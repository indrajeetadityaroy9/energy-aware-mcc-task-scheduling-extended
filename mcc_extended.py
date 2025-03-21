from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, NamedTuple
from copy import deepcopy
import time as time_module
import heapq
import logging
from math import inf
import random

from data import ExecutionTier, SchedulingState, core_execution_times, cloud_execution_times, edge_execution_times, generate_realistic_network_conditions, generate_realistic_power_models, add_task_attributes
from utils import format_schedule_3tier, validate_task_dependencies

# Constants and configuration data
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EdgeAssignment:
    """Tracks task assignment to an edge node"""
    edge_id: int  # Which edge node (E_m where m is 1...M)
    core_id: int  # Which core on that edge node


class ExecutionUnit(NamedTuple):
    """Represents a specific execution unit in the three-tier architecture"""
    tier: ExecutionTier
    location: Tuple[int, int] = None  # (node_id, core_id) for edge; core_id for device; None for cloud

    def __str__(self):
        if self.tier == ExecutionTier.DEVICE:
            return f"Device(Core {self.location[0]})"
        elif self.tier == ExecutionTier.EDGE:
            return f"Edge(Node {self.location[0]}, Core {self.location[1]})"
        else:
            return "Cloud"


@dataclass
class TaskMigrationState:
    """State information for task migrations during optimization"""
    time: float  # Total completion time
    energy: float  # Total energy consumption
    efficiency: float  # Energy reduction per unit time
    task_id: int  # ID of task being migrated
    # Source and target tiers/locations
    source_tier: ExecutionTier
    target_tier: ExecutionTier
    source_location: Optional[Tuple[int, int]] = None
    target_location: Optional[Tuple[int, int]] = None
    # Performance metrics
    time_increase: float = 0.0  # Increase in completion time
    energy_reduction: float = 0.0  # Reduction in energy consumption
    # Migration characteristics
    migration_complexity: int = 0  # Measure of migration complexity


class Task:
    def __init__(self, id, pred_task=None, succ_task=None, complexity=None, data_intensity=None):
        # Basic task graph structure
        self.id = id  # Task identifier v_i in DAG G=(V,E)
        self.succ_tasks = succ_task or []  # succ(v_i): Immediate successors
        self.pred_tasks = pred_task or []
        self.complexity = complexity if complexity is not None else random.uniform(0.5, 5.0)
        self.data_intensity = data_intensity if data_intensity is not None else random.uniform(0.2, 2.0)
        # Local execution times (T_i^l,k) - Section II.B
        self.local_execution_times = core_execution_times.get(id, [])
        # Cloud execution phases - Section II.B
        # [T_i^s: sending time, T_i^c: cloud computing time, T_i^r: receiving time]
        self.cloud_execution_times = cloud_execution_times
        # Edge execution times (T_i^e,m) - Section III extension
        self.edge_execution_times = {}
        for key, value in edge_execution_times.items():
            task_id, edge_id, core_id = key
            if task_id == id:
                self.edge_execution_times[(edge_id, core_id)] = value
        # ==== DATA TRANSFER PARAMETERS ====
        # Data sizes for all possible transfers
        self.data_sizes = -1
        # ==== ASSIGNMENT AND EXECUTION STATE ====
        # Current execution tier and location
        self.execution_tier = ExecutionTier.DEVICE  # Default to device
        self.device_core = -1  # If on device, which core (-1 = unassigned)
        self.edge_assignment = None  # If on edge, EdgeAssignment object
        # Execution migration path
        self.execution_path = []  # List of (tier, location) tuples showing migration path
        # ==== FINISH TIMES - Original Model ====
        # As defined in Section II.C
        self.FT_l = 0  # FT_i^l:  Local core finish time
        self.FT_ws = 0  # FT_i^ws: Wireless sending finish time (upload complete)
        self.FT_c = 0  # FT_i^c:  Cloud computation finish time
        self.FT_wr = 0  # FT_i^wr: Wireless receiving finish time (download complete)
        # ==== FINISH TIMES - Edge Extension ====
        # Edge execution finish times (FT_i^e,m)
        self.FT_edge = {}  # Map (edge_id) → finish time on that edge
        # Edge sending finish times (FT_i^(es),m)
        self.FT_edge_send = {}  # Map (source_edge, target) → sending finish time
        # target can be another edge id, 'device', or 'cloud'
        # Edge receiving finish times (when results arrive)
        self.FT_edge_receive = {}  # Map (edge_id) → time when results received
        # ==== READY TIMES - Original Model ====
        # Ready times as defined in equations 3-6
        self.RT_l = -1  # RT_i^l:  Ready time for local execution
        self.RT_ws = -1  # RT_i^ws: Ready time for wireless sending (upload)
        self.RT_c = -1  # RT_i^c:  Ready time for cloud execution
        self.RT_wr = -1  # RT_i^wr: Ready time for receiving results (download)
        # ==== READY TIMES - Edge Extension ====
        # Ready times for edge execution (RT_i^e,m)
        self.RT_edge = {}  # Map (edge_id) → ready time on that edge
        # Ready times for sending from edge (RT_i^(es),m)
        self.RT_edge_send = {}  # Map (source_edge, target) → ready time for sending
        # ==== SCHEDULING PARAMETERS ====
        # Task priority and scheduling state
        self.priority_score = None  # priority(v_i) for scheduling (equation 15)
        self.is_scheduled = SchedulingState.UNSCHEDULED
        # Final task completion time (across all possible execution locations)
        self.completion_time = -1
        self.execution_unit_task_start_times = None  # Will be set during scheduling
        self.edge_assignment = None
        self.device_core = -1

    def get_edge_execution_time(self, edge_id, core_id):
        """Get the execution time for a specific edge node and core"""
        # Try task-specific edge execution times first
        key = (self.id, edge_id, core_id)
        if 'edge_execution_times' in globals() and key in edge_execution_times:
            return edge_execution_times[key]

        # Next try the task's own execution times dictionary
        if hasattr(self, 'edge_execution_times'):
            key = (edge_id, core_id)
            if key in self.edge_execution_times:
                return self.edge_execution_times[key]

        # Calculate a reasonable fallback based on local execution times
        if hasattr(self, 'local_execution_times') and self.local_execution_times:
            # For data-intensive tasks (odd IDs), edge is slightly faster
            if self.id % 2 == 1:
                avg_local = sum(self.local_execution_times) / len(self.local_execution_times)
                return avg_local * 0.9
            # For compute-intensive tasks, edge is between device and cloud
            else:
                min_local = min(self.local_execution_times)
                cloud_time = sum(self.cloud_execution_times)
                return (min_local + cloud_time) / 2

        # Ultimate fallback
        return 5.0

    def calculate_data_transfer_time(self, source_tier, target_tier, upload_rates_dict, download_rates_dict,
                                     source_location=None, target_location=None):
        # Device to Cloud (original case)
        if source_tier == ExecutionTier.DEVICE and target_tier == ExecutionTier.CLOUD:
            if 'device_to_cloud' not in self.data_sizes:
                logger.warning(f"Missing data size for device_to_cloud transfer for task {self.id}")
                return self.cloud_execution_times[0]  # Fallback to default
            data_size = self.data_sizes['device_to_cloud']
            rate = upload_rates_dict.get('device_to_cloud', 1.0)
            return 0 if rate == 0 else data_size / rate

        # Cloud to Device (original case)
        elif source_tier == ExecutionTier.CLOUD and target_tier == ExecutionTier.DEVICE:
            if 'cloud_to_device' not in self.data_sizes:
                logger.warning(f"Missing data size for cloud_to_device transfer for task {self.id}")
                return self.cloud_execution_times[2]  # Fallback to default
            data_size = self.data_sizes['cloud_to_device']
            rate = download_rates_dict.get('cloud_to_device', 1.0)
            return 0 if rate == 0 else data_size / rate

        # Device to Edge
        elif source_tier == ExecutionTier.DEVICE and target_tier == ExecutionTier.EDGE:
            if target_location is None:
                logger.error(f"Target location required for device to edge transfer for task {self.id}")
                return float('inf')
            edge_id, _ = target_location
            key = f'device_to_edge{edge_id}'
            if key not in self.data_sizes:
                logger.warning(f"Missing data size for {key} transfer for task {self.id}")
                return 3.0  # Reasonable default similar to cloud
            data_size = self.data_sizes[key]
            rate = upload_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Edge to Device
        elif source_tier == ExecutionTier.EDGE and target_tier == ExecutionTier.DEVICE:
            if source_location is None:
                logger.error(f"Source location required for edge to device transfer for task {self.id}")
                return float('inf')
            edge_id, _ = source_location
            key = f'edge{edge_id}_to_device'
            if key not in self.data_sizes:
                logger.warning(f"Missing data size for {key} transfer for task {self.id}")
                return 1.0  # Reasonable default
            data_size = self.data_sizes[key]
            rate = download_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Edge to Cloud
        elif source_tier == ExecutionTier.EDGE and target_tier == ExecutionTier.CLOUD:
            if source_location is None:
                logger.error(f"Source location required for edge to cloud transfer for task {self.id}")
                return float('inf')
            edge_id, _ = source_location
            key = f'edge{edge_id}_to_cloud'
            if key not in self.data_sizes:
                logger.warning(f"Missing data size for {key} transfer for task {self.id}")
                return 2.0  # Reasonable default
            data_size = self.data_sizes[key]
            rate = upload_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Cloud to Edge
        elif source_tier == ExecutionTier.CLOUD and target_tier == ExecutionTier.EDGE:
            if target_location is None:
                logger.error(f"Target location required for cloud to edge transfer for task {self.id}")
                return float('inf')
            edge_id, _ = target_location
            key = f'cloud_to_edge{edge_id}'
            if key not in self.data_sizes:
                logger.warning(f"Missing data size for {key} transfer for task {self.id}")
                return 1.0  # Reasonable default
            data_size = self.data_sizes[key]
            rate = download_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Edge to Edge migration
        elif source_tier == ExecutionTier.EDGE and target_tier == ExecutionTier.EDGE:
            if source_location is None or target_location is None:
                logger.error(f"Source and target locations required for edge to edge transfer for task {self.id}")
                return float('inf')
            source_edge_id, _ = source_location
            target_edge_id, _ = target_location
            if source_edge_id == target_edge_id:
                return 0  # Same edge, no transfer needed
            key = f'edge{source_edge_id}_to_edge{target_edge_id}'
            if key not in self.data_sizes:
                logger.warning(f"Missing data size for {key} transfer for task {self.id}")
                return 1.5  # Reasonable default
            data_size = self.data_sizes[key]
            rate = upload_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Default case - unknown transfer path
        logger.error(f"Unknown transfer path from {source_tier} to {target_tier} for task {self.id}")
        return float('inf')  # Effectively prevents this transfer option

@dataclass
class OptimizationMetrics:
    initial_time: float = 0.0
    initial_energy: float = 0.0
    current_time: float = 0.0
    current_energy: float = 0.0
    best_time: float = float('inf')
    best_energy: float = float('inf')
    iterations: int = 0
    migrations: int = 0
    evaluations: int = 0
    time_violations: int = 0
    energy_improvements: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0

    def start(self, initial_time: float, initial_energy: float):
        self.initial_time = initial_time
        self.initial_energy = initial_energy
        self.current_time = initial_time
        self.current_energy = initial_energy
        self.best_time = initial_time
        self.best_energy = initial_energy
        self.start_time = time_module.time()

    def update(self, new_time: float, new_energy: float, evaluations: int = 0):
        self.iterations += 1
        self.evaluations += evaluations
        self.elapsed_time = time_module.time() - self.start_time

        # Check if time constraint violated
        if new_time > self.initial_time * 1.1:  # Allow 10% increase
            self.time_violations += 1
        # Check if energy improved
        if new_energy < self.current_energy:
            self.energy_improvements += 1
            self.migrations += 1
        # Update current metrics
        self.current_time = new_time
        self.current_energy = new_energy
        # Update best metrics
        if new_energy < self.best_energy:
            self.best_energy = new_energy
            self.best_time = new_time

    def get_summary(self) -> Dict[str, Any]:
        return {
            "initial_time": self.initial_time,
            "initial_energy": self.initial_energy,
            "final_time": self.current_time,
            "final_energy": self.current_energy,
            "time_change_pct": (self.current_time - self.initial_time) / self.initial_time * 100,
            "energy_reduction_pct": (self.initial_energy - self.current_energy) / self.initial_energy * 100,
            "iterations": self.iterations,
            "migrations": self.migrations,
            "evaluations": self.evaluations,
            "time_violations": self.time_violations,
            "energy_improvements": self.energy_improvements,
            "elapsed_time": self.elapsed_time,
            "evaluation_rate": self.evaluations / max(1, self.elapsed_time)
        }


class MigrationCache:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = {}
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: tuple) -> Optional[Tuple[float, float]]:
        self.access_count += 1

        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]

        self.miss_count += 1
        return None

    def put(self, key: tuple, value: Tuple[float, float]) -> None:
        if len(self.cache) >= self.capacity:
            logger.info(f"Clearing migration cache (size: {len(self.cache)})")
            self.cache.clear()

        self.cache[key] = value


class SequenceManager:
    def __init__(self, num_device_cores: int, num_edge_nodes: int, num_edge_cores_per_node: int):
        self.num_device_cores = num_device_cores
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores_per_node = num_edge_cores_per_node
        self.total_units = num_device_cores + num_edge_nodes * num_edge_cores_per_node + 1
        self.sequences = [[] for _ in range(self.total_units)]
        self.unit_to_index_map = {}

        # Populate the map for device cores
        for core_id in range(num_device_cores):
            unit = ExecutionUnit(ExecutionTier.DEVICE, (core_id,))
            self.unit_to_index_map[unit] = core_id

        # Populate the map for edge cores
        offset = num_device_cores
        for node_id in range(num_edge_nodes):
            for core_id in range(num_edge_cores_per_node):
                unit = ExecutionUnit(ExecutionTier.EDGE, (node_id, core_id))
                index = offset + node_id * num_edge_cores_per_node + core_id
                self.unit_to_index_map[unit] = index

        # Add the cloud
        cloud_index = self.total_units - 1
        self.unit_to_index_map[ExecutionUnit(ExecutionTier.CLOUD)] = cloud_index

    def get_cloud_index(self):
        """Return the index in self.sequences corresponding to the cloud resource."""
        return self.total_units - 1

    def set_all_sequences(self, sequences: List[List[int]]) -> None:
        if len(sequences) != self.total_units:
            raise ValueError(f"Expected {self.total_units} sequences, got {len(sequences)}")
        self.sequences = deepcopy(sequences)

    def build_sequences_from_tasks(self, tasks):
        # Initialize empty sequences
        device_sequences = [[] for _ in range(self.num_device_cores)]
        edge_sequences = [[] for _ in range(self.num_edge_nodes * self.num_edge_cores_per_node)]
        cloud_sequence = []

        # Populate sequences based on current task assignments
        for task in tasks:
            if task.execution_tier == ExecutionTier.DEVICE:
                core_id = task.device_core
                if 0 <= core_id < self.num_device_cores:
                    device_sequences[core_id].append(task.id)
            elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
                edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based
                core_id = task.edge_assignment.core_id - 1  # Convert to 0-based
                if (0 <= edge_id < self.num_edge_nodes and
                        0 <= core_id < self.num_edge_cores_per_node):
                    edge_idx = edge_id * self.num_edge_cores_per_node + core_id
                    edge_sequences[edge_idx].append(task.id)
            elif task.execution_tier == ExecutionTier.CLOUD:
                cloud_sequence.append(task.id)

        # Combine all sequences
        all_sequences = device_sequences + edge_sequences + [cloud_sequence]

        # Set the sequences
        self.set_all_sequences(all_sequences)

        return self  # For method chaining


def primary_assignment(
        tasks,
        num_device_cores=3,
        num_edge_nodes=2,
        num_edge_cores_per_node=2,
        upload_rates=None,
        download_rates=None
):
    """
    Assign each task to whichever tier (device, edge, or cloud) yields the minimal
    single-task finish time. This is a 'best guess' ignoring concurrency.

    Args:
        tasks: List of Task objects to assign
        num_device_cores: Number of device cores available
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node
        upload_rates: (Optional) dict of upload rates for transfers
        download_rates: (Optional) dict of download rates for transfers
    """
    from math import inf

    if upload_rates is None:
        upload_rates = {}
    if download_rates is None:
        download_rates = {}

    # Iterate through each task and pick the best resource
    for task in tasks:
        # Track best (time, tier, location) for this task
        best_time = inf
        best_tier = None
        best_dev_core = -1
        best_edge_id = -1
        best_edge_core = -1

        #############################################
        # 1) Evaluate all DEVICE cores
        #############################################
        if task.local_execution_times:
            for dev_core in range(min(num_device_cores, len(task.local_execution_times))):
                t_local = task.local_execution_times[dev_core]
                # Optional: add a small overhead if data must be “fetched” from anywhere
                # For initial scheduling, we can ignore or treat it as 0.
                total_time = t_local
                if total_time < best_time:
                    best_time = total_time
                    best_tier = ExecutionTier.DEVICE
                    best_dev_core = dev_core
                    best_edge_id = -1
                    best_edge_core = -1

        #############################################
        # 2) Evaluate all EDGE nodes / cores
        #############################################
        # For each edge node, each core
        for e_id in range(1, num_edge_nodes + 1):
            for c_id in range(1, num_edge_cores_per_node + 1):
                # Obtain known edge execution time
                edge_time = task.get_edge_execution_time(e_id, c_id)
                if edge_time is None or edge_time == float('inf'):
                    continue

                # Basic assumption: to run at the edge, you might upload from device => edge,
                # and eventually download edge => device if needed.
                # For now, we just do single-task “exec time + up/down overhead”:
                # (Adjust or remove if you want no overhead in the initial pass.)
                up_key = f'device_to_edge{e_id}'
                up_size = task.data_sizes.get(up_key, 0)
                up_rate = upload_rates.get(up_key, 1.0)
                upload_time = up_size / up_rate if up_rate > 0 else 0

                down_key = f'edge{e_id}_to_device'
                down_size = task.data_sizes.get(down_key, 0)
                down_rate = download_rates.get(down_key, 1.0)
                download_time = down_size / down_rate if down_rate > 0 else 0

                total_time = upload_time + edge_time + download_time
                if total_time < best_time:
                    best_time = total_time
                    best_tier = ExecutionTier.EDGE
                    best_edge_id = e_id
                    best_edge_core = c_id
                    best_dev_core = -1

        #############################################
        # 3) Evaluate CLOUD
        #############################################
        if task.cloud_execution_times:
            # Typically T_send + T_cloud + T_receive
            t_send, t_cloud, t_recv = task.cloud_execution_times
            # You could also incorporate real data sizes if you prefer
            total_time = t_send + t_cloud + t_recv
            if total_time < best_time:
                best_time = total_time
                best_tier = ExecutionTier.CLOUD
                best_dev_core = -1
                best_edge_id = -1
                best_edge_core = -1

        else:
            # If no cloud times are defined, skip it
            pass

        # -----------------------------
        #  Update the chosen assignment
        # -----------------------------
        task.execution_tier = best_tier

        if best_tier == ExecutionTier.DEVICE:
            # Assign to the best device core
            task.device_core = best_dev_core
            # We can store an initial “finish time” guess
            task.FT_l = best_time
            # Clear out edge/cloud data
            task.edge_assignment = None
            task.FT_edge.clear()
            task.FT_edge_receive.clear()
            task.FT_ws = task.FT_c = task.FT_wr = 0

            logger.info(f"Task {task.id} => DEVICE (core {best_dev_core}) ~Time={best_time:.2f}")

        elif best_tier == ExecutionTier.EDGE:
            # Create EdgeAssignment
            task.edge_assignment = EdgeAssignment(edge_id=best_edge_id, core_id=best_edge_core)
            # Record an initial finish time
            task.FT_edge[best_edge_id] = best_time
            # Some small guess for “receive time” if needed
            task.FT_edge_receive[best_edge_id] = best_time
            # Clear out device/cloud data
            task.device_core = -1
            task.FT_l = 0
            task.FT_ws = task.FT_c = task.FT_wr = 0

            logger.info(f"Task {task.id} => EDGE (node {best_edge_id}, core {best_edge_core}) ~Time={best_time:.2f}")

        else:
            # Cloud assignment
            task.device_core = -1
            task.edge_assignment = None
            task.FT_l = 0
            # Based on T_send + T_cloud + T_recv
            if len(task.cloud_execution_times) == 3:
                t_send, t_cloud, t_recv = task.cloud_execution_times
                task.RT_ws = 0
                task.FT_ws = t_send
                task.RT_c = task.FT_ws
                task.FT_c = task.RT_c + t_cloud
                task.RT_wr = task.FT_c
                task.FT_wr = task.RT_wr + t_recv
            else:
                # fallback
                task.FT_wr = best_time

            # Clear out edge data
            task.FT_edge.clear()
            task.FT_edge_receive.clear()

            logger.info(f"Task {task.id} => CLOUD ~Time={best_time:.2f}")

    logger.info("Finished primary assignment for all tasks.\n")


def task_prioritizing(tasks):
    """
    Calculates a 'priority' (often called rank in HEFT) for each task based on:
        w[i] + max( priority( successor ) )
    where w[i] is the estimated execution cost on the assigned resource.

    This ensures that tasks with large compute times and that lead to
    other large tasks get higher priority.

    Args:
        tasks: List of Task objects that have already been assigned.
    """
    from math import inf
    num_tasks = len(tasks)
    # We'll store the computed cost for each task in an array w:
    w = [0.0] * num_tasks
    task_index_map = {task.id: i for i, task in enumerate(tasks)}

    # 1) Compute w[i] = the single-task cost on its assigned resource
    for i, task in enumerate(tasks):
        if task.execution_tier == ExecutionTier.DEVICE:
            # If assigned to device, cost = local_execution_time for that chosen core
            core_idx = task.device_core
            if 0 <= core_idx < len(task.local_execution_times):
                w[i] = task.local_execution_times[core_idx]
            else:
                w[i] = inf

        elif task.execution_tier == ExecutionTier.EDGE:
            # If assigned to an edge node, cost = stored edge_execution_time
            if task.edge_assignment:
                e_id = task.edge_assignment.edge_id
                c_id = task.edge_assignment.core_id
                # Attempt to get the stored execution time
                exec_time = task.get_edge_execution_time(e_id, c_id)
                w[i] = exec_time if exec_time else inf
            else:
                w[i] = inf

        else:
            # If assigned to cloud, cost = T_send + T_cloud + T_recv
            if task.cloud_execution_times and len(task.cloud_execution_times) == 3:
                w[i] = sum(task.cloud_execution_times)
            else:
                w[i] = inf

    # 2) Compute the recursive priority: priority(i) = w[i] + max_{succ} priority(succ)
    computed_priority = {}
    # Map each task ID => list of successor IDs
    succ_map = {task.id: [st.id for st in task.succ_tasks] for task in tasks}

    def compute_priority(tid):
        if tid in computed_priority:
            return computed_priority[tid]

        i = task_index_map[tid]
        successors = succ_map[tid]
        if not successors:
            # If no successors, priority = w[i]
            computed_priority[tid] = w[i]
        else:
            # max successor priority
            max_succ = 0
            for s_id in successors:
                val = compute_priority(s_id)
                if val > max_succ:
                    max_succ = val
            computed_priority[tid] = w[i] + max_succ

        return computed_priority[tid]

    # Compute priority for all tasks
    for task in tasks:
        compute_priority(task.id)

    # Finally, store each task’s priority_score
    for task in tasks:
        task.priority_score = computed_priority[task.id]

    logger.info("Finished priority calculation for all tasks.\n")
    for t in tasks:
        logger.info(f"Task {t.id} => priority_score = {t.priority_score:.2f}")


def calculate_realistic_execution_time(task, execution_unit, system_state=None):
    """
    Calculate realistic execution time for a task on a specific execution unit.

    Parameters:
        task: Task object to execute
        execution_unit: Target execution unit (device core, edge, cloud)
        system_state: Current state of system resources (load, contention)

    Returns:
        Tuple of (execution_time, confidence_level)
    """
    # If no system state is provided, use a default state
    if system_state is None:
        # Create a default system state with reasonable values
        system_state = {
            'loads': {},  # Load per execution unit
            'memory_contention': {0: 0.1, 1: 0.1, 2: 0.1},  # Memory contention per tier
            'io_contention': {0: 0.1, 1: 0.1, 2: 0.1},  # I/O contention per tier
            'execution_history': {},  # Task execution count per unit
            'frequencies': {},  # CPU frequency per unit (normalized)
        }
        # Set default load and frequency for this execution unit
        system_state['loads'][execution_unit] = 0.5  # Default 50% load
        system_state['frequencies'][execution_unit] = 1.0  # Default full frequency

    # Get base execution time from task profile
    if execution_unit.tier == ExecutionTier.DEVICE:
        base_time = task.local_execution_times[execution_unit.location[0]]
    elif execution_unit.tier == ExecutionTier.EDGE:
        edge_id, core_id = execution_unit.location
        base_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
    else:  # Cloud
        base_time = task.cloud_execution_times[1]  # Pure computation time

    # Get system load for this execution unit
    unit_load = system_state['loads'].get(execution_unit, 0.5)  # Default to 50% load

    # CPU contention effect - execution time increases non-linearly with load
    # At 50% load, minimal impact. At 100% load, up to 2x slowdown
    if unit_load <= 0.5:
        contention_factor = 1.0 + (unit_load * 0.2)  # Up to 10% slowdown at 50% load
    else:
        contention_factor = 1.1 + ((unit_load - 0.5) * 1.8)  # 10-100% slowdown between 50-100% load

    # Memory access contention
    memory_contention = system_state['memory_contention'].get(execution_unit.tier.value, 0.1)
    memory_factor = 1.0 + memory_contention

    # I/O contention (affects disk/network bound tasks more)
    io_intensity = getattr(task, 'io_intensity', 0.3)  # Default: 30% I/O bound
    io_contention = system_state['io_contention'].get(execution_unit.tier.value, 0.1)
    io_factor = 1.0 + (io_contention * io_intensity)

    # Cache effects - repeated executions of same task are faster
    # Lookup task execution history on this unit
    execution_count = system_state['execution_history'].get((task.id, execution_unit), 0)
    cache_factor = 1.0 if execution_count == 0 else max(0.85, 1.0 - (0.05 * min(execution_count, 3)))

    # Startup overhead - first execution includes startup cost
    startup_overhead = 0
    if execution_count == 0:
        if execution_unit.tier == ExecutionTier.CLOUD:
            startup_overhead = random.uniform(0.2, 0.5)  # Cloud VM/container startup
        elif execution_unit.tier == ExecutionTier.EDGE:
            startup_overhead = random.uniform(0.1, 0.3)  # Edge container startup

    # Dynamic frequency scaling effect
    cpu_frequency = system_state['frequencies'].get(execution_unit, 1.0)  # Normalized frequency
    frequency_factor = 1.0 / cpu_frequency  # Lower frequency = longer execution

    # Random variation (±5%)
    random_factor = random.uniform(0.95, 1.05)

    # Combine all factors
    adjusted_time = (base_time * contention_factor * memory_factor * io_factor *
                     cache_factor * frequency_factor * random_factor +
                     startup_overhead)

    # Calculate confidence level in estimate
    # Higher variation in factors = lower confidence
    factor_variation = abs(contention_factor - 1.0) + abs(memory_factor - 1.0) + abs(io_factor - 1.0)
    confidence_level = max(0.5, 1.0 - (factor_variation / 2) - (0.1 if execution_count == 0 else 0))

    return adjusted_time, confidence_level

# First, define a modified version of get_execution_unit_from_index
def get_execution_unit_from_index(target_unit_idx, sequence_manager=None):
    """
    Converts a linear index to an ExecutionUnit using system configuration.

    Args:
        target_unit_idx: Linear index representing an execution unit
        sequence_manager: Optional SequenceManager containing system configuration

    Returns:
        ExecutionUnit corresponding to the provided index
    """
    # Get system configuration
    if sequence_manager:
        num_device_cores = sequence_manager.num_device_cores
        num_edge_nodes = sequence_manager.num_edge_nodes
        num_edge_cores_per_node = sequence_manager.num_edge_cores_per_node
    else:
        # Default values if sequence_manager not provided
        num_device_cores = 3  # Default to 3 device cores
        num_edge_nodes = 2  # Default to 2 edge nodes
        num_edge_cores_per_node = 2  # Default to 2 cores per edge node

    # Check if index is for a device core
    if target_unit_idx < num_device_cores:
        return ExecutionUnit(ExecutionTier.DEVICE, (target_unit_idx,))

    # Check if index is for an edge core
    edge_offset = num_device_cores
    total_edge_cores = num_edge_nodes * num_edge_cores_per_node

    if target_unit_idx < edge_offset + total_edge_cores:
        edge_index = target_unit_idx - edge_offset
        node_id = edge_index // num_edge_cores_per_node
        core_id = edge_index % num_edge_cores_per_node
        return ExecutionUnit(ExecutionTier.EDGE, (node_id, core_id))

    # Must be cloud
    return ExecutionUnit(ExecutionTier.CLOUD)


def get_current_execution_unit(task):
    """
    Gets the current execution unit for a task.

    Args:
        task: Task object

    Returns:
        ExecutionUnit where the task is currently scheduled
    """
    if task.execution_tier == ExecutionTier.DEVICE:
        return ExecutionUnit(ExecutionTier.DEVICE, (task.device_core,))
    elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
        return ExecutionUnit(ExecutionTier.EDGE, (
            task.edge_assignment.edge_id - 1,  # Convert to 0-based
            task.edge_assignment.core_id - 1  # Convert to 0-based
        ))
    else:  # Cloud
        return ExecutionUnit(ExecutionTier.CLOUD)


def generate_migration_cache_key(tasks, task_idx, source_unit, target_unit):
    """
    Generate a unique cache key for a migration scenario.

    Args:
        tasks: List of all tasks
        task_idx: Index of task being migrated
        source_unit: Source ExecutionUnit
        target_unit: Target ExecutionUnit

    Returns:
        tuple: Unique identifier for this migration
    """
    # Encode task
    task_key = task_idx

    # Encode source unit
    source_key = (source_unit.tier.value,)
    if source_unit.tier == ExecutionTier.DEVICE:
        source_key += (source_unit.location[0],)
    elif source_unit.tier == ExecutionTier.EDGE:
        source_key += (source_unit.location[0], source_unit.location[1])

    # Encode target unit
    target_key = (target_unit.tier.value,)
    if target_unit.tier == ExecutionTier.DEVICE:
        target_key += (target_unit.location[0],)
    elif target_unit.tier == ExecutionTier.EDGE:
        target_key += (target_unit.location[0], target_unit.location[1])

    # Encode current task assignments (can be simplified for performance)
    # Only include critical information like dependencies
    assignments = tuple(
        (t.id, t.execution_tier.value) for t in tasks
        if t.id in [p.id for p in tasks[task_idx].pred_tasks] or
        t.id in [s.id for s in tasks[task_idx].succ_tasks]
    )

    return (task_key, source_key, target_key, assignments)


class ThreeTierTaskScheduler:
    def __init__(self, tasks, num_cores=3, num_edge_nodes=2, edge_cores_per_node=2,
                 upload_rates=None, download_rates=None):
        self.tasks = tasks
        self.k = num_cores  # Number of device cores
        self.M = num_edge_nodes  # Number of edge nodes
        self.edge_cores = edge_cores_per_node

        self.upload_rates = upload_rates or {}
        self.download_rates = download_rates or {}

        self.edge_to_device_ready = [0.0] * self.M

        # Ensure we have default rates for all edge nodes if not provided
        for e_id in range(1, self.M + 1):
            edge_to_device_key = f'edge{e_id}_to_device'
            if edge_to_device_key not in self.download_rates:
                # Default rate - could be based on data from elsewhere in the code
                self.download_rates[edge_to_device_key] = 1.0

        # Earliest-ready times for concurrency:
        self.core_earliest_ready = [0.0] * self.k  # device cores
        self.edge_core_earliest_ready = [
            [0.0] * self.edge_cores for _ in range(self.M)
        ]
        self.ws_ready = 0.0  # single channel device->cloud (if modeling one-at-a-time)
        self.wr_ready = 0.0  # single channel cloud->device
        # Edge↔device, edge↔edge, etc. readiness
        self.device_to_edge_ready = [0.0] * self.M
        self.edge_to_cloud_ready = [0.0] * self.M
        self.cloud_to_edge_ready = [0.0] * self.M
        self.edge_to_edge_ready = [[0.0] * self.M for _ in range(self.M)]

        # One sequence list per resource (device cores + edge cores + cloud):
        total_resources = self.k + self.M * self.edge_cores + 1
        self.sequences = [[] for _ in range(total_resources)]

        # NEW: A place to record the final "minimal-delay" schedule decisions
        # so you have an explicit dictionary of {task_id: {...}}
        self.final_min_delay_schedule = {}

    def get_edge_core_index(self, edge_id, core_id):
        """Return the index in self.sequences corresponding to (edge_id, core_id)."""
        return self.k + edge_id * self.edge_cores + core_id

    def get_cloud_index(self):
        """Return the index in self.sequences corresponding to the cloud resource."""
        return self.k + self.M * self.edge_cores

    # -------------------------------------------------------------------------
    #                 A) Strict Topological + Priority Ordering
    # -------------------------------------------------------------------------
    def schedule_tasks_topo_priority(self):
        """
        Main entry point for Step 1: minimal-delay scheduling.

        1. Compute in-degrees for all tasks (topological ordering).
        2. Build a max-heap keyed on negative priority.
        3. Repeatedly pop the highest-priority task whose in-degree is zero.
        4. Per-task resource selection to find earliest finishing option.
        5. Update the schedule and reduce in-degree of successors.
        """
        # Build in-degree table
        in_degs = {t.id: 0 for t in self.tasks}
        task_map = {t.id: t for t in self.tasks}

        for t in self.tasks:
            for succ in t.succ_tasks:
                in_degs[succ.id] += 1

        # Initialize a max-heap (store -priority so highest priority is popped first)
        ready_heap = []
        for t in self.tasks:
            if in_degs[t.id] == 0:
                heapq.heappush(ready_heap, (-t.priority_score, t.id))

        # Keep scheduling until no more tasks remain
        while ready_heap:
            # Pop highest-priority task
            neg_priority, tid = heapq.heappop(ready_heap)
            task = task_map[tid]

            # Make sure its predecessors have finished - THIS IS CRITICAL
            pred_finish = self.earliest_pred_finish_time(task)
            if pred_finish == float('inf'):
                # Some predecessor not done or not scheduled properly
                continue

            # Find earliest finish among all resources
            best_finish, best_option = self.find_earliest_finish_among_all_resources(task, pred_finish)

            # Schedule the task
            if best_option is None or best_finish == float('inf'):
                continue

            # Execute the schedule decision
            self.execute_schedule_decision(task, pred_finish, best_finish, best_option)

            # With task scheduled, reduce in-degree of successors
            for succ in task.succ_tasks:
                in_degs[succ.id] -= 1
                if in_degs[succ.id] == 0:
                    heapq.heappush(ready_heap, (-succ.priority_score, succ.id))

        # Once done, tasks that are scheduled have a final finish time
        # Let's do a quick check or store final results in self.final_min_delay_schedule
        self.record_final_schedule()
        self.validate_final_schedule()

    # -------------------------------------------------------------------------
    #                 B) Per-Task Resource Selection
    # -------------------------------------------------------------------------
    def find_earliest_finish_among_all_resources(self, task, pred_finish):
        """
        Tries scheduling 'task' on:
          - each device core
          - each edge node & core
          - cloud
        Returns (earliest_finish_time, best_option) for the minimal finishing time.
        best_option is a tuple describing which resource we used.
        """
        best_finish_time = float('inf')
        best_option = None

        # 1) Check device cores
        device_ready_times = []
        for core_id in range(self.k):
            # Use max of pred_finish (which accounts for edge-to-device transfers)
            # and when the core is available
            start_time = max(pred_finish, self.core_earliest_ready[core_id])
            if core_id < len(task.local_execution_times):
                exec_time = task.local_execution_times[core_id]
            else:
                exec_time = float('inf')
            finish_time = start_time + exec_time
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_option = ("DEVICE", core_id)

        # 2) Check edge nodes
        for e_id in range(self.M):
            # Calculate earliest time the task can start on that edge
            # We need to calculate edge ready time for each edge node individually
            # and make sure we use this per-edge ready time instead of the global pred_finish
            edge_ready_time = self.calculate_edge_ready_time(task, e_id)
            start_time = edge_ready_time  # Use edge-specific ready time, not global pred_finish

            # Among that edge's cores, pick the earliest finishing core
            chosen_finish, chosen_core = self.find_best_edge_core(task, e_id, start_time)
            if chosen_finish < best_finish_time:
                best_finish_time = chosen_finish
                best_option = ("EDGE", e_id, chosen_core)

        # 3) Check cloud
        #   "cloud_ready" is the earliest we can upload
        cloud_ready = max(pred_finish, self.calculate_cloud_upload_ready_time(task))
        cloud_finish = self.get_cloud_finish(task, cloud_ready)
        if cloud_finish < best_finish_time:
            best_finish_time = cloud_finish
            best_option = ("CLOUD",)

        return best_finish_time, best_option

    def find_best_edge_core(self, task, edge_id, start_time):
        """
        Among the edge_id's cores, find which yields the earliest finish time.
        Return (finish_time, core_id).
        """
        best_finish = float('inf')
        best_core = None
        for core_id in range(self.edge_cores):
            actual_start = max(start_time, self.edge_core_earliest_ready[edge_id][core_id])
            exec_time = task.get_edge_execution_time(edge_id + 1, core_id + 1) or float('inf')
            finish_time = actual_start + exec_time
            if finish_time < best_finish:
                best_finish = finish_time
                best_core = core_id
        return best_finish, best_core

    # -------------------------------------------------------------------------
    #                 C) Actually Schedule the Chosen Option
    # -------------------------------------------------------------------------
    def execute_schedule_decision(self, task, pred_finish, finish_time, best_option):
        """
        Given the resource kind (DEVICE, EDGE, or CLOUD) in best_option,
        actually schedule the task by setting start time, finish time,
        resource usage, etc. This method records the concurrency-based
        start time in task.execution_unit_task_start_times so that later
        formatting/logging can display the correct time (rather than 0.0).

        For edge tasks, properly calculates and records edge-to-device transfer times
        and edge-to-edge transfer times.
        """
        resource_kind = best_option[0]

        if resource_kind == "DEVICE":
            # e.g. best_option = ("DEVICE", core_id)
            core_id = best_option[1]

            # The device can't start this task before:
            # 1) The task's predecessors are finished (pred_finish),
            # 2) The core is free (self.core_earliest_ready[core_id]).
            start = max(pred_finish, self.core_earliest_ready[core_id])

            # The finish_time was chosen by the scheduling logic,
            # but we can also confirm it as start + local_exec_time.
            self.core_earliest_ready[core_id] = finish_time

            # Record times
            task.FT_l = finish_time
            task.execution_finish_time = finish_time
            task.execution_tier = ExecutionTier.DEVICE
            task.device_core = core_id
            task.is_scheduled = SchedulingState.SCHEDULED

            # Make sure execution_unit_task_start_times is large enough
            total_units = self.k + self.M * self.edge_cores + 1
            if (not hasattr(task, 'execution_unit_task_start_times') or
                    task.execution_unit_task_start_times is None or
                    len(task.execution_unit_task_start_times) < total_units):
                task.execution_unit_task_start_times = [-1.0] * total_units

            # Store the concurrency-based start time at index=core_id for device
            task.execution_unit_task_start_times[core_id] = start

            # Also record in the sequences list
            self.sequences[core_id].append(task.id)

        elif resource_kind == "EDGE":
            # e.g. best_option = ("EDGE", e_id, c_id)
            e_id, c_id = best_option[1], best_option[2]

            # Calculate the concurrency-based start time for this edge node+core
            # The earliest we can start is after pred_finish and when the edge core is free
            start = max(pred_finish, self.edge_core_earliest_ready[e_id][c_id])
            self.edge_core_earliest_ready[e_id][c_id] = finish_time

            # Record the finishing time
            if not hasattr(task, 'FT_edge'):
                task.FT_edge = {}
            task.FT_edge[e_id] = finish_time
            task.execution_finish_time = finish_time
            task.execution_tier = ExecutionTier.EDGE
            task.edge_assignment = EdgeAssignment(edge_id=e_id + 1, core_id=c_id + 1)
            task.is_scheduled = SchedulingState.SCHEDULED

            # Calculate edge-to-device transfer time
            if not hasattr(self, 'edge_to_device_ready'):
                self.edge_to_device_ready = [0.0] * self.M

            edge_to_device_key = f'edge{e_id + 1}_to_device'
            data_size = task.data_sizes.get(edge_to_device_key, 1.0)
            download_rate = self.download_rates.get(edge_to_device_key, 1.0)
            transfer_time = data_size / download_rate if download_rate > 0 else 0

            # Determine when we can start transmitting results back to device,
            # considering contention on the edge-to-device channel
            transfer_start = max(finish_time, self.edge_to_device_ready[e_id])

            # Calculate when results arrive at device
            receive_time = transfer_start + transfer_time

            # Store this time in the task
            if not hasattr(task, 'FT_edge_receive'):
                task.FT_edge_receive = {}
            task.FT_edge_receive[e_id] = receive_time

            # Update channel readiness
            self.edge_to_device_ready[e_id] = receive_time

            # ---- Precompute all potential edge-to-edge transfers ----
            # Initialize tracking for edge-to-edge transfers
            if not hasattr(task, 'FT_edge_send'):
                task.FT_edge_send = {}

            # Reserve times on all potential edge-to-edge channels from this edge
            for target_e_id in range(self.M):
                if target_e_id != e_id:  # Skip self-transfers
                    # Calculate when this task's data would be available to other edges
                    edge_to_edge_key = f'edge{e_id + 1}_to_edge{target_e_id + 1}'
                    data_size = task.data_sizes.get(edge_to_edge_key, 1.0)
                    rate = self.upload_rates.get(edge_to_edge_key, 1.0)

                    # Potential transfer could start after task finishes
                    potential_transfer_start = max(finish_time, self.edge_to_edge_ready[e_id][target_e_id])
                    transfer_time = data_size / rate if rate > 0 else 0
                    potential_transfer_finish = potential_transfer_start + transfer_time

                    # Record this potential transfer time
                    task.FT_edge_send[(e_id, target_e_id)] = potential_transfer_finish

                    # Update the edge-to-edge channel availability
                    self.edge_to_edge_ready[e_id][target_e_id] = potential_transfer_finish

                    logger.debug(
                        f"Potential edge-to-edge: Task {task.id} Edge{e_id + 1}→Edge{target_e_id + 1} "
                        f"would be available at {potential_transfer_finish:.3f}"
                    )

            # Make sure execution_unit_task_start_times is large enough
            total_units = self.k + self.M * self.edge_cores + 1
            if (not hasattr(task, 'execution_unit_task_start_times') or
                    task.execution_unit_task_start_times is None or
                    len(task.execution_unit_task_start_times) < total_units):
                task.execution_unit_task_start_times = [-1.0] * total_units

            # Convert (e_id, c_id) into a sequence index
            seq_idx = self.get_edge_core_index(e_id, c_id)
            task.execution_unit_task_start_times[seq_idx] = start
            self.sequences[seq_idx].append(task.id)

        elif resource_kind == "CLOUD":
            # e.g. best_option = ("CLOUD",)
            # We allow only one-at-a-time upload if modeling a single channel
            cloud_ready = max(pred_finish, self.ws_ready)
            self.ws_ready = cloud_ready + task.cloud_execution_times[0]  # T_send
            # The overall finish_time is (cloud_ready + T_send + T_cloud + T_recv)
            # (already computed in find_earliest_finish_among_all_resources)

            task.RT_ws = cloud_ready
            task.FT_ws = cloud_ready + task.cloud_execution_times[0]
            task.RT_c = task.FT_ws
            task.FT_c = task.RT_c + task.cloud_execution_times[1]
            task.RT_wr = task.FT_c
            task.FT_wr = task.RT_wr + task.cloud_execution_times[2]
            task.execution_finish_time = finish_time
            task.execution_tier = ExecutionTier.CLOUD
            task.device_core = -1
            task.edge_assignment = None
            task.is_scheduled = SchedulingState.SCHEDULED

            # Clear any edge execution data
            if hasattr(task, 'FT_edge'):
                task.FT_edge = {}
            if hasattr(task, 'FT_edge_receive'):
                task.FT_edge_receive = {}

            total_units = self.k + self.M * self.edge_cores + 1
            if (not hasattr(task, 'execution_unit_task_start_times') or
                    task.execution_unit_task_start_times is None or
                    len(task.execution_unit_task_start_times) < total_units):
                task.execution_unit_task_start_times = [-1.0] * total_units

            cloud_idx = self.get_cloud_index()
            # Record the time at which upload begins
            task.execution_unit_task_start_times[cloud_idx] = cloud_ready
            self.sequences[cloud_idx].append(task.id)

        else:
            logger.error(f"Unknown resource kind: {resource_kind}")

    # -------------------------------------------------------------------------
    #   Helper: earliest time all preds have completed
    # -------------------------------------------------------------------------
    def earliest_pred_finish_time(self, task):
        """
        For strict precedence, a task can't start until all predecessors are done.
        Return the max finish time among all of task's predecessors,
        properly accounting for edge-to-device and edge-to-edge transfer times.
        """
        if not task.pred_tasks:
            return 0.0

        latest = 0.0
        for pred in task.pred_tasks:
            # If not scheduled => block
            if pred.is_scheduled != SchedulingState.SCHEDULED or not hasattr(pred, 'execution_finish_time'):
                return float('inf')

            # Determine when this predecessor's results are available based on tier
            if pred.execution_tier == ExecutionTier.DEVICE:
                # Local task - available as soon as it finishes
                pred_finish = getattr(pred, 'FT_l', 0.0)

            elif pred.execution_tier == ExecutionTier.CLOUD:
                # Cloud task - results available after download
                pred_finish = getattr(pred, 'FT_wr', 0.0)

            elif pred.execution_tier == ExecutionTier.EDGE:
                # Edge task handling depends on the destination tier
                if not pred.edge_assignment:
                    logger.warning(f"Edge assignment missing for predecessor {pred.id}")
                    return float('inf')

                pred_edge_id = pred.edge_assignment.edge_id - 1  # 0-based index

                # Determine the target execution location for the current task
                if task.execution_tier == ExecutionTier.DEVICE:
                    # Edge -> Device transfer
                    if hasattr(pred, 'FT_edge_receive') and pred_edge_id in pred.FT_edge_receive:
                        pred_finish = pred.FT_edge_receive[pred_edge_id]
                    else:
                        # If not recorded, calculate it now
                        if not hasattr(pred, 'FT_edge') or pred_edge_id not in pred.FT_edge:
                            logger.warning(f"Missing edge finish time for predecessor {pred.id}")
                            return float('inf')

                        edge_finish = pred.FT_edge[pred_edge_id]
                        edge_to_device_key = f'edge{pred_edge_id + 1}_to_device'
                        data_size = pred.data_sizes.get(edge_to_device_key, 1.0)
                        download_rate = self.download_rates.get(edge_to_device_key, 1.0)
                        transfer_time = data_size / download_rate if download_rate > 0 else 0

                        # Account for channel contention
                        if hasattr(self, 'edge_to_device_ready'):
                            transfer_start = max(edge_finish, self.edge_to_device_ready[pred_edge_id])
                            pred_finish = transfer_start + transfer_time
                        else:
                            pred_finish = edge_finish + transfer_time

                elif task.execution_tier == ExecutionTier.EDGE:
                    # First check if the current task is assigned to an edge
                    if not task.edge_assignment:
                        logger.warning(f"Edge assignment missing for task {task.id}")
                        return float('inf')

                    task_edge_id = task.edge_assignment.edge_id - 1

                    if pred_edge_id == task_edge_id:
                        # Same edge node - core-to-core communication only
                        pred_finish = pred.FT_edge.get(pred_edge_id, float('inf'))

                        # Add minor delay for core-to-core transfer if on different cores
                        if task.edge_assignment.core_id != pred.edge_assignment.core_id:
                            pred_finish += 0.1  # Small overhead for core-to-core
                    else:
                        # Different edge nodes - need edge-to-edge transfer
                        if not hasattr(pred, 'FT_edge') or pred_edge_id not in pred.FT_edge:
                            logger.warning(f"Missing edge finish time for predecessor {pred.id}")
                            return float('inf')

                        # Check if a pre-calculated edge-to-edge transfer exists
                        transfer_key = (pred_edge_id, task_edge_id)
                        if hasattr(pred, 'FT_edge_send') and transfer_key in pred.FT_edge_send:
                            # Use pre-calculated time
                            pred_finish = pred.FT_edge_send[transfer_key]
                            logger.info(f"Using pre-calculated edge-to-edge transfer: "
                                        f"Task {pred.id}→{task.id}, "
                                        f"Edge {pred_edge_id + 1}→{task_edge_id + 1}, "
                                        f"Finish: {pred_finish:.3f}")
                        else:
                            # Calculate edge-to-edge transfer time
                            edge_finish = pred.FT_edge[pred_edge_id]

                            data_key = f'edge{pred_edge_id + 1}_to_edge{task_edge_id + 1}'
                            data_size = pred.data_sizes.get(data_key, 1.0)
                            transfer_rate = self.upload_rates.get(data_key, 1.0)
                            transfer_time = data_size / transfer_rate if transfer_rate > 0 else 0

                            # Account for channel contention
                            if hasattr(self, 'edge_to_edge_ready'):
                                transfer_start = max(edge_finish, self.edge_to_edge_ready[pred_edge_id][task_edge_id])
                                transfer_end = transfer_start + transfer_time

                                # Update channel availability (this is crucial)
                                self.edge_to_edge_ready[pred_edge_id][task_edge_id] = transfer_end
                                pred_finish = transfer_end

                                # Store calculated time for reuse
                                if not hasattr(pred, 'FT_edge_send'):
                                    pred.FT_edge_send = {}
                                pred.FT_edge_send[transfer_key] = transfer_end

                                logger.info(f"Calculated edge-to-edge transfer: "
                                            f"Task {pred.id}→{task.id}, "
                                            f"Edge {pred_edge_id + 1}→{task_edge_id + 1}, "
                                            f"Start: {transfer_start:.3f}, "
                                            f"End: {transfer_end:.3f}")
                            else:
                                # Simple calculation without contention tracking
                                pred_finish = edge_finish + transfer_time

                elif task.execution_tier == ExecutionTier.CLOUD:
                    # Edge -> Cloud transfer needed
                    if not hasattr(pred, 'FT_edge') or pred_edge_id not in pred.FT_edge:
                        logger.warning(f"Missing edge finish time for predecessor {pred.id}")
                        return float('inf')

                    edge_finish = pred.FT_edge[pred_edge_id]
                    data_key = f'edge{pred_edge_id + 1}_to_cloud'
                    data_size = pred.data_sizes.get(data_key, 1.0)
                    upload_rate = self.upload_rates.get(data_key, 1.0)
                    transfer_time = data_size / upload_rate if upload_rate > 0 else 0

                    # Account for channel contention
                    if hasattr(self, 'edge_to_cloud_ready'):
                        transfer_start = max(edge_finish, self.edge_to_cloud_ready[pred_edge_id])
                        transfer_end = transfer_start + transfer_time

                        # Update channel availability
                        self.edge_to_cloud_ready[pred_edge_id] = transfer_end
                        pred_finish = transfer_end
                    else:
                        pred_finish = edge_finish + transfer_time
            else:
                logger.error(f"Unknown execution tier for predecessor {pred.id}")
                return float('inf')

            if pred_finish <= 0:
                logger.warning(f"Invalid finish time for predecessor {pred.id}: {pred_finish}")
                return float('inf')

            latest = max(latest, pred_finish)

        return latest

    # -------------------------------------------------------------------------
    #   Helper: earliest time we can upload to the cloud
    # -------------------------------------------------------------------------
    def calculate_cloud_upload_ready_time(self, task):
        # For a single channel, you might just require we wait for ws_ready
        # plus the predecessor's finish. If you want "infinite concurrency,"
        # you'd skip the shared ws_ready. For now we do:
        return max(self.ws_ready, self.earliest_pred_finish_time(task))

    # -------------------------------------------------------------------------
    #   Helper: compute finishing time if assigned to cloud
    # -------------------------------------------------------------------------
    def get_cloud_finish(self, task, upload_start):
        t_send, t_cloud, t_recv = task.cloud_execution_times
        # If you assume unlimited concurrency in the cloud, it's just:
        return upload_start + t_send + t_cloud + t_recv

    # -------------------------------------------------------------------------
    #   Helper: earliest time a task can start on edge e_id
    # -------------------------------------------------------------------------
    def calculate_edge_ready_time(self, task, edge_id):
        """
        Revised calculation of the ready time for executing 'task' on a given edge node (0-based index edge_id)
        without caching. This method computes, for each predecessor of the task, the time at which its output is
        available on the target edge node by considering:

          - DEVICE → EDGE transfers,
          - EDGE → EDGE transfers (for a predecessor on a different edge),
          - CLOUD → EDGE transfers.

        For EDGE → EDGE transfers, the channel's availability is updated using the maximum of its current value
        and the newly computed transfer finish time.

        Parameters:
          task: The task for which to compute the ready time.
          edge_id: The 0-based index of the target edge node.

        Returns:
          float: The earliest time at which 'task' can start on edge 'edge_id'.
        """
        if not task.pred_tasks:
            return 0.0  # Entry task has no predecessors

        max_ready_time = 0.0

        for pred_task in task.pred_tasks:
            # If any predecessor is unscheduled, we cannot compute a valid ready time.
            if pred_task.is_scheduled != SchedulingState.SCHEDULED or not hasattr(pred_task, 'execution_finish_time'):
                logger.warning(f"Predecessor task {pred_task.id} of task {task.id} not scheduled")
                return float('inf')

            # Initialize ready_time for this predecessor.
            ready_time = 0.0

            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # DEVICE → EDGE transfer:
                pred_finish = pred_task.FT_l
                if pred_finish <= 0:
                    logger.warning(f"Invalid finish time for DEVICE predecessor {pred_task.id}")
                    return float('inf')
                data_key = f'device_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = self.upload_rates.get(data_key, 1.0)
                channel_avail = self.device_to_edge_ready[edge_id]
                transfer_start = max(pred_finish, channel_avail)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time
                # Update the device-to-edge channel availability.
                self.device_to_edge_ready[edge_id] = ready_time

            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # EDGE → EDGE transfer:
                if not pred_task.edge_assignment:
                    logger.warning(f"Edge assignment missing for EDGE predecessor {pred_task.id}")
                    return float('inf')
                pred_edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based index

                if pred_edge_id == edge_id:
                    # Predecessor is on the same edge:
                    if pred_edge_id not in pred_task.FT_edge:
                        logger.warning(f"Missing FT_edge[{pred_edge_id}] for predecessor {pred_task.id}")
                        return float('inf')
                    ready_time = pred_task.FT_edge[pred_edge_id]
                    # Add a small overhead if the child is assigned to a different core.
                    if task.edge_assignment and (task.edge_assignment.core_id != pred_task.edge_assignment.core_id):
                        ready_time += 0.1
                else:
                    # Predecessor is on a different edge: compute EDGE → EDGE transfer delay.
                    if pred_edge_id not in pred_task.FT_edge:
                        logger.warning(f"Missing FT_edge[{pred_edge_id}] for EDGE predecessor {pred_task.id}")
                        return float('inf')
                    data_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                    data_size = task.data_sizes.get(data_key, 0)
                    rate = self.upload_rates.get(data_key, 1.0)
                    current_channel_avail = self.edge_to_edge_ready[pred_edge_id][edge_id]
                    pred_finish = pred_task.FT_edge[pred_edge_id]
                    transfer_start = max(pred_finish, current_channel_avail)
                    transfer_time = data_size / rate if rate > 0 else 0
                    new_ready_time = transfer_start + transfer_time
                    # Robustly update the channel availability using max().
                    self.edge_to_edge_ready[pred_edge_id][edge_id] = max(current_channel_avail, new_ready_time)
                    ready_time = self.edge_to_edge_ready[pred_edge_id][edge_id]
                    # Optionally, record the computed transfer time for reuse.
                    if not hasattr(pred_task, 'FT_edge_send'):
                        pred_task.FT_edge_send = {}
                    pred_task.FT_edge_send[(pred_edge_id, edge_id)] = ready_time
                    logger.info(f"Calculated EDGE→EDGE transfer: Task {pred_task.id}→Task {task.id}, "
                                f"from Edge {pred_edge_id + 1} to Edge {edge_id + 1}, "
                                f"transfer start {transfer_start:.3f}, delay {transfer_time:.3f}, ready at {ready_time:.3f}")

            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # CLOUD → EDGE transfer:
                pred_finish = pred_task.FT_c
                if pred_finish <= 0:
                    logger.warning(f"Invalid finish time for CLOUD predecessor {pred_task.id}")
                    return float('inf')
                data_key = f'cloud_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = self.download_rates.get(data_key, 1.0)
                channel_avail = self.cloud_to_edge_ready[edge_id]
                transfer_start = max(pred_finish, channel_avail)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time
                # Update the cloud-to-edge channel availability.
                self.cloud_to_edge_ready[edge_id] = ready_time

            else:
                logger.error(f"Unknown execution tier for predecessor {pred_task.id}")
                return float('inf')

            # Sanity check for ready_time.
            if ready_time <= 0:
                logger.warning(f"Computed non-positive ready time for predecessor {pred_task.id}: {ready_time}")
                return float('inf')

            max_ready_time = max(max_ready_time, ready_time)

        # Also ensure that the device-to-edge channel's current state is considered.
        max_ready_time = max(max_ready_time, self.device_to_edge_ready[edge_id])
        logger.debug(f"Task {task.id} computed ready time on EDGE {edge_id + 1}: {max_ready_time:.3f}")
        return max_ready_time

    # -------------------------------------------------------------------------
    #   3) Record the Final "Minimal-Delay" Schedule
    # -------------------------------------------------------------------------
    def record_final_schedule(self):
        """
        After all tasks are scheduled, store the final decisions in a dictionary
        for easy reference. The user can use self.final_min_delay_schedule to
        get the minimal-delay schedule mapping {task_id: {...}}.
        """
        self.final_min_delay_schedule.clear()
        total_units = self.k + self.M * self.edge_cores + 1
        for task in self.tasks:
            finish_t = getattr(task, 'execution_finish_time', None)
            tier_name = task.execution_tier.name if hasattr(task, 'execution_tier') else "UNASSIGNED"

            record = {
                'task_id': task.id,
                'tier': tier_name,
                'finish_time': finish_t
            }

            # If device:
            if task.execution_tier == ExecutionTier.DEVICE:
                record['device_core'] = task.device_core
                # check the start time you recorded in execution_unit_task_start_times
                if task.execution_unit_task_start_times:
                    record['start_time'] = task.execution_unit_task_start_times[task.device_core]
            elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
                e_id = task.edge_assignment.edge_id - 1
                c_id = task.edge_assignment.core_id - 1
                record['edge_node'] = e_id
                record['edge_core'] = c_id
                seq_idx = self.get_edge_core_index(e_id, c_id)
                if task.execution_unit_task_start_times:
                    record['start_time'] = task.execution_unit_task_start_times[seq_idx]
            elif task.execution_tier == ExecutionTier.CLOUD:
                cloud_idx = self.get_cloud_index()
                if task.execution_unit_task_start_times:
                    record['start_time'] = task.execution_unit_task_start_times[cloud_idx]

            self.final_min_delay_schedule[task.id] = record

    def validate_final_schedule(self):
        """
        Optional: check for any obvious precedence violations or scheduling anomalies
        after the final schedule is set.
        """
        for task in self.tasks:
            if task.is_scheduled != SchedulingState.SCHEDULED:
                continue
            pred_finish = self.earliest_pred_finish_time(task)
            if task.execution_finish_time < pred_finish:
                print(f"ERROR: Precedence violation for Task {task.id} -> finishes at {task.execution_finish_time}, "
                      f"but predecessor finishes at {pred_finish}.")


def total_time_3tier(tasks):
    """
    Returns the overall application completion time for the three-tier system.

    Args:
        tasks: List of all tasks in the application

    Returns:
        float: Maximum finish time among exit tasks
    """
    # Identify exit tasks (tasks with no successors)
    exit_tasks = [t for t in tasks if not t.succ_tasks]

    if not exit_tasks:
        logger.warning("No exit tasks found in task graph")
        # Fallback: use maximum finish time among all tasks
        return max(task.execution_finish_time for task in tasks
                   if hasattr(task, 'execution_finish_time'))

    # Find maximum finish time among exit tasks
    max_finish_time = 0.0

    for task in exit_tasks:
        finish_time = 0.0

        # Get finish time based on execution tier
        if task.execution_tier == ExecutionTier.DEVICE:
            finish_time = task.FT_l
        elif task.execution_tier == ExecutionTier.CLOUD:
            finish_time = task.FT_wr  # When results are received
        elif task.execution_tier == ExecutionTier.EDGE:
            if task.edge_assignment:
                edge_id = task.edge_assignment.edge_id - 1
                # Use edge_receive time if available (when results arrive at device)
                if hasattr(task, 'FT_edge_receive') and edge_id in task.FT_edge_receive:
                    finish_time = task.FT_edge_receive[edge_id]
                else:
                    # Fallback to edge execution finish time
                    finish_time = task.FT_edge.get(edge_id, 0.0)

        max_finish_time = max(max_finish_time, finish_time)

    return max_finish_time


def total_energy_3tier_with_rf(
        tasks,
        device_power_profiles,
        rf_power,  # from mobile_power_models['rf']
        upload_rates,  # e.g. {'device_to_edge': 8.0, 'device_to_cloud': 3.0}
        default_signal_strength=70.0
):
    """
    Computes total *device-side* energy in a three-tier scenario.

    Args:
        tasks: List of Task objects, each with .execution_tier, .device_core, etc.
        device_power_profiles: e.g. mobile_power_models['device']
            A dict of {core_id: {
               'idle_power': float,
               'dynamic_power': function(load)->float,
               ...
            }}
        rf_power: from mobile_power_models['rf'], a dict with:
            {
                'device_to_edge': lambda data_rate, signal_strength: returns power (Watts),
                'device_to_cloud': lambda data_rate, signal_strength: returns power (Watts)
            }
        upload_rates: dict of data rates (Mbps), e.g. {
            'device_to_edge': 8.0,
            'device_to_cloud': 3.0
        }
        default_signal_strength: fallback for signal strength (dBm or custom scale)

    Returns:
        total_energy (float): total device energy consumption in Joules
    """
    total_energy = 0.0

    for task in tasks:
        # -------------------------------------------
        # 1) Local execution on device
        # -------------------------------------------
        if task.execution_tier == ExecutionTier.DEVICE:
            # The device core that ran this task
            core_id = task.device_core
            start_times = getattr(task, 'execution_unit_task_start_times', None) or []
            # compute actual runtime
            if 0 <= core_id < len(start_times):
                start_t = start_times[core_id]
                exec_time = task.execution_finish_time - start_t
            else:
                exec_time = 0.0

            # Now get the device core's power model
            core_info = device_power_profiles.get(core_id, {})
            idle_pwr = core_info.get('idle_power', 0.0)
            dyn_func = core_info.get('dynamic_power', lambda load: 0.0)
            # For simplicity, assume load=1.0 while it's running
            load = 1.0
            # total power = idle + dynamic
            pwr_during_task = idle_pwr + dyn_func(load)
            # energy = power * time
            E_local = pwr_during_task * exec_time

            total_energy += E_local

        # -------------------------------------------
        # 2) Task assigned to an edge node
        #    => device->edge upload energy
        # -------------------------------------------
        elif task.execution_tier == ExecutionTier.EDGE:
            # We assume the device is responsible for sending data to the edge
            # so device uses radio power for T_send
            # data size
            if task.edge_assignment:
                edge_id = task.edge_assignment.edge_id  # 1-based
                ds_key = f'device_to_edge{edge_id}'
            else:
                ds_key = 'device_to_edge'  # fallback

            data_size = task.data_sizes.get(ds_key, 0.0)  # in MB or other unit
            data_rate_mbps = upload_rates.get('device_to_edge', 5.0)
            # Convert data_size to Mb if your data_size is in MB:
            # data_size_mb * 8 = data_size in Mbits
            # Then T_send = data_size_in_Mbits / data_rate_mbps
            data_in_mbits = data_size * 8.0
            T_send = data_in_mbits / data_rate_mbps if data_rate_mbps > 0 else 0.0

            # radio power from rf_power
            # we pass data_rate_mbps, plus a default signal
            radio_pwr = rf_power['device_to_edge'](data_rate_mbps, default_signal_strength)

            E_upload = radio_pwr * T_send
            total_energy += E_upload

        # -------------------------------------------
        # 3) Task assigned to the cloud
        #    => device->cloud upload energy
        # -------------------------------------------
        elif task.execution_tier == ExecutionTier.CLOUD:
            data_size = task.data_sizes.get('device_to_cloud', 0.0)
            data_rate_mbps = upload_rates.get('device_to_cloud', 2.0)
            data_in_mbits = data_size * 8.0
            T_send = data_in_mbits / data_rate_mbps if data_rate_mbps > 0 else 0.0

            radio_pwr = rf_power['device_to_cloud'](data_rate_mbps, default_signal_strength)
            E_upload = radio_pwr * T_send
            total_energy += E_upload

        else:
            # If for some reason there's an unknown tier, do nothing
            pass

    return total_energy

def optimize_task_scheduling(tasks, sequence_manager, T_final,
                             power_models, upload_rates, download_rates,
                             migration_cache=None, max_iterations=100):
    """
    Implements the task migration algorithm extended to a three-tier architecture.
    Optimizes energy consumption while maintaining completion time constraints.

    Args:
        tasks: List of tasks from application graph G=(V,E)
        sequence_manager: SequenceManager object with current execution sequences
        T_final: Target completion time constraint T_max
        power_models: Power consumption models for device, edge, and RF components
        upload_rates: Dictionary of upload rates for different connections
        download_rates: Dictionary of download rates for different connections
        migration_cache: Optional cache for reusing migration evaluations
        max_iterations: Maximum number of migration iterations to perform

    Returns:
        tuple: (tasks, sequence_manager) with optimized scheduling
    """
    # Initialize migration cache if not provided
    if migration_cache is None:
        migration_cache = MigrationCache(capacity=10000)

    # Store system configuration
    num_device_cores = len(sequence_manager.sequences[:sequence_manager.num_device_cores])
    num_edge_nodes = sequence_manager.num_edge_nodes
    num_edge_cores_per_node = sequence_manager.num_edge_cores_per_node

    # Calculate initial application completion time and energy consumption
    current_time = total_time_3tier(tasks)
    current_energy = total_energy_3tier_with_rf(
        tasks=tasks,
        device_power_profiles=power_models['device'],
        rf_power=power_models['rf'],
        upload_rates=upload_rates
    )

    # Initialize optimization metrics
    metrics = OptimizationMetrics()
    metrics.start(current_time, current_energy)

    # Iterative improvement loop
    iteration = 0
    energy_improved = True

    while energy_improved and iteration < max_iterations:
        # Store current metrics for comparison
        previous_energy = current_energy

        # Define time constraint - allow some flexibility
        T_max = T_final * 1.05  # 5% scheduling flexibility

        # Generate and evaluate all valid migration options
        migration_candidates = []

        # For each task, evaluate possible migration targets
        for task_idx, task in enumerate(tasks):
            current_unit = get_current_execution_unit(task)

            # Evaluate migrations to all possible execution units
            total_execution_units = (
                    num_device_cores +  # Device cores
                    num_edge_nodes * num_edge_cores_per_node +  # Edge cores
                    1  # Cloud
            )

            for target_unit_idx in range(total_execution_units):
                # Skip self-migration (same execution unit)
                target_unit = get_execution_unit_from_index(target_unit_idx)
                if current_unit == target_unit:
                    continue

                # Perform migration trial
                trial_time, trial_energy = evaluate_migration(
                    tasks=tasks,
                    task_idx=task_idx,
                    sequence_manager=sequence_manager,
                    source_unit=current_unit,
                    target_unit=target_unit,
                    migration_cache=migration_cache,
                    power_models=power_models,
                    upload_rates=upload_rates,
                    download_rates=download_rates
                )

                # Store migration candidate if energy reduction is possible
                if trial_energy < previous_energy:
                    energy_reduction = previous_energy - trial_energy
                    time_increase = max(0, trial_time - current_time)

                    # Calculate efficiency ratio (energy reduction per unit time)
                    if time_increase <= 0:
                        efficiency = float('inf')  # Prioritize cases with no time increase
                    else:
                        efficiency = energy_reduction / time_increase

                    migration_candidates.append(
                        TaskMigrationState(
                            time=trial_time,
                            energy=trial_energy,
                            efficiency=efficiency,
                            energy_reduction=energy_reduction,
                            time_increase=time_increase,
                            task_id=task.id,
                            source_tier=current_unit.tier,
                            target_tier=target_unit.tier,
                            source_location=current_unit.location,
                            target_location=target_unit.location
                        )
                    )

        # Sort migration candidates by:
        # 1. First prioritize those that don't increase completion time
        # 2. Then by energy reduction (highest first)
        # 3. If time increases, consider efficiency ratio
        migration_candidates.sort(
            key=lambda m: (
                0 if m.time <= current_time else 1,  # No time increase first
                -m.energy_reduction,  # Highest energy reduction
                -m.efficiency  # Highest efficiency ratio
            )
        )

        # Select best migration
        best_migration = None
        for candidate in migration_candidates:
            # Skip if it would exceed time constraint
            if candidate.time > T_max:
                continue

            # Accept first valid candidate (already sorted optimally)
            best_migration = candidate
            break

        # Exit if no valid migrations remain
        if best_migration is None:
            break

        # 1. Update task's execution tier and location
        source_unit = ExecutionUnit(
            tier=best_migration.source_tier,
            location=best_migration.source_location
        )
        target_unit = ExecutionUnit(
            tier=best_migration.target_tier,
            location=best_migration.target_location
        )

        # 2. Construct new sequences using linear-time algorithm
        sequence_manager = construct_sequence(
            tasks=tasks,
            task_id=best_migration.task_id,
            source_unit=source_unit,
            target_unit=target_unit,
            sequence_manager=sequence_manager
        )

        # 3. Apply kernel algorithm for O(N) rescheduling
        tasks = kernel_algorithm_3tier(
            tasks=tasks,
            sequence_manager=sequence_manager,
            upload_rates=upload_rates,
            download_rates=download_rates
        )

        # 4. Update successor ready times
        migrated_task = next(t for t in tasks if t.id == best_migration.task_id)
        for succ_task in migrated_task.succ_tasks:
            # Recalculate ready time for successor
            new_ready_time = calculate_task_ready_time(succ_task, tasks, target_unit)
            # Update scheduling state if now ready
            if new_ready_time <= current_time and succ_task.is_scheduled != SchedulingState.SCHEDULED:
                succ_task.is_scheduled = SchedulingState.SCHEDULED

        # Calculate new energy consumption and completion time
        current_time = total_time_3tier(tasks)
        current_energy = total_energy_3tier_with_rf(
            tasks=tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        # Update metrics
        metrics.update(current_time, current_energy)
        energy_improved = current_energy < previous_energy
        iteration += 1

        # Manage cache size
        if len(migration_cache.cache) > migration_cache.capacity * 0.9:
            migration_cache.clear()

    # Log optimization results
    return tasks, sequence_manager


def evaluate_migration(tasks, task_idx, sequence_manager, source_unit, target_unit,
                       migration_cache, power_models, upload_rates, download_rates):
    """
    Evaluates a potential task migration between different execution units in a three-tier architecture.

    Args:
        tasks: List of all tasks
        task_idx: Index of task to be migrated
        sequence_manager: SequenceManager object with current execution sequences
        source_unit: Source ExecutionUnit (tier and location)
        target_unit: Target ExecutionUnit (tier and location)
        migration_cache: MigrationCache object for storing evaluation results
        power_models: Dictionary of power consumption models
        upload_rates: Dictionary of upload rates for different connections
        download_rates: Dictionary of download rates for different connections

    Returns:
        tuple: (time, energy) representing application completion time and energy consumption after migration
    """
    # Generate cache key for this migration scenario
    cache_key = generate_migration_cache_key(tasks, task_idx, source_unit, target_unit)

    # Check cache for previously evaluated scenario
    cached_result = migration_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    # Create copies to avoid modifying original state
    tasks_copy = deepcopy(tasks)
    sequences_copy = deepcopy(sequence_manager.sequences)
    sequence_manager_copy = SequenceManager(
        num_device_cores=sequence_manager.num_device_cores,
        num_edge_nodes=sequence_manager.num_edge_nodes,
        num_edge_cores_per_node=sequence_manager.num_edge_cores_per_node
    )
    sequence_manager_copy.set_all_sequences(sequences_copy)

    # Get task to migrate
    task_to_migrate = tasks_copy[task_idx]

    # Update task's execution unit
    if target_unit.tier == ExecutionTier.DEVICE:
        task_to_migrate.execution_tier = ExecutionTier.DEVICE
        task_to_migrate.device_core = target_unit.location[0]
        task_to_migrate.edge_assignment = None
    elif target_unit.tier == ExecutionTier.EDGE:
        task_to_migrate.execution_tier = ExecutionTier.EDGE
        task_to_migrate.device_core = -1
        task_to_migrate.edge_assignment = EdgeAssignment(
            edge_id=target_unit.location[0] + 1,  # Convert to 1-based indexing
            core_id=target_unit.location[1] + 1  # Convert to 1-based indexing
        )
    else:  # Cloud
        task_to_migrate.execution_tier = ExecutionTier.CLOUD
        task_to_migrate.device_core = -1
        task_to_migrate.edge_assignment = None

    # Construct new sequence
    source_idx = sequence_manager_copy.unit_to_index_map[source_unit]
    target_idx = sequence_manager_copy.unit_to_index_map[target_unit]

    # Remove task from source sequence
    if task_to_migrate.id in sequence_manager_copy.sequences[source_idx]:
        sequence_manager_copy.sequences[source_idx].remove(task_to_migrate.id)

    # Find insertion point in target sequence based on task ready time
    target_sequence = sequence_manager_copy.sequences[target_idx]

    # Calculate ready time based on task dependencies
    ready_time = calculate_task_ready_time(task_to_migrate, tasks_copy, target_unit)

    # Get execution times of tasks already in target sequence
    task_start_times = []
    for task_id in target_sequence:
        task = next((t for t in tasks_copy if t.id == task_id), None)
        if task and task.execution_unit_task_start_times:
            start_time = task.execution_unit_task_start_times[target_idx]
            task_start_times.append((task_id, start_time))

    # Sort by start time
    task_start_times.sort(key=lambda x: x[1])

    # Find insertion point
    insertion_idx = 0
    for i, (tid, start_time) in enumerate(task_start_times):
        if start_time >= ready_time:
            insertion_idx = i
            break
        insertion_idx = i + 1

    # Insert task at appropriate position
    sequence_manager_copy.sequences[target_idx].insert(insertion_idx, task_to_migrate.id)

    # Apply kernel algorithm for rescheduling
    kernel_algorithm_3tier(
        tasks=tasks_copy,
        sequence_manager=sequence_manager_copy,
        upload_rates=upload_rates,
        download_rates=download_rates
    )

    # Calculate new metrics
    new_time = total_time_3tier(tasks_copy)
    new_energy = total_energy_3tier_with_rf(
        tasks=tasks_copy,
        device_power_profiles=power_models['device'],
        rf_power=power_models['rf'],
        upload_rates=upload_rates
    )

    # Cache and return results
    result = (new_time, new_energy)
    migration_cache.put(cache_key, result)

    return result

def calculate_task_ready_time(task, tasks, target_unit, upload_rates=None, download_rates=None):
    """
    Calculate the ready time for a task on a specific target execution unit.
    Takes into account data transfer times between different tiers.

    Args:
        task: Task to calculate ready time for
        tasks: All tasks in the application (needed for lookup)
        target_unit: Target ExecutionUnit (tier and location)
        upload_rates: Dictionary of upload rates for different connections
        download_rates: Dictionary of download rates for different connections

    Returns:
        float: Earliest time the task can start execution
    """
    if not task.pred_tasks:
        return 0.0  # Entry task

    # Initialize default rates if not provided
    upload_rates = upload_rates or {}
    download_rates = download_rates or {}

    # Find maximum ready time based on predecessors
    max_ready_time = 0.0

    for pred in task.pred_tasks:
        if pred.is_scheduled != SchedulingState.SCHEDULED:
            return float('inf')  # Predecessor not scheduled yet

        pred_finish = 0.0

        if pred.execution_tier == ExecutionTier.DEVICE:
            pred_finish = pred.FT_l
        elif pred.execution_tier == ExecutionTier.CLOUD:
            pred_finish = pred.FT_wr  # When results are received
        elif pred.execution_tier == ExecutionTier.EDGE:
            if pred.edge_assignment:
                edge_id = pred.edge_assignment.edge_id - 1  # 0-based
                if hasattr(pred, 'FT_edge_receive') and edge_id in pred.FT_edge_receive:
                    pred_finish = pred.FT_edge_receive[edge_id]
                else:
                    pred_finish = pred.FT_edge.get(edge_id, 0.0)
        else:
            pred_finish = 0.0

        # Add data transfer time based on source and target tiers
        transfer_time = 0.0

        # Calculate transfer time from predecessor location to target unit
        if pred.execution_tier != target_unit.tier:
            # Different tiers - need data transfer
            if pred.execution_tier == ExecutionTier.DEVICE and target_unit.tier == ExecutionTier.EDGE:
                # Device → Edge transfer
                edge_id = target_unit.location[0]
                data_key = f'device_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0.0)
                rate = upload_rates.get(data_key, 1.0)
                transfer_time = data_size / rate if rate > 0 else 0.0

            elif pred.execution_tier == ExecutionTier.DEVICE and target_unit.tier == ExecutionTier.CLOUD:
                # Device → Cloud transfer
                data_size = task.data_sizes.get('device_to_cloud', 0.0)
                rate = upload_rates.get('device_to_cloud', 1.0)
                transfer_time = data_size / rate if rate > 0 else 0.0

            elif pred.execution_tier == ExecutionTier.EDGE and target_unit.tier == ExecutionTier.DEVICE:
                # Edge → Device transfer
                if pred.edge_assignment:
                    edge_id = pred.edge_assignment.edge_id - 1
                    data_key = f'edge{edge_id + 1}_to_device'
                    data_size = task.data_sizes.get(data_key, 0.0)
                    rate = download_rates.get(data_key, 1.0)
                    transfer_time = data_size / rate if rate > 0 else 0.0

            elif pred.execution_tier == ExecutionTier.EDGE and target_unit.tier == ExecutionTier.CLOUD:
                # Edge → Cloud transfer
                if pred.edge_assignment:
                    edge_id = pred.edge_assignment.edge_id - 1
                    data_key = f'edge{edge_id + 1}_to_cloud'
                    data_size = task.data_sizes.get(data_key, 0.0)
                    rate = upload_rates.get(data_key, 1.0)
                    transfer_time = data_size / rate if rate > 0 else 0.0

            elif pred.execution_tier == ExecutionTier.EDGE and target_unit.tier == ExecutionTier.EDGE:
                # Edge → Edge transfer (if different edge nodes)
                if pred.edge_assignment:
                    source_edge_id = pred.edge_assignment.edge_id - 1
                    target_edge_id = target_unit.location[0]
                    if source_edge_id != target_edge_id:
                        data_key = f'edge{source_edge_id + 1}_to_edge{target_edge_id + 1}'
                        data_size = task.data_sizes.get(data_key, 0.0)
                        rate = upload_rates.get(data_key, 1.0)
                        transfer_time = data_size / rate if rate > 0 else 0.0

            elif pred.execution_tier == ExecutionTier.CLOUD and target_unit.tier == ExecutionTier.DEVICE:
                # Cloud → Device transfer
                data_size = task.data_sizes.get('cloud_to_device', 0.0)
                rate = download_rates.get('cloud_to_device', 1.0)
                transfer_time = data_size / rate if rate > 0 else 0.0

            elif pred.execution_tier == ExecutionTier.CLOUD and target_unit.tier == ExecutionTier.EDGE:
                # Cloud → Edge transfer
                edge_id = target_unit.location[0]
                data_key = f'cloud_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0.0)
                rate = download_rates.get(data_key, 1.0)
                transfer_time = data_size / rate if rate > 0 else 0.0

        # Total ready time = predecessor finish + transfer time
        ready_time = pred_finish + transfer_time
        max_ready_time = max(max_ready_time, ready_time)

    return max_ready_time


def construct_sequence(tasks, task_id, source_unit, target_unit, sequence_manager):
    """
    Implements the linear-time rescheduling algorithm for three-tier architecture.
    Constructs new execution sequences after task migration while preserving task precedence.

    Args:
        tasks: List of all tasks in the application
        task_id: ID of task being migrated
        source_unit: Source ExecutionUnit (tier and location)
        target_unit: Target ExecutionUnit (tier and location)
        sequence_manager: SequenceManager object with current execution sequences

    Returns:
        Updated sequence_manager after task migration
    """
    # Step 1: Find the task to migrate
    task_to_migrate = next((t for t in tasks if t.id == task_id), None)
    if not task_to_migrate:
        logger.error(f"Task {task_id} not found in task list")
        return sequence_manager

    # Step 2: Convert execution units to sequence indices
    source_seq_idx = sequence_manager.unit_to_index_map.get(source_unit)
    target_seq_idx = sequence_manager.unit_to_index_map.get(target_unit)

    if source_seq_idx is None or target_seq_idx is None:
        logger.error(f"Invalid source or target execution unit")
        return sequence_manager

    # Step 3: Get source and target sequences
    source_sequence = sequence_manager.sequences[source_seq_idx]
    target_sequence = sequence_manager.sequences[target_seq_idx]

    # Step 4: Remove task from source sequence if present
    if task_id in source_sequence:
        source_sequence.remove(task_id)

    # Step 5: Calculate ready time for the migrating task on the target execution unit
    # This is the earliest time the task can start on the target unit
    ready_time = 0.0

    if hasattr(task_to_migrate, 'RT_l') and target_unit.tier == ExecutionTier.DEVICE:
        ready_time = task_to_migrate.RT_l
    elif hasattr(task_to_migrate, 'RT_ws') and target_unit.tier == ExecutionTier.CLOUD:
        ready_time = task_to_migrate.RT_ws
    elif hasattr(task_to_migrate, 'RT_edge') and target_unit.tier == ExecutionTier.EDGE:
        edge_id = target_unit.location[0]
        ready_time = task_to_migrate.RT_edge.get(edge_id, 0.0)

    # If no specific ready time is available, calculate from predecessors
    if ready_time <= 0:
        for pred in task_to_migrate.pred_tasks:
            pred_finish = 0.0

            # Get predecessor finish time based on execution tier
            if pred.execution_tier == ExecutionTier.DEVICE:
                pred_finish = pred.FT_l
            elif pred.execution_tier == ExecutionTier.CLOUD:
                pred_finish = pred.FT_wr  # When device receives results
            elif pred.execution_tier == ExecutionTier.EDGE:
                if pred.edge_assignment:
                    edge_id = pred.edge_assignment.edge_id - 1  # 0-based
                    if hasattr(pred, 'FT_edge_receive') and edge_id in pred.FT_edge_receive:
                        pred_finish = pred.FT_edge_receive[edge_id]
                    elif hasattr(pred, 'FT_edge') and edge_id in pred.FT_edge:
                        pred_finish = pred.FT_edge[edge_id]

            ready_time = max(ready_time, pred_finish)

    # Step 6: Get start times for tasks in target sequence
    task_start_times = []
    for tid in target_sequence:
        task = next((t for t in tasks if t.id == tid), None)
        if not task or not hasattr(task, 'execution_unit_task_start_times'):
            continue

        start_time = task.execution_unit_task_start_times[target_seq_idx]
        if start_time >= 0:
            task_start_times.append((tid, start_time))

    # Sort tasks by start time
    task_start_times.sort(key=lambda x: x[1])

    # Step 7: Find insertion point for task based on ready_time
    insertion_idx = 0
    for i, (tid, start_time) in enumerate(task_start_times):
        if start_time >= ready_time:
            insertion_idx = target_sequence.index(tid)
            break
        if i == len(task_start_times) - 1:
            # After all existing tasks
            insertion_idx = len(target_sequence)

    # Step 8: Insert task at the determined position
    target_sequence.insert(insertion_idx, task_id)

    # Step 9: Update task's execution tier and location
    task_to_migrate.execution_tier = target_unit.tier

    if target_unit.tier == ExecutionTier.DEVICE:
        task_to_migrate.device_core = target_unit.location[0]
        task_to_migrate.edge_assignment = None
    elif target_unit.tier == ExecutionTier.EDGE:
        task_to_migrate.device_core = -1
        task_to_migrate.edge_assignment = EdgeAssignment(
            edge_id=target_unit.location[0] + 1,  # Convert to 1-based indexing
            core_id=target_unit.location[1] + 1  # Convert to 1-based indexing
        )
    else:  # CLOUD
        task_to_migrate.device_core = -1
        task_to_migrate.edge_assignment = None

    # Update sequence manager with modified sequences
    sequence_manager.sequences[source_seq_idx] = source_sequence
    sequence_manager.sequences[target_seq_idx] = target_sequence

    #logger.info(f"Migrated Task {task_id} from {source_unit} to {target_unit}")
    return sequence_manager


def kernel_algorithm_3tier(tasks, sequence_manager, upload_rates=None, download_rates=None):
    """
    Implements the kernel (rescheduling) algorithm for three-tier architecture.
    Provides linear-time task rescheduling for the task migration phase.

    Args:
        tasks: List of all tasks in the application
        sequence_manager: SequenceManager with execution sequences for all units
        upload_rates: Dictionary of upload rates for different connections
        download_rates: Dictionary of download rates for different connections

    Returns:
        Updated tasks with new scheduling times
    """
    # Initialize the three-tier kernel scheduler
    scheduler = ThreeTierKernelScheduler(
        tasks=tasks,
        sequence_manager=sequence_manager,
        upload_rates=upload_rates or {},
        download_rates=download_rates or {}
    )

    # Initialize ready task queue
    ready_queue = scheduler.initialize_queue()

    # Main scheduling loop - process until all tasks are scheduled
    while ready_queue:
        # Get next ready task from queue
        current_task = ready_queue.popleft()

        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on execution tier
        if current_task.execution_tier == ExecutionTier.DEVICE:
            # Schedule on assigned device core
            scheduler.schedule_device_task(current_task)

        elif current_task.execution_tier == ExecutionTier.EDGE:
            # Schedule on assigned edge node and core
            scheduler.schedule_edge_task(current_task)

        else:  # ExecutionTier.CLOUD
            # Schedule three-phase cloud execution
            scheduler.schedule_cloud_task(current_task)

        # Update dependency and sequence readiness for all tasks
        newly_ready_tasks = []
        for task in tasks:
            if task.is_scheduled == SchedulingState.KERNEL_SCHEDULED:
                continue  # Skip already scheduled tasks

            # Update readiness state
            scheduler.update_task_readiness(task)

            # Check if task is now ready for scheduling
            if scheduler.is_task_ready(task) and task not in ready_queue:
                newly_ready_tasks.append(task)

        # Add newly ready tasks to queue
        ready_queue.extend(newly_ready_tasks)

    # Reset scheduling state for next iteration
    for task in tasks:
        if task.is_scheduled == SchedulingState.KERNEL_SCHEDULED:
            task.is_scheduled = SchedulingState.SCHEDULED

    return tasks


class ThreeTierKernelScheduler:
    """
    Kernel scheduler implementation for three-tier architecture.
    Manages task scheduling across device cores, edge nodes, and cloud.
    """

    def __init__(self, tasks, sequence_manager, upload_rates, download_rates):
        """Initialize scheduler state for three-tier architecture."""
        self.tasks = tasks
        self.sequence_manager = sequence_manager
        self.upload_rates = upload_rates
        self.download_rates = download_rates

        # Get system configuration
        self.num_device_cores = sequence_manager.num_device_cores
        self.num_edge_nodes = sequence_manager.num_edge_nodes
        self.num_edge_cores_per_node = sequence_manager.num_edge_cores_per_node

        # Initialize execution unit readiness tracking
        # 1. Device cores
        self.device_cores_ready = [0.0] * self.num_device_cores

        # 2. Edge cores (per node)
        self.edge_cores_ready = [
            [0.0] * self.num_edge_cores_per_node
            for _ in range(self.num_edge_nodes)
        ]

        # 3. Cloud phases
        self.cloud_upload_ready = 0.0  # Wireless sending channel
        self.cloud_compute_ready = 0.0  # Cloud computation resources
        self.cloud_download_ready = 0.0  # Wireless receiving channel

        # 4. Communication channels
        # Device <-> Edge channels
        self.device_to_edge_ready = [0.0] * self.num_edge_nodes
        self.edge_to_device_ready = [0.0] * self.num_edge_nodes

        # Edge <-> Cloud channels
        self.edge_to_cloud_ready = [0.0] * self.num_edge_nodes
        self.cloud_to_edge_ready = [0.0] * self.num_edge_nodes

        # Edge <-> Edge channels
        self.edge_to_edge_ready = [
            [0.0] * self.num_edge_nodes
            for _ in range(self.num_edge_nodes)
        ]

        # Initialize task readiness tracking vectors
        self.dependency_ready = {}  # Maps task ID -> number of unscheduled predecessors
        self.sequence_ready = {}  # Maps task ID -> sequence readiness state

        # Initialize the readiness tracking for all tasks
        self.initialize_task_readiness()

    def initialize_task_readiness(self):
        """Initialize task dependency and sequence readiness tracking."""
        # Track dependency readiness (similar to ready1 vector)
        for task in self.tasks:
            # Count number of predecessors not yet scheduled
            self.dependency_ready[task.id] = len(task.pred_tasks)

        # Track sequence readiness (similar to ready2 vector)
        for task in self.tasks:
            # Default: not in any sequence
            self.sequence_ready[task.id] = -1

        # Mark first task in each sequence as ready
        for seq_idx, sequence in enumerate(self.sequence_manager.sequences):
            if sequence:  # Non-empty sequence
                first_task_id = sequence[0]
                self.sequence_ready[first_task_id] = 0  # Ready in sequence

    def initialize_queue(self):
        """Initialize queue with tasks that are ready for scheduling."""
        from collections import deque

        # A task is ready if:
        # 1. All predecessors are scheduled (dependency_ready[id] == 0)
        # 2. It's the first task in its sequence (sequence_ready[id] == 0)
        ready_queue = deque()

        for task in self.tasks:
            if (self.dependency_ready[task.id] == 0 and
                    self.sequence_ready[task.id] == 0 and
                    task.is_scheduled != SchedulingState.KERNEL_SCHEDULED):
                ready_queue.append(task)

        return ready_queue

    def update_task_readiness(self, task):
        """Update dependency and sequence readiness state for a task."""
        # 1. Update dependency readiness (count unscheduled predecessors)
        unscheduled_preds = sum(
            1 for pred in task.pred_tasks
            if pred.is_scheduled != SchedulingState.KERNEL_SCHEDULED
        )
        self.dependency_ready[task.id] = unscheduled_preds

        # 2. Update sequence readiness
        # Find task's sequence and position
        for seq_idx, sequence in enumerate(self.sequence_manager.sequences):
            if task.id in sequence:
                task_pos = sequence.index(task.id)

                if task_pos == 0:
                    # First task in sequence is ready if all predecessors scheduled
                    self.sequence_ready[task.id] = 0
                else:
                    # Check if previous task in sequence is scheduled
                    prev_task_id = sequence[task_pos - 1]
                    prev_task = next((t for t in self.tasks if t.id == prev_task_id), None)

                    if prev_task and prev_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED:
                        self.sequence_ready[task.id] = 0  # Ready in sequence
                    else:
                        self.sequence_ready[task.id] = 1  # Not ready in sequence

                break  # Found the sequence containing this task

    def compute_edge_task_ready_time(self, task, target_edge_id):
        """
        Revised calculation of ready time for executing task on target_edge_id.
        Ensures edge-to-edge transfers are properly accounted for.
        """
        if not task.pred_tasks:
            return 0.0  # Entry task has no predecessors

        max_ready_time = 0.0

        for pred in task.pred_tasks:
            if pred.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')  # Predecessor not scheduled yet

            ready_time = 0.0

            # === Edge-to-Edge Transfer Enhancement ===
            if pred.execution_tier == ExecutionTier.EDGE:
                if not pred.edge_assignment:
                    return float('inf')

                pred_edge_id = pred.edge_assignment.edge_id - 1  # 0-based

                # Same edge - no transfer needed
                if pred_edge_id == target_edge_id:
                    if not hasattr(pred, 'FT_edge') or pred_edge_id not in pred.FT_edge:
                        return float('inf')
                    ready_time = pred.FT_edge[pred_edge_id]
                else:
                    # Different edges - need to transfer data
                    if not hasattr(pred, 'FT_edge') or pred_edge_id not in pred.FT_edge:
                        return float('inf')

                    # Get edge execution finish time
                    pred_finish = pred.FT_edge[pred_edge_id]

                    # Get edge-to-edge transfer parameters
                    data_key = f'edge{pred_edge_id + 1}_to_edge{target_edge_id + 1}'
                    data_size = task.data_sizes.get(data_key, 1.0)
                    rate = self.upload_rates.get(data_key, 1.0)

                    # Calculate transfer start time considering channel availability
                    transfer_start = max(pred_finish, self.edge_to_edge_ready[pred_edge_id][target_edge_id])
                    transfer_time = data_size / rate if rate > 0 else 0

                    # Calculate ready time after transfer
                    ready_time = transfer_start + transfer_time

                    # Update channel availability
                    self.edge_to_edge_ready[pred_edge_id][target_edge_id] = ready_time

            # Handle other predecessor types
            elif pred.execution_tier == ExecutionTier.DEVICE:
                pred_finish = pred.FT_l
                if pred_finish <= 0:
                    return float('inf')

                # Calculate device-to-edge transfer
                data_key = f'device_to_edge{target_edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = self.upload_rates.get(data_key, 1.0)

                channel_avail = self.device_to_edge_ready[target_edge_id]
                transfer_start = max(pred_finish, channel_avail)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time

                # Update channel availability
                self.device_to_edge_ready[target_edge_id] = ready_time

            elif pred.execution_tier == ExecutionTier.CLOUD:
                pred_finish = pred.FT_c
                if pred_finish <= 0:
                    return float('inf')

                # Calculate cloud-to-edge transfer
                data_key = f'cloud_to_edge{target_edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = self.download_rates.get(data_key, 1.0)

                channel_avail = self.cloud_to_edge_ready[target_edge_id]
                transfer_start = max(pred_finish, channel_avail)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time

                # Update channel availability
                self.cloud_to_edge_ready[target_edge_id] = ready_time

            max_ready_time = max(max_ready_time, ready_time)

        return max_ready_time

    def is_task_ready(self, task):
        """Check if a task is ready for scheduling."""
        return (self.dependency_ready[task.id] == 0 and
                self.sequence_ready[task.id] == 0 and
                task.is_scheduled != SchedulingState.KERNEL_SCHEDULED)

    def schedule_device_task(self, task):
        """Schedule a task for execution on a device core."""
        # Verify task is assigned to device
        if task.execution_tier != ExecutionTier.DEVICE or task.device_core < 0:
            logger.error(f"Task {task.id} not properly assigned to device")
            return

        core_id = task.device_core

        # Calculate ready time from predecessors
        pred_ready_time = self.calculate_device_ready_time(task)

        # Determine actual start time based on core availability
        start_time = max(pred_ready_time, self.device_cores_ready[core_id])

        # Store start time for concurrency tracking
        if not hasattr(task, 'execution_unit_task_start_times') or task.execution_unit_task_start_times is None:
            task.execution_unit_task_start_times = [-1.0] * len(self.sequence_manager.sequences)

        task.execution_unit_task_start_times[core_id] = start_time

        # Calculate finish time
        execution_time = task.local_execution_times[core_id]
        finish_time = start_time + execution_time

        # Update task timing information
        task.RT_l = pred_ready_time
        task.FT_l = finish_time
        task.execution_finish_time = finish_time

        # Clear cloud and edge execution times
        task.RT_ws = task.FT_ws = task.RT_c = task.FT_c = task.RT_wr = task.FT_wr = -1
        if hasattr(task, 'FT_edge'):
            task.FT_edge = {}
        if hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}

        # Update core availability
        self.device_cores_ready[core_id] = finish_time

    def schedule_edge_task(self, task):
        """
        Enhanced edge task scheduling function with improved channel management
        """
        # Verify task is assigned to edge
        if task.execution_tier != ExecutionTier.EDGE or not task.edge_assignment:
            logger.error(f"Task {task.id} not properly assigned to edge")
            return

        # Get edge node and core (convert to 0-based)
        edge_id = task.edge_assignment.edge_id - 1
        core_id = task.edge_assignment.core_id - 1

        # Calculate ready time from predecessors and data transfers
        pred_ready_time = self.compute_edge_task_ready_time(task, edge_id)

        # Determine actual start time based on edge core availability
        start_time = max(pred_ready_time, self.edge_cores_ready[edge_id][core_id])

        # Store start time for concurrency tracking
        if not hasattr(task, 'execution_unit_task_start_times') or task.execution_unit_task_start_times is None:
            task.execution_unit_task_start_times = [-1.0] * len(self.sequence_manager.sequences)

        # Find execution unit index in sequence_manager
        edge_unit = ExecutionUnit(ExecutionTier.EDGE, (edge_id, core_id))
        unit_idx = self.sequence_manager.unit_to_index_map.get(edge_unit, -1)

        if unit_idx >= 0:
            task.execution_unit_task_start_times[unit_idx] = start_time

        # Calculate finish time
        execution_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
        finish_time = start_time + execution_time

        # Update task timing information
        if not hasattr(task, 'RT_edge'):
            task.RT_edge = {}
        if not hasattr(task, 'FT_edge'):
            task.FT_edge = {}

        task.RT_edge[edge_id] = pred_ready_time
        task.FT_edge[edge_id] = finish_time
        task.execution_finish_time = finish_time

        # Calculate edge-to-device transfer time for results
        edge_to_device_key = f'edge{edge_id + 1}_to_device'
        data_size = task.data_sizes.get(edge_to_device_key, 0.0)
        download_rate = self.download_rates.get(edge_to_device_key, 1.0)
        transfer_time = data_size / download_rate if download_rate > 0 else 0.0

        # Determine transfer start time based on channel availability
        transfer_start = max(finish_time, self.edge_to_device_ready[edge_id])
        receive_time = transfer_start + transfer_time

        # Update result receive time
        if not hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}
        task.FT_edge_receive[edge_id] = receive_time

        # Update edge core and channel availability
        self.edge_cores_ready[edge_id][core_id] = finish_time
        self.edge_to_device_ready[edge_id] = receive_time

        # Pre-calculate all potential edge-to-edge transfers
        for target_edge_id in range(self.num_edge_nodes):
            if target_edge_id != edge_id:
                data_key = f'edge{edge_id + 1}_to_edge{target_edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 1.0)
                rate = self.upload_rates.get(data_key, 1.0)

                # Calculate transfer time
                transfer_start = max(finish_time, self.edge_to_edge_ready[edge_id][target_edge_id])
                transfer_time = data_size / rate if rate > 0 else 0
                transfer_finish = transfer_start + transfer_time

                # Update channel availability and record in task
                self.edge_to_edge_ready[edge_id][target_edge_id] = transfer_finish

                if not hasattr(task, 'FT_edge_send'):
                    task.FT_edge_send = {}
                task.FT_edge_send[(edge_id, target_edge_id)] = transfer_finish

                logger.debug(f"Edge-to-Edge transfer: Task {task.id}, Edge {edge_id + 1}→{target_edge_id + 1}, "
                             f"Available at: {transfer_finish:.3f}")

        # Also calculate edge-to-cloud transfer time
        data_key = f'edge{edge_id + 1}_to_cloud'
        data_size = task.data_sizes.get(data_key, 1.0)
        rate = self.upload_rates.get(data_key, 1.0)

        transfer_start = max(finish_time, self.edge_to_cloud_ready[edge_id])
        transfer_time = data_size / rate if rate > 0 else 0
        transfer_finish = transfer_start + transfer_time

        # Update channel availability
        self.edge_to_cloud_ready[edge_id] = transfer_finish

        if not hasattr(task, 'FT_edge_send'):
            task.FT_edge_send = {}
        task.FT_edge_send[('edge', 'cloud')] = transfer_finish

        logger.debug(f"Edge-to-Cloud transfer: Task {task.id}, Edge {edge_id + 1}→Cloud, "
                     f"Available at: {transfer_finish:.3f}")

        # Clear device and cloud execution times
        task.RT_l = task.FT_l = -1
        task.RT_ws = task.FT_ws = task.RT_c = task.FT_c = task.RT_wr = task.FT_wr = -1

    def schedule_cloud_task(self, task):
        """Schedule a task for three-phase execution on the cloud."""
        # Verify task is assigned to cloud
        if task.execution_tier != ExecutionTier.CLOUD:
            logger.error(f"Task {task.id} not properly assigned to cloud")
            return

        # ---- Phase 1: Upload ----
        # Calculate ready time for cloud upload
        upload_ready_time = self.calculate_cloud_upload_ready_time(task)

        # Determine actual upload start time based on channel availability
        upload_start = max(upload_ready_time, self.cloud_upload_ready)

        # Store start time for concurrency tracking
        if not hasattr(task, 'execution_unit_task_start_times') or task.execution_unit_task_start_times is None:
            task.execution_unit_task_start_times = [-1.0] * len(self.sequence_manager.sequences)

        # Find cloud index in sequence_manager
        cloud_unit = ExecutionUnit(ExecutionTier.CLOUD)
        cloud_idx = self.sequence_manager.unit_to_index_map.get(cloud_unit, -1)

        if cloud_idx >= 0:
            task.execution_unit_task_start_times[cloud_idx] = upload_start

        # Calculate upload finish time
        upload_time = task.cloud_execution_times[0]  # T_send
        upload_finish = upload_start + upload_time

        # ---- Phase 2: Cloud Computation ----
        # Calculate ready time for cloud computation
        compute_ready_time = max(
            upload_finish,  # Must finish uploading
            self.cloud_compute_ready  # Cloud resources must be available
        )

        # Calculate computation finish time
        compute_time = task.cloud_execution_times[1]  # T_cloud
        compute_finish = compute_ready_time + compute_time

        # ---- Phase 3: Download ----
        # Calculate ready time for result download
        download_ready_time = max(
            compute_finish,  # Must finish computation
            self.cloud_download_ready  # Download channel must be available
        )

        # Calculate download finish time
        download_time = task.cloud_execution_times[2]  # T_receive
        download_finish = download_ready_time + download_time

        # Update task timing information
        task.RT_ws = upload_ready_time
        task.FT_ws = upload_finish
        task.RT_c = compute_ready_time
        task.FT_c = compute_finish
        task.RT_wr = download_ready_time
        task.FT_wr = download_finish
        task.execution_finish_time = download_finish

        # Clear device and edge execution times
        task.RT_l = task.FT_l = -1
        if hasattr(task, 'FT_edge'):
            task.FT_edge = {}
        if hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}

        # Update channel and resource availability
        self.cloud_upload_ready = upload_finish
        self.cloud_compute_ready = compute_finish
        self.cloud_download_ready = download_finish

    def calculate_device_ready_time(self, task):
        """Calculate ready time for a task on a device core."""
        if not task.pred_tasks:
            return 0.0  # Entry task

        max_ready_time = 0.0

        for pred in task.pred_tasks:
            pred_finish = 0.0

            # Different logic based on predecessor execution tier
            if pred.execution_tier == ExecutionTier.DEVICE:
                # Local execution - results available immediately
                pred_finish = pred.FT_l

            elif pred.execution_tier == ExecutionTier.CLOUD:
                # Cloud execution - must wait for download
                pred_finish = pred.FT_wr

            elif pred.execution_tier == ExecutionTier.EDGE:
                # Edge execution - must wait for edge-to-device transfer
                if pred.edge_assignment:
                    edge_id = pred.edge_assignment.edge_id - 1  # 0-based
                    if hasattr(pred, 'FT_edge_receive') and edge_id in pred.FT_edge_receive:
                        pred_finish = pred.FT_edge_receive[edge_id]
                    else:
                        # Fallback to edge execution finish time
                        pred_finish = pred.FT_edge.get(edge_id, 0.0)

            max_ready_time = max(max_ready_time, pred_finish)

        return max_ready_time

    def calculate_cloud_upload_ready_time(self, task):
        """
        Revised calculation of ready time for cloud upload.
        Ensures edge-to-cloud transfers are properly accounted for.
        """
        if not task.pred_tasks:
            return max(self.cloud_upload_ready, 0.0)  # Consider channel availability

        max_pred_finish = 0.0

        for pred in task.pred_tasks:
            if pred.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')  # Predecessor not scheduled yet

            pred_finish = 0.0

            if pred.execution_tier == ExecutionTier.DEVICE:
                pred_finish = pred.FT_l
            elif pred.execution_tier == ExecutionTier.CLOUD:
                pred_finish = pred.FT_wr
            elif pred.execution_tier == ExecutionTier.EDGE:
                if pred.edge_assignment:
                    edge_id = pred.edge_assignment.edge_id - 1  # 0-based
                    if hasattr(pred, 'FT_edge') and edge_id in pred.FT_edge:
                        pred_finish = pred.FT_edge[edge_id]
                    else:
                        pred_finish = 0.0

                    # Calculate edge-to-cloud transfer time
                    data_key = f'edge{edge_id + 1}_to_cloud'
                    data_size = task.data_sizes.get(data_key, 1.0)
                    rate = self.upload_rates.get(data_key, 1.0)
                    transfer_time = data_size / rate if rate > 0 else 0.0

                    # Determine transfer start time based on edge finish and channel availability
                    transfer_start = max(pred_finish, self.edge_to_cloud_ready[edge_id])
                    pred_finish = transfer_start + transfer_time

            max_pred_finish = max(max_pred_finish, pred_finish)

        # Consider wireless channel availability
        ready_time = max(max_pred_finish, self.cloud_upload_ready)

        return ready_time


if __name__ == "__main__":
    # 1) Realistic network conditions
    upload_rates, download_rates = generate_realistic_network_conditions()

    # 2) Generate mobile power models (which includes 'device' cores and 'rf')
    mobile_power_models = generate_realistic_power_models(device_type='mobile', battery_level=65)
    device_power_profiles = mobile_power_models.get('device', {})
    wireless_rf_power_profiles = mobile_power_models.get('rf', {})

    # 3) Build or define your DAG tasks
    task20 = Task(id=20, succ_task=[])
    task19 = Task(id=19, succ_task=[])
    task18 = Task(id=18, succ_task=[])
    task17 = Task(id=17, succ_task=[])
    task16 = Task(id=16, succ_task=[task19])
    task15 = Task(id=15, succ_task=[task19])
    task14 = Task(id=14, succ_task=[task18, task19])
    task13 = Task(id=13, succ_task=[task17, task18])
    task12 = Task(id=12, succ_task=[task17])
    task11 = Task(id=11, succ_task=[task15, task16])
    task10 = Task(id=10, succ_task=[task11, task15])
    task9 = Task(id=9, succ_task=[task13, task14])
    task8 = Task(id=8, succ_task=[task12, task13])
    task7 = Task(id=7, succ_task=[task12])
    task6 = Task(id=6, succ_task=[task10, task11])
    task5 = Task(id=5, succ_task=[task9, task10])
    task4 = Task(id=4, succ_task=[task8, task9])
    task3 = Task(id=3, succ_task=[task7, task8])
    task2 = Task(id=2, succ_task=[task7, task8])
    task1 = Task(id=1, succ_task=[task7])

    # Set predecessors
    task1.pred_tasks = []
    task2.pred_tasks = []
    task3.pred_tasks = []
    task4.pred_tasks = []
    task5.pred_tasks = []
    task6.pred_tasks = []
    task7.pred_tasks = [task1, task2, task3]
    task8.pred_tasks = [task3, task4]
    task9.pred_tasks = [task4, task5]
    task10.pred_tasks = [task5, task6]
    task11.pred_tasks = [task6, task10]
    task12.pred_tasks = [task7, task8]
    task13.pred_tasks = [task8, task9]
    task14.pred_tasks = [task9, task10]
    task15.pred_tasks = [task10, task11]
    task16.pred_tasks = [task11]
    task17.pred_tasks = [task12, task13]
    task18.pred_tasks = [task13, task14]
    task19.pred_tasks = [task14, task15, task16]
    task20.pred_tasks = [task12]

    predefined_tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12, task13,
                        task14, task15, task16, task17, task18, task19, task20]

    # 4) Enhance tasks with complexity, data sizes, etc.
    tasks = add_task_attributes(predefined_tasks=predefined_tasks)

    print("\nTask Graph Summary:")
    for t in tasks:
        preds = [p.id for p in t.pred_tasks]
        succs = [s.id for s in t.succ_tasks]
        task_type = getattr(t, 'task_type', 'unknown')
        tier_str = t.execution_tier.name if hasattr(t, 'execution_tier') else "UNASSIGNED"
        print(f"Task {t.id}: Type={task_type}, Tier={tier_str}, "
              f"C={t.complexity:.2f}, D={t.data_intensity:.2f}")
        print(f"  Predecessors: {preds} | Successors: {succs}")
        print("  DataSizes:", t.data_sizes)
        print()

    # 5) Primary assignment => pick best single-task location
    primary_assignment(tasks)

    # 6) Priority ranking (HEFT-like)
    task_prioritizing(tasks)

    # 7) Build ThreeTierTaskScheduler
    scheduler = ThreeTierTaskScheduler(
        tasks,
        num_cores=3,
        num_edge_nodes=2,
        edge_cores_per_node=2,
        upload_rates=upload_rates,
        download_rates=download_rates
    )

    # 8) Topological+priority scheduling
    scheduler.schedule_tasks_topo_priority()

    # 9) Print final schedule (basic overview)
    print("\nFinal Task Schedule:")
    for t in tasks:
        ft = getattr(t, 'execution_finish_time', None)
        tier_name = t.execution_tier.name if hasattr(t, 'execution_tier') else "UNASSIGNED"
        if t.execution_tier == ExecutionTier.DEVICE:
            res_str = f"Device Core {t.device_core}"
        elif t.execution_tier == ExecutionTier.EDGE and t.edge_assignment:
            res_str = f"Edge(Node={t.edge_assignment.edge_id}, Core={t.edge_assignment.core_id})"
        else:
            res_str = "Cloud"
        print(f"Task {t.id} => Tier={tier_name}, Finish={ft:.2f}, Resource={res_str}")

    # 10) Calculate total time and energy
    T_total = total_time_3tier(tasks)
    E_total = total_energy_3tier_with_rf(
        tasks,
        device_power_profiles=device_power_profiles,
        rf_power=wireless_rf_power_profiles,
        upload_rates=upload_rates,
        default_signal_strength=70.0
    )

    print("\n=== SCHEDULING RESULTS ===")
    print(f"Total Completion Time (Three-Tier): {T_total:.2f}")
    print(f"Total Energy Consumption: {E_total:.2f} Joules")

    # 11) Print a fully formatted schedule table (showing start times)
    schedule_str = format_schedule_3tier(tasks, scheduler)
    print("\nFormatted schedule (3-tier concurrency-based start times):")
    print(schedule_str)

    # 12) Validate the schedule to check for dependency violations
    is_valid, violations = validate_task_dependencies(tasks)
    if is_valid:
        print("\nSchedule is valid: no dependency violations.")
    else:
        print("\nSchedule has DAG/time violations:")
        for v in violations:
            print(f" - {v}")

    # ========================================================
    # STEP TWO: ENERGY OPTIMIZATION PHASE
    # ========================================================
    if is_valid:  # Only proceed with optimization if initial schedule is valid
        print("\n=== STARTING ENERGY OPTIMIZATION PHASE ===")
        print(f"Initial completion time: {T_total:.2f}")
        print(f"Initial energy consumption: {E_total:.2f} Joules")

        # Create and populate a SequenceManager using the new method
        sequence_manager = SequenceManager(
            num_device_cores=3,
            num_edge_nodes=2,
            num_edge_cores_per_node=2
        ).build_sequences_from_tasks(tasks)

        # Set time constraint - typically we'd allow some flexibility
        T_max = T_total * 1.2  # Allow 20% increase in completion time

        # Create migration cache
        migration_cache = MigrationCache(capacity=10000)

        # Set power models
        power_models = {
            'device': device_power_profiles,
            'rf': wireless_rf_power_profiles
        }

        # Run the optimization algorithm
        print("\nRunning task migration optimization...\n")
        optimized_tasks, optimized_sequence_manager = optimize_task_scheduling(
            tasks=tasks,
            sequence_manager=sequence_manager,
            T_final=T_max,
            power_models=power_models,
            upload_rates=upload_rates,
            download_rates=download_rates,
            migration_cache=migration_cache,
            max_iterations=50
        )

        # Calculate the new metrics
        optimized_time = total_time_3tier(optimized_tasks)
        optimized_energy = total_energy_3tier_with_rf(
            tasks=optimized_tasks,
            device_power_profiles=device_power_profiles,
            rf_power=wireless_rf_power_profiles,
            upload_rates=upload_rates
        )

        # Print the optimized schedule
        print("\n=== OPTIMIZED SCHEDULE RESULTS ===")
        print(f"Final Completion Time: {optimized_time:.2f} (Initial: {T_total:.2f})")
        print(f"Final Energy Consumption: {optimized_energy:.2f} Joules (Initial: {E_total:.2f})")
        print(f"Energy Reduction: {((E_total - optimized_energy) / E_total * 100):.2f}%")
        print(f"Completion Time Change: {((optimized_time - T_total) / T_total * 100):.2f}%")

        # Compare task assignments before and after optimization
        print("\nTask Assignment Changes:")
        for task in optimized_tasks:
            # Find the original task
            original_task = next((t for t in tasks if t.id == task.id), None)

            if original_task:
                original_tier = original_task.execution_tier.name
                optimized_tier = task.execution_tier.name

                # Get resource information
                if original_task.execution_tier == ExecutionTier.DEVICE:
                    original_resource = f"Device Core {original_task.device_core}"
                elif original_task.execution_tier == ExecutionTier.EDGE and original_task.edge_assignment:
                    original_resource = f"Edge(Node={original_task.edge_assignment.edge_id}, Core={original_task.edge_assignment.core_id})"
                else:
                    original_resource = "Cloud"

                if task.execution_tier == ExecutionTier.DEVICE:
                    optimized_resource = f"Device Core {task.device_core}"
                elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
                    optimized_resource = f"Edge(Node={task.edge_assignment.edge_id}, Core={task.edge_assignment.core_id})"
                else:
                    optimized_resource = "Cloud"

                # Only print tasks that changed
                if original_tier != optimized_tier or original_resource != optimized_resource:
                    print(
                        f"Task {task.id}: {original_tier}({original_resource}) → {optimized_tier}({optimized_resource})")

        # Validate the optimized schedule
        is_valid, violations = validate_task_dependencies(optimized_tasks)
        if is_valid:
            print("\nOptimized schedule is valid: no dependency violations.")
        else:
            print("\nOptimized schedule has DAG/time violations:")
            for v in violations:
                print(f" - {v}")

        # Print the optimized schedule table
        optimized_schedule_str = format_schedule_3tier(optimized_tasks, scheduler)
        print("\nFormatted optimized schedule:")
        print(optimized_schedule_str)