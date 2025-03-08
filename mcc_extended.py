import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set, Any, NamedTuple, Callable
from copy import deepcopy
from collections import deque
import time as time_module
import random

# --------------------------------------------------
# 1. Logging setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------
# 2. Global execution-time dictionaries
#    (DEVICE, EDGE, and CLOUD core times)
# --------------------------------------------------
device_core_execution_times = {
    1: [9, 7, 5],
    2: [8, 6, 5],
    3: [6, 5, 4],
    4: [7, 5, 3],
    5: [5, 4, 2],
    6: [7, 6, 4],
    7: [8, 5, 3],
    8: [6, 4, 2],
    9: [5, 3, 2],
    10: [7, 4, 2],
    11: [12, 3, 3],
    12: [12, 8, 4],
    13: [11, 3, 2],
    14: [12, 11, 4],
    15: [13, 4, 2],
    16: [9, 7, 3],
    17: [9, 3, 3],
    18: [13, 9, 2],
    19: [10, 5, 3],
    20: [12, 5, 4]
}

edge_core_execution_times = {
    (1, 1, 1): 8, (1, 1, 2): 6, (1, 2, 1): 7, (1, 2, 2): 5,
    (2, 1, 1): 7, (2, 1, 2): 5, (2, 2, 1): 6, (2, 2, 2): 4,
}

cloud_execution_times = [3, 1, 1]

# Number of Edge Nodes (for quick reference if needed)
M = 2


class ExecutionTier(Enum):
    """Defines where a task can be executed in the three-tier architecture."""
    DEVICE = 0  # Mobile device (local cores)
    EDGE = 1  # Edge nodes (intermediate tier)
    CLOUD = 2  # Cloud platform


class SchedulingState(Enum):
    """Task scheduling algorithm states."""
    UNSCHEDULED = 0
    SCHEDULED = 1
    KERNEL_SCHEDULED = 2


@dataclass
class EdgeAssignment:
    """Tracks a task’s assignment to an edge node."""
    edge_id: int  # Which edge node (E_m where m is 1...M)
    core_id: int  # Which core on that edge node


class ExecutionUnit(NamedTuple):
    """
    Represents a single execution resource in the system:
      - tier: which tier (DEVICE, EDGE, CLOUD)
      - location: tuple describing core or node+core, depending on tier
    """
    tier: ExecutionTier
    location: Tuple[int, int] = None  # (node_id, core_id) for edge; (core_id,) for device; None for cloud

    def __str__(self):
        if self.tier == ExecutionTier.DEVICE:
            return f"Device(Core {self.location[0]})"
        elif self.tier == ExecutionTier.EDGE:
            return f"Edge(Node {self.location[0]}, Core {self.location[1]})"
        else:
            return "Cloud"


@dataclass
class TaskMigrationState:
    """
    Holds information about migrating a task from one tier (and location) to another, along with before/after metrics for time and energy.
    """
    time: float  # Total completion time after the migration
    energy: float  # Total energy consumption after the migration
    efficiency: float  # Energy reduction per unit time of increase
    task_id: int  # ID of the task being migrated

    source_tier: ExecutionTier
    target_tier: ExecutionTier
    source_location: Optional[Tuple[int, int]] = None
    target_location: Optional[Tuple[int, int]] = None

    time_increase: float = 0.0
    energy_reduction: float = 0.0
    old_task_finish_time: float = 0.0
    new_task_finish_time: float = 0.0

    migration_complexity: int = 0

    def __post_init__(self):
        """
        Assign a 'migration_complexity' value based on how far or complex the migration is
        """
        if self.source_tier == self.target_tier:
            if self.source_tier == ExecutionTier.DEVICE:
                self.migration_complexity = 1
            elif self.source_tier == ExecutionTier.EDGE:
                # Edge to edge
                if (self.source_location and self.target_location
                        and self.source_location[0] == self.target_location[0]):
                    # Same edge node, different core
                    self.migration_complexity = 1
                else:
                    # Different edge nodes
                    self.migration_complexity = 2
            else:
                # Cloud to cloud (rare or not used)
                self.migration_complexity = 0
        else:
            # Cross-tier
            src_idx = self.source_tier.value
            tgt_idx = self.target_tier.value
            self.migration_complexity = 1 + abs(src_idx - tgt_idx)


class SequenceManager:
    """
    Manages a list of task-execution sequences—one sequence per execution unit
    (device cores, edge cores, and cloud). Provides convenient methods to
    set and retrieve those sequences.
    """

    def __init__(self, num_device_cores: int, num_edge_nodes: int, num_edge_cores_per_node: int):
        self.num_device_cores = num_device_cores
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores_per_node = num_edge_cores_per_node

        # Calculate total number of execution units: device cores + edge cores + 1 cloud
        self.total_units = (
                num_device_cores
                + (num_edge_nodes * num_edge_cores_per_node)
                + 1  # 1 for the Cloud
        )

        # Initialize an empty list of sequences
        self.sequences = [[] for _ in range(self.total_units)]

        # Map each (tier, location) → sequence index
        self.unit_to_index_map = {}

        # Populate mapping for device cores
        for core_id in range(num_device_cores):
            unit = ExecutionUnit(ExecutionTier.DEVICE, (core_id,))
            self.unit_to_index_map[unit] = core_id

        # Populate mapping for each edge core
        offset = num_device_cores
        for node_id in range(num_edge_nodes):
            for core_id in range(num_edge_cores_per_node):
                unit = ExecutionUnit(ExecutionTier.EDGE, (node_id, core_id))
                index = offset + node_id * num_edge_cores_per_node + core_id
                self.unit_to_index_map[unit] = index

        # Finally, add cloud
        cloud_index = self.total_units - 1
        self.unit_to_index_map[ExecutionUnit(ExecutionTier.CLOUD)] = cloud_index

    def set_all_sequences(self, sequences: List[List[int]]) -> None:
        if len(sequences) != self.total_units:
            raise ValueError(f"Expected {self.total_units} sequences, got {len(sequences)}")

        # Use deepcopy to avoid accidental side effects
        self.sequences = deepcopy(sequences)


@dataclass
class OptimizationMetrics:
    """
    Tracks the evolution of key metrics across multiple optimization
    iterations (e.g., total time, total energy, number of migrations).
    """
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
        # Check if time constraint was violated (allow +10% from the initial schedule)
        if new_time > self.initial_time * 1.1:
            self.time_violations += 1
        # Check if energy improved
        if new_energy < self.current_energy:
            self.energy_improvements += 1
            self.migrations += 1
        # Update current values
        self.current_time = new_time
        self.current_energy = new_energy
        # Update best-so-far
        if new_energy < self.best_energy:
            self.best_energy = new_energy
            self.best_time = new_time

    def get_summary(self) -> Dict[str, Any]:
        return {
            "initial_time": self.initial_time,
            "initial_energy": self.initial_energy,
            "final_time": self.current_time,
            "final_energy": self.current_energy,
            "time_change_pct": (
                (self.current_time - self.initial_time) / self.initial_time * 100
                if self.initial_time != 0 else 0
            ),
            "energy_reduction_pct": (
                (self.initial_energy - self.current_energy) / self.initial_energy * 100
                if self.initial_energy != 0 else 0
            ),
            "iterations": self.iterations,
            "migrations": self.migrations,
            "evaluations": self.evaluations,
            "time_violations": self.time_violations,
            "energy_improvements": self.energy_improvements,
            "elapsed_time": self.elapsed_time,
            "evaluation_rate": (
                self.evaluations / self.elapsed_time if self.elapsed_time > 0 else 0
            ),
        }


class MigrationCache:
    """
    Simple caching mechanism to store the results of
    task-migration evaluations (time, energy) and avoid repeated re-calculations.
    """

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


# --------------------------------------------------
# 1. Task Class
# --------------------------------------------------
class Task:
    """
    Represents a single task in the DAG, with information about:
      - ID and predecessor/successor relationships
      - Execution times on device, edge, and cloud
      - Execution-tier assignment (device, edge, or cloud)
      - Various timestamps (finish times, ready times) for scheduling
    """

    def __init__(self, id, pred_task=None, succ_task=None):
        # Basic IDs and links
        self.id = id  # Task identifier: v_i
        self.succ_tasks = succ_task or []  # succ(v_i): immediate successors
        self.pred_tasks = pred_task or []  # pred(v_i): immediate predecessors

        # ------------------------------------------------
        # Execution time parameters
        # ------------------------------------------------
        # Local (device) execution times: T_i^(l,k)
        self.core_execution_times = device_core_execution_times.get(id, [])
        # Cloud execution times: [T_i^s, T_i^c, T_i^r]
        self.cloud_execution_times = cloud_execution_times  # global default: [3, 1, 1]

        # Edge (node/core) execution times: T_i^(e,m), if available
        #   - e.g. edge_core_execution_times[(task_id, edge_id, core_id)]
        self.edge_core_execution_times = {}
        for key, value in edge_core_execution_times.items():
            (tid, e_id, c_id) = key
            if tid == self.id:
                self.edge_core_execution_times[(e_id, c_id)] = value

        # ------------------------------------------------
        # Current assignment and path
        # ------------------------------------------------
        self.execution_tier = ExecutionTier.DEVICE  # Default to device
        self.device_core = -1  # If on device, which core
        self.edge_assignment = None  # If on edge, EdgeAssignment object
        self.execution_path = []  # Tracks any migration path
        # ------------------------------------------------
        # Various finish times
        # ------------------------------------------------
        self.FT_l = 0  # Local finish time
        self.FT_ws = 0  # Wireless sending finish time (upload)
        self.FT_c = 0  # Cloud computation finish time
        self.FT_wr = 0  # Wireless receiving finish time (download)
        # Edge-specific finish times, if executed on edge
        self.FT_edge = {}  # e.g. FT_edge[m] for finishing on edge m
        self.FT_edge_send = {}  # Track edge→{cloud, device, or edge} send completion
        self.FT_edge_receive = {}  # Track edge→device or cloud→edge receive completion
        # Ready times for each phase
        self.RT_l = -1
        self.RT_ws = -1
        self.RT_c = -1
        self.RT_wr = -1
        self.RT_edge = {}  # RT_edge[m]
        self.RT_edge_send = {}  # RT_edge_send[(m, target)]
        # Scheduling attributes
        self.priority_score = None
        self.is_scheduled = SchedulingState.UNSCHEDULED
        self.completion_time = -1
        self.execution_unit_task_start_times = None

    def calculate_local_finish_time(self, core, start_time):
        """
        Returns the finish time if the task runs on a particular local core,
        starting at 'start_time'.
        """
        if core < 0 or core >= len(self.core_execution_times):
            raise ValueError(f"Invalid core {core} for task {self.id}")
        return start_time + self.core_execution_times[core]

    def calculate_cloud_finish_times(self, upload_start_time):
        """
        For convenience, calculates the three cloud phases (upload, compute, download)
        finish times given an initial 'upload_start_time'.
        """
        upload_finish = upload_start_time + self.cloud_execution_times[0]
        cloud_finish = upload_finish + self.cloud_execution_times[1]
        download_finish = cloud_finish + self.cloud_execution_times[2]
        return upload_finish, cloud_finish, download_finish

    def calculate_edge_finish_time(self, edge_id, core_id, start_time):
        """
        Returns the finish time if the task runs on a particular edge's core,
        starting at 'start_time'.
        """
        exec_time = self.get_edge_execution_time(edge_id, core_id)
        return float('inf') if exec_time is None else (start_time + exec_time)

    def get_edge_execution_time(self, edge_id, core_id):
        """
        Retrieve or approximate T_i^(e, m) for the given (edge_id, core_id).
        If not found, we do a fallback heuristic.
        """
        # Check explicit dictionary
        key = (edge_id, core_id)
        if key in self.edge_core_execution_times:
            return self.edge_core_execution_times[key]

        # Fallback: if no explicit edge times, try an average device vs. cloud approach
        if self.core_execution_times:
            avg_local = sum(self.core_execution_times) / len(self.core_execution_times)
            # We blend local vs. cloud to guess an edge time
            return (avg_local + sum(self.cloud_execution_times)) / 2.0

        # Ultimate fallback
        return 5.0

    def get_overall_finish_time(self):
        """
        Returns the overall completion time of this task by checking
        whichever finish time is relevant (e.g., local, cloud, or edge).
        """
        finish_times = []

        # If local
        if self.FT_l > 0:
            finish_times.append(self.FT_l)
        # If cloud
        if self.FT_wr > 0:
            finish_times.append(self.FT_wr)
        # If edge
        if self.FT_edge_receive:
            # Edge results are typically considered complete after returning to device
            max_receive = max(self.FT_edge_receive.values())
            finish_times.append(max_receive)

        return max(finish_times) if finish_times else -1


# --------------------------------------------------
# 1. ThreeTierKernelScheduler
# --------------------------------------------------
def _calculate_device_to_cloud_ready_time(task):
    """
    If migrating from device to cloud, find earliest time the device→cloud channel
    is available after all dependencies are scheduled.
    """
    if not task.pred_tasks:
        return 0
    max_ready = 0
    for pred in task.pred_tasks:
        if pred.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            return float('inf')
        if pred.execution_tier == ExecutionTier.DEVICE:
            max_ready = max(max_ready, pred.FT_l)
        elif pred.execution_tier == ExecutionTier.CLOUD:
            max_ready = max(max_ready, pred.FT_ws)
        elif pred.execution_tier == ExecutionTier.EDGE:
            # Must come back to device first
            if pred.edge_assignment:
                e_id = pred.edge_assignment.edge_id - 1
                if e_id in pred.FT_edge_receive:
                    max_ready = max(max_ready, pred.FT_edge_receive[e_id])
    return max_ready


class ThreeTierKernelScheduler:
    def __init__(self, tasks: List[Any], sequences: List[List[int]], num_device_cores=3, num_edge_nodes=2,
                 num_edge_cores_per_node=2):
        """
        Initialize with the list of all tasks, existing sequences,
        and hardware configuration (device cores, edge nodes, cores per edge).
        """
        self.tasks = tasks
        self.sequences = sequences
        self.num_device_cores = num_device_cores
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores_per_node = num_edge_cores_per_node
        # Track readiness for device cores, edge cores, channels (upload/download)
        self.device_cores_ready = [0] * num_device_cores
        self.edge_cores_ready = [[0] * num_edge_cores_per_node for _ in range(num_edge_nodes)]
        # Channel readiness (time when channel is free)
        self.device_to_cloud_ready = 0
        self.cloud_to_device_ready = 0
        self.device_to_edge_ready = [0] * num_edge_nodes
        self.edge_to_device_ready = [0] * num_edge_nodes
        self.edge_to_cloud_ready = [0] * num_edge_nodes
        self.cloud_to_edge_ready = [0] * num_edge_nodes
        self.edge_to_edge_ready = [[0] * num_edge_nodes for _ in range(num_edge_nodes)]
        # Channel usage logs (for analysis); each entry is a queue of usage intervals
        self.channel_contention_queue = {
            'device_to_cloud': [], 'cloud_to_device': []
        }
        for edge_id in range(num_edge_nodes):
            self.channel_contention_queue[f'device_to_edge{edge_id + 1}'] = []
            self.channel_contention_queue[f'edge{edge_id + 1}_to_device'] = []
            self.channel_contention_queue[f'edge{edge_id + 1}_to_cloud'] = []
            self.channel_contention_queue[f'cloud_to_edge{edge_id + 1}'] = []
            for other_edge in range(num_edge_nodes):
                if edge_id != other_edge:
                    self.channel_contention_queue[f'edge{edge_id + 1}_to_edge{other_edge + 1}'] = []

        # For each task, we track how many predecessors remain unscheduled
        # Also track its "sequence_ready" status
        self.dependency_ready, self.sequence_ready = self._initialize_task_state()

    def _initialize_task_state(self):
        """
        Initialize arrays that track how many unscheduled predecessors remain for each task,
        and which tasks are first in their respective sequences.
        """
        dependency_ready = [len(task.pred_tasks) for task in self.tasks]
        sequence_ready = [-1] * len(self.tasks)

        # For each sequence, mark the first task as sequence_ready=0
        for sequence in self.sequences:
            if sequence:
                first_task_id = sequence[0]
                # Convert to 0-based index
                sequence_ready[first_task_id - 1] = 0

        return dependency_ready, sequence_ready

    def update_task_state(self, task):
        """
        Update the 'dependency_ready' and 'sequence_ready' fields
        for a given task, based on its predecessors and position in sequence.
        """
        task_idx = task.id - 1
        # Update the count of unscheduled predecessors
        self.dependency_ready[task_idx] = sum(
            1 for pred_task in task.pred_tasks
            if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
        )

        # Update sequence-based readiness (whether its sequence predecessor is finished)
        for s_idx, sequence in enumerate(self.sequences):
            if task.id in sequence:
                pos = sequence.index(task.id)
                if pos == 0:
                    # First in sequence
                    self.sequence_ready[task_idx] = 0
                else:
                    pred_id = sequence[pos - 1]
                    pred_obj = self.tasks[pred_id - 1]
                    # if predecessor is kernel-scheduled, mark this as 0
                    self.sequence_ready[
                        task_idx] = 0 if pred_obj.is_scheduled == SchedulingState.KERNEL_SCHEDULED else 1
                break

    def initialize_queue(self):
        """
        Build a queue of tasks that are ready for scheduling (dependency=0, sequence=0, not kernel-scheduled).
        """
        from collections import deque
        return deque(
            task for task in self.tasks
            if (
                    self.sequence_ready[task.id - 1] == 0
                    and self.dependency_ready[task.id - 1] == 0
                    and task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )
        )

    def schedule_device_task(self, task):
        """
        Schedules 'task' on its designated device core (task.device_core).
        Updates start & finish times, channel usage, etc.
        """
        core_id = task.device_core
        # Calculate data-ready time
        ready_time = self._calculate_device_ready_time(task)
        # Start after both core is free and data is ready
        start_time = max(self.device_cores_ready[core_id], ready_time)
        exec_time = task.core_execution_times[core_id]
        finish_time = start_time + exec_time
        self.device_cores_ready[core_id] = finish_time

        # Update the task’s local finish time
        task.FT_l = finish_time

        # Make sure we have an array of start-times for each resource
        total_units = (self.num_device_cores
                       + self.num_edge_nodes * self.num_edge_cores_per_node
                       + 1)  # Cloud is 1
        if not hasattr(task, 'execution_unit_task_start_times') or not task.execution_unit_task_start_times:
            task.execution_unit_task_start_times = [-1] * total_units

        task.execution_unit_task_start_times[core_id] = start_time

        # Clear edge/cloud times
        task.FT_edge = {}
        task.FT_edge_receive = {}
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        logger.debug(f"Scheduled task {task.id} on device core {core_id} from {start_time} to {finish_time}")

    def schedule_edge_task(self, task):
        """
        Schedules 'task' on the specified edge node & core (task.edge_assignment.edge_id/core_id).
        Respects data dependencies (device→edge, cloud→edge, or edge→edge).
        """
        if not task.edge_assignment:
            logger.error(f"Task {task.id} missing edge assignment")
            return False

        e_id = task.edge_assignment.edge_id - 1  # 0-based
        c_id = task.edge_assignment.core_id - 1

        # Compute earliest time the data is ready on this edge
        ready_time = self._calculate_edge_ready_time(task, e_id)
        if ready_time == float('inf'):
            logger.error(f"Task {task.id} has invalid dependencies for edge execution")
            return False

        # Start after edge core is free
        start_time = max(ready_time, self.edge_cores_ready[e_id][c_id])

        # Execution time
        exec_time = task.get_edge_execution_time(e_id + 1, c_id + 1)
        if exec_time is None:
            logger.error(f"No edge exec time for task {task.id}, edge {e_id + 1}, core {c_id + 1}")
            return False

        finish_time = start_time + exec_time
        self.edge_cores_ready[e_id][c_id] = finish_time

        # Update the task's edge finish dictionary
        if not hasattr(task, 'FT_edge'):
            task.FT_edge = {}
        task.FT_edge[e_id] = finish_time
        task.FT_edge[e_id + 1] = finish_time  # store 1-based as well

        # Calculate edge→device result transfer
        self._calculate_edge_to_device_transfer(task, e_id, finish_time)

        # Record the start time in the global array
        total_units = (self.num_device_cores
                       + self.num_edge_nodes * self.num_edge_cores_per_node
                       + 1)
        if not task.execution_unit_task_start_times:
            task.execution_unit_task_start_times = [-1] * total_units
        seq_index = self.get_edge_core_index(e_id, c_id)
        task.execution_unit_task_start_times[seq_index] = start_time

        # Clear device & cloud times
        task.FT_l = 0
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        # Mark the task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # Insert the task ID into the relevant sequence
        self.sequences[seq_index].append(task.id)
        return True

    def schedule_cloud_task(self, task):
        """
        Schedules 'task' in the cloud. This includes computing the
        upload time from source, the cloud compute time, and the
        download time back to device (or edge).
        """
        # We rely on global or external function 'update_cloud_timing' or similar approach
        # but here we do it inline for clarity:

        # 1) Figure out how the data arrives in the cloud (from device or edge)
        if hasattr(task, 'execution_tier'):
            source_tier = task.execution_tier
        else:
            source_tier = ExecutionTier.DEVICE

        # If needed, adjust times. (We assume we have a function to do so.)
        update_cloud_timing(task, self.tasks)  # This function is typically external

        # Possibly handle channel contentions for device->cloud or edge->cloud, etc.
        # For simplicity, we assume the function above accounts for them.

        # Mark as SCHEDULED
        task.is_scheduled = SchedulingState.SCHEDULED

        # Record the start time in the "cloud" slot of the array
        if not task.execution_unit_task_start_times:
            total_units = self.num_device_cores + (self.num_edge_nodes * self.num_edge_cores_per_node) + 1
            task.execution_unit_task_start_times = [-1] * total_units
        cloud_idx = self.num_device_cores + (self.num_edge_nodes * self.num_edge_cores_per_node)
        task.execution_unit_task_start_times[cloud_idx] = task.RT_ws

        # Clear device/edge fields
        task.FT_l = 0
        task.FT_edge = {}
        task.FT_edge_receive = {}

        logger.debug(f"Scheduled task {task.id} on cloud")

    def get_edge_core_index(self, edge_id, core_id):
        """
        Convert (edge_id, core_id) into the correct sequence index within 'self.sequences'.
        The device cores occupy indices [0..k-1],
        the edge cores occupy [k..k + M*C -1],
        and the cloud is at the last index.
        """
        return self.num_device_cores + edge_id * self.num_edge_cores_per_node + core_id

    # ------------------------------------------------------
    # Internal calculations for data readiness
    # ------------------------------------------------------
    def _calculate_device_ready_time(self, task):
        """
        When is the data ready for this task to start on device?
        We check all predecessors, plus any needed data transfers from edge or cloud.
        """
        if not task.pred_tasks:
            return 0
        max_ready = 0
        epsilon = 1e-6

        for pred in task.pred_tasks:
            if pred.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')
            if pred.execution_tier == ExecutionTier.DEVICE:
                max_ready = max(max_ready, pred.FT_l + epsilon)
            elif pred.execution_tier == ExecutionTier.CLOUD:
                max_ready = max(max_ready, pred.FT_wr + epsilon)
            elif pred.execution_tier == ExecutionTier.EDGE:
                if not pred.edge_assignment:
                    return float('inf')
                e_id = pred.edge_assignment.edge_id - 1
                if not hasattr(pred, 'FT_edge_receive') or e_id not in pred.FT_edge_receive:
                    # We compute it if missing
                    if e_id in pred.FT_edge:
                        edge_finish = pred.FT_edge[e_id]
                        data_key = f'edge{e_id + 1}_to_device'
                        rate = download_rates.get(data_key, 2.0)
                        data_size = 1.0
                        if pred.id in task_data_sizes and data_key in task_data_sizes[pred.id]:
                            data_size = task_data_sizes[pred.id][data_key]
                        transfer_time = data_size / rate if rate > 0 else 0
                        transfer_start = max(edge_finish, self.edge_to_device_ready[e_id])
                        receive_time = transfer_start + transfer_time + epsilon
                        self.edge_to_device_ready[e_id] = receive_time
                        if not hasattr(pred, 'FT_edge_receive'):
                            pred.FT_edge_receive = {}
                        pred.FT_edge_receive[e_id] = receive_time
                        max_ready = max(max_ready, receive_time)
                    else:
                        return float('inf')
                else:
                    max_ready = max(max_ready, pred.FT_edge_receive[e_id])

        return max_ready

    def _calculate_edge_ready_time(self, task, edge_id):
        if not task.pred_tasks:
            return 0
        max_ready = 0
        epsilon = 1e-6

        for pred in task.pred_tasks:
            if pred.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')
            # Check pred tier
            if pred.execution_tier == ExecutionTier.DEVICE:
                # device->edge
                pred_finish = pred.FT_l
                data_key = f'device_to_edge{edge_id + 1}'
                rate = upload_rates.get(data_key, 1.5)
                data_size = 3.0
                if task.id in task_data_sizes and data_key in task_data_sizes[task.id]:
                    data_size = task_data_sizes[task.id][data_key]
                transfer_time = data_size / rate if rate > 0 else 0
                transfer_start = max(pred_finish, self.device_to_edge_ready[edge_id])
                transfer_finish = transfer_start + transfer_time + epsilon
                self.device_to_edge_ready[edge_id] = transfer_finish
                max_ready = max(max_ready, transfer_finish)
            elif pred.execution_tier == ExecutionTier.CLOUD:
                # cloud->edge
                pred_finish = pred.FT_c
                data_key = f'cloud_to_edge{edge_id + 1}'
                rate = download_rates.get(data_key, 3.0)
                data_size = 1.0
                if task.id in task_data_sizes and data_key in task_data_sizes[task.id]:
                    data_size = task_data_sizes[task.id][data_key]
                transfer_time = data_size / rate if rate > 0 else 0
                transfer_start = max(pred_finish, self.cloud_to_edge_ready[edge_id])
                transfer_finish = transfer_start + transfer_time + epsilon
                self.cloud_to_edge_ready[edge_id] = transfer_finish
                max_ready = max(max_ready, transfer_finish)
            elif pred.execution_tier == ExecutionTier.EDGE:
                if not pred.edge_assignment:
                    return float('inf')
                p_eid = pred.edge_assignment.edge_id - 1
                if p_eid not in pred.FT_edge:
                    return float('inf')
                edge_finish = pred.FT_edge[p_eid]
                if p_eid == edge_id:
                    # Same edge node
                    max_ready = max(max_ready, edge_finish + epsilon)
                else:
                    # edge->edge
                    data_key = f'edge{p_eid + 1}_to_edge{edge_id + 1}'
                    rate = upload_rates.get(data_key, 3.0)
                    data_size = 1.5
                    if task.id in task_data_sizes and data_key in task_data_sizes[task.id]:
                        data_size = task_data_sizes[task.id][data_key]
                    transfer_time = data_size / rate if rate > 0 else 0
                    transfer_start = max(edge_finish, self.edge_to_edge_ready[p_eid][edge_id])
                    transfer_finish = transfer_start + transfer_time + epsilon
                    self.edge_to_edge_ready[p_eid][edge_id] = transfer_finish
                    max_ready = max(max_ready, transfer_finish)
        return max_ready

    def _calculate_edge_to_device_transfer(self, task, edge_id, edge_finish_time):
        """
        After finishing on edge 'edge_id', results typically go back to the device.
        This method calculates that timing.
        """
        data_key = f'edge{edge_id + 1}_to_device'
        rate_key = data_key
        if task.id in task_data_sizes and data_key in task_data_sizes[task.id]:
            data_size = task_data_sizes[task.id][data_key]
        else:
            data_size = 1.0
        rate = download_rates.get(rate_key, 2.0)
        download_time = data_size / rate if rate > 0 else 0

        epsilon = 1e-6
        download_start = max(edge_finish_time + epsilon, self.edge_to_device_ready[edge_id])
        download_finish = download_start + download_time
        self.edge_to_device_ready[edge_id] = download_finish

        if not hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}
        task.FT_edge_receive[edge_id] = download_finish
        task.FT_edge_receive[edge_id + 1] = download_finish

        self.channel_contention_queue[f'edge{edge_id + 1}_to_device'].append({
            'task_id': task.id,
            'start': download_start,
            'finish': download_finish
        })


class ThreeTierTaskScheduler:
    """
    Another high-level scheduler that decides, for each task,
    whether to execute on device, edge, or cloud, in a more
    monolithic scheduling pass.
    """

    def __init__(self, tasks, num_cores=3, num_edge_nodes=2, edge_cores_per_node=2):
        self.tasks = tasks
        self.k = num_cores  # number of device cores
        self.M = num_edge_nodes  # number of edge nodes
        self.edge_cores = edge_cores_per_node

        # Resource readiness
        self.core_earliest_ready = [0] * self.k
        self.edge_core_earliest_ready = [[0] * self.edge_cores for _ in range(self.M)]

        # Wireless channels to/from cloud
        self.ws_ready = 0  # device → cloud
        self.wr_ready = 0  # cloud → device

        # device → edge channels
        self.device_to_edge_ready = [0] * self.M
        # edge → device channels
        self.edge_to_device_ready = [0] * self.M

        # edge → cloud channels
        self.edge_to_cloud_ready = [0] * self.M
        # cloud → edge channels
        self.cloud_to_edge_ready = [0] * self.M

        # edge → edge
        self.edge_to_edge_ready = [[0] * self.M for _ in range(self.M)]

        # Channel contention logs
        self.channel_contention_queue = {
            'device_to_cloud': [],
            'cloud_to_device': []
        }
        for e in range(self.M):
            self.channel_contention_queue[f'device_to_edge{e + 1}'] = []
            self.channel_contention_queue[f'edge{e + 1}_to_device'] = []
            self.channel_contention_queue[f'edge{e + 1}_to_cloud'] = []
            self.channel_contention_queue[f'cloud_to_edge{e + 1}'] = []
            for other_e in range(self.M):
                if other_e != e:
                    self.channel_contention_queue[f'edge{e + 1}_to_edge{other_e + 1}'] = []

        # Prepare a list of execution sequences
        total_resources = self.k + self.M * self.edge_cores + 1
        self.sequences = [[] for _ in range(total_resources)]

    def get_edge_core_index(self, edge_id, core_id):
        return self.k + edge_id * self.edge_cores + core_id

    def get_cloud_index(self):
        return self.k + self.M * self.edge_cores

    def get_priority_ordered_tasks(self):
        """
        Return a list of task IDs sorted by their 'priority_score'
        (descending).
        """
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)  # higher priority first
        return [item[1] for item in task_priority_list]

    def classify_entry_tasks(self, priority_order):
        """
        Partition tasks into entry tasks (no predecessors) vs.
        non-entry tasks, preserving priority order.
        """
        entry_tasks = []
        non_entry_tasks = []
        for tid in priority_order:
            t = self.tasks[tid - 1]
            if not t.pred_tasks:
                entry_tasks.append(t)
            else:
                non_entry_tasks.append(t)
        return entry_tasks, non_entry_tasks

    def calculate_local_ready_time(self, task):
        """
        For local (device) execution, determine earliest time 'task' can start
        by checking all predecessor completion times (device or downloaded from cloud/edge).
        """
        if not task.pred_tasks:
            return 0
        max_ready_time = 0
        for pred_task in task.pred_tasks:
            if pred_task.is_scheduled != SchedulingState.SCHEDULED:
                return float('inf')
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                max_ready_time = max(max_ready_time, pred_task.FT_l)
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                max_ready_time = max(max_ready_time, pred_task.FT_wr)
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                if pred_task.edge_assignment:
                    e_id = pred_task.edge_assignment.edge_id - 1
                    if hasattr(pred_task, 'FT_edge_receive') and e_id in pred_task.FT_edge_receive:
                        max_ready_time = max(max_ready_time, pred_task.FT_edge_receive[e_id])
        return max_ready_time

    def calculate_cloud_upload_ready_time(self, task):
        """
        Earliest time the task can start uploading to the cloud, based on
        predecessor finish times, plus any needed data transfer to device first.
        """
        if not task.pred_tasks:
            return 0
        max_ready_time = 0
        for pred_task in task.pred_tasks:
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                max_ready_time = max(max_ready_time, pred_task.FT_l)
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                max_ready_time = max(max_ready_time, pred_task.FT_ws)
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                if pred_task.edge_assignment:
                    e_id = pred_task.edge_assignment.edge_id - 1
                    # If data was returned to device, we check that time
                    if hasattr(pred_task, 'FT_edge_receive') and e_id in pred_task.FT_edge_receive:
                        max_ready_time = max(max_ready_time, pred_task.FT_edge_receive[e_id])
        # Also consider the device→cloud channel readiness
        max_ready_time = max(max_ready_time, self.ws_ready)
        return max_ready_time

    def calculate_edge_ready_time(self, task, edge_id):
        """
        Earliest time data is available on the specified edge node
        (edge_id) for 'task'. Must consider device→edge, cloud→edge,
        or edge→edge transfers for each predecessor.
        """
        if not task.pred_tasks:
            return 0
        max_ready_time = 0
        for pred_task in task.pred_tasks:
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                max_ready_time = max(max_ready_time, pred_task.FT_l)
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                max_ready_time = max(max_ready_time, pred_task.FT_c)
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                if pred_task.edge_assignment:
                    pe_id = pred_task.edge_assignment.edge_id - 1
                    if pe_id == edge_id:
                        max_ready_time = max(max_ready_time, pred_task.FT_edge.get(pe_id, 0))
                    else:
                        # different edge → need edge→edge xfer
                        max_ready_time = max(max_ready_time, pred_task.FT_edge.get(pe_id, 0))
        max_ready_time = max(max_ready_time, self.device_to_edge_ready[edge_id])
        return max_ready_time

    def identify_optimal_local_core(self, task, ready_time=None):
        """
        Among all device cores, find which yields the earliest finish time
        for this 'task' if it starts not earlier than 'ready_time'.
        """
        if ready_time is None:
            ready_time = self.calculate_local_ready_time(task)
        best_finish = float('inf')
        best_core = -1
        best_start = float('inf')
        for core in range(self.k):
            start = max(ready_time, self.core_earliest_ready[core])
            finish = start + task.core_execution_times[core]
            if finish < best_finish:
                best_finish = finish
                best_core = core
                best_start = start
        return best_core, best_start, best_finish

    def identify_optimal_edge_core(self, task, edge_id, ready_time=None):
        """
        Among all cores of the specified edge_id, find which yields earliest finish time.
        """
        if ready_time is None:
            ready_time = self.calculate_edge_ready_time(task, edge_id)
        best_finish = float('inf')
        best_core = -1
        best_start = float('inf')

        for c_id in range(self.edge_cores):
            start = max(ready_time, self.edge_core_earliest_ready[edge_id][c_id])
            exec_time = task.get_edge_execution_time(edge_id + 1, c_id + 1)
            if exec_time is None:
                continue
            finish = start + exec_time
            if finish < best_finish:
                best_finish = finish
                best_core = c_id
                best_start = start
        return best_core, best_start, best_finish

    def identify_optimal_edge_node(self, task):
        """
        Among all edge nodes, pick the one (and core) that yields earliest finish.
        """
        best_finish = float('inf')
        best_edge_id = -1
        best_core_id = -1
        best_start = float('inf')
        for e_id in range(self.M):
            ready_time = self.calculate_edge_ready_time(task, e_id)
            if ready_time == float('inf'):
                continue
            c_id, st, ft = self.identify_optimal_edge_core(task, e_id, ready_time)
            if c_id >= 0 and ft < best_finish:
                best_finish = ft
                best_edge_id = e_id
                best_core_id = c_id
                best_start = st
        return best_edge_id, best_core_id, best_start, best_finish

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """
        Assign 'task' to device core 'core' with the given start/finish times.
        Update all relevant fields and resource readiness.
        """
        task.FT_l = finish_time
        task.execution_finish_time = finish_time

        total_units = self.k + self.M * self.edge_cores + 1
        task.execution_unit_task_start_times = [-1] * total_units
        task.execution_unit_task_start_times[core] = start_time

        self.core_earliest_ready[core] = finish_time

        task.assignment = core
        task.execution_tier = ExecutionTier.DEVICE
        task.device_core = core
        task.edge_assignment = None
        task.FT_edge = {}
        task.FT_edge_receive = {}
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[core].append(task.id)

    def schedule_on_edge(self, task, edge_id, core_id, start_time, finish_time):
        """
        Assign 'task' to the (edge_id, core_id) with the given start/finish times.
        Update all relevant fields and resource readiness, including
        result transfer to device if needed.
        """
        # Check for migrations if already was on some other resource
        # For simplicity, we omit the details or do a partial approach.

        # Set up array of start times
        total_units = self.k + self.M * self.edge_cores + 1
        task.execution_unit_task_start_times = task.execution_unit_task_start_times or [-1] * total_units

        # Record the start time in the correct sequence index
        seq_idx = self.get_edge_core_index(edge_id, core_id)
        task.execution_unit_task_start_times[seq_idx] = start_time

        # Mark the actual finish time on this edge
        if not hasattr(task, 'FT_edge'):
            task.FT_edge = {}
        task.FT_edge[edge_id] = finish_time
        task.FT_edge[edge_id + 1] = finish_time
        task.execution_finish_time = finish_time

        # compute transfer time back to device
        data_key = f'edge{edge_id + 1}_to_device'
        if task.id in task_data_sizes and data_key in task_data_sizes[task.id]:
            dsize = task_data_sizes[task.id][data_key]
        else:
            dsize = 1.0
        rate = download_rates.get(data_key, 2.0)
        down_time = dsize / rate if rate > 0 else 0

        e_start = max(finish_time, self.edge_to_device_ready[edge_id])
        e_finish = e_start + down_time
        self.edge_to_device_ready[edge_id] = e_finish

        if not hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}
        task.FT_edge_receive[edge_id] = e_finish
        task.FT_edge_receive[edge_id + 1] = e_finish

        self.channel_contention_queue[f'edge{edge_id + 1}_to_device'].append({
            'task_id': task.id,
            'start': e_start,
            'finish': e_finish
        })

        self.edge_core_earliest_ready[edge_id][core_id] = finish_time

        task.assignment = seq_idx
        task.execution_tier = ExecutionTier.EDGE
        task.device_core = -1
        task.edge_assignment = EdgeAssignment(edge_id=edge_id + 1, core_id=core_id + 1)
        task.execution_path = task.execution_path or []
        task.execution_path.append((ExecutionTier.EDGE, (edge_id + 1, core_id + 1)))

        task.FT_l = 0
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[seq_idx].append(task.id)

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready,
                          receive_finish):
        """
        Assign 'task' to the cloud. The times are precomputed with an external
        routine (e.g. 'calculate_cloud_phases_timing').
        """
        # Adjust if there's any ordering constraints, but typically
        # we assume it's consistent with the data we pass.

        task.RT_ws = send_ready
        task.FT_ws = send_finish
        task.RT_c = cloud_ready
        task.FT_c = cloud_finish
        task.RT_wr = receive_ready
        task.FT_wr = receive_finish

        task.execution_finish_time = receive_finish

        # Channel usage logs
        if task.execution_tier == ExecutionTier.DEVICE:
            self.ws_ready = send_finish
            self.channel_contention_queue['device_to_cloud'].append({
                'task_id': task.id, 'start': send_ready, 'finish': send_finish
            })
            self.wr_ready = receive_finish
            self.channel_contention_queue['cloud_to_device'].append({
                'task_id': task.id, 'start': receive_ready, 'finish': receive_finish
            })
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            e_id = task.edge_assignment.edge_id - 1
            k1 = f'edge{e_id + 1}_to_cloud'
            self.channel_contention_queue[k1].append({
                'task_id': task.id, 'start': send_ready, 'finish': send_finish
            })
            k2 = f'cloud_to_edge{e_id + 1}'
            self.channel_contention_queue[k2].append({
                'task_id': task.id, 'start': receive_ready, 'finish': receive_finish
            })

        total_units = self.k + self.M * self.edge_cores + 1
        if not hasattr(task, 'execution_unit_task_start_times') or not task.execution_unit_task_start_times:
            task.execution_unit_task_start_times = [-1] * total_units
        c_idx = self.get_cloud_index()
        task.execution_unit_task_start_times[c_idx] = send_ready

        task.assignment = c_idx
        task.execution_tier = ExecutionTier.CLOUD
        task.device_core = -1
        task.edge_assignment = None
        if not hasattr(task, 'execution_path') or not task.execution_path:
            task.execution_path = [(ExecutionTier.DEVICE, None), (ExecutionTier.CLOUD, None)]
        else:
            task.execution_path.append((ExecutionTier.CLOUD, None))

        task.FT_l = 0
        task.FT_edge = {}
        task.FT_edge_receive = {}
        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[c_idx].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        """
        For each entry task, pick the tier (device or edge or cloud) based on
        initial assignment, then do the scheduling with earliest
        finish time in mind.
        """
        # We'll partition them by current execution_tier
        device_tasks = []
        edge_tasks = []
        cloud_tasks = []

        for t in entry_tasks:
            if t.execution_tier == ExecutionTier.DEVICE:
                device_tasks.append(t)
            elif t.execution_tier == ExecutionTier.EDGE:
                edge_tasks.append(t)
            else:
                cloud_tasks.append(t)

        # 1) device tasks
        for t in device_tasks:
            core, st, ft = self.identify_optimal_local_core(t, 0)
            if core < 0 or ft == float('inf'):
                # fallback
                logger.info(f"No local core for entry task {t.id}, fallback to edge or cloud.")
                if self.M > 0:
                    edge_tasks.append(t)
                else:
                    cloud_tasks.append(t)
            else:
                self.schedule_on_local_core(t, core, st, ft)

        # 2) edge tasks
        edge_fallbacks = []
        for t in edge_tasks:
            best_e, best_c, st, ft = self.identify_optimal_edge_node(t)
            if best_e < 0 or ft == float('inf'):
                edge_fallbacks.append(t)
            else:
                self.schedule_on_edge(t, best_e, best_c, st, ft)
        cloud_tasks.extend(edge_fallbacks)

        # 3) cloud tasks
        cloud_fallbacks = []
        for t in cloud_tasks:
            # We can do a “calculate_cloud_phases_timing” or something simpler
            # Suppose we rely on 'calculate_cloud_phases_timing'
            # or we do inline approximations
            # We'll just assume it’s properly updated by a helper function:
            # e.g. "timing = self.calculate_cloud_phases_timing(t)"
            # or an external approach. Then call 'schedule_on_cloud(...)'
            # For brevity, a placeholder approach:
            t.RT_ws = self.ws_ready
            upload_finish = t.RT_ws + t.cloud_execution_times[0]
            cloud_start = upload_finish
            cloud_finish = cloud_start + t.cloud_execution_times[1]
            download_start = cloud_finish
            download_finish = download_start + t.cloud_execution_times[2]

            if upload_finish == float('inf'):
                cloud_fallbacks.append(t)
                continue

            self.schedule_on_cloud(t, t.RT_ws, upload_finish,
                                   cloud_start, cloud_finish,
                                   download_start, download_finish)

        for t in cloud_fallbacks:
            # If it can't do cloud scheduling, we do a final fallback to device or edge
            core, st, ft = self.identify_optimal_local_core(t, 0)
            if core >= 0 and ft < float('inf'):
                self.schedule_on_local_core(t, core, st, ft)
            else:
                # try edge
                best_e, best_c, st2, ft2 = self.identify_optimal_edge_node(t)
                if best_e >= 0 and ft2 < float('inf'):
                    self.schedule_on_edge(t, best_e, best_c, st2, ft2)
                else:
                    logger.error(f"Entry task {t.id} cannot be scheduled anywhere.")
                    t.is_scheduled = SchedulingState.UNSCHEDULED

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """
        Schedules tasks that have predecessors, after the entry tasks are scheduled.
        For each task, we evaluate local vs. edge vs. cloud finish times and pick the best.
        """
        for t in non_entry_tasks:
            logger.info(f"Scheduling non-entry task {t.id}")

            local_ready = self.calculate_local_ready_time(t)
            local_finish = float('inf')
            best_core = None
            best_start = float('inf')
            if local_ready < float('inf'):
                c, st, ft = self.identify_optimal_local_core(t, local_ready)
                local_finish = ft
                best_core = c
                best_start = st

            edge_finish = float('inf')
            edge_id, core_id = -1, -1
            edge_start = 0
            # gather ready times
            best_e, best_c, st2, ft2 = self.identify_optimal_edge_node(t)
            edge_finish = ft2
            edge_id = best_e
            core_id = best_c
            edge_start = st2

            cloud_finish = float('inf')
            # assume we do a quick or approximate approach:
            upload_ready = self.calculate_cloud_upload_ready_time(t)
            if upload_ready < float('inf'):
                # do ephemeral computations
                us = upload_ready
                uf = us + t.cloud_execution_times[0]
                cs = uf
                cf = cs + t.cloud_execution_times[1]
                rs = cf
                rf = rs + t.cloud_execution_times[2]
                cloud_finish = rf

            logger.info(f"Task {t.id} finish times - local: {local_finish}, edge: {edge_finish}, cloud: {cloud_finish}")

            if local_finish <= edge_finish and local_finish <= cloud_finish and local_finish < float('inf'):
                self.schedule_on_local_core(t, best_core, best_start, local_finish)
            elif edge_finish <= local_finish and edge_finish <= cloud_finish and edge_finish < float('inf'):
                self.schedule_on_edge(t, edge_id, core_id, edge_start, edge_finish)
            elif cloud_finish < float('inf'):
                self.schedule_on_cloud(t, us, uf, cs, cf, rs, rf)
            else:
                logger.error(f"Task {t.id} cannot find a valid scheduling option.")
                t.is_scheduled = SchedulingState.UNSCHEDULED


def primary_assignment(tasks, edge_nodes=None):
    """
    Basic heuristic that assigns each task to the 'best' tier
    (device, edge, or cloud) *before* scheduling, based on comparing
    known device/edge/cloud times.
    """
    global device_core_execution_times, edge_core_execution_times, cloud_execution_times, logger

    if edge_nodes is None:
        edge_nodes = M  # from global if needed

    for task in tasks:
        # min local time
        t_l_min = min(task.core_execution_times) if task.core_execution_times else float('inf')

        # min edge time
        t_edge_min = float('inf')
        best_edge_id = -1
        best_core_id = -1
        for e_id in range(1, edge_nodes + 1):
            for c_id in range(1, 3):  # or however many cores are assumed
                k = (task.id, e_id, c_id)
                if k in edge_core_execution_times:
                    e_time = edge_core_execution_times[k]
                    if e_time < t_edge_min:
                        t_edge_min = e_time
                        best_edge_id = e_id
                        best_core_id = c_id

        # cloud total time
        t_cloud = sum(task.cloud_execution_times)

        # simple heuristic
        if (task.id % 2 == 1) and (t_edge_min < t_l_min * 1.1):
            # data-intensive => prefer edge
            task.execution_tier = ExecutionTier.EDGE
            task.edge_assignment = EdgeAssignment(best_edge_id, best_core_id)
        else:
            if t_l_min <= t_edge_min and t_l_min <= t_cloud:
                task.execution_tier = ExecutionTier.DEVICE
            elif t_edge_min <= t_l_min and t_edge_min <= t_cloud:
                task.execution_tier = ExecutionTier.EDGE
                task.edge_assignment = EdgeAssignment(best_edge_id, best_core_id)
            else:
                task.execution_tier = ExecutionTier.CLOUD


def task_prioritizing(tasks):
    """
    Computes a 'priority_score' for each task,
    typically used in scheduling order.
    """
    global logger
    num_tasks = len(tasks)
    w = [0] * num_tasks

    for i, t in enumerate(tasks):
        if t.execution_tier == ExecutionTier.CLOUD:
            w[i] = sum(t.cloud_execution_times)
        elif t.execution_tier == ExecutionTier.EDGE:
            if t.edge_assignment:
                # approximate cost: device→edge + edge_exec + edge→device
                e_id, c_id = t.edge_assignment.edge_id, t.edge_assignment.core_id
                e_time = t.edge_core_execution_times.get((e_id, c_id), 3.0)
                # approximate upload and download
                w[i] = e_time + 3.0
            else:
                w[i] = float('inf')
        else:  # device
            if t.core_execution_times:
                w[i] = sum(t.core_execution_times) / len(t.core_execution_times)
            else:
                w[i] = float('inf')

    computed_scores = {}

    def calc_priority(task_idx):
        if task_idx < 0 or task_idx >= num_tasks:
            return 0
        task_obj = tasks[task_idx]
        if task_obj.id in computed_scores:
            return computed_scores[task_obj.id]

        if not task_obj.succ_tasks:
            sc = w[task_idx]
            computed_scores[task_obj.id] = sc
            return sc

        max_succ = 0
        for s in task_obj.succ_tasks:
            s_idx = s.id - 1
            sp = calc_priority(s_idx)
            max_succ = max(max_succ, sp)
        sc = w[task_idx] + max_succ
        computed_scores[task_obj.id] = sc
        return sc

    for i in range(num_tasks):
        calc_priority(i)

    for t in tasks:
        t.priority_score = computed_scores.get(t.id, 0)


def validate_three_tier_schedule(tasks: List[Any], sequences: List[List[int]],
                                 num_device_cores: int, num_edge_nodes: int,
                                 num_edge_cores_per_node: int) -> Dict[str, Any]:
    """
    Checks for consistency of assignments, no overlap, correct sequence usage, etc.
    Also calls validate_task_dependencies for more thorough checks.
    """
    validation = {
        "valid": True,
        "dependency_violations": 0,
        "sequence_violations": 0,
        "resource_violations": 0,
        "tier_violations": 0,
        "unscheduled_tasks": 0,
        "issues": []
    }

    # Check if tasks are assigned
    for t in tasks:
        scheduled_ok = False
        if t.execution_tier == ExecutionTier.DEVICE:
            if t.device_core >= 0 and t.device_core < num_device_cores:
                scheduled_ok = True
            else:
                validation["issues"].append(f"Task {t.id} invalid device_core {t.device_core}")
        elif t.execution_tier == ExecutionTier.EDGE:
            if t.edge_assignment and 1 <= t.edge_assignment.edge_id <= num_edge_nodes and 1 <= t.edge_assignment.core_id <= num_edge_cores_per_node:
                scheduled_ok = True
            else:
                validation["issues"].append(f"Task {t.id} invalid edge assignment {t.edge_assignment}")
        elif t.execution_tier == ExecutionTier.CLOUD:
            scheduled_ok = True
        else:
            validation["issues"].append(f"Task {t.id} unknown tier")

        if not scheduled_ok:
            validation["unscheduled_tasks"] += 1

    # Check sequences
    seq_count = {}
    for seq_idx, seq in enumerate(sequences):
        for tid in seq:
            seq_count[tid] = seq_count.get(tid, 0) + 1

    for t in tasks:
        if t.id not in seq_count:
            # Not in any sequence
            validation["sequence_violations"] += 1
            validation["issues"].append(f"Task {t.id} is missing from sequences")
        else:
            if seq_count[t.id] > 1:
                validation["sequence_violations"] += 1
                validation["issues"].append(f"Task {t.id} appears in {seq_count[t.id]} sequences")

    valid_deps, dep_violations = validate_task_dependencies(tasks)
    if not valid_deps:
        validation["dependency_violations"] += 1
        validation["issues"].append("Task dependency constraints violated")
        for v in dep_violations:
            validation["issues"].append(f"{v['type']} - {v['detail']}")

    validation["valid"] = (
            validation["dependency_violations"] == 0 and
            validation["sequence_violations"] == 0 and
            validation["unscheduled_tasks"] == 0
    )
    return validation


def validate_task_dependencies(tasks, epsilon=1e-6):
    """
    Checks that for each task, it does not start before
    any predecessor’s data is ready.
    Returns (bool, list_of_violations).
    """
    violations = []
    for t in tasks:
        if t.is_scheduled == SchedulingState.UNSCHEDULED:
            continue
        start_time = None

        if t.execution_tier == ExecutionTier.DEVICE and t.device_core >= 0:
            exec_time = t.core_execution_times[t.device_core] if t.core_execution_times else 0
            start_time = t.FT_l - exec_time
        elif t.execution_tier == ExecutionTier.CLOUD:
            start_time = t.RT_ws
        elif t.execution_tier == ExecutionTier.EDGE and t.edge_assignment:
            e_id = t.edge_assignment.edge_id - 1
            c_id = t.edge_assignment.core_id - 1
            if e_id in t.FT_edge:
                exec_time = t.get_edge_execution_time(e_id + 1, c_id + 1) or 0
                start_time = t.FT_edge[e_id] - exec_time

        if start_time is None:
            # might not be scheduled or missing data
            continue

        for p in t.pred_tasks:
            pred_ready = None
            if p.execution_tier == ExecutionTier.DEVICE:
                pred_ready = p.FT_l
            elif p.execution_tier == ExecutionTier.CLOUD:
                pred_ready = p.FT_wr
            elif p.execution_tier == ExecutionTier.EDGE:
                if p.edge_assignment:
                    pe_id = p.edge_assignment.edge_id - 1
                    if pe_id in p.FT_edge:
                        pred_ready = p.FT_edge[pe_id]
                    # Also consider time to device if needed
            if pred_ready is not None and start_time < pred_ready - epsilon:
                violations.append({
                    'type': 'Dependency Violation',
                    'task': t.id,
                    'predecessor': p.id,
                    'task_start': start_time,
                    'pred_ready': pred_ready,
                    'detail': f"Task {t.id} starts at {start_time}, but pred {p.id} not done until {pred_ready}"
                })

    is_valid = (len(violations) == 0)
    return is_valid, violations


# --------------------------------------------------
# 4. three_tier_kernel_algorithm
# --------------------------------------------------
def three_tier_kernel_algorithm(tasks, sequences, num_device_cores, num_edge_nodes, num_edge_cores_per_node):
    """
    An example kernel-based scheduling pass that looks at 'tasks' and 'sequences',
    attempts to schedule tasks in a certain order, and updates
    tasks' finishing times, etc.
    """
    scheduler = ThreeTierKernelScheduler(
        tasks, sequences, num_device_cores, num_edge_nodes, num_edge_cores_per_node
    )

    cloud_tasks = [t.id for t in tasks if t.execution_tier == ExecutionTier.CLOUD]
    logger.info(f"Starting kernel algorithm with cloud tasks: {cloud_tasks}")

    queue = scheduler.initialize_queue()

    device_scheduled = []
    edge_scheduled = []
    cloud_scheduled = []

    while queue:
        cur_task = queue.popleft()
        cur_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        if cur_task.execution_tier == ExecutionTier.DEVICE:
            scheduler.schedule_device_task(cur_task)
            device_scheduled.append(cur_task.id)
        elif cur_task.execution_tier == ExecutionTier.EDGE:
            scheduler.schedule_edge_task(cur_task)
            edge_scheduled.append(cur_task.id)
        else:
            # cloud
            if cur_task.FT_ws <= 0 or cur_task.FT_c <= 0 or cur_task.FT_wr <= 0:
                # approximate or adjust
                update_cloud_timing(cur_task, tasks)
            scheduler.schedule_cloud_task(cur_task)
            cloud_scheduled.append(cur_task.id)

        for t in tasks:
            scheduler.update_task_state(t)
            if (scheduler.dependency_ready[t.id - 1] == 0
                    and scheduler.sequence_ready[t.id - 1] == 0
                    and t.is_scheduled != SchedulingState.KERNEL_SCHEDULED
                    and t not in queue):
                queue.append(t)

    logger.info(
        f"Kernel scheduling complete. device={len(device_scheduled)}, edge={len(edge_scheduled)}, cloud={len(cloud_scheduled)}")

    # Reset tasks' is_scheduled to SCHEDULED
    for t in tasks:
        t.is_scheduled = SchedulingState.SCHEDULED

    return tasks


# --------------------------------------------------
# 5. Additional time/energy calculations
# --------------------------------------------------
def calculate_energy_consumption(task, core_powers, rf_power, upload_rates, download_rates):
    """
    Approximates the energy usage for 'task' based on where it ran and
    how much data it transferred.
    """
    global task_data_sizes
    if task.execution_tier == ExecutionTier.DEVICE:
        if task.device_core < 0 or task.device_core >= len(task.core_execution_times):
            return 0
        return core_powers.get(task.device_core, 1.0) * task.core_execution_times[task.device_core]

    # fallback approach if not enough detail:
    total_energy = 0
    if not hasattr(task, 'execution_path') or not task.execution_path:
        # no path => single-tier approach
        if task.execution_tier == ExecutionTier.CLOUD:
            return rf_power.get('device_to_cloud', 0.5) * task.cloud_execution_times[0]
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            e_id = task.edge_assignment.edge_id
            c_id = task.edge_assignment.core_id
            return core_powers.get(c_id, 1.0) * 3.0  # fallback
        return 0

    # if we track migrations in execution_path, we can accumulate
    for i, (tier, location) in enumerate(task.execution_path[:-1]):
        next_tier, next_loc = task.execution_path[i + 1]
        # device -> edge
        if tier == ExecutionTier.DEVICE and next_tier == ExecutionTier.EDGE:
            e_id, _ = next_loc
            key = f'device_to_edge{e_id}'
            data_sz = task_data_sizes.get(task.id, {}).get(key, 0)
            r = upload_rates.get(key, 1.0)
            t = data_sz / r if r > 0 else 0
            total_energy += rf_power.get(key, 0.4) * t
        elif tier == ExecutionTier.DEVICE and next_tier == ExecutionTier.CLOUD:
            total_energy += rf_power.get('device_to_cloud', 0.5) * task.cloud_execution_times[0]
        # Similarly for edge->edge, edge->cloud, cloud->device, etc.

    return total_energy


def total_energy(tasks, core_powers, rf_power, upload_rates, download_rates):
    return sum(calculate_energy_consumption(t, core_powers, rf_power, upload_rates, download_rates)
               for t in tasks)


def total_time(tasks):
    """
    Overall completion time of the DAG in a 3-tier scenario.
    Typically the max finish time among exit tasks.
    """
    exit_tasks = [t for t in tasks if not t.succ_tasks]
    if not exit_tasks:
        exit_tasks = tasks

    max_ct = 0
    for e in exit_tasks:
        ft = e.get_overall_finish_time()
        max_ct = max(max_ct, ft)
    return max_ct


def initialize_migration_choices_three_tier(tasks, num_device_cores, num_edge_nodes, num_edge_cores_per_node):
    """
    Creates a matrix [n_tasks x total_units] of booleans indicating
    if a (task, unit) migration is valid.
    """
    total_units = num_device_cores + num_edge_nodes * num_edge_cores_per_node + 1
    import numpy as np
    migration_choices = np.zeros((len(tasks), total_units), dtype=bool)

    for i, task in enumerate(tasks):
        tier = task.execution_tier
        if tier == ExecutionTier.DEVICE:
            # can migrate to other device cores, edge, or cloud
            for c in range(num_device_cores):
                if c != task.device_core:
                    migration_choices[i, c] = True
            offset = num_device_cores
            for e_id in range(num_edge_nodes):
                for c_id in range(num_edge_cores_per_node):
                    idx = offset + e_id * num_edge_cores_per_node + c_id
                    migration_choices[i, idx] = True
            # cloud
            migration_choices[i, total_units - 1] = True
        elif tier == ExecutionTier.EDGE:
            # can go to device, other edges, or cloud
            for c in range(num_device_cores):
                migration_choices[i, c] = True
            offset = num_device_cores
            for e_id in range(num_edge_nodes):
                for c_id in range(num_edge_cores_per_node):
                    idx = offset + e_id * num_edge_cores_per_node + c_id
                    # skip if same edge node+core
                    if (task.edge_assignment
                            and e_id == (task.edge_assignment.edge_id - 1)
                            and c_id == (task.edge_assignment.core_id - 1)):
                        continue
                    migration_choices[i, idx] = True
            # cloud
            migration_choices[i, total_units - 1] = True
        elif tier == ExecutionTier.CLOUD:
            # can move to device or edge
            for c in range(num_device_cores):
                migration_choices[i, c] = True
            offset = num_device_cores
            for e_id in range(num_edge_nodes):
                for c_id in range(num_edge_cores_per_node):
                    idx = offset + e_id * num_edge_cores_per_node + c_id
                    migration_choices[i, idx] = True
        else:
            logger.warning(f"Task {task.id} unknown tier => no migration")

    return migration_choices


# --------------------------------------------------
# 1. update_cloud_timing
# --------------------------------------------------
def update_cloud_timing(task, tasks=None):
    """
    Adjusts the cloud-based finish times (FT_ws, FT_c, FT_wr)
    based on the largest predecessor finish time plus
    standard phases for cloud upload/compute/download.
    """
    if not task.pred_tasks:
        upload_ready = 0.0
    else:
        pred_finish_times = []
        for p in task.pred_tasks:
            if p.execution_tier == ExecutionTier.DEVICE:
                pred_finish_times.append(p.FT_l)
            elif p.execution_tier == ExecutionTier.CLOUD:
                pred_finish_times.append(p.FT_ws)
            elif p.execution_tier == ExecutionTier.EDGE:
                if p.edge_assignment:
                    e_id = p.edge_assignment.edge_id - 1
                    if hasattr(p, 'FT_edge_receive') and e_id in p.FT_edge_receive:
                        pred_finish_times.append(p.FT_edge_receive[e_id])
        upload_ready = max(pred_finish_times) if pred_finish_times else 0.0

    task.RT_ws = upload_ready
    upload_time = task.cloud_execution_times[0]
    task.FT_ws = task.RT_ws + upload_time

    task.RT_c = task.FT_ws
    compute_time = task.cloud_execution_times[1]
    task.FT_c = task.RT_c + compute_time

    task.RT_wr = task.FT_c
    download_time = task.cloud_execution_times[2]
    task.FT_wr = task.RT_wr + download_time


# --------------------------------------------------
# 2. construct_sequence_three_tier
# --------------------------------------------------
def construct_sequence_three_tier(tasks, task_id, target_unit_index, sequences,
                                  num_device_cores, num_edge_nodes, num_edge_cores_per_node):
    """
    Migrate the task with 'task_id' to the resource identified by 'target_unit_index'.
    This means removing it from its old sequence and appending it to the new sequence.
    Then update the task’s 'execution_tier' and 'device_core' or 'edge_assignment'
    accordingly.
    """
    t = next((x for x in tasks if x.id == task_id), None)
    if not t:
        logger.error(f"Task {task_id} not found for migration")
        return sequences

    source_index = next((i for i, seq in enumerate(sequences) if task_id in seq), -1)
    if source_index < 0:
        logger.error(f"Task {task_id} not in any sequence")
        return sequences

    sequences[source_index].remove(task_id)

    # Convert the target_unit_index into (tier, location)
    target_unit = get_execution_unit_from_index(
        target_unit_index, num_device_cores, num_edge_nodes, num_edge_cores_per_node
    )

    sequences[target_unit_index].append(task_id)

    t.execution_tier = target_unit.tier
    if target_unit.tier == ExecutionTier.DEVICE:
        logger.info(f"Migrate task {task_id} to device core {target_unit.location[0]}")
        t.device_core = target_unit.location[0]
        t.edge_assignment = None
    elif target_unit.tier == ExecutionTier.EDGE:
        logger.info(
            f"Migrate task {task_id} to edge node {target_unit.location[0] + 1}, core {target_unit.location[1] + 1}")
        e_n, c_n = target_unit.location
        t.device_core = -1
        t.edge_assignment = EdgeAssignment(e_n + 1, c_n + 1)
    else:
        logger.info(f"Migrate task {task_id} to cloud")
        t.device_core = -1
        t.edge_assignment = None
        t.FT_l = 0
        update_cloud_timing(t)  # optionally adjust times

    return sequences


# --------------------------------------------------
# 3. total_energy_consumption_three_tier
# --------------------------------------------------
def total_energy_consumption_three_tier(tasks, device_core_powers=None, edge_node_powers=None,
                                        rf_power=None, upload_rates=None, download_rates=None):
    """
    Aggregates energy usage across all tasks in the system
    using the 'calculate_energy_consumption_three_tier' helper.
    """
    return sum(
        calculate_energy_consumption_three_tier(
            t, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
        ) for t in tasks
    )


def calculate_energy_consumption_three_tier(task, device_core_powers=None, edge_node_powers=None,
                                            rf_power=None, upload_rates=None, download_rates=None):
    """
    Another approach to compute energy, referencing the
    'execution_path' for each segment’s data transfer, etc.
    """
    if not device_core_powers:
        device_core_powers = {}
    if not edge_node_powers:
        edge_node_powers = {}
    if not rf_power:
        rf_power = {}
    if not upload_rates:
        upload_rates = {}
    if not download_rates:
        download_rates = {}

    # For brevity, a simplified approach:
    if task.execution_tier == ExecutionTier.DEVICE:
        c_id = task.device_core
        core_pwr = device_core_powers.get(c_id, 30.0)
        ex_time = 0
        if (task.core_execution_times and c_id >= 0 and c_id < len(task.core_execution_times)):
            ex_time = task.core_execution_times[c_id]
        return core_pwr * ex_time
    elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
        e_id = task.edge_assignment.edge_id
        c_id = task.edge_assignment.core_id
        edge_pwr = edge_node_powers.get((e_id, c_id), 0.5)
        ex_time = task.get_edge_execution_time(e_id, c_id) or 3.0
        return edge_pwr * ex_time
    elif task.execution_tier == ExecutionTier.CLOUD:
        # naive approach:
        rf_p = rf_power.get('device_to_cloud', 0.05)
        upload_t = task.cloud_execution_times[0]
        return rf_p * upload_t
    return 0.0


# --------------------------------------------------
# 4. optimize_task_scheduling_three_tier
# --------------------------------------------------
def optimize_task_scheduling_three_tier(tasks, sequences, initial_time,
                                        time_constraint_factor=1.5, max_iterations=50,
                                        max_evaluations_per_iteration=20,
                                        early_stopping_threshold=0.01,
                                        num_device_cores=3, num_edge_nodes=2,
                                        num_edge_cores_per_node=2, device_core_powers=None,
                                        edge_node_powers=None, rf_power=None,
                                        upload_rates=None, download_rates=None):
    """
    Main iterative optimization routine that attempts
    to reduce energy usage (subject to some time constraint).
    Uses a 'migration_cache' to avoid redundant calculations.
    """

    start_t = time_module.time()
    current_tasks = deepcopy(tasks)
    current_sequences = [s.copy() for s in sequences]
    sequence_mgr = SequenceManager(num_device_cores, num_edge_nodes, num_edge_cores_per_node)
    sequence_mgr.set_all_sequences(current_sequences)
    migration_cache = MigrationCache(capacity=20000)

    initial_energy = total_energy_consumption_three_tier(
        current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )
    metrics = OptimizationMetrics()
    metrics.start(initial_time, initial_energy)
    max_time = initial_time * time_constraint_factor

    valid, _ = validate_task_dependencies(current_tasks)
    if not valid:
        logger.error("Initial schedule has dependency violations; continuing anyway...")

    consecutive_no_improvement = 0

    for iteration in range(max_iterations):
        if consecutive_no_improvement >= 3:
            logger.info(f"Early stopping after {iteration} iterations (no improvement).")
            break

        prev_energy = metrics.current_energy
        prev_tasks = deepcopy(current_tasks)
        prev_sequences = [seq.copy() for seq in current_sequences]

        # Build migration_choices matrix
        migration_choices = initialize_migration_choices_three_tier(
            current_tasks, num_device_cores, num_edge_nodes, num_edge_cores_per_node
        )

        # Evaluate possible migrations. We'll store (task_idx, target_idx, new_time, new_energy)
        migration_results = []
        for i, t in enumerate(current_tasks):
            # example: prioritize testing cloud migration
            cloud_idx = num_device_cores + (num_edge_nodes * num_edge_cores_per_node)
            if migration_choices[i][cloud_idx]:
                new_time, new_energy = evaluate_migration_three_tier(
                    current_tasks, current_sequences, i, cloud_idx,
                    migration_cache, three_tier_kernel_algorithm, construct_sequence_three_tier,
                    total_energy_consumption_three_tier,
                    num_device_cores, num_edge_nodes, num_edge_cores_per_node,
                    device_core_powers, edge_node_powers, rf_power,
                    upload_rates, download_rates
                )
                if new_time < float('inf') and new_energy < float('inf'):
                    migration_results.append((i, cloud_idx, new_time, new_energy))

            if len(migration_results) >= max_evaluations_per_iteration:
                break

        best = identify_optimal_migration_three_tier(migration_results,
                                                     metrics.current_time,
                                                     metrics.current_energy,
                                                     max_time)
        metrics.update(metrics.current_time, metrics.current_energy, len(migration_results))

        if not best:
            logger.info(f"Iteration {iteration + 1}: no valid migration found.")
            consecutive_no_improvement += 1
            continue

        try:
            # Apply it
            task_id = best.task_id
            target_idx = 0
            if best.target_tier == ExecutionTier.DEVICE:
                target_idx = best.target_location[0]
            elif best.target_tier == ExecutionTier.EDGE:
                e, c = best.target_location
                target_idx = num_device_cores + e * num_edge_cores_per_node + c
            else:
                target_idx = num_device_cores + num_edge_nodes * num_edge_cores_per_node

            current_sequences = construct_sequence_three_tier(
                current_tasks, task_id, target_idx, current_sequences,
                num_device_cores, num_edge_nodes, num_edge_cores_per_node
            )
            three_tier_kernel_algorithm(current_tasks, current_sequences,
                                        num_device_cores, num_edge_nodes, num_edge_cores_per_node)

            valid, vios = validate_task_dependencies(current_tasks)
            if not valid:
                logger.warning(f"Migration {task_id} => {best.target_tier} caused constraint violation. Rolling back.")
                current_tasks = prev_tasks
                current_sequences = prev_sequences
                consecutive_no_improvement += 1
                metrics.update(metrics.current_time, prev_energy)
                continue

            new_t = calculate_total_time(current_tasks)
            new_e = total_energy_consumption_three_tier(
                current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
            )
            metrics.update(new_t, new_e, 0)
            if new_e < prev_energy:
                consecutive_no_improvement = 0
                improv = (prev_energy - new_e) / prev_energy
                logger.info(
                    f"Iteration {iteration + 1}: energy improved from {prev_energy:.2f} to {new_e:.2f} => {improv * 100:.2f}%")
                if improv < early_stopping_threshold:
                    consecutive_no_improvement += 1
            else:
                consecutive_no_improvement += 1
        except Exception as e:
            logger.error(f"Error in migration: {e}")
            current_tasks = prev_tasks
            current_sequences = prev_sequences
            consecutive_no_improvement += 1
            continue

    final_t = calculate_total_time(current_tasks)
    final_e = total_energy_consumption_three_tier(
        current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )
    metrics.update(final_t, final_e)
    valid, vios = validate_task_dependencies(current_tasks)
    if not valid:
        logger.warning("Final schedule has dependency violations!")
        for v in vios[:5]:
            logger.warning(f"{v['type']}: {v['detail']}")

    logger.info(
        f"Optimization done in {time_module.time() - start_t:.2f}s. final energy={final_e:.2f}, final time={final_t:.2f}")
    return current_tasks, current_sequences, metrics.get_summary()


# --------------------------------------------------
# 5. Additional small utilities
# --------------------------------------------------
def calculate_total_time(tasks_list):
    # Find exit tasks (tasks with no successors)
    exit_tasks = [task for task in tasks_list if not task.succ_tasks]
    if not exit_tasks:
        exit_tasks = tasks_list  # If no explicit exit tasks, use all tasks

    # Find maximum finish time across all exit tasks
    max_finish = 0
    for task in exit_tasks:
        # Calculate based on execution tier
        if task.execution_tier == ExecutionTier.DEVICE:
            finish = task.FT_l
        elif task.execution_tier == ExecutionTier.CLOUD:
            finish = task.FT_wr
        elif task.execution_tier == ExecutionTier.EDGE:
            if task.edge_assignment and hasattr(task, 'FT_edge_receive'):
                edge_id = task.edge_assignment.edge_id - 1
                finish = task.FT_edge_receive.get(edge_id, 0)
            else:
                finish = 0
        else:
            finish = 0

        max_finish = max(max_finish, finish)

    return max_finish


def identify_optimal_migration_three_tier(
        migration_trials_results: List[Tuple[int, int, float, float]],
        tasks: List[Any],
        current_time: float,
        current_energy: float,
        max_time: float,
        priority_energy_reduction: bool = True
) -> Optional[TaskMigrationState]:
    """
    From a list of potential migrations (each a tuple (task_idx, target_unit_index, new_time, new_energy)),
    pick the best migration that yields a net energy reduction while
    respecting the overall time constraint (i.e., ensuring we don't exceed 'max_time').

    Args:
        migration_trials_results:
            A list of (task_idx, target_unit_index, new_time, new_energy), where:
              - task_idx: index of the task in 'tasks'
              - target_unit_index: the linear index of the new resource (device core, edge core, or cloud)
              - new_time: the total completion time after this migration (already pre-computed)
              - new_energy: the total energy consumption after this migration (already pre-computed)
        tasks:
            The list of all Task objects (so we can map task_idx back to a Task).
        current_time:
            The current total completion time (before migration).
        current_energy:
            The current total energy consumption (before migration).
        max_time:
            The maximum allowable completion time (e.g., initial_time * time_constraint_factor).
        priority_energy_reduction:
            If True, we prioritize maximizing energy reduction over other factors. (Currently,
            this function implements a simple ratio-based approach, so you can tweak for your needs.)

    Returns:
        A TaskMigrationState describing the best migration, or None if no valid migration is found.
    """

    if not migration_trials_results:
        return None

    # Determine how much additional time we can afford relative to current_time
    allowed_time_increase = max_time - current_time

    # Filter candidate migrations that give us a positive energy reduction
    # and do not exceed the allowable time.
    candidates = []
    for (tidx, tgt_idx, mig_time, mig_en) in migration_trials_results:
        energy_reduction = current_energy - mig_en
        time_increase = mig_time - current_time

        # Only keep if it reduces energy AND doesn't exceed max_time by too much.
        if energy_reduction > 0 and time_increase <= allowed_time_increase:
            candidates.append({
                'task_idx': tidx,
                'target_idx': tgt_idx,
                'mig_time': mig_time,
                'mig_energy': mig_en,
                'energy_reduction': energy_reduction,
                'time_increase': time_increase
            })

    if not candidates:
        return None

    # Compute a simple "score" for each candidate – e.g., energy_reduction / time_increase
    # so that bigger is better (more reduction, less time penalty).
    # For extremely small time_increase, avoid division by zero.
    for c in candidates:
        c['score'] = c['energy_reduction'] / max(c['time_increase'], 1e-6)

    # Sort by descending score and pick the top
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]

    # Identify the actual Task object and compute its source/target units
    source_task = tasks[best['task_idx']]
    source_unit = get_current_execution_unit(source_task)

    target_unit = get_execution_unit_from_index(
        best['target_idx'],
        globals().get('num_device_cores', 3),  # If your code sets these globally
        globals().get('num_edge_nodes', 2),  # otherwise pass them in explicitly
        globals().get('num_edge_cores_per_node', 2)
    )

    # Construct and return the TaskMigrationState
    return TaskMigrationState(
        time=best['mig_time'],  # The new total completion time if we do this migration
        energy=best['mig_energy'],  # The new total energy if we do this migration
        efficiency=best['score'],  # The ratio-based “score” for this migration
        task_id=best['task_idx'] + 1,  # Task ID is index+1 if tasks are 0-based
        source_tier=source_unit.tier,
        target_tier=target_unit.tier,
        source_location=source_unit.location,
        target_location=target_unit.location,
        time_increase=best['time_increase'],
        energy_reduction=best['energy_reduction']
    )


def evaluate_migration_three_tier(tasks, sequences, task_idx, target_unit_index,
                                  migration_cache, kernel_algorithm_func,
                                  construct_sequence_func, energy_calc_func,
                                num_device_cores, num_edge_nodes,
                                  num_edge_cores_per_node, device_core_powers, edge_node_powers,
                                  rf_power, upload_rates, download_rates):
    """
    Temporarily apply a migration (task_idx -> target_unit_index),
    run kernel scheduling, measure total time & energy. Then revert.
    Return (time, energy) or (inf, inf) if it fails.
    """
    from copy import deepcopy
    t_id = task_idx + 1
    cache_key = generate_migration_cache_key(tasks, t_id,
                                             get_current_execution_unit(tasks[task_idx]),
                                             get_execution_unit_from_index(target_unit_index, num_device_cores,
                                                                           num_edge_nodes, num_edge_cores_per_node)
                                             )
    cached = migration_cache.get(cache_key)
    if cached:
        return cached

    tasks_copy = deepcopy(tasks)
    seq_copy = [s.copy() for s in sequences]
    try:
        seq_copy = construct_sequence_func(tasks_copy, t_id, target_unit_index, seq_copy,
                                           num_device_cores, num_edge_nodes, num_edge_cores_per_node)
        kernel_algorithm_func(tasks_copy, seq_copy, num_device_cores, num_edge_nodes, num_edge_cores_per_node)
        valid, _ = validate_task_dependencies(tasks_copy)
        if not valid:
            migration_cache.put(cache_key, (float('inf'), float('inf')))
            return float('inf'), float('inf')

        new_t = calculate_total_time(tasks_copy)
        new_e = energy_calc_func(tasks_copy, device_core_powers, edge_node_powers, rf_power,
                                 upload_rates, download_rates)
        migration_cache.put(cache_key, (new_t, new_e))
        return new_t, new_e
    except:
        return float('inf'), float('inf')


def generate_migration_cache_key(tasks, task_id, source_unit, target_unit):
    """
    Unique key that identifies the 'before' state plus the specific migration
    (task_id, source_unit -> target_unit).
    """
    # encode current assignment of all tasks
    assignments = tuple(encode_task_assignment(t) for t in tasks)
    return task_id, encode_execution_unit(source_unit), encode_execution_unit(target_unit), assignments


def get_current_execution_unit(task):
    if task.execution_tier == ExecutionTier.DEVICE:
        return ExecutionUnit(task.execution_tier, (task.device_core,))
    elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
        return ExecutionUnit(task.execution_tier,
                             (task.edge_assignment.edge_id - 1,
                              task.edge_assignment.core_id - 1))
    else:
        return ExecutionUnit(ExecutionTier.CLOUD)


def get_execution_unit_from_index(index, num_device_cores, num_edge_nodes, num_edge_cores_per_node):
    if index < num_device_cores:
        return ExecutionUnit(ExecutionTier.DEVICE, (index,))
    edge_offset = num_device_cores
    total_edge = num_edge_nodes * num_edge_cores_per_node
    if index < edge_offset + total_edge:
        e_idx = index - edge_offset
        n_id = e_idx // num_edge_cores_per_node
        c_id = e_idx % num_edge_cores_per_node
        return ExecutionUnit(ExecutionTier.EDGE, (n_id, c_id))
    return ExecutionUnit(ExecutionTier.CLOUD)


def encode_execution_unit(unit):
    tier_val = unit.tier.value
    if unit.tier == ExecutionTier.DEVICE:
        return tier_val, unit.location[0]
    elif unit.tier == ExecutionTier.EDGE:
        return tier_val, unit.location[0], unit.location[1]
    return (tier_val,)


def encode_task_assignment(t):
    if t.execution_tier == ExecutionTier.DEVICE:
        return 0, t.device_core
    elif t.execution_tier == ExecutionTier.EDGE and t.edge_assignment:
        return 1, t.edge_assignment.edge_id, t.edge_assignment.core_id
    else:
        return (2,)  # cloud


import numpy as np
import random
from copy import deepcopy
from typing import Dict, Tuple, List, Any
import time as time_module


def generate_realistic_network_parameters(
        num_tasks: int = 10,
        num_edge_nodes: int = 2,
        seed: int = None
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Generate realistic network parameters using statistical distributions that
    better reflect real-world conditions.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # ---- NETWORK SPEED PARAMETERS ----
    # Base parameters for network conditions (Mbps)
    # Using log-normal distributions for network speeds (common in real networks)
    base_internet_upload = max(0.5, np.random.lognormal(mean=0.5, sigma=0.4))
    base_internet_download = max(1.0, np.random.lognormal(mean=1.5, sigma=0.5))
    base_wifi_speed = max(3.0, np.random.lognormal(mean=2.0, sigma=0.6))
    base_edge_backhaul = max(5.0, np.random.lognormal(mean=2.3, sigma=0.4))

    # ---- POWER CONSUMPTION PARAMETERS ----
    # Power consumption for different transmission modes (W)
    # Power consumption often follows gamma distribution (always positive, right-skewed)
    base_cellular_power = max(0.3, np.random.gamma(shape=2.0, scale=0.2))
    base_wifi_power = max(0.1, np.random.gamma(shape=1.5, scale=0.1))
    base_edge_power = max(0.2, np.random.gamma(shape=2.0, scale=0.1))

    # ---- GENERATE RATE DICTIONARIES WITH CONSTRAINTS ----
    upload_rates = {}
    download_rates = {}
    rf_power = {}

    # Device to cloud rates (cellular)
    upload_rates['device_to_cloud'] = base_internet_upload
    download_rates['cloud_to_device'] = base_internet_download
    rf_power['device_to_cloud'] = base_cellular_power
    rf_power['cloud_to_device'] = base_cellular_power * 0.7  # Receiving uses less power

    # Device to edge rates (WiFi/local)
    for e in range(1, num_edge_nodes + 1):
        # Add variability between edge nodes (some might be closer/farther)
        edge_factor = np.random.uniform(0.8, 1.2)

        # Upload rates - edge connections are faster than cloud
        edge_upload = base_wifi_speed * edge_factor
        upload_rates[f'device_to_edge{e}'] = edge_upload

        # Download rates - typically faster than upload for consumer connections
        edge_download = edge_upload * np.random.uniform(1.2, 1.8)
        download_rates[f'edge{e}_to_device'] = edge_download

        # RF power - WiFi uses less power than cellular
        edge_power_factor = np.random.uniform(0.9, 1.1)
        rf_power[f'device_to_edge{e}'] = base_wifi_power * edge_power_factor
        rf_power[f'edge{e}_to_device'] = base_wifi_power * edge_power_factor * 0.7

        # Edge to cloud rates (better backhaul)
        upload_rates[f'edge{e}_to_cloud'] = base_edge_backhaul * edge_factor
        download_rates[f'cloud_to_edge{e}'] = base_edge_backhaul * edge_factor * np.random.uniform(1.1, 1.5)
        rf_power[f'edge{e}_to_cloud'] = base_edge_power * edge_power_factor
        rf_power[f'cloud_to_edge{e}'] = base_edge_power * edge_power_factor * 0.7

        # Edge to edge rates
        for e2 in range(1, num_edge_nodes + 1):
            if e != e2:
                # Edge-to-edge transfers typically fast but variable
                edge_to_edge_factor = np.random.uniform(0.7, 1.3)
                upload_rates[f'edge{e}_to_edge{e2}'] = base_edge_backhaul * edge_to_edge_factor
                rf_power[f'edge{e}_to_edge{e2}'] = base_edge_power * edge_power_factor * edge_to_edge_factor

    # ---- GENERATE TASK DATA SIZES WITH CONSTRAINTS ----
    task_data_sizes = {}

    for task_id in range(1, num_tasks + 1):
        task_data_sizes[task_id] = {}

        # Base data size varies by task (using log-normal as data sizes are often right-skewed)
        # Data-intensive vs. compute-intensive pattern based on task ID
        if task_id % 2 == 1:  # Odd IDs: data-intensive (larger input, smaller output)
            base_input_size = max(1.0, np.random.lognormal(mean=1.5, sigma=0.5))
            base_output_size = max(0.5, np.random.lognormal(mean=0.0, sigma=0.5))
        else:  # Even IDs: compute-intensive (smaller input, varying output)
            base_input_size = max(0.5, np.random.lognormal(mean=0.7, sigma=0.4))
            base_output_size = max(0.8, np.random.lognormal(mean=1.0, sigma=0.6))

        # Device to cloud (typically needs more data than local transfers)
        cloud_factor = np.random.uniform(1.1, 1.5)
        task_data_sizes[task_id]['device_to_cloud'] = base_input_size * cloud_factor
        task_data_sizes[task_id]['cloud_to_device'] = base_output_size * cloud_factor

        # Device to edge transfers
        for e in range(1, num_edge_nodes + 1):
            edge_data_factor = np.random.uniform(0.8, 1.2)
            task_data_sizes[task_id][f'device_to_edge{e}'] = base_input_size * edge_data_factor
            task_data_sizes[task_id][f'edge{e}_to_device'] = base_output_size * edge_data_factor

            # Edge to cloud transfers
            task_data_sizes[task_id][f'edge{e}_to_cloud'] = base_input_size * edge_data_factor * 0.9
            task_data_sizes[task_id][f'cloud_to_edge{e}'] = base_output_size * edge_data_factor * 0.9

            # Edge to edge transfers
            for e2 in range(1, num_edge_nodes + 1):
                if e != e2:
                    task_data_sizes[task_id][f'edge{e}_to_edge{e2}'] = base_input_size * edge_data_factor * 0.95

    return task_data_sizes, rf_power, upload_rates, download_rates


def create_constrained_parameter_scenarios(
        num_scenarios: int = 5,
        num_tasks: int = 10,
        num_edge_nodes: int = 2,
        base_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Create multiple network parameter scenarios with realistic constraints.
    """
    scenarios = []

    for i in range(num_scenarios):
        seed = base_seed + i

        # Generate base parameters
        task_data_sizes, rf_power, upload_rates, download_rates = generate_realistic_network_parameters(
            num_tasks, num_edge_nodes, seed
        )

        # Create scenario with specific characteristics
        scenario_type = i % 5  # Create 5 types of scenarios

        if scenario_type == 0:
            # Baseline/average scenario
            name = f"baseline_{i}"
            # No modifications needed

        elif scenario_type == 1:
            # Good network, high energy scenario
            name = f"good_network_high_energy_{i}"
            # Improve all network rates by 50%
            for key in upload_rates:
                upload_rates[key] *= 1.5
            for key in download_rates:
                download_rates[key] *= 1.5
            # But increase power consumption by 30%
            for key in rf_power:
                rf_power[key] *= 1.3

        elif scenario_type == 2:
            # Poor network, low energy scenario
            name = f"poor_network_low_energy_{i}"
            # Reduce all network rates by 40%
            for key in upload_rates:
                upload_rates[key] *= 0.6
            for key in download_rates:
                download_rates[key] *= 0.6
            # But decrease power consumption by 20%
            for key in rf_power:
                rf_power[key] *= 0.8

        elif scenario_type == 3:
            # Edge-favorable scenario
            name = f"edge_favorable_{i}"
            # Make edge connections much better than cloud
            for e in range(1, num_edge_nodes + 1):
                upload_rates[f'device_to_edge{e}'] *= 2.0
                download_rates[f'edge{e}_to_device'] *= 2.0
                rf_power[f'device_to_edge{e}'] *= 0.7
                rf_power[f'edge{e}_to_device'] *= 0.7

                # Reduce data sizes for edge transfers
                for task_id in task_data_sizes:
                    task_data_sizes[task_id][f'device_to_edge{e}'] *= 0.7
                    task_data_sizes[task_id][f'edge{e}_to_device'] *= 0.7

        elif scenario_type == 4:
            # Cloud-favorable scenario
            name = f"cloud_favorable_{i}"
            # Make cloud connections better, edge worse
            upload_rates['device_to_cloud'] *= 1.8
            download_rates['cloud_to_device'] *= 1.8
            rf_power['device_to_cloud'] *= 0.7

            # Make edge connections worse
            for e in range(1, num_edge_nodes + 1):
                upload_rates[f'device_to_edge{e}'] *= 0.7
                download_rates[f'edge{e}_to_device'] *= 0.7

                # Increase data sizes for edge transfers
                for task_id in task_data_sizes:
                    task_data_sizes[task_id][f'device_to_edge{e}'] *= 1.3
                    task_data_sizes[task_id][f'edge{e}_to_device'] *= 1.3

        # Store scenario
        scenarios.append({
            'name': name,
            'task_data_sizes': task_data_sizes,
            'rf_power': rf_power,
            'upload_rates': upload_rates,
            'download_rates': download_rates,
            'seed': seed
        })

    return scenarios


def run_scheduler_with_params(
        tasks,
        task_data_sizes,
        rf_power,
        upload_rates,
        download_rates,
        device_core_powers,
        edge_node_powers
):
    """
    Run the scheduler with a specific set of parameters.

    Returns a dictionary with performance metrics.
    """
    # Create deep copy of tasks to avoid modifications between runs
    tasks_copy = deepcopy(tasks)

    # Use primary_assignment to decide the initial execution tier for each task
    primary_assignment(tasks_copy, edge_nodes=2)

    # Compute task priority scores for scheduling order
    task_prioritizing(tasks_copy)

    # Create a three-tier scheduler instance
    scheduler = ThreeTierTaskScheduler(tasks_copy, num_cores=3, num_edge_nodes=2, edge_cores_per_node=2)

    # Classify and schedule tasks
    entry_tasks = [task for task in tasks_copy if not task.pred_tasks]
    non_entry_tasks = [task for task in tasks_copy if task.pred_tasks]

    scheduler.schedule_entry_tasks(entry_tasks)
    scheduler.schedule_non_entry_tasks(non_entry_tasks)

    # Calculate initial metrics
    T_initial = total_time(tasks_copy)
    E_initial = total_energy_consumption_three_tier(
        tasks_copy, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )

    # Optimize the schedule
    optimized_tasks, optimized_sequences, opt_metrics = optimize_task_scheduling_three_tier(
        tasks_copy, scheduler.sequences, T_initial,
        num_device_cores=3, num_edge_nodes=2, num_edge_cores_per_node=2,
        device_core_powers=device_core_powers, edge_node_powers=edge_node_powers,
        rf_power=rf_power, upload_rates=upload_rates, download_rates=download_rates
    )

    # Calculate final metrics
    T_final = total_time(optimized_tasks)
    E_final = total_energy_consumption_three_tier(
        optimized_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )

    # Count tasks per tier
    cloud_tasks = sum(1 for t in optimized_tasks if t.execution_tier == ExecutionTier.CLOUD)
    edge_tasks = sum(1 for t in optimized_tasks if t.execution_tier == ExecutionTier.EDGE)
    device_tasks = sum(1 for t in optimized_tasks if t.execution_tier == ExecutionTier.DEVICE)

    # Validate the final schedule
    validation_result = validate_three_tier_schedule(
        optimized_tasks, optimized_sequences,
        num_device_cores=3, num_edge_nodes=2, num_edge_cores_per_node=2
    )

    # Return metrics
    return {
        'initial_time': T_initial,
        'initial_energy': E_initial,
        'final_time': T_final,
        'final_energy': E_final,
        'time_change_pct': ((T_final - T_initial) / T_initial) * 100 if T_initial > 0 else 0,
        'energy_reduction_pct': ((E_initial - E_final) / E_initial) * 100 if E_initial > 0 else 0,
        'cloud_tasks': cloud_tasks,
        'edge_tasks': edge_tasks,
        'device_tasks': device_tasks,
        'is_valid': validation_result["valid"],
        'validation_issues': validation_result["issues"],
        'optimized_tasks': optimized_tasks,
        'optimized_sequences': optimized_sequences
    }


if __name__ == '__main__':
    # Create a sample task graph similar to the original framework
    task10 = Task(10)
    task9 = Task(9, succ_task=[task10])
    task8 = Task(8, succ_task=[task10])
    task7 = Task(7, succ_task=[task10])
    task6 = Task(6, succ_task=[task8])
    task5 = Task(5, succ_task=[task9])
    task4 = Task(4, succ_task=[task8, task9])
    task3 = Task(3, succ_task=[task7])
    task2 = Task(2, succ_task=[task8, task9])
    task1 = Task(1, succ_task=[task2, task3, task4, task5, task6])

    # Set up predecessor relationships
    task10.pred_tasks = [task7, task8, task9]
    task9.pred_tasks = [task2, task4, task5]
    task8.pred_tasks = [task2, task4, task6]
    task7.pred_tasks = [task3]
    task6.pred_tasks = [task1]
    task5.pred_tasks = [task1]
    task4.pred_tasks = [task1]
    task3.pred_tasks = [task1]
    task2.pred_tasks = [task1]
    task1.pred_tasks = []

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]

    # Print task graph summary
    print("Task Graph:")
    for task in tasks:
        preds = [p.id for p in task.pred_tasks]
        succs = [s.id for s in task.succ_tasks]
        print(f"Task {task.id}: Predecessors {preds} | Successors {succs}")

    # Define device and edge power consumption values (keep these constant)
    device_core_powers = {0: 30.0, 1: 40.0, 2: 50.0}  # High power consumption for device cores
    edge_node_powers = {
        (1, 1): 15.0, (1, 2): 18.0,  # First edge node cores
        (2, 1): 12.0, (2, 2): 14.0  # Second edge node cores (slightly more efficient)
    }

    # APPROACH 1: RUN WITH A SINGLE SET OF RANDOMIZED PARAMETERS
    print("\n===== RUNNING WITH RANDOMIZED PARAMETERS =====")
    # Generate realistic parameters
    task_data_sizes, rf_power, upload_rates, download_rates = generate_realistic_network_parameters(
        num_tasks=10, num_edge_nodes=2, seed=42
    )

    # Print some of the generated parameters
    print("\nGenerated Parameter Examples:")
    print(f"Upload rate (device→cloud): {upload_rates['device_to_cloud']:.2f} Mbps")
    print(f"Upload rate (device→edge1): {upload_rates['device_to_edge1']:.2f} Mbps")
    print(f"Download rate (cloud→device): {download_rates['cloud_to_device']:.2f} Mbps")
    print(f"RF power (device→cloud): {rf_power['device_to_cloud']:.2f} W")
    print(f"Task 1 data size (device→cloud): {task_data_sizes[1]['device_to_cloud']:.2f} MB")

    # Run the scheduler with these parameters
    result = run_scheduler_with_params(
        tasks, task_data_sizes, rf_power, upload_rates, download_rates,
        device_core_powers, edge_node_powers
    )

    # Print results
    print("\nSingle Run Results:")
    print(f"Initial Time: {result['initial_time']:.2f}")
    print(f"Initial Energy: {result['initial_energy']:.2f}")
    print(f"Final Time: {result['final_time']:.2f}")
    print(f"Final Energy: {result['final_energy']:.2f}")
    print(f"Time Change: {result['time_change_pct']:+.2f}%")
    print(f"Energy Reduction: {result['energy_reduction_pct']:+.2f}%")
    print(
        f"Task Distribution: Cloud={result['cloud_tasks']}, Edge={result['edge_tasks']}, Device={result['device_tasks']}")

    # Print final schedule
    print("\nFinal Schedule:")
    for i, seq in enumerate(result['optimized_sequences']):
        print(f"Unit {i}: {seq}")

    # APPROACH 2: RUN ACROSS MULTIPLE SCENARIOS
    print("\n\n===== TESTING ACROSS MULTIPLE SCENARIOS =====")

    # Generate test scenarios
    num_scenarios = 5  # Generate 5 different scenarios
    scenarios = create_constrained_parameter_scenarios(
        num_scenarios=num_scenarios, num_tasks=10, num_edge_nodes=2, base_seed=42
    )

    # Run the scheduler on each scenario
    scenario_results = {}

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}...")
        # Run the scheduler with scenario parameters
        scenario_result = run_scheduler_with_params(
            tasks,
            scenario['task_data_sizes'],
            scenario['rf_power'],
            scenario['upload_rates'],
            scenario['download_rates'],
            device_core_powers,
            edge_node_powers
        )

        # Store results
        scenario_results[scenario['name']] = scenario_result

    # Print comparative results
    print("\n===== COMPARATIVE RESULTS ACROSS SCENARIOS =====")
    print(
        f"{'Scenario':<25} {'Init Time':<10} {'Final Time':<10} {'Time Δ%':<10} {'Energy Δ%':<10} {'Cloud':<6} {'Edge':<6} {'Device':<6}")
    print("-" * 85)

    for name, result in scenario_results.items():
        print(f"{name:<25} {result['initial_time']:<10.2f} {result['final_time']:<10.2f} "
              f"{result['time_change_pct']:+<10.2f} {result['energy_reduction_pct']:+<10.2f} "
              f"{result['cloud_tasks']:<6} {result['edge_tasks']:<6} {result['device_tasks']:<6}")

    # Calculate and print averages
    avg_time_change = sum(r['time_change_pct'] for r in scenario_results.values()) / len(scenario_results)
    avg_energy_reduction = sum(r['energy_reduction_pct'] for r in scenario_results.values()) / len(scenario_results)
    avg_cloud = sum(r['cloud_tasks'] for r in scenario_results.values()) / len(scenario_results)
    avg_edge = sum(r['edge_tasks'] for r in scenario_results.values()) / len(scenario_results)
    avg_device = sum(r['device_tasks'] for r in scenario_results.values()) / len(scenario_results)

    print("-" * 85)
    print(f"{'AVERAGE':<25} {'':<10} {'':<10} "
          f"{avg_time_change:+<10.2f} {avg_energy_reduction:+<10.2f} "
          f"{avg_cloud:<6.1f} {avg_edge:<6.1f} {avg_device:<6.1f}")

    # Analyze the scenarios to find patterns
    print("\n===== ANALYSIS OF OPTIMIZATION PATTERNS =====")

    # 1. Which scenario had the best energy reduction?
    best_energy_scenario = max(scenario_results.items(), key=lambda x: x[1]['energy_reduction_pct'])
    print(f"Best energy reduction: {best_energy_scenario[0]} ({best_energy_scenario[1]['energy_reduction_pct']:.2f}%)")

    # 2. Which scenario had the worst time change?
    worst_time_scenario = min(scenario_results.items(), key=lambda x: x[1]['time_change_pct'])
    print(f"Worst time performance: {worst_time_scenario[0]} ({worst_time_scenario[1]['time_change_pct']:.2f}%)")

    # 3. Which scenario used edge nodes most effectively?
    max_edge_scenario = max(scenario_results.items(), key=lambda x: x[1]['edge_tasks'])
    print(f"Most edge utilization: {max_edge_scenario[0]} ({max_edge_scenario[1]['edge_tasks']} tasks)")

    print("\nConclusion: The MCC framework shows variable performance across different network conditions.")
    print("This demonstrates the importance of dynamic parameter adaptation in real-world deployments.")
