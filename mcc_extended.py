from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, NamedTuple
from copy import deepcopy
import time as time_module
import networkx as nx
import heapq
import logging
from math import inf

# Constants and configuration data
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

core_execution_times = {
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

edge_execution_times = {
    # Format: (task_id, edge_node_id, core_id): execution_time
    (1, 1, 1): 8, (1, 1, 2): 6, (1, 2, 1): 7, (1, 2, 2): 5,
    (2, 1, 1): 7, (2, 1, 2): 5, (2, 2, 1): 6, (2, 2, 2): 4,
}
# Original cloud execution parameters
# [T_send, T_cloud, T_receive]
cloud_execution_times = [3, 1, 1]
# Number of edge nodes in the system
M = 2  # M edge nodes {E_1, E_2, ..., E_M}


####################################
# SECTION: CORE DATA STRUCTURES
####################################

class ExecutionTier(Enum):
    """Defines where a task can be executed in the three-tier architecture"""
    DEVICE = 0  # Mobile device (local cores)
    EDGE = 1  # Edge nodes (intermediate tier)
    CLOUD = 2  # Cloud platform


class SchedulingState(Enum):
    """Task scheduling algorithm states"""
    UNSCHEDULED = 0  # Initial state
    SCHEDULED = 1  # After initial minimal-delay scheduling
    KERNEL_SCHEDULED = 2  # After energy optimization


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

    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        # Set migration complexity based on source and target tiers
        if self.source_tier == self.target_tier:
            if self.source_tier == ExecutionTier.DEVICE:
                # Device to different device core
                self.migration_complexity = 1
            elif self.source_tier == ExecutionTier.EDGE:
                # Edge to different edge
                if self.source_location and self.target_location and self.source_location[0] == self.target_location[0]:
                    # Same edge node, different core
                    self.migration_complexity = 1
                else:
                    # Different edge nodes
                    self.migration_complexity = 2
            else:
                # Cloud to cloud (shouldn't happen)
                self.migration_complexity = 0
        else:
            # Cross-tier migration
            src_idx = self.source_tier.value
            tgt_idx = self.target_tier.value
            # Set complexity based on distance between tiers
            self.migration_complexity = 1 + abs(src_idx - tgt_idx)


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

    def calculate_local_finish_time(self, core, start_time):
        """Calculate finish time if task runs on a local device core"""
        if core < 0 or core >= len(self.local_execution_times):
            raise ValueError(f"Invalid core ID {core} for task {self.id}")
        return start_time + self.local_execution_times[core]

    def calculate_cloud_finish_times(self, upload_start_time):
        """Calculate all cloud-related finish times"""
        # Calculate upload finish time (FT_i^ws)
        upload_finish = upload_start_time + self.cloud_execution_times[0]
        # Calculate cloud computation finish time (FT_i^c)
        cloud_finish = upload_finish + self.cloud_execution_times[1]
        # Calculate download finish time (FT_i^wr)
        download_finish = cloud_finish + self.cloud_execution_times[2]
        return upload_finish, cloud_finish, download_finish

    def calculate_edge_finish_time(self, edge_id, core_id, start_time):
        """Calculate finish time if task runs on a specific edge core"""
        # Get execution time for this edge node and core
        exec_time = self.get_edge_execution_time(edge_id, core_id)
        if exec_time is None:
            logger.warning(f"No execution time data for task {self.id} on edge {edge_id}, core {core_id}")
            return float('inf')
        return start_time + exec_time

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

    def calculate_edge_ready_time(self, task, edge_id, device_to_edge_ready=None,
                                  edge_to_edge_ready=None, cloud_to_edge_ready=None,
                                  upload_rates=None, download_rates=None):
        """
        Fixed calculation of ready time for edge execution.

        Parameters:
            task: The task to calculate ready time for
            edge_id: Edge node ID (0-based)
            device_to_edge_ready: Dictionary of device to edge channel readiness times
            edge_to_edge_ready: Matrix of edge to edge channel readiness times
            cloud_to_edge_ready: Dictionary of cloud to edge channel readiness times
            upload_rates: Dictionary of upload rates for different channels
            download_rates: Dictionary of download rates for different channels
        """
        if not task.pred_tasks:
            return 0  # Entry task

        # Default values if not provided
        device_to_edge_ready = device_to_edge_ready or {}
        edge_to_edge_ready = edge_to_edge_ready or [[0 for _ in range(10)] for _ in range(10)]
        cloud_to_edge_ready = cloud_to_edge_ready or {}
        upload_rates = upload_rates or {}
        download_rates = download_rates or {}

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            # Skip unscheduled predecessors
            if pred_task.is_scheduled != SchedulingState.SCHEDULED:
                return float('inf')

            # Calculate ready time based on predecessor location
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor on device, need device→edge transfer
                pred_finish = pred_task.FT_l

                # Add device→edge transfer time
                data_key = f'device_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = upload_rates.get(data_key, 1.0)

                # Check channel availability and calculate transfer time
                channel_available = device_to_edge_ready.get(edge_id, 0)
                transfer_start = max(pred_finish, channel_available)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time

            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor on some edge node
                if not pred_task.edge_assignment:
                    return float('inf')

                pred_edge_id = pred_task.edge_assignment.edge_id - 1

                if pred_edge_id == edge_id:
                    # Same edge node - core-to-core time might be needed
                    core_id = pred_task.edge_assignment.core_id - 1
                    ready_time = pred_task.FT_edge.get(pred_edge_id, float('inf'))

                    # Add small core-to-core transfer time if different cores
                    if task.edge_assignment and task.edge_assignment.core_id - 1 != core_id:
                        ready_time += 0.1  # Small overhead for core-to-core transfer
                else:
                    # Different edge - need edge-to-edge transfer
                    if pred_edge_id not in pred_task.FT_edge:
                        return float('inf')

                    pred_finish = pred_task.FT_edge[pred_edge_id]

                    # Calculate edge-to-edge transfer time
                    data_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                    data_size = task.data_sizes.get(data_key, 0)
                    rate = upload_rates.get(data_key, 1.0)

                    # Check edge-to-edge channel availability
                    channel_available = edge_to_edge_ready[pred_edge_id][edge_id]
                    transfer_start = max(pred_finish, channel_available)
                    transfer_time = data_size / rate if rate > 0 else 0
                    ready_time = transfer_start + transfer_time

            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Cloud to edge transfer
                pred_finish = pred_task.FT_c

                # Add cloud-to-edge transfer time
                data_key = f'cloud_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = download_rates.get(data_key, 1.0)

                # Check channel availability
                channel_available = cloud_to_edge_ready.get(edge_id, 0)
                transfer_start = max(pred_finish, channel_available)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time
            else:
                return float('inf')

            max_ready_time = max(max_ready_time, ready_time)

        # Also consider the device-to-edge channel availability
        max_ready_time = max(max_ready_time, device_to_edge_ready.get(edge_id, 0))

        return max_ready_time

    def calculate_ready_time_cloud_upload(self, download_rates):
        if not self.pred_tasks:
            return 0  # Entry task, ready at time 0

        max_ready_time = 0

        for pred_task in self.pred_tasks:
            # Ensure predecessor has been scheduled
            if pred_task.is_scheduled == SchedulingState.UNSCHEDULED:
                logger.warning(f"Predecessor task {pred_task.id} of task {self.id} not yet scheduled")
                return float('inf')  # Not ready until predecessor is scheduled

            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor executed locally
                if pred_task.FT_l <= 0:
                    logger.warning(f"Invalid finish time for predecessor {pred_task.id} on device")
                    return float('inf')
                ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor uploaded to cloud already
                if pred_task.FT_ws <= 0:
                    logger.warning(f"Invalid sending finish time for predecessor {pred_task.id} to cloud")
                    return float('inf')
                ready_time = pred_task.FT_ws
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor executed on edge
                if not pred_task.edge_assignment:
                    logger.warning(f"Edge assignment missing for predecessor {pred_task.id}")
                    return float('inf')

                pred_edge_id = pred_task.edge_assignment.edge_id
                # Check if data was sent to cloud directly from edge
                cloud_key = ('edge', 'cloud')
                if cloud_key in pred_task.FT_edge_send:
                    # Data already sent to cloud
                    ready_time = pred_task.FT_edge_send[cloud_key]
                else:
                    # Need to wait for edge execution and then add transfer time
                    if pred_edge_id not in pred_task.FT_edge:
                        logger.warning(f"Missing finish time for predecessor {pred_task.id} on edge {pred_edge_id}")
                        return float('inf')

                    edge_finish_time = pred_task.FT_edge[pred_edge_id]
                    # Calculate time to transfer from edge to device first
                    transfer_to_device = pred_task.calculate_data_transfer_time(
                        ExecutionTier.EDGE, ExecutionTier.DEVICE,
                        upload_rates_dict={},  # Not needed for this direction
                        download_rates_dict=download_rates,
                        source_location=(pred_edge_id, 0)
                    )
                    ready_time = edge_finish_time + transfer_to_device
            else:
                logger.error(f"Unknown execution tier for predecessor {pred_task.id}")
                return float('inf')

            max_ready_time = max(max_ready_time, ready_time)

        return max_ready_time

    def get_overall_finish_time(self):
        finish_times = []
        if self.FT_l > 0:
            finish_times.append(self.FT_l)
        # Add cloud finish time if applicable (when results received)
        if self.FT_wr > 0:
            finish_times.append(self.FT_wr)
        # Add edge finish times if applicable
        for edge_id, finish_time in self.FT_edge_receive.items():
            if finish_time > 0:
                finish_times.append(finish_time)
        if not finish_times:
            return -1  # Not scheduled yet
        return max(finish_times)


####################################
# SECTION: UTILITY CLASSES
####################################

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

        # Calculate total number of execution units
        self.total_units = (
                num_device_cores +  # Device cores
                num_edge_nodes * num_edge_cores_per_node +  # Edge cores
                1  # Cloud platform
        )

        # Initialize sequences for all execution units
        self.sequences = [[] for _ in range(self.total_units)]
        # Map to quickly find sequence index for a specific execution unit
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
        # Add cloud
        cloud_index = self.total_units - 1
        self.unit_to_index_map[ExecutionUnit(ExecutionTier.CLOUD)] = cloud_index

    def set_all_sequences(self, sequences: List[List[int]]) -> None:
        if len(sequences) != self.total_units:
            raise ValueError(f"Expected {self.total_units} sequences, got {len(sequences)}")
        self.sequences = deepcopy(sequences)


####################################
# SECTION: TASK GRAPH MANAGEMENT
####################################
def generate_task_graph(
        num_tasks=10,
        density=0.3,
        complexity_range=(0.5, 5.0),
        data_intensity_range=(0.2, 2.0),
        task_type_weights=None,
        predefined_tasks=None
):
    """
    Generates or enhances a DAG of tasks with complexity, data intensity,
    and data_sizes. Logs key attributes to confirm proper assignment.
    """
    if predefined_tasks is not None:
        # Just enhance the existing tasks instead of generating random DAG
        return enhance_predefined_tasks(
            predefined_tasks,
            complexity_range=complexity_range,
            data_intensity_range=data_intensity_range,
            task_type_weights=task_type_weights
        )

    # Otherwise, build a random DAG from scratch
    G = nx.DiGraph()

    # Add nodes
    for i in range(1, num_tasks + 1):
        G.add_node(i)

    # If no custom distribution of compute/data/balanced is provided
    if task_type_weights is None:
        task_type_weights = {
            'compute': 0.3,
            'data': 0.3,
            'balanced': 0.4
        }

    # Randomly assign task type
    task_types = {}
    for i in range(1, num_tasks + 1):
        task_type = random.choices(
            list(task_type_weights.keys()),
            weights=list(task_type_weights.values())
        )[0]
        task_types[i] = task_type

        # Set complexity and data intensity
        if task_type == 'compute':
            complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
            data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
        elif task_type == 'data':
            complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
            data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
        else:  # 'balanced'
            complexity = random.uniform(complexity_range[0], complexity_range[1])
            data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[1])

        # Store results in G for reference
        G.nodes[i]['complexity'] = complexity
        G.nodes[i]['data_intensity'] = data_intensity
        G.nodes[i]['type'] = task_type

    # Build edges with given density, ensuring DAG remains acyclic
    nodes = list(range(1, num_tasks + 1))
    random.shuffle(nodes)

    # Basic approach: add edges with probability `density`
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < density:
                G.add_edge(nodes[i], nodes[j])

    # Create Task objects
    tasks = {}
    for i in range(1, num_tasks + 1):
        task_obj = Task(
            id=i,
            complexity=G.nodes[i]['complexity'],
            data_intensity=G.nodes[i]['data_intensity']
        )
        # (Optionally) fill data_sizes here or in 'enhance_predefined_tasks'
        # e.g. task_obj.data_sizes = {...}
        tasks[i] = task_obj

    # Add predecessor/successor references
    for i in range(1, num_tasks + 1):
        # preds
        preds = list(G.predecessors(i))
        for p in preds:
            tasks[i].pred_tasks.append(tasks[p])
        # succs
        succs = list(G.successors(i))
        for s in succs:
            tasks[i].succ_tasks.append(tasks[s])

    # Log out each newly created task's attributes
    for t_id, t_obj in tasks.items():
        logger.info(
            f"[generate_task_graph] Created Task {t_id} | "
            f"Type={G.nodes[t_id]['type']} | "
            f"C={t_obj.complexity:.2f} | D={t_obj.data_intensity:.2f} | "
            f"DataSizes={t_obj.data_sizes}"
        )

    return list(tasks.values())


def enhance_predefined_tasks(
        tasks,
        complexity_range=(0.5, 5.0),
        data_intensity_range=(0.2, 2.0),
        task_type_weights=None
):
    """
    Adds or refines attributes (complexity, data intensity, type, data_sizes)
    to a pre-existing list of Task objects. Logs the changes to confirm success.
    """
    import random

    if task_type_weights is None:
        task_type_weights = {
            'compute': 0.3,
            'data': 0.3,
            'balanced': 0.4
        }

    # Build a small graph from the tasks to identify structure
    import networkx as nx
    G = nx.DiGraph()
    for t in tasks:
        G.add_node(t.id)
    for t in tasks:
        for succ in t.succ_tasks:
            G.add_edge(t.id, succ.id)

    # For each task, randomly choose or infer a type, complexity, data intensity
    # if they aren't already set or we want to override them
    for task in tasks:
        # Heuristic: if it has no type assigned, randomly pick one
        if not hasattr(task, 'task_type') or task.task_type is None:
            ttype = random.choices(
                list(task_type_weights.keys()),
                weights=list(task_type_weights.values())
            )[0]
            task.task_type = ttype
        else:
            ttype = task.task_type

        # Depending on type, set complexity/data_intensity if needed
        if task.complexity is None:
            if ttype == 'compute':
                task.complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
            elif ttype == 'data':
                task.complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
            else:
                task.complexity = random.uniform(complexity_range[0], complexity_range[1])

        if task.data_intensity is None:
            if ttype == 'data':
                task.data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
            elif ttype == 'compute':
                task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
            else:
                task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[1])

        # You can also generate 'data_sizes' if needed for each possible transfer:
        if not task.data_sizes or task.data_sizes == -1:
            # Example: fill with random or fixed placeholders
            task.data_sizes = {
                'device_to_cloud': random.uniform(1.0, 5.0),
                'cloud_to_device': random.uniform(1.0, 5.0),
                'device_to_edge1': random.uniform(0.5, 2.0),
                'edge1_to_device': random.uniform(0.5, 2.0),
                # extend to edge2, etc. as needed
            }

        # Log the final assigned attributes for verification
        logger.info(
            f"[enhance_predefined_tasks] Enhanced Task {task.id}: "
            f"Type={task.task_type}, "
            f"C={task.complexity:.2f}, "
            f"D={task.data_intensity:.2f}, "
            f"DataSizes={task.data_sizes}"
        )

    return tasks


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


####################################
# SECTION: SIMULATION MODELS
####################################
def generate_power_models(device_type='mobile', battery_level=100):
    """
    Generate power consumption models that vary with load.

    Parameters:
        device_type: Type of device ('mobile', 'edge_server', 'cloud_server')
        battery_level: Current battery percentage (mobile only)

    Returns:
        Dictionary of power models for different processing units
    """
    # Base power models with load-dependent functions
    power_models = {
        'device': {},
        'edge': {},
        'cloud': {},
        'rf': {}
    }

    # Mobile device power characteristics
    if device_type == 'mobile':
        # Battery efficiency drops as battery level decreases
        battery_factor = 1.0 if battery_level > 30 else 1.0 + (30 - battery_level) * 0.01

        # Different cores have different power profiles
        # Power = idle_power + (load * dynamic_power * frequency^2)
        power_models['device'] = {
            0: {  # High-performance core
                'idle_power': 0.1 * battery_factor,
                'dynamic_power': lambda load: (0.2 + 1.8 * load) * battery_factor,
                'frequency_range': (0.8, 2.4),  # GHz
                'current_frequency': 2.0,
                'dvfs_enabled': True
            },
            1: {  # Mid-range core
                'idle_power': 0.05 * battery_factor,
                'dynamic_power': lambda load: (0.1 + 1.4 * load) * battery_factor,
                'frequency_range': (0.6, 1.8),
                'current_frequency': 1.6,
                'dvfs_enabled': True
            },
            2: {  # Efficiency core
                'idle_power': 0.03 * battery_factor,
                'dynamic_power': lambda load: (0.05 + 0.95 * load) * battery_factor,
                'frequency_range': (0.5, 1.5),
                'current_frequency': 1.2,
                'dvfs_enabled': True
            }
        }

        # RF transmission power varies with data rate
        power_models['rf'] = {
            'device_to_edge': lambda data_rate: 0.1 + 0.4 * (data_rate / 10),
            'device_to_cloud': lambda data_rate: 0.15 + 0.6 * (data_rate / 5),
        }

    # Edge server power characteristics
    elif device_type == 'edge_server':
        for edge_id in range(1, 3):  # Two edge nodes
            for core_id in range(1, 3):  # Two cores per node
                efficiency = 1.0 - 0.1 * (edge_id - 1) - 0.05 * (core_id - 1)  # First edge, first core most efficient
                power_models['edge'][(edge_id, core_id)] = {
                    'idle_power': 5.0 * efficiency,
                    'dynamic_power': lambda load: (3.0 + 12.0 * load) * efficiency,
                    'frequency_range': (1.0, 3.2),
                    'current_frequency': 2.8,
                    'dvfs_enabled': True
                }

    # Cloud server power characteristics
    elif device_type == 'cloud_server':
        power_models['cloud'] = {
            'idle_power': 50.0,
            'dynamic_power': lambda load: (20.0 + 180.0 * load),
            'frequency_range': (2.0, 4.0),
            'current_frequency': 3.5,
            'virtualization_overhead': 0.1  # 10% overhead due to virtualization
        }

    return power_models


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


def initialize_edge_execution_times(tasks=None):
    """
    Create reasonable edge execution times for all tasks

    Parameters:
        tasks: Optional list of Task objects to update directly

    Returns:
        Dictionary of edge execution times
    """
    global edge_execution_times

    # Clear existing dictionary if it exists
    if 'edge_execution_times' in globals():
        edge_execution_times.clear()
    else:
        edge_execution_times = {}

    for task_id in range(1, 11):  # For tasks 1-10 in our sample
        # Get local execution times for this task
        local_times = core_execution_times.get(task_id, [9, 7, 5])
        min_local = min(local_times)
        avg_local = sum(local_times) / len(local_times)

        # Cloud time for comparison
        cloud_time = sum(cloud_execution_times)

        # Different edge performance profiles for different task types
        for edge_id in range(1, 3):  # Two edge nodes
            for core_id in range(1, 3):  # Two cores per edge
                # Data intensive tasks (odd IDs in our example) - edge performs well
                if task_id % 2 == 1:
                    # Edge slightly better than best local core
                    edge_time = min_local * 0.9
                # Compute intensive tasks (even IDs) - cloud might perform better
                else:
                    # Edge between local and cloud performance
                    edge_time = (min_local + cloud_time) / 2

                # Add variability between edge nodes and cores
                if edge_id == 2:
                    edge_time *= 1.1  # Second edge node slightly slower
                if core_id == 2:
                    edge_time *= 1.05  # Second core slightly slower

                # Store the edge execution time
                edge_execution_times[(task_id, edge_id, core_id)] = edge_time

                # Also update each task object directly if tasks were provided
                if tasks is not None:
                    task = next((t for t in tasks if t.id == task_id), None)
                    if task:
                        if not hasattr(task, 'edge_execution_times'):
                            task.edge_execution_times = {}
                        task.edge_execution_times[(edge_id, core_id)] = edge_time

    task_count = len(edge_execution_times) // 4  # 4 entries per task (2 nodes × 2 cores)
    logger.info(f"Initialized edge execution times for {task_count} tasks")
    return edge_execution_times


import random
import datetime


def generate_realistic_network_conditions(time_of_day=None):
    """
    Generates separate dictionaries for upload_rates and download_rates
    based on time-of-day congestion patterns.

    Returns:
        (upload_rates, download_rates): two dicts
    """

    # Use current local hour if not specified
    if time_of_day is None:
        time_of_day = datetime.datetime.now().hour

    # Base rates in Mbps (upload direction)
    base_upload = {
        'device_to_edge': 10.0,  # e.g., device -> edge
        'edge_to_edge': 30.0,  # e.g., edge1 -> edge2
        'edge_to_cloud': 50.0,  # e.g., edge -> cloud
        'device_to_cloud': 5.0,  # e.g., device -> cloud
    }

    # Base rates in Mbps (download direction)
    base_download = {
        'edge_to_device': 12.0,  # e.g., edge -> device
        'cloud_to_edge': 60.0,  # e.g., cloud -> edge
        'edge_to_edge': 30.0,  # e.g., edge2 -> edge1 (symmetric for simplicity)
        'cloud_to_device': 6.0,  # e.g., cloud -> device
    }

    # Time-of-day factor (peak/off-peak)
    tod_factor = 1.0
    if 9 <= time_of_day <= 11 or 19 <= time_of_day <= 21:
        tod_factor = 0.7  # 30% slowdown at peak hours
    elif 0 <= time_of_day <= 5:
        tod_factor = 1.3  # 30% speed-up at late night

    # Random fluctuation (±15%)
    random_factor = random.uniform(0.85, 1.15)

    # Build final rate dicts
    upload_rates = {}
    download_rates = {}

    # Apply factors to each base rate
    for link, base_rate in base_upload.items():
        effective_rate = base_rate * tod_factor * random_factor
        upload_rates[link] = effective_rate

    for link, base_rate in base_download.items():
        effective_rate = base_rate * tod_factor * random_factor
        download_rates[link] = effective_rate

    # 5% chance to degrade a random link by 70%
    if random.random() < 0.05:
        # Decide if we degrade an upload or download link
        choice_is_upload = bool(random.getrandbits(1))
        if choice_is_upload:
            trouble_link = random.choice(list(upload_rates.keys()))
            upload_rates[trouble_link] *= 0.3
        else:
            trouble_link = random.choice(list(download_rates.keys()))
            download_rates[trouble_link] *= 0.3

    return upload_rates, download_rates


####################################
# SECTION: UTILITY FUNCTIONS
####################################

def get_execution_unit_from_index(index: int) -> ExecutionUnit:
    """
    Converts a linear index to an ExecutionUnit using globally available parameters.

    Args:
        index: Linear index representing an execution unit

    Returns:
        ExecutionUnit corresponding to the provided index
    """
    # Get the structure parameters from globals or use defaults
    num_device_cores = globals().get('num_device_cores', 3)
    num_edge_nodes = globals().get('num_edge_nodes', 2)
    num_edge_cores_per_node = globals().get('num_edge_cores_per_node', 2)

    # Device cores
    if index < num_device_cores:
        return ExecutionUnit(ExecutionTier.DEVICE, (index,))

    # Edge cores
    edge_offset = num_device_cores
    total_edge_cores = num_edge_nodes * num_edge_cores_per_node

    if index < edge_offset + total_edge_cores:
        edge_index = index - edge_offset
        node_id = edge_index // num_edge_cores_per_node
        core_id = edge_index % num_edge_cores_per_node
        return ExecutionUnit(ExecutionTier.EDGE, (node_id, core_id))

    # Cloud
    return ExecutionUnit(ExecutionTier.CLOUD)


def get_current_execution_unit(task: Any) -> ExecutionUnit:
    """
    Get the current execution unit for a task.
    Works with both three-tier tasks and original MCC tasks.

    Args:
        task: Task object

    Returns:
        ExecutionUnit where the task is currently scheduled
    """
    if hasattr(task, 'execution_tier'):
        # Three-tier task
        tier = task.execution_tier

        if tier == ExecutionTier.DEVICE:
            return ExecutionUnit(tier, (task.device_core,))
        elif tier == ExecutionTier.EDGE and task.edge_assignment:
            return ExecutionUnit(tier, (
                task.edge_assignment.edge_id - 1,  # Convert to 0-based
                task.edge_assignment.core_id - 1  # Convert to 0-based
            ))
        else:
            return ExecutionUnit(ExecutionTier.CLOUD)
    else:
        # Original MCC task
        if hasattr(task, 'is_core_task') and task.is_core_task:
            return ExecutionUnit(ExecutionTier.DEVICE, (task.assignment,))
        else:
            return ExecutionUnit(ExecutionTier.CLOUD)


def generate_migration_cache_key(tasks: List[Any], task_id: int, source_unit: ExecutionUnit,
                                 target_unit: ExecutionUnit) -> Tuple:
    """
    Generate a cache key for a specific migration.

    Args:
        tasks: List of all tasks
        task_id: ID of task being migrated
        source_unit: Source execution unit
        target_unit: Target execution unit

    Returns:
        Tuple that uniquely identifies this migration
    """
    # Encode execution units
    source_key = encode_execution_unit(source_unit)
    target_key = encode_execution_unit(target_unit)

    # Encode current task assignments
    assignments = tuple(encode_task_assignment(task) for task in tasks)

    # Combine everything into a single key
    return task_id, source_key, target_key, assignments


def encode_execution_unit(unit: ExecutionUnit) -> Tuple:
    """
    Encode an execution unit into a hashable format for caching.

    Args:
        unit: ExecutionUnit to encode

    Returns:
        Tuple representation of the execution unit
    """
    tier_value = unit.tier.value

    if unit.tier == ExecutionTier.DEVICE:
        return tier_value, unit.location[0]  # (DEVICE, core_id)
    elif unit.tier == ExecutionTier.EDGE:
        return tier_value, unit.location[0], unit.location[1]  # (EDGE, node_id, core_id)
    else:
        return (tier_value,)  # (CLOUD,)


def encode_task_assignment(task: Any) -> Tuple:
    """
    Encode a task assignment into a hashable format for caching.

    Args:
        task: Task object to encode

    Returns:
        Tuple representation of the task assignment
    """
    if hasattr(task, 'execution_tier'):
        # Three-tier task
        tier = task.execution_tier.value

        if task.execution_tier == ExecutionTier.DEVICE:
            return tier, task.device_core
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            return tier, task.edge_assignment.edge_id, task.edge_assignment.core_id
        else:
            return (tier,)
    else:
        # Original MCC task
        if hasattr(task, 'is_core_task') and task.is_core_task:
            return 0, task.assignment  # (DEVICE, core_id)
        else:
            return (2,)  # (CLOUD,)


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

        # NEW: A place to record the final “minimal-delay” schedule decisions
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
            start_time = max(pred_finish, self.calculate_edge_ready_time(task, e_id))
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
        best_finish = inf
        best_core = None
        for core_id in range(self.edge_cores):
            actual_start = max(start_time, self.edge_core_earliest_ready[edge_id][core_id])
            exec_time = task.get_edge_execution_time(edge_id + 1, core_id + 1) or inf
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

        For edge tasks, properly calculates and records edge-to-device transfer times.
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

            logger.info(
                f"Task {task.id} scheduled on DEVICE core={core_id} from {start:.2f} to {finish_time:.2f}"
            )

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

            logger.info(
                f"Task {task.id} scheduled on EDGE node={e_id}, core={c_id} from {start:.2f} to {finish_time:.2f} "
                f"(results arrive at device at {receive_time:.2f})"
            )

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

            logger.info(
                f"Task {task.id} scheduled on CLOUD from {cloud_ready:.2f} to {finish_time:.2f}"
            )

        else:
            logger.error(f"Unknown resource kind: {resource_kind}")

    # -------------------------------------------------------------------------
    #   Helper: earliest time all preds have completed
    # -------------------------------------------------------------------------
    def earliest_pred_finish_time(self, task):
        """
        For strict precedence, a task can't start until all predecessors are done.
        Return the max finish time among all of task's predecessors,
        properly accounting for edge-to-device transfer times.
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
                pred_finish = pred.FT_l

            elif pred.execution_tier == ExecutionTier.CLOUD:
                # Cloud task - results available after download
                pred_finish = pred.FT_wr

            elif pred.execution_tier == ExecutionTier.EDGE:
                # Edge task - must wait for edge-to-device transfer
                if not pred.edge_assignment:
                    logger.warning(f"Edge assignment missing for predecessor {pred.id}")
                    return float('inf')

                edge_id = pred.edge_assignment.edge_id - 1  # 0-based index

                # Check if edge-to-device transfer time is recorded
                if hasattr(pred, 'FT_edge_receive') and edge_id in pred.FT_edge_receive:
                    pred_finish = pred.FT_edge_receive[edge_id]
                else:
                    # If not recorded, calculate it now
                    if not hasattr(pred, 'FT_edge') or edge_id not in pred.FT_edge:
                        logger.warning(f"Missing edge finish time for predecessor {pred.id}")
                        return float('inf')

                    edge_finish = pred.FT_edge[edge_id]
                    edge_to_device_key = f'edge{edge_id + 1}_to_device'
                    data_size = pred.data_sizes.get(edge_to_device_key, 1.0)
                    download_rate = self.download_rates.get(edge_to_device_key, 1.0)
                    transfer_time = data_size / download_rate if download_rate > 0 else 0

                    # Store calculated time for future reference
                    if not hasattr(pred, 'FT_edge_receive'):
                        pred.FT_edge_receive = {}

                    pred_finish = edge_finish + transfer_time
                    pred.FT_edge_receive[edge_id] = pred_finish

                    logger.info(
                        f"Calculated result arrival time for task {pred.id}: edge finish {edge_finish:.2f} + transfer {transfer_time:.2f} = {pred_finish:.2f}")
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
        # plus the predecessor’s finish. If you want “infinite concurrency,”
        # you’d skip the shared ws_ready. For now we do:
        return max(self.ws_ready, self.earliest_pred_finish_time(task))

    # -------------------------------------------------------------------------
    #   Helper: compute finishing time if assigned to cloud
    # -------------------------------------------------------------------------
    def get_cloud_finish(self, task, upload_start):
        t_send, t_cloud, t_recv = task.cloud_execution_times
        # If you assume unlimited concurrency in the cloud, it’s just:
        return upload_start + t_send + t_cloud + t_recv

    # -------------------------------------------------------------------------
    #   Helper: earliest time a task can start on edge e_id
    # -------------------------------------------------------------------------
    def calculate_edge_ready_time(self, task, e_id):
        # In the simplest approach, you look at each predecessor’s finish time
        # plus data transfer from that predecessor’s location to this edge.
        # This code is a placeholder that you may refine:
        pred_finish = self.earliest_pred_finish_time(task)
        # If you also track e.g. device->edge or edge->edge readiness, do:
        # e.g. return max(pred_finish, self.device_to_edge_ready[e_id])
        return max(pred_finish, self.device_to_edge_ready[e_id])

    # -------------------------------------------------------------------------
    #   3) Record the Final “Minimal-Delay” Schedule
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
    Returns the overall application completion time for the three-tier system
    by analogy to Equation (10) in the original MCC paper.

    We assume each task has .execution_finish_time set after scheduling.
    """
    # Identify exit tasks: tasks with no successors
    exit_tasks = [t for t in tasks if not t.succ_tasks]
    if not exit_tasks:
        return 0.0  # If your DAG has no explicit exit tasks, fallback

    # Each task in your code has .execution_finish_time after scheduling
    return max(t.execution_finish_time for t in exit_tasks)


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


def generate_realistic_power_models(device_type='mobile', battery_level=100):
    """
    Generate realistic power consumption models that vary with load.

    Parameters:
        device_type: Type of device ('mobile', 'edge_server', 'cloud_server')
        battery_level: Current battery percentage (mobile only)

    Returns:
        Dictionary of power models for different processing units
    """
    # Base power models with load-dependent functions
    power_models = {
        'device': {},
        'edge': {},
        'cloud': {},
        'rf': {}
    }

    # Mobile device power characteristics
    if device_type == 'mobile':
        # Battery efficiency drops as battery level decreases
        battery_factor = 1.0 if battery_level > 30 else 1.0 + (30 - battery_level) * 0.01

        # Different cores have different power profiles
        # Power = idle_power + (load * dynamic_power * frequency^2)
        power_models['device'] = {
            0: {  # High-performance core
                'idle_power': 0.1 * battery_factor,
                'dynamic_power': lambda load: (0.2 + 1.8 * load) * battery_factor,
                'frequency_range': (0.8, 2.4),  # GHz
                'current_frequency': 2.0,
                'dvfs_enabled': True
            },
            1: {  # Mid-range core
                'idle_power': 0.05 * battery_factor,
                'dynamic_power': lambda load: (0.1 + 1.4 * load) * battery_factor,
                'frequency_range': (0.6, 1.8),
                'current_frequency': 1.6,
                'dvfs_enabled': True
            },
            2: {  # Efficiency core
                'idle_power': 0.03 * battery_factor,
                'dynamic_power': lambda load: (0.05 + 0.95 * load) * battery_factor,
                'frequency_range': (0.5, 1.5),
                'current_frequency': 1.2,
                'dvfs_enabled': True
            }
        }

        # RF transmission power varies with signal strength and data rate
        power_models['rf'] = {
            'device_to_edge': lambda data_rate, signal_strength:
            (0.1 + 0.4 * (data_rate / 10) * (1 + (70 - signal_strength) * 0.02)) * battery_factor,
            'device_to_cloud': lambda data_rate, signal_strength:
            (0.15 + 0.6 * (data_rate / 5) * (1 + (70 - signal_strength) * 0.03)) * battery_factor,
        }

    # Edge server power characteristics - more power efficient than mobile, less than cloud
    elif device_type == 'edge_server':
        for edge_id in range(1, 3):  # Two edge nodes
            for core_id in range(1, 3):  # Two cores per node
                efficiency = 1.0 - 0.1 * (edge_id - 1) - 0.05 * (core_id - 1)  # First edge, first core most efficient
                power_models['edge'][(edge_id, core_id)] = {
                    'idle_power': 5.0 * efficiency,
                    'dynamic_power': lambda load: (3.0 + 12.0 * load) * efficiency,
                    'frequency_range': (1.0, 3.2),
                    'current_frequency': 2.8,
                    'dvfs_enabled': True
                }

    # Cloud server power characteristics - highest absolute power but most efficient per computation
    elif device_type == 'cloud_server':
        power_models['cloud'] = {
            'idle_power': 50.0,
            'dynamic_power': lambda load: (20.0 + 180.0 * load),
            'frequency_range': (2.0, 4.0),
            'current_frequency': 3.5,
            'virtualization_overhead': 0.1  # 10% overhead due to virtualization
        }

    return power_models


def format_schedule_3tier(tasks, scheduler):
    """
    Builds a formatted table of the scheduled tasks in a three-tier environment.
    Shows:
        Task  |  Tier        |  Start Time  |  Finish Time |  Resource

    Args:
        tasks: List of Task objects that have:
               - execution_tier
               - device_core
               - edge_assignment
               - execution_finish_time
               - execution_unit_task_start_times
        scheduler: A reference to the ThreeTierTaskScheduler (or whichever object
                   provides get_edge_core_index() and get_cloud_index()) so we can
                   determine the correct index for edge/cloud resources.

    Returns:
        A multi-line string containing the formatted schedule table.
    """

    header = f"{'Task':<5}  {'Tier':<10}  {'Start':>7}  {'Finish':>7}  Resource"
    sep_line = "-" * len(header)

    lines = [header, sep_line]

    for t in tasks:
        # Retrieve finish time
        finish_time = getattr(t, 'execution_finish_time', 0.0)
        tier_name = (t.execution_tier.name if hasattr(t, 'execution_tier') and t.execution_tier
                     else "UNASSIGNED")

        # Default or fallback if we cannot find the correct start
        start_time = 0.0
        resource_str = "N/A"

        if t.execution_tier == ExecutionTier.DEVICE:
            # e.g. "Device Core 2"
            core_id = getattr(t, 'device_core', -1)
            resource_str = f"Device Core {core_id}"
            # If we have valid start times array and a valid core index
            if (t.execution_unit_task_start_times
                    and 0 <= core_id < len(t.execution_unit_task_start_times)):
                start_time = t.execution_unit_task_start_times[core_id]

        elif t.execution_tier == ExecutionTier.EDGE and t.edge_assignment:
            # e.g. "Edge(Node=1, Core=2)"
            e_id = t.edge_assignment.edge_id - 1  # 0-based
            c_id = t.edge_assignment.core_id - 1  # 0-based
            resource_str = f"Edge(Node={e_id + 1}, Core={c_id + 1})"

            # get the sequence index from your scheduler method
            seq_idx = scheduler.get_edge_core_index(e_id, c_id)

            if (t.execution_unit_task_start_times
                    and 0 <= seq_idx < len(t.execution_unit_task_start_times)):
                start_time = t.execution_unit_task_start_times[seq_idx]

        elif t.execution_tier == ExecutionTier.CLOUD:
            resource_str = "Cloud"
            # If your code places the cloud start time at cloud_idx
            cloud_idx = scheduler.get_cloud_index()
            if (t.execution_unit_task_start_times
                    and 0 <= cloud_idx < len(t.execution_unit_task_start_times)):
                start_time = t.execution_unit_task_start_times[cloud_idx]

        # Build a single row for the table
        lines.append(
            f"{t.id:<5}  {tier_name:<10}  {start_time:7.2f}  {finish_time:7.2f}  {resource_str}"
        )

    return "\n".join(lines)


def validate_three_tier_schedule(
        tasks: List[Any],
        sequences: List[List[int]],
        num_device_cores: int = 3,
        num_edge_nodes: int = 2,
        num_edge_cores_per_node: int = 2
) -> Dict[str, Any]:
    """
    Validates a three-tier schedule for:
     - Whether tasks are assigned or left unscheduled
     - Whether DAG dependency constraints are violated
     - Whether tasks appear in the resource sequences
     - Whether tasks are assigned to valid resources
     - Whether tier-specific fields are consistent
    Returns a dictionary summarizing validity and any issues.
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

    # ----------------------------------------------------
    # 1) Check if all tasks are scheduled
    # ----------------------------------------------------
    for task in tasks:
        # We'll see if it's "scheduled" by checking key attributes
        scheduled = False
        if hasattr(task, 'execution_tier'):
            # For your three-tier tasks, we expect one of the three:
            if task.execution_tier == ExecutionTier.DEVICE:
                # Must have device_core >= 0
                if (hasattr(task, 'device_core') and
                        task.device_core is not None and
                        task.device_core >= 0):
                    scheduled = True
            elif task.execution_tier == ExecutionTier.EDGE:
                # Must have a valid EdgeAssignment
                if (hasattr(task, 'edge_assignment') and
                        task.edge_assignment is not None):
                    scheduled = True
            elif task.execution_tier == ExecutionTier.CLOUD:
                # If it's assigned to the Cloud, we assume "scheduled"
                scheduled = True
        else:
            # Original MCC-like
            if hasattr(task, 'assignment') and task.assignment >= 0:
                scheduled = True

        if not scheduled:
            validation["unscheduled_tasks"] += 1
            task_id = getattr(task, 'id', '?')
            validation["issues"].append(f"Task {task_id} has no valid assignment")

    # ----------------------------------------------------
    # 2) Check DAG dependency constraints
    # ----------------------------------------------------
    # We’ll define a small helper to confirm that each child
    # appears after all its predecessors.
    # This function might either check scheduling times or just that "predecessors exist".
    # If you want to do time-based checks, you'd have to gather the actual start/finish times
    # from the tasks. Otherwise, we do a structural check that you haven't assigned children
    # with missing or contradictory preds. For brevity, we'll do a structural check here.

    if not validate_task_dependencies(tasks):
        # If your code has a separate function for that, fine.
        validation["valid"] = False
        validation["dependency_violations"] += 1
        validation["issues"].append("Task dependency constraints are violated")

    # ----------------------------------------------------
    # 3) Check sequence consistency
    #    Ensure each task appears 0 or 1 times in your resource sequences
    # ----------------------------------------------------

    # We’ll see how many tasks exist total
    task_counts = {}
    for t in tasks:
        tid = getattr(t, 'id', None)
        if tid is not None:
            task_counts[tid] = task_counts.get(tid, 0) + 1

    # Count how many times each task ID appears in the sequences
    sequence_task_counts = {}
    for seq in sequences:
        for tid in seq:
            sequence_task_counts[tid] = sequence_task_counts.get(tid, 0) + 1

    # Check for tasks that appear multiple times in the sequences
    for tid, count in sequence_task_counts.items():
        if count > 1:
            validation["sequence_violations"] += 1
            validation["issues"].append(f"Task {tid} appears in {count} resource sequences")

    # Check for tasks that never appear
    for tid, count in task_counts.items():
        if tid not in sequence_task_counts:
            validation["sequence_violations"] += 1
            validation["issues"].append(f"Task {tid} is not in any sequence")

    # ----------------------------------------------------
    # 4) Check resource assignments are valid
    # ----------------------------------------------------
    # E.g. device_core must be in [0, num_device_cores-1].
    # EdgeAssignment must be in [1..num_edge_nodes], [1..num_edge_cores_per_node].
    for task in tasks:
        if hasattr(task, 'execution_tier'):
            if task.execution_tier == ExecutionTier.DEVICE:
                # Must have 0 <= device_core < num_device_cores
                if (hasattr(task, 'device_core') and
                        (task.device_core < 0 or task.device_core >= num_device_cores)):
                    tid = getattr(task, 'id', '?')
                    validation["resource_violations"] += 1
                    validation["issues"].append(
                        f"Task {tid} assigned to invalid device core {task.device_core}"
                    )

            elif task.execution_tier == ExecutionTier.EDGE:
                if (hasattr(task, 'edge_assignment') and
                        task.edge_assignment):
                    edge_id = task.edge_assignment.edge_id
                    core_id = task.edge_assignment.core_id
                    # Usually edge_id in [1..M], core_id in [1..num_edge_cores_per_node]
                    if (edge_id < 1 or edge_id > num_edge_nodes):
                        validation["resource_violations"] += 1
                        validation["issues"].append(
                            f"Task {getattr(task, 'id', '?')} assigned to invalid edge node {edge_id}"
                        )
                    if (core_id < 1 or core_id > num_edge_cores_per_node):
                        validation["resource_violations"] += 1
                        validation["issues"].append(
                            f"Task {getattr(task, 'id', '?')} assigned to invalid edge core {core_id}"
                        )

            elif task.execution_tier == ExecutionTier.CLOUD:
                # Typically no specific resource index checks for cloud
                pass

    # ----------------------------------------------------
    # 5) Tier consistency checks
    # ----------------------------------------------------
    # E.g. if it's a device task, do we have FT_l > 0? If edge, do we see FT_edge? etc.
    for task in tasks:
        tid = getattr(task, 'id', '?')
        if task.execution_tier == ExecutionTier.DEVICE:
            # It's possible the code hasn't yet set FT_l if concurrency pass assigned it,
            # but let's see if you want to enforce that. We'll do a soft check:
            if hasattr(task, 'edge_assignment') and task.edge_assignment:
                validation["tier_violations"] += 1
                validation["issues"].append(
                    f"Task {tid} is a device task but has an edge_assignment"
                )

            # If we want to confirm it was actually scheduled and has a positive finish time:
            if hasattr(task, 'FT_l') and task.FT_l <= 0:
                # This might be a clue that it never ran or concurrency pass didn't set it
                validation["tier_violations"] += 1
                validation["issues"].append(
                    f"Task {tid} is device-tier but has no local finish time"
                )

        elif task.execution_tier == ExecutionTier.EDGE:
            # We check if it has FT_edge
            if not hasattr(task, 'edge_assignment') or not task.edge_assignment:
                validation["tier_violations"] += 1
                validation["issues"].append(
                    f"Task {tid} is edge-tier but has no edge_assignment"
                )

        elif task.execution_tier == ExecutionTier.CLOUD:
            # Check if it has some FT_ws, FT_c, FT_wr
            if (getattr(task, 'FT_ws', 0) <= 0 or
                    getattr(task, 'FT_c', 0) <= 0 or
                    getattr(task, 'FT_wr', 0) <= 0):
                validation["tier_violations"] += 1
                validation["issues"].append(
                    f"Task {tid} is cloud-tier but missing some cloud times"
                )

    # ----------------------------------------------------
    # Summarize
    # ----------------------------------------------------
    if (validation["dependency_violations"] > 0 or
            validation["sequence_violations"] > 0 or
            validation["resource_violations"] > 0 or
            validation["tier_violations"] > 0 or
            validation["unscheduled_tasks"] > 0):
        validation["valid"] = False

    return validation


def validate_task_dependencies(tasks, epsilon=1e-9):
    """
    Verifies that each scheduled task starts strictly after all of its immediate
    predecessors finish, factoring in how data travels between tiers (device, edge, cloud).

    Args:
        tasks: List of Task objects with fields:
               - is_scheduled: SchedulingState
               - execution_tier: ExecutionTier (DEVICE, EDGE, CLOUD)
               - device_core: int (>=0 if device tier)
               - FT_l, FT_ws, FT_c, FT_wr: float times for device & cloud
               - FT_edge, FT_edge_receive, FT_edge_send for edge tasks
               - local_execution_times, cloud_execution_times, etc. for durations
               - pred_tasks: list of immediate predecessor tasks
        epsilon: float tolerance to allow small rounding differences

    Returns:
        (is_valid, violations): Tuple where:
          - is_valid: bool, True if no violations
          - violations: list of dicts describing each violation
    """
    violations = []

    for task in tasks:
        # Skip tasks that aren't scheduled at all
        if task.is_scheduled == SchedulingState.UNSCHEDULED:
            continue

        # For each immediate predecessor, we must ensure that predecessor finishes
        # before this task can start. The exact check depends on both tasks' tiers.
        for pred_task in getattr(task, 'pred_tasks', []):
            # If predecessor is unscheduled, we can't fully check timing
            # but we can record a violation (or skip it).
            if pred_task.is_scheduled == SchedulingState.UNSCHEDULED:
                violations.append({
                    'type': 'Unscheduled Predecessor',
                    'task': task.id,
                    'predecessor': pred_task.id,
                    'detail': f"Task {task.id} depends on unscheduled task {pred_task.id}"
                })
                continue

            # 1) If the child (this task) is on the DEVICE tier
            if task.execution_tier == ExecutionTier.DEVICE:
                # We'll figure out the child's local start time by
                # (child_finish_time - child's local_execution_time on that core).
                if (not hasattr(task, 'device_core') or
                        task.device_core < 0 or
                        not hasattr(task, 'FT_l')):
                    violations.append({
                        'type': 'Invalid Device Execution',
                        'task': task.id,
                        'detail': f"Task {task.id} is device-tier but missing device_core/FT_l"
                    })
                    continue

                # How long does the child take on that core?
                core_idx = task.device_core
                if (not hasattr(task, 'local_execution_times') or
                        core_idx >= len(task.local_execution_times)):
                    violations.append({
                        'type': 'Invalid Device Core Index',
                        'task': task.id,
                        'detail': f"Task {task.id} has local_execution_times but index {core_idx} is invalid"
                    })
                    continue

                child_finish = task.FT_l
                child_exec_time = task.local_execution_times[core_idx]
                child_start = child_finish - child_exec_time

                # Now see how the predecessor finishes (depending on its tier).
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # The predecessor's FT_l must be <= child_start
                    pred_finish = getattr(pred_task, 'FT_l', 0.0)
                    if (pred_finish - child_start) > epsilon:
                        violations.append({
                            'type': 'Device-Device Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} starts at {child_start:.3f} but pred {pred_task.id} ends at {pred_finish:.3f}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # The predecessor's FT_wr must be <= child_start
                    pred_finish = getattr(pred_task, 'FT_wr', 0.0)
                    if (pred_finish - child_start) > epsilon:
                        violations.append({
                            'type': 'Cloud-Device Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} starts at {child_start:.3f} but cloud {pred_task.id} finishes receiving at {pred_finish:.3f}"
                        })

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # The predecessor's results must arrive at device
                    # i.e. FT_edge_receive[...] must be <= child_start
                    if not hasattr(pred_task, 'edge_assignment') or not pred_task.edge_assignment:
                        violations.append({
                            'type': 'Missing Edge Assignment',
                            'predecessor': pred_task.id,
                            'detail': f"Edge predecessor {pred_task.id} is missing edge_assignment"
                        })
                        continue

                    edge_id = pred_task.edge_assignment.edge_id - 1
                    pred_finish = float('inf')
                    if hasattr(pred_task, 'FT_edge_receive'):
                        # Attempt the dictionary get for that edge
                        pred_finish = pred_task.FT_edge_receive.get(edge_id,
                                                                    pred_task.FT_edge_receive.get(edge_id + 1,
                                                                                                  float('inf')))
                    if (pred_finish - child_start) > epsilon:
                        violations.append({
                            'type': 'Edge-Device Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} starts at {child_start:.3f} but edge {pred_task.id} arrives at {pred_finish:.3f}"
                        })

            # 2) If the child is on the CLOUD tier
            elif task.execution_tier == ExecutionTier.CLOUD:
                # We'll approximate child's "upload start" = FT_ws - T_i^s
                if (not hasattr(task, 'FT_ws') or not hasattr(task, 'cloud_execution_times')):
                    violations.append({
                        'type': 'Invalid Cloud Execution',
                        'task': task.id,
                        'detail': f"Task {task.id} is cloud-tier but missing FT_ws or cloud_execution_times"
                    })
                    continue

                t_send = task.cloud_execution_times[0]
                upload_start = task.FT_ws - t_send

                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # The predecessor's FT_l must be <= upload_start
                    pred_finish = getattr(pred_task, 'FT_l', 0.0)
                    if (pred_finish - upload_start) > epsilon:
                        violations.append({
                            'type': 'Device-Cloud Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} uploads at {upload_start:.3f} but pred {pred_task.id} ends device at {pred_finish:.3f}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # The predecessor's FT_ws must be <= child's upload_start
                    pred_finish = getattr(pred_task, 'FT_ws', 0.0)
                    if (pred_finish - upload_start) > epsilon:
                        violations.append({
                            'type': 'Cloud-Cloud Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} uploads at {upload_start:.3f} but pred {pred_task.id} done uploading at {pred_finish:.3f}"
                        })

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # Check if data was sent directly from edge to cloud
                    if hasattr(pred_task, 'FT_edge_send') and ('edge', 'cloud') in pred_task.FT_edge_send:
                        # Data was directly sent from edge to cloud
                        pred_finish = pred_task.FT_edge_send[('edge', 'cloud')]
                    else:
                        # Data needs to go through device first
                        if not hasattr(pred_task, 'edge_assignment') or not pred_task.edge_assignment:
                            violations.append({
                                'type': 'Missing Edge Assignment',
                                'predecessor': pred_task.id,
                                'detail': f"Edge predecessor {pred_task.id} is missing edge_assignment"
                            })
                            continue

                        edge_id = pred_task.edge_assignment.edge_id - 1
                        # First check if we have edge-to-device transfer time recorded
                        if hasattr(pred_task, 'FT_edge_receive') and edge_id in pred_task.FT_edge_receive:
                            # Results arrived at device, so we use that time
                            pred_finish = pred_task.FT_edge_receive[edge_id]
                        else:
                            # Fallback to edge execution finish time
                            if hasattr(pred_task, 'FT_edge') and edge_id in pred_task.FT_edge:
                                pred_finish = pred_task.FT_edge[edge_id]
                            else:
                                violations.append({
                                    'type': 'Missing Edge Finish Time',
                                    'predecessor': pred_task.id,
                                    'detail': f"Edge predecessor {pred_task.id} has no finish time for edge {edge_id + 1}"
                                })
                                continue

                    if (pred_finish - upload_start) > epsilon:
                        violations.append({
                            'type': 'Edge-Cloud Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} uploads at {upload_start:.3f} but edge {pred_task.id} finishes at {pred_finish:.3f}"
                        })

            # 3) If the child is on the EDGE tier
            elif task.execution_tier == ExecutionTier.EDGE:
                # We'll figure out child's start time from FT_edge and exec_time
                if (not hasattr(task, 'edge_assignment') or
                        not hasattr(task, 'FT_edge')):
                    violations.append({
                        'type': 'Invalid Edge Execution',
                        'task': task.id,
                        'detail': f"Task {task.id} is edge-tier but missing assignment or FT_edge"
                    })
                    continue

                e_id = task.edge_assignment.edge_id - 1
                if e_id in task.FT_edge:
                    child_finish = task.FT_edge[e_id]
                else:
                    child_finish = float('inf')

                # Look up how long the child runs on that edge
                if hasattr(task, 'get_edge_execution_time'):
                    c_id = task.edge_assignment.core_id
                    exec_time = task.get_edge_execution_time(e_id + 1, c_id)
                else:
                    exec_time = 0.0
                child_start = child_finish - exec_time

                # Get actual start time if available from execution_unit_task_start_times
                if hasattr(task, 'execution_unit_task_start_times') and task.execution_unit_task_start_times:
                    # Convert (e_id, c_id) into a sequence index (similar to scheduler's get_edge_core_index)
                    seq_idx = len(task.local_execution_times) + e_id * 2 + c_id - 1  # Approximation
                    if 0 <= seq_idx < len(task.execution_unit_task_start_times):
                        recorded_start = task.execution_unit_task_start_times[seq_idx]
                        if recorded_start > 0:
                            child_start = recorded_start

                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # For device->edge transfers:
                    # - Check if the predecessor's device execution is complete
                    # - Data needs to be transferred from device to edge
                    pred_finish = getattr(pred_task, 'FT_l', 0.0)

                    # Add a minimal transfer time estimate
                    # Ideally, we would use actual device-to-edge transfer times if recorded
                    if pred_finish <= 0:
                        violations.append({
                            'type': 'Invalid Device Finish Time',
                            'predecessor': pred_task.id,
                            'detail': f"Device predecessor {pred_task.id} has invalid finish time"
                        })
                        continue

                    if (pred_finish - child_start) > epsilon:
                        violations.append({
                            'type': 'Device-Edge Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} starts on edge at {child_start:.3f} but device {pred_task.id} finishes at {pred_finish:.3f}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # For cloud->edge transfers:
                    # - Check if predecessor's cloud computation is complete (FT_c)
                    # - Data needs to be transferred from cloud to edge
                    pred_cloud_finish = getattr(pred_task, 'FT_c', 0.0)

                    # Check if we have a specific cloud-to-edge transfer recorded
                    pred_finish = pred_cloud_finish
                    if hasattr(pred_task, 'FT_cloud_to_edge') and e_id in pred_task.FT_cloud_to_edge:
                        pred_finish = pred_task.FT_cloud_to_edge[e_id]

                    if pred_finish <= 0:
                        violations.append({
                            'type': 'Invalid Cloud Finish Time',
                            'predecessor': pred_task.id,
                            'detail': f"Cloud predecessor {pred_task.id} has invalid finish time"
                        })
                        continue

                    if (pred_finish - child_start) > epsilon:
                        violations.append({
                            'type': 'Cloud-Edge Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} starts on edge at {child_start:.3f} but cloud {pred_task.id} finishes at {pred_finish:.3f}"
                        })

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # For edge->edge transfers:
                    if not hasattr(pred_task, 'edge_assignment') or not pred_task.edge_assignment:
                        violations.append({
                            'type': 'Missing Edge Assignment',
                            'predecessor': pred_task.id,
                            'detail': f"Edge predecessor {pred_task.id} is missing edge_assignment"
                        })
                        continue

                    pred_e_id = pred_task.edge_assignment.edge_id - 1

                    # Check if predecessor is on the same edge
                    if pred_e_id == e_id:
                        # Same edge, check if predecessor's execution is complete
                        if hasattr(pred_task, 'FT_edge') and pred_e_id in pred_task.FT_edge:
                            pred_finish = pred_task.FT_edge[pred_e_id]
                        else:
                            violations.append({
                                'type': 'Missing Edge Finish Time',
                                'predecessor': pred_task.id,
                                'detail': f"Edge predecessor {pred_task.id} has no finish time for edge {pred_e_id + 1}"
                            })
                            continue
                    else:
                        # Different edges, check for edge-to-edge transfer
                        if hasattr(pred_task, 'FT_edge_send') and (pred_e_id, e_id) in pred_task.FT_edge_send:
                            # We have direct edge-to-edge transfer time
                            pred_finish = pred_task.FT_edge_send[(pred_e_id, e_id)]
                        else:
                            # Fall back to edge execution finish time plus a minimal transfer time
                            if hasattr(pred_task, 'FT_edge') and pred_e_id in pred_task.FT_edge:
                                pred_finish = pred_task.FT_edge[
                                                  pred_e_id] + 1.0  # Add minimal edge-to-edge transfer time
                            else:
                                violations.append({
                                    'type': 'Missing Edge Finish Time',
                                    'predecessor': pred_task.id,
                                    'detail': f"Edge predecessor {pred_task.id} has no finish time for edge {pred_e_id + 1}"
                                })
                                continue

                    if (pred_finish - child_start) > epsilon:
                        violations.append({
                            'type': 'Edge-Edge Dependency',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Child {task.id} starts on edge {e_id + 1} at {child_start:.3f} but predecessor {pred_task.id} on edge {pred_e_id + 1} finishes at {pred_finish:.3f}"
                        })

    # Finally, if we found no violations, we are good
    is_valid = (len(violations) == 0)
    return is_valid, violations


def plot_three_tier_gantt(tasks, scheduler, title="Three-Tier Schedule"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Get basic dimensions from scheduler
    num_device_cores = scheduler.k
    num_edge_nodes = scheduler.M
    num_edge_cores_per_node = scheduler.edge_cores

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))

    # Map task IDs to task objects for easy lookup
    task_map = {t.id: t for t in tasks}

    # Define colors for different execution units and operations
    colors = {
        'device': 'lightcoral',
        'edge': 'lightgreen',
        'cloud_send': 'lightskyblue',
        'cloud_compute': 'royalblue',
        'cloud_receive': 'mediumslateblue',
        'edge_to_device': 'palegreen',  # Transfer from edge to device
    }

    # Helper function to add centered text to bars
    def add_centered_text(ax, start, duration, y_level, task_id):
        center_x = start + duration / 2

        # Pre-measure text to see if it fits in the bar
        renderer = ax.figure.canvas.get_renderer()
        text_obj = ax.text(0, 0, f"T{task_id}", fontsize=10, fontweight='bold')
        bbox = text_obj.get_window_extent(renderer=renderer)
        text_obj.remove()

        trans = ax.transData.inverted()
        text_width = trans.transform((bbox.width, 0))[0] - trans.transform((0, 0))[0]

        if text_width > duration * 0.8:
            # Text won't fit inside bar, so place it above
            ax.text(center_x, y_level + 0.3, f"T{task_id}",
                    va='bottom', ha='center',
                    color='black', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        else:
            # Text fits inside bar
            ax.text(center_x, y_level, f"T{task_id}",
                    va='center', ha='center',
                    color='black', fontsize=10, fontweight='bold')

    # Calculate maximum completion time across all tasks
    max_completion_time = max(t.execution_finish_time for t in tasks)

    # Prepare y-positions for each resource
    yticks = []
    ytick_labels = []

    # Position counter (from bottom to top)
    y_pos = 0
    y_positions = {}

    # Cloud positions (bottom of chart)
    y_positions['cloud_send'] = y_pos
    yticks.append(y_pos)
    ytick_labels.append('Cloud Upload')
    y_pos += 1

    y_positions['cloud_compute'] = y_pos
    yticks.append(y_pos)
    ytick_labels.append('Cloud Compute')
    y_pos += 1

    y_positions['cloud_receive'] = y_pos
    yticks.append(y_pos)
    ytick_labels.append('Cloud Download')
    y_pos += 1

    # Edge transfer channels
    for e_id in range(num_edge_nodes):
        y_positions[f'edge{e_id}_to_device'] = y_pos
        yticks.append(y_pos)
        ytick_labels.append(f'Edge {e_id + 1} → Device')
        y_pos += 1

    # Edge node cores
    for e_id in range(num_edge_nodes):
        for c_id in range(num_edge_cores_per_node):
            y_positions[f'edge{e_id}_core{c_id}'] = y_pos
            yticks.append(y_pos)
            ytick_labels.append(f'Edge {e_id + 1} Core {c_id + 1}')
            y_pos += 1

    # Device cores
    for core_id in range(num_device_cores):
        y_positions[f'device_core{core_id}'] = y_pos
        yticks.append(y_pos)
        ytick_labels.append(f'Device Core {core_id + 1}')
        y_pos += 1

    # Plot tasks on device cores
    for task in tasks:
        if task.execution_tier == ExecutionTier.DEVICE:
            core_id = task.device_core
            y_level = y_positions[f'device_core{core_id}']

            if hasattr(task, 'execution_unit_task_start_times') and task.execution_unit_task_start_times:
                start_time = task.execution_unit_task_start_times[core_id]
                duration = task.FT_l - start_time

                ax.barh(y_level, duration, left=start_time, height=0.6,
                        align='center', color=colors['device'], edgecolor='black')
                add_centered_text(ax, start_time, duration, y_level, task.id)

    # Plot tasks on edge nodes
    for task in tasks:
        if task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            e_id = task.edge_assignment.edge_id - 1  # Convert to 0-based
            c_id = task.edge_assignment.core_id - 1  # Convert to 0-based
            y_level = y_positions[f'edge{e_id}_core{c_id}']

            # Edge execution
            if hasattr(task, 'execution_unit_task_start_times') and task.execution_unit_task_start_times:
                seq_idx = scheduler.get_edge_core_index(e_id, c_id)
                if seq_idx < len(task.execution_unit_task_start_times):
                    start_time = task.execution_unit_task_start_times[seq_idx]
                    if e_id in task.FT_edge:
                        finish_time = task.FT_edge[e_id]
                        duration = finish_time - start_time

                        ax.barh(y_level, duration, left=start_time, height=0.6,
                                align='center', color=colors['edge'], edgecolor='black')
                        add_centered_text(ax, start_time, duration, y_level, task.id)

            # Edge-to-device transfer
            if hasattr(task, 'FT_edge') and hasattr(task, 'FT_edge_receive'):
                if e_id in task.FT_edge and e_id in task.FT_edge_receive:
                    start_time = task.FT_edge[e_id]
                    finish_time = task.FT_edge_receive[e_id]
                    duration = finish_time - start_time

                    y_level = y_positions[f'edge{e_id}_to_device']
                    ax.barh(y_level, duration, left=start_time, height=0.6,
                            align='center', color=colors['edge_to_device'], edgecolor='black', hatch='///')
                    add_centered_text(ax, start_time, duration, y_level, task.id)

    # Plot tasks on cloud
    for task in tasks:
        if task.execution_tier == ExecutionTier.CLOUD:
            # Cloud sending phase
            send_start = task.RT_ws
            send_duration = task.FT_ws - task.RT_ws
            y_level = y_positions['cloud_send']
            ax.barh(y_level, send_duration, left=send_start, height=0.6,
                    align='center', color=colors['cloud_send'], edgecolor='black')
            add_centered_text(ax, send_start, send_duration, y_level, task.id)

            # Cloud computing phase
            compute_start = task.RT_c
            compute_duration = task.FT_c - task.RT_c
            y_level = y_positions['cloud_compute']
            ax.barh(y_level, compute_duration, left=compute_start, height=0.6,
                    align='center', color=colors['cloud_compute'], edgecolor='black')
            add_centered_text(ax, compute_start, compute_duration, y_level, task.id)

            # Cloud receiving phase
            receive_start = task.RT_wr
            receive_duration = task.FT_wr - task.RT_wr
            y_level = y_positions['cloud_receive']
            ax.barh(y_level, receive_duration, left=receive_start, height=0.6,
                    align='center', color=colors['cloud_receive'], edgecolor='black')
            add_centered_text(ax, receive_start, receive_duration, y_level, task.id)

    # Configure axis
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Execution Resource")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0, max_completion_time * 1.05)

    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['device'], edgecolor='black', label='Device Execution'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['edge'], edgecolor='black', label='Edge Execution'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['cloud_send'], edgecolor='black', label='Cloud Upload'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['cloud_compute'], edgecolor='black', label='Cloud Computation'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['cloud_receive'], edgecolor='black', label='Cloud Download'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['edge_to_device'], edgecolor='black', hatch='///',
                      label='Edge→Device Transfer')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Draw vertical lines for key time points to make it easier to track dependencies
    for time_point in range(int(max_completion_time) + 1):
        ax.axvline(x=time_point, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

####################################
# SECTION: MAIN PROGRAM EXECUTION
####################################
if __name__ == "__main__":
    # 1) Realistic network conditions
    upload_rates, download_rates = generate_realistic_network_conditions()

    # 2) Generate mobile power models (which includes 'device' cores and 'rf')
    mobile_power_models = generate_realistic_power_models(device_type='mobile', battery_level=65)
    device_power_profiles = mobile_power_models.get('device', {})
    rf_power = mobile_power_models.get('rf', {})

    # (Optional) If you want to populate or re-populate edge_execution_times:
    # initialize_edge_execution_times()

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

    predefined_tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12, task13, task14,
             task15, task16, task17, task18, task19, task20]

    # 4) Enhance tasks with complexity, data sizes, etc.
    tasks = generate_task_graph(predefined_tasks=predefined_tasks)

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
        upload_rates=upload_rates,  # uses the real dictionary from step 1
        download_rates=download_rates  # ditto
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

    # 10) total_time_3tier => max finish time among exit tasks
    T_total = total_time_3tier(tasks)

    # 11) total_energy_3tier_with_rf => device energy usage
    E_total = total_energy_3tier_with_rf(
        tasks,
        device_power_profiles=device_power_profiles,
        rf_power=rf_power,
        # You can pass the same dictionary from step 1, or a subset:
        upload_rates={
            'device_to_edge': upload_rates.get('device_to_edge', 6.0),
            'device_to_cloud': upload_rates.get('device_to_cloud', 3.0)
        },
        default_signal_strength=65.0
    )

    print("\n=== SCHEDULING RESULTS ===")
    print(f"Total Completion Time (Three-Tier): {T_total:.2f}")
    print(f"Total Device Energy Consumption: {E_total:.4f} J")

    # 12) Print a fully formatted schedule table (showing start times)
    schedule_str = format_schedule_3tier(tasks, scheduler)
    print("\nFormatted schedule (3-tier concurrency-based start times):")
    print(schedule_str)

    is_valid, violations = validate_task_dependencies(tasks)
    if is_valid:
        print("\nSchedule is valid: no dependency violations.")
    else:
        print("\nSchedule has DAG/time violations:")
        for v in violations:
            print(f" - {v}")
    plot_three_tier_gantt(tasks,scheduler)