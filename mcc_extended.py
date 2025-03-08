import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set, Any, NamedTuple, Callable
from copy import deepcopy
from collections import deque
import time as time_module
import random
import networkx as nx
import matplotlib.pyplot as plt
import datetime

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
        self.pred_tasks = pred_task
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

    def calculate_ready_time_local(self):
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
                # Predecessor executed in cloud, need to wait for results
                if pred_task.FT_wr <= 0:
                    logger.warning(f"Invalid receiving finish time for predecessor {pred_task.id} from cloud")
                    return float('inf')
                ready_time = pred_task.FT_wr
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor executed on edge, need to wait for results
                if not pred_task.edge_assignment:
                    logger.warning(f"Edge assignment missing for predecessor {pred_task.id}")
                    return float('inf')

                pred_edge_id = pred_task.edge_assignment.edge_id
                if pred_edge_id not in pred_task.FT_edge_receive:
                    logger.warning(f"Missing receive time for predecessor {pred_task.id} from edge {pred_edge_id}")
                    return float('inf')

                ready_time = pred_task.FT_edge_receive[pred_edge_id]
            else:
                logger.error(f"Unknown execution tier for predecessor {pred_task.id}")
                return float('inf')

            max_ready_time = max(max_ready_time, ready_time)

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
def generate_task_graph(num_tasks=10, density=0.3, complexity_range=(0.5, 5.0), data_intensity_range=(0.2, 2.0),
                        task_type_weights=None, predefined_tasks=None):
    """
    Generates a realistic task graph based on application patterns.
    Can use either a randomly generated graph or enhance a predefined task graph.

    Args:
        num_tasks: Number of tasks to generate (only used if predefined_tasks is None)
        density: Edge density of the task graph (0-1) (only used if predefined_tasks is None)
        complexity_range: Range for computational complexity
        data_intensity_range: Range for data intensity
        task_type_weights: Weights for different task types (compute-intensive, data-intensive, balanced)
        predefined_tasks: List of predefined Task objects (optional)

    Returns:
        List of Task objects with dependencies and realistic attributes
    """
    # If predefined tasks are provided, use those instead of generating random tasks
    if predefined_tasks is not None:
        return enhance_predefined_tasks(predefined_tasks, complexity_range, data_intensity_range, task_type_weights)

    # Create a directed acyclic graph for random generation
    G = nx.DiGraph()

    # Add nodes
    for i in range(1, num_tasks + 1):
        G.add_node(i)

    # Determine task types distribution
    if task_type_weights is None:
        task_type_weights = {
            'compute': 0.3,  # Compute-intensive tasks
            'data': 0.3,  # Data-intensive tasks
            'balanced': 0.4  # Balanced tasks
        }

    # Assign task types and characteristics
    task_types = {}
    for i in range(1, num_tasks + 1):
        task_type = random.choices(list(task_type_weights.keys()),
                                   weights=list(task_type_weights.values()))[0]
        task_types[i] = task_type

        # Assign complexity and data intensity based on task type
        if task_type == 'compute':
            # Compute-intensive: high complexity, low data intensity
            complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
            data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
        elif task_type == 'data':
            # Data-intensive: low complexity, high data intensity
            complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
            data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
        else:
            # Balanced: moderate complexity and data intensity
            complexity = random.uniform(complexity_range[0], complexity_range[1])
            data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[1])

        G.nodes[i]['complexity'] = complexity
        G.nodes[i]['data_intensity'] = data_intensity
        G.nodes[i]['type'] = task_type

    # Add edges (dependencies) using topological order to ensure acyclicity
    nodes = list(range(1, num_tasks + 1))
    random.shuffle(nodes)

    # Ensure at least one entry task (no predecessors)
    entry_tasks = nodes[:max(1, int(num_tasks * 0.1))]

    # Ensure at least one exit task (no successors)
    exit_tasks = nodes[-max(1, int(num_tasks * 0.1)):]

    # Add edges with probability based on density
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[j] not in exit_tasks and nodes[i] not in entry_tasks:
                if random.random() < density:
                    G.add_edge(nodes[i], nodes[j])

    # Ensure all nodes except entry tasks have at least one predecessor
    for node in nodes:
        if node not in entry_tasks and len(list(G.predecessors(node))) == 0:
            # Add edge from a random previous node
            predecessors = [n for n in nodes if n < node and n != node]
            if predecessors:
                G.add_edge(random.choice(predecessors), node)

    # Ensure all nodes except exit tasks have at least one successor
    for node in nodes:
        if node not in exit_tasks and len(list(G.successors(node))) == 0:
            # Add edge to a random next node
            successors = [n for n in nodes if n > node and n != node]
            if successors:
                G.add_edge(node, random.choice(successors))

    # Create Task objects
    tasks = {}
    for i in range(1, num_tasks + 1):
        tasks[i] = Task(i,
                        complexity=G.nodes[i]['complexity'],
                        data_intensity=G.nodes[i]['data_intensity'])

    # Set predecessors and successors
    for i in range(1, num_tasks + 1):
        for pred in G.predecessors(i):
            if tasks[i].pred_tasks is None:
                tasks[i].pred_tasks = []
            tasks[i].pred_tasks.append(tasks[pred])

        for succ in G.successors(i):
            if tasks[i].succ_tasks is None:
                tasks[i].succ_tasks = []
            tasks[i].succ_tasks.append(tasks[succ])

    # Convert to list and return
    return list(tasks.values())


def enhance_predefined_tasks(tasks, complexity_range=(0.5, 5.0), data_intensity_range=(0.2, 2.0),
                             task_type_weights=None):
    """
    Enhances predefined task objects with complexity and data intensity values.

    Args:
        tasks: List of predefined Task objects
        complexity_range: Range for computational complexity
        data_intensity_range: Range for data intensity
        task_type_weights: Weights for different task types

    Returns:
        Enhanced task list with realistic attributes
    """
    # Determine task types distribution
    if task_type_weights is None:
        task_type_weights = {
            'compute': 0.3,  # Compute-intensive tasks
            'data': 0.3,  # Data-intensive tasks
            'balanced': 0.4  # Balanced tasks
        }

    # Build a task lookup dictionary for easy access
    task_dict = {task.id: task for task in tasks}

    # Create a graph representation to analyze the structure
    G = nx.DiGraph()

    # Add nodes and edges
    for task in tasks:
        G.add_node(task.id)
        for succ in task.succ_tasks:
            G.add_edge(task.id, succ.id)

    # Identify entry and exit tasks
    entry_tasks = [task.id for task in tasks if not task.pred_tasks]
    exit_tasks = [task.id for task in tasks if not task.succ_tasks]

    # Assign task types and characteristics
    for task in tasks:
        # Determine task type based on position in graph
        # Edge tasks are likely data-intensive, intermediate tasks more compute-intensive
        if task.id in entry_tasks:
            probabilities = [0.2, 0.6, 0.2]  # [compute, data, balanced]
        elif task.id in exit_tasks:
            probabilities = [0.2, 0.6, 0.2]  # [compute, data, balanced]
        else:
            # Tasks in the middle of the graph are more likely compute-intensive
            probabilities = [0.6, 0.2, 0.2]  # [compute, data, balanced]

            # If it has many predecessors, it's likely a reduction operation (compute-intensive)
            if len(task.pred_tasks) > 2:
                probabilities = [0.7, 0.1, 0.2]

        task_type = random.choices(list(task_type_weights.keys()), weights=probabilities)[0]

        # Assign complexity and data intensity based on task type
        if task_type == 'compute':
            # Compute-intensive: high complexity, low data intensity
            task.complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
        elif task_type == 'data':
            # Data-intensive: low complexity, high data intensity
            task.complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
            task.data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
        else:
            # Balanced: moderate complexity and data intensity
            task.complexity = random.uniform(complexity_range[0], complexity_range[1])
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[1])

        # Store task type for later reference
        task.task_type = task_type

    return tasks


def primary_assignment(tasks, edge_nodes=None, edge_cores_per_node=2):
    """
    Performs initial assignment of tasks to execution tiers based on minimizing execution time.
    Takes into account task characteristics for more intelligent placement.

    Args:
        tasks: List of tasks to assign
        edge_nodes: Number of edge nodes in the system (default: use global M)
        edge_cores_per_node: Number of cores per edge node (default: 2)
    """
    if edge_nodes is None:
        edge_nodes = M  # Use global edge nodes count

    for task in tasks:
        # Calculate T_i^l_min (minimum local execution time)
        if not task.local_execution_times:
            t_l_min = float('inf')
            logger.warning(f"Task {task.id} has no local execution times defined")
            best_device_core = -1
        else:
            t_l_min = min(task.local_execution_times)
            best_device_core = task.local_execution_times.index(t_l_min)

        # Calculate minimum edge execution time
        t_edge_min = float('inf')
        best_edge_id = -1
        best_core_id = -1

        # Check all edge nodes and cores
        for edge_id in range(1, edge_nodes + 1):
            for core_id in range(1, edge_cores_per_node + 1):
                # Try different ways to get edge execution time
                edge_time = None

                # Try task-specific edge execution times from global dictionary
                key = (task.id, edge_id, core_id)
                if 'edge_execution_times' in globals() and key in edge_execution_times:
                    edge_time = edge_execution_times[key]

                # Try task's own edge execution times
                if edge_time is None and hasattr(task, 'edge_execution_times'):
                    key = (edge_id, core_id)
                    if key in task.edge_execution_times:
                        edge_time = task.edge_execution_times[key]

                # Calculate a reasonable fallback if execution time is still missing
                if edge_time is None and hasattr(task, 'local_execution_times') and task.local_execution_times:
                    # For data-intensive tasks (odd IDs), edge is slightly faster
                    if hasattr(task, 'task_type') and task.task_type == 'data':
                        avg_local = sum(task.local_execution_times) / len(task.local_execution_times)
                        edge_time = avg_local * 0.9
                    # For compute-intensive tasks, edge is between device and cloud
                    elif hasattr(task, 'task_type') and task.task_type == 'compute':
                        min_local = min(task.local_execution_times)
                        cloud_time = sum(task.cloud_execution_times)
                        edge_time = (min_local + cloud_time) / 2

                if edge_time is None:
                    continue  # Skip if no execution time available

                if edge_time < t_edge_min:
                    t_edge_min = edge_time
                    best_edge_id = edge_id
                    best_core_id = core_id

        # Calculate cloud execution time (sum of all phases)
        if hasattr(task, 'cloud_execution_times') and task.cloud_execution_times:
            t_cloud = sum(task.cloud_execution_times)
        else:
            t_cloud = float('inf')
            logger.warning(f"Task {task.id} has no cloud execution times defined")

        # Consider task characteristics for more intelligent placement
        is_data_intensive = False

        # Check explicit task type if available
        if hasattr(task, 'task_type') and task.task_type == 'data':
            is_data_intensive = True
        # Otherwise infer from data intensity vs complexity ratio
        elif hasattr(task, 'data_intensity') and hasattr(task, 'complexity'):
            is_data_intensive = task.data_intensity > task.complexity * 1.5
        # As a fallback, use task ID (odd IDs are data-intensive)
        elif task.id % 2 == 1:  # Odd IDs considered data-intensive
            is_data_intensive = True

        # Log comparison information
        logger.info(
            f"Task {task.id}: Device={t_l_min:.2f}, Edge={t_edge_min:.2f}, Cloud={t_cloud:.2f}, Data-intensive={is_data_intensive}")

        # Make assignment decision based on execution time and task characteristics
        if is_data_intensive and t_edge_min < t_l_min * 1.2:  # Allow edge to be slightly slower for data-intensive tasks
            # Data-intensive tasks often benefit from edge execution even if slightly slower
            task.execution_tier = ExecutionTier.EDGE
            task.edge_assignment = EdgeAssignment(edge_id=best_edge_id, core_id=best_core_id)
            logger.info(
                f"Task {task.id} assigned to EDGE (data-intensive task, node {best_edge_id}, core {best_core_id})")
        elif t_l_min <= t_edge_min and t_l_min <= t_cloud:
            # Device is fastest
            task.execution_tier = ExecutionTier.DEVICE
            logger.info(f"Task {task.id} assigned to DEVICE (core {best_device_core})")
        elif t_edge_min <= t_l_min and t_edge_min <= t_cloud:
            # Edge is fastest
            task.execution_tier = ExecutionTier.EDGE
            task.edge_assignment = EdgeAssignment(edge_id=best_edge_id, core_id=best_core_id)
            logger.info(f"Task {task.id} assigned to EDGE (node {best_edge_id}, core {best_core_id})")
        else:
            # Cloud is fastest
            task.execution_tier = ExecutionTier.CLOUD
            logger.info(f"Task {task.id} assigned to CLOUD")

        # Initialize timing parameters for the assigned execution tier
        if task.execution_tier == ExecutionTier.DEVICE:
            # Initialize device timing parameters
            task.device_core = best_device_core
            task.FT_l = t_l_min  # Initial estimate
            # Clear other execution options
            task.edge_assignment = None
            task.FT_edge = {}
            task.FT_edge_receive = {}
            task.RT_ws = task.FT_ws = task.RT_c = task.FT_c = task.RT_wr = task.FT_wr = 0

        elif task.execution_tier == ExecutionTier.EDGE:
            # Initialize edge timing parameters
            task.edge_assignment = EdgeAssignment(edge_id=best_edge_id, core_id=best_core_id)
            # Set initial finish time estimates
            if not hasattr(task, 'FT_edge') or task.FT_edge is None:
                task.FT_edge = {}
            task.FT_edge[best_edge_id] = t_edge_min  # Initial estimate
            # Also store in 0-based index for compatibility
            task.FT_edge[best_edge_id - 1] = t_edge_min

            if not hasattr(task, 'FT_edge_receive') or task.FT_edge_receive is None:
                task.FT_edge_receive = {}
            # Estimate result receive time (add simple transfer delay)
            transfer_delay = 1.0  # Simple default
            task.FT_edge_receive[best_edge_id] = t_edge_min + transfer_delay
            task.FT_edge_receive[best_edge_id - 1] = t_edge_min + transfer_delay
            # Clear other execution options
            task.device_core = -1
            task.FT_l = 0
            task.RT_ws = task.FT_ws = task.RT_c = task.FT_c = task.RT_wr = task.FT_wr = 0

        else:  # Cloud execution
            # Initialize cloud timing parameters
            if not hasattr(task, 'cloud_execution_times') or not task.cloud_execution_times:
                task.cloud_execution_times = [3, 1, 1]  # Default values

            # Initialize cloud timing parameters with actual timing values
            task.RT_ws = 0
            task.FT_ws = task.RT_ws + task.cloud_execution_times[0]
            task.RT_c = task.FT_ws
            task.FT_c = task.RT_c + task.cloud_execution_times[1]
            task.RT_wr = task.FT_c
            task.FT_wr = task.RT_wr + task.cloud_execution_times[2]
            # Clear other execution options
            task.device_core = -1
            task.edge_assignment = None
            task.FT_l = 0
            task.FT_edge = {}
            task.FT_edge_receive = {}


def task_prioritizing(tasks, download_rates, upload_rates):
    """
    Extended implementation of "Task Prioritizing" phase for three-tier architecture.
    Calculates priority levels for each task to determine scheduling order.
    """
    num_tasks = len(tasks)
    w = [0] * num_tasks

    # Step 1: Calculate computation costs (wi) for each task
    for i, task in enumerate(tasks):
        if i >= num_tasks:
            continue  # Skip if index out of range

        if task.execution_tier == ExecutionTier.CLOUD:
            # For cloud tasks:
            w[i] = (task.cloud_execution_times[0] +  # Ti^s
                    task.cloud_execution_times[1] +  # Ti^c
                    task.cloud_execution_times[2])  # Ti^r

        elif task.execution_tier == ExecutionTier.EDGE:
            # For edge tasks:
            if task.edge_assignment:
                edge_id = task.edge_assignment.edge_id
                core_id = task.edge_assignment.core_id

                # Get execution time for assigned core
                key = (edge_id, core_id)
                if key in task.edge_execution_times:
                    edge_time = task.edge_execution_times[key]

                    # Add transfer times
                    d2e_key = f'device_to_edge{edge_id}'
                    e2d_key = f'edge{edge_id}_to_device'

                    # Data sizes with defaults
                    d2e_data_size = task.data_sizes.get(d2e_key, 3.0)
                    e2d_data_size = task.data_sizes.get(e2d_key, 1.0)

                    # Transfer rates with defaults
                    d2e_rate = upload_rates.get(d2e_key, 1.5)
                    e2d_rate = download_rates.get(e2d_key, 2.0)

                    # Calculate transfer times
                    d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate
                    e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate

                    # Total edge execution time
                    w[i] = d2e_time + edge_time + e2d_time
                else:
                    # Fallback if specific edge time not available
                    # Calculate average of all edge times for this task
                    all_edge_times = [v for k, v in task.edge_execution_times.items()]
                    if all_edge_times:
                        avg_edge_time = sum(all_edge_times) / len(all_edge_times)
                        # Assume average transfer times
                        w[i] = 3.0 + avg_edge_time + 1.0  # Reasonable defaults
                    else:
                        logger.warning(f"No edge execution times for task {task.id}")
                        w[i] = float('inf')
            else:
                logger.warning(f"Edge task {task.id} has no edge assignment")
                w[i] = float('inf')

        else:  # ExecutionTier.DEVICE
            # For local tasks:
            if task.local_execution_times:
                # Average computation time across all local cores
                w[i] = sum(task.local_execution_times) / len(task.local_execution_times)
            else:
                logger.warning(f"No local execution times for task {task.id}")
                w[i] = float('inf')

    # Cache for memoization of priority calculations
    computed_priority_scores = {}

    def calculate_priority(task_idx):
        """
        Recursive implementation of priority calculation.
        Extended for three-tier architecture but follows the same principle.
        """
        # Ensure valid task index
        if task_idx < 0 or task_idx >= num_tasks:
            return 0

        task = tasks[task_idx]

        # Memoization check
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]

        # Base case: Exit tasks
        if not task.succ_tasks:
            score = w[task_idx]
            computed_priority_scores[task.id] = score
            return score

        # Recursive case: Non-exit tasks
        max_successor_priority = 0
        for succ_task in task.succ_tasks:
            # Find index of successor task
            succ_idx = next((i for i, t in enumerate(tasks) if t.id == succ_task.id), -1)
            if succ_idx >= 0:
                succ_priority = calculate_priority(succ_idx)
                max_successor_priority = max(max_successor_priority, succ_priority)

        task_priority = w[task_idx] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all tasks using recursive algorithm
    for i, task in enumerate(tasks):
        if i < num_tasks:
            calculate_priority(i)

    # Update priority scores in task objects
    for task in tasks:
        if task.id in computed_priority_scores:
            task.priority_score = computed_priority_scores[task.id]
        else:
            logger.warning(f"Priority score not calculated for task {task.id}")
            task.priority_score = 0


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


def generate_realistic_network_conditions(time_of_day=None):
    """
    Generate realistic network conditions based on time-of-day congestion patterns.

    Parameters:
        time_of_day: Current hour (0-23), or None to use system time

    Returns:
        Dictionary of network conditions for all connection types
    """
    # Use current time if not specified
    if time_of_day is None:
        time_of_day = datetime.datetime.now().hour

    # Base transfer rates in Mbps
    base_rates = {
        'device_to_edge': 10.0,  # Local high-speed connection
        'edge_to_device': 12.0,  # Download usually faster than upload
        'device_to_cloud': 5.0,  # WAN connection
        'cloud_to_device': 6.0,  # Cloud download
        'edge_to_cloud': 50.0,  # Edge nodes have better backhaul
        'cloud_to_edge': 60.0,  # Cloud to edge backhaul
        'edge_to_edge': 30.0  # Direct edge connection
    }

    # Time-of-day factors (network congestion patterns)
    # Busiest at 9-11am and 7-9pm
    tod_factor = 1.0
    if 9 <= time_of_day <= 11 or 19 <= time_of_day <= 21:
        tod_factor = 0.7  # 30% degradation during peak hours
    elif 0 <= time_of_day <= 5:
        tod_factor = 1.3  # 30% improvement during night

    # Random fluctuation (±15%)
    random_factor = random.uniform(0.85, 1.15)

    # Apply time-of-day factor to base rates
    network_conditions = {}
    for link_type, base_rate in base_rates.items():
        effective_rate = base_rate * tod_factor * random_factor
        network_conditions[link_type] = effective_rate

    # Add jitter and latency
    network_conditions['latencies'] = {
        'device_to_edge': random.uniform(5, 15),  # 5-15ms
        'device_to_cloud': random.uniform(50, 150),  # 50-150ms
        'edge_to_cloud': random.uniform(20, 60)  # 20-60ms
    }

    # Add random packet loss rates
    network_conditions['packet_loss'] = {
        'device_to_edge': random.uniform(0.001, 0.01),  # 0.1-1%
        'device_to_cloud': random.uniform(0.005, 0.02),  # 0.5-2%
        'edge_to_cloud': random.uniform(0.001, 0.005)  # 0.1-0.5%
    }

    # Simulate occasional network issues
    if random.random() < 0.05:  # 5% chance of a network issue
        problem_link = random.choice(list(base_rates.keys()))
        # 70% degradation on a random link
        network_conditions[problem_link] *= 0.3

    return network_conditions


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


def generate_migration_cache_key(tasks: List[Any], task_id: int, source_unit: ExecutionUnit, target_unit: ExecutionUnit) -> Tuple:
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


####################################
# SECTION 11: MAIN PROGRAM EXECUTION
####################################

if __name__ == "__main__":
    # Main program execution code
    pass
