from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Dictionary storing execution times for tasks on different cores on the mobile device
# Key: task ID (1-20)
# Value: List of execution times [core1_time, core2_time, core3_time]
# This implements T_i^l_k from Section II.B of the paper
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

# Edge node execution times for tasks
# Key: tuple (task_id, edge_node_id, core_id)
# Value: execution time
# This implements execution time on edge nodes for the extended three-tier model
edge_execution_times = {
    # Format: (task_id, edge_node_id, core_id): execution_time
    # Example values for M=2 edge nodes, each with 2 cores
    (1, 1, 1): 8, (1, 1, 2): 6, (1, 2, 1): 7, (1, 2, 2): 5,
    (2, 1, 1): 7, (2, 1, 2): 5, (2, 2, 1): 6, (2, 2, 2): 4,
    # Add more entries for all tasks and edge nodes
}

# Original cloud execution parameters
# [T_send, T_cloud, T_receive]
cloud_execution_times = [3, 1, 1]

# Number of edge nodes in the system
M = 2  # M edge nodes {E_1, E_2, ..., E_M}

# Edge node power consumption parameters
# Format: (edge_node_id, core_id): (alpha, gamma, frequency)
edge_power_params = {
    (1, 1): (1.2, 2.5, 1.8),  # Edge 1, Core 1: (alpha, gamma, frequency)
    (1, 2): (1.0, 2.3, 2.1),  # Edge 1, Core 2
    (2, 1): (1.3, 2.6, 1.7),  # Edge 2, Core 1
    (2, 2): (1.1, 2.4, 2.0),  # Edge 2, Core 2
}

# Communication rates and data sizes for the extended three-tier model
# Upload rates between tiers (in data units per time unit)
upload_rates = {
    'device_to_edge1': 2.0,  # R^s_{d→e1}
    'device_to_edge2': 1.8,  # R^s_{d→e2}
    'device_to_cloud': 1.5,  # R^s (original rate)
    'edge1_to_edge2': 3.0,  # R^s_{e1→e2}
    'edge2_to_edge1': 3.0,  # R^s_{e2→e1}
    'edge1_to_cloud': 4.0,  # R^s_{e1→c}
    'edge2_to_cloud': 3.8  # R^s_{e2→c}
}

# Download rates between tiers
download_rates = {
    'cloud_to_device': 2.0,  # R^r (original rate)
    'cloud_to_edge1': 4.5,  # R^r_{c→e1}
    'cloud_to_edge2': 4.2,  # R^r_{c→e2}
    'edge1_to_device': 3.0,  # R^r_{e1→d}
    'edge2_to_device': 2.8  # R^r_{e2→d}
}

# Power consumption for RF communication
rf_power = {
    'device_to_edge1': 0.4,  # P^s_{d→e1}
    'device_to_edge2': 0.45,  # P^s_{d→e2}
    'device_to_cloud': 0.5,  # P^s (original)
    'edge1_to_edge2': 0.3,  # P^s_{e1→e2}
    'edge2_to_edge1': 0.3,  # P^s_{e2→e1}
    'edge1_to_cloud': 0.4,  # P^s_{e1→c}
    'edge2_to_cloud': 0.42,  # P^s_{e2→c}
    'edge1_to_device': 0.3,  # P^s_{e1→d}
    'edge2_to_device': 0.35  # P^s_{e2→d}
}

# Data size configurations for tasks
# Key: task_id
# Value: dict of data sizes for different transfer paths
task_data_sizes = {
    # Example for task 1
    1: {
        'device_to_edge1': 2.0,  # data_i^(d→e1)
        'device_to_edge2': 2.0,  # data_i^(d→e2)
        'device_to_cloud': 3.0,  # data_i (original)
        'edge1_to_edge2': 1.5,  # data_i^(e1→e2)
        'edge2_to_edge1': 1.5,  # data_i^(e2→e1)
        'edge1_to_cloud': 2.0,  # data_i^(e1→c)
        'edge2_to_cloud': 2.0,  # data_i^(e2→c)
        'cloud_to_device': 1.0,  # data_i' (original)
        'edge1_to_device': 0.8,  # data_i^(e1→d)
        'edge2_to_device': 0.8  # data_i^(e2→d)
    },
    # Add more entries for all tasks
}


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


@dataclass
class ExecutionResult:
    """Stores the execution outcomes of a task"""
    start_time: float = 0.0  # When execution started
    finish_time: float = 0.0  # When execution completed (FT)
    energy: float = 0.0  # Energy consumed


@dataclass
class TaskMigrationState:
    """Tracks task migration decisions"""
    time: float  # T_total: Completion time after migration
    energy: float  # E_total: Energy consumption after migration
    efficiency: float  # Energy reduction per unit time
    task_index: int  # v_tar: Task selected for migration
    source_tier: ExecutionTier
    target_tier: ExecutionTier
    source_location: Optional[Tuple[int, int]] = None  # (edge_id, core_id) if applicable
    target_location: Optional[Tuple[int, int]] = None  # (edge_id, core_id) if applicable


class Task:
    def __init__(self, id, pred_tasks=None, succ_tasks=None):
        # Basic task graph structure
        self.id = id  # Task identifier v_i in DAG G=(V,E)
        self.pred_tasks = pred_tasks or []  # pred(v_i): Immediate predecessors
        self.succ_tasks = succ_tasks or []  # succ(v_i): Immediate successors

        # ==== EXECUTION TIME PARAMETERS ====
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
        self.data_sizes = task_data_sizes.get(id, {})

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
        key = (edge_id, core_id)
        return self.edge_execution_times.get(key)

    def calculate_data_transfer_time(self, source_tier, target_tier,
                                     upload_rates_dict, download_rates_dict,
                                     source_location=None, target_location=None):
        """
        Calculate time to transfer data between locations

        Parameters:
        - source_tier: ExecutionTier of source
        - target_tier: ExecutionTier of target
        - upload_rates_dict: Dictionary of upload rates between tiers
        - download_rates_dict: Dictionary of download rates between tiers
        - source_location: For edge tier, tuple of (edge_id, core_id)
        - target_location: For edge tier, tuple of (edge_id, core_id)

        Returns:
        - Transfer time in time units
        """
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

    def calculate_ready_time_edge(self, edge_id, upload_rates_dict, download_rates_dict):
        """
        Calculate ready time (RT_i^e,m) for execution on edge node E_m
        Based on the ready time formula from Section III extension:

        RT_i^e,m = max_{v_p ∈ pred(v_i)} (FT_p(X_p) + Δ_p→m)

        where Δ_p→m is the data transfer time from predecessor's location to edge m
        """
        if not self.pred_tasks:
            return 0  # Entry task, ready at time 0

        max_ready_time = 0

        for pred_task in self.pred_tasks:
            # Ensure predecessor has been scheduled
            if pred_task.is_scheduled == SchedulingState.UNSCHEDULED:
                logger.warning(f"Predecessor task {pred_task.id} of task {self.id} not yet scheduled")
                return float('inf')  # Not ready until predecessor is scheduled

            # Determine where predecessor executed (device, edge, or cloud)
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor on device local core
                pred_finish_time = pred_task.FT_l
                if pred_finish_time <= 0:
                    logger.warning(f"Invalid finish time for predecessor {pred_task.id} on device")
                    return float('inf')

                # Time to transfer data from device to this edge
                transfer_time = pred_task.calculate_data_transfer_time(
                    ExecutionTier.DEVICE, ExecutionTier.EDGE,
                    upload_rates_dict, download_rates_dict,
                    target_location=(edge_id, 0)  # core_id doesn't matter for transfer
                )

            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor in cloud
                pred_finish_time = pred_task.FT_c
                if pred_finish_time <= 0:
                    logger.warning(f"Invalid finish time for predecessor {pred_task.id} on cloud")
                    return float('inf')

                # Time to transfer data from cloud to this edge
                transfer_time = pred_task.calculate_data_transfer_time(
                    ExecutionTier.CLOUD, ExecutionTier.EDGE,
                    upload_rates_dict, download_rates_dict,
                    target_location=(edge_id, 0)
                )

            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor on some edge node
                if not pred_task.edge_assignment:
                    logger.warning(f"Edge assignment missing for predecessor {pred_task.id}")
                    return float('inf')

                pred_edge_id = pred_task.edge_assignment.edge_id
                # Ensure the finish time for this edge is recorded
                if pred_edge_id not in pred_task.FT_edge:
                    logger.warning(f"Missing finish time for predecessor {pred_task.id} on edge {pred_edge_id}")
                    return float('inf')

                pred_finish_time = pred_task.FT_edge[pred_edge_id]

                if pred_edge_id == edge_id:
                    # Same edge node, no transfer needed
                    transfer_time = 0
                else:
                    # Different edge node, need edge-to-edge transfer
                    transfer_time = pred_task.calculate_data_transfer_time(
                        ExecutionTier.EDGE, ExecutionTier.EDGE,
                        upload_rates_dict, download_rates_dict,
                        source_location=(pred_edge_id, 0),
                        target_location=(edge_id, 0)
                    )

            else:
                logger.error(f"Unknown execution tier for predecessor {pred_task.id}")
                return float('inf')

            # Update max ready time
            current_ready_time = pred_finish_time + transfer_time
            max_ready_time = max(max_ready_time, current_ready_time)

        return max_ready_time

    def calculate_ready_time_local(self):
        """
        Calculate ready time for local execution (RT_i^l)
        Based on equation (3) from Section II.C
        """
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

    def calculate_ready_time_cloud_upload(self):
        """
        Calculate ready time for cloud upload (RT_i^ws)
        Based on equation (4) from Section II.C and extended for edge-to-cloud transfers
        """
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
        """
        Calculate the overall finish time for this task across all execution units
        """
        finish_times = []

        # Add device finish time if applicable
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

    def validate_dependencies(self):
        """
        Validate that all dependency constraints are satisfied for this task
        """
        if not self.pred_tasks:
            return True  # Entry task, no dependencies to check

        if self.is_scheduled == SchedulingState.UNSCHEDULED:
            return False  # Task not scheduled yet

        # Check if executing on device
        if self.execution_tier == ExecutionTier.DEVICE:
            ready_time = self.calculate_ready_time_local()
            if ready_time == float('inf'):
                return False
            # Check if start time respects ready time
            for i, ft in enumerate(self.local_execution_times):
                if i == self.device_core:
                    # This is the core we're using
                    start_time = self.FT_l - ft
                    if start_time < ready_time:
                        logger.error(f"Task {self.id} starts at {start_time} but ready time is {ready_time}")
                        return False

        # Check if executing on cloud
        elif self.execution_tier == ExecutionTier.CLOUD:
            ready_time = self.calculate_ready_time_cloud_upload()
            if ready_time == float('inf'):
                return False
            # Check if upload start time respects ready time
            upload_time = self.cloud_execution_times[0]
            start_time = self.FT_ws - upload_time
            if start_time < ready_time:
                logger.error(f"Task {self.id} cloud upload starts at {start_time} but ready time is {ready_time}")
                return False

        # Check if executing on edge
        elif self.execution_tier == ExecutionTier.EDGE:
            if not self.edge_assignment:
                return False

            edge_id = self.edge_assignment.edge_id
            core_id = self.edge_assignment.core_id

            ready_time = self.calculate_ready_time_edge(edge_id, upload_rates, download_rates)
            if ready_time == float('inf'):
                return False

            # Check if edge execution start time respects ready time
            if edge_id not in self.FT_edge:
                return False

            exec_time = self.get_edge_execution_time(edge_id, core_id)
            if exec_time is None:
                return False

            start_time = self.FT_edge[edge_id] - exec_time
            if start_time < ready_time:
                logger.error(f"Task {self.id} edge execution starts at {start_time} but ready time is {ready_time}")
                return False

        return True  # All dependency constraints satisfied


# Standalone utility functions for calculating time and energy metrics
def total_time(tasks):
    """
    Extended implementation of total completion time calculation for three-tier model
    Extends equation (10) to include edge execution finish times

    Parameters:
    - tasks: List of Task objects

    Returns:
    - Total completion time (maximum finish time across all exit tasks)
    """
    if not tasks:
        return 0

    exit_tasks = [task for task in tasks if not task.succ_tasks]
    if not exit_tasks:
        # If no explicit exit tasks, use all tasks
        exit_tasks = tasks

    max_completion_time = 0
    for task in exit_tasks:
        # Calculate maximum finish time for this task across all execution units
        task_finish_times = []

        # Device execution
        if task.FT_l > 0:
            task_finish_times.append(task.FT_l)

        # Cloud execution
        if task.FT_wr > 0:
            task_finish_times.append(task.FT_wr)

        # Edge execution
        if task.FT_edge_receive and len(task.FT_edge_receive) > 0:
            for edge_id, time in task.FT_edge_receive.items():
                if time > 0:
                    task_finish_times.append(time)

        if task_finish_times:
            task_max_finish = max(task_finish_times)
            max_completion_time = max(max_completion_time, task_max_finish)
        else:
            logger.warning(f"Exit task {task.id} has no valid finish times")

    return max_completion_time


def calculate_energy_consumption(task, core_powers, rf_power, upload_rates, download_rates):
    """
    Calculate energy consumption for a single task in the three-tier model
    Extended from equations (7) and (8) to include edge execution and transfers

    Parameters:
    - task: The Task object
    - core_powers: Dictionary mapping core IDs to their power consumption
    - rf_power: Dictionary of RF power consumption rates for different transfers
    - upload_rates: Dictionary of upload rates between tiers
    - download_rates: Dictionary of download rates between tiers

    Returns:
    - Energy consumption for this task
    """
    # Device-tier execution (original equation 7)
    if task.execution_tier == ExecutionTier.DEVICE:
        if task.device_core < 0 or task.device_core >= len(task.local_execution_times):
            logger.warning(f"Task {task.id} has invalid device core {task.device_core}")
            return 0  # Invalid core assignment
        # E_i^l = P_k × T_i^l,k
        return core_powers.get(task.device_core, 1.0) * task.local_execution_times[task.device_core]

    # Calculate energy based on the execution path
    total_energy = 0

    # Empty execution path check
    if not task.execution_path or len(task.execution_path) < 2:
        # If no execution path is tracked, use current assignment
        if task.execution_tier == ExecutionTier.CLOUD:
            # E_i^c = P^s × T_i^s (device to cloud, original equation 8)
            return rf_power.get('device_to_cloud', 0.5) * task.cloud_execution_times[0]
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            # Device to edge transfer
            edge_id = task.edge_assignment.edge_id
            key = f'device_to_edge{edge_id}'
            # E = P^s × (data_i^(d→e) / R^s_(d→e))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            return rf_power.get(key, 0.4) * transfer_time
        return 0

    # Track energy consumption through the execution path
    for i, (tier, location) in enumerate(task.execution_path[:-1]):
        next_tier, next_location = task.execution_path[i + 1]

        # Device to Edge transfer
        if tier == ExecutionTier.DEVICE and next_tier == ExecutionTier.EDGE:
            edge_id, _ = next_location
            key = f'device_to_edge{edge_id}'
            # E = P^s × (data_i^(d→e) / R^s_(d→e))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power.get(key, 0.4) * transfer_time

        # Device to Cloud transfer (original equation 8)
        elif tier == ExecutionTier.DEVICE and next_tier == ExecutionTier.CLOUD:
            # E_i^c = P^s × T_i^s
            total_energy += rf_power.get('device_to_cloud', 0.5) * task.cloud_execution_times[0]

        # Edge to Edge migration
        elif tier == ExecutionTier.EDGE and next_tier == ExecutionTier.EDGE:
            source_edge_id, _ = location
            target_edge_id, _ = next_location
            key = f'edge{source_edge_id}_to_edge{target_edge_id}'
            # E = P^s × (data_i^(e→e') / R^s_(e→e'))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power.get(key, 0.3) * transfer_time

        # Edge to Cloud transfer
        elif tier == ExecutionTier.EDGE and next_tier == ExecutionTier.CLOUD:
            edge_id, _ = location
            key = f'edge{edge_id}_to_cloud'
            # E = P^s × (data_i^(e→c) / R^s_(e→c))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power.get(key, 0.4) * transfer_time

        # Edge to Device return transfer
        elif tier == ExecutionTier.EDGE and next_tier == ExecutionTier.DEVICE:
            edge_id, _ = location
            key = f'edge{edge_id}_to_device'
            # E = P^s × (data_i^(e→d) / R^r_(e→d))
            data_size = task.data_sizes.get(key, 0)
            rate = download_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power.get(key, 0.3) * transfer_time

    return total_energy


def total_energy(tasks, core_powers, rf_power, upload_rates, download_rates):
    """
    Calculate total energy consumption across all tasks in the three-tier model
    Extends equation (9) to account for all possible execution paths

    Parameters:
    - tasks: List of Task objects
    - core_powers: Dictionary mapping core IDs to their power consumption
    - rf_power: Dictionary of RF power consumption rates for different transfers
    - upload_rates: Dictionary of upload rates between tiers
    - download_rates: Dictionary of download rates between tiers

    Returns:
    - Total energy consumption across all tasks
    """
    return sum(
        calculate_energy_consumption(task, core_powers, rf_power, upload_rates, download_rates)
        for task in tasks
    )


def primary_assignment(tasks, edge_nodes=None):
    """
    Extended implementation of "Primary Assignment" phase for three-tier architecture.
    Determines which tasks should be executed on device, edge, or cloud.

    Parameters:
    - tasks: List of Task objects
    - edge_nodes: Number of edge nodes available (default: 2)
    """
    if edge_nodes is None:
        edge_nodes = M  # Use global edge nodes count from earlier code

    for task in tasks:
        # Calculate T_i^l_min (minimum local execution time)
        if not task.local_execution_times:
            logger.warning(f"Task {task.id} has no local execution times")
            t_l_min = float('inf')
        else:
            t_l_min = min(task.local_execution_times)

        # Calculate minimum edge execution time across all edge nodes
        # For each edge node, find the fastest core
        edge_times = []
        for edge_id in range(1, edge_nodes + 1):
            edge_cores_times = []
            # Check all cores in this edge node
            for core_id in range(1, 3):  # Assuming 2 cores per edge node
                key = (edge_id, core_id)
                if key in task.edge_execution_times:
                    edge_cores_times.append(task.edge_execution_times[key])

            if edge_cores_times:
                # Calculate total edge execution time including transfers
                min_edge_core_time = min(edge_cores_times)

                # Add transfer times: device→edge + edge→device
                d2e_key = f'device_to_edge{edge_id}'
                e2d_key = f'edge{edge_id}_to_device'

                # Data sizes for transfers
                d2e_data_size = task.data_sizes.get(d2e_key, 3.0)  # Default similar to cloud
                e2d_data_size = task.data_sizes.get(e2d_key, 1.0)  # Default similar to cloud

                # Transfer rates
                d2e_rate = upload_rates.get(d2e_key, 1.5)  # Default similar to cloud
                e2d_rate = download_rates.get(e2d_key, 2.0)  # Default similar to cloud

                # Calculate transfer times
                d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate
                e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate

                # Total edge execution time = upload + processing + download
                total_edge_time = d2e_time + min_edge_core_time + e2d_time
                edge_times.append((total_edge_time, edge_id, edge_cores_times.index(min_edge_core_time) + 1))

        # Get minimum edge execution time if any
        t_edge_min = float('inf')
        best_edge_id = -1
        best_core_id = -1

        if edge_times:
            t_edge_min, best_edge_id, best_core_id = min(edge_times, key=lambda x: x[0])

        # Calculate T_i^re (cloud execution time)
        t_re = (task.cloud_execution_times[0] +  # T_i^s (send)
                task.cloud_execution_times[1] +  # T_i^c (cloud)
                task.cloud_execution_times[2])  # T_i^r (receive)

        # Determine optimal execution tier
        if t_l_min <= t_edge_min and t_l_min <= t_re:
            # Local execution is fastest
            task.execution_tier = ExecutionTier.DEVICE
            # Will assign specific core later in execution unit selection
        elif t_edge_min <= t_l_min and t_edge_min <= t_re:
            # Edge execution is fastest
            task.execution_tier = ExecutionTier.EDGE
            task.edge_assignment = EdgeAssignment(edge_id=best_edge_id, core_id=best_core_id)
        else:
            # Cloud execution is fastest
            task.execution_tier = ExecutionTier.CLOUD


def task_prioritizing(tasks):
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


def validate_task_dependencies(tasks):
    """
    Validate that task dependency constraints are properly maintained
    across the three-tier architecture.

    Parameters:
    - tasks: List of Task objects

    Returns:
    - Tuple (is_valid, violations)
    """
    violations = []

    for task in tasks:
        # Skip unscheduled tasks
        if task.is_scheduled == SchedulingState.UNSCHEDULED:
            continue

        for pred_task in task.pred_tasks:
            # Skip if predecessor is not scheduled
            if pred_task.is_scheduled == SchedulingState.UNSCHEDULED:
                violations.append({
                    'type': 'Unscheduled Predecessor',
                    'task': task.id,
                    'predecessor': pred_task.id,
                    'detail': f"Task {task.id} depends on unscheduled task {pred_task.id}"
                })
                continue

            # Check device execution
            if task.execution_tier == ExecutionTier.DEVICE:
                # Calculate when this task starts on the device
                if task.device_core < 0 or task.FT_l <= 0:
                    violations.append({
                        'type': 'Invalid Device Execution',
                        'task': task.id,
                        'detail': f"Task {task.id} has invalid device execution parameters"
                    })
                    continue

                task_start_time = task.FT_l - task.local_execution_times[task.device_core]

                # Check dependency based on predecessor's execution tier
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Device predecessor must finish before task starts
                    if pred_task.FT_l > task_start_time:
                        violations.append({
                            'type': 'Device-Device Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts at {task_start_time} before predecessor {pred_task.id} finishes at {pred_task.FT_l}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Cloud predecessor must return results before task starts
                    if pred_task.FT_wr > task_start_time:
                        violations.append({
                            'type': 'Cloud-Device Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts at {task_start_time} before cloud results from {pred_task.id} arrive at {pred_task.FT_wr}"
                        })

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # Edge predecessor must return results to device before task starts
                    if not pred_task.edge_assignment:
                        violations.append({
                            'type': 'Missing Edge Assignment',
                            'task': pred_task.id,
                            'detail': f"Edge task {pred_task.id} has no edge assignment"
                        })
                        continue

                    edge_id = pred_task.edge_assignment.edge_id
                    if edge_id not in pred_task.FT_edge_receive:
                        violations.append({
                            'type': 'Missing Edge Receive Time',
                            'task': pred_task.id,
                            'edge': edge_id,
                            'detail': f"Edge task {pred_task.id} has no receive time for edge {edge_id}"
                        })
                        continue

                    receive_time = pred_task.FT_edge_receive[edge_id]
                    if receive_time > task_start_time:
                        violations.append({
                            'type': 'Edge-Device Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts at {task_start_time} before edge results from {pred_task.id} arrive at {receive_time}"
                        })

            # Check cloud execution
            elif task.execution_tier == ExecutionTier.CLOUD:
                # Calculate when this task starts uploading to cloud
                if task.FT_ws <= 0:
                    violations.append({
                        'type': 'Invalid Cloud Execution',
                        'task': task.id,
                        'detail': f"Task {task.id} has invalid cloud upload parameters"
                    })
                    continue

                upload_start_time = task.FT_ws - task.cloud_execution_times[0]

                # Check dependency based on predecessor's execution tier
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Device predecessor must finish before upload starts
                    if pred_task.FT_l > upload_start_time:
                        violations.append({
                            'type': 'Device-Cloud Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts uploading at {upload_start_time} before predecessor {pred_task.id} finishes at {pred_task.FT_l}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # For cloud-to-cloud, only need to wait for upload to complete
                    if pred_task.FT_ws > upload_start_time:
                        violations.append({
                            'type': 'Cloud-Cloud Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts uploading at {upload_start_time} before predecessor {pred_task.id} finishes uploading at {pred_task.FT_ws}"
                        })

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # Need to check edge-to-cloud transfer if applicable
                    if not pred_task.edge_assignment:
                        violations.append({
                            'type': 'Missing Edge Assignment',
                            'task': pred_task.id,
                            'detail': f"Edge task {pred_task.id} has no edge assignment"
                        })
                        continue

                    # Check if data was sent directly to cloud from edge
                    cloud_key = ('edge', 'cloud')
                    if cloud_key in pred_task.FT_edge_send:
                        cloud_send_time = pred_task.FT_edge_send[cloud_key]
                        if cloud_send_time > upload_start_time:
                            violations.append({
                                'type': 'Edge-Cloud Direct Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'detail': f"Task {task.id} starts uploading at {upload_start_time} before edge {pred_task.id} finishes sending to cloud at {cloud_send_time}"
                            })
                    else:
                        # Otherwise need to wait for data to return to device
                        edge_id = pred_task.edge_assignment.edge_id
                        if edge_id not in pred_task.FT_edge_receive:
                            violations.append({
                                'type': 'Missing Edge Receive Time',
                                'task': pred_task.id,
                                'edge': edge_id,
                                'detail': f"Edge task {pred_task.id} has no receive time for edge {edge_id}"
                            })
                            continue

                        receive_time = pred_task.FT_edge_receive[edge_id]
                        if receive_time > upload_start_time:
                            violations.append({
                                'type': 'Edge-Device-Cloud Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'detail': f"Task {task.id} starts uploading at {upload_start_time} before edge results from {pred_task.id} arrive at device at {receive_time}"
                            })

            # Check edge execution
            elif task.execution_tier == ExecutionTier.EDGE:
                # Verify edge assignment
                if not task.edge_assignment:
                    violations.append({
                        'type': 'Missing Edge Assignment',
                        'task': task.id,
                        'detail': f"Edge task {task.id} has no edge assignment"
                    })
                    continue

                edge_id = task.edge_assignment.edge_id
                core_id = task.edge_assignment.core_id

                # Calculate when this task starts on the edge
                if edge_id not in task.FT_edge:
                    violations.append({
                        'type': 'Invalid Edge Execution',
                        'task': task.id,
                        'edge': edge_id,
                        'detail': f"Task {task.id} has no finish time for edge {edge_id}"
                    })
                    continue

                # Get execution time on this edge core
                exec_time = task.get_edge_execution_time(edge_id, core_id)
                if exec_time is None:
                    violations.append({
                        'type': 'Missing Edge Execution Time',
                        'task': task.id,
                        'edge': edge_id,
                        'core': core_id,
                        'detail': f"Task {task.id} has no execution time for edge {edge_id}, core {core_id}"
                    })
                    continue

                edge_start_time = task.FT_edge[edge_id] - exec_time

                # Check dependency based on predecessor's execution tier
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Need to account for device-to-edge transfer time
                    transfer_time = pred_task.calculate_data_transfer_time(
                        ExecutionTier.DEVICE, ExecutionTier.EDGE,
                        upload_rates, download_rates,
                        target_location=(edge_id, 0)
                    )

                    # Device predecessor must finish and data must transfer before edge task starts
                    ready_time = pred_task.FT_l + transfer_time
                    if ready_time > edge_start_time:
                        violations.append({
                            'type': 'Device-Edge Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts on edge at {edge_start_time} before data from {pred_task.id} arrives at {ready_time}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Need to account for cloud-to-edge transfer
                    transfer_time = pred_task.calculate_data_transfer_time(
                        ExecutionTier.CLOUD, ExecutionTier.EDGE,
                        upload_rates, download_rates,
                        target_location=(edge_id, 0)
                    )

                    # Cloud computation must finish and data must transfer before edge task starts
                    ready_time = pred_task.FT_c + transfer_time
                    if ready_time > edge_start_time:
                        violations.append({
                            'type': 'Cloud-Edge Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts on edge at {edge_start_time} before data from cloud {pred_task.id} arrives at {ready_time}"
                        })

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # Check edge-to-edge transfer if needed
                    if not pred_task.edge_assignment:
                        violations.append({
                            'type': 'Missing Edge Assignment',
                            'task': pred_task.id,
                            'detail': f"Edge task {pred_task.id} has no edge assignment"
                        })
                        continue

                    pred_edge_id = pred_task.edge_assignment.edge_id

                    if pred_edge_id not in pred_task.FT_edge:
                        violations.append({
                            'type': 'Missing Edge Finish Time',
                            'task': pred_task.id,
                            'edge': pred_edge_id,
                            'detail': f"Edge task {pred_task.id} has no finish time for edge {pred_edge_id}"
                        })
                        continue

                    pred_finish_time = pred_task.FT_edge[pred_edge_id]

                    if pred_edge_id == edge_id:
                        # Same edge node, no transfer needed
                        if pred_finish_time > edge_start_time:
                            violations.append({
                                'type': 'Same-Edge Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'edge': edge_id,
                                'detail': f"Task {task.id} starts on edge {edge_id} at {edge_start_time} before predecessor {pred_task.id} finishes at {pred_finish_time}"
                            })
                    else:
                        # Different edge nodes, need to account for edge-to-edge transfer
                        transfer_time = pred_task.calculate_data_transfer_time(
                            ExecutionTier.EDGE, ExecutionTier.EDGE,
                            upload_rates, download_rates,
                            source_location=(pred_edge_id, 0),
                            target_location=(edge_id, 0)
                        )

                        ready_time = pred_finish_time + transfer_time
                        if ready_time > edge_start_time:
                            violations.append({
                                'type': 'Edge-Edge Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'source_edge': pred_edge_id,
                                'target_edge': edge_id,
                                'detail': f"Task {task.id} starts on edge {edge_id} at {edge_start_time} before data from edge {pred_edge_id} task {pred_task.id} arrives at {ready_time}"
                            })

    is_valid = len(violations) == 0
    return is_valid, violations


def print_validation_report(tasks):
    """Print detailed dependency validation report"""
    is_valid, violations = validate_task_dependencies(tasks)

    print("\nTask Dependency Validation Report")
    print("=" * 60)

    if is_valid:
        print("✓ All task dependency constraints are satisfied!")
        print(f"✓ Validated {len(tasks)} tasks with proper precedence relationships")
    else:
        print(f"✗ Found {len(violations)} dependency constraint violations:")

        # Group violations by type for easier analysis
        violation_types = {}
        for v in violations:
            v_type = v['type']
            if v_type not in violation_types:
                violation_types[v_type] = []
            violation_types[v_type].append(v)

        # Print summary by violation type
        print("\nViolation Summary:")
        for v_type, v_list in violation_types.items():
            print(f"- {v_type}: {len(v_list)} occurrences")

        # Print detailed violations
        print("\nDetailed Violations:")
        for i, v in enumerate(violations, 1):
            print(f"\n{i}. {v['type']}:")
            print(f"   {v['detail']}")

    return is_valid