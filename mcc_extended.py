from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any

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
        self.edge_execution_times = edge_execution_times

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
        return start_time + exec_time

    def get_edge_execution_time(self, edge_id, core_id):
        """Get the execution time for a specific edge node and core"""
        key = (self.id, edge_id, core_id)
        return self.edge_execution_times.get(key, 5)  # Default to 5 time units if not specified

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
            return self.cloud_execution_times[0]  # T_i^s

        # Cloud to Device (original case)
        elif source_tier == ExecutionTier.CLOUD and target_tier == ExecutionTier.DEVICE:
            return self.cloud_execution_times[2]  # T_i^r

        # Device to Edge
        elif source_tier == ExecutionTier.DEVICE and target_tier == ExecutionTier.EDGE:
            edge_id, _ = target_location
            key = f'device_to_edge{edge_id}'
            # T_i^(d→e) = data_i^(d→e) / R^s_(d→e)
            data_size = self.data_sizes.get(key, 0)
            rate = upload_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Edge to Device
        elif source_tier == ExecutionTier.EDGE and target_tier == ExecutionTier.DEVICE:
            edge_id, _ = source_location
            key = f'edge{edge_id}_to_device'
            # T_i^(e→d) = data_i^(e→d) / R^r_(e→d)
            data_size = self.data_sizes.get(key, 0)
            rate = download_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Edge to Cloud
        elif source_tier == ExecutionTier.EDGE and target_tier == ExecutionTier.CLOUD:
            edge_id, _ = source_location
            key = f'edge{edge_id}_to_cloud'
            # T_i^(e→c) = data_i^(e→c) / R^s_(e→c)
            data_size = self.data_sizes.get(key, 0)
            rate = upload_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Cloud to Edge
        elif source_tier == ExecutionTier.CLOUD and target_tier == ExecutionTier.EDGE:
            edge_id, _ = target_location
            key = f'cloud_to_edge{edge_id}'
            # T_i^(c→e) = data_i^(c→e) / R^r_(c→e)
            data_size = self.data_sizes.get(key, 0)
            rate = download_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Edge to Edge migration
        elif source_tier == ExecutionTier.EDGE and target_tier == ExecutionTier.EDGE:
            source_edge_id, _ = source_location
            target_edge_id, _ = target_location
            key = f'edge{source_edge_id}_to_edge{target_edge_id}'
            # T_i^(e→e') = data_i^(e→e') / R^s_(e→e')
            data_size = self.data_sizes.get(key, 0)
            rate = upload_rates_dict.get(key, 1.0)
            return 0 if rate == 0 else data_size / rate

        # Default case
        return 0

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
            # Determine where predecessor executed (device, edge, or cloud)
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor on device local core
                pred_finish_time = pred_task.FT_l
                # Time to transfer data from device to this edge
                transfer_time = pred_task.calculate_data_transfer_time(
                    ExecutionTier.DEVICE, ExecutionTier.EDGE,
                    upload_rates_dict, download_rates_dict,
                    target_location=(edge_id, 0)  # core_id doesn't matter for transfer
                )

            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor in cloud
                pred_finish_time = pred_task.FT_c
                # Time to transfer data from cloud to this edge
                transfer_time = pred_task.calculate_data_transfer_time(
                    ExecutionTier.CLOUD, ExecutionTier.EDGE,
                    upload_rates_dict, download_rates_dict,
                    target_location=(edge_id, 0)
                )

            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor on some edge node
                pred_edge_id = pred_task.edge_assignment.edge_id
                pred_finish_time = pred_task.FT_edge.get(pred_edge_id, 0)

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
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor executed locally
                ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor executed in cloud, need to wait for results
                ready_time = pred_task.FT_wr
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor executed on edge, need to wait for results
                pred_edge_id = pred_task.edge_assignment.edge_id
                ready_time = pred_task.FT_edge_receive.get(pred_edge_id, 0)

            max_ready_time = max(max_ready_time, ready_time)

        return max_ready_time

    def calculate_ready_time_cloud_upload(self):
        """
        Calculate ready time for cloud upload (RT_i^ws)
        Based on equation (4) from Section II.C
        """
        if not self.pred_tasks:
            return 0  # Entry task, ready at time 0

        max_ready_time = 0

        for pred_task in self.pred_tasks:
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor executed locally
                ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor uploaded to cloud
                ready_time = pred_task.FT_ws
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor executed on edge
                pred_edge_id = pred_task.edge_assignment.edge_id
                # If results were sent to device, use that finish time
                ready_time = pred_task.FT_edge_receive.get(pred_edge_id, 0)

            max_ready_time = max(max_ready_time, ready_time)

        return max_ready_time


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
        task_finish_times = [
            task.FT_l,  # Device local execution finish time
            task.FT_wr  # Cloud execution (results received) finish time
        ]

        # Add edge execution finish times if available
        if task.FT_edge_receive and len(task.FT_edge_receive) > 0:
            task_finish_times.append(max(task.FT_edge_receive.values()))

        max_completion_time = max(max_completion_time, max(task_finish_times))

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
            return 0  # Invalid core assignment
        # E_i^l = P_k × T_i^l,k
        return core_powers[task.device_core] * task.local_execution_times[task.device_core]

    # Calculate energy based on the execution path
    total_energy = 0

    # Empty execution path check
    if not task.execution_path or len(task.execution_path) < 2:
        # If no execution path is tracked, use current assignment
        if task.execution_tier == ExecutionTier.CLOUD:
            # E_i^c = P^s × T_i^s (device to cloud, original equation 8)
            return rf_power['device_to_cloud'] * task.cloud_execution_times[0]
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            # Device to edge transfer
            edge_id = task.edge_assignment.edge_id
            key = f'device_to_edge{edge_id}'
            # E = P^s × (data_i^(d→e) / R^s_(d→e))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            return rf_power[key] * transfer_time
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
            total_energy += rf_power[key] * transfer_time

        # Device to Cloud transfer (original equation 8)
        elif tier == ExecutionTier.DEVICE and next_tier == ExecutionTier.CLOUD:
            # E_i^c = P^s × T_i^s
            total_energy += rf_power['device_to_cloud'] * task.cloud_execution_times[0]

        # Edge to Edge migration
        elif tier == ExecutionTier.EDGE and next_tier == ExecutionTier.EDGE:
            source_edge_id, _ = location
            target_edge_id, _ = next_location
            key = f'edge{source_edge_id}_to_edge{target_edge_id}'
            # E = P^s × (data_i^(e→e') / R^s_(e→e'))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power[key] * transfer_time

        # Edge to Cloud transfer
        elif tier == ExecutionTier.EDGE and next_tier == ExecutionTier.CLOUD:
            edge_id, _ = location
            key = f'edge{edge_id}_to_cloud'
            # E = P^s × (data_i^(e→c) / R^s_(e→c))
            data_size = task.data_sizes.get(key, 0)
            rate = upload_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power[key] * transfer_time

        # Edge to Device return transfer (add this for completeness)
        elif tier == ExecutionTier.EDGE and next_tier == ExecutionTier.DEVICE:
            edge_id, _ = location
            key = f'edge{edge_id}_to_device'
            # E = P^s × (data_i^(e→d) / R^r_(e→d))
            data_size = task.data_sizes.get(key, 0)
            rate = download_rates.get(key, 1.0)
            transfer_time = 0 if rate == 0 else data_size / rate
            total_energy += rf_power[key] * transfer_time

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
        t_l_min = min(task.local_execution_times)

        # Calculate minimum edge execution time across all edge nodes
        # For each edge node, find the fastest core
        edge_times = []
        for edge_id in range(1, edge_nodes + 1):
            edge_cores_times = []
            for core_id in range(1, 3):  # Assuming 2 cores per edge node
                key = (task.id, edge_id, core_id)
                if key in edge_execution_times:
                    edge_cores_times.append(edge_execution_times[key])

            if edge_cores_times:
                # Calculate total edge execution time including transfers
                min_edge_core_time = min(edge_cores_times)

                # Add transfer times: device→edge + edge→device
                d2e_key = f'device_to_edge{edge_id}'
                e2d_key = f'edge{edge_id}_to_device'

                # Data sizes for transfers
                d2e_data_size = task.data_sizes.get(d2e_key, 0)
                e2d_data_size = task.data_sizes.get(e2d_key, 0)

                # Transfer rates
                d2e_rate = upload_rates.get(d2e_key, 1.0)
                e2d_rate = download_rates.get(e2d_key, 1.0)

                # Calculate transfer times
                d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate
                e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate

                # Total edge execution time = upload + processing + download
                total_edge_time = d2e_time + min_edge_core_time + e2d_time
                edge_times.append(total_edge_time)

        # Get minimum edge execution time if any
        t_edge_min = min(edge_times) if edge_times else float('inf')

        # Calculate T_i^re (cloud execution time)
        t_re = (task.cloud_execution_times[0] +  # T_i^s (send)
                task.cloud_execution_times[1] +  # T_i^c (cloud)
                task.cloud_execution_times[2])  # T_i^r (receive)

        # Determine optimal execution tier
        if t_l_min <= t_edge_min and t_l_min <= t_re:
            # Local execution is fastest
            task.execution_tier = ExecutionTier.DEVICE
        elif t_edge_min <= t_l_min and t_edge_min <= t_re:
            # Edge execution is fastest
            task.execution_tier = ExecutionTier.EDGE
            # Find which edge node gave this minimum time
            for edge_id in range(1, edge_nodes + 1):
                # Recalculate edge time for this specific edge node
                # This is slightly redundant but ensures correct assignment
                edge_cores_times = []
                for core_id in range(1, 3):
                    key = (task.id, edge_id, core_id)
                    if key in edge_execution_times:
                        edge_cores_times.append(edge_execution_times[key])

                if edge_cores_times:
                    min_edge_core_time = min(edge_cores_times)
                    d2e_key = f'device_to_edge{edge_id}'
                    e2d_key = f'edge{edge_id}_to_device'
                    d2e_data_size = task.data_sizes.get(d2e_key, 0)
                    e2d_data_size = task.data_sizes.get(e2d_key, 0)
                    d2e_rate = upload_rates.get(d2e_key, 1.0)
                    e2d_rate = download_rates.get(e2d_key, 1.0)
                    d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate
                    e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate
                    total_edge_time = d2e_time + min_edge_core_time + e2d_time

                    if abs(total_edge_time - t_edge_min) < 1e-9:  # Float comparison
                        # This is the edge node with minimum time
                        task.edge_assignment = EdgeAssignment(edge_id=edge_id, core_id=0)  # Core assigned later
                        break
        else:
            # Cloud execution is fastest
            task.execution_tier = ExecutionTier.CLOUD


def task_prioritizing(tasks):
    """
    Extended implementation of "Task Prioritizing" phase for three-tier architecture.
    Calculates priority levels for each task to determine scheduling order.
    """
    w = [0] * len(tasks)

    # Step 1: Calculate computation costs (wi) for each task
    for i, task in enumerate(tasks):
        if task.execution_tier == ExecutionTier.CLOUD:
            # For cloud tasks:
            w[i] = (task.cloud_execution_times[0] +  # Ti^s
                    task.cloud_execution_times[1] +  # Ti^c
                    task.cloud_execution_times[2])  # Ti^r

        elif task.execution_tier == ExecutionTier.EDGE:
            # For edge tasks:
            if task.edge_assignment:
                edge_id = task.edge_assignment.edge_id
                # Calculate average execution time across cores on this edge node
                edge_times = []
                for core_id in range(1, 3):  # Assuming 2 cores per edge node
                    key = (task.id, edge_id, core_id)
                    if key in edge_execution_times:
                        edge_times.append(edge_execution_times[key])

                if edge_times:
                    # Average computation time on the assigned edge
                    avg_edge_time = sum(edge_times) / len(edge_times)

                    # Add transfer times
                    d2e_key = f'device_to_edge{edge_id}'
                    e2d_key = f'edge{edge_id}_to_device'
                    d2e_data_size = task.data_sizes.get(d2e_key, 0)
                    e2d_data_size = task.data_sizes.get(e2d_key, 0)
                    d2e_rate = upload_rates.get(d2e_key, 1.0)
                    e2d_rate = download_rates.get(e2d_key, 1.0)
                    d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate
                    e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate

                    # Total edge execution time
                    w[i] = d2e_time + avg_edge_time + e2d_time
                else:
                    # Fallback if no edge times available
                    w[i] = float('inf')
            else:
                # Edge assignment missing - error case
                w[i] = float('inf')

        else:  # ExecutionTier.DEVICE
            # For local tasks:
            # Average computation time across all local cores
            w[i] = sum(task.local_execution_times) / len(task.local_execution_times)

    # Cache for memoization of priority calculations
    computed_priority_scores = {}

    def calculate_priority(task):
        """
        Recursive implementation of priority calculation.
        Extended for three-tier architecture but follows the same principle.
        """
        # Memoization check
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]

        # Base case: Exit tasks
        if not task.succ_tasks:  # Note: Fixed attribute name to match previous code
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]

        # Recursive case: Non-exit tasks
        max_successor_priority = 0
        if task.succ_tasks:  # Fixed attribute name
            max_successor_priority = max(calculate_priority(successor)
                                         for successor in task.succ_tasks)

        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all tasks using recursive algorithm
    for task in tasks:
        calculate_priority(task)

    # Update priority scores in task objects
    for task in tasks:
        task.priority_score = computed_priority_scores[task.id]


class ThreeTierTaskScheduler:
    """
    Comprehensive implementation of the initial scheduling algorithm for three-tier architecture.
    Extends the original InitialTaskScheduler to include edge nodes as an intermediate tier.
    """

    def __init__(self, tasks, num_local_cores=3, num_edge_nodes=2, cores_per_edge=2):
        """
        Initialize the scheduler with tasks and resources across three tiers.

        Args:
            tasks: List of Task objects representing the application DAG
            num_local_cores: Number of cores on the mobile device (K)
            num_edge_nodes: Number of edge computing nodes (M)
            cores_per_edge: Number of cores per edge node
        """
        self.tasks = tasks
        self.k = num_local_cores  # K cores on mobile device
        self.m = num_edge_nodes  # M edge nodes
        self.cores_per_edge = cores_per_edge  # Cores per edge node

        # === DEVICE RESOURCE TRACKING ===
        # Local core availability
        self.core_earliest_ready = [0] * self.k  # When each local core becomes available

        # === CLOUD RESOURCE TRACKING ===
        # Wireless channel availability for device-cloud communication
        self.ws_ready = 0  # Next available time for RF sending channel (device to cloud)
        self.wr_ready = 0  # Next available time for RF receiving channel (cloud to device)

        # === EDGE RESOURCE TRACKING ===
        # Edge node core availability
        self.edge_core_earliest_ready = {
            (edge_id, core_id): 0
            for edge_id in range(1, self.m + 1)
            for core_id in range(1, self.cores_per_edge + 1)
        }

        # === COMMUNICATION CHANNEL TRACKING ===
        # Device-Edge communication channels
        self.device_to_edge_ready = {edge_id: 0 for edge_id in range(1, self.m + 1)}
        self.edge_to_device_ready = {edge_id: 0 for edge_id in range(1, self.m + 1)}

        # Edge-Cloud communication channels
        self.edge_to_cloud_ready = {edge_id: 0 for edge_id in range(1, self.m + 1)}
        self.cloud_to_edge_ready = {edge_id: 0 for edge_id in range(1, self.m + 1)}

        # Edge-Edge communication channels
        self.edge_to_edge_ready = {
            (src, dst): 0
            for src in range(1, self.m + 1)
            for dst in range(1, self.m + 1) if src != dst
        }

        # === TASK SEQUENCES ===
        # Sequences of tasks scheduled on each execution unit
        self.sequences = {}

        # Local cores
        for i in range(self.k):
            self.sequences[f'device_core_{i}'] = []

        # Edge nodes and cores
        for edge_id in range(1, self.m + 1):
            for core_id in range(1, self.cores_per_edge + 1):
                self.sequences[f'edge_{edge_id}_core_{core_id}'] = []

        # Cloud
        self.sequences['cloud'] = []

        # Initialize task execution time information
        self._initialize_task_data()

    def _initialize_task_data(self):
        """Initialize task execution time data structures if needed."""
        for task in self.tasks:
            # Initialize finish time tracking for edge nodes if not already present
            if not hasattr(task, 'FT_edge') or task.FT_edge is None:
                task.FT_edge = {}

            # Initialize edge sending finish times if not already present
            if not hasattr(task, 'FT_edge_send') or task.FT_edge_send is None:
                task.FT_edge_send = {}

            # Initialize edge receiving finish times if not already present
            if not hasattr(task, 'FT_edge_receive') or task.FT_edge_receive is None:
                task.FT_edge_receive = {}

            # Initialize ready times for edge nodes if not already present
            if not hasattr(task, 'RT_edge') or task.RT_edge is None:
                task.RT_edge = {}

    def get_priority_ordered_tasks(self):
        """
        Orders tasks by priority scores calculated in the task_prioritizing phase.
        Returns a list of task IDs in descending order of priority.
        """
        # Sort tasks by their priority scores (higher first)
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)

        return [item[1] for item in task_priority_list]

    def classify_entry_tasks(self, priority_order):
        """
        Separates tasks into entry and non-entry tasks while maintaining priority order.
        Entry tasks have no predecessors and can start immediately.

        Args:
            priority_order: List of task IDs in priority order

        Returns:
            Tuple of (entry_tasks, non_entry_tasks)
        """
        entry_tasks = []
        non_entry_tasks = []

        for task_id in priority_order:
            # Find the task object from its ID
            task = next((t for t in self.tasks if t.id == task_id), None)
            if task:
                if not task.pred_tasks:
                    # Entry tasks have no predecessors
                    entry_tasks.append(task)
                else:
                    # Non-entry tasks have at least one predecessor
                    non_entry_tasks.append(task)

        return entry_tasks, non_entry_tasks

    def calculate_ready_time_local(self, task):
        """
        Calculates ready time for local execution on device cores.
        Implements the extended ready time formula for the three-tier architecture.

        Args:
            task: The task for which to calculate ready time

        Returns:
            The ready time for local execution
        """
        if not task.pred_tasks:
            return 0  # Entry task, ready at time 0

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            # Case 1: Predecessor executed locally on device
            local_finish = pred_task.FT_l if hasattr(pred_task, 'FT_l') and pred_task.FT_l > 0 else 0

            # Case 2: Predecessor executed in cloud
            cloud_receive = pred_task.FT_wr if hasattr(pred_task, 'FT_wr') and pred_task.FT_wr > 0 else 0

            # Case 3: Predecessor executed on edge node(s)
            edge_receive = 0

            if (hasattr(pred_task, 'execution_tier') and
                    pred_task.execution_tier == ExecutionTier.EDGE and
                    hasattr(pred_task, 'edge_assignment') and
                    pred_task.edge_assignment):

                edge_id = pred_task.edge_assignment.edge_id

                # Get when task finished on the edge node
                edge_finish = pred_task.FT_edge.get(edge_id, 0) if hasattr(pred_task, 'FT_edge') else 0

                # Check if results were transferred back to device
                if hasattr(pred_task, 'FT_edge_receive') and edge_id in pred_task.FT_edge_receive:
                    edge_receive = pred_task.FT_edge_receive[edge_id]
                else:
                    # Calculate edge-to-device transfer time
                    e2d_key = f'edge{edge_id}_to_device'
                    data_size = pred_task.data_sizes.get(e2d_key, 0) if hasattr(pred_task, 'data_sizes') else 0
                    rate = download_rates.get(e2d_key, 1.0)

                    # Δ_j,m->d = data_j^(m->d) / R_m->d^s
                    transfer_time = 0 if rate == 0 else data_size / rate

                    # FT_j^wr,(m->d) = FT_j^e,m + Δ_j,m->d
                    edge_receive = edge_finish + transfer_time

            # Take maximum of all possible finish times for this predecessor
            pred_finish = max(local_finish, cloud_receive, edge_receive)

            # Update max ready time across all predecessors
            max_ready_time = max(max_ready_time, pred_finish)

        return max_ready_time

    def calculate_ready_time_cloud(self, task):
        """
        Calculates ready time for cloud offloading.
        Implements the wireless sending ready time (RT_i^ws) calculation.

        Args:
            task: The task for which to calculate ready time

        Returns:
            The ready time for wireless sending
        """
        if not task.pred_tasks:
            return self.ws_ready  # Entry task, ready when channel is available

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if hasattr(pred_task, 'execution_tier'):
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Predecessor on device: wait for local finish
                    max_ready_time = max(max_ready_time, pred_task.FT_l if hasattr(pred_task, 'FT_l') else 0)

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Predecessor on cloud: wait for sending completion
                    max_ready_time = max(max_ready_time, pred_task.FT_ws if hasattr(pred_task, 'FT_ws') else 0)

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # Predecessor on edge: depends on whether we need data transfer
                    if hasattr(pred_task, 'edge_assignment') and pred_task.edge_assignment:
                        edge_id = pred_task.edge_assignment.edge_id

                        # Need to get data from edge first
                        if hasattr(pred_task, 'FT_edge_receive') and edge_id in pred_task.FT_edge_receive:
                            max_ready_time = max(max_ready_time, pred_task.FT_edge_receive[edge_id])
                        else:
                            # Calculate time when data would arrive from edge
                            edge_finish = pred_task.FT_edge.get(edge_id, 0) if hasattr(pred_task, 'FT_edge') else 0
                            e2d_key = f'edge{edge_id}_to_device'
                            data_size = pred_task.data_sizes.get(e2d_key, 0) if hasattr(pred_task, 'data_sizes') else 0
                            rate = download_rates.get(e2d_key, 1.0)
                            transfer_time = 0 if rate == 0 else data_size / rate
                            max_ready_time = max(max_ready_time, edge_finish + transfer_time)

        # Also consider wireless channel availability
        return max(max_ready_time, self.ws_ready)

    def calculate_ready_time_edge(self, task, edge_id):
        """
        Calculates ready time for execution on a specific edge node.

        Args:
            task: The task for which to calculate ready time
            edge_id: The edge node ID

        Returns:
            The ready time for edge execution
        """
        if not task.pred_tasks:
            return self.device_to_edge_ready[edge_id]  # Entry task, ready when channel is available

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if hasattr(pred_task, 'execution_tier'):
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Predecessor on device: wait for local finish
                    max_ready_time = max(max_ready_time, pred_task.FT_l if hasattr(pred_task, 'FT_l') else 0)

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Predecessor on cloud: wait for results to return
                    max_ready_time = max(max_ready_time, pred_task.FT_wr if hasattr(pred_task, 'FT_wr') else 0)

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    # Predecessor on edge: depends on whether it's the same edge
                    if hasattr(pred_task, 'edge_assignment') and pred_task.edge_assignment:
                        pred_edge_id = pred_task.edge_assignment.edge_id

                        if pred_edge_id == edge_id:
                            # Same edge node: just wait for execution to finish
                            edge_finish = pred_task.FT_edge.get(edge_id, 0) if hasattr(pred_task, 'FT_edge') else 0
                            max_ready_time = max(max_ready_time, edge_finish)
                        else:
                            # Different edge node: need edge-to-edge transfer
                            # This is complex - for simplicity, we'll assume data comes back to device first
                            if hasattr(pred_task, 'FT_edge_receive') and pred_edge_id in pred_task.FT_edge_receive:
                                max_ready_time = max(max_ready_time, pred_task.FT_edge_receive[pred_edge_id])
                            else:
                                # Calculate time when data would arrive from edge to device
                                edge_finish = pred_task.FT_edge.get(pred_edge_id, 0) if hasattr(pred_task,
                                                                                                'FT_edge') else 0
                                e2d_key = f'edge{pred_edge_id}_to_device'
                                data_size = pred_task.data_sizes.get(e2d_key, 0) if hasattr(pred_task,
                                                                                            'data_sizes') else 0
                                rate = download_rates.get(e2d_key, 1.0)
                                transfer_time = 0 if rate == 0 else data_size / rate
                                max_ready_time = max(max_ready_time, edge_finish + transfer_time)

        # Consider device-to-edge channel availability
        return max(max_ready_time, self.device_to_edge_ready[edge_id])

    def identify_optimal_local_core(self, task, ready_time=0):
        """
        Finds optimal local core assignment for a task to minimize finish time.

        Args:
            task: The task to schedule
            ready_time: Ready time for local execution

        Returns:
            Tuple of (core_id, start_time, finish_time)
        """
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')

        for core in range(self.k):
            # Calculate earliest possible start time on this core
            start_time = max(ready_time, self.core_earliest_ready[core])

            # Calculate finish time using local execution time for this core
            finish_time = start_time + task.local_execution_times[core]

            # Choose core that gives earliest finish time
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core
                best_start_time = start_time

        return best_core, best_start_time, best_finish_time

    def identify_optimal_edge_node(self, task, ready_time=0):
        """
        Finds optimal edge node and core assignment to minimize finish time.

        Args:
            task: The task to schedule
            ready_time: Base ready time (e.g., from predecessors)

        Returns:
            Tuple with edge assignment details and timing information
        """
        best_finish_time = float('inf')
        best_edge_id = -1
        best_core_id = -1
        best_start_time = float('inf')
        best_upload_start = 0
        best_upload_finish = 0
        best_download_start = 0
        best_download_finish = 0

        # Try each edge node
        for edge_id in range(1, self.m + 1):
            # Calculate data transfer times to this edge
            d2e_key = f'device_to_edge{edge_id}'
            e2d_key = f'edge{edge_id}_to_device'

            # Data sizes for transfers
            d2e_data_size = task.data_sizes.get(d2e_key, 0) if hasattr(task, 'data_sizes') else 0
            e2d_data_size = task.data_sizes.get(e2d_key, 0) if hasattr(task, 'data_sizes') else 0

            # Transfer rates
            d2e_rate = upload_rates.get(d2e_key, 1.0)
            e2d_rate = download_rates.get(e2d_key, 1.0)

            # Calculate transfer times
            d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate
            e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate

            # Upload start time depends on task ready time and channel availability
            edge_ready_time = self.calculate_ready_time_edge(task, edge_id)
            upload_start = max(ready_time, edge_ready_time, self.device_to_edge_ready[edge_id])
            upload_finish = upload_start + d2e_time

            # Try each core on this edge node
            for core_id in range(1, self.cores_per_edge + 1):
                # Execution can start after upload completes and core is available
                core_key = (edge_id, core_id)
                exec_start = max(upload_finish, self.edge_core_earliest_ready[core_key])

                # Get execution time for this edge node and core
                exec_time = task.get_edge_execution_time(edge_id, core_id) if hasattr(task,
                                                                                      'get_edge_execution_time') else 5
                exec_finish = exec_start + exec_time

                # Download can start after execution completes and channel is available
                download_start = max(exec_finish, self.edge_to_device_ready[edge_id])
                download_finish = download_start + e2d_time

                # Consider the total finish time (when results are available on device)
                if download_finish < best_finish_time:
                    best_finish_time = download_finish
                    best_edge_id = edge_id
                    best_core_id = core_id
                    best_start_time = exec_start
                    best_finish_exec = exec_finish
                    best_upload_start = upload_start
                    best_upload_finish = upload_finish
                    best_download_start = download_start
                    best_download_finish = download_finish

        return (best_edge_id, best_core_id, best_start_time, best_finish_exec,
                best_upload_start, best_upload_finish, best_download_start, best_download_finish)

    def calculate_cloud_phases_timing(self, task):
        """
        Calculates timing for the three-phase cloud execution model.

        Args:
            task: The task to schedule

        Returns:
            Tuple of timing details for cloud execution phases
        """
        # Phase 1: RF Sending Phase
        send_ready = task.RT_ws if hasattr(task, 'RT_ws') else self.calculate_ready_time_cloud(task)
        send_time = task.cloud_execution_times[0] if hasattr(task, 'cloud_execution_times') else 3
        send_finish = max(send_ready, self.ws_ready) + send_time

        # Phase 2: Cloud Computing Phase
        # Ready time depends on completion of sending and cloud predecessors
        cloud_ready = send_finish

        for pred_task in task.pred_tasks:
            if hasattr(pred_task, 'execution_tier') and pred_task.execution_tier == ExecutionTier.CLOUD:
                cloud_ready = max(cloud_ready, pred_task.FT_c if hasattr(pred_task, 'FT_c') else 0)

            # Add edge-to-cloud predecessor consideration
            elif (hasattr(pred_task, 'execution_tier') and
                  pred_task.execution_tier == ExecutionTier.EDGE and
                  hasattr(pred_task, 'edge_assignment') and
                  pred_task.edge_assignment):

                edge_id = pred_task.edge_assignment.edge_id
                # If this edge sent data to cloud for this predecessor
                edge_to_cloud_key = (edge_id, 'cloud')
                if hasattr(pred_task, 'FT_edge_send') and edge_to_cloud_key in pred_task.FT_edge_send:
                    cloud_ready = max(cloud_ready, pred_task.FT_edge_send[edge_to_cloud_key])

        # Cloud execution time
        cloud_time = task.cloud_execution_times[1] if hasattr(task, 'cloud_execution_times') else 1
        cloud_finish = cloud_ready + cloud_time

        # Phase 3: RF Receiving Phase
        receive_ready = cloud_finish
        receive_time = task.cloud_execution_times[2] if hasattr(task, 'cloud_execution_times') else 1
        receive_finish = max(receive_ready, self.wr_ready) + receive_time

        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """
        Schedules a task on a local core.

        Args:
            task: The task to schedule
            core: The core ID
            start_time: When the task will start
            finish_time: When the task will finish
        """
        # Set task execution details
        task.execution_tier = ExecutionTier.DEVICE
        task.device_core = core
        task.RT_l = start_time - task.local_execution_times[core]  # Back-calculate ready time
        task.FT_l = finish_time

        # Clear cloud and edge execution information
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0
        task.edge_assignment = None

        # Set overall execution finish time
        task.execution_finish_time = finish_time

        # Update core availability
        self.core_earliest_ready[core] = finish_time

        # Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # Add to appropriate sequence
        self.sequences[f'device_core_{core}'].append(task.id)

    def schedule_on_edge(self, task, edge_id, core_id, start_time, finish_time,
                         upload_start, upload_finish, download_start, download_finish):
        """
        Schedules a task on an edge node.

        Args:
            task: The task to schedule
            edge_id: The edge node ID
            core_id: The core ID on the edge node
            start_time: When execution will start on the edge
            finish_time: When execution will finish on the edge
            upload_start: When data upload to edge starts
            upload_finish: When data upload to edge finishes
            download_start: When result download from edge starts
            download_finish: When result download from edge finishes
        """
        # Set task execution tier and assignment
        task.execution_tier = ExecutionTier.EDGE
        task.edge_assignment = EdgeAssignment(edge_id=edge_id, core_id=core_id)

        # Clear local and cloud finish times
        task.FT_l = 0
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        # Set edge execution timing
        task.RT_edge[edge_id] = upload_finish  # Ready after upload completes
        task.FT_edge[edge_id] = finish_time

        # Set edge transfer timing
        d2e_key = (0, edge_id)  # 0 represents device
        task.FT_edge_send[d2e_key] = upload_finish

        e2d_key = edge_id
        task.FT_edge_receive[e2d_key] = download_finish

        # Set overall execution finish time (when results are available on device)
        task.execution_finish_time = download_finish

        # Update resource availability
        self.edge_core_earliest_ready[(edge_id, core_id)] = finish_time
        self.device_to_edge_ready[edge_id] = upload_finish
        self.edge_to_device_ready[edge_id] = download_finish

        # Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # Add to appropriate sequence
        self.sequences[f'edge_{edge_id}_core_{core_id}'].append(task.id)

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready,
                          receive_finish):
        """
        Schedules a task on the cloud.

        Args:
            task: The task to schedule
            send_ready: When data upload to cloud can start
            send_finish: When data upload to cloud completes
            cloud_ready: When cloud execution can start
            cloud_finish: When cloud execution completes
            receive_ready: When result download from cloud can start
            receive_finish: When result download from cloud completes
        """
        # Set task execution tier
        task.execution_tier = ExecutionTier.CLOUD

        # Clear local and edge finish times
        task.FT_l = 0
        task.edge_assignment = None

        # Set cloud execution timing
        task.RT_ws = send_ready
        task.FT_ws = send_finish
        task.RT_c = cloud_ready
        task.FT_c = cloud_finish
        task.RT_wr = receive_ready
        task.FT_wr = receive_finish

        # Set overall execution finish time
        task.execution_finish_time = receive_finish

        # Update wireless channel availability
        self.ws_ready = send_finish
        self.wr_ready = receive_finish

        # Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # Add to cloud sequence
        self.sequences['cloud'].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        """
        Schedules tasks with no predecessors.
        Process in order: local tasks first, then edge tasks, then cloud tasks.

        Args:
            entry_tasks: List of tasks with no predecessors
        """
        # Categorize entry tasks by their execution tier
        local_tasks = []
        edge_tasks = []
        cloud_tasks = []

        for task in entry_tasks:
            if hasattr(task, 'execution_tier'):
                if task.execution_tier == ExecutionTier.DEVICE:
                    local_tasks.append(task)
                elif task.execution_tier == ExecutionTier.EDGE:
                    edge_tasks.append(task)
                elif task.execution_tier == ExecutionTier.CLOUD:
                    cloud_tasks.append(task)
            else:
                # Default to local if not specified
                local_tasks.append(task)

        # Schedule local tasks first (lowest communication overhead)
        for task in local_tasks:
            core, start_time, finish_time = self.identify_optimal_local_core(task)
            self.schedule_on_local_core(task, core, start_time, finish_time)

        # Schedule edge tasks next
        for task in edge_tasks:
            # Find best edge node and timing
            (edge_id, core_id, start_time, finish_time,
             upload_start, upload_finish, download_start, download_finish) = self.identify_optimal_edge_node(task)

            if edge_id != -1:
                # Schedule on edge
                self.schedule_on_edge(task, edge_id, core_id, start_time, finish_time,
                                      upload_start, upload_finish, download_start, download_finish)
            else:
                # Fallback to local if no valid edge found
                core, start_time, finish_time = self.identify_optimal_local_core(task)
                self.schedule_on_local_core(task, core, start_time, finish_time)

        # Schedule cloud tasks last (highest communication overhead)
        for task in cloud_tasks:
            # Calculate ready time for cloud upload
            task.RT_ws = self.calculate_ready_time_cloud(task)

            # Calculate cloud execution timing
            timing = self.calculate_cloud_phases_timing(task)

            # Schedule on cloud
            self.schedule_on_cloud(task, *timing)

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """
        Schedules tasks that have predecessors.
        For each task, evaluate options across all tiers and choose the one that minimizes finish time.

        Args:
            non_entry_tasks: List of tasks with predecessors
        """
        for task in non_entry_tasks:
            # Calculate ready times for all execution units
            local_ready_time = self.calculate_ready_time_local(task)
            cloud_ready_time = self.calculate_ready_time_cloud(task)

            # Find best local core option
            local_core, local_start, local_finish = self.identify_optimal_local_core(task, local_ready_time)

            # Find best edge node option
            (edge_id, core_id, edge_start, edge_finish,
             upload_start, upload_finish, download_start, download_finish) = self.identify_optimal_edge_node(task,
                                                                                                             local_ready_time)

            # Set task's cloud ready time for cloud scheduling
            task.RT_ws = cloud_ready_time

            # Calculate cloud option
            cloud_timing = self.calculate_cloud_phases_timing(task)
            cloud_finish = cloud_timing[5]  # The receive_finish time

            # Choose execution path with earliest finish time
            if local_finish <= download_finish and local_finish <= cloud_finish:
                # Local execution is fastest
                self.schedule_on_local_core(task, local_core, local_start, local_finish)

            elif download_finish <= local_finish and download_finish <= cloud_finish:
                # Edge execution is fastest
                self.schedule_on_edge(task, edge_id, core_id, edge_start, edge_finish,
                                      upload_start, upload_finish, download_start, download_finish)

            else:
                # Cloud execution is fastest
                self.schedule_on_cloud(task, *cloud_timing)

    def calculate_execution_unit_load(self):
        """
        Calculates the load (utilization) of each execution unit.

        Returns:
            Dictionary mapping execution unit names to their utilization percentage
        """
        # Track total execution time and span for each unit
        unit_exec_time = {}
        unit_span = {}

        # Initialize with zero execution time
        for unit in self.sequences.keys():
            unit_exec_time[unit] = 0
            unit_span[unit] = 0

        # Calculate execution time for each unit
        for unit, task_ids in self.sequences.items():
            if not task_ids:
                continue

            # Find tasks from IDs
            unit_tasks = [next((t for t in self.tasks if t.id == task_id), None) for task_id in task_ids]
            unit_tasks = [t for t in unit_tasks if t is not None]

            if not unit_tasks:
                continue

            # Calculate total execution time
            if 'device_core' in unit:
                core = int(unit.split('_')[-1])
                unit_exec_time[unit] = sum(task.local_execution_times[core] for task in unit_tasks)

                # Find span (difference between first start and last finish)
                first_start = min(self.core_earliest_ready[core] - task.local_execution_times[core]
                                  for task in unit_tasks)
                last_finish = max(task.FT_l for task in unit_tasks)
                unit_span[unit] = last_finish - first_start

            elif 'edge' in unit:
                # Extract edge and core IDs
                parts = unit.split('_')
                edge_id = int(parts[1])
                core_id = int(parts[3])

                # Sum execution times
                unit_exec_time[unit] = sum(task.get_edge_execution_time(edge_id, core_id)
                                           if hasattr(task, 'get_edge_execution_time') else 5
                                           for task in unit_tasks)

                # Find span
                try:
                    first_start = min(task.RT_edge.get(edge_id, 0) for task in unit_tasks)
                    last_finish = max(task.FT_edge.get(edge_id, 0) for task in unit_tasks)
                    unit_span[unit] = last_finish - first_start
                except:
                    unit_span[unit] = 0

            elif unit == 'cloud':
                # For cloud, use cloud execution time
                unit_exec_time[unit] = sum(task.cloud_execution_times[1]
                                           if hasattr(task, 'cloud_execution_times') else 1
                                           for task in unit_tasks)

                # Find span
                first_start = min(task.RT_c for task in unit_tasks if hasattr(task, 'RT_c') and task.RT_c > 0)
                last_finish = max(task.FT_c for task in unit_tasks if hasattr(task, 'FT_c') and task.FT_c > 0)
                unit_span[unit] = last_finish - first_start

        # Calculate utilization
        unit_utilization = {}
        for unit in self.sequences.keys():
            if unit_span[unit] > 0:
                unit_utilization[unit] = unit_exec_time[unit] / unit_span[unit] * 100
            else:
                unit_utilization[unit] = 0

        return unit_utilization

    def execute(self):
        """
        Executes the full initial scheduling algorithm.

        Returns:
            Dictionary of task sequences for each execution unit
        """
        # Get tasks in priority order
        priority_ordered_tasks = self.get_priority_ordered_tasks()

        # Classify entry and non-entry tasks
        entry_tasks, non_entry_tasks = self.classify_entry_tasks(priority_ordered_tasks)

        # Schedule entry tasks
        self.schedule_entry_tasks(entry_tasks)

        # Schedule non-entry tasks
        self.schedule_non_entry_tasks(non_entry_tasks)

        return self.sequences

    def calculate_total_time(self):
        """
        Calculates the total completion time of the application.

        Returns:
            The application completion time
        """
        max_finish_time = 0

        for task in self.tasks:
            if hasattr(task, 'execution_finish_time'):
                max_finish_time = max(max_finish_time, task.execution_finish_time)
            elif hasattr(task, 'FT_l') and task.FT_l > 0:
                max_finish_time = max(max_finish_time, task.FT_l)
            elif hasattr(task, 'FT_wr') and task.FT_wr > 0:
                max_finish_time = max(max_finish_time, task.FT_wr)
            elif hasattr(task, 'FT_edge_receive') and task.FT_edge_receive:
                max_finish_time = max(max_finish_time, max(task.FT_edge_receive.values()))

        return max_finish_time

    def calculate_total_energy(self):
        """
        Calculates the total energy consumption.

        Returns:
            The total energy consumption
        """
        total_energy = 0

        # Core power consumption values
        core_powers = [1, 2, 4]  # Example values, should be from global config

        for task in self.tasks:
            if hasattr(task, 'execution_tier'):
                if task.execution_tier == ExecutionTier.DEVICE:
                    # Local execution
                    if hasattr(task, 'device_core') and 0 <= task.device_core < len(core_powers):
                        core = task.device_core
                        energy = core_powers[core] * task.local_execution_times[core]
                        total_energy += energy

                elif task.execution_tier == ExecutionTier.CLOUD:
                    # Cloud offloading - only count transmission energy
                    if hasattr(task, 'cloud_execution_times'):
                        # E_i^c = P^s × T_i^s
                        send_energy = rf_power.get('device_to_cloud', 0.5) * task.cloud_execution_times[0]
                        total_energy += send_energy

                elif task.execution_tier == ExecutionTier.EDGE:
                    # Edge execution - count transmission energy
                    if hasattr(task, 'edge_assignment') and task.edge_assignment:
                        edge_id = task.edge_assignment.edge_id

                        # Device to edge transfer
                        d2e_key = f'device_to_edge{edge_id}'
                        d2e_power = rf_power.get(d2e_key, 0.4)

                        # Calculate transfer time
                        d2e_data_size = task.data_sizes.get(d2e_key, 0) if hasattr(task, 'data_sizes') else 0
                        d2e_rate = upload_rates.get(d2e_key, 1.0)
                        d2e_time = 0 if d2e_rate == 0 else d2e_data_size / d2e_rate

                        # Add upload energy
                        total_energy += d2e_power * d2e_time

                        # Edge to device transfer (results)
                        e2d_key = f'edge{edge_id}_to_device'
                        e2d_power = rf_power.get(e2d_key, 0.3)

                        # Calculate transfer time
                        e2d_data_size = task.data_sizes.get(e2d_key, 0) if hasattr(task, 'data_sizes') else 0
                        e2d_rate = download_rates.get(e2d_key, 1.0)
                        e2d_time = 0 if e2d_rate == 0 else e2d_data_size / e2d_rate

                        # Add download energy
                        total_energy += e2d_power * e2d_time

        return total_energy


# Main implementation for the example
if __name__ == "__main__":
    # Create task graph with 10 tasks as specified
    tasks = []
    for i in range(1, 11):
        tasks.append(Task(id=i))

    # Set task dependencies as specified in the example
    tasks[0].pred_tasks = []
    tasks[0].succ_tasks = [tasks[1], tasks[2], tasks[3], tasks[4], tasks[5]]

    tasks[1].pred_tasks = [tasks[0]]
    tasks[1].succ_tasks = [tasks[7], tasks[8]]

    tasks[2].pred_tasks = [tasks[0]]
    tasks[2].succ_tasks = [tasks[6]]

    tasks[3].pred_tasks = [tasks[0]]
    tasks[3].succ_tasks = [tasks[7], tasks[8]]

    tasks[4].pred_tasks = [tasks[0]]
    tasks[4].succ_tasks = [tasks[8]]

    tasks[5].pred_tasks = [tasks[0]]
    tasks[5].succ_tasks = [tasks[7]]

    tasks[6].pred_tasks = [tasks[2]]
    tasks[6].succ_tasks = [tasks[9]]

    tasks[7].pred_tasks = [tasks[1], tasks[3], tasks[5]]
    tasks[7].succ_tasks = [tasks[9]]

    tasks[8].pred_tasks = [tasks[1], tasks[3], tasks[4]]
    tasks[8].succ_tasks = [tasks[9]]

    tasks[9].pred_tasks = [tasks[6], tasks[7], tasks[8]]
    tasks[9].succ_tasks = []

    # Configure task execution parameters
    for i, task in enumerate(tasks):
        # Set local execution times
        task.local_execution_times = core_execution_times[i + 1]

        # Set edge execution times (create faster options than local cores)
        min_local_time = min(task.local_execution_times)

        # Edge 1 cores (20% faster than local)
        edge_execution_times[(i + 1, 1, 1)] = min_local_time * 0.8
        edge_execution_times[(i + 1, 1, 2)] = min_local_time * 0.85

        # Edge 2 cores (40% faster than local)
        edge_execution_times[(i + 1, 2, 1)] = min_local_time * 0.6
        edge_execution_times[(i + 1, 2, 2)] = min_local_time * 0.65

        # Set data sizes for transfers
        task.data_sizes = {
            'device_to_edge1': 2.0,
            'device_to_edge2': 2.0,
            'device_to_cloud': 3.0,
            'edge1_to_edge2': 1.5,
            'edge2_to_edge1': 1.5,
            'edge1_to_cloud': 2.0,
            'edge2_to_cloud': 2.0,
            'cloud_to_device': 1.0,
            'edge1_to_device': 0.8,
            'edge2_to_device': 0.8
        }

    # STEP 1: Initial Scheduling
    print("Performing initial three-tier scheduling...")

    # Phase 1: Primary assignment
    primary_assignment(tasks)

    # Phase 2: Task prioritizing
    task_prioritizing(tasks)

    # Phase 3: Execute scheduler
    scheduler = ThreeTierTaskScheduler(tasks)
    scheduler.execute()

    # Calculate completion time and energy consumption
    completion_time = scheduler.calculate_total_time()
    energy = scheduler.calculate_total_energy()

    print(f"INITIAL SCHEDULING APPLICATION COMPLETION TIME: {completion_time}")
    print(f"INITIAL APPLICATION ENERGY CONSUMPTION: {energy}")

    # Print initial task schedule
    print("INITIAL TASK SCHEDULE: \n")
    print("Task Scheduling Details:")
    print("-" * 80)

    for task in tasks:
        print(f"\nTask ID        : {task.id}")

        if task.execution_tier == ExecutionTier.DEVICE:
            print(f"Assignment     : Core {task.device_core + 1}")
            print(f"Execution Window: {task.RT_l:.2f} → {task.FT_l:.2f}")
        elif task.execution_tier == ExecutionTier.CLOUD:
            print(f"Assignment     : Cloud")
            print(f"Send Phase     : {task.RT_ws:.2f} → {task.FT_ws:.2f}")
            print(f"Cloud Phase    : {task.RT_c:.2f} → {task.FT_c:.2f}")
            print(f"Receive Phase  : {task.RT_wr:.2f} → {task.FT_wr:.2f}")
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            edge_id = task.edge_assignment.edge_id
            core_id = task.edge_assignment.core_id
            print(f"Assignment     : Edge Node {edge_id}, Core {core_id}")

            # Get device to edge transfer timing
            d2e_key = (0, edge_id)
            upload_finish = task.FT_edge_send.get(d2e_key, 0)
            upload_start = max(0, upload_finish - 2)  # Approximate
            print(f"Device→Edge    : {upload_start:.2f} → {upload_finish:.2f}")

            # Get edge execution timing
            exec_start = task.RT_edge.get(edge_id, upload_finish)
            exec_finish = task.FT_edge.get(edge_id, 0)
            print(f"Edge Execution : {exec_start:.2f} → {exec_finish:.2f}")

            # Get edge to device transfer timing
            e2d_key = edge_id
            download_finish = task.FT_edge_receive.get(e2d_key, 0)
            download_start = max(exec_finish, download_finish - 1)  # Approximate
            print(f"Edge→Device    : {download_start:.2f} → {download_finish:.2f}")

        print("-" * 40)