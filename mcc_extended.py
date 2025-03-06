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