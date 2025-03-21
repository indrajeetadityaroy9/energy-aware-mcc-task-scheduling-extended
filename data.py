from enum import Enum
import random
from typing import Dict, Callable, Any

####################################
# SECTION: CORE EXECUTION TIMES
####################################

# core_execution_times:
# Dictionary mapping task IDs (1 to 20) to a list of local execution times (e.g., on different cores).
# These represent the time it takes for a task to execute on various local cores.
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

# cloud_execution_times:
# List representing the three phases for cloud execution:
# T_send (time to send data to cloud), T_cloud (time to compute in cloud), and T_receive (time to receive results).
cloud_execution_times = [3, 1, 1]


####################################
# FUNCTION: initialize_edge_execution_times
####################################
def initialize_edge_execution_times(tasks=None):
    """
    Compute and return edge execution times for each task (IDs 1 to 20) on available edge nodes and their cores.

    For each task, the function:
      1. Retrieves local execution times from core_execution_times.
      2. Computes a baseline 'base_edge_time' by averaging the best local time and total cloud time.
      3. If task objects are provided, refines 'base_edge_time' based on the task's characteristics:
         - 'data' tasks: assumed to be data-intensive, so they run faster on the edge.
         - 'compute' tasks: assumed to be compute-intensive, so the edge time is an average of local and cloud times.
         - 'balanced' tasks: use a weighted average with a slight bias.
      4. Applies an adjustment factor based on task complexity and data intensity.
      5. Loops over each edge node and each core within the node, applying variability:
         - The second edge node is 10% slower.
         - The second core on each node is 5% slower.
      6. Saves computed execution times in a dictionary and, if tasks are provided, updates each task object's attribute.

    Parameters:
        tasks (list, optional): List of Task objects that have attributes such as id, type, complexity, and data_intensity.

    Returns:
        dict: A dictionary with keys (task_id, edge_id, core_id) and values as the computed edge execution times.
    """
    # Dictionary to store computed edge execution times.
    edge_execution_times = {}

    # Precompute the total cloud execution time (sum of sending, computing, and receiving times).
    cloud_time = sum(cloud_execution_times)  # For [3, 1, 1], cloud_time equals 5.

    # Process tasks with IDs 1 to 20.
    for task_id in range(1, 21):
        # Retrieve local execution times for this task; if not available, use a default list.
        local_times = core_execution_times.get(task_id, [9, 7, 5])
        min_local = min(local_times)  # Best (minimum) local execution time.

        # Default calculation: average of min_local and cloud_time.
        base_edge_time = (min_local + cloud_time) / 2.0

        # If task objects are provided, refine base_edge_time using task characteristics.
        task_obj = None
        if tasks is not None:
            # Find the matching task object using the task ID.
            task_obj = next((t for t in tasks if t.id == task_id), None)
            if task_obj is not None:
                # Retrieve task type; default to 'balanced' if missing.
                task_type = getattr(task_obj, 'type', 'balanced').lower()
                # Retrieve task complexity and data intensity, with default values if not provided.
                complexity = getattr(task_obj, 'complexity', 3.0)
                data_intensity = getattr(task_obj, 'data_intensity', 1.0)

                if task_type == 'data':
                    # Data-intensive tasks: edge is more efficient.
                    base_edge_time = 0.9 * min_local
                    # If data intensity is above average, reduce time slightly more.
                    adjustment_factor = 1.0 - 0.05 * max(data_intensity - 1.0, 0)
                elif task_type == 'compute':
                    # Compute-intensive tasks: average of local and cloud times.
                    base_edge_time = (min_local + cloud_time) / 2.0
                    # Increase execution time if task complexity is above average.
                    adjustment_factor = 1.0 + 0.05 * max(complexity - 3.0, 0)
                elif task_type == 'balanced':
                    # Balanced tasks: slightly favor local performance and average with cloud.
                    base_edge_time = (0.95 * min_local + cloud_time) / 2.0
                    # Minor adjustment based on difference between complexity and data intensity.
                    adjustment_factor = 1.0 + 0.03 * ((complexity - 3.0) - (data_intensity - 1.0))
                else:
                    # For any unknown type, fall back to the default.
                    base_edge_time = (min_local + cloud_time) / 2.0
                    adjustment_factor = 1.0

                # Clamp the adjustment_factor between 0.5 and 1.5 to avoid extreme modifications.
                adjustment_factor = max(0.5, min(adjustment_factor, 1.5))
                # Apply the adjustment factor to the base_edge_time.
                base_edge_time *= adjustment_factor

        # Loop over edge nodes and cores.
        # (Assumption: 2 edge nodes and 2 cores per node.)
        for edge_id in range(1, 3):
            for core_id in range(1, 3):
                computed_time = base_edge_time  # Start with the computed base time.

                # Apply variability: if it's the second edge node, increase time by 10%.
                if edge_id == 2:
                    computed_time *= 1.1
                # If it's the second core on a node, increase time by an additional 5%.
                if core_id == 2:
                    computed_time *= 1.05

                # Save the computed edge execution time for the task on this specific edge node and core.
                edge_execution_times[(task_id, edge_id, core_id)] = computed_time

                # If Task objects are provided, update the task's attribute with the computed time.
                if tasks is not None and task_obj is not None:
                    if not hasattr(task_obj, 'edge_execution_times'):
                        task_obj.edge_execution_times = {}
                    task_obj.edge_execution_times[(edge_id, core_id)] = computed_time

    return edge_execution_times


# Initialize edge execution times (global variable for later use).
edge_execution_times = initialize_edge_execution_times()


####################################
# FUNCTION: generate_realistic_power_models
####################################
def generate_realistic_power_models(
        device_type: str = 'mobile',
        battery_level: int = 100,
        num_edge_nodes: int = 2,
        num_edge_cores: int = 2
) -> Dict[str, Dict[Any, Any]]:
    """
    Generate realistic power consumption models that vary with load for different device types.

    Parameters:
        device_type (str): The type of device to model. Options are:
                           'mobile' (for mobile devices),
                           'edge_server' (for edge nodes),
                           'cloud_server' (for cloud servers).
        battery_level (int): The current battery level (percentage) for mobile devices.
                             This affects the power consumption profile.
        num_edge_nodes (int): Number of edge nodes available (only used for 'edge_server').
        num_edge_cores (int): Number of cores per edge node (only used for 'edge_server').

    Returns:
        Dict[str, Dict]: A dictionary with keys 'device', 'edge', 'cloud', and 'rf'.
                         Each key maps to a dictionary of power model parameters. For 'rf',
                         the dictionary maps strings to callable functions computing RF power.
    """
    power_models: Dict[str, Dict[Any, Any]] = {
        'device': {},
        'edge': {},
        'cloud': {},
        'rf': {}
    }

    if device_type == 'mobile':
        # Adjust battery factor: if battery level is low, efficiency decreases.
        battery_factor = 1.0 if battery_level > 30 else 1.0 + (30 - battery_level) * 0.01

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
        # Define RF power models for mobile transmissions.
        rf_model: Dict[str, Callable[[float, float], float]] = {
            'device_to_edge': lambda data_rate, signal_strength: (
                    (0.1 + 0.4 * (data_rate / 10) * (1 + (70 - signal_strength) * 0.02)) * battery_factor
            ),
            'device_to_cloud': lambda data_rate, signal_strength: (
                    (0.15 + 0.6 * (data_rate / 5) * (1 + (70 - signal_strength) * 0.03)) * battery_factor
            ),
        }
        power_models['rf'] = rf_model

    elif device_type == 'edge_server':
        # For edge servers, randomize efficiency to simulate a mix of homogeneous and heterogeneous hardware.
        for edge_id in range(1, num_edge_nodes + 1):
            for core_id in range(1, num_edge_cores + 1):
                # With 50% chance, treat the current edge node as homogeneous:
                # all cores have similar efficiency (sampled from a narrow range).
                if random.random() < 0.5:
                    efficiency = random.uniform(0.9, 1.0)
                else:
                    # Otherwise, use a heterogeneous model: base efficiency decreases with edge_id and core_id,
                    # then add a small random perturbation to simulate variation.
                    base_efficiency = 1.0 - 0.1 * (edge_id - 1) - 0.05 * (core_id - 1)
                    efficiency = base_efficiency * random.uniform(0.9, 1.1)

                power_models['edge'][(edge_id, core_id)] = {
                    'idle_power': 5.0 * efficiency,  # Idle power scales with efficiency.
                    'dynamic_power': (lambda load, eff=efficiency: (3.0 + 12.0 * load) * eff),
                    'frequency_range': (1.0, 3.2),  # Frequency range in GHz.
                    'current_frequency': 2.8,  # Current operating frequency.
                    'dvfs_enabled': True  # DVFS is enabled.
                }
        # For edge servers, the RF models are left empty (or can be added if needed).
        power_models['rf'] = {}

    elif device_type == 'cloud_server':
        power_models['cloud'] = {
            'idle_power': 50.0,
            'dynamic_power': lambda load: (20.0 + 180.0 * load),
            'frequency_range': (2.0, 4.0),
            'current_frequency': 3.5,
            'virtualization_overhead': 0.1  # 10% overhead due to virtualization.
        }
        power_models['rf'] = {}
    else:
        raise ValueError("Unsupported device_type. Choose 'mobile', 'edge_server', or 'cloud_server'.")

    return power_models


####################################
# FUNCTION: generate_realistic_network_conditions
####################################
def generate_realistic_network_conditions():
    """
    Generate network conditions (upload and download rates) based solely on base rates, random variability,
    and occasional degradation events. There is no time-of-day dependency in this model.

    Returns:
        tuple: Two dictionaries (upload_rates, download_rates) mapping link types to effective network rates (Mbps).
    """
    # Base upload rates in Mbps for different links.
    base_upload = {
        'device_to_edge': 10.0,  # From mobile device to edge node.
        'edge_to_edge': 30.0,  # Between two edge nodes.
        'edge_to_cloud': 50.0,  # From edge node to cloud.
        'device_to_cloud': 5.0,  # Directly from mobile device to cloud.
    }

    # Base download rates in Mbps for different links.
    base_download = {
        'edge_to_device': 12.0,  # From edge node to mobile device.
        'cloud_to_edge': 60.0,  # From cloud to edge node.
        'edge_to_edge': 30.0,  # Between edge nodes (assumed symmetric).
        'cloud_to_device': 6.0,  # From cloud to mobile device.
    }

    # Generate a random factor (±15%) to simulate natural network fluctuations.
    random_factor = random.uniform(0.85, 1.15)

    # Dictionaries to store the effective upload and download rates.
    upload_rates = {}
    download_rates = {}

    # Compute effective upload rates by applying the random factor.
    for link, base_rate in base_upload.items():
        effective_rate = base_rate * random_factor
        upload_rates[link] = effective_rate

    # Compute effective download rates by applying the random factor.
    for link, base_rate in base_download.items():
        effective_rate = base_rate * random_factor
        download_rates[link] = effective_rate

    # Occasionally, with a 5% chance, degrade one random link by 70% to simulate a network outage.
    if random.random() < 0.05:
        choice_is_upload = bool(random.getrandbits(1))
        if choice_is_upload:
            trouble_link = random.choice(list(upload_rates.keys()))
            upload_rates[trouble_link] *= 0.3  # Degrade the link significantly.
        else:
            trouble_link = random.choice(list(download_rates.keys()))
            download_rates[trouble_link] *= 0.3

    return upload_rates, download_rates


def add_task_attributes(
        predefined_tasks,
        num_edge_nodes=2,
        complexity_range=(0.5, 5.0),
        data_intensity_range=(0.2, 2.0),
        task_type_weights=None
):
    """
    Enhances a predefined task graph with realistic characteristics.
    Only adds attributes without modifying the graph structure.

    Parameters:
        predefined_tasks: List of Task objects with pred_tasks and succ_tasks already defined
        num_edge_nodes: Number of edge computing nodes in the system
        complexity_range: Range of possible computational complexity values
        data_intensity_range: Range of possible data intensity values
        task_type_weights: Distribution of task types (compute/data/balanced)

    Returns:
        The same task list with enhanced characteristics
    """

    # Default task type weights if not provided
    if task_type_weights is None:
        task_type_weights = {
            'compute': 0.3,
            'data': 0.3,
            'balanced': 0.4
        }

    # For each task, assign type, complexity, data intensity
    for task in predefined_tasks:
        # Assign task type (compute, data, balanced)
        task.task_type = random.choices(
            list(task_type_weights.keys()),
            weights=list(task_type_weights.values())
        )[0]

        # Set complexity based on task type
        if task.task_type == 'compute':
            task.complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
        elif task.task_type == 'data':
            task.complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
        else:  # 'balanced'
            task.complexity = random.uniform(complexity_range[0], complexity_range[1])

        # Set data intensity based on task type
        if task.task_type == 'data':
            task.data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
        elif task.task_type == 'compute':
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
        else:  # 'balanced'
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[1])

        # Generate data sizes dynamically based on number of edge nodes
        task.data_sizes = {
            'device_to_cloud': random.uniform(1.0, 5.0),
            'cloud_to_device': random.uniform(1.0, 5.0),
        }

        # Add edge-specific data sizes
        for i in range(1, num_edge_nodes + 1):
            # Device <-> Edge transfers
            task.data_sizes[f'device_to_edge{i}'] = random.uniform(0.5, 2.0)
            task.data_sizes[f'edge{i}_to_device'] = random.uniform(0.5, 2.0)

            # Cloud <-> Edge transfers
            task.data_sizes[f'edge{i}_to_cloud'] = random.uniform(2.0, 4.0)
            task.data_sizes[f'cloud_to_edge{i}'] = random.uniform(1.0, 3.0)

            # Edge <-> Edge transfers (for all possible combinations)
            for j in range(1, num_edge_nodes + 1):
                if i != j:  # Skip self-transfers
                    task.data_sizes[f'edge{i}_to_edge{j}'] = random.uniform(1.0, 3.0)

        # Log the enhanced task attributes
        print(
            f"Task {task.id}: Type={task.task_type}, " f"Task Complexity={task.complexity:.2f}, Data Intensity={task.data_intensity:.2f}")

    return predefined_tasks


####################################
# SECTION: CORE DATA STRUCTURES
####################################

class ExecutionTier(Enum):
    """
    ExecutionTier defines where a task can be executed in the three-tier architecture:
      - DEVICE: Execution on the mobile device (using local cores).
      - EDGE: Execution on edge nodes (intermediate tier).
      - CLOUD: Execution on the cloud platform.
    """
    DEVICE = 0  # Mobile device (local cores)
    EDGE = 1  # Edge nodes (intermediate tier)
    CLOUD = 2  # Cloud platform


class SchedulingState(Enum):
    """
    SchedulingState defines the state of task scheduling in the algorithm:
      - UNSCHEDULED: Task has not been scheduled.
      - SCHEDULED: Task has been scheduled during the initial minimal-delay scheduling.
      - KERNEL_SCHEDULED: Task has been rescheduled after energy optimization.
    """
    UNSCHEDULED = 0  # Initial state
    SCHEDULED = 1  # After initial minimal-delay scheduling
    KERNEL_SCHEDULED = 2  # After energy optimization
