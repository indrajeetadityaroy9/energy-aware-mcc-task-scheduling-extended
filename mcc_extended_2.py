import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set, Any, NamedTuple, Callable
from copy import deepcopy
from collections import deque
import time as time_module

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
    1: {  # Task 1 (entry task - larger input/smaller output)
        'device_to_edge1': 4.2,
        'device_to_edge2': 4.2,
        'device_to_cloud': 5.0,
        'edge1_to_edge2': 2.5,
        'edge2_to_edge1': 2.5,
        'edge1_to_cloud': 3.8,
        'edge2_to_cloud': 3.8,
        'cloud_to_device': 1.2,
        'edge1_to_device': 1.0,
        'edge2_to_device': 1.0,
        'cloud_to_edge1': 0.9,
        'cloud_to_edge2': 0.9
    },
    2: {  # Task 2 (processes task 1's output - medium data size)
        'device_to_edge1': 3.5,
        'device_to_edge2': 3.5,
        'device_to_cloud': 4.2,
        'edge1_to_edge2': 2.0,
        'edge2_to_edge1': 2.0,
        'edge1_to_cloud': 3.2,
        'edge2_to_cloud': 3.2,
        'cloud_to_device': 2.8,
        'edge1_to_device': 2.2,
        'edge2_to_device': 2.2,
        'cloud_to_edge1': 2.0,
        'cloud_to_edge2': 2.0
    },
    3: {  # Task 3 (processes task 1's output - involves computation)
        'device_to_edge1': 2.8,
        'device_to_edge2': 2.8,
        'device_to_cloud': 3.5,
        'edge1_to_edge2': 1.8,
        'edge2_to_edge1': 1.8,
        'edge1_to_cloud': 2.5,
        'edge2_to_cloud': 2.5,
        'cloud_to_device': 1.5,
        'edge1_to_device': 1.2,
        'edge2_to_device': 1.2,
        'cloud_to_edge1': 1.0,
        'cloud_to_edge2': 1.0
    },
    4: {  # Task 4 (processes task 1's output - high compute task)
        'device_to_edge1': 3.0,
        'device_to_edge2': 3.0,
        'device_to_cloud': 3.8,
        'edge1_to_edge2': 2.2,
        'edge2_to_edge1': 2.2,
        'edge1_to_cloud': 2.8,
        'edge2_to_cloud': 2.8,
        'cloud_to_device': 3.2,
        'edge1_to_device': 2.5,
        'edge2_to_device': 2.5,
        'cloud_to_edge1': 2.0,
        'cloud_to_edge2': 2.0
    },
    5: {  # Task 5 (processes task 1's output - small data size)
        'device_to_edge1': 2.5,
        'device_to_edge2': 2.5,
        'device_to_cloud': 3.0,
        'edge1_to_edge2': 1.5,
        'edge2_to_edge1': 1.5,
        'edge1_to_cloud': 2.0,
        'edge2_to_cloud': 2.0,
        'cloud_to_device': 1.8,
        'edge1_to_device': 1.5,
        'edge2_to_device': 1.5,
        'cloud_to_edge1': 1.2,
        'cloud_to_edge2': 1.2
    },
    6: {  # Task 6 (processes task 1's output - medium-high data size)
        'device_to_edge1': 3.8,
        'device_to_edge2': 3.8,
        'device_to_cloud': 4.5,
        'edge1_to_edge2': 2.8,
        'edge2_to_edge1': 2.8,
        'edge1_to_cloud': 3.5,
        'edge2_to_cloud': 3.5,
        'cloud_to_device': 2.5,
        'edge1_to_device': 2.0,
        'edge2_to_device': 2.0,
        'cloud_to_edge1': 1.8,
        'cloud_to_edge2': 1.8
    },
    7: {  # Task 7 (processes task 3's output - medium data transfer)
        'device_to_edge1': 3.2,
        'device_to_edge2': 3.2,
        'device_to_cloud': 4.0,
        'edge1_to_edge2': 2.4,
        'edge2_to_edge1': 2.4,
        'edge1_to_cloud': 3.0,
        'edge2_to_cloud': 3.0,
        'cloud_to_device': 2.2,
        'edge1_to_device': 1.8,
        'edge2_to_device': 1.8,
        'cloud_to_edge1': 1.5,
        'cloud_to_edge2': 1.5
    },
    8: {  # Task 8 (processes outputs from 2,4,6 - larger data size)
        'device_to_edge1': 4.5,
        'device_to_edge2': 4.5,
        'device_to_cloud': 5.5,
        'edge1_to_edge2': 3.0,
        'edge2_to_edge1': 3.0,
        'edge1_to_cloud': 4.0,
        'edge2_to_cloud': 4.0,
        'cloud_to_device': 3.5,
        'edge1_to_device': 2.8,
        'edge2_to_device': 2.8,
        'cloud_to_edge1': 2.5,
        'cloud_to_edge2': 2.5
    },
    9: {  # Task 9 (processes outputs from 2,4,5 - larger data size)
        'device_to_edge1': 4.0,
        'device_to_edge2': 4.0,
        'device_to_cloud': 5.2,
        'edge1_to_edge2': 2.8,
        'edge2_to_edge1': 2.8,
        'edge1_to_cloud': 3.8,
        'edge2_to_cloud': 3.8,
        'cloud_to_device': 3.0,
        'edge1_to_device': 2.5,
        'edge2_to_device': 2.5,
        'cloud_to_edge1': 2.2,
        'cloud_to_edge2': 2.2
    },
    10: {  # Task 10 (final task - processes outputs from 7,8,9)
        'device_to_edge1': 5.0,
        'device_to_edge2': 5.0,
        'device_to_cloud': 6.0,
        'edge1_to_edge2': 3.5,
        'edge2_to_edge1': 3.5,
        'edge1_to_cloud': 4.5,
        'edge2_to_cloud': 4.5,
        'cloud_to_device': 4.0,
        'edge1_to_device': 3.2,
        'edge2_to_device': 3.2,
        'cloud_to_edge1': 3.0,
        'cloud_to_edge2': 3.0
    }
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
    """Enhanced TaskMigrationState for three-tier evaluations"""
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

    # Detailed timing information
    old_task_finish_time: float = 0.0  # Task's original finish time
    new_task_finish_time: float = 0.0  # Task's finish time after migration

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


class SequenceManager:
    """
    Enhanced sequence manager for three-tier architecture.

    Maintains execution sequences for all execution units across the three-tier architecture:
    - Device cores
    - Edge nodes (multiple cores per node)
    - Cloud platform

    Provides methods for sequence manipulation, insertion, and validation.
    """

    def __init__(self, num_device_cores: int, num_edge_nodes: int, num_edge_cores_per_node: int):
        """
        Initialize sequence manager with resource counts.

        Args:
            num_device_cores: Number of cores on the mobile device
            num_edge_nodes: Number of edge nodes
            num_edge_cores_per_node: Number of cores per edge node
        """
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
        """
        Set all sequences for all execution units.

        Args:
            sequences: List of sequences for all execution units
        """
        if len(sequences) != self.total_units:
            raise ValueError(f"Expected {self.total_units} sequences, got {len(sequences)}")

        self.sequences = deepcopy(sequences)


@dataclass
class OptimizationMetrics:
    """Tracks metrics during optimization"""
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
        """Initialize metrics at start of optimization"""
        self.initial_time = initial_time
        self.initial_energy = initial_energy
        self.current_time = initial_time
        self.current_energy = initial_energy
        self.best_time = initial_time
        self.best_energy = initial_energy
        self.start_time = time_module.time()

    def update(self, new_time: float, new_energy: float, evaluations: int = 0):
        """Update metrics after an iteration"""
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
        """Get summary of optimization metrics"""
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
    """
    Enhanced cache for three-tier migration evaluations.

    Provides efficient caching and retrieval of migration evaluation results
    to avoid redundant calculations in the optimization algorithm.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize cache with maximum capacity.

        Args:
            capacity: Maximum number of entries to store
        """
        self.capacity = capacity
        self.cache = {}
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: tuple) -> Optional[Tuple[float, float]]:
        """
        Get a cached value for a migration scenario.

        Args:
            key: Cache key from generate_key()

        Returns:
            Tuple (time, energy) if cached, None otherwise
        """
        self.access_count += 1

        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]

        self.miss_count += 1
        return None

    def put(self, key: tuple, value: Tuple[float, float]) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key from generate_key()
            value: Tuple (time, energy) to cache
        """
        # Check if we need to clear the cache
        if len(self.cache) >= self.capacity:
            logger.info(f"Clearing migration cache (size: {len(self.cache)})")
            self.cache.clear()

        self.cache[key] = value


class ThreeTierKernelScheduler:
    """
            Kernel scheduler for three-tier architecture.

            Implements linear-time rescheduling for all three tiers (device, edge, cloud)
            while maintaining task dependencies and resource constraints.
            """

    def __init__(self, tasks: List[Any], sequences: List[List[int]],
                 num_device_cores=3, num_edge_nodes=2, num_edge_cores_per_node=2):
        """
                Initialize the three-tier kernel scheduler.

                Args:
                    tasks: List of all tasks
                    sequences: Lists of task sequences for all execution units
                    num_device_cores: Number of device cores
                    num_edge_nodes: Number of edge nodes
                    num_edge_cores_per_node: Number of cores per edge node
                """
        self.tasks = tasks
        self.sequences = sequences
        self.num_device_cores = num_device_cores
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores_per_node = num_edge_cores_per_node

        # Resource timing trackers
        # Device cores
        self.device_cores_ready = [0] * num_device_cores

        # Edge cores
        self.edge_cores_ready = [
            [0] * num_edge_cores_per_node for _ in range(num_edge_nodes)
        ]

        # Communication channels
        # Device → Cloud
        self.device_to_cloud_ready = 0
        self.cloud_to_device_ready = 0

        # Device → Edge
        self.device_to_edge_ready = [0] * num_edge_nodes
        self.edge_to_device_ready = [0] * num_edge_nodes

        # Edge → Cloud
        self.edge_to_cloud_ready = [0] * num_edge_nodes
        self.cloud_to_edge_ready = [0] * num_edge_nodes

        # Edge → Edge
        self.edge_to_edge_ready = [
            [0] * num_edge_nodes for _ in range(num_edge_nodes)
        ]

        # Initialize task readiness tracking
        self.dependency_ready, self.sequence_ready = self._initialize_task_state()

    def _initialize_task_state(self):
        """
                Initialize task readiness tracking vectors.

                This is an extension of the original algorithm's ready1 and ready2 vectors
                to support the three-tier architecture.

                Returns:
                    Tuple (dependency_ready, sequence_ready)
                """
        # Initialize dependency tracking (ready1)
        # dependency_ready[j] is the number of immediate predecessors not yet scheduled
        dependency_ready = [len(task.pred_tasks) for task in self.tasks]

        # Initialize sequence position tracking (ready2)
        # sequence_ready[j] indicates if task is ready in its sequence:
        # -1: Task not in current sequence
        #  0: Task ready to execute (first in sequence or predecessor completed)
        #  1: Task waiting for predecessor in sequence
        sequence_ready = [-1] * len(self.tasks)

        # Process each execution sequence
        for sequence in self.sequences:
            if sequence:  # Non-empty sequence
                # Mark first task in sequence as ready
                sequence_ready[sequence[0] - 1] = 0

        return dependency_ready, sequence_ready

    def update_task_state(self, task):
        """
                Update readiness vectors for a task after scheduling changes.

                This is an extension of the original algorithm to maintain the
                invariants required for three-tier linear-time scheduling.

                Args:
                    task: Task object
                """
        task_idx = task.id - 1  # Convert to 0-based index

        # Only update state for unscheduled tasks
        if task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            # Update dependency tracking (ready1)
            self.dependency_ready[task_idx] = sum(
                1 for pred_task in task.pred_tasks
                if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )

            # Update sequence position tracking (ready2)
            for s_idx, sequence in enumerate(self.sequences):
                if task.id in sequence:
                    position = sequence.index(task.id)
                    if position > 0:
                        # Task has predecessor in sequence
                        pred_task_id = sequence[position - 1]
                        pred_task = self.tasks[pred_task_id - 1]

                        # Check if predecessor has been scheduled
                        self.sequence_ready[task_idx] = (
                            # 1: Waiting for predecessor
                            # 0: Predecessor completed
                            1 if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
                            else 0
                        )
                    else:
                        # First task in sequence
                        self.sequence_ready[task_idx] = 0
                    break

    def schedule_device_task(self, task):
        """
                Schedule a task on a device core.

                Args:
                    task: Task to schedule
                """
        # Get task's device core
        if hasattr(task, 'device_core'):
            core_id = task.device_core
        else:
            core_id = task.assignment

        # Calculate ready time
        ready_time = self._calculate_device_ready_time(task)

        # Calculate start time considering core availability
        start_time = max(self.device_cores_ready[core_id], ready_time)

        # Calculate finish time
        execution_time = task.local_execution_times[core_id]
        finish_time = start_time + execution_time

        # Update core availability
        self.device_cores_ready[core_id] = finish_time

        # Update task timing information
        task.FT_l = finish_time

        # Initialize execution timing array if needed
        if not hasattr(task, 'execution_unit_task_start_times') or task.execution_unit_task_start_times is None:
            total_units = self.num_device_cores + (self.num_edge_nodes * self.num_edge_cores_per_node) + 1
            task.execution_unit_task_start_times = [-1] * total_units

        # Record start time
        task.execution_unit_task_start_times[core_id] = start_time

        # Clear edge and cloud execution times
        if hasattr(task, 'FT_edge'):
            task.FT_edge = {}
        if hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        logger.debug(f"Scheduled task {task.id} on device core {core_id} from {start_time} to {finish_time}")

    def schedule_edge_task(self, task):
        """
                Schedule a task on an edge core.

                Args:
                    task: Task to schedule
                """
        if not hasattr(task, 'edge_assignment') or not task.edge_assignment:
            logger.error(f"Task {task.id} missing edge assignment")
            return

        # Get task's edge node and core (convert to 0-based indices)
        edge_id = task.edge_assignment.edge_id - 1
        core_id = task.edge_assignment.core_id - 1

        # Calculate ready time
        ready_time = self._calculate_edge_ready_time(task, edge_id)

        # Calculate start time considering core availability
        start_time = max(self.edge_cores_ready[edge_id][core_id], ready_time)

        # Get execution time for this edge core
        execution_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
        if execution_time is None:
            logger.error(f"Missing execution time for task {task.id} on edge {edge_id + 1}, core {core_id + 1}")
            return

        # Calculate finish time
        finish_time = start_time + execution_time

        # Update edge core availability
        self.edge_cores_ready[edge_id][core_id] = finish_time

        # Update task timing information
        if not hasattr(task, 'FT_edge'):
            task.FT_edge = {}
        task.FT_edge[edge_id] = finish_time
        task.FT_edge[edge_id + 1] = finish_time  # Also store with 1-based index for compatibility

        # Calculate and update edge→device transfer time
        self._calculate_edge_to_device_transfer(task, edge_id, finish_time)

        # Initialize execution timing array if needed
        if not hasattr(task, 'execution_unit_task_start_times') or task.execution_unit_task_start_times is None:
            total_units = self.num_device_cores + (self.num_edge_nodes * self.num_edge_cores_per_node) + 1
            task.execution_unit_task_start_times = [-1] * total_units

        # Calculate sequence index for this edge core
        edge_idx = self.num_device_cores + (edge_id * self.num_edge_cores_per_node) + core_id

        # Record start time
        task.execution_unit_task_start_times[edge_idx] = start_time

        # Clear device and cloud execution times
        task.FT_l = 0
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        logger.debug(
            f"Scheduled task {task.id} on edge {edge_id + 1}, core {core_id + 1} from {start_time} to {finish_time}")

    def _calculate_edge_to_device_transfer(self, task, edge_id, edge_finish_time):
        """
                Calculate edge→device transfer timing.

                Args:
                    task: Task object
                    edge_id: Edge node ID (0-based)
                    edge_finish_time: When edge execution finishes
                """
        # Check for data size information
        data_key = f'edge{edge_id + 1}_to_device'
        if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
            # Use default transfer time
            transfer_time = 1.0
        else:
            data_size = task.data_sizes[data_key]
            # Get download rate
            rate_key = f'edge{edge_id + 1}_to_device'
            if not 'download_rates' in globals() or rate_key not in download_rates:
                rate = 2.0  # Default rate
            else:
                rate = download_rates[rate_key]

            transfer_time = data_size / rate if rate > 0 else 0

        # Calculate transfer timing considering channel availability
        transfer_start = max(edge_finish_time, self.edge_to_device_ready[edge_id])
        transfer_finish = transfer_start + transfer_time

        # Update edge→device channel availability
        self.edge_to_device_ready[edge_id] = transfer_finish

        # Record edge→device transfer finish time
        if not hasattr(task, 'FT_edge_receive'):
            task.FT_edge_receive = {}
        task.FT_edge_receive[edge_id] = transfer_finish
        task.FT_edge_receive[edge_id + 1] = transfer_finish  # Also store with 1-based index

    def schedule_cloud_task(self, task):
        """
        Schedule a task on the cloud.
        Instead of using fixed or stale timing values, we always update cloud timing
        based on the current finish times of the task’s predecessors.
        """
        # Determine source tier for upload (if migrating from DEVICE or EDGE)
        if hasattr(task, 'execution_tier'):
            source_tier = task.execution_tier
        else:
            source_tier = ExecutionTier.DEVICE if hasattr(task,
                                                          'is_core_task') and task.is_core_task else ExecutionTier.CLOUD

        if source_tier == ExecutionTier.DEVICE:
            upload_ready = self._calculate_device_to_cloud_ready_time(task)
            channel_ready = self.device_to_cloud_ready
            upload_start = max(upload_ready, channel_ready)
            upload_time = task.cloud_execution_times[0]
            upload_finish = upload_start + upload_time
            self.device_to_cloud_ready = upload_finish
        elif source_tier == ExecutionTier.EDGE:
            edge_id = task.edge_assignment.edge_id - 1
            upload_ready = self._calculate_edge_to_cloud_ready_time(task, edge_id)
            channel_ready = self.edge_to_cloud_ready[edge_id]
            upload_start = max(upload_ready, channel_ready)
            data_key = f'edge{edge_id + 1}_to_cloud'
            if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                upload_time = 2.0
            else:
                data_size = task.data_sizes[data_key]
                rate_key = f'edge{edge_id + 1}_to_cloud'
                upload_time = task.cloud_execution_times[0]  # use the task's value directly
            upload_finish = upload_start + upload_time
            self.edge_to_cloud_ready[edge_id] = upload_finish
        else:
            # For a task already on cloud, let’s use its current value.
            upload_start = task.RT_ws if task.RT_ws > 0 else 0
            upload_time = task.cloud_execution_times[0]
            upload_finish = upload_start + upload_time

        # CRITICAL CHANGE: Always update cloud timing dynamically
        update_cloud_timing(task, self.tasks)

        # Now compute download phase based on source tier.
        if source_tier == ExecutionTier.DEVICE:
            download_start = task.FT_c
            download_time = task.cloud_execution_times[2]
            download_actual_start = max(download_start, self.cloud_to_device_ready)
            download_finish = download_actual_start + download_time
            self.cloud_to_device_ready = download_finish
        elif source_tier == ExecutionTier.EDGE:
            edge_id = task.edge_assignment.edge_id - 1
            download_start = task.FT_c
            data_key = f'cloud_to_edge{edge_id + 1}'
            if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                download_time = 1.0
            else:
                data_size = task.data_sizes[data_key]
                rate_key = f'cloud_to_edge{edge_id + 1}'
                download_time = task.cloud_execution_times[2]  # use the task's value
            download_actual_start = max(download_start, self.cloud_to_edge_ready[edge_id])
            download_finish = download_actual_start + download_time
            self.cloud_to_edge_ready[edge_id] = download_finish
        else:
            download_finish = upload_finish

        # Record cloud execution start in the task's timeline.
        task.execution_unit_task_start_times = task.execution_unit_task_start_times or \
                                               [-1] * (
                                                       self.num_device_cores + self.num_edge_nodes * self.num_edge_cores_per_node + 1)
        cloud_idx = self.num_device_cores + (self.num_edge_nodes * self.num_edge_cores_per_node)
        task.execution_unit_task_start_times[cloud_idx] = task.RT_ws

        # Clear local and edge finish times
        task.FT_l = 0
        task.FT_edge = {}
        task.FT_edge_receive = {}

        logger.debug(f"Scheduled task {task.id} on cloud: upload {task.RT_ws}-{task.FT_ws}, "
                     f"compute {task.RT_c}-{task.FT_c}, download {task.RT_wr}-{task.FT_wr}")

    def _calculate_device_ready_time(self, task):
        """
                Calculate ready time for device execution.

                Args:
                    task: Task object

                Returns:
                    Ready time for device execution
                """
        if not task.pred_tasks:
            return 0  # Entry tasks can start immediately

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')  # Predecessor not yet scheduled

            # Calculate ready time based on predecessor location
            pred_ready_time = 0

            if hasattr(pred_task, 'execution_tier'):
                # Three-tier task
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    pred_ready_time = pred_task.FT_l
                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    pred_ready_time = pred_task.FT_wr
                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    if not pred_task.edge_assignment:
                        return float('inf')

                    edge_id = pred_task.edge_assignment.edge_id - 1
                    if edge_id in pred_task.FT_edge_receive:
                        pred_ready_time = pred_task.FT_edge_receive[edge_id]
                    else:
                        # This should have been calculated during scheduling
                        return float('inf')
            else:
                # Original MCC task
                if hasattr(pred_task, 'is_core_task') and pred_task.is_core_task:
                    pred_ready_time = pred_task.FT_l
                else:
                    pred_ready_time = pred_task.FT_wr

            max_ready_time = max(max_ready_time, pred_ready_time)

        return max_ready_time

    def _calculate_edge_ready_time(self, task, edge_id):
        """
                Calculate ready time for edge execution.

                Args:
                    task: Task object
                    edge_id: Edge node ID (0-based)

                Returns:
                    Ready time for edge execution
                """
        if not task.pred_tasks:
            return 0  # Entry tasks can start immediately

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')  # Predecessor not yet scheduled

            # Calculate ready time based on predecessor location
            pred_ready_time = 0

            if hasattr(pred_task, 'execution_tier'):
                # Three-tier task
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Need device→edge transfer
                    device_finish = pred_task.FT_l

                    # Calculate transfer time
                    data_key = f'device_to_edge{edge_id + 1}'
                    if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                        transfer_time = 2.0  # Default transfer time
                    else:
                        data_size = task.data_sizes[data_key]
                        rate_key = f'device_to_edge{edge_id + 1}'
                        if not 'upload_rates' in globals() or rate_key not in upload_rates:
                            rate = 1.5  # Default rate
                        else:
                            rate = upload_rates[rate_key]

                        transfer_time = data_size / rate if rate > 0 else 0

                    # Calculate transfer timing considering channel availability
                    transfer_start = max(device_finish, self.device_to_edge_ready[edge_id])
                    transfer_finish = transfer_start + transfer_time

                    # Update device→edge channel availability
                    self.device_to_edge_ready[edge_id] = transfer_finish

                    pred_ready_time = transfer_finish

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Need cloud→edge transfer
                    cloud_finish = pred_task.FT_c

                    # Calculate transfer time
                    data_key = f'cloud_to_edge{edge_id + 1}'
                    if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                        transfer_time = 1.0  # Default transfer time
                    else:
                        data_size = task.data_sizes[data_key]
                        rate_key = f'cloud_to_edge{edge_id + 1}'
                        if not 'download_rates' in globals() or rate_key not in download_rates:
                            rate = 3.0  # Default rate
                        else:
                            rate = download_rates[rate_key]

                        transfer_time = data_size / rate if rate > 0 else 0

                    # Calculate transfer timing considering channel availability
                    transfer_start = max(cloud_finish, self.cloud_to_edge_ready[edge_id])
                    transfer_finish = transfer_start + transfer_time

                    # Update cloud→edge channel availability
                    self.cloud_to_edge_ready[edge_id] = transfer_finish

                    pred_ready_time = transfer_finish

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    if not pred_task.edge_assignment:
                        return float('inf')

                    pred_edge_id = pred_task.edge_assignment.edge_id - 1

                    if pred_edge_id == edge_id:
                        # Same edge node, no transfer needed
                        if pred_edge_id in pred_task.FT_edge:
                            pred_ready_time = pred_task.FT_edge[pred_edge_id]
                        else:
                            return float('inf')
                    else:
                        # Different edge node, need edge→edge transfer
                        if pred_edge_id not in pred_task.FT_edge:
                            return float('inf')

                        edge_finish = pred_task.FT_edge[pred_edge_id]

                        # Calculate transfer time
                        data_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                        if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                            transfer_time = 1.5  # Default transfer time
                        else:
                            data_size = task.data_sizes[data_key]
                            rate_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                            if not 'upload_rates' in globals() or rate_key not in upload_rates:
                                rate = 3.0  # Default rate
                            else:
                                rate = upload_rates[rate_key]

                            transfer_time = data_size / rate if rate > 0 else 0

                        # Calculate transfer timing considering channel availability
                        transfer_start = max(edge_finish, self.edge_to_edge_ready[pred_edge_id][edge_id])
                        transfer_finish = transfer_start + transfer_time

                        # Update edge→edge channel availability
                        self.edge_to_edge_ready[pred_edge_id][edge_id] = transfer_finish

                        pred_ready_time = transfer_finish
            else:
                # Original MCC task
                if hasattr(pred_task, 'is_core_task') and pred_task.is_core_task:
                    # Need device→edge transfer
                    device_finish = pred_task.FT_l

                    # Use default transfer time
                    transfer_time = 2.0

                    # Calculate transfer timing considering channel availability
                    transfer_start = max(device_finish, self.device_to_edge_ready[edge_id])
                    transfer_finish = transfer_start + transfer_time

                    # Update device→edge channel availability
                    self.device_to_edge_ready[edge_id] = transfer_finish

                    pred_ready_time = transfer_finish
                else:
                    # Need cloud→edge transfer
                    cloud_finish = pred_task.FT_c

                    # Use default transfer time
                    transfer_time = 1.0

                    # Calculate transfer timing considering channel availability
                    transfer_start = max(cloud_finish, self.cloud_to_edge_ready[edge_id])
                    transfer_finish = transfer_start + transfer_time

                    # Update cloud→edge channel availability
                    self.cloud_to_edge_ready[edge_id] = transfer_finish

                    pred_ready_time = transfer_finish

            max_ready_time = max(max_ready_time, pred_ready_time)

        return max_ready_time

    def _calculate_device_to_cloud_ready_time(self, task):
        """
                Calculate ready time for cloud upload from device.

                Args:
                    task: Task object

                Returns:
                    Ready time for cloud upload
                """
        if not task.pred_tasks:
            return 0  # Entry tasks can start immediately

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')  # Predecessor not yet scheduled

            # Calculate ready time based on predecessor location
            pred_ready_time = 0

            if hasattr(pred_task, 'execution_tier'):
                # Three-tier task
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    pred_ready_time = pred_task.FT_l
                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    pred_ready_time = pred_task.FT_ws
                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    if not pred_task.edge_assignment:
                        return float('inf')

                    edge_id = pred_task.edge_assignment.edge_id - 1
                    if edge_id in pred_task.FT_edge_receive:
                        # Need to wait for results to return to device
                        pred_ready_time = pred_task.FT_edge_receive[edge_id]
                    else:
                        # This should have been calculated during scheduling
                        return float('inf')
            else:
                # Original MCC task
                if hasattr(pred_task, 'is_core_task') and pred_task.is_core_task:
                    pred_ready_time = pred_task.FT_l
                else:
                    pred_ready_time = pred_task.FT_ws

            max_ready_time = max(max_ready_time, pred_ready_time)

        return max_ready_time

    def _calculate_edge_to_cloud_ready_time(self, task, edge_id):
        """
                Calculate ready time for cloud upload from edge.

                Args:
                    task: Task object
                    edge_id: Edge node ID (0-based)

                Returns:
                    Ready time for cloud upload
                """
        if not task.pred_tasks:
            return 0  # Entry tasks can start immediately

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
                return float('inf')  # Predecessor not yet scheduled

            # Calculate ready time based on predecessor location
            pred_ready_time = 0

            if hasattr(pred_task, 'execution_tier'):
                # Three-tier task
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Need device→edge transfer first
                    device_finish = pred_task.FT_l

                    # Calculate transfer time
                    data_key = f'device_to_edge{edge_id + 1}'
                    if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                        transfer_time = 2.0  # Default transfer time
                    else:
                        data_size = task.data_sizes[data_key]
                        rate_key = f'device_to_edge{edge_id + 1}'
                        if not 'upload_rates' in globals() or rate_key not in upload_rates:
                            rate = 1.5  # Default rate
                        else:
                            rate = upload_rates[rate_key]

                        transfer_time = data_size / rate if rate > 0 else 0

                    # Calculate transfer timing considering channel availability
                    transfer_start = max(device_finish, self.device_to_edge_ready[edge_id])
                    transfer_finish = transfer_start + transfer_time

                    # Update device→edge channel availability
                    self.device_to_edge_ready[edge_id] = transfer_finish

                    pred_ready_time = transfer_finish

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Just need to wait for upload to complete
                    pred_ready_time = pred_task.FT_ws

                elif pred_task.execution_tier == ExecutionTier.EDGE:
                    if not pred_task.edge_assignment:
                        return float('inf')

                    pred_edge_id = pred_task.edge_assignment.edge_id - 1

                    if pred_edge_id == edge_id:
                        # Same edge node, no transfer needed
                        if pred_edge_id in pred_task.FT_edge:
                            pred_ready_time = pred_task.FT_edge[pred_edge_id]
                        else:
                            return float('inf')
                    else:
                        # Different edge node, need edge→edge transfer
                        if pred_edge_id not in pred_task.FT_edge:
                            return float('inf')

                        edge_finish = pred_task.FT_edge[pred_edge_id]

                        # Calculate transfer time
                        data_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                        if not hasattr(task, 'data_sizes') or data_key not in task.data_sizes:
                            transfer_time = 1.5  # Default transfer time
                        else:
                            data_size = task.data_sizes[data_key]
                            rate_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                            if not 'upload_rates' in globals() or rate_key not in upload_rates:
                                rate = 3.0  # Default rate
                            else:
                                rate = upload_rates[rate_key]

                            transfer_time = data_size / rate if rate > 0 else 0

                        # Calculate transfer timing considering channel availability
                        transfer_start = max(edge_finish, self.edge_to_edge_ready[pred_edge_id][edge_id])
                        transfer_finish = transfer_start + transfer_time

                        # Update edge→edge channel availability
                        self.edge_to_edge_ready[pred_edge_id][edge_id] = transfer_finish

                        pred_ready_time = transfer_finish
            else:
                # Original MCC task
                if hasattr(pred_task, 'is_core_task') and pred_task.is_core_task:
                    # Need device→edge transfer first
                    device_finish = pred_task.FT_l

                    # Use default transfer time
                    transfer_time = 2.0

                    # Calculate transfer timing considering channel availability
                    transfer_start = max(device_finish, self.device_to_edge_ready[edge_id])
                    transfer_finish = transfer_start + transfer_time

                    # Update device→edge channel availability
                    self.device_to_edge_ready[edge_id] = transfer_finish

                    pred_ready_time = transfer_finish
                else:
                    # Just need to wait for upload to complete
                    pred_ready_time = pred_task.FT_ws

            max_ready_time = max(max_ready_time, pred_ready_time)

        return max_ready_time

    def initialize_queue(self):
        """
                Initialize LIFO stack for linear-time scheduling.

                Returns:
                    Queue of initially ready tasks
                """
        # Create LIFO stack (implemented as deque)
        # A task is ready when:
        # 1. All predecessors are scheduled
        # 2. It's first in its sequence or the task before it is scheduled
        return deque(
            task for task in self.tasks
            if (
                    self.sequence_ready[task.id - 1] == 0
                    and all(pred_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED
                            for pred_task in task.pred_tasks)
            )
        )


class ThreeTierTaskScheduler:
    """
    Implements the initial scheduling algorithm with three-tier architecture:
    device (local cores), edge nodes, and cloud.
    """

    def __init__(self, tasks, num_cores=3, num_edge_nodes=2, edge_cores_per_node=2):
        """
        Initialize scheduler with tasks and resources across all tiers.
        Enhanced with improved channel contention handling.
        """
        self.tasks = tasks
        self.k = num_cores  # K cores from paper
        self.M = num_edge_nodes  # M edge nodes
        self.edge_cores = edge_cores_per_node  # Cores per edge node

        # Resource timing tracking for device cores
        self.core_earliest_ready = [0] * self.k  # When each core becomes available

        # Resource timing tracking for edge nodes
        # Format: edge_core_earliest_ready[edge_id][core_id]
        self.edge_core_earliest_ready = [[0] * self.edge_cores for _ in range(self.M)]

        # Resource timing tracking for communication channels
        # Device channels
        self.ws_ready = 0  # Next available time for RF sending channel (device → cloud)
        self.wr_ready = 0  # Next available time for RF receiving channel (cloud → device)

        # Device to edge channels (sending)
        self.device_to_edge_ready = [0] * self.M  # When each d→e channel is available

        # Edge to device channels (receiving)
        self.edge_to_device_ready = [0] * self.M  # When each e→d channel is available

        # Edge to cloud channels
        self.edge_to_cloud_ready = [0] * self.M  # When each e→c channel is available

        # Cloud to edge channels
        self.cloud_to_edge_ready = [0] * self.M  # When each c→e channel is available

        # Edge to edge channels (for edge-to-edge transfers)
        self.edge_to_edge_ready = [[0] * self.M for _ in range(self.M)]

        # Enhanced channel contention tracking
        self.channel_contention_queue = {
            'device_to_cloud': [],
            'cloud_to_device': []
        }

        # Initialize channel queues for all edge connections
        for edge_id in range(self.M):
            self.channel_contention_queue[f'device_to_edge{edge_id + 1}'] = []
            self.channel_contention_queue[f'edge{edge_id + 1}_to_device'] = []
            self.channel_contention_queue[f'edge{edge_id + 1}_to_cloud'] = []
            self.channel_contention_queue[f'cloud_to_edge{edge_id + 1}'] = []

            for other_edge in range(self.M):
                if edge_id != other_edge:
                    self.channel_contention_queue[f'edge{edge_id + 1}_to_edge{other_edge + 1}'] = []

        # Channel locks for atomic operations
        self.channel_locks = {}
        for edge_id in range(self.M):
            self.channel_locks[f'd2e_{edge_id}'] = 0  # device→edge
            self.channel_locks[f'e2d_{edge_id}'] = 0  # edge→device
            self.channel_locks[f'e2c_{edge_id}'] = 0  # edge→cloud
            self.channel_locks[f'c2e_{edge_id}'] = 0  # cloud→edge
            for other_edge in range(self.M):
                if edge_id != other_edge:
                    self.channel_locks[f'e2e_{edge_id}_{other_edge}'] = 0

        # Execution sequences for all resources
        # Indices 0..k-1: Local cores
        # Indices k..k+M*edge_cores-1: Edge cores
        # Index k+M*edge_cores: Cloud
        total_resources = self.k + self.M * self.edge_cores + 1
        self.sequences = [[] for _ in range(total_resources)]

    def get_edge_core_index(self, edge_id, core_id):
        """
        Convert edge_id and core_id to a single index in the sequences list.

        Args:
            edge_id: ID of the edge node (0-based)
            core_id: ID of the core within the edge node (0-based)

        Returns:
            Index in the sequences list
        """
        return self.k + edge_id * self.edge_cores + core_id

    def get_cloud_index(self):
        """
        Get the index for cloud in the sequences list.

        Returns:
            Index for cloud
        """
        return self.k + self.M * self.edge_cores

    def get_priority_ordered_tasks(self):
        """
        Orders tasks by priority scores.
        Same as original implementation.
        """
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)  # Higher priority first
        return [item[1] for item in task_priority_list]

    def classify_entry_tasks(self, priority_order):
        """
        Separates tasks into entry and non-entry tasks while maintaining priority order.
        Same as original implementation.
        """
        entry_tasks = []
        non_entry_tasks = []

        for task_id in priority_order:
            task = self.tasks[task_id - 1]

            if not task.pred_tasks:
                entry_tasks.append(task)
            else:
                non_entry_tasks.append(task)

        return entry_tasks, non_entry_tasks

    def calculate_local_ready_time(self, task):
        """Calculate ready time for local execution with improved precedence handling."""
        if not task.pred_tasks:
            return 0  # Entry task

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.is_scheduled != SchedulingState.SCHEDULED:
                return float('inf')  # Predecessor not scheduled yet

            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor executed locally - just wait for it to finish
                ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor executed in cloud - wait for download to complete
                ready_time = pred_task.FT_wr
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor on edge - wait for edge-to-device transfer
                if not pred_task.edge_assignment:
                    return float('inf')

                edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based

                # IMPORTANT: Make sure edge-to-device transfer is complete
                if edge_id not in pred_task.FT_edge_receive:
                    # Calculate it if not already done
                    download_key = f'edge{edge_id + 1}_to_device'
                    download_size = pred_task.data_sizes.get(download_key, 0)
                    download_rate = download_rates.get(f'edge{edge_id + 1}_to_device', 1.0)

                    if download_rate > 0:
                        download_time = download_size / download_rate
                    else:
                        download_time = 0

                    edge_finish = pred_task.FT_edge.get(edge_id, 0)
                    ready_time = edge_finish + download_time
                    pred_task.FT_edge_receive[edge_id] = ready_time
                else:
                    ready_time = pred_task.FT_edge_receive[edge_id]
            else:
                return float('inf')

            max_ready_time = max(max_ready_time, ready_time)

        return max_ready_time

    def calculate_cloud_upload_ready_time(self, task):
        """
        Calculate ready time for uploading to cloud (RT_i^ws).
        Implements equation (4) extended for three-tier architecture.

        Args:
            task: Task object

        Returns:
            Ready time for cloud upload
        """
        if not task.pred_tasks:
            return 0  # Entry task

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor executed locally
                ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor uploaded to cloud
                ready_time = pred_task.FT_ws
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor executed on edge
                if not pred_task.edge_assignment:
                    return float('inf')  # Invalid state

                edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based

                # Check if data was sent to device already
                if edge_id in pred_task.FT_edge_receive:
                    ready_time = pred_task.FT_edge_receive[edge_id]
                elif ('edge', 'cloud') in pred_task.FT_edge_send:
                    # Data was sent directly to cloud from edge
                    ready_time = pred_task.FT_edge_send[('edge', 'cloud')]
                else:
                    return float('inf')  # Invalid state
            else:
                return float('inf')  # Invalid state

            max_ready_time = max(max_ready_time, ready_time)

        # Also consider the wireless sending channel availability
        max_ready_time = max(max_ready_time, self.ws_ready)

        return max_ready_time

    def calculate_edge_ready_time(self, task, edge_id):
        """
        Calculate ready time for execution on edge node (RT_i^e,m).
        Based on the formula: RT_i^e,m = max_{v_j ∈ pred(v_i)} (FT_j(X_j) + Δ_j→m)

        Args:
            task: Task object
            edge_id: ID of the edge node (0-based)

        Returns:
            Ready time for edge execution
        """
        if not task.pred_tasks:
            return 0  # Entry task

        max_ready_time = 0

        for pred_task in task.pred_tasks:
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                # Predecessor on device, need data transfer to edge
                ready_time = pred_task.FT_l

                # Add transfer time from device to edge
                data_key = f'device_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate_key = f'device_to_edge{edge_id + 1}'
                rate = upload_rates.get(rate_key, 1.0)

                if rate > 0:
                    transfer_time = data_size / rate
                else:
                    transfer_time = 0

                ready_time += transfer_time

            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Predecessor in cloud, need data transfer to edge
                ready_time = pred_task.FT_c  # Cloud computation finish

                # Add transfer time from cloud to edge
                data_key = f'cloud_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate_key = f'cloud_to_edge{edge_id + 1}'
                rate = download_rates.get(rate_key, 1.0)

                if rate > 0:
                    transfer_time = data_size / rate
                else:
                    transfer_time = 0

                ready_time += transfer_time

            elif pred_task.execution_tier == ExecutionTier.EDGE:
                # Predecessor on edge
                if not pred_task.edge_assignment:
                    return float('inf')  # Invalid state

                pred_edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based

                # Check if predecessor is on the same edge
                if pred_edge_id == edge_id:
                    # Same edge, no transfer needed
                    if pred_edge_id in pred_task.FT_edge:
                        ready_time = pred_task.FT_edge[pred_edge_id]
                    else:
                        return float('inf')  # Invalid state
                else:
                    # Different edge, need edge-to-edge transfer
                    if pred_edge_id in pred_task.FT_edge:
                        base_ready_time = pred_task.FT_edge[pred_edge_id]

                        # Add transfer time from edge to edge
                        data_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                        data_size = task.data_sizes.get(data_key, 0)
                        rate_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                        rate = upload_rates.get(rate_key, 1.0)

                        if rate > 0:
                            transfer_time = data_size / rate
                        else:
                            transfer_time = 0

                        ready_time = base_ready_time + transfer_time
                    else:
                        return float('inf')  # Invalid state
            else:
                return float('inf')  # Invalid state

            max_ready_time = max(max_ready_time, ready_time)

        # Also consider the edge channel availability
        max_ready_time = max(max_ready_time, self.device_to_edge_ready[edge_id])

        return max_ready_time

    def identify_optimal_local_core(self, task, ready_time=None):
        """
        Find optimal local core assignment for a task to minimize finish time.
        Similar to original implementation but calculates ready time if not provided.
        """
        if ready_time is None:
            ready_time = self.calculate_local_ready_time(task)
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')

        for core in range(self.k):
            # Calculate earliest possible start time on this core
            start_time = max(ready_time, self.core_earliest_ready[core])

            # Calculate finish time
            finish_time = start_time + task.local_execution_times[core]

            # Keep track of core that gives earliest finish time
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core
                best_start_time = start_time

        return best_core, best_start_time, best_finish_time

    def identify_optimal_edge_core(self, task, edge_id, ready_time=None):
        """
        Find optimal core on a specific edge node for a task.

        Args:
            task: Task object
            edge_id: ID of the edge node (0-based)
            ready_time: Minimum start time for the task (optional)

        Returns:
            Tuple of (core_id, start_time, finish_time) for the optimal core
        """
        if ready_time is None:
            ready_time = self.calculate_edge_ready_time(task, edge_id)
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')

        for core_id in range(self.edge_cores):
            # Calculate earliest possible start time on this edge core
            start_time = max(ready_time, self.edge_core_earliest_ready[edge_id][core_id])

            # Calculate finish time
            exec_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
            if exec_time is None:
                # No execution time data for this core
                continue

            finish_time = start_time + exec_time

            # Keep track of core that gives earliest finish time
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core_id
                best_start_time = start_time

        return best_core, best_start_time, best_finish_time

    def identify_optimal_edge_node(self, task, ready_times_dict=None):
        """
        Find optimal edge node for a task across all available edge nodes.

        Args:
            task: Task object
            ready_times_dict: Dictionary mapping edge_id to ready_time (optional)

        Returns:
            Tuple of (edge_id, core_id, start_time, finish_time) for the optimal edge node
        """
        if ready_times_dict is None:
            # Calculate ready times for all edge nodes
            ready_times_dict = {}
            for edge_id in range(self.M):
                ready_times_dict[edge_id] = self.calculate_edge_ready_time(task, edge_id)
        best_finish_time = float('inf')
        best_edge_id = -1
        best_core_id = -1
        best_start_time = float('inf')

        # Try each edge node
        for edge_id in range(self.M):
            ready_time = ready_times_dict.get(edge_id, 0)
            if ready_time == float('inf'):
                continue  # Skip if ready time is infinite (not ready)

            # Find optimal core on this edge
            core_id, start_time, finish_time = self.identify_optimal_edge_core(
                task, edge_id, ready_time)

            if core_id >= 0 and finish_time < best_finish_time:
                best_finish_time = finish_time
                best_edge_id = edge_id
                best_core_id = core_id
                best_start_time = start_time

        return best_edge_id, best_core_id, best_start_time, best_finish_time

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """
        Assigns a task to a local core and updates timing information.
        Similar to original implementation but adds execution_tier.
        """
        # Set task finish time on local core
        task.FT_l = finish_time
        task.execution_finish_time = finish_time

        # Initialize execution start times array
        task.execution_unit_task_start_times = [-1] * (self.k + self.M * self.edge_cores + 1)

        # Record actual start time on assigned core
        task.execution_unit_task_start_times[core] = start_time

        # Update core availability
        self.core_earliest_ready[core] = finish_time

        # Set task assignment
        task.assignment = core
        task.execution_tier = ExecutionTier.DEVICE
        task.device_core = core
        task.edge_assignment = None

        # Clear edge and cloud finish times
        task.FT_edge = {}
        task.FT_edge_receive = {}
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        # Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # Add task to execution sequence for this core
        self.sequences[core].append(task.id)

        for pred_task in task.pred_tasks:
            pred_finish = float('inf')
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                pred_finish = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                pred_finish = pred_task.FT_wr if task.execution_tier == ExecutionTier.DEVICE else pred_task.FT_ws
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                edge_id = pred_task.edge_assignment.edge_id - 1
                if task.execution_tier == ExecutionTier.DEVICE:
                    pred_finish = pred_task.FT_edge_receive.get(edge_id, float('inf'))
                elif task.execution_tier == ExecutionTier.EDGE:
                    if task.edge_assignment.edge_id - 1 == edge_id:
                        # Same edge node
                        pred_finish = pred_task.FT_edge.get(edge_id, float('inf'))
                    else:
                        # Different edge node - need transfer time
                        pred_finish = float('inf')  # Calculate edge-to-edge transfer time

            task_start = 0
            if task.execution_tier == ExecutionTier.DEVICE:
                task_start = task.execution_unit_task_start_times[task.device_core]
            elif task.execution_tier == ExecutionTier.EDGE:
                task_start = task.execution_unit_task_start_times[
                    self.get_edge_core_index(task.edge_assignment.edge_id - 1, task.edge_assignment.core_id - 1)]

            if task_start < pred_finish:
                logger.error(
                    f"Task {task.id} starts at {task_start} before predecessor {pred_task.id} finishes at {pred_finish}")

    def schedule_on_edge(self, task, edge_id, core_id, start_time, finish_time):
        """
        Enhanced method to properly handle edge scheduling and all transfers.
        Includes precedence validation, edge-to-edge transfers, and execution path tracking.

        Args:
            edge_id: 0-based index of the edge node
            core_id: 0-based index of the core within the edge node
        """
        # STEP 1: Verify precedence constraints
        verified_ready_time = self.calculate_edge_ready_time(task, edge_id)
        if verified_ready_time > start_time:
            # Adjust the start time to respect precedence constraints
            logger.warning(
                f"Adjusting start time for task {task.id} from {start_time:.2f} to {verified_ready_time:.2f}")
            start_time = verified_ready_time
            # Recalculate finish time
            exec_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
            if exec_time is None:
                logger.error(f"No execution time data for task {task.id} on edge {edge_id + 1}, core {core_id + 1}")
                return False
            finish_time = start_time + exec_time

        # STEP 2: Check for edge-to-edge migration
        original_edge_id = None
        is_migration = False
        if task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            original_edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based
            if original_edge_id != edge_id:
                is_migration = True
                # Calculate edge-to-edge transfer time
                data_key = f'edge{original_edge_id + 1}_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate = upload_rates.get(data_key, 1.0)
                transfer_time = data_size / rate if rate > 0 else 0

                # Calculate transfer timing
                transfer_start = max(
                    task.FT_edge.get(original_edge_id, 0),
                    self.edge_to_edge_ready[original_edge_id][edge_id]
                )
                transfer_finish = transfer_start + transfer_time

                # Update edge-to-edge channel availability
                self.edge_to_edge_ready[original_edge_id][edge_id] = transfer_finish

                # Adjust start time if needed to account for migration time
                if transfer_finish > start_time:
                    logger.warning(
                        f"Adjusting start time for task {task.id} migration from edge {original_edge_id + 1} to {edge_id + 1}")
                    start_time = transfer_finish
                    finish_time = start_time + exec_time

        # STEP 3: Handle device-to-edge transfer (if not migration)
        if not is_migration:
            # Calculate device→edge upload time and channel reservation
            data_key = f'device_to_edge{edge_id + 1}'
            data_size = task.data_sizes.get(data_key, 0)
            rate = upload_rates.get(data_key, 1.0)

            upload_time = data_size / rate if rate > 0 else 0

            # Calculate actual upload timing considering channel availability
            upload_start = max(
                # When data is ready to send (account for predecessor finish time)
                0,  # For entry tasks or migration
                # When channel becomes available
                self.device_to_edge_ready[edge_id]
            )
            upload_finish = upload_start + upload_time

            # Adjust task start time if upload delay affects it
            actual_start = max(start_time, upload_finish)
            if actual_start > start_time:
                # Recalculate finish time
                exec_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
                finish_time = actual_start + exec_time
                start_time = actual_start

            # Update device→edge channel availability
            self.device_to_edge_ready[edge_id] = upload_finish

            # Add to channel contention queue
            channel_key = f'device_to_edge{edge_id + 1}'
            self.channel_contention_queue[channel_key].append({
                'task_id': task.id,
                'start': upload_start,
                'finish': upload_finish
            })

        # STEP 4: Initialize execution timing information
        task.execution_unit_task_start_times = [-1] * (self.k + self.M * self.edge_cores + 1)

        # STEP 5: Set task finish time on edge
        # Use both 0-based and 1-based indices for edge_id to ensure consistency
        task.FT_edge[edge_id] = finish_time  # 0-based index for internal tracking
        task.FT_edge[edge_id + 1] = finish_time  # 1-based index for validation
        task.execution_finish_time = finish_time

        # STEP 6: Calculate and handle edge→device results transfer
        download_key = f'edge{edge_id + 1}_to_device'
        download_size = task.data_sizes.get(download_key, 0)
        download_rate = download_rates.get(download_key, 1.0)

        download_time = download_size / download_rate if download_rate > 0 else 0

        # Calculate download timing considering channel availability
        download_start = max(finish_time, self.edge_to_device_ready[edge_id])
        download_finish = download_start + download_time

        # Update edge→device channel availability
        self.edge_to_device_ready[edge_id] = download_finish

        # Record results received time at device - use both indices for consistency
        task.FT_edge_receive[edge_id] = download_finish  # 0-based index
        task.FT_edge_receive[edge_id + 1] = download_finish  # 1-based index

        # Add to channel contention queue
        channel_key = f'edge{edge_id + 1}_to_device'
        self.channel_contention_queue[channel_key].append({
            'task_id': task.id,
            'start': download_start,
            'finish': download_finish
        })

        # STEP 7: Record execution information
        # Record actual start time on assigned edge core
        sequence_idx = self.get_edge_core_index(edge_id, core_id)
        task.execution_unit_task_start_times[sequence_idx] = start_time

        # Update edge core availability
        self.edge_core_earliest_ready[edge_id][core_id] = finish_time

        # STEP 8: Set task assignment
        task.assignment = sequence_idx
        task.execution_tier = ExecutionTier.EDGE
        task.device_core = -1
        task.edge_assignment = EdgeAssignment(edge_id=edge_id + 1, core_id=core_id + 1)  # Convert to 1-based

        # STEP 9: Update execution path for energy calculations
        if not task.execution_path:
            # First assignment
            task.execution_path = [(ExecutionTier.DEVICE, None),
                                   (ExecutionTier.EDGE, (edge_id + 1, core_id + 1))]
        else:
            # Task is migrating
            task.execution_path.append((ExecutionTier.EDGE, (edge_id + 1, core_id + 1)))

        # STEP 10: Clear device and cloud finish times
        task.FT_l = 0
        task.FT_ws = 0
        task.FT_c = 0
        task.FT_wr = 0

        # STEP 11: Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # STEP 12: Add task to execution sequence for this edge core
        self.sequences[sequence_idx].append(task.id)

        # STEP 13: Final validation - check for precedence violations
        for pred_task in task.pred_tasks:
            pred_ready_time = float('-inf')

            if pred_task.execution_tier == ExecutionTier.DEVICE:
                pred_ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                if not pred_task.edge_assignment:
                    logger.warning(f"Missing edge assignment for predecessor {pred_task.id}")
                    continue

                pred_edge_id = pred_task.edge_assignment.edge_id - 1
                if pred_edge_id == edge_id:
                    pred_ready_time = pred_task.FT_edge.get(pred_edge_id, 0)
                else:
                    # Different edge - need transfer time
                    pred_finish = pred_task.FT_edge.get(pred_edge_id, 0)
                    # Calculate edge-to-edge transfer
                    data_key = f'edge{pred_edge_id + 1}_to_edge{edge_id + 1}'
                    data_size = task.data_sizes.get(data_key, 0)
                    rate = upload_rates.get(data_key, 1.0)
                    transfer_time = data_size / rate if rate > 0 else 0
                    pred_ready_time = pred_finish + transfer_time
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                # Cloud to edge transfer
                pred_ready_time = pred_task.FT_c + task.calculate_data_transfer_time(
                    ExecutionTier.CLOUD, ExecutionTier.EDGE,
                    upload_rates, download_rates,
                    target_location=(edge_id + 1, 0)
                )

            if start_time < pred_ready_time:
                logger.warning(
                    f"Precedence violation: Task {task.id} starts at {start_time:.2f} before predecessor {pred_task.id} ready at {pred_ready_time:.2f}")

        return True

    def schedule_entry_tasks(self, entry_tasks):
        """
        Schedules tasks with no predecessors (entry tasks) across the three-tier architecture.
        Entry tasks can start immediately but need to be scheduled optimally across
        all available resources.

        Args:
            entry_tasks: List of Task objects with no predecessors
        """
        # First pass: Group tasks by their primary assignment tier
        device_tasks = []
        edge_tasks = []
        cloud_tasks = []

        for task in entry_tasks:
            if task.execution_tier == ExecutionTier.DEVICE:
                device_tasks.append(task)
            elif task.execution_tier == ExecutionTier.EDGE:
                edge_tasks.append(task)
            elif task.execution_tier == ExecutionTier.CLOUD:
                cloud_tasks.append(task)
            else:
                logger.warning(f"Task {task.id} has unknown execution tier. Defaulting to device.")
                device_tasks.append(task)

        # Second pass: Schedule tasks on each tier in order:
        # 1. Device (local cores)
        # 2. Edge nodes
        # 3. Cloud
        # This scheduling order helps optimize communication channel usage

        # 1. First schedule tasks on local device cores
        for task in device_tasks:
            # Find optimal local core and calculate timing
            core, start_time, finish_time = self.identify_optimal_local_core(task, 0)  # Entry tasks can start at time 0

            if core < 0 or finish_time == float('inf'):
                # No suitable local core found, try edge or cloud instead
                logger.info(f"No suitable local core for task {task.id}, will try edge or cloud")
                # We'll handle this in fallback section
                if self.M > 0:  # If we have edge nodes
                    edge_tasks.append(task)
                else:
                    cloud_tasks.append(task)
            else:
                # Schedule on the selected local core
                self.schedule_on_local_core(task, core, start_time, finish_time)
                logger.info(f"Scheduled entry task {task.id} on local core {core}")

        # 2. Next schedule tasks on edge nodes
        edge_fallbacks = []  # Tasks that can't be scheduled on edge (will go to cloud)

        for task in edge_tasks:
            # Calculate ready times for all edge nodes (0 for entry tasks)
            ready_times = {edge_id: 0 for edge_id in range(self.M)}

            # Find optimal edge node and core
            edge_id, core_id, start_time, finish_time = self.identify_optimal_edge_node(task, ready_times)

            if edge_id < 0 or finish_time == float('inf'):
                # No suitable edge node found, will try cloud
                logger.info(f"No suitable edge node for task {task.id}, will try cloud")
                edge_fallbacks.append(task)
            else:
                # Schedule on the selected edge node and core
                self.schedule_on_edge(task, edge_id, core_id, start_time, finish_time)
                logger.info(f"Scheduled entry task {task.id} on edge node {edge_id}, core {core_id}")

        # Add edge fallbacks to cloud tasks
        cloud_tasks.extend(edge_fallbacks)

        # 3. Finally schedule tasks on cloud
        cloud_fallbacks = []  # Tasks that can't be scheduled on cloud

        for task in cloud_tasks:
            # Set task's wireless sending ready time
            task.RT_ws = self.ws_ready  # Entry tasks can start sending as soon as channel is free

            # Calculate cloud execution phases timing
            timing = self.calculate_cloud_phases_timing(task)

            if timing[0] == float('inf'):
                # Invalid timing, can't schedule on cloud
                logger.warning(f"Cannot schedule task {task.id} on cloud, invalid timing")
                cloud_fallbacks.append(task)
                continue

            # Schedule task on cloud
            self.schedule_on_cloud(task, *timing)
            logger.info(f"Scheduled entry task {task.id} on cloud")

        # Handle fallbacks - any tasks that couldn't be scheduled anywhere
        # This should be rare but we handle it to make the scheduler robust
        for task in cloud_fallbacks:
            # Last attempt: Try to schedule on any available resource

            # Try local cores
            core, start_time, finish_time = self.identify_optimal_local_core(task, 0)
            if core >= 0 and finish_time < float('inf'):
                self.schedule_on_local_core(task, core, start_time, finish_time)
                logger.info(f"Fallback: Scheduled task {task.id} on local core {core}")
                continue

            # Try edge nodes
            ready_times = {edge_id: 0 for edge_id in range(self.M)}
            edge_id, core_id, start_time, finish_time = self.identify_optimal_edge_node(task, ready_times)
            if edge_id >= 0 and finish_time < float('inf'):
                self.schedule_on_edge(task, edge_id, core_id, start_time, finish_time)
                logger.info(f"Fallback: Scheduled task {task.id} on edge node {edge_id}, core {core_id}")
                continue

            # If we reach here, task cannot be scheduled
            logger.error(f"Task {task.id} could not be scheduled on any resource")
            # Mark task as unscheduled
            task.is_scheduled = SchedulingState.UNSCHEDULED

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready,
                          receive_finish):
        """
        Enhanced method for cloud execution scheduling with improved precedence tracking.
        Includes execution path tracking and channel contention resolution.
        """
        # STEP 1: Verify precedence constraints
        verified_ready_time = self.calculate_cloud_upload_ready_time(task)
        if verified_ready_time > send_ready:
            # Adjust the start time to respect precedence constraints
            logger.warning(f"Adjusting send time for task {task.id} from {send_ready:.2f} to {verified_ready_time:.2f}")

            # Recalculate all timing parameters
            send_ready = verified_ready_time
            send_finish = send_ready + task.cloud_execution_times[0]
            cloud_ready = send_finish
            cloud_finish = cloud_ready + task.cloud_execution_times[1]
            receive_ready = cloud_finish

            # Determine target for results based on execution tier
            if task.execution_tier == ExecutionTier.DEVICE:
                # Results return to device
                receive_finish = max(self.wr_ready, receive_ready) + task.cloud_execution_times[2]
                self.wr_ready = receive_finish  # Update receiving channel availability
            elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
                # Results return to edge node
                edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based

                # Calculate cloud-to-edge download time
                data_key = f'cloud_to_edge{edge_id + 1}'
                data_size = task.data_sizes.get(data_key, 0)
                rate_key = f'cloud_to_edge{edge_id + 1}'
                rate = download_rates.get(rate_key, 1.0)

                if rate > 0:
                    download_time = data_size / rate
                else:
                    download_time = 0

                receive_finish = max(self.cloud_to_edge_ready[edge_id], receive_ready) + download_time
                self.cloud_to_edge_ready[edge_id] = receive_finish

        # STEP 2: Set timing parameters for three-phase cloud execution
        task.RT_ws = send_ready  # When we can start sending
        task.FT_ws = send_finish  # When sending completes

        task.RT_c = cloud_ready  # When cloud can start
        task.FT_c = cloud_finish  # When cloud computation ends

        task.RT_wr = receive_ready  # When results are ready
        task.FT_wr = receive_finish  # When results are received

        # STEP 3: Set overall execution finish time
        task.execution_finish_time = receive_finish

        # STEP 4: Track channel usage
        # Device to cloud channel (upload)
        if task.execution_tier == ExecutionTier.DEVICE:
            # Update channel contention tracking
            self.ws_ready = send_finish  # Update sending channel
            self.channel_contention_queue['device_to_cloud'].append({
                'task_id': task.id,
                'start': send_ready,
                'finish': send_finish
            })

            # Cloud to device channel (download)
            self.wr_ready = receive_finish  # Update receiving channel
            self.channel_contention_queue['cloud_to_device'].append({
                'task_id': task.id,
                'start': receive_ready,
                'finish': receive_finish
            })
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            # Edge to cloud channel
            edge_id = task.edge_assignment.edge_id - 1
            channel_key = f'edge{edge_id + 1}_to_cloud'
            self.channel_contention_queue[channel_key].append({
                'task_id': task.id,
                'start': send_ready,
                'finish': send_finish
            })

            # Cloud to edge channel
            channel_key = f'cloud_to_edge{edge_id + 1}'
            self.channel_contention_queue[channel_key].append({
                'task_id': task.id,
                'start': receive_ready,
                'finish': receive_finish
            })

        # STEP 5: Initialize execution start times array
        task.execution_unit_task_start_times = [-1] * (self.k + self.M * self.edge_cores + 1)

        # STEP 6: Record cloud execution start time
        cloud_idx = self.get_cloud_index()
        task.execution_unit_task_start_times[cloud_idx] = send_ready

        # STEP 7: Set task assignment
        task.assignment = cloud_idx
        task.execution_tier = ExecutionTier.CLOUD
        task.device_core = -1
        task.edge_assignment = None

        # STEP 8: Update execution path for energy calculations
        previous_tier = None
        previous_location = None

        if task.execution_path:
            if len(task.execution_path) > 0:
                previous_tier, previous_location = task.execution_path[-1]

        if not task.execution_path:
            # First assignment - assume from device
            task.execution_path = [(ExecutionTier.DEVICE, None), (ExecutionTier.CLOUD, None)]
        else:
            # Task is migrating
            task.execution_path.append((ExecutionTier.CLOUD, None))

        # STEP 9: Clear local and edge finish times
        task.FT_l = 0
        task.FT_edge = {}
        task.FT_edge_receive = {}

        # STEP 10: Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # STEP 11: Add to cloud execution sequence
        self.sequences[cloud_idx].append(task.id)

        # STEP 12: Final validation - check for precedence violations
        for pred_task in task.pred_tasks:
            pred_ready_time = float('-inf')

            if pred_task.execution_tier == ExecutionTier.DEVICE:
                pred_ready_time = pred_task.FT_l
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                pred_ready_time = pred_task.FT_ws  # For cloud-to-cloud, wait for upload to complete
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                if not pred_task.edge_assignment:
                    logger.warning(f"Missing edge assignment for predecessor {pred_task.id}")
                    continue

                edge_id = pred_task.edge_assignment.edge_id - 1

                # Check if data was sent directly to cloud
                cloud_key = ('edge', 'cloud')
                if cloud_key in pred_task.FT_edge_send:
                    pred_ready_time = pred_task.FT_edge_send[cloud_key]
                else:
                    # Otherwise need to wait for data to return to device
                    if edge_id in pred_task.FT_edge_receive:
                        pred_ready_time = pred_task.FT_edge_receive[edge_id]
                    else:
                        pred_ready_time = pred_task.FT_edge.get(edge_id, 0) + pred_task.calculate_data_transfer_time(
                            ExecutionTier.EDGE, ExecutionTier.DEVICE,
                            upload_rates, download_rates,
                            source_location=(edge_id + 1, 0)
                        )

            if send_ready < pred_ready_time:
                logger.warning(
                    f"Precedence violation: Task {task.id} starts cloud upload at {send_ready:.2f} before predecessor {pred_task.id} ready at {pred_ready_time:.2f}")

        return True

    def calculate_cloud_phases_timing(self, task):
        """
        Calculates timing for the three-phase cloud execution model.
        Extended for three-tier architecture to include edge-to-cloud transfers.

        Args:
            task: Task object

        Returns:
            Tuple of (send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish)
        """
        # Phase 1: Sending Phase (Upload)
        # Determine source tier
        original_tier = task.execution_tier

        if original_tier == ExecutionTier.DEVICE:
            # From device to cloud (as in original model)
            send_ready = task.RT_ws
            send_finish = send_ready + task.cloud_execution_times[0]
            self.ws_ready = send_finish  # Update sending channel availability
        elif original_tier == ExecutionTier.EDGE:
            # From edge to cloud
            if not task.edge_assignment:
                logger.error(f"Edge task {task.id} has no edge assignment for cloud offloading")
                return (float('inf'),) * 6  # Invalid state

            edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based

            # Calculate edge-to-cloud upload time
            data_key = f'edge{edge_id + 1}_to_cloud'
            data_size = task.data_sizes.get(data_key, 0)
            rate_key = f'edge{edge_id + 1}_to_cloud'
            rate = upload_rates.get(rate_key, 1.0)

            if rate > 0:
                upload_time = data_size / rate
            else:
                upload_time = 0

            send_ready = max(task.FT_edge.get(edge_id, 0), self.edge_to_cloud_ready[edge_id])
            send_finish = send_ready + upload_time

            # Update edge-to-cloud channel availability
            self.edge_to_cloud_ready[edge_id] = send_finish

            # Record send operation in task
            edge_send_key = ('edge', 'cloud')
            task.FT_edge_send[edge_send_key] = send_finish
        else:
            logger.error(f"Invalid execution tier {original_tier} for task {task.id}")
            return (float('inf'),) * 6  # Invalid state

        # Phase 2: Cloud Computing Phase
        cloud_ready = send_finish
        cloud_finish = cloud_ready + task.cloud_execution_times[1]

        # Phase 3: Receiving Phase (Download)
        receive_ready = cloud_finish

        # Determine target for results - send back to original tier
        if original_tier == ExecutionTier.DEVICE:
            # Results return to device
            receive_finish = max(self.wr_ready, receive_ready) + task.cloud_execution_times[2]
            self.wr_ready = receive_finish  # Update receiving channel availability
        elif original_tier == ExecutionTier.EDGE:
            # Results return to edge node
            edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based

            # Calculate cloud-to-edge download time
            data_key = f'cloud_to_edge{edge_id + 1}'
            data_size = task.data_sizes.get(data_key, 0)
            rate_key = f'cloud_to_edge{edge_id + 1}'
            rate = download_rates.get(rate_key, 1.0)

            if rate > 0:
                download_time = data_size / rate
            else:
                download_time = 0

            receive_finish = max(self.cloud_to_edge_ready[edge_id], receive_ready) + download_time

            # Update cloud-to-edge channel availability
            self.cloud_to_edge_ready[edge_id] = receive_finish
        else:
            receive_finish = receive_ready  # No download needed for pure cloud tasks

        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """
        Schedules tasks with predecessors (non-entry tasks) across the three-tier architecture.
        These tasks must respect precedence constraints and can only start when all their
        predecessors have completed.

        Args:
            non_entry_tasks: List of Task objects with predecessors
        """
        for task in non_entry_tasks:
            logger.info(f"Scheduling non-entry task {task.id}")

            # 1. Calculate ready times for all execution options
            # Ready time on local device cores
            device_ready_time = self.calculate_local_ready_time(task)

            # Ready times on all edge nodes
            edge_ready_times = {}
            for edge_id in range(self.M):
                edge_ready_times[edge_id] = self.calculate_edge_ready_time(task, edge_id)

            # Ready time for cloud upload
            cloud_ready_time = self.calculate_cloud_upload_ready_time(task)

            # Check if ready times are valid (could be invalid due to unscheduled predecessors)
            if device_ready_time == float('inf') and all(
                    rt == float('inf') for rt in edge_ready_times.values()) and cloud_ready_time == float('inf'):
                logger.error(f"Task {task.id} has no valid ready time - predecessors not properly scheduled")
                task.is_scheduled = SchedulingState.UNSCHEDULED
                continue

            # 2. Evaluate all execution options based on finish times

            # Option 1: Local execution (on device cores)
            local_finish = float('inf')
            if device_ready_time < float('inf'):
                # Get optimal local core
                core, local_start, local_finish = self.identify_optimal_local_core(task, device_ready_time)
                # If no valid local core is found, this remains infinity

            # Option 2: Edge execution
            edge_finish = float('inf')
            edge_id = -1
            core_id = -1
            edge_start = 0

            valid_edge_nodes = {eid: rt for eid, rt in edge_ready_times.items() if rt < float('inf')}
            if valid_edge_nodes:
                # Find optimal edge node and core
                edge_id, core_id, edge_start, edge_finish = self.identify_optimal_edge_node(task, valid_edge_nodes)
                # If no valid edge node is found, this remains infinity

            # Option 3: Cloud execution
            cloud_finish = float('inf')
            if cloud_ready_time < float('inf'):
                # Set task ready time for cloud upload
                task.RT_ws = cloud_ready_time

                # Calculate cloud execution phases timing
                timing = self.calculate_cloud_phases_timing(task)

                if timing[0] < float('inf'):
                    # Extract cloud finish time (when results are received)
                    cloud_finish = timing[-1]  # receive_finish

            # 3. Choose execution option with earliest finish time
            logger.info(
                f"Task {task.id} finish times - Local: {local_finish}, Edge: {edge_finish}, Cloud: {cloud_finish}")

            if local_finish <= edge_finish and local_finish <= cloud_finish and local_finish < float('inf'):
                # Local execution is fastest
                self.schedule_on_local_core(task, core, local_start, local_finish)
                logger.info(f"Scheduled task {task.id} on local core {core}")

            elif edge_finish <= local_finish and edge_finish <= cloud_finish and edge_finish < float('inf'):
                # Edge execution is fastest
                self.schedule_on_edge(task, edge_id, core_id, edge_start, edge_finish)
                logger.info(f"Scheduled task {task.id} on edge node {edge_id}, core {core_id}")

            elif cloud_finish < float('inf'):
                # Cloud execution is fastest
                task.RT_ws = cloud_ready_time
                timing = self.calculate_cloud_phases_timing(task)
                self.schedule_on_cloud(task, *timing)
                logger.info(f"Scheduled task {task.id} on cloud")

            else:
                # No valid option found - this is an error case
                logger.error(f"Task {task.id} has no valid execution option")
                task.is_scheduled = SchedulingState.UNSCHEDULED


class Task:
    def __init__(self, id, pred_task=None, succ_task=None):
        # Basic task graph structure
        self.id = id  # Task identifier v_i in DAG G=(V,E)
        self.succ_tasks = succ_task or []  # succ(v_i): Immediate successors
        self.pred_tasks = pred_task

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

    def calculate_edge_ready_time(self, task, edge_id):
        """Fixed calculation of ready time for edge execution."""
        if not task.pred_tasks:
            return 0  # Entry task

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
                channel_available = self.device_to_edge_ready[edge_id]
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
                    channel_available = self.edge_to_edge_ready[pred_edge_id][edge_id]
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
                channel_available = self.cloud_to_edge_ready[edge_id]
                transfer_start = max(pred_finish, channel_available)
                transfer_time = data_size / rate if rate > 0 else 0
                ready_time = transfer_start + transfer_time
            else:
                return float('inf')

            max_ready_time = max(max_ready_time, ready_time)

        # Also consider the device-to-edge channel availability
        max_ready_time = max(max_ready_time, self.device_to_edge_ready[edge_id])

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


def primary_assignment(tasks, edge_nodes=None):
    if edge_nodes is None:
        edge_nodes = M  # Use global edge nodes count

    for task in tasks:
        # Compute minimum local execution time
        t_l_min = min(task.local_execution_times) if task.local_execution_times else float('inf')

        # Compute minimum edge execution time (using available edge data)
        t_edge_min = float('inf')
        best_edge_id = -1
        best_core_id = -1
        for edge_id in range(1, edge_nodes + 1):
            for core_id in range(1, 3):  # Assuming 2 cores per edge node
                key = (edge_id, core_id)
                if key in task.edge_execution_times:
                    edge_time = task.edge_execution_times[key]
                    if edge_time < t_edge_min:
                        t_edge_min = edge_time
                        best_edge_id = edge_id
                        best_core_id = core_id

        # Use the true (unadjusted) cloud time
        t_re = sum(task.cloud_execution_times) if task.cloud_execution_times else float('inf')
        # Print for debugging
        print(f"Task {task.id}: Device={t_l_min}, Edge={t_edge_min}, Cloud={t_re}")

        # Choose the execution tier based solely on the three raw times
        if t_l_min <= t_edge_min and t_l_min <= t_re:
            task.execution_tier = ExecutionTier.DEVICE
        elif t_edge_min <= t_l_min and t_edge_min <= t_re:
            task.execution_tier = ExecutionTier.EDGE
            task.edge_assignment = EdgeAssignment(edge_id=best_edge_id, core_id=best_core_id)
        else:
            task.execution_tier = ExecutionTier.CLOUD

        if task.execution_tier == ExecutionTier.CLOUD:
            logger.info(f"Initializing cloud task {task.id}")
            # Use the task's cloud_execution_times as-is (no bias)
            task.cloud_execution_times = task.cloud_execution_times or [3, 1, 1]
            # Instead of fixed values, immediately update cloud timing based on predecessors
            update_cloud_timing(task, tasks)
            # Clear other tier assignments
            task.device_core = -1
            task.edge_assignment = None
            task.FT_l = 0
    return


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


def validate_three_tier_schedule(tasks: List[Any],
                                 sequences: List[List[int]],
                                 num_device_cores: int = 3,
                                 num_edge_nodes: int = 2,
                                 num_edge_cores_per_node: int = 2) -> Dict[str, Any]:
    """
            Perform comprehensive validation of a three-tier schedule.

            Args:
                tasks: List of all tasks
                sequences: Current sequences for all execution units
                num_device_cores: Number of device cores
                num_edge_nodes: Number of edge nodes
                num_edge_cores_per_node: Number of cores per edge node

            Returns:
                Dictionary with validation results
            """
    # Initialize validation results
    validation = {
        "valid": True,
        "dependency_violations": 0,
        "sequence_violations": 0,
        "resource_violations": 0,
        "tier_violations": 0,
        "unscheduled_tasks": 0,
        "issues": []
    }

    # Check if all tasks are scheduled
    for task in tasks:
        # Check if task has a valid assignment
        if hasattr(task, 'execution_tier'):
            # Three-tier task
            scheduled = False

            if task.execution_tier == ExecutionTier.DEVICE:
                if hasattr(task, 'device_core') and task.device_core >= 0:
                    scheduled = True
            elif task.execution_tier == ExecutionTier.EDGE:
                if hasattr(task, 'edge_assignment') and task.edge_assignment:
                    scheduled = True
            elif task.execution_tier == ExecutionTier.CLOUD:
                scheduled = True

            if not scheduled:
                validation["unscheduled_tasks"] += 1
                validation["issues"].append(f"Task {getattr(task, 'id', '?')} has no valid assignment")
        else:
            # Original MCC task
            if not hasattr(task, 'assignment') or task.assignment < 0:
                validation["unscheduled_tasks"] += 1
                validation["issues"].append(f"Task {getattr(task, 'id', '?')} has no valid assignment")

    # Check task dependencies
    if not validate_task_dependencies(tasks):
        validation["valid"] = False
        validation["dependency_violations"] += 1
        validation["issues"].append("Task dependency constraints are violated")

    # Check sequence consistency
    task_counts = {}
    for task in tasks:
        task_id = getattr(task, 'id', None)
        if task_id is not None:
            if task_id in task_counts:
                task_counts[task_id] += 1
            else:
                task_counts[task_id] = 1

    sequence_task_counts = {}
    for sequence in sequences:
        for task_id in sequence:
            if task_id in sequence_task_counts:
                sequence_task_counts[task_id] += 1
            else:
                sequence_task_counts[task_id] = 1

    # Check for tasks in multiple sequences
    for task_id, count in sequence_task_counts.items():
        if count > 1:
            validation["sequence_violations"] += 1
            validation["issues"].append(f"Task {task_id} appears in {count} sequences")

    # Check for missing tasks in sequences
    for task_id, count in task_counts.items():
        if task_id not in sequence_task_counts:
            validation["sequence_violations"] += 1
            validation["issues"].append(f"Task {task_id} is not in any sequence")

    # Check resource assignments
    for task in tasks:
        if hasattr(task, 'execution_tier'):
            # Three-tier task
            if task.execution_tier == ExecutionTier.DEVICE:
                if hasattr(task, 'device_core'):
                    if task.device_core < 0 or task.device_core >= num_device_cores:
                        validation["resource_violations"] += 1
                        validation["issues"].append(
                            f"Task {getattr(task, 'id', '?')} assigned to invalid device core {task.device_core}")
            elif task.execution_tier == ExecutionTier.EDGE:
                if hasattr(task, 'edge_assignment') and task.edge_assignment:
                    edge_id = task.edge_assignment.edge_id
                    core_id = task.edge_assignment.core_id

                    if edge_id < 1 or edge_id > num_edge_nodes:
                        validation["resource_violations"] += 1
                        validation["issues"].append(
                            f"Task {getattr(task, 'id', '?')} assigned to invalid edge node {edge_id}")

                    if core_id < 1 or core_id > num_edge_cores_per_node:
                        validation["resource_violations"] += 1
                        validation["issues"].append(
                            f"Task {getattr(task, 'id', '?')} assigned to invalid edge core {core_id}")
        else:
            # Original MCC task
            if hasattr(task, 'assignment'):
                if hasattr(task, 'is_core_task') and task.is_core_task:
                    if task.assignment < 0 or task.assignment >= num_device_cores:
                        validation["resource_violations"] += 1
                        validation["issues"].append(
                            f"Task {getattr(task, 'id', '?')} assigned to invalid device core {task.assignment}")

    # Check for tier consistency
    for task in tasks:
        if hasattr(task, 'execution_tier'):
            # Three-tier task
            if task.execution_tier == ExecutionTier.DEVICE:
                if hasattr(task, 'edge_assignment') and task.edge_assignment:
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is a device task but has edge assignment")

                if getattr(task, 'FT_l', 0) <= 0:
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is a device task but has no local finish time")

            elif task.execution_tier == ExecutionTier.EDGE:
                if not hasattr(task, 'edge_assignment') or not task.edge_assignment:
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is an edge task but has no edge assignment")

                if not hasattr(task, 'FT_edge') or not task.FT_edge:
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is an edge task but has no edge finish time")

            elif task.execution_tier == ExecutionTier.CLOUD:
                if getattr(task, 'FT_ws', 0) <= 0 or getattr(task, 'FT_c', 0) <= 0 or getattr(task, 'FT_wr',
                                                                                              0) <= 0:
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is a cloud task but has incomplete cloud timing")
        else:
            # Original MCC task
            if hasattr(task, 'is_core_task'):
                if task.is_core_task and getattr(task, 'FT_l', 0) <= 0:
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is a local task but has no local finish time")

                if not task.is_core_task and (
                        getattr(task, 'FT_ws', 0) <= 0 or getattr(task, 'FT_c', 0) <= 0 or getattr(task,
                                                                                                   'FT_wr',
                                                                                                   0) <= 0):
                    validation["tier_violations"] += 1
                    validation["issues"].append(
                        f"Task {getattr(task, 'id', '?')} is a cloud task but has incomplete cloud timing")

    # Update overall validation status
    validation["valid"] = (validation["dependency_violations"] == 0 and
                           validation["sequence_violations"] == 0 and
                           validation["resource_violations"] == 0 and
                           validation["tier_violations"] == 0 and
                           validation["unscheduled_tasks"] == 0)

    return validation


def validate_task_dependencies(tasks, epsilon=1e-9):
    """
    Validate that task dependency constraints are properly maintained
    across the three-tier architecture, with tolerance for floating-point imprecision.

    Parameters:
    - tasks: List of Task objects
    - epsilon: Small tolerance for floating-point comparisons (default: 1e-9)

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
                    # Apply floating-point tolerance
                    if pred_task.FT_l - task_start_time > epsilon:
                        violations.append({
                            'type': 'Device-Device Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts at {task_start_time} before predecessor {pred_task.id} finishes at {pred_task.FT_l}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # Cloud predecessor must return results before task starts
                    # Apply floating-point tolerance
                    if pred_task.FT_wr - task_start_time > epsilon:
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

                    edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based for internal lookup

                    # Check both 0-based and 1-based indexing for backward compatibility
                    has_receive_time = (edge_id in pred_task.FT_edge_receive or
                                        (edge_id + 1) in pred_task.FT_edge_receive)

                    if not has_receive_time:
                        violations.append({
                            'type': 'Missing Edge Receive Time',
                            'task': pred_task.id,
                            'edge': edge_id + 1,  # Report as 1-based for user clarity
                            'detail': f"Edge task {pred_task.id} has no receive time for edge {edge_id + 1}"
                        })
                        continue

                    # Get receive time (try both indices)
                    receive_time = pred_task.FT_edge_receive.get(edge_id,
                                                                 pred_task.FT_edge_receive.get(edge_id + 1,
                                                                                               float('inf')))

                    # Apply floating-point tolerance
                    if receive_time - task_start_time > epsilon:
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
                    # Apply floating-point tolerance
                    if pred_task.FT_l - upload_start_time > epsilon:
                        violations.append({
                            'type': 'Device-Cloud Dependency Violation',
                            'task': task.id,
                            'predecessor': pred_task.id,
                            'detail': f"Task {task.id} starts uploading at {upload_start_time} before predecessor {pred_task.id} finishes at {pred_task.FT_l}"
                        })

                elif pred_task.execution_tier == ExecutionTier.CLOUD:
                    # For cloud-to-cloud, only need to wait for upload to complete
                    # Apply floating-point tolerance
                    if pred_task.FT_ws - upload_start_time > epsilon:
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
                        # Apply floating-point tolerance
                        if cloud_send_time - upload_start_time > epsilon:
                            violations.append({
                                'type': 'Edge-Cloud Direct Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'detail': f"Task {task.id} starts uploading at {upload_start_time} before edge {pred_task.id} finishes sending to cloud at {cloud_send_time}"
                            })
                    else:
                        # Otherwise need to wait for data to return to device
                        edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based

                        # Check both 0-based and 1-based indexing for backward compatibility
                        has_receive_time = (edge_id in pred_task.FT_edge_receive or
                                            (edge_id + 1) in pred_task.FT_edge_receive)

                        if not has_receive_time:
                            violations.append({
                                'type': 'Missing Edge Receive Time',
                                'task': pred_task.id,
                                'edge': edge_id + 1,  # Report as 1-based for user clarity
                                'detail': f"Edge task {pred_task.id} has no receive time for edge {edge_id + 1}"
                            })
                            continue

                        # Get receive time (try both indices)
                        receive_time = pred_task.FT_edge_receive.get(edge_id,
                                                                     pred_task.FT_edge_receive.get(edge_id + 1,
                                                                                                   float('inf')))

                        # Apply floating-point tolerance
                        if receive_time - upload_start_time > epsilon:
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

                edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based for internal lookup
                core_id = task.edge_assignment.core_id - 1  # Convert to 0-based

                # Check both 0-based and 1-based indexing for backward compatibility
                has_finish_time = (edge_id in task.FT_edge or
                                   (edge_id + 1) in task.FT_edge)

                if not has_finish_time:
                    violations.append({
                        'type': 'Invalid Edge Execution',
                        'task': task.id,
                        'edge': edge_id + 1,  # Report as 1-based for user clarity
                        'detail': f"Task {task.id} has no finish time for edge {edge_id + 1}"
                    })
                    continue

                # Get execution time on this edge core
                exec_time = task.get_edge_execution_time(edge_id + 1, core_id + 1)
                if exec_time is None:
                    violations.append({
                        'type': 'Missing Edge Execution Time',
                        'task': task.id,
                        'edge': edge_id + 1,  # Report as 1-based for user clarity
                        'core': core_id + 1,  # Report as 1-based for user clarity
                        'detail': f"Task {task.id} has no execution time for edge {edge_id + 1}, core {core_id + 1}"
                    })
                    continue

                # Get finish time (try both indices)
                finish_time = task.FT_edge.get(edge_id,
                                               task.FT_edge.get(edge_id + 1, float('inf')))

                edge_start_time = finish_time - exec_time

                # Check dependency based on predecessor's execution tier
                if pred_task.execution_tier == ExecutionTier.DEVICE:
                    # Need to account for device-to-edge transfer time
                    transfer_time = pred_task.calculate_data_transfer_time(
                        ExecutionTier.DEVICE, ExecutionTier.EDGE,
                        upload_rates, download_rates,
                        target_location=(edge_id + 1, 0)
                    )

                    # Device predecessor must finish and data must transfer before edge task starts
                    ready_time = pred_task.FT_l + transfer_time
                    # Apply floating-point tolerance
                    if ready_time - edge_start_time > epsilon:
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
                        target_location=(edge_id + 1, 0)
                    )

                    # Cloud computation must finish and data must transfer before edge task starts
                    ready_time = pred_task.FT_c + transfer_time
                    # Apply floating-point tolerance
                    if ready_time - edge_start_time > epsilon:
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

                    pred_edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based

                    # Check both 0-based and 1-based indexing for backward compatibility
                    has_pred_finish_time = (pred_edge_id in pred_task.FT_edge or
                                            (pred_edge_id + 1) in pred_task.FT_edge)

                    if not has_pred_finish_time:
                        violations.append({
                            'type': 'Missing Edge Finish Time',
                            'task': pred_task.id,
                            'edge': pred_edge_id + 1,  # Report as 1-based for user clarity
                            'detail': f"Edge task {pred_task.id} has no finish time for edge {pred_edge_id + 1}"
                        })
                        continue

                    # Get predecessor finish time (try both indices)
                    pred_finish_time = pred_task.FT_edge.get(pred_edge_id,
                                                             pred_task.FT_edge.get(pred_edge_id + 1, float('inf')))

                    if pred_edge_id == edge_id:
                        # Same edge node, no transfer needed
                        # Apply floating-point tolerance - key change!
                        if pred_finish_time - edge_start_time > epsilon:
                            violations.append({
                                'type': 'Same-Edge Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'edge': edge_id + 1,  # Report as 1-based for user clarity
                                'detail': f"Task {task.id} starts on edge {edge_id + 1} at {edge_start_time} before predecessor {pred_task.id} finishes at {pred_finish_time}"
                            })
                    else:
                        # Different edge nodes, need to account for edge-to-edge transfer
                        transfer_time = pred_task.calculate_data_transfer_time(
                            ExecutionTier.EDGE, ExecutionTier.EDGE,
                            upload_rates, download_rates,
                            source_location=(pred_edge_id + 1, 0),
                            target_location=(edge_id + 1, 0)
                        )

                        ready_time = pred_finish_time + transfer_time
                        # Apply floating-point tolerance
                        if ready_time - edge_start_time > epsilon:
                            violations.append({
                                'type': 'Edge-Edge Dependency Violation',
                                'task': task.id,
                                'predecessor': pred_task.id,
                                'source_edge': pred_edge_id + 1,  # Report as 1-based for user clarity
                                'target_edge': edge_id + 1,  # Report as 1-based for user clarity
                                'detail': f"Task {task.id} starts on edge {edge_id + 1} at {edge_start_time} before data from edge {pred_edge_id + 1} task {pred_task.id} arrives at {ready_time}"
                            })

    is_valid = len(violations) == 0
    return is_valid, violations


def three_tier_kernel_algorithm(tasks, sequences, num_device_cores=3,
                                num_edge_nodes=2, num_edge_cores_per_node=2):
    """
    Enhanced kernel algorithm for three-tier architecture with better cloud task handling.

    Provides linear-time rescheduling for all tasks across device, edge, and cloud
    while maintaining task dependencies and resource constraints.

    Args:
        tasks: List of all tasks
        sequences: Lists of task sequences for all execution units
        num_device_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node

    Returns:
        Updated tasks with new schedule
    """
    # Initialize kernel scheduler
    scheduler = ThreeTierKernelScheduler(
        tasks, sequences, num_device_cores, num_edge_nodes, num_edge_cores_per_node
    )

    # Add debug to track cloud task migrations
    cloud_tasks = [t.id for t in tasks if t.execution_tier == ExecutionTier.CLOUD]
    logger.info(f"Starting kernel algorithm with cloud tasks: {cloud_tasks}")

    # Initialize LIFO stack with ready tasks
    queue = scheduler.initialize_queue()

    # Track scheduled tasks for reporting
    device_scheduled = []
    edge_scheduled = []
    cloud_scheduled = []

    # Preprocess cloud tasks to ensure proper dependency order
    # This creates a topological sorting of cloud tasks
    cloud_task_order = []
    cloud_task_objects = [t for t in tasks if t.execution_tier == ExecutionTier.CLOUD]

    # First build a dependency graph for cloud tasks
    cloud_deps = {t.id: set() for t in cloud_task_objects}
    for task in cloud_task_objects:
        for pred in task.pred_tasks:
            if pred.execution_tier == ExecutionTier.CLOUD:
                cloud_deps[task.id].add(pred.id)

    # Use topological sorting to determine cloud task execution order
    temp_mark = set()
    perm_mark = set()

    def visit(task_id):
        if task_id in perm_mark:
            return
        if task_id in temp_mark:
            # Cycle detected - break it by removing an edge
            # For scheduling purposes, we'll resolve to ensure progress
            return

        temp_mark.add(task_id)
        for dep_id in cloud_deps.get(task_id, set()):
            visit(dep_id)

        temp_mark.remove(task_id)
        perm_mark.add(task_id)
        cloud_task_order.append(task_id)

    # Visit all cloud tasks to build ordering
    for task in cloud_task_objects:
        if task.id not in perm_mark:
            visit(task.id)

    # Reverse to get correct order (predecessors first)
    cloud_task_order.reverse()

    # Map from task ID to order index (used for timing calculations)
    cloud_order_map = {task_id: idx for idx, task_id in enumerate(cloud_task_order)}

    # Main scheduling loop
    while queue:
        # Pop next ready task from stack
        current_task = queue.popleft()

        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on execution tier
        if hasattr(current_task, 'execution_tier'):
            # Three-tier task
            if current_task.execution_tier == ExecutionTier.DEVICE:
                scheduler.schedule_device_task(current_task)
                device_scheduled.append(current_task.id)
            elif current_task.execution_tier == ExecutionTier.EDGE:
                scheduler.schedule_edge_task(current_task)
                edge_scheduled.append(current_task.id)
            elif current_task.execution_tier == ExecutionTier.CLOUD:
                # CRITICAL FIX: Improved cloud scheduling with dependency awareness
                # Initialize cloud timing if missing or invalid
                if current_task.FT_ws <= 0 or current_task.FT_c <= 0 or current_task.FT_wr <= 0:
                    logger.info(f"Initializing cloud timing for task {current_task.id}")

                    # Calculate earliest start time based on predecessors
                    upload_ready_time = 0
                    for pred in current_task.pred_tasks:
                        if pred.execution_tier == ExecutionTier.DEVICE:
                            upload_ready_time = max(upload_ready_time, pred.FT_l)
                        elif pred.execution_tier == ExecutionTier.EDGE and pred.edge_assignment:
                            edge_id = pred.edge_assignment.edge_id - 1
                            if edge_id in pred.FT_edge:
                                upload_ready_time = max(upload_ready_time, pred.FT_edge[edge_id])
                        elif pred.execution_tier == ExecutionTier.CLOUD:
                            # Wait for predecessor's upload to complete
                            upload_ready_time = max(upload_ready_time, pred.FT_ws)

                    # Use cloud task order to determine staggered start times
                    # This ensures tasks are scheduled in dependency order
                    order_position = cloud_order_map.get(current_task.id, 0)

                    # Set timing parameters with base time of 2.0 units per task
                    base_time = 2.0
                    upload_time = current_task.cloud_execution_times[0] if hasattr(current_task,
                                                                                   'cloud_execution_times') else 3.0
                    compute_time = current_task.cloud_execution_times[1] if hasattr(current_task,
                                                                                    'cloud_execution_times') else 1.0
                    download_time = current_task.cloud_execution_times[2] if hasattr(current_task,
                                                                                     'cloud_execution_times') else 1.0

                    # Stagger start times based on order position and dependencies
                    upload_start = max(upload_ready_time, order_position * base_time)

                    # Set task timing parameters
                    current_task.RT_ws = upload_start
                    current_task.FT_ws = upload_start + upload_time
                    current_task.RT_c = current_task.FT_ws
                    current_task.FT_c = current_task.RT_c + compute_time
                    current_task.RT_wr = current_task.FT_c
                    current_task.FT_wr = current_task.RT_wr + download_time

                    # Make sure execution unit timing is updated
                    if hasattr(current_task, 'execution_unit_task_start_times'):
                        cloud_idx = num_device_cores + (num_edge_nodes * num_edge_cores_per_node)
                        if not current_task.execution_unit_task_start_times or len(
                                current_task.execution_unit_task_start_times) <= cloud_idx:
                            # Initialize or extend the array if needed
                            current_task.execution_unit_task_start_times = [-1] * (cloud_idx + 1)
                        current_task.execution_unit_task_start_times[cloud_idx] = current_task.RT_ws

                    # Clear device/edge attributes
                    current_task.device_core = -1
                    current_task.edge_assignment = None
                    current_task.FT_l = 0

                    logger.info(
                        f"Cloud task {current_task.id} timing: upload={current_task.FT_ws}, compute={current_task.FT_c}, download={current_task.FT_wr}")

                # Schedule the cloud task
                scheduler.schedule_cloud_task(current_task)
                cloud_scheduled.append(current_task.id)
        else:
            # Original MCC task
            if hasattr(current_task, 'is_core_task') and current_task.is_core_task:
                scheduler.schedule_device_task(current_task)
                device_scheduled.append(current_task.id)
            else:
                # Handle cloud task with original MCC attributes
                if not hasattr(current_task, 'RT_ws') or not hasattr(current_task, 'RT_c'):
                    # Initialize with reasonable defaults
                    current_task.RT_ws = 0
                    current_task.FT_ws = 3.0
                    current_task.RT_c = 3.0
                    current_task.FT_c = 4.0
                    current_task.RT_wr = 4.0
                    current_task.FT_wr = 5.0

                scheduler.schedule_cloud_task(current_task)
                cloud_scheduled.append(current_task.id)

        # Update ready1 and ready2 vectors for all tasks
        for task in tasks:
            scheduler.update_task_state(task)

            # Add newly ready tasks to stack
            task_idx = task.id - 1
            if task_idx < len(scheduler.dependency_ready) and task_idx < len(scheduler.sequence_ready):
                if (scheduler.dependency_ready[task_idx] == 0 and
                        scheduler.sequence_ready[task_idx] == 0 and
                        task.is_scheduled != SchedulingState.KERNEL_SCHEDULED and
                        task not in queue):
                    queue.append(task)

    # Report scheduling statistics
    logger.info(f"Kernel scheduling complete:")
    logger.info(f"  Device tasks: {len(device_scheduled)}")
    logger.info(f"  Edge tasks: {len(edge_scheduled)}")
    logger.info(f"  Cloud tasks: {len(cloud_scheduled)}")

    # Reset scheduling state for next iteration if needed
    for task in tasks:
        task.is_scheduled = SchedulingState.SCHEDULED

    return tasks


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


def initialize_migration_choices_three_tier(tasks: List[Any],
                                            num_device_cores: int,
                                            num_edge_nodes: int,
                                            num_edge_cores_per_node: int) -> np.ndarray:
    """
    Initializes possible migration choices for each task in the three-tier architecture.

    Creates a matrix where rows represent tasks and columns represent all possible
    execution units (device cores, edge cores, cloud). A True value indicates
    that migrating the task to that execution unit is a valid option.

    Args:
        tasks: List of all tasks
        num_device_cores: Number of cores on the mobile device
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node

    Returns:
        numpy.ndarray: Boolean matrix of migration choices [tasks × units]
    """
    # Calculate total number of execution units
    total_units = num_device_cores + (num_edge_nodes * num_edge_cores_per_node) + 1

    # Create matrix of migration possibilities [tasks × units]
    migration_choices = np.zeros((len(tasks), total_units), dtype=bool)

    # Iterate through all tasks to determine valid migration targets
    for i, task in enumerate(tasks):
        current_tier = None
        current_location = None

        # Determine current execution tier and location
        if hasattr(task, 'execution_tier'):
            # Three-tier task
            current_tier = task.execution_tier

            if current_tier == ExecutionTier.DEVICE:
                current_location = task.device_core
            elif current_tier == ExecutionTier.EDGE and task.edge_assignment:
                current_location = (
                    task.edge_assignment.edge_id - 1,  # Convert to 0-based index
                    task.edge_assignment.core_id - 1  # Convert to 0-based index
                )
        else:
            # Original MCC task
            if hasattr(task, 'is_core_task') and task.is_core_task:
                current_tier = ExecutionTier.DEVICE
                current_location = task.assignment
            else:
                current_tier = ExecutionTier.CLOUD
                current_location = None

        # Set migration options based on current location
        if current_tier == ExecutionTier.DEVICE:
            # Device task:
            # 1. Can migrate to any other device core
            # 2. Can migrate to any edge node core
            # 3. Can migrate to cloud

            # Set all device cores as options
            for core in range(num_device_cores):
                # Only consider migration if it's not the current core
                migration_choices[i, core] = (core != current_location)

            # Set all edge cores as options
            device_offset = num_device_cores
            for node in range(num_edge_nodes):
                for core in range(num_edge_cores_per_node):
                    idx = device_offset + (node * num_edge_cores_per_node) + core
                    migration_choices[i, idx] = True

            # Set cloud as an option
            migration_choices[i, total_units - 1] = True

        elif current_tier == ExecutionTier.EDGE:
            # Edge task:
            # 1. Can migrate to any device core
            # 2. Can migrate to other edge node cores (different node or different core)
            # 3. Can migrate to cloud

            # Set all device cores as options
            for core in range(num_device_cores):
                migration_choices[i, core] = True

            # Set edge cores as options
            device_offset = num_device_cores
            for node in range(num_edge_nodes):
                for core in range(num_edge_cores_per_node):
                    idx = device_offset + (node * num_edge_cores_per_node) + core

                    # Skip current edge core
                    if (current_location is not None and
                            node == current_location[0] and
                            core == current_location[1]):
                        migration_choices[i, idx] = False
                    else:
                        migration_choices[i, idx] = True

            # Set cloud as an option
            migration_choices[i, total_units - 1] = True

        elif current_tier == ExecutionTier.CLOUD:
            # Cloud task:
            # 1. Can migrate to any device core
            # 2. Can migrate to any edge node core

            # Set all device cores as options
            for core in range(num_device_cores):
                migration_choices[i, core] = True

            # Set all edge cores as options
            device_offset = num_device_cores
            for node in range(num_edge_nodes):
                for core in range(num_edge_cores_per_node):
                    idx = device_offset + (node * num_edge_cores_per_node) + core
                    migration_choices[i, idx] = True

            # Can't migrate to cloud again
            migration_choices[i, total_units - 1] = False

        else:
            logger.warning(f"Task {task.id} has unknown execution tier, limiting migration options")

    # Verify migrations are possible
    for i, task in enumerate(tasks):
        if not np.any(migration_choices[i]):
            logger.warning(f"Task {task.id} has no valid migration options")

    return migration_choices


def optimize_task_scheduling_three_tier(
        tasks: List[Any],
        sequences: List[List[int]],
        initial_time: float,
        time_constraint_factor: float = 1.5,
        max_iterations: int = 50,
        max_evaluations_per_iteration: int = 100,
        early_stopping_threshold: float = 0.01,
        num_device_cores: int = 3,
        num_edge_nodes: int = 2,
        num_edge_cores_per_node: int = 2,
        device_core_powers: Dict[int, float] = None,
        edge_node_powers: Dict[Tuple[int, int], float] = None,
        rf_power: Dict[str, float] = None,
        upload_rates: Dict[str, float] = None,
        download_rates: Dict[str, float] = None
) -> Tuple[List[Any], List[List[int]], Dict[str, Any]]:
    """
    Main optimization function for three-tier task scheduling with enhanced migration handling.

    Implements an iterative energy optimization algorithm that maintains completion time
    constraints while identifying and applying energy-efficient task migrations across
    all three tiers (device, edge, cloud).

    Args:
        tasks: List of all tasks
        sequences: Initial task sequences for all execution units
        initial_time: Initial completion time to use as constraint baseline
        time_constraint_factor: Maximum allowed increase in completion time (e.g., 1.5 = 50% increase)
        max_iterations: Maximum number of optimization iterations
        max_evaluations_per_iteration: Maximum migrations to evaluate per iteration
        early_stopping_threshold: Stop if energy improvement falls below this threshold
        num_device_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node
        device_core_powers: Power consumption of device cores
        edge_node_powers: Power consumption of edge nodes
        rf_power: RF power consumption rates
        upload_rates: Upload rates between tiers
        download_rates: Download rates between tiers

    Returns:
        Tuple (optimized_tasks, optimized_sequences, metrics)
    """
    # Store start time for benchmarking
    start_time = time_module.time()

    # Create deep copies of tasks and sequences to avoid modifying originals
    current_tasks = deepcopy(tasks)
    current_sequences = [seq.copy() for seq in sequences]

    # Initialize the sequence manager and migration cache
    sequence_manager = SequenceManager(
        num_device_cores=num_device_cores,
        num_edge_nodes=num_edge_nodes,
        num_edge_cores_per_node=num_edge_cores_per_node
    )
    sequence_manager.set_all_sequences(current_sequences)

    migration_cache = MigrationCache(capacity=20000)

    # Define a simple total time calculation function
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

    # Calculate initial energy consumption
    initial_energy = total_energy_consumption_three_tier(
        current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )

    # Initialize optimization metrics
    metrics = OptimizationMetrics()
    metrics.start(initial_time, initial_energy)

    # Set maximum allowed completion time
    max_time = initial_time * time_constraint_factor

    # Validate constraints before optimization
    valid, violations = validate_task_dependencies(current_tasks)
    if not valid:
        logger.error("Initial schedule has dependency violations! Optimization may fail.")
        for v in violations[:5]:  # Show first 5 violations
            logger.error(f"Violation: {v['type']} - {v['detail']}")

    # Main optimization loop
    energy_improved = True
    consecutive_no_improvement = 0

    logger.info(f"Starting optimization with {len(current_tasks)} tasks")
    logger.info(f"Initial metrics - Time: {initial_time:.2f}, Energy: {initial_energy:.2f}")

    for iteration in range(max_iterations):
        # Break if no energy improvement for multiple iterations
        if consecutive_no_improvement >= 3:
            logger.info(f"Early stopping after {iteration} iterations (no improvement)")
            break

        # Store current energy for comparison
        previous_energy = metrics.current_energy

        # Initialize migration choices
        migration_choices = initialize_migration_choices_three_tier(
            current_tasks, num_device_cores, num_edge_nodes, num_edge_cores_per_node
        )

        # Evaluate all possible migrations
        migration_results = []
        for task_idx in range(len(current_tasks)):
            task = current_tasks[task_idx]

            # Skip already scheduled cloud tasks
            if task.execution_tier == ExecutionTier.CLOUD:
                continue

            # For each task, try migrating to cloud first
            cloud_index = num_device_cores + (num_edge_nodes * num_edge_cores_per_node)

            # Skip if task is already on cloud or migration choice isn't valid
            if not migration_choices[task_idx][cloud_index]:
                continue

            # Evaluate migration to cloud
            migration_time, migration_energy = evaluate_migration_three_tier(
                current_tasks, current_sequences, task_idx, cloud_index,
                migration_cache, three_tier_kernel_algorithm, construct_sequence_three_tier,
                total_energy_consumption_three_tier, calculate_total_time,
                num_device_cores, num_edge_nodes, num_edge_cores_per_node,
                device_core_powers, edge_node_powers, rf_power,
                upload_rates, download_rates
            )

            # Add to results
            migration_results.append((task_idx, cloud_index, migration_time, migration_energy))

            # Limit evaluations per iteration
            if len(migration_results) >= max_evaluations_per_iteration:
                break

        # Find the best migration option
        best_migration = identify_optimal_migration_three_tier(
            migration_trials_results=migration_results,
            current_time=metrics.current_time,
            current_energy=metrics.current_energy,
            max_time=max_time
        )

        # Update metrics with evaluation count
        evaluations = len(migration_results)

        # Check if a valid migration was found
        if best_migration is None:
            logger.info(f"Iteration {iteration + 1}: No valid migration found")
            consecutive_no_improvement += 1
            metrics.update(metrics.current_time, metrics.current_energy, evaluations)
            continue

        # Apply the selected migration
        task_id = best_migration.task_id
        task_idx = task_id - 1  # Convert to 0-based index

        # Get target execution unit
        if best_migration.target_tier == ExecutionTier.DEVICE:
            target_core = best_migration.target_location[0]
            target_unit_index = target_core
        elif best_migration.target_tier == ExecutionTier.EDGE:
            node_id, core_id = best_migration.target_location
            target_unit_index = num_device_cores + (node_id * num_edge_cores_per_node) + core_id
        else:  # Cloud
            target_unit_index = num_device_cores + (num_edge_nodes * num_edge_cores_per_node)

        # Log migration details
        logger.info(f"Iteration {iteration + 1}: Migrating task {task_id} to {best_migration.target_tier.name} " +
                    f"(unit index: {target_unit_index})")

        try:
            # Verify the migration actually worked by calculating energy
            pre_migration_energy = total_energy_consumption_three_tier(
                current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
            )

            # Get original task state for debugging
            task = next((t for t in current_tasks if t.id == task_id), None)
            original_tier = task.execution_tier if task else None

            # Apply the migration
            current_sequences = construct_sequence_three_tier(
                current_tasks, task_id, target_unit_index, current_sequences,
                num_device_cores, num_edge_nodes, num_edge_cores_per_node
            )

            # Apply kernel algorithm to recalculate schedule
            three_tier_kernel_algorithm(
                current_tasks, current_sequences,
                num_device_cores, num_edge_nodes, num_edge_cores_per_node
            )

            # Calculate post-migration energy
            post_migration_energy = total_energy_consumption_three_tier(
                current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
            )

            # Verify energy improvement actually occurred
            energy_diff = pre_migration_energy - post_migration_energy
            logger.info(
                f"Migration energy change: {pre_migration_energy:.2f} → {post_migration_energy:.2f} (Diff: {energy_diff:.2f})")

            # Verify task was actually migrated
            new_task = next((t for t in current_tasks if t.id == task_id), None)
            new_tier = new_task.execution_tier if new_task else None

            if original_tier == new_tier:
                logger.error(f"Migration failed: Task {task_id} still at tier {new_tier}")
                # Force the migration directly as a fallback
                if new_task and best_migration.target_tier == ExecutionTier.CLOUD:
                    logger.info(f"Forcing migration of task {task_id} to cloud")
                    new_task.execution_tier = ExecutionTier.CLOUD
                    new_task.device_core = -1
                    new_task.edge_assignment = None
                    new_task.FT_l = 0
                    # Set cloud timing
                    new_task.RT_ws = 0
                    new_task.FT_ws = 3.0
                    new_task.RT_c = 3.0
                    new_task.FT_c = 4.0
                    new_task.RT_wr = 4.0
                    new_task.FT_wr = 5.0
                    # Remove from device/edge sequence
                    for seq_idx, seq in enumerate(
                            current_sequences[:num_device_cores + (num_edge_nodes * num_edge_cores_per_node)]):
                        if task_id in seq:
                            current_sequences[seq_idx].remove(task_id)
                            break
                    # Add to cloud sequence
                    cloud_idx = num_device_cores + (num_edge_nodes * num_edge_cores_per_node)
                    if task_id not in current_sequences[cloud_idx]:
                        current_sequences[cloud_idx].append(task_id)

                    # Run kernel again to fix dependencies
                    three_tier_kernel_algorithm(
                        current_tasks, current_sequences,
                        num_device_cores, num_edge_nodes, num_edge_cores_per_node
                    )

                    # Recalculate energy
                    post_migration_energy = total_energy_consumption_three_tier(
                        current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
                    )
                    energy_diff = pre_migration_energy - post_migration_energy
                    logger.info(
                        f"After forced migration: {pre_migration_energy:.2f} → {post_migration_energy:.2f} (Diff: {energy_diff:.2f})")

            if energy_diff <= 0:
                logger.warning(f"Migration didn't reduce energy! Investigating task {task_id}...")
                # Debug task state
                task = next((t for t in current_tasks if t.id == task_id), None)
                if task:
                    logger.info(f"Task {task_id} tier: {task.execution_tier}")
                    # Add more debug info
                    if task.execution_tier == ExecutionTier.CLOUD:
                        cloud_timing = f"Cloud timing - RT_ws: {task.RT_ws}, FT_ws: {task.FT_ws}, FT_c: {task.FT_c}, FT_wr: {task.FT_wr}"
                        logger.info(cloud_timing)

            # Validate constraints after migration
            if not validate_task_dependencies(current_tasks)[0]:
                logger.warning(f"Constraint violation after migrating task {task_id}!")
                # Try to recover by re-running kernel algorithm
                three_tier_kernel_algorithm(
                    current_tasks, current_sequences,
                    num_device_cores, num_edge_nodes, num_edge_cores_per_node
                )

            # Calculate new metrics
            new_time = calculate_total_time(current_tasks)
            new_energy = total_energy_consumption_three_tier(
                current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
            )

            # Update metrics
            metrics.update(new_time, new_energy, evaluations)

            # Check if energy improved
            energy_improved = new_energy < previous_energy

            if energy_improved:
                # Reset counter for consecutive non-improvements
                consecutive_no_improvement = 0

                # Calculate relative improvement
                relative_improvement = (previous_energy - new_energy) / previous_energy

                logger.info(
                    f"Iteration {iteration + 1}: Energy improved from {previous_energy:.2f} to {new_energy:.2f} "
                    f"({relative_improvement * 100:.2f}%), Time: {new_time:.2f}")

                # Check for early stopping
                if relative_improvement < early_stopping_threshold:
                    consecutive_no_improvement += 1
            else:
                consecutive_no_improvement += 1
                logger.info(f"Iteration {iteration + 1}: No energy improvement")

        except Exception as e:
            logger.error(f"Error during migration: {e}")
            consecutive_no_improvement += 1
            continue

    # Calculate final metrics
    final_time = calculate_total_time(current_tasks)
    final_energy = total_energy_consumption_three_tier(
        current_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )

    # Update metrics with final values
    metrics.update(final_time, final_energy)

    # Calculate overall improvements
    time_change = (final_time - initial_time) / initial_time * 100
    energy_reduction = (initial_energy - final_energy) / initial_energy * 100

    # Log optimization summary
    logger.info(f"Optimization completed in {time_module.time() - start_time:.2f} seconds")
    logger.info(f"Final metrics - Time: {final_time:.2f} ({time_change:+.2f}%), "
                f"Energy: {final_energy:.2f} ({energy_reduction:+.2f}%)")
    logger.info(f"Iterations: {metrics.iterations}, Migrations: {metrics.migrations}, "
                f"Evaluations: {metrics.evaluations}")

    # Return optimized tasks, sequences, and metrics
    return current_tasks, current_sequences, metrics.get_summary()


def update_cloud_timing(task, tasks):
    """
    Recalculate cloud timing parameters for a task based on its predecessors.

    Assumes that:
      - For a device predecessor, its finish time is task.FT_l.
      - For a cloud predecessor, its upload finish time is task.FT_ws.
      - For an edge predecessor, we use its result receive time stored in FT_edge_receive
        (using the predecessor’s assigned edge, converting 1-based to 0-based index).

    Updates the following attributes:
      - task.RT_ws, task.FT_ws, task.RT_c, task.FT_c, task.RT_wr, task.FT_wr.
    """
    if not task.pred_tasks:
        upload_ready = 0.0
    else:
        pred_finish_times = []
        for pred in task.pred_tasks:
            if pred.execution_tier == ExecutionTier.DEVICE:
                pred_finish_times.append(pred.FT_l)
            elif pred.execution_tier == ExecutionTier.CLOUD:
                pred_finish_times.append(pred.FT_ws)
            elif pred.execution_tier == ExecutionTier.EDGE:
                if pred.edge_assignment and pred.FT_edge_receive:
                    # Use 0-based index for FT_edge_receive
                    key = pred.edge_assignment.edge_id - 1
                    if key in pred.FT_edge_receive:
                        pred_finish_times.append(pred.FT_edge_receive[key])
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


def construct_sequence_three_tier(tasks, task_id, target_unit_index, sequences,
                                  num_device_cores=3, num_edge_nodes=2, num_edge_cores_per_node=2):
    """
    When migrating a task to cloud, remove it from its current sequence,
    add it to the cloud sequence, and update its execution tier and dynamic cloud timing.
    """
    task = next((t for t in tasks if t.id == task_id), None)
    if not task:
        logger.error(f"Task {task_id} not found for migration")
        return sequences

    target_unit = get_execution_unit_from_index(
        target_unit_index, num_device_cores, num_edge_nodes, num_edge_cores_per_node
    )

    # Find and remove the task from its current sequence.
    source_index = next((idx for idx, seq in enumerate(sequences) if task_id in seq), -1)
    if source_index == -1:
        logger.error(f"Task {task_id} not found in any sequence")
        return sequences
    logger.info(f"Removing task {task_id} from sequence {source_index}")
    sequences[source_index].remove(task_id)

    # Add to target sequence.
    target_index = target_unit_index  # Direct mapping
    logger.info(f"Adding task {task_id} to sequence {target_index}")
    sequences[target_index].append(task_id)

    # Update task assignment.
    task.execution_tier = target_unit.tier
    if target_unit.tier == ExecutionTier.DEVICE:
        logger.info(f"Updating task {task_id} to device core {target_unit.location[0]}")
        task.device_core = target_unit.location[0]
        task.edge_assignment = None
    elif target_unit.tier == ExecutionTier.EDGE:
        node_id, core_id = target_unit.location
        logger.info(f"Updating task {task_id} to edge {node_id + 1}, core {core_id + 1}")
        task.edge_assignment = EdgeAssignment(edge_id=node_id + 1, core_id=core_id + 1)
        task.device_core = -1
    else:  # Cloud
        logger.info(f"Updating task {task_id} to cloud")
        task.device_core = -1
        task.edge_assignment = None
        task.FT_l = 0
        # IMPORTANT: Dynamically update cloud timing instead of using fixed values.
        update_cloud_timing(task, tasks)

    return sequences


def calculate_task_ready_time(task, target_unit, all_tasks, task_lookup):
    """
    Calculate the earliest time a task can start on a target execution unit.
    This accounts for predecessor finish times and data transfer overheads.

    Args:
        task: Task being migrated
        target_unit: Target execution unit
        all_tasks: List of all tasks
        task_lookup: Dictionary mapping task ID to task object

    Returns:
        Earliest possible start time on the target unit
    """
    # Implementation depends on what globals are available - this is a minimal version
    if not hasattr(task, 'pred_tasks') or not task.pred_tasks:
        return 0  # Entry task

    max_ready_time = 0

    for pred_task in task.pred_tasks:
        # Simple implementation: just use the predecessor's finish time
        pred_finish_time = 0

        if hasattr(pred_task, 'execution_tier'):
            # Three-tier task
            if pred_task.execution_tier == ExecutionTier.DEVICE:
                pred_finish_time = getattr(pred_task, 'FT_l', 0)
            elif pred_task.execution_tier == ExecutionTier.CLOUD:
                pred_finish_time = getattr(pred_task, 'FT_wr', 0)
            elif pred_task.execution_tier == ExecutionTier.EDGE:
                if hasattr(pred_task, 'edge_assignment') and pred_task.edge_assignment:
                    edge_id = pred_task.edge_assignment.edge_id - 1  # Convert to 0-based

                    if hasattr(pred_task, 'FT_edge') and edge_id in pred_task.FT_edge:
                        pred_finish_time = pred_task.FT_edge[edge_id]
                    elif hasattr(pred_task, 'FT_edge_receive') and edge_id in pred_task.FT_edge_receive:
                        pred_finish_time = pred_task.FT_edge_receive[edge_id]
        else:
            # Original MCC task
            pred_finish_time = max(getattr(pred_task, 'FT_l', 0), getattr(pred_task, 'FT_wr', 0))

        max_ready_time = max(max_ready_time, pred_finish_time)

    # We could add transfer overhead here, but for simplicity, we'll just use predecessor finish time
    return max_ready_time


def estimate_transfer_overhead(task: Any, target_unit: ExecutionUnit) -> float:
    """
    Estimate data transfer overhead for moving a task to a target execution unit.

    Args:
        task: Task being migrated
        target_unit: Target execution unit

    Returns:
        Estimated transfer overhead (in time units)
    """
    # Default overhead values based on tier transitions
    DEFAULT_OVERHEADS = {
        (ExecutionTier.DEVICE, ExecutionTier.DEVICE): 0.0,  # No overhead for same-device migration
        (ExecutionTier.DEVICE, ExecutionTier.EDGE): 2.0,  # Device to edge
        (ExecutionTier.DEVICE, ExecutionTier.CLOUD): 3.0,  # Device to cloud
        (ExecutionTier.EDGE, ExecutionTier.DEVICE): 2.0,  # Edge to device
        (ExecutionTier.EDGE, ExecutionTier.EDGE): 1.5,  # Edge to edge (different nodes)
        (ExecutionTier.EDGE, ExecutionTier.CLOUD): 2.0,  # Edge to cloud
        (ExecutionTier.CLOUD, ExecutionTier.DEVICE): 3.0,  # Cloud to device
        (ExecutionTier.CLOUD, ExecutionTier.EDGE): 2.0,  # Cloud to edge
        (ExecutionTier.CLOUD, ExecutionTier.CLOUD): 0.0,  # No overhead for cloud-to-cloud
    }

    source_tier = None

    # Determine source tier
    if hasattr(task, 'execution_tier'):
        # Three-tier task
        source_tier = task.execution_tier
    else:
        # Original MCC task
        if hasattr(task, 'is_core_task') and task.is_core_task:
            source_tier = ExecutionTier.DEVICE
        else:
            source_tier = ExecutionTier.CLOUD

    # Same tier but different edge nodes
    if source_tier == ExecutionTier.EDGE and target_unit.tier == ExecutionTier.EDGE:
        if hasattr(task, 'edge_assignment') and task.edge_assignment:
            source_node = task.edge_assignment.edge_id - 1  # Convert to 0-based
            target_node = target_unit.location[0]

            if source_node == target_node:
                return 0.0  # Same edge node, no overhead

    # Look up default overhead
    return DEFAULT_OVERHEADS.get((source_tier, target_unit.tier), 2.0)


def total_energy_consumption_three_tier(tasks: List[Any],
                                        device_core_powers: Dict[int, float] = None,
                                        edge_node_powers: Dict[Tuple[int, int], float] = None,
                                        rf_power: Dict[str, float] = None,
                                        upload_rates: Dict[str, float] = None,
                                        download_rates: Dict[str, float] = None) -> float:
    """
    Calculate total energy consumption across all tasks in the three-tier architecture.

    Args:
        tasks: List of all tasks
        device_core_powers: Power consumption of device cores
        edge_node_powers: Power consumption of edge nodes
        rf_power: RF power consumption rates
        upload_rates: Upload rates between tiers
        download_rates: Download rates between tiers

    Returns:
        Total energy consumption
    """
    return sum(
        calculate_energy_consumption_three_tier(
            task, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
        )
        for task in tasks
    )


def identify_critical_path_tasks(tasks: List[Any], total_time_func: Callable) -> Set[int]:
    """
    Identify tasks on the critical path to prioritize for migration evaluation.

    Args:
        tasks: List of all tasks
        total_time_func: Function to calculate total time

    Returns:
        Set of task IDs on the critical path
    """
    # Find exit tasks (tasks with no successors)
    exit_tasks = [task for task in tasks if not task.succ_tasks]

    # Find the exit task that determines the completion time
    completion_time = total_time_func(tasks)
    critical_exit_task = None

    for task in exit_tasks:
        task_finish_time = 0

        if hasattr(task, 'execution_tier'):
            # Three-tier task
            if task.execution_tier == ExecutionTier.DEVICE:
                task_finish_time = task.FT_l
            elif task.execution_tier == ExecutionTier.CLOUD:
                task_finish_time = task.FT_wr
            elif task.execution_tier == ExecutionTier.EDGE:
                if task.edge_assignment:
                    edge_id = task.edge_assignment.edge_id - 1
                    if edge_id in task.FT_edge_receive:
                        task_finish_time = task.FT_edge_receive[edge_id]
        else:
            # Original MCC task
            task_finish_time = max(task.FT_l, task.FT_wr)

        if abs(task_finish_time - completion_time) < 1e-6:  # Allow for floating point imprecision
            critical_exit_task = task
            break

    # If no critical exit task found, use the first exit task
    if critical_exit_task is None and exit_tasks:
        critical_exit_task = exit_tasks[0]

    # Traverse backwards to find all tasks on the critical path
    critical_tasks = set()
    if critical_exit_task:
        critical_tasks.add(critical_exit_task.id)

        # Simple BFS to find critical path
        queue = [critical_exit_task]
        while queue:
            current_task = queue.pop(0)

            # Find the predecessor that determines the ready time
            if current_task.pred_tasks:
                critical_pred = max(
                    current_task.pred_tasks,
                    key=lambda t: max(getattr(t, 'FT_l', 0), getattr(t, 'FT_wr', 0))
                )
                critical_tasks.add(critical_pred.id)
                queue.append(critical_pred)

    return critical_tasks


def optimize_migration_search(tasks: List[Any],
                              sequences: List[List[int]],
                              current_time: float,
                              current_energy: float,
                              max_time: float,
                              migration_choices: np.ndarray,
                              migration_cache: Dict,
                              num_device_cores: int = 3,
                              num_edge_nodes: int = 2,
                              num_edge_cores_per_node: int = 2,
                              max_evaluations: int = 100,
                              kernel_algorithm_func: Callable = None,
                              construct_sequence_func: Callable = None,
                              energy_calculation_func: Callable = None,
                              total_time_func: Callable = None,
                              device_core_powers: Dict[int, float] = None,
                              edge_node_powers: Dict[Tuple[int, int], float] = None,
                              rf_power: Dict[str, float] = None,
                              upload_rates: Dict[str, float] = None,
                              download_rates: Dict[str, float] = None) -> Optional[TaskMigrationState]:
    """
    Optimize migration search to find the best migration more efficiently.

    Args:
        tasks: List of all tasks
        sequences: Current sequences for all execution units
        current_time: Current total completion time
        current_energy: Current total energy consumption
        max_time: Maximum allowable completion time
        migration_choices: Boolean matrix of valid migration choices
        migration_cache: Cache for memoizing results
        num_device_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node
        max_evaluations: Maximum number of migrations to evaluate
        kernel_algorithm_func: Function to run kernel algorithm
        construct_sequence_func: Function to construct sequence
        energy_calculation_func: Function to calculate energy
        total_time_func: Function to calculate total time
        device_core_powers: Power consumption of device cores
        edge_node_powers: Power consumption of edge nodes
        rf_power: RF power consumption rates
        upload_rates: Upload rates between tiers
        download_rates: Download rates between tiers

    Returns:
        TaskMigrationState for the optimal migration, or None if no valid migration found
    """
    # Start timing for performance benchmarking
    search_start_time = time_module.time()

    # Get tasks in critical path to prioritize
    critical_tasks = identify_critical_path_tasks(tasks, total_time_func)

    # Prioritize tasks that are likely to give good energy reductions
    task_priorities = prioritize_tasks_for_migration(
        tasks, migration_choices,
        device_core_powers, edge_node_powers,
        critical_tasks
    )

    # Create search space with priority-based sampling
    search_space = create_prioritized_search_space(
        task_priorities, migration_choices,
        num_device_cores, num_edge_nodes, num_edge_cores_per_node,
        max_evaluations
    )

    # Evaluate migrations in batches for efficiency
    results = batch_evaluate_migrations(
        tasks, sequences,
        migration_choices, migration_cache,
        num_device_cores, num_edge_nodes, num_edge_cores_per_node,
        min(max_evaluations // 10, 10),  # Batch size
        kernel_algorithm_func, construct_sequence_func,
        energy_calculation_func, total_time_func,
        device_core_powers, edge_node_powers, rf_power,
        upload_rates, download_rates
    )

    # Find optimal migration
    optimal_migration = identify_optimal_migration_three_tier(
        results, current_time, current_energy, max_time
    )

    # Log performance info
    search_time = time_module.time() - search_start_time
    logger.info(f"Migration search completed in {search_time:.3f}s, "
                f"evaluated {len(results)} of {len(search_space)} possible migrations")

    if optimal_migration:
        logger.info(f"Found optimal migration: {optimal_migration}")
    else:
        logger.info("No valid migration found")

    return optimal_migration


def prioritize_tasks_for_migration(tasks: List[Any],
                                   migration_choices: np.ndarray,
                                   device_core_powers: Dict[int, float],
                                   edge_node_powers: Dict[Tuple[int, int], float],
                                   critical_tasks: Set[int]) -> Dict[int, float]:
    """
    Prioritize tasks for migration evaluation based on potential energy reduction.

    Args:
        tasks: List of all tasks
        migration_choices: Boolean matrix of valid migration choices
        device_core_powers: Power consumption of device cores
        edge_node_powers: Power consumption of edge nodes
        critical_tasks: Set of task IDs on the critical path

    Returns:
        Dictionary mapping task IDs to priority scores
    """
    task_priorities = {}

    for i, task in enumerate(tasks):
        # Start with a base priority
        priority = 1.0

        # Tasks on critical path get higher priority
        if task.id in critical_tasks:
            priority *= 2.0

        # Tasks with high power consumption get higher priority (more potential for reduction)
        current_power = 0

        if hasattr(task, 'execution_tier'):
            # Three-tier task
            if task.execution_tier == ExecutionTier.DEVICE:
                current_power = device_core_powers.get(task.device_core, 1.0)
            elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
                edge_id = task.edge_assignment.edge_id
                core_id = task.edge_assignment.core_id
                current_power = edge_node_powers.get((edge_id, core_id), 1.5)
        else:
            # Original MCC task
            if hasattr(task, 'is_core_task') and task.is_core_task:
                current_power = device_core_powers.get(task.assignment, 1.0)

        # Scale priority by power
        priority *= (1.0 + current_power)

        # Tasks with more migration options get higher priority
        num_options = migration_choices[i].sum()
        priority *= (1.0 + 0.1 * num_options)

        # Store priority
        task_priorities[task.id] = priority

    return task_priorities


def create_prioritized_search_space(task_priorities: Dict[int, float],
                                    migration_choices: np.ndarray,
                                    num_device_cores: int,
                                    num_edge_nodes: int,
                                    num_edge_cores_per_node: int,
                                    max_samples: int) -> List[Tuple[int, int]]:
    """
    Create a prioritized search space for migration evaluation.

    Args:
        task_priorities: Dictionary mapping task IDs to priority scores
        migration_choices: Boolean matrix of valid migration choices
        num_device_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node
        max_samples: Maximum number of samples to return

    Returns:
        List of (task_idx, target_unit_index) tuples
    """
    # Get all valid migration options
    all_options = []

    num_tasks, num_units = migration_choices.shape
    for task_idx in range(num_tasks):
        task_id = task_idx + 1  # Convert to 1-based ID
        priority = task_priorities.get(task_id, 1.0)

        for target_unit_index in range(num_units):
            if migration_choices[task_idx, target_unit_index]:
                # Add migration option with priority
                all_options.append((priority, task_idx, target_unit_index))

    # Sort options by priority (descending)
    all_options.sort(reverse=True)

    # Take top options up to max_samples
    top_options = all_options[:max_samples]

    # Extract task_idx and target_unit_index
    return [(task_idx, target_unit_index) for _, task_idx, target_unit_index in top_options]


def identify_optimal_migration_three_tier(migration_trials_results: List[Tuple],
                                          current_time: float,
                                          current_energy: float,
                                          max_time: float,
                                          priority_energy_reduction: bool = True) -> Optional[TaskMigrationState]:
    logger.info(f"Evaluating {len(migration_trials_results)} possible migrations")
    allowed_time_increase = current_time * 0.5  # Allow 50% time increase

    candidate_migrations = []
    for task_idx, target_unit_index, migration_time, migration_energy in migration_trials_results:
        if migration_time == float('inf') or migration_energy == float('inf'):
            continue
        task_id = task_idx + 1
        energy_reduction = current_energy - migration_energy
        time_increase = migration_time - current_time
        if energy_reduction > 0 and time_increase <= allowed_time_increase:
            source_unit = get_current_execution_unit(tasks[task_idx])
            target_unit = get_execution_unit_from_index(
                target_unit_index,
                globals().get('num_device_cores', 3),
                globals().get('num_edge_nodes', 2),
                globals().get('num_edge_cores_per_node', 2)
            )
            candidate_migrations.append((task_id, energy_reduction, time_increase,
                                         migration_time, migration_energy,
                                         target_unit_index, source_unit, target_unit))
    # Sort candidates by energy reduction (highest first)
    candidate_migrations.sort(key=lambda x: x[1], reverse=True)
    if candidate_migrations:
        best = candidate_migrations[0]
        logger.info(f"Selected migration: Task {best[0]} with energy reduction {best[1]:.2f}")
        return TaskMigrationState(
            time=best[3],
            energy=best[4],
            efficiency=best[1] / max(0.1, best[2]),
            task_id=best[0],
            source_tier=best[6].tier,
            target_tier=best[7].tier,
            source_location=best[6].location,
            target_location=best[7].location,
            energy_reduction=best[1],
            time_increase=best[2]
        )
    logger.info("No suitable migration found")
    return None


def get_execution_unit_from_index(index: int, num_device_cores: int,
                                  num_edge_nodes: int, num_edge_cores_per_node: int) -> ExecutionUnit:
    """
    Converts a linear index to an ExecutionUnit.

    Args:
        index: Linear index into the execution units
        num_device_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node

    Returns:
        ExecutionUnit corresponding to the index
    """
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


def batch_evaluate_migrations(tasks: List[Any],
                              sequences: List[List[int]],
                              migration_choices: np.ndarray,
                              migration_cache: Dict,
                              num_device_cores: int = 3,
                              num_edge_nodes: int = 2,
                              num_edge_cores_per_node: int = 3,
                              batch_size: int = 10,
                              kernel_algorithm_func: Callable = None,
                              construct_sequence_func: Callable = None,
                              energy_calculation_func: Callable = None,
                              total_time_func: Callable = None,
                              device_core_powers: Dict[int, float] = None,
                              edge_node_powers: Dict[Tuple[int, int], float] = None,
                              rf_power: Dict[str, float] = None,
                              upload_rates: Dict[str, float] = None,
                              download_rates: Dict[str, float] = None) -> List[Tuple]:
    """
    Evaluate multiple migrations in batches for improved efficiency.

    Args:
        tasks: List of all tasks
        sequences: Current sequences for all execution units
        migration_choices: Boolean matrix of valid migration choices
        migration_cache: Cache for memoizing results
        num_device_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores_per_node: Number of cores per edge node
        batch_size: Number of migrations to evaluate in parallel
        kernel_algorithm_func: Function to run kernel algorithm
        construct_sequence_func: Function to construct sequence
        energy_calculation_func: Function to calculate energy
        total_time_func: Function to calculate total time
        device_core_powers: Power consumption of device cores
        edge_node_powers: Power consumption of edge nodes
        rf_power: RF power consumption rates
        upload_rates: Upload rates between tiers
        download_rates: Download rates between tiers

    Returns:
        List of (task_idx, target_unit_index, time, energy) tuples
    """
    if kernel_algorithm_func is None or construct_sequence_func is None:
        raise ValueError("Kernel algorithm and construct sequence functions must be provided")

    if energy_calculation_func is None:
        energy_calculation_func = total_energy_consumption_three_tier

    if total_time_func is None:
        total_time_func = lambda tasks: max(
            max(task.FT_l, task.FT_wr) for task in tasks if not task.succ_tasks
        )

    # Find all valid migration options
    migration_options = []

    num_tasks, num_units = migration_choices.shape
    for task_idx in range(num_tasks):
        for target_unit_index in range(num_units):
            if migration_choices[task_idx, target_unit_index]:
                migration_options.append((task_idx, target_unit_index))

    # Shuffle migration options to avoid bias
    # This is important for exploring the search space more effectively
    import random
    random.shuffle(migration_options)

    # Process migrations in batches
    results = []
    for i in range(0, len(migration_options), batch_size):
        batch = migration_options[i:i + batch_size]

        for task_idx, target_unit_index in batch:
            # Evaluate migration
            time_result, energy_result = evaluate_migration_three_tier(
                tasks, sequences, task_idx, target_unit_index,
                migration_cache, kernel_algorithm_func, construct_sequence_func,
                energy_calculation_func, total_time_func,
                num_device_cores, num_edge_nodes, num_edge_cores_per_node,
                device_core_powers, edge_node_powers, rf_power,
                upload_rates, download_rates
            )

            # Store result
            results.append((task_idx, target_unit_index, time_result, energy_result))

    return results


def evaluate_migration_three_tier(tasks,
                                  sequences,
                                  task_idx,
                                  target_unit_index,
                                  migration_cache,
                                  kernel_algorithm_func,
                                  construct_sequence_func,
                                  energy_calculation_func,
                                  total_time_func,
                                  num_device_cores=3,
                                  num_edge_nodes=2,
                                  num_edge_cores_per_node=2,
                                  device_core_powers=None,
                                  edge_node_powers=None,
                                  rf_power=None,
                                  upload_rates=None,
                                  download_rates=None):
    """
    Evaluates a potential task migration in the three-tier architecture.
    """
    # Convert task_idx from 0-based to 1-based task ID
    task_id = task_idx + 1

    # Start measuring computation time (for performance benchmarking)
    eval_start_time = time_module.time() if 'time_module' in globals() else time_module.time()

    # Step 1: Get source execution unit for the task
    source_unit = get_current_execution_unit(tasks[task_idx])

    # Step 2: Get target execution unit from index
    target_unit = get_execution_unit_from_index(
        target_unit_index,
        num_device_cores,
        num_edge_nodes,
        num_edge_cores_per_node
    )

    # Step 3: Check if this migration has been evaluated before
    cache_key = generate_migration_cache_key(
        tasks, task_id, source_unit, target_unit
    )

    # Use the proper method to get from cache based on the object type
    if hasattr(migration_cache, 'get') and callable(migration_cache.get):
        # It's a MigrationCache object with a get method
        cached_result = migration_cache.get(cache_key)
    elif isinstance(migration_cache, dict):
        # It's a regular dictionary
        cached_result = migration_cache.get(cache_key)
    else:
        # Unsupported cache type
        logger.warning(f"Unsupported migration cache type: {type(migration_cache)}")
        cached_result = None

    if cached_result:
        return cached_result

    # Step 4: Create deep copies to avoid modifying original state
    tasks_copy = deepcopy(tasks)
    sequences_copy = [seq.copy() for seq in sequences]

    # Step 5: Apply migration by constructing new sequence
    try:
        sequences_copy = construct_sequence_func(
            tasks_copy,
            task_id,
            target_unit_index,
            sequences_copy,
            num_device_cores,
            num_edge_nodes,
            num_edge_cores_per_node
        )
    except Exception as e:
        logger.error(f"Error constructing sequence for task {task_id} to unit {target_unit}: {e}")
        # Return infinity values to indicate invalid migration
        return float('inf'), float('inf')

    # Step 6: Apply kernel algorithm to recalculate schedule
    try:
        kernel_algorithm_func(
            tasks_copy,
            sequences_copy,
            num_device_cores,
            num_edge_nodes,
            num_edge_cores_per_node
        )
    except Exception as e:
        logger.error(f"Error in kernel algorithm for task {task_id} to unit {target_unit}: {e}")
        return float('inf'), float('inf')

    # Step 7: Calculate new metrics
    try:
        migration_time = total_time_func(tasks_copy)
        migration_energy = energy_calculation_func(
            tasks_copy,
            device_core_powers=device_core_powers,
            edge_node_powers=edge_node_powers,
            rf_power=rf_power,
            upload_rates=upload_rates,
            download_rates=download_rates
        )
    except Exception as e:
        logger.error(f"Error calculating metrics for task {task_id} to unit {target_unit}: {e}")
        return float('inf'), float('inf')

    # Step 8: Cache results - use the proper method based on the object type
    result = (migration_time, migration_energy)
    if hasattr(migration_cache, 'put') and callable(migration_cache.put):
        # It's a MigrationCache object with a put method
        migration_cache.put(cache_key, result)
    elif isinstance(migration_cache, dict):
        # It's a regular dictionary
        migration_cache[cache_key] = result
    else:
        # Unsupported cache type - log warning but continue
        logger.warning(f"Cannot cache result - unsupported migration cache type: {type(migration_cache)}")

    # For performance benchmarking
    eval_time = (time_module.time() if 'time_module' in globals() else time_module.time()) - eval_start_time
    if eval_time > 0.1:  # Log slow evaluations
        logger.debug(f"Slow migration evaluation: {eval_time:.3f}s for task {task_id} to {target_unit}")

    return migration_time, migration_energy


def generate_migration_cache_key(tasks: List[Any],
                                 task_id: int,
                                 source_unit: ExecutionUnit,
                                 target_unit: ExecutionUnit) -> Tuple:
    """
    Generates a cache key for memoizing migration evaluations.

    Args:
        tasks: List of all tasks
        task_id: ID of task being migrated (1-based)
        source_unit: Source execution unit
        target_unit: Target execution unit

    Returns:
        Tuple that uniquely identifies this migration scenario
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
    Encode an execution unit into a hashable tuple.

    Args:
        unit: ExecutionUnit to encode

    Returns:
        Tuple representation of the execution unit
    """
    tier_value = unit.tier.value

    if unit.tier == ExecutionTier.DEVICE:
        return (tier_value, unit.location[0])  # (DEVICE, core_id)
    elif unit.tier == ExecutionTier.EDGE:
        return (tier_value, unit.location[0], unit.location[1])  # (EDGE, node_id, core_id)
    else:
        return (tier_value,)  # (CLOUD,)


def encode_task_assignment(task: Any) -> Tuple:
    """
    Encode a task's current assignment into a hashable tuple.

    Args:
        task: Task object

    Returns:
        Tuple representation of the task's assignment
    """
    if hasattr(task, 'execution_tier'):
        # Three-tier task
        tier = task.execution_tier.value

        if task.execution_tier == ExecutionTier.DEVICE:
            return (tier, task.device_core)
        elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
            return (tier, task.edge_assignment.edge_id, task.edge_assignment.core_id)
        else:
            return (tier,)
    else:
        # Original MCC task
        if hasattr(task, 'is_core_task') and task.is_core_task:
            return (0, task.assignment)  # (DEVICE, core_id)
        else:
            return (2,)  # (CLOUD,)


def calculate_energy_consumption_three_tier(task: Any,
                                            device_core_powers: Dict[int, float] = None,
                                            edge_node_powers: Dict[Tuple[int, int], float] = None,
                                            rf_power: Dict[str, float] = None,
                                            upload_rates: Dict[str, float] = None,
                                            download_rates: Dict[str, float] = None) -> float:
    if device_core_powers is None:
        device_core_powers = {0: 30.0, 1: 40.0, 2: 50.0}
    if edge_node_powers is None:
        edge_node_powers = {(1, 1): 0.5, (1, 2): 0.6, (2, 1): 0.4, (2, 2): 0.5}
    if rf_power is None:
        rf_power = {'device_to_cloud': 0.05, 'edge1_to_cloud': 0.05, 'edge2_to_cloud': 0.05}

    if hasattr(task, 'execution_tier'):
        execution_tier = task.execution_tier
    else:
        execution_tier = ExecutionTier.DEVICE if hasattr(task,
                                                         'is_core_task') and task.is_core_task else ExecutionTier.CLOUD

    if execution_tier == ExecutionTier.DEVICE:
        core_id = task.device_core if hasattr(task, 'device_core') else getattr(task, 'assignment', 0)
        core_power = device_core_powers.get(core_id, 30.0)
        exec_time = task.local_execution_times[core_id] if task.local_execution_times and core_id < len(
            task.local_execution_times) else 5.0
        return core_power * exec_time

    elif execution_tier == ExecutionTier.EDGE:
        if not task.edge_assignment:
            return 0.0
        edge_id = task.edge_assignment.edge_id
        core_id = task.edge_assignment.core_id
        edge_power = edge_node_powers.get((edge_id, core_id), 0.5)
        exec_time = task.get_edge_execution_time(edge_id, core_id) or 3.0
        return edge_power * exec_time

    elif execution_tier == ExecutionTier.CLOUD:
        # Use actual cloud timing without a bias factor.
        rf_power_value = rf_power.get('device_to_cloud', 0.05)
        upload_time = task.cloud_execution_times[0] if task.cloud_execution_times else 3.0
        return rf_power_value * upload_time

    return 0.0


if __name__ == '__main__':
    # Create a sample task graph similar to the original framework.
    # Tasks 1–10 with various predecessor–successor relationships.
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

    # Use primary_assignment to decide the initial execution tier for each task.
    primary_assignment(tasks, edge_nodes=2)

    # Compute task priority scores for scheduling order.
    task_prioritizing(tasks)

    # Create a three-tier scheduler instance.
    # For this example, assume 3 device cores, 2 edge nodes (each with 2 cores).
    scheduler = ThreeTierTaskScheduler(tasks, num_cores=3, num_edge_nodes=2, edge_cores_per_node=2)

    # Classify tasks: entry tasks (no predecessors) and non–entry tasks.
    entry_tasks = [task for task in tasks if not task.pred_tasks]
    non_entry_tasks = [task for task in tasks if task.pred_tasks]

    # Schedule entry tasks first.
    scheduler.schedule_entry_tasks(entry_tasks)
    # Then schedule non–entry tasks.
    scheduler.schedule_non_entry_tasks(non_entry_tasks)

    # After initial scheduling, calculate overall completion time and energy consumption.
    T_initial = total_time(tasks)

    # Define power and rate dictionaries for energy calculations.
    device_core_powers = {0: 30.0, 1: 40.0, 2: 50.0}
    edge_node_powers = {(1, 1): 0.5, (1, 2): 0.6, (2, 1): 0.4, (2, 2): 0.5}
    rf_power = {
        'device_to_cloud': 0.05,
        'edge1_to_cloud': 0.05,
        'edge2_to_cloud': 0.05,
        'device_to_edge1': 0.4,
        'device_to_edge2': 0.45,
        'edge1_to_device': 0.3,
        'edge2_to_device': 0.35
    }
    upload_rates = {
        'device_to_edge1': 2.0,
        'device_to_edge2': 1.8,
        'device_to_cloud': 1.5,
        'edge1_to_cloud': 4.0,
        'edge2_to_cloud': 3.8
    }
    download_rates = {
        'cloud_to_device': 2.0,
        'cloud_to_edge1': 4.5,
        'cloud_to_edge2': 4.2,
        'edge1_to_device': 3.0,
        'edge2_to_device': 2.8
    }

    E_initial = total_energy_consumption_three_tier(
        tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )

    print("\nINITIAL SCHEDULE:")
    print(f"Completion Time: {T_initial}")
    print(f"Energy Consumption: {E_initial}")
    print("Execution Sequences (per unit):")
    for i, seq in enumerate(scheduler.sequences):
        print(f"Unit {i}: {seq}")

    # Optimize the schedule using the migration optimization framework.
    optimized_tasks, optimized_sequences, opt_metrics = optimize_task_scheduling_three_tier(
        tasks, scheduler.sequences, T_initial,
        num_device_cores=3, num_edge_nodes=2, num_edge_cores_per_node=2,
        device_core_powers=device_core_powers, edge_node_powers=edge_node_powers,
        rf_power=rf_power, upload_rates=upload_rates, download_rates=download_rates
    )

    # After optimization, recalculate completion time and energy.
    T_final = total_time(optimized_tasks)
    E_final = total_energy_consumption_three_tier(
        optimized_tasks, device_core_powers, edge_node_powers, rf_power, upload_rates, download_rates
    )

    print("\nFINAL SCHEDULE AFTER OPTIMIZATION:")
    print(f"Completion Time: {T_final}")
    print(f"Energy Consumption: {E_final}")
    print("Final Execution Sequences:")
    for i, seq in enumerate(optimized_sequences):
        print(f"Unit {i}: {seq}")

    # Validate the final schedule.
    validation_result = validate_three_tier_schedule(
        optimized_tasks, optimized_sequences,
        num_device_cores=3, num_edge_nodes=2, num_edge_cores_per_node=2
    )
    if validation_result["valid"]:
        print("\nFinal schedule validation: ✓ All constraints satisfied!")
    else:
        print("\nFinal schedule validation: ✗ Violations detected:")
        for issue in validation_result["issues"]:
            print(issue)
