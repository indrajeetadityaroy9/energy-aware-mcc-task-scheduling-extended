# File 1: gnn_marl_integration.py
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch.distributions import Categorical
import random
import numpy as np
from collections import deque
import copy

from data import ExecutionTier, SchedulingState, add_task_attributes, generate_realistic_power_models, \
    generate_realistic_network_conditions
from mcc_extended import total_time_3tier, total_energy_3tier_with_rf, EdgeAssignment, construct_sequence, \
    get_current_execution_unit, get_execution_unit_from_index, \
    ThreeTierKernelScheduler, primary_assignment, task_prioritizing, ThreeTierTaskScheduler, Task, SequenceManager, \
    MigrationCache
from utils import validate_task_dependencies, format_schedule_3tier


# GNN Model for Task Representation
class TaskGNN(pyg_nn.MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TaskGNN, self).__init__(aggr='mean')
        self.lin_start = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.lin_start(x)
        x = self.activation(x)
        x = self.propagate(edge_index, x=x)
        x = self.lin_hidden(x)
        x = self.activation(x)
        x = self.lin_out(x)
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


# Multi-Agent Reinforcement Learning Agent
class MARLAgent:
    def __init__(self, gnn_model, policy_net, device):
        self.gnn_model = gnn_model
        self.policy_net = policy_net
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.gnn_model.parameters()) + list(self.policy_net.parameters()),
            lr=0.001
        )

    def select_action(self, state):
        with torch.no_grad():
            gnn_output = self.gnn_model(state.x.to(self.device), state.edge_index.to(self.device))
            policy_input = torch.cat([gnn_output, state.resource_info.to(self.device)], dim=1)
            action_logits = self.policy_net(policy_input)
            action_probs = torch.softmax(action_logits, dim=1)
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            dist = Categorical(action_probs)
            action = dist.sample()
        return action  # Return the tensor directly


def apply_action(tasks, sequence_manager, actions):
    new_tasks = copy.deepcopy(tasks)
    new_sequence_manager = copy.deepcopy(sequence_manager)

    # Assuming actions is a tensor where each element corresponds to a task
    for task_idx, task in enumerate(new_tasks):
        action = actions[task_idx].item()  # Convert tensor to integer

        # Determine target execution unit based on action
        num_device_cores = new_sequence_manager.num_device_cores
        num_edge_nodes = new_sequence_manager.num_edge_nodes
        num_edge_cores_per_node = new_sequence_manager.num_edge_cores_per_node

        if action == 0:
            continue  # No change

        if 1 <= action <= num_device_cores:
            task.execution_tier = ExecutionTier.DEVICE
            task.device_core = action - 1
            task.edge_assignment = None
        elif num_device_cores < action <= num_device_cores + num_edge_nodes * num_edge_cores_per_node:
            offset = num_device_cores
            edge_core_idx = action - offset - 1
            edge_id = edge_core_idx // num_edge_cores_per_node
            core_id = edge_core_idx % num_edge_cores_per_node
            task.execution_tier = ExecutionTier.EDGE
            task.device_core = -1
            task.edge_assignment = EdgeAssignment(edge_id + 1, core_id + 1)
        else:
            task.execution_tier = ExecutionTier.CLOUD
            task.device_core = -1
            task.edge_assignment = None

    # Update sequence manager (Note: This part may need adjustment based on your logic)
    # The original code used 'task.id' which refers to the last task in the loop.
    # Ensure this is handled correctly for your use case.
    new_sequence_manager = construct_sequence(
        new_tasks,
        task.id,
        get_current_execution_unit(task),
        get_execution_unit_from_index(action - 1, new_sequence_manager),
        new_sequence_manager
    )

    return new_tasks, new_sequence_manager


# State Representation Class
class State:
    def __init__(self, x, edge_index, resource_info):
        self.x = x
        self.edge_index = edge_index
        self.resource_info = resource_info

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.resource_info = self.resource_info.to(device)
        return self


# GNN-based MARL Optimization
def gnn_marl_optimize_task_scheduling(tasks, sequence_manager, T_final,
                                      power_models, upload_rates, download_rates,
                                      migration_cache=None, max_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a sample state to determine actual input dimensions
    sample_state = generate_state(tasks, sequence_manager, power_models, upload_rates, download_rates)
    input_dim = sample_state.x.shape[1]  # Number of task features
    resource_info_dim = sample_state.resource_info.shape[1]  # Number of resource features

    # Initialize GNN and policy network with correct dimensions
    hidden_dim = 64
    output_dim = 4  # Number of possible actions
    gnn_model = TaskGNN(input_dim, hidden_dim, hidden_dim).to(device)
    policy_net = nn.Sequential(
        nn.Linear(hidden_dim + resource_info_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    ).to(device)

    # Create MARL agents (one per execution unit)
    num_device_cores = sequence_manager.num_device_cores
    num_edge_nodes = sequence_manager.num_edge_nodes
    num_edge_cores_per_node = sequence_manager.num_edge_cores_per_node
    total_units = num_device_cores + num_edge_nodes * num_edge_cores_per_node + 1
    agents = [MARLAgent(copy.deepcopy(gnn_model), copy.deepcopy(policy_net), device)
              for _ in range(total_units)]

    # Experience replay buffer
    replay_buffer = deque(maxlen=10000)

    # Training parameters
    batch_size = 32
    episodes = max_iterations  # Use the max_iterations parameter

    for episode in range(episodes):
        # Reset environment
        current_tasks = copy.deepcopy(tasks)
        current_sequence_manager = copy.deepcopy(sequence_manager)
        current_time = total_time_3tier(current_tasks)
        current_energy = total_energy_3tier_with_rf(
            current_tasks,
            power_models['device'],
            power_models['rf'],
            upload_rates
        )

        # Generate initial state
        state = generate_state(current_tasks, current_sequence_manager, power_models, upload_rates, download_rates)

        # Collect experience
        for agent in agents:
            action = agent.select_action(state)
            # Apply action (task migration)
            new_tasks, new_sequence_manager = apply_action(current_tasks, current_sequence_manager, action)
            new_time = total_time_3tier(new_tasks)
            new_energy = total_energy_3tier_with_rf(
                new_tasks,
                power_models['device'],
                power_models['rf'],
                upload_rates
            )
            reward = calculate_reward(current_time, current_energy, new_time, new_energy)
            next_state = generate_state(new_tasks, new_sequence_manager, power_models, upload_rates, download_rates)
            done = False  # We'll run for fixed episodes

            # Use migration cache if provided
            if migration_cache:
                cache_key = (tuple(current_tasks), action)
                cached_result = migration_cache.get(cache_key)
                if cached_result:
                    new_time, new_energy = cached_result
                else:
                    migration_cache.put(cache_key, (new_time, new_energy))

            replay_buffer.append((state, action, reward, next_state, done))

            # Update current state
            state = next_state
            current_tasks = new_tasks
            current_sequence_manager = new_sequence_manager
            current_time = new_time
            current_energy = new_energy

        # Training step
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            for i, agent in enumerate(agents):
                agent.update(states, actions, rewards, next_states, dones)

    # After training, apply the optimized policy
    optimized_tasks, optimized_sequence_manager = apply_trained_policy(tasks, sequence_manager, agents)

    return optimized_tasks, optimized_sequence_manager


def generate_state(tasks, sequence_manager, power_models, upload_rates, download_rates):
    """
    Generate a state representation for the current scheduling environment.

    Args:
        tasks: List of Task objects
        sequence_manager: SequenceManager with current execution sequences
        power_models: Dictionary of power consumption models
        upload_rates: Dictionary of upload rates for different connections
        download_rates: Dictionary of download rates for different connections

    Returns:
        State object containing graph representation of tasks and resources
    """
    # Create task feature matrix
    num_tasks = len(tasks)
    x = torch.zeros(num_tasks, 14)  # Increased features to include network and power info
    for i, task in enumerate(tasks):
        x[i, 0] = task.complexity
        x[i, 1] = task.data_intensity

        # Current execution information
        x[i, 2] = task.execution_tier.value
        if task.execution_tier == ExecutionTier.DEVICE:
            x[i, 3] = task.device_core
        elif task.execution_tier == ExecutionTier.EDGE:
            x[i, 3] = task.edge_assignment.edge_id if task.edge_assignment else -1
            x[i, 4] = task.edge_assignment.core_id if task.edge_assignment else -1
        else:
            x[i, 3] = -1
            x[i, 4] = -1

        # Timing information
        x[i, 5] = task.execution_finish_time
        x[i, 6] = task.FT_l if hasattr(task, 'FT_l') else -1
        x[i, 7] = task.FT_ws if hasattr(task, 'FT_ws') else -1
        x[i, 8] = task.FT_wr if hasattr(task, 'FT_wr') else -1

        # Network rates
        if task.execution_tier == ExecutionTier.DEVICE:
            x[i, 9] = upload_rates.get('device_to_edge', 0.0)
            x[i, 10] = upload_rates.get('device_to_cloud', 0.0)
            x[i, 11] = download_rates.get('edge_to_device', 0.0)
            x[i, 12] = download_rates.get('cloud_to_device', 0.0)
        elif task.execution_tier == ExecutionTier.EDGE:
            edge_id = task.edge_assignment.edge_id if task.edge_assignment else 0
            x[i, 9] = upload_rates.get(f'edge{edge_id}_to_cloud', 0.0)
            x[i, 10] = download_rates.get(f'cloud_to_edge{edge_id}', 0.0)
            x[i, 11] = np.mean([v for k, v in upload_rates.items() if k.startswith('edge')])
            x[i, 12] = np.mean([v for k, v in download_rates.items() if k.startswith('edge')])
        else:
            x[i, 9] = upload_rates.get('cloud_to_device', 0.0)
            x[i, 10] = download_rates.get('device_to_cloud', 0.0)
            x[i, 11] = upload_rates.get('cloud_to_edge', 0.0)
            x[i, 12] = download_rates.get('edge_to_cloud', 0.0)

        # Power model information
        if task.execution_tier == ExecutionTier.DEVICE:
            core_id = task.device_core
            if core_id >= 0 and core_id < len(power_models['device']):
                power_model = power_models['device'][core_id].get('dynamic_power', 0.0)
                x[i, 13] = power_model(1.0) if callable(power_model) else power_model
            else:
                x[i, 13] = 0.0
        elif task.execution_tier == ExecutionTier.EDGE:
            power_model = power_models.get('edge', {}).get('dynamic_power', 0.0)
            x[i, 13] = power_model(1.0) if callable(power_model) else power_model
        else:
            power_model = power_models.get('cloud', {}).get('dynamic_power', 0.0)
            x[i, 13] = power_model(1.0) if callable(power_model) else power_model

    # Create edge index from task dependencies
    edge_index = []
    for task in tasks:
        for pred in task.pred_tasks:
            # Convert to 0-based indexing if tasks are 1-based
            pred_id = pred.id - 1
            task_id = task.id - 1

            # Check if IDs are valid
            if pred_id < 0 or pred_id >= num_tasks:
                raise ValueError(
                    f"Invalid predecessor ID: {pred.id} (0-based: {pred_id}) for task {task.id}, num_tasks={num_tasks}")
            if task_id < 0 or task_id >= num_tasks:
                raise ValueError(f"Invalid task ID: {task.id} (0-based: {task_id}), num_tasks={num_tasks}")

            edge_index.append([pred_id, task_id])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create resource information
    resource_info = torch.zeros(num_tasks, 5)
    for i, task in enumerate(tasks):
        # Convert to 0-based index
        task_id = task.id - 1

        if task.execution_tier == ExecutionTier.DEVICE:
            resource_info[task_id, 0] = 1.0  # Device indicator
            resource_info[task_id, 1] = task.device_core
        elif task.execution_tier == ExecutionTier.EDGE:
            resource_info[task_id, 2] = 1.0  # Edge indicator
            resource_info[task_id, 3] = task.edge_assignment.edge_id if task.edge_assignment else -1
            resource_info[task_id, 4] = task.edge_assignment.core_id if task.edge_assignment else -1
        else:
            resource_info[task_id, 0] = 0.0  # Cloud indicator

    return State(x, edge_index, resource_info)


def apply_action(tasks, sequence_manager, action):
    # Implement action application logic based on your specific action space
    # This is a placeholder implementation
    new_tasks = copy.deepcopy(tasks)
    new_sequence_manager = copy.deepcopy(sequence_manager)

    # Example: action represents migrating a task to a different execution unit
    task_id = action // 100
    target_unit = action % 100

    # Find the task to migrate
    task_to_migrate = next((t for t in new_tasks if t.id == task_id), None)
    if task_to_migrate:
        # Determine target execution unit
        if target_unit < sequence_manager.num_device_cores:
            task_to_migrate.execution_tier = ExecutionTier.DEVICE
            task_to_migrate.device_core = target_unit
            task_to_migrate.edge_assignment = None
        else:
            # Convert to edge node and core
            edge_nodes = sequence_manager.num_edge_nodes
            edge_cores = sequence_manager.num_edge_cores_per_node
            offset = sequence_manager.num_device_cores
            edge_idx = (target_unit - offset) // edge_cores
            core_idx = (target_unit - offset) % edge_cores
            task_to_migrate.execution_tier = ExecutionTier.EDGE
            task_to_migrate.device_core = -1
            task_to_migrate.edge_assignment = EdgeAssignment(edge_idx + 1, core_idx + 1)

    return new_tasks, new_sequence_manager


def calculate_reward(old_time, old_energy, new_time, new_energy):
    # Reward function that balances time and energy considerations
    time_improvement = old_time - new_time
    energy_improvement = old_energy - new_energy
    return 0.7 * time_improvement + 0.3 * energy_improvement  # Weighted combination


def apply_trained_policy(tasks, sequence_manager, agents):
    """
    Apply the trained MARL policy to generate an optimized schedule.

    Args:
        tasks: List of Task objects to optimize
        sequence_manager: SequenceManager with current execution sequences
        agents: List of trained MARL agents

    Returns:
        optimized_tasks: Tasks with optimized scheduling decisions
        optimized_sequence_manager: Updated SequenceManager
    """
    # Create deep copies to avoid modifying original objects
    optimized_tasks = copy.deepcopy(tasks)
    optimized_sequence_manager = copy.deepcopy(sequence_manager)

    # Get system configuration
    num_device_cores = optimized_sequence_manager.num_device_cores
    num_edge_nodes = optimized_sequence_manager.num_edge_nodes
    num_edge_cores_per_node = optimized_sequence_manager.num_edge_cores_per_node
    total_units = num_device_cores + num_edge_nodes * num_edge_cores_per_node + 1

    # Ensure we have enough agents
    if len(agents) != total_units:
        raise ValueError(f"Number of agents ({len(agents)}) must match number of execution units ({total_units})")

    # Process each task to potentially migrate it
    for task in optimized_tasks:
        # Determine which agent to use based on current execution unit
        if task.execution_tier == ExecutionTier.DEVICE:
            if task.device_core < 0 or task.device_core >= num_device_cores:
                # Invalid device core assignment, treat as unscheduled
                current_agent_idx = -1
            else:
                current_agent_idx = task.device_core
        elif task.execution_tier == ExecutionTier.EDGE:
            if not task.edge_assignment:
                # Invalid edge assignment, treat as unscheduled
                current_agent_idx = -1
            else:
                edge_id = task.edge_assignment.edge_id - 1  # Convert to 0-based
                core_id = task.edge_assignment.core_id - 1  # Convert to 0-based
                if (edge_id < 0 or edge_id >= num_edge_nodes or
                        core_id < 0 or core_id >= num_edge_cores_per_node):
                    # Invalid edge assignment, treat as unscheduled
                    current_agent_idx = -1
                else:
                    current_agent_idx = num_device_cores + edge_id * num_edge_cores_per_node + core_id
        else:  # CLOUD
            current_agent_idx = total_units - 1  # Last agent handles cloud decisions

        # If current assignment is invalid, use a default agent
        if current_agent_idx < 0 or current_agent_idx >= len(agents):
            # Use a default agent (could be randomly selected or a specific one)
            current_agent_idx = len(agents) - 1  # Use cloud agent as default

        # Get the appropriate agent
        agent = agents[current_agent_idx]

        # Generate state for this task
        state = generate_state([task], optimized_sequence_manager, power_models, upload_rates, download_rates)

        # Select action using the agent's policy
        action = agent.select_action(state)

        # Apply the action to potentially migrate the task
        optimized_tasks, optimized_sequence_manager = apply_single_action(
            optimized_tasks, optimized_sequence_manager, task.id, action
        )

    # After processing all tasks, run the kernel algorithm to update the schedule
    scheduler = ThreeTierKernelScheduler(
        optimized_tasks,
        optimized_sequence_manager,
        upload_rates=upload_rates,
        download_rates=download_rates
    )

    # Initialize ready queue
    ready_queue = scheduler.initialize_queue()

    # Main scheduling loop
    while ready_queue:
        # Get next ready task from queue
        current_task = ready_queue.popleft()

        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on execution tier
        if current_task.execution_tier == ExecutionTier.DEVICE:
            scheduler.schedule_device_task(current_task)
        elif current_task.execution_tier == ExecutionTier.EDGE:
            scheduler.schedule_edge_task(current_task)
        else:  # ExecutionTier.CLOUD
            scheduler.schedule_cloud_task(current_task)

        # Update dependency and sequence readiness for all tasks
        newly_ready_tasks = []
        for task in optimized_tasks:
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
    for task in optimized_tasks:
        if task.is_scheduled == SchedulingState.KERNEL_SCHEDULED:
            task.is_scheduled = SchedulingState.SCHEDULED

    return optimized_tasks, optimized_sequence_manager


def apply_single_action(tasks, sequence_manager, task_id, action):
    new_tasks = copy.deepcopy(tasks)
    new_sequence_manager = copy.deepcopy(sequence_manager)

    # Convert task_id to a Python integer if it's a tensor
    if isinstance(task_id, torch.Tensor):
        task_id = task_id.item()  # Extract scalar value
    elif not isinstance(task_id, int):
        raise TypeError(f"task_id must be an integer or tensor, got {type(task_id)}")

    # Find the task to migrate
    task_to_migrate = next((t for t in new_tasks if t.id == task_id), None)
    if not task_to_migrate:
        return new_tasks, new_sequence_manager

    # Determine target execution unit based on action
    num_device_cores = new_sequence_manager.num_device_cores
    num_edge_nodes = new_sequence_manager.num_edge_nodes
    num_edge_cores_per_node = new_sequence_manager.num_edge_cores_per_node

    # Action encoding:
    # 0: No change
    # 1 - num_device_cores: Migrate to device core (action-1)
    # num_device_cores+1 to num_device_cores + num_edge_nodes*num_edge_cores_per_node: Migrate to edge
    # Last action: Migrate to cloud

    if action == 0:
        return new_tasks, new_sequence_manager

    if 1 <= action <= num_device_cores:
        task_to_migrate.execution_tier = ExecutionTier.DEVICE
        task_to_migrate.device_core = action - 1
        task_to_migrate.edge_assignment = None
    elif num_device_cores < action <= num_device_cores + num_edge_nodes * num_edge_cores_per_node:
        offset = num_device_cores
        edge_core_idx = action - offset - 1
        edge_id = edge_core_idx // num_edge_cores_per_node
        core_id = edge_core_idx % num_edge_cores_per_node
        task_to_migrate.execution_tier = ExecutionTier.EDGE
        task_to_migrate.device_core = -1
        task_to_migrate.edge_assignment = EdgeAssignment(edge_id + 1, core_id + 1)
    else:
        task_to_migrate.execution_tier = ExecutionTier.CLOUD
        task_to_migrate.device_core = -1
        task_to_migrate.edge_assignment = None

    # Update sequence manager
    new_sequence_manager = construct_sequence(
        new_tasks,
        task_id,
        get_current_execution_unit(task_to_migrate),
        get_execution_unit_from_index(action - 1, new_sequence_manager),
        new_sequence_manager
    )

    return new_tasks, new_sequence_manager


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
        optimized_tasks, optimized_sequence_manager = gnn_marl_optimize_task_scheduling(
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
