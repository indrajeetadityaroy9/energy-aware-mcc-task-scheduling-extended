# q_learning_mcc.py
import numpy as np
import random
from copy import deepcopy
import time
import pickle
import os
from collections import defaultdict, namedtuple

from data import ExecutionTier
# Import necessary components from mcc_extended.py
from mcc_extended import (
    Task, SequenceManager, primary_assignment, task_prioritizing,
    ThreeTierTaskScheduler, total_time_3tier, total_energy_3tier_with_rf,
    generate_realistic_network_conditions, generate_realistic_power_models,
    add_task_attributes, format_schedule_3tier, validate_task_dependencies, get_current_execution_unit,
    get_execution_unit_from_index, construct_sequence, kernel_algorithm_3tier
)


class QScheduler:
    """
    Q-learning-based scheduler for task migration in three-tier MCC environments.
    Specifically targets the energy optimization phase while maintaining time constraints.
    """

    def __init__(self,
                 alpha=0.1,  # Learning rate
                 gamma=0.9,  # Discount factor
                 epsilon=0.1,  # Exploration rate
                 time_penalty=100.0,  # Penalty for time constraint violations
                 max_episodes=100,  # Maximum number of training episodes
                 max_iterations=500,  # Maximum iterations per episode
                 q_table_path=None):  # Path to load/save Q-table

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.time_penalty = time_penalty
        self.max_episodes = max_episodes
        self.max_iterations = max_iterations
        self.q_table = defaultdict(float)  # Initialize empty Q-table
        self.q_table_path = q_table_path

        # Load Q-table if path is specified and file exists
        if q_table_path and os.path.exists(q_table_path):
            try:
                with open(q_table_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Loaded Q-table from {q_table_path} with {len(self.q_table)} entries")
            except Exception as e:
                print(f"Error loading Q-table: {e}")

    def save_q_table(self):
        """Save Q-table to disk for future use"""
        if self.q_table_path:
            try:
                with open(self.q_table_path, 'wb') as f:
                    pickle.dump(self.q_table, f)
                print(f"Saved Q-table to {self.q_table_path} with {len(self.q_table)} entries")
            except Exception as e:
                print(f"Error saving Q-table: {e}")

    def get_state_hash(self, tasks):
        """
        Create a hashable representation of the current task assignments.
        Focuses on execution tiers and locations to keep state space manageable.
        """
        state_components = []
        for task in tasks:
            if task.execution_tier == ExecutionTier.DEVICE:
                state_components.append((task.id, 0, task.device_core))
            elif task.execution_tier == ExecutionTier.EDGE and task.edge_assignment:
                state_components.append((
                    task.id,
                    1,
                    task.edge_assignment.edge_id,
                    task.edge_assignment.core_id
                ))
            elif task.execution_tier == ExecutionTier.CLOUD:
                state_components.append((task.id, 2))

        # Sort by task ID to ensure consistent ordering
        return tuple(sorted(state_components))

    def get_possible_actions(self, tasks, sequence_manager):
        """
        Identify all valid migration actions from the current state.
        Each action is a (task_id, target_unit_idx) pair.
        """
        actions = []
        for task in tasks:
            current_unit = get_current_execution_unit(task)

            # Calculate total execution units
            total_units = (
                    sequence_manager.num_device_cores +
                    sequence_manager.num_edge_nodes * sequence_manager.num_edge_cores_per_node +
                    1  # Cloud
            )

            # Consider all possible target units
            for target_unit_idx in range(total_units):
                target_unit = get_execution_unit_from_index(
                    target_unit_idx, sequence_manager
                )

                # Skip self-migration
                if current_unit.tier == target_unit.tier:
                    if (current_unit.tier == ExecutionTier.DEVICE and
                            current_unit.location[0] == target_unit.location[0]):
                        continue
                    elif (current_unit.tier == ExecutionTier.EDGE and
                          current_unit.location == target_unit.location):
                        continue
                    elif current_unit.tier == ExecutionTier.CLOUD:
                        continue

                # Add valid migration action
                actions.append((task.id, target_unit_idx))

        return actions

    def evaluate_action(self, tasks, sequence_manager, task_id, target_unit_idx,
                        power_models, upload_rates, download_rates, T_max):
        """
        Evaluate the effect of a single migration action on energy and time.
        Returns new tasks, sequence_manager, time, energy, and reward.
        """
        # Create deep copies to avoid modifying originals
        temp_tasks = deepcopy(tasks)
        temp_seq_manager = deepcopy(sequence_manager)

        # Find the task to migrate
        task_to_migrate = next((t for t in temp_tasks if t.id == task_id), None)
        if not task_to_migrate:
            return tasks, sequence_manager, float('inf'), float('inf'), -float('inf')

        # Get source and target units
        source_unit = get_current_execution_unit(task_to_migrate)
        target_unit = get_execution_unit_from_index(target_unit_idx, temp_seq_manager)

        # Execute migration
        temp_seq_manager = construct_sequence(
            temp_tasks, task_id, source_unit, target_unit, temp_seq_manager
        )
        temp_tasks = kernel_algorithm_3tier(
            temp_tasks, temp_seq_manager, upload_rates, download_rates
        )

        # Calculate new metrics
        new_time = total_time_3tier(temp_tasks)
        new_energy = total_energy_3tier_with_rf(
            temp_tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        # Calculate reward (energy reduction with time constraint penalty)
        # For original tasks
        original_energy = total_energy_3tier_with_rf(
            tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        energy_reduction = original_energy - new_energy
        time_violation = max(0, new_time - T_max)

        # Reward is energy reduction minus time penalty
        reward = energy_reduction - (time_violation * self.time_penalty)

        return temp_tasks, temp_seq_manager, new_time, new_energy, reward

    def select_action(self, state, available_actions):
        """
        Select action using epsilon-greedy strategy.
        With probability epsilon, choose random action for exploration.
        Otherwise, choose best action based on Q-values.
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(available_actions) if available_actions else None
        else:
            # Exploitation: best action based on Q-values
            if not available_actions:
                return None

            # Find action with highest Q-value
            best_q = -float('inf')
            best_actions = []

            for action in available_actions:
                q_value = self.q_table[(state, action)]
                if q_value > best_q:
                    best_q = q_value
                    best_actions = [action]
                elif q_value == best_q:
                    best_actions.append(action)

            # If multiple best actions, randomly choose one
            return random.choice(best_actions)

    def learn(self, tasks, sequence_manager, T_max, power_models,
              upload_rates, download_rates):
        """
        Main Q-learning training loop.
        Runs multiple episodes to learn optimal migration policy.
        """
        print(f"Starting Q-learning training with {self.max_episodes} episodes...")
        start_time = time.time()

        # Metrics tracking
        episode_rewards = []
        episode_energies = []
        episode_times = []

        # Initial metrics
        initial_time = total_time_3tier(tasks)
        initial_energy = total_energy_3tier_with_rf(
            tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        best_energy = initial_energy
        best_tasks = None
        best_seq_manager = None

        print(f"Initial time: {initial_time:.2f}, Initial energy: {initial_energy:.2f}")

        # Training loop - multiple episodes
        for episode in range(self.max_episodes):
            episode_start = time.time()

            # Reset environment for new episode
            current_tasks = deepcopy(tasks)
            current_seq_manager = deepcopy(sequence_manager)

            # Training performance metrics
            total_reward = 0
            iterations = 0
            migrations = 0

            current_time = initial_time
            current_energy = initial_energy

            # Run a single episode
            done = False
            while not done and iterations < self.max_iterations:
                iterations += 1

                # Get current state
                state = self.get_state_hash(current_tasks)

                # Get possible actions
                actions = self.get_possible_actions(current_tasks, current_seq_manager)
                if not actions:
                    break  # No valid migrations left

                # Select action using epsilon-greedy
                action = self.select_action(state, actions)
                if action is None:
                    break

                # Execute action and get reward
                task_id, target_unit_idx = action
                next_tasks, next_seq_manager, new_time, new_energy, reward = self.evaluate_action(
                    current_tasks, current_seq_manager, task_id, target_unit_idx,
                    power_models, upload_rates, download_rates, T_max
                )

                # Get next state
                next_state = self.get_state_hash(next_tasks)

                # Update Q-value using Q-learning update rule
                # Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

                # Get max Q-value for next state
                next_actions = self.get_possible_actions(next_tasks, next_seq_manager)
                max_next_q = max(
                    [self.q_table[(next_state, next_a)] for next_a in next_actions],
                    default=0
                )

                # Update Q-value
                self.q_table[(state, action)] = (1 - self.alpha) * self.q_table[(state, action)] + \
                                                self.alpha * (reward + self.gamma * max_next_q)

                # Track migration
                if reward > 0:
                    migrations += 1

                # Update state and metrics
                current_tasks = next_tasks
                current_seq_manager = next_seq_manager
                current_time = new_time
                current_energy = new_energy
                total_reward += reward

                # Check if energy improved
                if current_energy < best_energy and current_time <= T_max:
                    best_energy = current_energy
                    best_tasks = deepcopy(current_tasks)
                    best_seq_manager = deepcopy(current_seq_manager)

                # Check termination conditions
                if iterations >= self.max_iterations or not next_actions:
                    done = True

            # End of episode
            episode_duration = time.time() - episode_start
            episode_rewards.append(total_reward)
            episode_energies.append(current_energy)
            episode_times.append(current_time)

            # Log progress
            print(f"Episode {episode + 1}/{self.max_episodes}: " +
                  f"Time={current_time:.2f}, Energy={current_energy:.2f}, " +
                  f"Reward={total_reward:.2f}, Migrations={migrations}, " +
                  f"Duration={episode_duration:.2f}s")

            # Decay exploration rate
            self.epsilon = max(0.01, self.epsilon * 0.95)

        # Save Q-table
        self.save_q_table()

        # Return best solution found during training
        if best_tasks is None:
            best_tasks = tasks
            best_seq_manager = sequence_manager

        training_time = time.time() - start_time
        print(f"Q-learning training completed in {training_time:.2f} seconds")
        print(f"Initial energy: {initial_energy:.2f}, Best energy: {best_energy:.2f}")
        print(f"Energy reduction: {(initial_energy - best_energy) / initial_energy * 100:.2f}%")

        return best_tasks, best_seq_manager

    def execute_learned_policy(self, tasks, sequence_manager, T_max, power_models,
                               upload_rates, download_rates):
        """
        Execute the learned policy without exploration.
        Uses Q-values to make optimal migration decisions.
        """
        print("Executing learned policy...")

        # Create copies to avoid modifying originals
        current_tasks = deepcopy(tasks)
        current_seq_manager = deepcopy(sequence_manager)

        # Initial metrics
        initial_time = total_time_3tier(current_tasks)
        initial_energy = total_energy_3tier_with_rf(
            current_tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        print(f"Initial time: {initial_time:.2f}, Initial energy: {initial_energy:.2f}")

        # Performance tracking
        iterations = 0
        migrations = 0

        # Execute policy without exploration (epsilon=0)
        old_epsilon = self.epsilon
        self.epsilon = 0

        try:
            # Keep making migrations until no more beneficial ones exist
            while iterations < self.max_iterations:
                iterations += 1

                # Get current state and actions
                state = self.get_state_hash(current_tasks)
                actions = self.get_possible_actions(current_tasks, current_seq_manager)

                if not actions:
                    break  # No valid migrations left

                # Select best action based on learned Q-values
                action = self.select_action(state, actions)
                if action is None:
                    break

                # Execute action
                task_id, target_unit_idx = action
                next_tasks, next_seq_manager, new_time, new_energy, reward = self.evaluate_action(
                    current_tasks, current_seq_manager, task_id, target_unit_idx,
                    power_models, upload_rates, download_rates, T_max
                )

                # Only apply migration if it improves energy and meets time constraint
                if new_energy < total_energy_3tier_with_rf(
                        current_tasks, power_models['device'],
                        power_models['rf'], upload_rates) and new_time <= T_max:
                    current_tasks = next_tasks
                    current_seq_manager = next_seq_manager
                    migrations += 1
                    print(f"  Migration {migrations}: Task {task_id} moved to unit {target_unit_idx}, " +
                          f"New energy: {new_energy:.2f}, New time: {new_time:.2f}")
                else:
                    # Skip this migration as it's not beneficial
                    continue

                # Check termination condition
                if iterations >= self.max_iterations:
                    break
        finally:
            # Restore original epsilon
            self.epsilon = old_epsilon

        # Final metrics
        final_time = total_time_3tier(current_tasks)
        final_energy = total_energy_3tier_with_rf(
            current_tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        print(f"Execution completed with {migrations} migrations")
        print(f"Final time: {final_time:.2f}, Final energy: {final_energy:.2f}")
        print(f"Energy reduction: {(initial_energy - final_energy) / initial_energy * 100:.2f}%")

        return current_tasks, current_seq_manager


# Main function to demonstrate usage
def optimize_with_q_learning(
        tasks,
        sequence_manager,
        T_max,
        power_models,
        upload_rates,
        download_rates,
        q_table_path='q_table.pkl',
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1,
        time_penalty=100.0,
        max_episodes=100,
        max_iterations=500,
        train_new_model=True):
    """
    Main function to optimize task scheduling using Q-learning.

    Args:
        tasks: List of Task objects
        sequence_manager: SequenceManager for the three-tier environment
        T_max: Maximum allowed completion time
        power_models: Dictionary of power models for different tiers
        upload_rates: Dictionary of upload rates for different connections
        download_rates: Dictionary of download rates for different connections
        q_table_path: Path to save/load Q-table
        learning_rate: Alpha parameter for Q-learning
        discount_factor: Gamma parameter for Q-learning
        exploration_rate: Epsilon parameter for exploration-exploitation
        time_penalty: Penalty factor for time constraint violations
        max_episodes: Maximum number of training episodes
        max_iterations: Maximum iterations per episode
        train_new_model: Whether to train a new model or use existing Q-table

    Returns:
        Tuple of (optimized_tasks, optimized_sequence_manager)
    """
    # Initialize Q-learning scheduler
    q_scheduler = QScheduler(
        alpha=learning_rate,
        gamma=discount_factor,
        epsilon=exploration_rate,
        time_penalty=time_penalty,
        max_episodes=max_episodes,
        max_iterations=max_iterations,
        q_table_path=q_table_path
    )

    # Train model if requested
    if train_new_model:
        print("Training Q-learning model...")
        optimized_tasks, optimized_seq_manager = q_scheduler.learn(
            tasks, sequence_manager, T_max, power_models, upload_rates, download_rates
        )
    else:
        # Execute learned policy without additional training
        print("Using existing Q-table without further training...")
        optimized_tasks, optimized_seq_manager = q_scheduler.execute_learned_policy(
            tasks, sequence_manager, T_max, power_models, upload_rates, download_rates
        )

    return optimized_tasks, optimized_seq_manager


# Example usage in context
if __name__ == "__main__":

    # 1) Setup the environment
    upload_rates, download_rates = generate_realistic_network_conditions()
    mobile_power_models = generate_realistic_power_models(device_type='mobile', battery_level=65)

    # 2) Define example tasks (create a simple task graph)
    task3 = Task(id=3, succ_task=[])
    task2 = Task(id=2, succ_task=[task3])
    task1 = Task(id=1, succ_task=[task2, task3])

    task1.pred_tasks = []
    task2.pred_tasks = [task1]
    task3.pred_tasks = [task1, task2]

    tasks = [task1, task2, task3]
    tasks = add_task_attributes(tasks)

    # 3) Run the initial scheduling algorithm
    primary_assignment(tasks)
    task_prioritizing(tasks)

    scheduler = ThreeTierTaskScheduler(
        tasks,
        num_cores=3,
        num_edge_nodes=2,
        edge_cores_per_node=2,
        upload_rates=upload_rates,
        download_rates=download_rates
    )

    scheduler.schedule_tasks_topo_priority()

    # 4) Calculate initial metrics
    initial_time = total_time_3tier(tasks)
    initial_energy = total_energy_3tier_with_rf(
        tasks=tasks,
        device_power_profiles=mobile_power_models['device'],
        rf_power=mobile_power_models['rf'],
        upload_rates=upload_rates
    )

    print(f"Initial time: {initial_time:.2f}, Initial energy: {initial_energy:.2f}")

    # 5) Create sequence manager
    sequence_manager = SequenceManager(
        num_device_cores=3,
        num_edge_nodes=2,
        num_edge_cores_per_node=2
    ).build_sequences_from_tasks(tasks)

    # 6) Run Q-learning optimization
    T_max = initial_time * 1.2  # Allow 20% increase in completion time

    power_models = {
        'device': mobile_power_models['device'],
        'rf': mobile_power_models['rf']
    }

    optimized_tasks, optimized_seq_manager = optimize_with_q_learning(
        tasks=tasks,
        sequence_manager=sequence_manager,
        T_max=T_max,
        power_models=power_models,
        upload_rates=upload_rates,
        download_rates=download_rates,
        q_table_path='q_table.pkl',
        max_episodes=50,  # Reduced for demonstration
        train_new_model=True
    )

    # 7) Calculate final metrics
    final_time = total_time_3tier(optimized_tasks)
    final_energy = total_energy_3tier_with_rf(
        tasks=optimized_tasks,
        device_power_profiles=mobile_power_models['device'],
        rf_power=mobile_power_models['rf'],
        upload_rates=upload_rates
    )

    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Initial time: {initial_time:.2f}, Final time: {final_time:.2f}")
    print(f"Initial energy: {initial_energy:.2f}, Final energy: {final_energy:.2f}")
    print(f"Energy reduction: {(initial_energy - final_energy) / initial_energy * 100:.2f}%")

    # 8) Validate the optimized schedule
    is_valid, violations = validate_task_dependencies(optimized_tasks)
    print(f"\nOptimized schedule is valid: {is_valid}")
    if not is_valid:
        for v in violations:
            print(f"  - {v['detail']}")