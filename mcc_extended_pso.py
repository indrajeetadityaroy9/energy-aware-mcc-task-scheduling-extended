import numpy as np
import random
from copy import deepcopy
import time
import pickle
import os
from collections import defaultdict

from data import generate_realistic_network_conditions, generate_realistic_power_models, add_task_attributes
# Import necessary components from mcc_extended.py
from mcc_extended import (
    Task, ExecutionTier, SchedulingState,
    get_current_execution_unit, get_execution_unit_from_index,
    kernel_algorithm_3tier, construct_sequence, total_time_3tier,
    total_energy_3tier_with_rf, validate_task_dependencies, primary_assignment, task_prioritizing,
    ThreeTierTaskScheduler, SequenceManager
)


class RobustQScheduler:
    """
    Robust Q-learning scheduler that handles Q-table format inconsistencies.
    """

    def __init__(self,
                 alpha=0.1,
                 gamma=0.9,
                 epsilon=0.3,
                 time_penalty=50.0,
                 max_episodes=100,
                 max_iterations=500,
                 q_table_path=None,
                 reset_q_table=False,
                 verbose=True):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.time_penalty = time_penalty
        self.max_episodes = max_episodes
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.q_table = defaultdict(float)
        self.q_table_path = q_table_path

        # Either reset the Q-table or load from disk if available
        if reset_q_table:
            print("Starting with a fresh Q-table")
        elif q_table_path and os.path.exists(q_table_path) and not reset_q_table:
            try:
                with open(q_table_path, 'rb') as f:
                    loaded_table = pickle.load(f)
                    # Convert to defaultdict if it's a regular dict
                    if isinstance(loaded_table, dict):
                        self.q_table = defaultdict(float, loaded_table)
                    else:
                        self.q_table = loaded_table
                print(f"Loaded Q-table from {q_table_path} with {len(self.q_table)} entries")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                print("Starting with a fresh Q-table instead")

    def save_q_table(self, path=None):
        """Save Q-table to disk for future use"""
        save_path = path or self.q_table_path
        if save_path:
            try:
                # Save as a regular dictionary to reduce file size
                with open(save_path, 'wb') as f:
                    pickle.dump(dict(self.q_table), f)
                print(f"Saved Q-table to {save_path} with {len(self.q_table)} entries")
            except Exception as e:
                print(f"Error saving Q-table: {e}")

    def get_state_hash(self, tasks):
        """Create a consistent hashable state representation"""
        state_components = []
        for task in tasks:
            # Include only essential information
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

        # Sort by task ID for consistency
        return tuple(sorted(state_components))

    def get_q_value(self, state, action):
        """
        Safely get Q-value with fallback for missing keys.
        Tries multiple key formats for backward compatibility.
        """
        # Format 1: Direct tuple (state, action)
        key1 = (state, action)

        # Format 2: Flattened key
        key2 = (state, action[0], action[1])

        # Format 3: String hash for compatibility with older versions
        key3 = str(hash(str(state) + str(action)))

        # Try each format in order
        if key1 in self.q_table:
            return self.q_table[key1]
        elif key2 in self.q_table:
            return self.q_table[key2]
        elif key3 in self.q_table:
            return self.q_table[key3]
        else:
            # Not found in any format, return default value
            return 0.0

    def update_q_value(self, state, action, value):
        """Consistently update Q-value using the primary key format"""
        key = (state, action)
        self.q_table[key] = value

    def get_possible_actions(self, tasks, sequence_manager):
        """Generate valid migration actions"""
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

                # Skip self-migration (fixed comparison)
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

        if self.verbose and actions:
            print(f"Found {len(actions)} valid migration actions")
        elif self.verbose:
            print("No valid migration actions found")

        return actions

    def select_action(self, state, available_actions):
        """
        Robust action selection with error handling for Q-table access.
        """
        if not available_actions:
            return None

        # Exploration with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Exploitation with safe Q-value access
        best_q = float('-inf')
        best_actions = []

        for action in available_actions:
            try:
                q_value = self.get_q_value(state, action)

                if q_value > best_q:
                    best_q = q_value
                    best_actions = [action]
                elif q_value == best_q:
                    best_actions.append(action)
            except Exception as e:
                if self.verbose:
                    print(f"Error accessing Q-value for state {state}, action {action}: {e}")
                continue

        # If no valid Q-values found, choose randomly
        if not best_actions:
            return random.choice(available_actions)

        return random.choice(best_actions)

    def evaluate_action(self, tasks, sequence_manager, task_id, target_unit_idx,
                        power_models, upload_rates, download_rates, T_max):
        """Evaluate a migration action and calculate reward"""
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

        # Calculate original energy before migration
        original_energy = total_energy_3tier_with_rf(
            tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        try:
            # Execute migration
            temp_seq_manager = construct_sequence(
                temp_tasks, task_id, source_unit, target_unit, temp_seq_manager
            )

            # Reschedule with kernel algorithm
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

            # Calculate reward
            energy_reduction = original_energy - new_energy
            time_violation = max(0, new_time - T_max)
            time_penalty = time_violation * self.time_penalty

            # Check schedule validity
            is_valid, violations = validate_task_dependencies(temp_tasks)
            dependency_penalty = 0 if is_valid else 500 * len(violations)

            # Invalid time penalty
            completion_penalty = 1000 if new_time <= 0 else 0

            # Total reward
            reward = energy_reduction - time_penalty - dependency_penalty - completion_penalty

            if self.verbose and (energy_reduction > 0):
                print(f"  Task {task_id} migration: Energy change: {energy_reduction:.2f}, " +
                      f"Time: {new_time:.2f}/{T_max:.2f}, Valid: {is_valid}, Reward: {reward:.2f}")

            return temp_tasks, temp_seq_manager, new_time, new_energy, reward

        except Exception as e:
            if self.verbose:
                print(f"Error evaluating migration for task {task_id}: {e}")
            return tasks, sequence_manager, float('inf'), float('inf'), -float('inf')

    def learn(self, tasks, sequence_manager, T_max, power_models,
              upload_rates, download_rates):
        """
        Enhanced learning algorithm with better error handling.
        """
        print(f"Starting Q-learning training with {self.max_episodes} episodes...")
        start_time = time.time()

        # Initial metrics
        initial_time = total_time_3tier(tasks)
        initial_energy = total_energy_3tier_with_rf(
            tasks,
            device_power_profiles=power_models['device'],
            rf_power=power_models['rf'],
            upload_rates=upload_rates
        )

        print(f"Initial time: {initial_time:.2f}, Initial energy: {initial_energy:.2f}")
        print(f"Time constraint: {T_max:.2f}")

        best_energy = initial_energy
        best_tasks = deepcopy(tasks)
        best_seq_manager = deepcopy(sequence_manager)

        # Track learning progress
        episode_rewards = []
        episode_migrations = []
        successful_migrations = 0

        # Training loop
        for episode in range(self.max_episodes):
            episode_start = time.time()

            # Start fresh in each episode
            current_tasks = deepcopy(tasks)
            current_seq_manager = deepcopy(sequence_manager)

            # Episode metrics
            total_reward = 0
            iterations = 0
            migrations = 0
            current_time = initial_time
            current_energy = initial_energy

            # Run a single episode
            while iterations < self.max_iterations:
                iterations += 1

                try:
                    # Get current state
                    state = self.get_state_hash(current_tasks)

                    # Get possible actions
                    actions = self.get_possible_actions(current_tasks, current_seq_manager)
                    if not actions:
                        if self.verbose:
                            print(f"No valid actions at iteration {iterations}")
                        break

                    # Select action
                    action = self.select_action(state, actions)
                    if action is None:
                        break

                    # Execute action
                    task_id, target_unit_idx = action
                    next_tasks, next_seq_manager, new_time, new_energy, reward = self.evaluate_action(
                        current_tasks, current_seq_manager, task_id, target_unit_idx,
                        power_models, upload_rates, download_rates, T_max
                    )

                    # Skip invalid migrations
                    if new_time == float('inf') or new_energy == float('inf'):
                        continue

                    # Get next state
                    next_state = self.get_state_hash(next_tasks)

                    # Get max Q-value for next state
                    next_actions = self.get_possible_actions(next_tasks, next_seq_manager)
                    max_next_q = 0.0
                    for next_a in next_actions:
                        next_q = self.get_q_value(next_state, next_a)
                        max_next_q = max(max_next_q, next_q)

                    # Update Q-value
                    old_q = self.get_q_value(state, action)
                    new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * max_next_q)
                    self.update_q_value(state, action, new_q)

                    # Only apply migration if it improves energy and meets constraints
                    if new_energy < current_energy and new_time <= T_max:
                        is_valid, _ = validate_task_dependencies(next_tasks)
                        if is_valid:
                            current_tasks = next_tasks
                            current_seq_manager = next_seq_manager
                            current_time = new_time
                            current_energy = new_energy
                            total_reward += reward
                            migrations += 1

                            if self.verbose:
                                print(f"  Migration applied: Energy={current_energy:.2f}, Time={current_time:.2f}")

                    # Update best solution
                    if current_energy < best_energy and current_time <= T_max:
                        is_valid, _ = validate_task_dependencies(current_tasks)
                        if is_valid:
                            best_energy = current_energy
                            best_tasks = deepcopy(current_tasks)
                            best_seq_manager = deepcopy(current_seq_manager)
                            if self.verbose:
                                print(f"  New best solution: Energy={best_energy:.2f}")

                except Exception as e:
                    print(f"Error in iteration {iterations}: {e}")
                    continue

            # End of episode
            episode_duration = time.time() - episode_start
            episode_rewards.append(total_reward)
            episode_migrations.append(migrations)
            successful_migrations += migrations

            # Log progress
            print(f"Episode {episode + 1}/{self.max_episodes}: " +
                  f"Time={current_time:.2f}, Energy={current_energy:.2f}, " +
                  f"Reward={total_reward:.2f}, Migrations={migrations}, " +
                  f"Duration={episode_duration:.2f}s")

            # Decay exploration rate
            self.epsilon = max(0.01, self.epsilon * 0.95)

            # Early stopping if we've made good progress
            if current_energy < initial_energy * 0.7:  # 30% reduction achieved
                if self.verbose:
                    print("Significant energy reduction achieved - ending training early")
                break

        # Save the final Q-table
        self.save_q_table()

        # Return best solution found
        training_time = time.time() - start_time
        print(f"Q-learning training completed in {training_time:.2f} seconds")
        print(f"Initial energy: {initial_energy:.2f}, Best energy: {best_energy:.2f}")
        print(f"Energy reduction: {(initial_energy - best_energy) / initial_energy * 100:.2f}%")
        print(f"Total successful migrations: {successful_migrations}")

        # Check final solution validity
        is_valid, violations = validate_task_dependencies(best_tasks)
        if not is_valid:
            print(f"WARNING: Final solution has {len(violations)} dependency violations!")
            # Fall back to original tasks if best solution is invalid
            return tasks, sequence_manager

        return best_tasks, best_seq_manager


# Main function to demonstrate usage
def optimize_with_q_learning(
        tasks,
        sequence_manager,
        T_max,
        power_models,
        upload_rates,
        download_rates,
        q_table_path='robust_q_table.pkl',
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.3,
        time_penalty=50.0,
        max_episodes=50,
        max_iterations=500,
        reset_q_table=False,
        verbose=True):
    """
    Optimize task scheduling using robust Q-learning implementation.

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
        reset_q_table: Whether to start with a fresh Q-table
        verbose: Enable verbose output

    Returns:
        Tuple of (optimized_tasks, optimized_sequence_manager)
    """
    # Initialize robust Q-learning scheduler
    q_scheduler = RobustQScheduler(
        alpha=learning_rate,
        gamma=discount_factor,
        epsilon=exploration_rate,
        time_penalty=time_penalty,
        max_episodes=max_episodes,
        max_iterations=max_iterations,
        q_table_path=q_table_path,
        reset_q_table=reset_q_table,
        verbose=verbose
    )

    print("Training Q-learning model...")
    optimized_tasks, optimized_seq_manager = q_scheduler.learn(
        tasks, sequence_manager, T_max, power_models, upload_rates, download_rates
    )

    return optimized_tasks, optimized_seq_manager


# Example usage in context
if __name__ == "__main__":

    # 1) Setup the environment
    upload_rates, download_rates = generate_realistic_network_conditions()
    mobile_power_models = generate_realistic_power_models(device_type='mobile', battery_level=65)

    # 2) Define example tasks (create a simple task graph)
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

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12, task13,
                        task14, task15, task16, task17, task18, task19, task20]
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
