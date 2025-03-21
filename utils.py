import matplotlib.pyplot as plt
from data import ExecutionTier, SchedulingState
import networkx as nx

def plot_three_tier_gantt(tasks, scheduler, title="Three-Tier Schedule"):
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


def create_and_visualize_task_graph(nodes, save_path=None, formats=None, dpi=300,show_allocation=False, final_allocation=None):
    """
    Creates and visualizes a task graph with color coding for task types and allocation.

    Parameters:
        nodes: List of Task objects
        save_path: Path to save the visualization
        formats: List of file formats to save (e.g., ['png', 'pdf'])
        dpi: Resolution for raster formats
        show_allocation: Whether to show the final allocation (device/edge/cloud)
        final_allocation: Dictionary mapping task IDs to their final allocation
    """
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes:
        # Store necessary attributes for visualization
        attributes = {
            'task_type': node.task_type if hasattr(node, 'task_type') else 'unknown'
        }
        G.add_node(node.id, **attributes)

    # Add edges
    for node in nodes:
        for child in node.succ_tasks:
            G.add_edge(node.id, child.id)

    plt.figure(figsize=(12, 14))

    # Use hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')

    # Define colors for task types
    task_type_colors = {
        'compute': '#FF9999',  # Red for compute-intensive
        'data': '#99CCFF',  # Blue for data-intensive
        'balanced': '#99FF99',  # Green for balanced
        'unknown': '#DDDDDD'  # Gray for unknown
    }

    # Define colors and shapes for allocation
    allocation_colors = {
        'device': '#FFA500',  # Orange for device
        'edge': '#800080',  # Purple for edge
        'cloud': '#00BFFF',  # Sky blue for cloud
        'unassigned': '#FFFFFF'  # White for unassigned
    }

    # Assign node colors based on task type or allocation
    if show_allocation and final_allocation:
        node_colors = [allocation_colors[G.nodes[n]['allocation']] for n in G.nodes]
        edge_color = '#666666'  # Darker edges for better contrast
    else:
        node_colors = [task_type_colors[G.nodes[n]['task_type']] for n in G.nodes]
        edge_color = '#333333'

    # Create node labels
    node_labels = {}
    for node_id in G.nodes:
        node_labels[node_id] = str(node_id)

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, width=1.5, edge_color=edge_color)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')

    # Add a legend for task types
    if not show_allocation or not final_allocation:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"{task_type}")
            for task_type, color in task_type_colors.items() if task_type != 'unknown'
        ]
        plt.legend(handles=legend_elements, loc='upper left', title="Task Types")
    else:
        # Add a legend for allocation
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"{alloc}")
            for alloc, color in allocation_colors.items() if alloc != 'unassigned'
        ]
        plt.legend(handles=legend_elements, loc='upper left', title="Allocation")

    plt.title("Task Dependency Graph", fontsize=18, pad=20)
    plt.axis('off')

    # Save visualization if path is provided
    if save_path and formats:
        plt.tight_layout()

        for fmt in formats:
            full_path = f"{save_path}.{fmt}"
            try:
                if fmt in ['pdf', 'svg', 'eps']:
                    plt.savefig(full_path, format=fmt, bbox_inches='tight', pad_inches=0.1)
                else:
                    plt.savefig(full_path, format=fmt, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                print(f"Successfully saved visualization as {full_path}")
            except Exception as e:
                print(f"Error saving {fmt} format: {str(e)}")

    return plt.gcf()