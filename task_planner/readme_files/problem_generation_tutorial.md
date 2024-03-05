
# Problem Builder Tutorial

The problem builder aims to create motion planning problems related to manipulation tasks.

## Problem Modeling

The Problem Builder categorizes problems into three types based on their complexity and constraints type `simple`, `sequential`, and `crossing`. Moreover, the modeling approach can be divided into three placement strategies `rectangular placement`, `linear placement`, and `circular placement`. Each problem type requires specific `placement_parameters` for proper modeling.

### Rectangular Placement
Applicable for arranging multiple objects on a plane, such as moving cups on a surface or items on a shelf.

### Linear Placement
Suitable for manipulating objects constrained to linear motion, such as opening drawers or sliding doors.

### Circular Placement
Ideal for objects that rotate around a center, like turning doorknobs or twisting bottle caps.

## Usage

To generate a problem, follow these steps:

1. Define the problem type and upload the environment and object meshes.
2. Configure the `<problem>.yaml` file according to your problem specifications.
3. Change the `config.yaml` file in the `problem` directory with your problem config file path.
4. Run the problem generator script using the following command:

```bash
rosrun task_planner problem_generator.py
```

The script will perform the following actions:

- Load the problem configuration from the config file.
- Initialize the robot scene with the specified initial joint states.
- Build the foliations for the task based on the environment and grasp parameters.
- Use the sampler to generate motion planning problems by sampling intersections of foliations.
- Visualize the problem in the ROS environment using markers.
- Save the generated problem to `check` directory 


## Configuration File Structure

The problem's configuration file should be defined in YAML format. Below is a detailed explanation of the key sections:

### Environment Section
Defines the paths to the environment and object meshes, grasp poses file, and specifies the poses of the environment and table top.

```yaml
environment:
  env_mesh_path: <path_to_environment_mesh>
  manipulated_object_mesh_path: <path_to_object_mesh>
  grasp_poses_file: <path_to_grasp_poses_file>
  env_pose:
    frame_id: "base_link"
    position:
      x: <x_position>
      y: <y_position>
      z: <z_position>
    orientation:
      x: <x_orientation>
      y: <y_orientation>
      z: <z_orientation>
      w: <w_orientation>
  table_top_pose:
    - [1, 0, 0, <x_table_top>]
    - [0, 1, 0, <y_table_top>]
    - [0, 0, 1, <z_table_top>]
    - [0, 0, 0, 1]
```

### Grasp Parameters Section
Specifies the number of grasp samples and the rotation matrix applied to the grasps.

```yaml
grasp_parameters:
  num_samples: <number_of_grasp_samples>
  rotated_matrix:
    - [1, 0, 0, <x_rotation>]
    - [0, 1, 0, <y_rotation>]
    - [0, 0, 1, <z_rotation>]
    - [0, 0, 0, 1]
```

### Foliation Parameters Section
Defines the parameters for the foliation process, including sliding sigma and tolerances for orientation and position.

```yaml
foliation_parameters:
  sliding_sigma: <sigma_value_for_sliding>
  orientation_tolerance: [<x_tolerance>, <y_tolerance>, <z_tolerance>]
  position_tolerance: [<x_position_tolerance>, <y_position_tolerance>, <z_position_tolerance>]
```

### Initial Joint State Section
Lists the initial positions of the robot's joints.

```yaml
initial_joint_state:
  joint_name_1: <initial_position>
  joint_name_2: <initial_position>
  # ...
```

### Task Parameters Section
Contains the task name, number of samples for intersections, and the path to save the generated problem.

```yaml
task_parameters:
  task_name: <name_of_the_task>
  num_samples: <number_of_samples_for_intersection>
  save_path: <path_to_save_generated_problem>
```

### Placement Parameters Section
Describes the type of placement and the additional parameters based on the chosen placement type.

For `linear` type:
```yaml
placement_parameters:
  type: "linear"
  start_position: [<x>, <y>, <z>]
  end_position: [<x>, <y>, <z>]
  steps: <steps>
```

For `circular` type:

```yaml
placement_parameters:
  type: "circular"
  center_position: [<x_center>, <y_center>, <z_center>]
  radius: <radius_value>
  start_angle: <start_angle_rad>
  end_angle: <end_angle_rad>
  steps: <steps>
```

For `rectangular` type:

```yaml
placement_parameters:
  type: "rectangular"
  layers:
    - num_of_row: <number_of_rows>
      num_of_col: <number_of_columns>
      x_shift: <x_shift_value>
      y_shift: <y_shift_value>
      z_shift: <z_shift_value>
    # Additional layers can be added here, e.g shelf have multiple layers
```

## Preview Problem

The `ProblemVisualizer` class publishes markers representing the environment, object placements, and other relevant aspects of the problem to the `/problem_visualization_marker_array` topic. You can preview the problem with RVIZ.

## Saving and Loading Problems

Generated problems can be saved to a file specified in the configuration using the `save` method of the `FoliatedProblem` class. To load a saved problem, use the `load` method.

## Pre-defined Problems and Models

### Problems

The following table lists the pre-defined problems available in the `problems` folder:

### Rectangular Placement

| Problem File                 | Type                   | Description                                           |
|------------------------------|------------------------|-------------------------------------------------------|
| `maze.yaml`         | Crossing Rectangular   | A complex maze problem involving crossing paths.      |
| `maze_real.yaml`    | Crossing Rectangular   | A realistic maze scenario with more intricate paths.  |
| `shelf.yaml`      | Sequential Rectangular | A task involving sequential object placement on a shelf. |
| `simple_table.yaml`          | Simple Rectangular     | A straightforward task of object placement on a table.|

### Linear Placement

| Problem File                 | Type            | Description                                           |
|------------------------------|-----------------|-------------------------------------------------------|
| `simple_open_drawer.yaml`    | Crossing Linear | A task simulating the opening of a drawer.            |

### Circular Placement

| Problem File                 | Type              | Description                                           |
|------------------------------|-------------------|-------------------------------------------------------|
| `slide_cup.yaml`      | Crossing Circular | A task involving the circular sliding of a cup.       |
| `open_door.yaml`    | Crossing Circular | A task simulating the opening of a door with crossing paths.|

### Models

The following table lists the available models in the `mesh_dir` folder, each with associated grasp poses and a brief description:

| Model File                  | Grasp Poses File     | Description                                         |
|-----------------------------|----------------------|-----------------------------------------------------|
| `cup.stl`                   | `cup.npz`            | A cup model with grasp poses for manipulation tasks.|
| `lid.stl`                   | `lid.npz`            | A lid model with grasp poses for tasks like twisting open.|
| `desk_with_bottle_frame.stl` | -                    | A desk frame model with a mounted bottle for lid-related tasks.|
| `drawer.stl`                | `drawer.npz`  | A scaled drawer model with grasp poses for opening tasks.|
| `desk_with_drawer_frame.stl` | -                  | A desk frame with an integrated drawer for related tasks.|
| `desk_with_item.stl`        | -                    | A cluttered desk model simulating a maze-like scenario.|
| `desk_with_shelf.stl`       | -                    | A desk with an attached shelf for sequential placement tasks.|
| `desk_set.stl`              | -                    | A standard desk model for various placement tasks.    |
| `desk.stl`                  | -                    | Another variant of a desk model for placement tasks.  |
| `table.stl`                 | -                    | A simple table model for basic manipulation tasks.    |
| `maze.stl`                  | -                    | A maze model for complex navigation and manipulation tasks.|