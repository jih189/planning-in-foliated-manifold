
# Problem Generation Tutorial

This tutorial will guide you through the process of creating a problem with foliated structure.

Similar to mode transition graph, we consider any action which can be represented by foliation structure as a mode. The mode transition graph is a directed graph where the nodes represent the modes and the edges represent the actions that can be taken to transition from one mode to another. Given a start configuration with a start mode and a goal configuration with a goal mode, the problem is to find a path from the start to the goal that is feasible with respect to the mode transition graph.

Ideally, after FoliatedRepMap adding both start and goal, the planning system will first sample certain points from intersections, so there is a possible path between the start and goal. Then, the system will produce a task sequence that motion planner can solve. During the motion planning in one manifold of a foliation, once a point is sampled, the motion planning will try to extend it to one intersection to the next manifold. Thus, the total algorithm should be like
```
for loop:
  current_task = task_planner.generate_task()
  constraints = current_task.get_constraints()
  start_configuration = current_task.get_start()
  goal_constraints = current_task.get_goal_constraints()

  (
    solution_path,
    configurations_with_status,
    intersection_configurations
  )
  = 
  motion_planner.solve(
    start_configuration, 
    constraints, 
    goal_constraints
  )

  task_planner.update(
    solution_path,
    configurations_with_status,
    intersection_configurations
  )

  # check if the problem is solved or not.
  if task_planner.is_finished():
    return task_planner.get_solution()
```

Here is the problem configuration file.
```
foliation:
  - name: "foliation_1"
  - co_parameter_type: grasp or placement
  - co_parameter_set: file_path to the co-parameter set

Intersection:
  - foliation1: "foliation_1"
  - foliation2: "foliation_2"
  - object_pose_constraint(This is only used when co-parameter type of two foliations are grasps): ...

```