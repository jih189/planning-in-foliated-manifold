# Task and motion planner tutorial
As a planning framework, user has freedom to develop their own task planner and motion planner which are required in foliated_planning_framework. In this tutorial, we will introduce how to develop a task planner and motion planner.
## Task planner
Task planner is used to generate a sequence of motion task. We provide a example named "jiaming_task_planner" to define the task planner. Here are the functions user need to define in the task planner:

<b>reset_task_planner</b>

This function will be called in the beginning of solve function. It is used to reset the task planner.

<b>add_manifold</b>

This function receives a manifold info and manifold id, which is usually foliation id and co-parameter id.

<b>add_intersection</b>

This function receives the start manifold id and goal manifold id where the intersection is connected. The intersection_detail_ contains the information of the intersection which is defined in IntersectionDetail.

<b>set_start_and_goal</b>

This function receives the start manifold id and goal manifold id. It is used to set the start and goal of the task planner.

<b>generate_task_sequence</b>

This function is used to generate a sequence of motion task. It returns a list of motion task whose class is defined in foliated_base_class.py. If there is no solution, it returns an empty list.

<b>update</b>

This function is used to update the task planner. It receives the task_graph_info which is used to guide the task planner which part of the task graph is used to solve. The plan should contain the information as experience to guide the task planner.

### optional function
<b>read_pointcloud</b>

This function is used to read the pointcloud of the environment to update the task planner. 

## Motion planner
The motion planner should define the following functions:
<b>prepare_planner</b>

This function is used to prepare the planner.

<b>_plan</b>

This function is used to plan a motion task. It receives a motion task and returns a motion plan. If there is no solution, it returns an empty list. All the motion plan should be class BaseTaskMotion. You need to define your own task motion which can be used to visualize the motion later.

<b>shutdown_planner</b>

This function is used to shutdown the planner.

---
# Visualizer
You may notice that there is a visualizer in the foliated_planning_framework. This visualizer is used to visualize the planning process. Here we procide a tutorial to define your own visualizer.

<b>visualize_plan</b>

This function receives a list of task motion, then you can visualize the motion in the list.
