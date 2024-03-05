<h1> Foliated Manifold Framework Tutorial </h1>

## Recall the definitions
<h3> <b>Manifold</b> </h3>
A manifold is a set of configurations under certain constraints. In this project, we will only consider the manifold that is a set of arm configuration statisfying some constraints.

<h3> <b>Intersection</b> </h3>
An abstract edge between two manifolds. Usually, intersection is a single configuration. However, this is not always true. It could be a motion or action. For example, in pick and place task, the robot need to open gripper to transit from grasping manifold to ungrasping manifold. Therefore, the intersection between grasping manifold and ungrasping manifold is an action to open gripper and approach the object.

<h3> <b>Foliation</b> </h3>
A foliation is a set of manifolds defined by a common constraints with a list of co_parameters. Two manifolds are similar if they have similar co_parameters. Furthermore, two manifolds from the same foliation do not have intersection.

***
# Construct foliation and intersection
The first important thing is to define the foliation and intersection. In this project, we provide a template that user can over write them for their own use. The template is in <b>foliated_base_class.py</b> (<b>BaseFoliation</b> and <b>BaseIntersection</b>). We provide the example in <b>manipulation_foliations_and_intersections.py</b> to show how to define your customized foliation and intersection.

## Customized foliation
For the foliation, we have provide the constructor function which you should not overwirte, and the save and load function which you should over write.

Everytimes, you initialize your customized foliation, you should provide those four variables:
1. <b>foliation name</b>: the name of the foliation.
2. <b>constraint parameters</b>: the constraint parameters of the foliation. Basically, this is just a directory storing the shared parameters of the constraints in foliation.
3. <b>co-parameters</b>: the list of co-parameter of the foliation. This is a list of co-parameters used to define a manifold with constraint function.
4. <b>similarity matrix</b>: the similarity matrix of the foliation. This is a matrix(_numpy array_) that defines the similarity between each two co-parameters. If two co-parameters are similar, then the corresponding manifolds are similar.

### Here is a template:
```
custom_foliation = BaseFoliation(
    foliation_name='custom_foliation', # string
    constraint_parameters=constraint_parameters, # dictionary
    co_parameters=co_parameters, # list
    similarity_matrix=similarity_matrix # numpy array
)
```
### Here is an example
We use regrasping as an example. In regrasping foliation, each manifold is defined by a placement of the object.
```
foliation_regrasp = ManipulationFoliation(foliation_name='regrasp', 
                                       constraint_parameters={
                                           "frame_id": "base_link",
                                           "is_object_in_hand": False,
                                           ... # other parameters of the constraint
                                       }, 
                                       co_parameters=feasible_placements,
                                       similarity_matrix=np.identity(feasible_placements.__len__()))
```

Then, the save and load function must be over written as follows:
```
from foliated_base_class import BaseFoliation
# define the foliation class
class ManipulationFoliation(BaseFoliation):
    def save(self, dir_path):
        # save current foliation to file
    @staticmethod
    def load(self, dir_path):
        # load the foliation from file, and return the foliation
```

## Customized intersection
In manipulation, the intersection between manifolds may not be only a configuration. It can be anything such as a action or motion. For example, in pick and place task, the robot need to open gripper to transit from grasping manifold to ungrasping manifold. Therefore, we need to define the intersection class as follows:
```
from foliated_base_class import BaseIntersection
class ManipulationIntersection(BaseIntersection):
    def __init__(self, ...):
        # define the action of intersection
    def inverse(self):
        # define the inverse of intersection action
    def get_edge_configurations(self):
        # return the edge configurations of the intersection
    def save(self, file_path):
        # save the intersection to file
    def get_task_motion(self, ...):
        # return the task motion of the intersection. This task motion is used to visualize later. It must be type of BaseTaskMotion class.
    @staticmethod
    def load(file_path):
        # load the intersection from file
```
***
# How to construct the foliated problem
To use the foliation planning framework, you need to construct the foliated problem, so you can pass it to the task planner to generate the plan and use the motion planner to solve it. The optimal way is to construct the foliated problem and save it into file. Then, you can load it later for planning. 

## Steps of constructing the foliated problem

1.  create foliations
2.  create foliated intersections
3.  use them to construct the foliated problem

In this project, we provide <b>create_foliated_problem.py</b> as an example to show how to construct the foliated problem. In this example, we will construct a foliated problem containing two foliations and one foliated intersection. The first foliation is the regrasping foliation, and the second foliation is the sliding foliation.

First of all, you need to import the following packages:
```
from foliated_base_class import FoliatedProblem, FoliatedIntersection
from manipulation_foliations_and_intersections import ManipulationFoliation, ManipulationIntersection
```
1. **FoliatedProblem** is a class representing a foliated problem, but it requires user to define the foliation and intersection.
2. **FoliatedIntersection** is a class representing the intersection between two foliations. It requires user to define the sampling function to sample the intersection. The important thing is that this class is not a intersection class, but a class we use to sample the intersection, and user need to defined it by passing sampling functions.
3. Both **ManipulationFoliation** and **ManipulationIntersection** are the customized foliation and intersection defined by user.

## Create Foliations
```
foliation_regrasp = ManipulationFoliation(foliation_name='regrasp', 
                                                constraint_parameters={
                                                    ...
                                                }, 
                                                co_parameters=feasible_placements,
                                                similarity_matrix=...)
foliation_slide = ManipulationFoliation(foliation_name='slide', 
                                            constraint_parameters={
                                                ...
                                            }, 
                                            co_parameters=feasible_grasps,
                                            similarity_matrix=...)
```

## Create Foliated Intersections
To create the foliated Intersection, you need to provide three functions:
1. **prepare_sampling_function**: This function is used to prepare the sampler. 
2. **sampling_function**: This function is used to sample the intersection. As input, this function will receives two list of co_parameter list from two foliations, then return four variables: **sample_success_flag**, **selected_co_parameter_from_foliation_1**, **selected_co_parameter_from_foliation_2**, and a child class of BaseIntersection(the class defined by you previously, such as ManipulationIntersection).
3. **sampling_done_function**: This function is used to delete the sampler.

In our example, we have those function named slide_regrasp_sampling_function, prepare_sampling_function, sampling_done_function. Then, we can create the foliated intersection as follows:
```
foliated_intersection = FoliatedIntersection(foliation_slide, foliation_regrasp, slide_regrasp_sampling_function, prepare_sampling_function, sampling_done_function)
```
One thing must be noticed is that the order of foliations to this function must be the same as the order of co_parameters in sampling_function. That is, the first co_parameter list in sampling_function is from the first foliation, and the second co_parameter list in sampling_function is from the second foliation.

## Create Foliated Problem
After initializing a foliated problem object, you need to provide both foliations and foliated intersections to it. Then, you can sample the intersections by calling the following function:
```
foliated_problem = FoliatedProblem("foliated problem name")
foliated_problem.set_foliation_n_foliated_intersection([foliation_regrasp, foliation_slide],[foliated_intersection])
foliated_problem.sample_intersections(3000)
```
The **sample_intersections** function will sample the intersection by calling the sampling_function in foliated_intersection. The input of this function is the number of sampling attempts. The more sampling attempts, the more intersections will be sampled.

## Set Start and Goal Manifold Candidates
For the evaluation later, you may need to sample random start and goal manifold candidates, so you can time the planning time and success rate. Therefore, for a foliated problem, you need to set the start and goal manifold candidates as follows:
```
foliated_problem.set_start_manifold_candidates(start_candidates)
foliated_problem.set_goal_manifold_candidates(goal_candidates)
```
where both **start_candidates** and **goal_candidates** are the list of manifold candidates. Each manifold candidate is a tuple of foliation index and co_parameter index. For example, (0, 0) means the first manifold of the first foliation.

## Save and Load Foliated Problem
To save the foliated problem, you can call the following function:
```
foliated_problem.save(dir_path)
```

To load the foliated problem, you can call the following function:
```
foliated_problem = FoliatedProblem.load(ManipulationFoliation, ManipulationIntersection, dir_path)
```
<b>Warning</b>:
After loading the foliated problem, you can't call sampling again.

***
# How to use the foliated planning framework
To use foliated planning framework, you need to provide three things:
1. **task planner**: a class that can generate the plan based on the foliated problem.
2. **motion planner**: a class that can solve the plan generated by task planner.
3. **visualizer**: a class that can visualize the plan generated by task planner and solved by motion planner.

## Import packages
```
from foliated_base_class import FoliatedProblem, FoliatedIntersection
from manipulation_foliations_and_intersections import ManipulationFoliation, ManipulationIntersection
from foliated_planning_framework import FoliatedPlanningFramework
from jiaming_task_planner import MTGTaskPlanner, ...
from jiaming_motion_planner import MoveitMotionPlanner
from jiaming_visualizer import MoveitVisualizer
```

## Create task planner, motion planner, and visualizer
```
task_planner = ...
motion_planner = ...
visualizer = ...
```

## Create foliated planning framework
```
foliated_planning_framework = FoliatedPlanningFramework()
foliated_planning_framework.setTaskPlanner(task_planner)
foliated_planning_framework.setMotionPlanner(motion_planner)

# set the visualizer, this is optional
foliated_planning_framework.setVisualizer(visualizer)
```

## Set foliated problem and both start and goal
After creating the foliated planning framework, you need to set the foliated problem and both start and goal. The **setStartAndGoal** function requires the following input:
1. **start_foliation_index**: the index of start foliation in foliated problem.
2. **start_co_parameter_index**: the index of start co_parameter in start foliation.
3. **start_intersection**: the intersection of start manifold.
4. **goal_foliation_index**: the index of goal foliation in foliated problem.
5. **goal_co_parameter_index**: the index of goal co_parameter in goal foliation.
6. **goal_intersection**: the intersection of goal manifold.
```
# set the foliated problem
foliated_planning_framework.setFoliatedProblem(loaded_foliated_problem)

# set the start and goal
foliated_planning_framework.setStartAndGoal(
  0, 0,
  ManipulationIntersection(action='start', ...),
  0, 14,
  ManipulationIntersection(action='goal', ...)
)
```

During the evaluation, you can sample random start and goal manifolds by calling the following function( This function will set both start and goal manifold candidates, and is used in the evaluation):
```
foliated_planning_framework.sampleStartAndGoal()
```

## Plan and visualize
After setting the foliated problem and both start and goal, you can call the following function to plan and visualize the plan.
```
found_solution, solution_trajectory = foliated_planning_framework.solve()

# visualize the solution
foliated_planning_framework.visualizeSolutionTrajectory(solution_trajectory)
```

Then, you need to call the following function to shutdown the framework.
```
foliated_planning_framework.shutdown()
```