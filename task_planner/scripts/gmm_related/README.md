# Generation of GMMs

A Gaussian mixture model is generated from all possible valid configurations of the robot. The distributions are generated in the configuration space. (i.e. dimension of mean = dof of arm). The goal is to partition the self-collision free configuration space into a mixture of distributions such that valid configurations can be sampled from these distributions. 

The file `gmm_generation.py` contains the script necessary to compute the distributions information.

First, trajectory data is generated from the `data_generation` package.

For this work, 10000 valid paths are generated from an obstacle free environment. Run 
```
rosrun data_generation empty_world_trajectory_generation.py 1 10000
```

To create the distributions, edit the paths in `task_planner/scripts/gmm_related/gmm_generation.py`.
Also tune the hyper-parameters

```
rosrun task_planner gmm_generation.py
```

These are the operations it performs
1. Read all trajectories from all environments from the given directory
2. Estimates the number of components of the gaussian mixture by picking the maximum number of waypoints used across all trajectories.
3. Compute the Gaussian Mixture Model (either using dirichlet process or without) and save it to files

For more details (and visualization), refer to `jupyter_note_tests/GMMTest.ipynb`
