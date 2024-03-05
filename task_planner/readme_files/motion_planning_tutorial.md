# Motion planning tutorial
In this project, we will use CDistributionRRT which will read both constraints and distribution to plan the motion trajectory. We also provide an example code CDistributionRRT_example.ipynb in jupyter_note_tests.

For more details, 

1. To use CDistributionRRT, you must set path constraint, or it will cause error.
2. You can save SamplingDistribution into a list and pass it to motion planner with function set_distribution
3. To set sample ratio p to define the probability to sample uniformly(1-p) or with distribution(with p), you can modify the value of sample_ratio under "CDISTRIBUTIONRRTConfigDefault" in fetch_ros/fetch_moveit_config/config/ompl_planning.yaml.