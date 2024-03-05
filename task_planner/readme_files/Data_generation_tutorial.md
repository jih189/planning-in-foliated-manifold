# Dataset generation tutorial

In this project, we have a predictor to read the pointcloud and predict the feasibility of each predefined distribution of GMM. For this purpose, we need to generate the dataset. Here, we assume we have GMM already. (you need to have a fake move group launching for the following command)

First, generate a set of self-collision-free arm configurations with following code.
```
rosrun data_generation random_joint_state_generation.py
```
This code will generate a directory(jiaming_manipulation/data_generation/gmm_data) with joint_names.npy and valid_robot_states.npy. joint_names.py contains the name of all joint of the robot, while valid_robot_state.npy contains all valid robot state(only joint values).

Second, based on those self-collision-free arm configurations, we randomly generate a scene and valid tag with the following command:
```
rosrun data_generation pointcloud_joint_state_validity_generation.py _random_value:=<random_value> _total_env_num:=<number of scene to generate>
```
Here, the random value is used as seed, while the total_env_num is the number of scene you want to generate. After generation, for each scene i, there will be a dir named "env_i". In this directory, there are two files. map_i.ply is the pointcloud of the obstacle, while valid_tag.npy is a vector of valid flag for each arm configuration in the valid_robot_states.npy.

Third, load the gmm and predict the feasibility of each predefined distribution with the following command:
```
rosrun task_planner prepare_gmm_dataset.py
```

This command will first load the gmm, then find the gmm_data in the data_generation, the read the valid_robot_states.npy. Based on the gmm and valid_tag.npy in each env_i, it can calculate the total number of sampled joint state of each distribution as total_count_dic.npy. Then, it also produces the valid number of each distribution as valid_count_dic.npy. Both of them are dictionary.

# To download the dataset(internal use only)
We use s3cmd to save all data, so you need to download the s3cmd first.
```
apt-get install s3cmd
```
Then you can put the configuration file into ~/.s3cfg. Then, you can list all files in the s3 bucket with the following command:
```
s3cmd ls s3://my-bucket
```
It should print out all files in the bucket like this:
```
2023-08-07 06:37  66233424   s3://my-bucket/gmm-data-0000
2023-08-07 22:28  66188139   s3://my-bucket/gmm-data-1000
2023-08-07 06:39  66214753   s3://my-bucket/gmm-data-2000
2023-08-07 06:46  66256142   s3://my-bucket/gmm-data-3000
2023-08-07 06:27  66182459   s3://my-bucket/gmm-data-4000
2023-08-07 06:30  66210384   s3://my-bucket/gmm-data-5000
2023-08-07 06:32  66216336   s3://my-bucket/gmm-data-6000
2023-08-07 06:33  66204653   s3://my-bucket/gmm-data-7000
2023-08-05 17:38    990042   s3://my-bucket/my-folder
```
Then you can download each of them with the following command (for example, you want to download gmm-data-0000):
```
s3cmd get s3://my-bucket/gmm-data-0000
```
or you can use the following command to download all of them (In this project, you should download all of them):
```
s3cmd get s3://my-bucket --recursive
```
You can unzip and merge them together with the following command into gmm_data:
```
unzip my-folder && 
unzip gmm-data-0000 &&
unzip gmm-data-1000 &&
unzip gmm-data-2000 &&
unzip gmm-data-3000 &&
unzip gmm-data-4000 &&
unzip gmm-data-5000 &&
unzip gmm-data-6000 &&
unzip gmm-data-7000 
```