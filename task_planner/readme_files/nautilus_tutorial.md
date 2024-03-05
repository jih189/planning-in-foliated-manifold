## Nautilus
To get access to PRP Nautilus cluster. You need to install [kubectl tool](https://docs.nationalresearchplatform.org/userdocs/start/quickstart/). Then, for this project, jiaming has created an account for using that, so you can ask him for the config for this project. Once you got it, create a .kube directory and place the config find into it.
```
mkdir ~/.kube
cd .kube
# place the 'config' file here.
```

Run the following code to check whether you can access
```
kubectl get pods | grep jiaming
```
You should see a pod named 'jiaming-http-....'. If you can't find it, then you can ask jiaming to launch the container.

To enter it, you can use the following code to enter the container. You should replace the ... with what you see in the above command.
```
kubectl exec -it jiaming-http-... -- /bin/bash
```

Once you enter the container, you should build the jiaming-manipulation workspace by the following code
```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/jih189/jiaming_manipulation.git
cd jiaming_manipulation
rm -r manipulation_test rail_segmentation
cd ../..
catkin build
source devel/setup.bash
```
There are some packages you should delete because we need simplify the docker container.