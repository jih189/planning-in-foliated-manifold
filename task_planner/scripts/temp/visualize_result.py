#!/usr/bin/env python
import json
import rospkg
import matplotlib.pyplot as plt


def find_median(nums):
    sorted_nums = sorted(nums)
    n = len(sorted_nums)

    # If the list has an odd number of values, return the middle one.
    if n % 2 == 1:
        return sorted_nums[n // 2]
    # If the list has an even number of values, return the average of the two middle ones.
    else:
        left_mid = sorted_nums[(n - 1) // 2]
        right_mid = sorted_nums[n // 2]
        return (left_mid + right_mid) / 2


if __name__ == "__main__":
    """
    load the json file and visualize the result
    """

    rospack = rospkg.RosPack()
    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    result_json_path = package_path + "/evaluated_data_dir/evaluated_data.json"
    # result_json_path = package_path + '/evaluated_data_dir/maze_evaluated_data.json'

    with open(result_json_path) as f:
        data = json.load(f)

    # find all task planner names
    task_planner_names = list(set([d["task_planner"] for d in data]))

    print("task planner names: ", task_planner_names)

    total_time = [0.0 for i in range(len(task_planner_names))]
    total_distance = [0.0 for i in range(len(task_planner_names))]
    experiment_count = [0.0 for i in range(len(task_planner_names))]
    success_count = [0.0 for i in range(len(task_planner_names))]
    times = [[] for i in range(len(task_planner_names))]
    distances = [[] for i in range(len(task_planner_names))]

    for i in range(len(data)):
        for j in range(len(task_planner_names)):
            if data[i]["task_planner"] == task_planner_names[j]:
                experiment_count[j] += 1
                if data[i]["found_solution"]:
                    success_count[j] += 1
                    total_time[j] += data[i]["time"]
                    total_distance[j] += data[i]["total_distance"]
                    times[j].append(data[i]["time"])
                    distances[j].append(data[i]["total_distance"])

    average_time = [0.0 for i in range(len(task_planner_names))]
    average_distance = [0.0 for i in range(len(task_planner_names))]
    success_rate = [0.0 for i in range(len(task_planner_names))]

    for i in range(len(task_planner_names)):
        if success_count[i] == 0:
            print("task planner: ", task_planner_names[i])
            print("always fail")
        else:
            average_time[i] = total_time[i] / success_count[i]
            average_distance[i] = total_distance[i] / success_count[i]
            success_rate[i] = success_count[i] / experiment_count[i]
            print("task planner: ", task_planner_names[i])
            print("average time: ", average_time[i])
            print("average distance: ", average_distance[i])
            print("success rate: ", success_rate[i])

    replaced_task_planner_names = [
        "MTG",
        "MDP",
        "MTG/FoliatedRepMap",
        "MDP/FoliatedRepMap",
    ]

    # visulize the average time, average distance, and success rate for each planner in plt
    plt.figure(0)
    plt.subplot(211)
    plt.boxplot(times, showmeans=True, meanline=True)
    plt.ylabel("Time")
    plt.xticks(range(1, len(task_planner_names) + 1), replaced_task_planner_names)
    # show the value in the table as well
    for i in range(len(task_planner_names)):
        plt.text(
            i + 1,
            average_time[i],
            str(round(average_time[i], 2)),
            ha="center",
            va="center",
        )

        # get the median value of times[i]
        median = find_median(times[i])
        plt.text(i + 1, median, str(round(median, 2)), ha="center", va="center")

    # # set max y value
    # plt.ylim(0, 141)

    # plt.subplot(212)
    # plt.boxplot(distances)
    # plt.ylabel('Distance')
    # plt.xticks(range(1, len(task_planner_names) + 1), replaced_task_planner_names)
    # # show the value in the table as well
    # for i in range(len(task_planner_names)):
    #     plt.text(i + 1, average_distance[i], str(round(average_distance[i], 2)), ha='center', va='center')

    plt.subplot(212)
    # plt.bar(task_planner_names, success_rate)
    plt.bar(range(len(task_planner_names)), success_rate)
    plt.ylabel("Success Rate")
    plt.xticks(range(len(task_planner_names)), replaced_task_planner_names)
    # show the value in percentage in the table as well
    for i in range(len(task_planner_names)):
        plt.text(
            i,
            success_rate[i],
            str(round(success_rate[i] * 100, 2)) + "%",
            ha="center",
            va="bottom",
        )
    plt.show()
