import numpy as np
import networkx as nx
from foliated_base_class import (
    BaseTaskPlanner,
    BaseIntersection,
    Task,
)

class MTGTaskPlanner(BaseTaskPlanner):
    def __init__(self, planner_name_="MTGTaskPlanner", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()  # python 2
        # super().__init__() # python 3
        self.planner_name = planner_name_
        self.parameter_dict = parameter_dict_

    # MTGTaskPlanner
    def reset_task_planner(self):
        self.task_graph = nx.DiGraph()
        self.manifold_info = {}  # the detail of each manifold
        self.incomming_manifold_intersections = (
            {}
        )  # the incomming intersections of each manifold
        self.outgoing_manifold_intersections = (
            {}
        )  # the outgoing intersections of each manifold
        self.new_intersection_id = 0

        # self.reset_manifold_similarity_table()
        self.total_similiarity_table = {}

    # MTGTaskPlanner
    def add_manifold(self, manifold_info_, manifold_id_):
        self.manifold_info[
            manifold_id_
        ] = manifold_info_  # the manifold_info is a dict with keys: foliation, co_parameter.

        self.incomming_manifold_intersections[manifold_id_] = []
        self.outgoing_manifold_intersections[manifold_id_] = []

    # MTGTaskPlanner
    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        intersection_from_1_to_2_id = self.new_intersection_id
        self.new_intersection_id += 1

        # add node for intersection from manifold 1 to manifold 2
        self.task_graph.add_node(
            intersection_from_1_to_2_id,
            intersection=intersection_detail_,
            previous_manifold_id=manifold_id1_,
            next_manifold_id=manifold_id2_,
        )

        for i in self.incomming_manifold_intersections[manifold_id1_]:
            self.task_graph.add_edge(
                i, intersection_from_1_to_2_id, weight=0.0, manifold_id=manifold_id1_
            )
        for i in self.outgoing_manifold_intersections[manifold_id2_]:
            self.task_graph.add_edge(
                intersection_from_1_to_2_id, i, weight=0.0, manifold_id=manifold_id2_
            )

        self.outgoing_manifold_intersections[manifold_id1_].append(
            intersection_from_1_to_2_id
        )
        self.incomming_manifold_intersections[manifold_id2_].append(
            intersection_from_1_to_2_id
        )

    # MTGTaskPlanner
    def set_start_and_goal(
        self,
        start_manifold_id_,
        start_intersection_,
        goal_manifold_id_,
        goal_intersection_,
    ):
        # if start and goal are set, then remove them from the task graph
        if self.task_graph.has_node("start"):
            self.task_graph.remove_node("start")

        if self.task_graph.has_node("goal"):
            self.task_graph.remove_node("goal")

        configuration_of_start, _ = start_intersection_.get_edge_configurations()
        self.task_graph.add_node(
            "start",
            intersection=IntersectionDetail(
                start_intersection_,
                configuration_of_start,
                configuration_of_start,
                False,
            ),
            previous_manifold_id="start",
            next_manifold_id=start_manifold_id_,
        )

        configuration_of_goal, _ = goal_intersection_.get_edge_configurations()
        self.task_graph.add_node(
            "goal",
            intersection=IntersectionDetail(
                goal_intersection_, configuration_of_goal, configuration_of_goal, True
            ),
            previous_manifold_id=goal_manifold_id_,
            next_manifold_id="goal",
        )

        for i in self.outgoing_manifold_intersections[start_manifold_id_]:
            self.task_graph.add_edge(
                "start", i, weight=0.0, manifold_id=start_manifold_id_
            )

        for i in self.incomming_manifold_intersections[goal_manifold_id_]:
            self.task_graph.add_edge(i, "goal", weight=0.0, manifold_id=goal_manifold_id_)

    # MTGTaskPlanner
    def generate_task_sequence(self):
        # check the connectivity of the task graph from start to goal
        if not nx.has_path(self.task_graph, "start", "goal"):
            print("no connection between start and goal!")
            return []

        # find the shortest path from start to goal
        shortest_path = nx.shortest_path(
            self.task_graph, "start", "goal", weight="weight"
        )
        task_sequence = []

        # construct the task sequence.
        for node1, node2 in zip(shortest_path[:-1], shortest_path[1:]):
            task = Task(
                manifold_detail_=self.manifold_info[
                    self.task_graph.edges[node1, node2]["manifold_id"]
                ],
                start_configuration_=nx.get_node_attributes(
                    self.task_graph, "intersection"
                )[node1].configuration_in_manifold2,
                goal_configuration_=nx.get_node_attributes(
                    self.task_graph, "intersection"
                )[node2].configuration_in_manifold1,
                next_motion_=nx.get_node_attributes(self.task_graph, "intersection")[
                    node2
                ].intersection_data,
                use_atlas=False,
            )

            task.set_task_graph_info((node1, node2))

            task_sequence.append(task)

        return task_sequence

    # MTGTaskPlanner
    def update(self, task_graph_info_, plan_, manifold_constraint_):
        # if current task is faled to solve, then we can increate the weight of the edge which is similar to the current task.
        # the similarity is defined as the product of the similarity of the previous manifold, the next manifold, and the current similarity.
        if not plan_[0]:
            # get the current manifold id, previous manifold id and next manifold id of the task.
            current_manifold_id = self.task_graph.edges[task_graph_info_]["manifold_id"]
            previous_manifold_id = self.task_graph.nodes[task_graph_info_[0]][
                "previous_manifold_id"
            ]
            next_manifold_id = self.task_graph.nodes[task_graph_info_[1]][
                "next_manifold_id"
            ]

            for (
                e_start_node,
                e_goal_node,
                e_current_manifold_id,
            ) in self.task_graph.edges.data("manifold_id"):
                # find all the edges having the same foliation with the current task.
                if current_manifold_id[0] == e_current_manifold_id[0]:
                    e_previous_manifold_id = self.task_graph.nodes[e_start_node][
                        "previous_manifold_id"
                    ]
                    e_next_manifold_id = self.task_graph.nodes[e_goal_node][
                        "next_manifold_id"
                    ]

                    previous_similarity_score = 0
                    next_similarity_score = 0

                    if (
                        previous_manifold_id == "start"
                        or e_previous_manifold_id == "start"
                    ):
                        if previous_manifold_id == e_previous_manifold_id:
                            previous_similarity_score = 1.0
                        else:
                            # previous similarity score is 0, so we can skip this edge.
                            continue
                    else:
                        previous_similarity_score = self.total_similiarity_table[
                            previous_manifold_id[0]
                        ][e_previous_manifold_id[1], previous_manifold_id[1]]

                    if next_manifold_id == "goal" or e_next_manifold_id == "goal":
                        if next_manifold_id == e_next_manifold_id:
                            next_similarity_score = 1.0
                        else:
                            # next similarity score is 0, so we can skip this edge.
                            continue
                    else:
                        next_similarity_score = self.total_similiarity_table[
                            next_manifold_id[0]
                        ][e_next_manifold_id[1], next_manifold_id[1]]

                    current_similarity_score = self.total_similiarity_table[
                        current_manifold_id[0]
                    ][e_current_manifold_id[1], current_manifold_id[1]]

                    total_similarity_score = (
                        current_similarity_score
                        * previous_similarity_score
                        * next_similarity_score
                    )

                    self.task_graph.edges[(e_start_node, e_goal_node)]["weight"] += (
                        1.0 * total_similarity_score
                    )
