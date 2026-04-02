import numpy as np


class AichildGraph:
    """Graph definition for 23 body+foot keypoints (COCO WholeBody subset 0..22)."""

    def __init__(self, max_hop=10, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()
        self.A = self._get_adjacency()

    def _get_edge(self):
        num_node = 23
        # Skeleton links for body + feet (0-indexed).
        neighbor_link = [
            (15, 13), (13, 11),
            (16, 14), (14, 12),
            (11, 12),
            (5, 11), (6, 12),
            (5, 6),
            (5, 7), (6, 8),
            (7, 9), (8, 10),
            (1, 2),
            (0, 1), (0, 2),
            (1, 3), (2, 4),
            (3, 5), (4, 6),
            (15, 17), (15, 18), (15, 19),
            (16, 20), (16, 21), (16, 22),
        ]

        # Parent joint for each keypoint index, used to build bone vectors.
        connect_joint = np.array([
            0,   # nose -> nose
            0,   # left eye -> nose
            0,   # right eye -> nose
            1,   # left ear -> left eye
            2,   # right ear -> right eye
            3,   # left shoulder -> left ear
            4,   # right shoulder -> right ear
            5,   # left elbow -> left shoulder
            6,   # right elbow -> right shoulder
            7,   # left wrist -> left elbow
            8,   # right wrist -> right elbow
            5,   # left hip -> left shoulder
            6,   # right hip -> right shoulder
            11,  # left knee -> left hip
            12,  # right knee -> right hip
            13,  # left ankle -> left knee
            14,  # right ankle -> right knee
            15,  # left big toe -> left ankle
            15,  # left small toe -> left ankle
            15,  # left heel -> left ankle
            16,  # right big toe -> right ankle
            16,  # right small toe -> right ankle
            16,  # right heel -> right ankle
        ])

        parts = [
            np.array([5, 7, 9]),                     # left_arm
            np.array([6, 8, 10]),                    # right_arm
            np.array([11, 13, 15, 17, 18, 19]),      # left_leg
            np.array([12, 14, 16, 20, 21, 22]),      # right_leg
            np.array([0, 1, 2, 3, 4]),               # head
        ]

        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        adjacency = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            adjacency[i, j] = 1
            adjacency[j, i] = 1

        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer = [np.linalg.matrix_power(adjacency, d) for d in range(self.max_hop + 1)]
        arrive = (np.stack(transfer) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive[d]] = d
        return hop_dis

    def _normalize_digraph(self, adjacency):
        degree = np.sum(adjacency, 0)
        dn = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            if degree[i] > 0:
                dn[i, i] = degree[i] ** (-1)
        return np.dot(adjacency, dn)

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A
