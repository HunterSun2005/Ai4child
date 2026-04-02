import numpy as np

# COCO-WholeBody indices:
# - body+foot: 0..22
# - face: 23..90
# - left hand: 91..111
# - right hand: 112..132
DEFAULT_NON_FACE_KEYPOINTS = list(range(23)) + list(range(91, 133))

BODY_AND_FOOT_LINKS = [
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

HAND_LINKS = [
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
    (112, 113), (113, 114), (114, 115), (115, 116),
    (112, 117), (117, 118), (118, 119), (119, 120),
    (112, 121), (121, 122), (122, 123), (123, 124),
    (112, 125), (125, 126), (126, 127), (127, 128),
    (112, 129), (129, 130), (130, 131), (131, 132),
]

BRIDGE_LINKS = [
    (9, 91),    # left wrist -> left hand root
    (10, 112),  # right wrist -> right hand root
]

PARENT_MAP = {
    # body + foot
    0: 0, 1: 0, 2: 0, 3: 1, 4: 2,
    5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 5, 12: 6, 13: 11, 14: 12, 15: 13, 16: 14,
    17: 15, 18: 15, 19: 15, 20: 16, 21: 16, 22: 16,
    # left hand
    91: 9, 92: 91, 93: 92, 94: 93, 95: 94,
    96: 91, 97: 96, 98: 97, 99: 98,
    100: 91, 101: 100, 102: 101, 103: 102,
    104: 91, 105: 104, 106: 105, 107: 106,
    108: 91, 109: 108, 110: 109, 111: 110,
    # right hand
    112: 10, 113: 112, 114: 113, 115: 114, 116: 115,
    117: 112, 118: 117, 119: 118, 120: 119,
    121: 112, 122: 121, 123: 122, 124: 123,
    125: 112, 126: 125, 127: 126, 128: 127,
    129: 112, 130: 129, 131: 130, 132: 131,
}

PART_GROUPS = [
    [5, 7, 9],                              # left_arm
    [6, 8, 10],                             # right_arm
    [11, 13, 15, 17, 18, 19],               # left_leg
    [12, 14, 16, 20, 21, 22],               # right_leg
    [0, 1, 2, 3, 4],                        # head
    list(range(91, 112)),                   # left_hand
    list(range(112, 133)),                  # right_hand
]


class AichildGraph:
    """Graph definition for configurable COCO-WholeBody keypoint subsets."""

    def __init__(self, keypoint_indices=None, max_hop=10, dilation=1):
        if keypoint_indices is None:
            keypoint_indices = DEFAULT_NON_FACE_KEYPOINTS
        self.keypoint_indices = [int(x) for x in keypoint_indices]
        if len(set(self.keypoint_indices)) != len(self.keypoint_indices):
            raise ValueError("Duplicate keypoint indices found in graph config.")

        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()
        self.A = self._get_adjacency()

    def _get_edge(self):
        local_of = {orig_idx: i for i, orig_idx in enumerate(self.keypoint_indices)}
        num_node = len(self.keypoint_indices)

        neighbor_link = []
        for a, b in BODY_AND_FOOT_LINKS + BRIDGE_LINKS + HAND_LINKS:
            if a in local_of and b in local_of:
                neighbor_link.append((local_of[a], local_of[b]))

        connect_joint = np.arange(num_node, dtype=np.int64)
        for orig_idx, local_idx in local_of.items():
            parent_orig = PARENT_MAP.get(orig_idx, orig_idx)
            parent_local = local_of.get(parent_orig, local_idx)
            connect_joint[local_idx] = parent_local

        parts = []
        covered = set()
        for group in PART_GROUPS:
            local_group = [local_of[idx] for idx in group if idx in local_of]
            if not local_group:
                continue
            parts.append(np.array(local_group, dtype=np.int64))
            covered.update(local_group)

        if len(covered) < num_node:
            remain = np.array(sorted(set(range(num_node)) - covered), dtype=np.int64)
            parts.append(remain)

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
