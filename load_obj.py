
import numpy as np

class KDTreeNumba():
    def __init__(self, tree, min_ax, max_ax, triangles):
        self.tree = tree
        self.min_ax = min_ax
        self.max_ax = max_ax
        self.triangles = triangles

    def get_node_children(self, node_index):
        node = self.tree[node_index]
        left_index = 2*node_index+1
        right_index = 2*node_index+2
        return left_index, self.tree[left_index], right_index, self.tree[right_index]


    def get_node_parent(self, node_index):
        return int((node_index-1)/2)


def number_of_nodes_in_tree(kd_tree):
    if kd_tree is None:
        return num_nodes

    num_nodes = 1

    if kd_tree.left is not None:
        num_nodes += number_of_nodes_in_tree(kd_tree.left)
    if kd_tree.right is not None:
        num_nodes += number_of_nodes_in_tree(kd_tree.right)

    return num_nodes

def convert_kd_tree_to_numba(kd_tree):
    n_nodes = number_of_nodes_in_tree(kd_tree)*2
    tree = np.zeros(shape=(n_nodes,3))
    min_ax = np.zeros(n_nodes)
    max_ax = np.zeros(n_nodes)
    triangles = np.zeros(shape=(n_nodes,3,3))

    def convert_node_to_array(node, index):
        tree[index] = node.center
        min_ax[index] = node.min_ax
        max_ax[index] = node.max_ax
        triangles[index] = node.triangle

        if node.left is not None:
            convert_node_to_array(node.left, 2*index+1)
        if node.right is not None:
            convert_node_to_array(node.right, 2*index+2)

    convert_node_to_array(kd_tree, 0)

    return tree, min_ax, max_ax, triangles

class Node():
    def __init__(self, triangle, center, min_ax, max_ax, axis, left=None, right=None):
        self.triangle = triangle
        self.center = center
        self.min_ax = min_ax
        self.max_ax = max_ax
        self.axis = axis

        self.left = left
        self.right = right

def construct_kd_tree(triangles, axis=0):
    if len(triangles) == 0:
        return None

    # sort triangles with the first axis of the center
    vals = list(sorted(triangles, key=lambda x: get_triangle_center(x)[axis]))

    median = len(triangles) // 2

    # print(len(vals))

    left = construct_kd_tree(vals[:median])
    right = construct_kd_tree(vals[median+1:])

    return Node(vals[median],
                get_triangle_center(vals[median]),
                get_triangle_center(vals[0])[axis],
                get_triangle_center(vals[-1])[axis],
                axis,
                left=left,
                right=right)

def intersect_kd_tree_node(ray, node):
    if node is None:
        return None

    if node.left is None and node.right is None:
        return node.triangle

    if ray.direction[0] == 0:
        return None

    t = (node.center[0] - ray.origin[0]) / ray.direction[0]

    if node.min_ax < t < node.max_ax:

        if node.center[0] < t:
            return intersect_kd_tree_node(ray, node.left)
        else:
            return intersect_kd_tree_node(ray, node.right)

    else:
        return None


def intersesct_kd_tree(ray, kd_tree, index=0):
    axis = 0
    left_index, left, right_index, right = kd_tree.get_node_children(index)

    # if left_index >= len(kd_tree.tree):
    #     return None
    # elif right_index >= len(kd_tree.tree):
    #     return None

    if ray.direction[axis] == 0:
        return None

    t = (kd_tree.tree[index][axis] - ray.origin[axis]) / ray.direction[axis]

    if kd_tree.min_ax[index] < t < kd_tree.max_ax[index]:

        if kd_tree.tree[index][0] < t:
            return intersesct_kd_tree(ray, kd_tree, left_index)
        else:
            return intersesct_kd_tree(ray, kd_tree, right_index)



def get_triangle_center(triangle):
    return (triangle[0] + triangle[1] + triangle[2]) / 3

def get_triangles(vertices, faces):
    triangles = []
    for face in faces:
        triangles.append([vertices[face[0]-1], vertices[face[1]-1], vertices[face[2]-1]])
    return triangles

def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            if line.startswith("f "):
                line_data = line.split(" ")
                faces.append([int(x) for x in line_data[1:]])

            if line.startswith("v "):
                line_data = line.split(" ")
                vertices.append([float(x) for x in line_data[1:]])

    triangles = get_triangles(vertices, faces)

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64), np.array(triangles, dtype=np.float64)


if __name__ == "__main__":
    vertices, faces, triangles = load_obj("objects/teapot.obj")

    from ray import Ray
    from primitives import Mesh

    r = Ray(np.array([0,0,0], np.float64), np.array([1,0,0], np.float64))

    kd_tree = construct_kd_tree(triangles)
    print(intersect_kd_tree_node(r, kd_tree))




    tree, min_ax, max_ax, triangles = convert_kd_tree_to_numba(kd_tree)
    kd_tree_numba = KDTreeNumba(tree, min_ax, max_ax, triangles)
    print(kd_tree_numba)




    print(intersesct_kd_tree(r, kd_tree_numba))


    # print(intersect_kd_tree(r, kd_tree))

    # mesh = Mesh(vertices, faces, triangles)
    # print(mesh.intersect(r))
