import numpy as np

class Node():
    def __init__(self, point, min_ax, max_ax, axis, left=None, right=None):
        self.point = point
        self.min_ax = min_ax
        self.max_ax = max_ax
        self.axis = axis

        self.left = left
        self.right = right

def construct_kd_tree(points, axis=0):
    if len(points) == 0:
        return None


    # sort triangles with the first axis of the center
    vals = list(sorted(points, key=lambda x: x[axis]))
    median = len(points) // 2

    tree = [vals[median]]

    index = 0

    to_append_children = [0]

    while True:
        if len(to_append_children) == 0:
            break

        index = to_append_children.pop(0)

        if len(vals) <= index:
            break

        if len(vals) <= index*2+1:
            break

        if len(vals) <= index*2+2:
            break

        left = vals[index*2+1]
        right = vals[index*2+2]

        tree.append(left)
        tree.append(right)

        to_append_children.append(index*2+1)
        to_append_children.append(index*2+2)

    return tree






    # left = construct_kd_tree(vals[:median])
    # right = construct_kd_tree(vals[median+1:])

    # print(left,right)

    # return Node(vals[median],
    #             vals[0][axis],
    #             vals[-1][axis],
    #             axis,
    #             left=left,
    #             right=right)


points = np.random.rand(10,3)
node = construct_kd_tree(points)
