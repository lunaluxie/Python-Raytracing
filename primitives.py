import numba as nb
from numba import literal_unroll
import numpy as np

from ray import Ray


@nb.experimental.jitclass([
    ("position", nb.float64[:]),
    ("radius", nb.float64),
])
class Sphere():
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def normal(self, P):
        return (P - self.position) / self.radius

    def intersect(self, ray):
        a = np.dot(ray.direction, ray.direction)
        b = 2*np.dot(ray.origin-self.position, ray.direction)
        c = np.dot(ray.origin-self.position, ray.origin-self.position) - self.radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return [np.inf], [np.array([0,0,0],dtype=np.float64)]
        else:
            t1 = (-b - np.sqrt(discriminant))/(2*a)
            t2 = (-b + np.sqrt(discriminant))/(2*a)
            return [t1,t2], [self.normal(ray.P(t)) for t in [t1,t2]]

@nb.experimental.jitclass([
    ("vertices", nb.float64[:,:]),
    ("faces", nb.int64[:,:]),
    ("triangles", nb.float64[:,:,:]),
])
class Mesh():
    def __init__(self, vertices, faces, triangles):
        self.vertices = vertices
        self.faces = faces
        self.triangles = triangles

    @staticmethod
    def point_in_triangle(P, triangle):
        A,B,C = triangle

        # Move the triangle so that the point becomes the triangles origin
        A -= P
        B -= P
        C -= P

        u = np.cross(B, C)
        v = np.cross(C, A)
        w = np.cross(A, B)


        #Test to see if the normals are facing the same direction, return false if not
        if np.dot(u, v) < 0:
            return False
        if np.dot(u, w) < 0:
            return False

        # All normals facing the same way, return true
        return True


    def normal(self, P):
        for triangle in self.triangles:

            # check if point is on triangle
            if self.point_in_triangle(P, triangle):
                A,B,C = triangle
                edgeAB = B - A
                edgeAC = C - A
                normal = np.cross(edgeAB, edgeAC)
                return normal

        return None

    def intersect(self, ray):
        closest_t = np.inf
        closest_normal = np.array([0,0,0], dtype=np.float64)

        for triangle in self.triangles:
            A,B,C = triangle
            edgeAB = B - A
            edgeAC = C - A
            normalVector = np.cross(edgeAB, edgeAC)
            ao = ray.origin - A
            dao = np.cross(ao, ray.direction)

            determinant = -np.dot(ray.direction, normalVector)
            invDet = 1 / determinant

            t = np.dot(ao, normalVector) * invDet

            if t < ray.tmin or t > ray.tmax:
                continue

            if t < closest_t:
                closest_t = t
                closest_normal = normalVector

        return [closest_t], [closest_normal]


@nb.njit
def closest_intersection(ray, objects):
    closest_object = None
    closest_t = np.inf
    closest_index = -1
    closest_normal = np.array([0,0,0], dtype=np.float64)
    for i, obj in enumerate(literal_unroll(objects)):
        ts, normals = obj.intersect(ray)
        for t, normal in zip(ts, normals):
            if ray.tmin<=t<=ray.tmax:
                if t<closest_t:
                    closest_object = obj
                    closest_t = t
                    closest_index = i
                    closest_normal = normal

    return closest_object, closest_t, closest_index, closest_normal


if __name__ == "__main__":

    ss = (Sphere(np.array([0,0,0], np.float64), 1),
          Sphere(np.array([0,0,1], np.float64), 1))

    r = Ray(np.array([0,0,-2], np.float64), np.array([0,0,1], np.float64))

    closest_object, closest_t, closest_index, closest_normal = closest_intersection(r, ss)

    P = r.P(closest_t)
    print(closest_object.normal(P))
    print(closest_normal)




    from load_obj import load_obj
    vertices, faces, triangles = load_obj("objects/teapot.obj")

    triangles = np.array(triangles, dtype=np.float64)

    teapot = Mesh(vertices, faces, triangles)

    r = Ray(np.array([0,0,-2], np.float64), np.array([0,0,1], np.float64))

    ts, normals = teapot.intersect(r)

    print(normals)
