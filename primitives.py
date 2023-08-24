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
            return [np.inf]
        else:
            t1 = (-b - np.sqrt(discriminant))/(2*a)
            t2 = (-b + np.sqrt(discriminant))/(2*a)
            return [t1,t2]


@nb.njit
def closest_intersection(ray, objects):
    closest_object = None
    closest_t = np.inf
    closest_index = -1
    for i, obj in enumerate(literal_unroll(objects)):
        ts = obj.intersect(ray)
        for t in ts:
            if ray.tmin<=t<=ray.tmax:
                if t<closest_t:
                    closest_object = obj
                    closest_t = t
                    closest_index = i

    return closest_object, closest_t, closest_index


if __name__ == "__main__":

    ss = (Sphere(np.array([0,0,0], np.float64), 1),
          Sphere(np.array([0,0,1], np.float64), 1))

    r = Ray(np.array([0,0,-2], np.float64), np.array([0,0,1], np.float64))

    closest_object, closest_t, closest_index = closest_intersection(r, ss)

    P = r.P(closest_t)
    closest_object.normal(P)