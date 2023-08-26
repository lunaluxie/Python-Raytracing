import numba as nb
import numpy as np

from primitives import closest_intersection, Sphere
from ray import Ray

@nb.njit
def roll_ray_type(specular_chance, reflection_chance, refraction_chance):
    """
    Returns a tuple of booleans that indicate whether to do specular, reflection or refraction
    """
    r = np.random.rand()
    do_specular = r < specular_chance
    do_reflection = specular_chance < r < specular_chance + reflection_chance
    do_refraction = specular_chance + reflection_chance < r < specular_chance + reflection_chance + refraction_chance


    return do_specular, do_reflection, do_refraction

@nb.njit
def schlick(IOR, c):
    # https://graphicscompendium.com/raytracing/11-fresnel-beer
    F0 = (IOR-1)**2/(IOR+1)**2

    return F0 + (1-F0)*c**5

@nb.njit()
def sample_uniform_hemisphere(r1,r2):
    sin_theta = np.sqrt(1-r1**2)
    phi = 2*np.pi*r2

    x = sin_theta*np.cos(phi)
    y = r1 # cos(theta)
    z = sin_theta*np.sin(phi)

    return np.array([x,y,z])

@nb.njit()
def create_local_coordinate_system(N):
    N_norm = np.sqrt(N[0]**2 + N[1]**2 + N[2]**2)
    if (N[0]>N[1]):
        Nt = np.array([N[2], 0, -N[0]]) / N_norm
    else:
        Nt = np.array([0, -N[2], N[1]]) / N_norm
    Nb = np.cross(N,Nt)

    return Nb, Nt, N


@nb.njit()
def local2world(P_local, N_world):
    Nt, Nb, _ = create_local_coordinate_system(N_world)

    A = np.array([[Nb[0], N_world[0], Nt[0]],
                  [Nb[1], N_world[1], Nt[1]],
                  [Nb[2], N_world[2], Nt[2]]])

    C = np.zeros(3)
    for i in nb.prange(3):
        for j in nb.prange(3):
            C[i] += A[i,j] * P_local[j]

    return C


@nb.njit
def trace_ray(ray, objects, materials, skybox, depth=1, max_depth=5):

    closest_t, closest_index, closest_normal = closest_intersection(ray, objects)

    if closest_t == np.inf:
        sphere = Sphere(np.array([0,0,0], np.float64), 1)
        closest_t, closest_index, closest_normal = closest_intersection(ray, (sphere,))

        return skybox.get_color(ray.P(closest_t)-ray.origin)
    else:
        if depth > max_depth:
            return np.array([0,0,0], dtype=np.float64)

        material = materials[closest_index]


        local_color = np.array([0,0,0], np.float64)

        # calculate whether to do specular, reflection or refraction
        do_specular, do_reflection, do_refraction = roll_ray_type(material.specular_chance, material.reflection_chance, material.refraction_chance)

        P = ray.P(closest_t)
        N = closest_normal
        ray_dir_dot_N = np.dot(ray.direction, N)
        if do_specular:
            ray_dir = ray.direction - 2*np.dot(ray.direction, N)*N
            new_ray = Ray(P, ray_dir, ray.tmin, ray.tmax)
        elif do_reflection:
            ray_dir = ray.direction - 2*np.dot(ray.direction, N)*N
            new_ray = Ray(P, ray_dir, ray.tmin, ray.tmax)
        elif do_refraction:
            d_Re = ray.direction - 2*(ray_dir_dot_N)*N

            # are we inside or outside the sphere?
            out_to_in = ray_dir_dot_N < 0

            IOR = material.refractive_index if out_to_in else 1/material.refractive_index
            nl = N if out_to_in else -N

            cos_theta = np.dot(ray.direction, nl)
            cos2_phi = 1.0 - IOR**2 * (1-cos_theta**2)

            if cos2_phi < 0:
                ray_dir = d_Re
            else:
                d_Tr = (IOR*ray.direction - (IOR*cos_theta + np.sqrt(cos2_phi))*nl)
                c = 1.0 - (-cos_theta if out_to_in else np.dot(d_Tr, N))
                Re = schlick(IOR, c)

                p_Re = 0.25 + 0.5 * Re

                if np.random.rand() < p_Re:
                    ray_dir = d_Re
                else:
                    Tr = 1.0 - Re
                    ray_dir = d_Tr

            new_ray = Ray(P, ray_dir, ray.tmin, ray.tmax)
        else:
            N_world = np.array([0,1,0])
            r1 = np.random.rand()
            r2 = np.random.rand() # cos(theta)
            local_coordinate = sample_uniform_hemisphere(r1,r2)
            ray_dir = local2world(local_coordinate, N_world)

            ray_prob = 1/(2*np.pi)
            new_ray = Ray(P, ray_dir, ray.tmin, ray.tmax)

        ## Color ##
        throughput = material.albedo

        if do_refraction:
            throughput = np.ones(3) * np.exp(-material.absorption * closest_t)


        return material.emissive + material.albedo * trace_ray(new_ray, objects, materials, skybox, depth+1, max_depth)
