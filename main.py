import numba as nb
from numba import literal_unroll
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

from primitives import Sphere, Mesh
from materials import Material
from camera import Camera
from ray import Ray
from trace import trace_ray
from skybox import SkyBox
from load_obj import load_obj

@nb.njit(parallel=True)
def render(camera, objects, materials, skybox):
    xmin = -int(camera.resolution[0]//2)+1
    xmax = int(camera.resolution[0]//2)
    ymin = -int(camera.resolution[1]//2)+1
    ymax = int(camera.resolution[1]//2)

    for x in nb.prange(xmin, xmax):
        for y in nb.prange(ymin, ymax):
            color = np.array([0,0,0], dtype=np.float64)
            n_samples = 100
            print(x,y)
            for _ in nb.prange(n_samples):
                noise = 1
                x_ = x + (np.random.rand()-0.5)*noise
                y_ = y + (np.random.rand()-0.5)*noise

                D = camera.canvas2world(x_, y_)
                ray = Ray(camera.position, D, 0.001, np.inf)

                color += trace_ray(ray, objects, materials, skybox)

            color /= n_samples
            color_r = min(color[0], 1)
            color_g = min(color[1], 1)
            color_b = min(color[2], 1)
            color = np.array([color_r, color_g, color_b], dtype=np.float64)

            camera.write_pixel(x, y, color)

    return camera.pixels




if __name__ == "__main__":
    camera = Camera(position=np.array([0,0,-5],np.float64))


    teapot = load_obj("objects/teapot.obj")

    objects = (teapot, Sphere(position=np.array([-5,0,0],np.float64), radius=0.5))
    object_materials = (Material(albedo=np.array([1,1,1], dtype=np.float64),
                                 emissive=np.array([0.0,0.0,0.0], dtype=np.float64),
                                 specular_chance=0.3,
                                 reflection_chance=0.2,
                                 refraction_chance=0,
                                 absorption=0,
                                 refractive_index=1),
                        Material(albedo=np.array([0,0,0], dtype=np.float64),
                                 emissive=np.array([5.0,5.0,5.0], dtype=np.float64),
                                 specular_chance=0.3,
                                 reflection_chance=0.2,
                                 refraction_chance=0,
                                 absorption=0,
                                 refractive_index=1),)

    # objects = (
    #     Sphere(position=np.array((0,0,3),np.float64), radius=1),
    #     Sphere(position=np.array((-2,0,4),np.float64),  radius=1),
    #     Sphere(position=np.array((2,0,4),np.float64), radius=1),
    #     Sphere(position=np.array((0,-5001,0),np.float64), radius=5000),
    #     Sphere(position=np.array((-5,0,1),np.float64), radius=0.5),
    #     Sphere(position=np.array((-2,0,1),np.float64), radius=0.5),
    # )


    # object_materials = (Material(albedo=np.array([1,1,1], dtype=np.float64),
    #                              emissive=np.array([0,0,0], dtype=np.float64),
    #                              specular_chance=0.0,
    #                              reflection_chance=0.0,
    #                              refraction_chance=1.0,
    #                              absorption=0.5,
    #                              refractive_index=1.8),
    #                     Material(albedo=np.array([0,0,1], dtype=np.float64),
    #                              emissive=np.array([0,0,0], dtype=np.float64),
    #                              specular_chance=0.0,
    #                              reflection_chance=0.0,
    #                              refraction_chance=0.0,
    #                              absorption=0,
    #                              refractive_index=1.0),
    #                     Material(albedo=np.array([1,1,1], dtype=np.float64),
    #                              emissive=np.array([0,0,0], dtype=np.float64),
    #                              specular_chance=0.0,
    #                              reflection_chance=0.0,
    #                              refraction_chance=1.0,
    #                              absorption=0.5,
    #                              refractive_index=1.0),
    #                     Material(albedo=np.array([0,1,1], dtype=np.float64),
    #                              emissive=np.array([0,0,0], dtype=np.float64),
    #                              specular_chance=0.0,
    #                              reflection_chance=0.0,
    #                              refraction_chance=0.0,
    #                              absorption=0.0,
    #                              refractive_index=1.0),
    #                     Material(albedo=np.array([0,0,0], dtype=np.float64),
    #                              emissive=np.array([12,12,12], dtype=np.float64),
    #                              specular_chance=0.0,
    #                              reflection_chance=0.0,
    #                              refraction_chance=0.0,
    #                              absorption=0,
    #                              refractive_index=1.0),
    #                     Material(albedo=np.array([0,0,0], dtype=np.float64),
    #                              emissive=np.array([12,12,12], dtype=np.float64),
    #                              specular_chance=0.0,
    #                              reflection_chance=0.0,
    #                              refraction_chance=0.0,
    #                              absorption=0,
    #                              refractive_index=1.0),)

    image = Image.open("skyboxes/miramar.jpeg")
    image = np.array(image, dtype=np.float64)
    skybox = SkyBox(image)

    pixels = render(camera, objects, object_materials, skybox)

    plt.imshow(pixels)
    plt.gca().axis("off")
    plt.show()
