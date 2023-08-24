import numba as nb
import numpy as np


@nb.experimental.jitclass([
    ("albedo", nb.float64[:]),
    ("emissive", nb.float64[:]),
    ("specular_chance", nb.float64),
    ("reflection_chance", nb.float64),
    ("refraction_chance", nb.float64),
    ("refractive_index", nb.float64),
])
class Material():
    def __init__(self, albedo, emissive, specular_chance, reflection_chance, refraction_chance, refractive_index):
        self.albedo = albedo
        self.emissive = emissive
        self.specular_chance = specular_chance
        self.reflection_chance = reflection_chance
        self.refraction_chance = refraction_chance
        self.refractive_index = refractive_index
