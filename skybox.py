import numba as nb
import numpy as np


@nb.experimental.jitclass([
    ("image", nb.float64[:,:,:]),
    ("images", nb.float64[:,:,:,:]),
])
class SkyBox():
    def __init__(self, image):
        self.image = image

        height, total_width, _ = self.image.shape
        face_width = total_width // 4

        self.images = np.array([
                self.image[face_width:2*face_width, 2*face_width:3*face_width], # POSITIVE X
                self.image[face_width:2*face_width, 0:face_width],              # NEGATIVE X
                self.image[0:face_width, face_width:2*face_width],              # POSITIVE Y
                self.image[2*face_width:3*face_width, face_width:2*face_width], # NEGATIVE Y
                self.image[face_width:2*face_width, face_width:2*face_width],   # POSITIVE Z
                self.image[face_width:2*face_width, 3*face_width:4*face_width], # NEGATIVE Z
                ], dtype=np.float64)


    def sphere2pixel(self, P):
        x,y,z = P
        abs_x, abs_y, abs_z = np.abs(P)

        x_positive = x>0
        y_positive = y>0
        z_positive = z>0

        # POSITIVE X
        if x_positive and abs_x >= abs_y and abs_x >= abs_z:
            max_axis = abs_x
            uc = -z
            vc = y
            index = 0

        # NEGATIVE X
        if not x_positive and abs_x >= abs_y and abs_x >= abs_z:
            max_axis = abs_x
            uc = z
            vc = y
            index = 1

        # POSITIVE Y
        if y_positive and abs_y >= abs_x and abs_y >= abs_z:
            max_axis = abs_y
            uc = x
            vc = -z
            index = 2

        # NEGATIVE Y
        if not y_positive and abs_y >= abs_x and abs_y >= abs_z:
            max_axis = abs_y
            uc = x
            vc = z
            index = 3

        # POSITIVE Z
        if z_positive and abs_z >= abs_x and abs_z >= abs_y:
            max_axis = abs_z
            uc = x
            vc = y
            index = 4

        # NEGATIVE Z
        if not z_positive and abs_z >= abs_x and abs_z >= abs_y:
            max_axis = abs_z
            uc = -x
            vc = y
            index = 5

        # convert from (-1,1) to (0,1)
        u = 0.5 * (uc / max_axis + 1)
        v = 0.5 * (vc / max_axis + 1)

        pixel = [v,u]

        pixel = (int(self.images[index].shape[0]-pixel[0]*self.images[index].shape[0]),
                    int(pixel[1]*self.images[index].shape[1])-1)

        return pixel[0], pixel[1], index

    def get_color(self, P):
        x,y, index = self.sphere2pixel(P)
        return self.images[index][x,y]



@nb.experimental.jitclass([
    ("image", nb.float64[:,:,:]),
    ("images", nb.float64[:,:,:,:]),
])
class SkyBox():
    def __init__(self, image):
        self.image = image

        height, total_width, _ = self.image.shape
        face_width = total_width // 4


        self.images = np.zeros((6,face_width,face_width,3), dtype=np.float64)

        images = [
                self.image[face_width:2*face_width, 2*face_width:3*face_width], # POSITIVE X
                self.image[face_width:2*face_width, 0:face_width], # NEGATIVE X
                self.image[0:face_width, face_width:2*face_width], # POSITIVE Y
                self.image[2*face_width:3*face_width, face_width:2*face_width], # NEGATIVE Y
                self.image[face_width:2*face_width, face_width:2*face_width], # POSITIVE Z
                self.image[face_width:2*face_width, 3*face_width:4*face_width], # NEGATIVE Z
                ]

        for i, image in enumerate(images):
            self.images[i] = image


    def sphere2pixel(self, P):
        x,y,z = P
        abs_x, abs_y, abs_z = np.abs(P)

        x_positive = x>0
        y_positive = y>0
        z_positive = z>0

        # POSITIVE X
        if x_positive and abs_x >= abs_y and abs_x >= abs_z:
            max_axis = abs_x
            uc = -z
            vc = y
            index = 0

        # NEGATIVE X
        if not x_positive and abs_x >= abs_y and abs_x >= abs_z:
            max_axis = abs_x
            uc = z
            vc = y
            index = 1

        # POSITIVE Y
        if y_positive and abs_y >= abs_x and abs_y >= abs_z:
            max_axis = abs_y
            uc = x
            vc = -z
            index = 2

        # NEGATIVE Y
        if not y_positive and abs_y >= abs_x and abs_y >= abs_z:
            max_axis = abs_y
            uc = x
            vc = z
            index = 3

        # POSITIVE Z
        if z_positive and abs_z >= abs_x and abs_z >= abs_y:
            max_axis = abs_z
            uc = x
            vc = y
            index = 4

        # NEGATIVE Z
        if not z_positive and abs_z >= abs_x and abs_z >= abs_y:
            max_axis = abs_z
            uc = -x
            vc = y
            index = 5

        # convert from (-1,1) to (0,1)
        u = 0.5 * (uc / max_axis + 1)
        v = 0.5 * (vc / max_axis + 1)

        pixel = [v,u]

        pixel = (int(self.images[index].shape[0]-pixel[0]*self.images[index].shape[0]),
                    int(pixel[1]*self.images[index].shape[1])-1)

        return pixel[0], pixel[1], index

    def get_color(self, P):
        x,y, index = self.sphere2pixel(P)
        return self.images[index][x,y]/255



if __name__ == "__main__":
    from PIL import Image

    image = Image.open("skyboxes/miramar.jpeg")
    image = np.array(image, dtype=np.float64)
    SkyBox(image)