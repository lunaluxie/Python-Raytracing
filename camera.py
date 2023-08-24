import numba as nb
import numpy as np

@nb.experimental.jitclass([
    ("position", nb.float64[:]),
    ("orientation", nb.float64[:,:]),
    ("resolution", nb.int64[:]),
    ("pixels", nb.float64[:, :, :]),
    ("d", nb.float64),
    ("w", nb.float64),
    ("h", nb.float64),
])
class Camera():
    def __init__(self,
                position=np.array([0,0,0],np.float64),
                orientation=np.array([[1,0,0],[0,1,0],[0,0,1]], np.float64),
                resolution=np.array([100,100], dtype=np.int64),
                d=1, w=1, h=1):
        self.position = position
        self.orientation = orientation


        # distance from camera to canvas
        self.d = d

        # width and height of camera frame otherwise known as the FoV
        self.w = w
        self.h = h

        self.resolution = resolution

        self.pixels = np.zeros((resolution[1], resolution[0], 3), dtype=np.float64)

    def canvas2world(self, x, y):
        return np.array((x*self.w/self.resolution[0], y*self.h/self.resolution[1], self.d),dtype=np.float64)

    def _center2pixel(self, x, y):
        return (self.resolution[0]//2 - y, self.resolution[1]//2 + x)

    def write_pixel(self, x, y, color):
        x, y = self._center2pixel(x, y)
        self.pixels[x,y] = color


if __name__ == "__main__":
    c = Camera()

    print(c.canvas2world(0,0))