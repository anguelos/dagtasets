import unittest
from scenethecizer.synthecizer import *


class Tester(unittest.TestCase):

    def test_geometry_clip(self):
        e = 1e-09
        image = np.zeros([100,150,3])
        image[10:66,33:98,1]=1
        image[50:82, 10:140, 0] = 1
        ltrb = np.array([[0, 0, 200, 200],
                         [0, 55, 65, 65],
                         [55, 0, 65, 65],
                         [55, 55, 200, 65],
                         [55, 55, 65, 200],
                         [55, 55, 65, 65]],dtype='double')
        cliper=GeometricCliper([40,50,80,90])

        # Testing coordinate functor
        XY_list = cliper([ltrb[:,0],ltrb[:,2]],[ltrb[:,1],ltrb[:,3]])
        assert (XY_list[0][0] == [40, 40, 55, 55, 55, 55]).all()
        assert (XY_list[0][1] == [80, 65, 65, 80, 65, 65]).all()
        assert (XY_list[1][0] == [50, 55, 50, 55, 55, 55]).all()
        assert (XY_list[1][1] == [90, 65, 65, 65, 90, 65]).all()

        out_image, out_ltrb = cliper.apply_on_image(image, ltrb)

        # Testing image_wide bboxes.
        assert (out_ltrb == np.array([[40,50,80,90], [40, 55, 65, 65],
                                      [55, 50, 65, 65], [55, 55, 80, 65],
                                      [55, 55, 65, 90], [55, 55, 65, 65]],
                                     dtype='double')).all()

        # MSE of the sampling inside the clipping area
        mse = ((out_image[50:90,40:80,:] - image[50:90,40:80,:])**2).mean()
        assert mse < e


if __name__ == '__main__':
    unittest.main()
