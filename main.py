from PIL import Image as pimg
import numpy as np
import cv2.aruco as aruco
import cv2
import time
np.set_printoptions(precision=2, suppress=True)

# based on this guide: https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
class Calibration:
    def __init__(self):
        # marker locations in robot world frame (clockwise, starting top left)
        #top left corners
        id0_x = 220
        id0_y = 140
        id1_x = 110
        id1_y = -270
        id2_x = 610
        id2_y = 130
        id3_x = 510
        id3_y = -290
        aruco_coordinates_0 = np.array([[id0_x, id0_y, 0], [id0_x, id0_y-40, 0], [id0_x-40, id0_y-40, 0], [id0_x-40, id0_y, 0]])
        aruco_coordinates_1 = np.array([[id1_x, id1_y, 0], [id1_x, id1_y-40, 0], [id1_x-40, id1_y-40, 0], [id1_x-40, id1_y, 0]])
        aruco_coordinates_2 = np.array([[id2_x, id2_y, 0], [id2_x, id2_y-40, 0], [id2_x-40, id2_y-40, 0], [id2_x-40, id2_y, 0]])
        aruco_coordinates_3 = np.array([[id3_x, id3_y, 0], [id3_x, id3_y-40, 0], [id3_x-40, id3_y-40, 0], [id3_x-40, id3_y, 0]])
        self.markers = np.array([aruco_coordinates_0, aruco_coordinates_1, aruco_coordinates_2, aruco_coordinates_3], dtype=np.float32)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # ids = np.float32(np.array([0, 1, 2, 3]))
        # board = aruco.Board_create(self.markers, aruco_dict, ids)


        self.default_intrinsic_matrix = np.array([[1371.46, 0, 976.27], [0, 1370.65, 571.26], [0, 0, 1]])
        default_distortion = np.array([0.10501412, -0.21740769,  0.00152855, -0.00110849,  0.08781253], dtype=np.float32)
        self.distortion = default_distortion.T
        self.reference_image = np.array([])

    def calibrate(self, np_image, x_coordinate, y_coordinate, world_z):
        assert x_coordinate > 0 and y_coordinate > 0 and world_z >= 0, "[FATAL] aruco calibrate got invalid x, y or z"
        timer = time.time()

        if len(self.reference_image) == 0:
            self.reference_image = np_image
        opencv_image_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("a", opencv_image_gray)
        #cv2.waitKey()
        (all_marker_corners, detected_ids, rejected_image_points) = aruco.detectMarkers(opencv_image_gray, self.aruco_dict)
        all_marker_corners = np.array(all_marker_corners).reshape((len(detected_ids), 4, 2)) #opencv stupid
        for marker_corner in all_marker_corners:
            for corner in marker_corner:
                cv2.circle(np_image, (corner[0], corner[1]), 5, (0, 0, 255), 2)
        img_small = cv2.resize(np_image, (1536, 864))
        #cv2.imshow("detected_corners", img_small)
        #cv2.waitKey()
        detected_ids = np.array(detected_ids).reshape((len(detected_ids))) #opencv stupid
        if len(detected_ids) <= 3:
            print("[WARNING] calibration found less than 4 markers")
        assert (len(detected_ids) >= 3), "Cannot work with 2 or less markers"

        #putting all the coordinates into arrays understood by solvePNP
        marker_world_coordinates = None
        image_coordinates = None
        for i in range(len(detected_ids)):
            if i == 0:
                marker_world_coordinates = self.markers[detected_ids[i]]
                image_coordinates = all_marker_corners[i]
            else:
                marker_world_coordinates = np.concatenate((marker_world_coordinates, self.markers[detected_ids[i]]))
                image_coordinates = np.concatenate((image_coordinates, all_marker_corners[i]))

        # finding exstrinsic camera parameters
        error, r_vector, t_vector = cv2.solvePnP(marker_world_coordinates, image_coordinates, self.default_intrinsic_matrix, self.distortion)

        r_matrix, jac = cv2.Rodrigues(r_vector)
        r_matrix_inverse = np.linalg.inv(r_matrix)
        intrinsic_matrix_inverse = np.linalg.inv(self.default_intrinsic_matrix)

        # finding correct scaling factor by adjusting it until the calculated Z is very close to 0, mathematically correct way didn't work ¯\_(ツ)_/¯
        scaling_factor = 940
        index = 0
        while True:
            pixel_coordinates = np.array([[x_coordinate, y_coordinate, 1]]).T
            pixel_coordinates = scaling_factor * pixel_coordinates
            xyz_c = intrinsic_matrix_inverse.dot(pixel_coordinates)
            xyz_c = xyz_c - t_vector
            world_coordinates = r_matrix_inverse.dot(xyz_c)
            index += 1
            if index > 1000:
                raise Exception("aruco.py: scaling factor finding is taking longer than 1000 iterations")
            if world_coordinates[2][0] > world_z + 0.5:
                scaling_factor += 1
            elif world_coordinates[2][0] < world_z - 0.5:
                scaling_factor -= 1
            else:
                break
        #print("[INFO] Calibration took %.2f seconds" % (time.time() - timer))
        return np.array([world_coordinates[0][0], world_coordinates[1][0], world_coordinates[2][0]])

if __name__ == "__main__":
    use_camera = False
    try:
        c = Calibration()
        #img = cv2.imread("test.jpg")
        if use_camera:
            camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            _, img = camera.read()
        else:
            img = cv2.imread("test.jpg")
        img_undistort = cv2.undistort(img, c.default_intrinsic_matrix, c.distortion)
        #cv2.imshow("cap", img)
        #cv2.imshow("undistorted", img_undistort)
        #cv2.waitKey()
        print(c.calibrate(img, 953, 47, 0))
    finally:
        camera.release()