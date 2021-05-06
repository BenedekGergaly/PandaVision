from PIL import Image as pimg
import numpy as np
import cv2.aruco as aruco
import cv2
import time
import math

np.set_printoptions(precision=2, suppress=True)


# based on this guide: https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
class Marker:
    def __init__(self):
        self.id = None
        self.corners = None
        self.object_id = None
        self.center = None
        self.top_center = None


class Item:
    def __init__(self):
        self.name = None
        self.id = None
        self.center = None
        self.orientation = None
        self.id_to_name_dict = {10: "black_top_normal",
                           11: "black_top_flipped",
                           20: "black_bottom_normal",
                           21: "black_bottom_flipped",
                           30: "blue_top_normal",
                           31: "blue_top_flipped",
                           40: "blue_bottom_normal",
                           41: "blue_bottom_flipped",
                           50: "white_top_normal",
                           51: "white_top_flipped",
                           60: "white_bottom_normal",
                           61: "white_bottom_flip",
                           70: "pcb_normal",
                           71: "pcb_flipped"
                           }

    def __repr__(self):
        return f"{self.name} at {self.center}"


class ObjectFinder:
    def __init__(self):
        # marker locations in robot world frame (clockwise, starting top left)
        # top left corners, in robot world frame, in mm
        id0_x = 220
        id0_y = 140
        id1_x = 110
        id1_y = -270
        id2_x = 610
        id2_y = 130
        id3_x = 510
        id3_y = -290
        aruco_coordinates_0 = np.array([[id0_x, id0_y, 0], [id0_x, id0_y - 40, 0], [id0_x - 40, id0_y - 40, 0], [id0_x - 40, id0_y, 0]])
        aruco_coordinates_1 = np.array([[id1_x, id1_y, 0], [id1_x, id1_y - 40, 0], [id1_x - 40, id1_y - 40, 0], [id1_x - 40, id1_y, 0]])
        aruco_coordinates_2 = np.array([[id2_x, id2_y, 0], [id2_x, id2_y - 40, 0], [id2_x - 40, id2_y - 40, 0], [id2_x - 40, id2_y, 0]])
        aruco_coordinates_3 = np.array([[id3_x, id3_y, 0], [id3_x, id3_y - 40, 0], [id3_x - 40, id3_y - 40, 0], [id3_x - 40, id3_y, 0]])
        self.marker_reference_coordinates = np.array([aruco_coordinates_0, aruco_coordinates_1, aruco_coordinates_2, aruco_coordinates_3], dtype=np.float32)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # ids = np.float32(np.array([0, 1, 2, 3]))
        # board = aruco.Board_create(self.markers, aruco_dict, ids)

        self.default_intrinsic_matrix = np.array([[1371.46, 0, 976.27], [0, 1370.65, 571.26], [0, 0, 1]]) #for Logitech C922
        default_distortion = np.array([0.10501412, -0.21740769, 0.00152855, -0.00110849, 0.08781253], dtype=np.float32) #for Logitech C922
        self.distortion = default_distortion.T
        self.reference_image = np.array([])
        self.intrinsic_matrix_inverse = None
        self.t_vector = None
        self.r_matrix_inverse = None

    def find_objects(self, np_image, world_z=0, debug=False): #world_z is height from table in mm
        timer = time.time()

        if len(self.reference_image) == 0:
            self.reference_image = np_image
        opencv_image_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("a", opencv_image_gray)
        # cv2.waitKey()
        (detected_corners, detected_ids, rejected_image_points) = aruco.detectMarkers(opencv_image_gray, self.aruco_dict)

        #drawing and showing the detected and rejected markers
        img_debug = aruco.drawDetectedMarkers(np_image, detected_corners, detected_ids, borderColor=(0, 255, 0))
        img_debug = aruco.drawDetectedMarkers(img_debug, rejected_image_points, borderColor=(0, 0, 255))
        img_small = cv2.resize(img_debug, (1536, 864))
        if debug:
            cv2.imshow("detected_corners", img_small)
            cv2.waitKey()

        #removing unnecessary dimensions from arrays
        detected_corners = np.array(detected_corners)
        detected_corners = np.array(detected_corners).reshape((len(detected_ids), 4, 2))  # opencv stupid
        detected_ids = np.array(detected_ids).reshape((len(detected_ids)))  # opencv stupid

        #filling in marker objects
        all_markers = []
        for i, id in enumerate(detected_ids):
            marker = Marker()
            marker.id = id
            marker.corners = detected_corners[i]
            all_markers.append(marker)

        calibration_markers = []
        object_markers = []

        #seperating into calibration and object markers. IDs less than 10 are reserved for calibration markers.
        for marker in all_markers:
            if marker.id < 10:
                calibration_markers.append(marker)
            else:
                object_markers.append(marker)

        #accuarcy is okay with only 3 markers
        if len(calibration_markers) <= 3:
            print("[WARNING] calibration found less than 4 markers")
        assert (len(calibration_markers) >= 3), "Cannot work with 2 or less markers"

        for marker in object_markers:
            marker.center = (marker.corners[0] + marker.corners[2]) / 2
            marker.top_center = (marker.corners[0] + marker.corners[1]) / 2

        # putting all the coordinates into arrays understood by solvePNP
        marker_world_coordinates = []
        image_coordinates = []
        for marker in calibration_markers:
            for corner_id, corner in enumerate(marker.corners):
                image_coordinates.append(corner)
                marker_world_coordinates.append(self.marker_reference_coordinates[marker.id][corner_id])
        marker_world_coordinates = np.array(marker_world_coordinates)
        image_coordinates = np.array(image_coordinates)

        # finding exstrinsic camera parameters
        error, r_vector, self.t_vector = cv2.solvePnP(marker_world_coordinates, image_coordinates, self.default_intrinsic_matrix, self.distortion)

        #matrix magic
        r_matrix, jac = cv2.Rodrigues(r_vector)
        self.r_matrix_inverse = np.linalg.inv(r_matrix)
        self.intrinsic_matrix_inverse = np.linalg.inv(self.default_intrinsic_matrix)

        detected_items = []
        for marker in object_markers:
            center_w = self._solve(marker.center, world_z)
            top_center_w = self._solve(marker.top_center, world_z)
            center_vector_w = top_center_w - center_w
            angle = math.atan2(center_vector_w[1], center_vector_w[0])
            item = Item()
            item.center = center_w
            item.orientation = angle
            item.id = marker.id
            item.name = item.id_to_name_dict[item.id]
            detected_items.append(item)

        # print("[INFO] Calibration took %.2f seconds" % (time.time() - timer))
        return detected_items

    def _solve(self, pixel_coordinates, world_z):
        scaling_factor = 940
        index = 0
        # finding correct scaling factor by adjusting it until the calculated Z is very close to 0, mathematically correct way didn't work ¯\_(ツ)_/¯
        while True:
            pixel_coordinates_copy = np.array([[pixel_coordinates[0], pixel_coordinates[1], 1]]).T
            pixel_coordinates_copy = scaling_factor * pixel_coordinates_copy
            xyz_c = self.intrinsic_matrix_inverse.dot(pixel_coordinates_copy)
            xyz_c = xyz_c - self.t_vector
            world_coordinates = self.r_matrix_inverse.dot(xyz_c)
            index += 1
            if index > 1000:
                raise Exception("Scaling factor finding is taking longer than 1000 iterations. This is a design time issue.")
            if world_coordinates[2][0] > world_z + 0.5:
                scaling_factor += 1
            elif world_coordinates[2][0] < world_z - 0.5:
                scaling_factor -= 1
            else:
                break
        world_coordinates = np.reshape(world_coordinates, (3)) #removing unecessary dimension
        return world_coordinates


if __name__ == "__main__":
    use_camera = True
    camera_device_id = 1
    object_finder = ObjectFinder()
    if use_camera:
        camera = cv2.VideoCapture(camera_device_id, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(1)
        _, img = camera.read()
    else:
        img = cv2.imread("test.jpg")
    img_undistort = cv2.undistort(img, object_finder.default_intrinsic_matrix, object_finder.distortion)
    # cv2.imshow("cap", img)
    # cv2.imshow("undistorted", img_undistort)
    # cv2.waitKey()
    print(object_finder.find_objects(img, 0))
