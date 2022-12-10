import cv2
import imutils as imutils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist


class ChessboardFinder:
    # here we use the first model to find the corners of the chessboard
    cam_m = np.array([[3.13479737e+03, 0.00000000e+00, 2.04366415e+03],
                      [0.00000000e+00, 3.13292625e+03, 1.50698424e+03],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_m = np.array([[2.08959569e-01, -9.49127601e-01, -
    2.70203242e-03, -1.20066339e-04, 1.33323676e+00]])
    pieceLabels = ["EMPTY", "W_PAWN", "B_PAWN", "W_QUEEN", "B_QUEEN",
                   "W_KING", "B_KING", "W_ROOK", "B_ROOK",
                   "W_KNIGHT", "B_KNIGHT", "W_BISHOP", "B_BISHOP"]
    maximum_pieces = [np.inf, 8, 8, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    # we store the maximal starting amount of pieces, this helps us discard unlikely setups during our scans
    destination_coordinates = np.array([[0, 80, 0], [80, 80, 0], [0, 0, 0]], dtype=np.float32)

    def __init__(self, detection_model_location, classification_model_location):
        tf.config.optimizer.set_jit(True)
        phys_dev = tf.config.experimental.list_physical_devices('GPU')
        if len(phys_dev) > 0:
            tf.config.experimental.set_memory_growth(phys_dev[0], True)

        self.detection_model = keras.models.load_model(detection_model_location)
        self.classification_model = keras.models.load_model(classification_model_location)

    def find_corners(self, img):
        self.prepareimg(img)
        predictions = self.detection_model.predict(
            np.expand_dims(self.img_color_rgb, axis=0))

        if predictions is None:
            print("Cannot find chessboard. Please try a different input.")
            return []

        self.corner_pts = predictions

        return self.corner_pts

    def find_board(self, img, corners):
        # with an image and found corners, find the edges of the board
        self.img_nn = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_nn = self.img_nn / 255
        scale = img.shape[1] / 512
        scaled_corners = np.array(corners) * scale
        _, r, t = cv2.solvePnP(self.destination_coordinates, scaled_corners, self.cam_m, self.dist_m)
        imgs = []
        for i in range(8):
            for j in range(8):
                cell_img = self.get_square_img(7 - i, j, r, t)
                imgs.append(cell_img)
        imgs = np.array(imgs)
        predictions = self.classification_model.predict(imgs, batch_size=8)
        return predictions


    def get_square_img(self, cellX, cellY, r, t):
        black = np.zeros((100, 100, 1), dtype="uint8")
        cv2.circle(black, (5 + 10 * cellX, 5 + 10 * cellY), 5, 255, 0)
        low_pos = np.argwhere(black)
        up_pos = low_pos.copy()
        up_pos[:, 2] = 13
        pos = np.concatenate((low_pos, up_pos)).astype(np.float32)
        img_pts, _ = cv2.projectPoints(pos, r, t, self.cam_m, self.dist_m)
        img_pts = img_pts.reshape(-1, 2)
        rect = cv2.minAreaRect(img_pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # stretch the rotated rectangle to the straightened one
        temp_img = cv2.warpPerspective(
            self.img_nn, M, (int(width), int(height)))
        if temp_img.shape[0] < temp_img.shape[1]:
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height = temp_img.shape[0]
        width = temp_img.shape[1]
        aspRatio = width / height
        given_ratio = 1 / 2
        if aspRatio > given_ratio:
            new_width = height * given_ratio
            diff = round((width - new_width) / 2)
            new_mat = temp_img[:, diff:(width - diff), :]
        elif aspRatio < given_ratio:
            new_height = width / given_ratio
            diff = round((height - new_height) / 2)
            new_mat = temp_img[diff:(height - diff), :, :]
        else:
            new_mat = temp_img
        new_mat = imutils.resize(new_mat, width=100)
        return cv2.resize(new_mat, (100, 200)).reshape(200, 100, 3)

        # ---------------------

    def rotate_and_predict(self, angle):

        rot_imgs = imutils.rotate(self.img_color_rgb, angle=-angle)
        predictions = self.detection_model.predict(
            np.expand_dims(rot_imgs, axis=0))
        if self.overlappingPoints(predictions):
            return None
        # Correct rotation of predicted points
        m = cv2.getRotationMatrix2D((256, 192), angle, 1)
        predictions[0, :, 2] = 1
        predictions[0, :, 0:2] = (m @ predictions[0].T).T
        return predictions

    def overlappingPoints(self, predictions):

        points = predictions[0, 0:4, 0:2]
        distances = np.triu(cdist(points, points))
        distances[distances == 0] = np.inf
        indexlist = np.argwhere(distances < 30)
        return not len(indexlist) == 0

    # ---------------------------

    def prepareimg(self, img):
        # resize and prepare the image color
        self.img_color_rgb = cv2.resize(img, (512, 384))
        self.img_color_rgb = cv2.cvtColor(self.img_color_rgb, cv2.COLOR_BGR2RGB)
