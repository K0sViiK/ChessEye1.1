import cv2
from chessboardfinder import ChessboardFinder


class chessboard_scanner():

    def typeout(self, predictions):
        fen_output = ""
        predictions.reshape(64)
        predictions.reshape(64)
        for i in range(8):
            empty_spaces = 0
            for j in range(8):
                if self.fen_sign[predictions[i * 8 + j]] == "e":
                    empty_spaces += 1
                    if j == 7:
                        fen_output += str(empty_spaces)
                else:
                    if empty_spaces > 0:
                        fen_output += str(empty_spaces)
                        empty_spaces = 0
                        fen_output += self.fen_sign[predictions[i * 8 + j]]
                    else:
                        fen_output += self.fen_sign[predictions[i * 8 + j]]
            if i != 7:
                fen_output += "/"
        return fen_output

    def main(self):
        detector = ChessboardFinder("models/detection", "models/classification.h5")
        capture = cv2.imread('img1.jpg')

        sample = cv2.resize(capture, (1000, 750))
        cv2.imshow('img', sample)
        cv2.waitKey(0)

        corners = detector.find_corners(capture)
        if len(corners) == 4:
            predictions = detector.find_board(capture, corners)
            self.typeout(self, predictions)



if __name__ == '__main__':
    cbs = chessboard_scanner()
    cbs.main()
