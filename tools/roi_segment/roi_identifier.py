from typing import List
import cv2
import numpy as np
import dlib
import json

class Roi:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.classifier_dlib = dlib.shape_predictor(
            "tools/roi_segment/shape_predictor_81_face_landmarks.dat")
        self.regions_of_interest = self.read_regions_of_interest()
        self.landmarks = None
        self.ret = None

    def detect_faces(self, image: np.ndarray):
        """
        Detect faces in the image and return bounding boxes in (x, y, w, h) format.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        list
            A list of face bounding boxes [x, y, w, h].
        """
        detected = self.face_detector(image, 1)
        face_bboxes = []

        for rect in detected:
            x = rect.left()
            y = rect.top()
            w = rect.right() - rect.left()
            h = rect.bottom() - rect.top()
            face_bboxes = [x, y, w, h]

        return face_bboxes

    def _facial_landmarks(self, image: np.ndarray, face_points):
        """
        Detects faces in an image and extracts their corresponding facial landmarks.

        Parameters:
        ----------
        image : numpy.ndarray
            The input image where faces and facial landmarks need to be detected.
        ret : list, optional
            A list of face bounding boxes in the format [x_coord, y_coord, width, height].
            If not provided, it will be calculated using `FACE_DETECTOR`.

        Returns:
        -------
        tuple or None
            - `landmarks`: A list of matrices, where each matrix contains the `(x, y)` coordinates
              of facial landmarks for a detected face.
            - `ret`: A list of rectangles in the format `rectangles[[(x1, y1), (x2, y2)]]`.
            Returns `None` if no faces are detected in the image.
        """
        # If ret is not provided, use FACE_DETECTOR to calculate it
        self.ret = dlib.rectangles()
        for x, y, w, h in face_points:
            self.ret.append(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))

        self.landmarks = [
            np.matrix([[p.x, p.y] for p in self.classifier_dlib(image, r).parts()])
            for r in self.ret
        ]

        return self.landmarks, self.ret

    def extract_rois(self, frame: np.ndarray, face: np.ndarray, roi_count: int = 21):
        """
        Extracts all regions of interest (ROIs) from the image.

        Args:
            frame (np.ndarray): Input image.
            roi_count (int): Number of ROIs to extract.

        Returns:
            List[np.ndarray]: List of extracted ROIs.
        """
        self._facial_landmarks(frame, face)
        return [self._get_roi(frame, i) for i in range(1, roi_count + 1)]

    def _get_roi(self, image: np.ndarray, roi: int):
        """
        Processes facial landmarks to calculate and crop specific regions of interest.

        Args:
            image (np.ndarray): Input image.
            roi (int): Index of the ROI to extract.

        Returns:
            Optional[np.ndarray]: Cropped ROI or None if no face is detected.
        """
        try:
            if self.ret is None:
                print('No faces detected in the image.')
                raise RuntimeError('No faces detected in the image.')

            idx_r = self.main_face_index(self.ret)

            lm = self.landmarks[idx_r]
            index = self.regions_of_interest.get(str(roi))
            points = lm[index]

            # Find the bounding box of the ROI
            x, y, w, h = cv2.boundingRect(points)

            # Crop the ROI from the original image
            roi_cropped = image[y:y + h, x:x + w]

            return np.asarray(roi_cropped)
        except RuntimeError as e:
            print(f'Error extracting ROI: {e}')
            return None

    @staticmethod
    def main_face_index(ret: List[dlib.rectangle]) -> int:
        """
        Returns the index of the main face (the largest detected face).

        Args:
            ret (List[dlib.rectangle]): List of detected face rectangles.

        Returns:
            int: Index of the main face.
        """
        if not ret:
            return 0
        return max(enumerate(ret), key=lambda x: x[1].area())[0]

    @staticmethod
    def read_regions_of_interest():
        with open("tools/roi_segment/regions_of_interest.json", "r") as data:
            regions_of_interest = json.load(data)
        return regions_of_interest