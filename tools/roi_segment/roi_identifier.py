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

    def _facial_landmarks(self, image: np.ndarray):
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

        ret = self.face_detector(image, 1)

        if not ret:
            return None, None

        # Format ret to look like rectangles[[(x1, y1), (x2, y2)]]
        # Extract facial landmarks
        landmarks = [
            np.matrix([[p.x, p.y] for p in self.classifier_dlib(image, r).parts()])
            for r in ret
        ]

        return landmarks, ret

    def extract_rois(self, frame: np.ndarray, roi_count: int = 21):
        """
        Extracts all regions of interest (ROIs) from the image.

        Args:
            frame (np.ndarray): Input image.
            roi_count (int): Number of ROIs to extract.

        Returns:
            List[np.ndarray]: List of extracted ROIs.
        """
        return [self._get_roi(frame, i) for i in range(1, roi_count + 1)]

    def reset_landmark(self, image: np.ndarray):
        """
        Detects landmarks in the image.

        Args:
            image (np.ndarray): Input image.
        """
        faces = self.face_detector(image, 1)
        if not faces:
            self.landmarks = None
            self.ret = None
            return

        self.landmarks, self.ret = self._facial_landmarks(image)

    def get_landmarks(self):
        """Returns the current landmarks safely."""
        return self.landmarks, self.ret

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