import os
import time

import cv2
import numpy as np
from utils_process import crop_outer_part, \
    morphological_transformation_foreground, morphological_transformation_mask
from utils_process import save_picture, opt_triangular_points, resize_frame
from collections import Counter

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class EchoProcess:
    """
	Class that preprocess the echo videos with cropping and resizing

	Parameters

	save_foreground: boolean
		true if you want to save the foreground mask
	save_overlay: boolean
		true if you want to save the overlay mask
	validate_cropping: boolean
		true if you want to save a frame for each echo where 
		is depicted the triangle that is used to crop the image
	save_cropped_video: boolean
		true if you want to save the cropped video
	resize: boolean
		true if you want to resize every video to a standard height
		and width given by side_length
	save_frames: boolean
		true if you want to save all the frames of each video
	verbose: boolean
		true if you want to print messages on terminal
	side_length: int
		standard size that is used to resize the echo 
	
	Variables: 
	echo_info: dict
		dictionary with all the metinformation of the echo
	echo: Echo object
		echo on which the preprocessing is currently working
	triangular_mask: 2D float array
		mask that is used to crop the current echo. The values are
		set to 1 for the pixel that are considered, and Nan for the 
		cropped pixe
	left_v: int 2d tuple 
		coordinates rispectively of the (x_low y_left) of the foreground
		mask
	right_v: int 2d tuple
		coordinates rispectively of the (x_high, y_right) of the foreground
		mask
	matrix_3d: 3D float array 
		matrix that represents the video of the echo
	"""

    def __init__(self, save_foreground=False, save_overlay=False,
                 validate_cropping=False,
                 save_cropped_video=False, resize=True, save_frames=False,
                 save_pickle_echo=False, verbose=False, crop_meta_data=True,
                 side_length=400):

        # parameters
        self.save_foreground = save_foreground
        self.save_overlay = save_overlay
        self.validate_cropping = validate_cropping
        self.save_cropped_video = save_cropped_video
        self.resize = resize
        self.save_frames = save_frames
        self.save_pickle_echo = save_pickle_echo
        self.verbose = verbose
        self.side_length = side_length
        self.crop_meta_data = crop_meta_data

        # variables
        self.echo_info = []
        self.echo = []
        self.triangular_mask = []
        self.left_v = []
        self.right_v = []
        self.matrix_3d = []

    def process_dataset(self, data_set):
        """
		Preprocess a DataCollection object
		
		Parameters
		----------
		data_set: DataCollection object
			DataCollection object to be preprocessed
		"""
        self.data_folder = os.path.basename(data_set.data_folder)
        echos = data_set.echos
        n_echos = len(echos)
        for j, ech in enumerate(echos):
            if self.verbose:
                print("echo: ", j + 1, "/", n_echos)
            self.extract_echo(ech)
            ech.set_3d_array(self.matrix_3d)
            if self.save_frames:
                if self.verbose:
                    print("Saving frames...")
                ech.save_frames()
            if self.save_pickle_echo:
                if self.verbose:
                    print("Saving pickle...")
                ech.save_pickle()

    def extract_echo(self, echo):
        """
		Open and preprocess an Echo object
		
		Parameters
		----------
		echo: Echo object
			echo that is preprocessed (cropped and resized)

		Returns
		----------
		matrix_3d: 3D float array
			3d matrix of the echo video cropped and resized
		"""
        self.echo = echo
        self.echo_info = self.echo.get_info()
        if self.verbose:
            print("Working on echo at:", self.echo_info["file_path"])

        if self.crop_meta_data:
            self.generate_foreground_mask(echo)
            self.create_triangle_mask()
            mask, top_left, bottom_right = self.triangular_mask, self.left_v, self.right_v
        else:
            mask = np.ones(shape=(self.echo.height, self.echo.height))
            top_left = (0, 0)
            bottom_right = (self.echo.height, self.echo.height)

        if self.resize:
            new_height = self.side_length
            new_width = self.side_length
        else:
            new_height = bottom_right[0] - top_left[0]
            new_width = bottom_right[1] - top_left[1]

        file_path = self.echo_info["file_path"]
        video_as_3d_array = np.zeros([new_height, new_width])

        if self.save_cropped_video:
            out_file_dir = os.path.join(DIR_PATH, '..', 'out',
                                        self.data_folder, 'videos', 'cropped')
            out_file_path = os.path.join(out_file_dir,
                                         self.echo_info["name"]) + '.avi'
            os.makedirs(out_file_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(out_file_path, fourcc, self.echo_info["fps"],
                                  (new_width, new_height), isColor=False)

        cap = cv2.VideoCapture(file_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.multiply(frame, mask)
                frame = frame[top_left[0]:bottom_right[0],
                        top_left[1]:bottom_right[1]]
                if self.resize:
                    frame = resize_frame(frame, self.side_length)
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                video_as_3d_array = np.dstack((video_as_3d_array, frame))
                frame = np.uint8(frame)
                if self.save_cropped_video:
                    out.write(frame)
            else:
                break
        cap.release()

        if self.save_cropped_video:
            out.release()

        self.matrix_3d = video_as_3d_array[:, :, 1:]

        # crop box and segmentation masks accordingly

        if self.echo.labels['box'] is not None:
            box = np.multiply(self.echo.labels['box'], mask)
            box = box[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            box = np.nan_to_num(box).astype(np.uint8)

            self.echo.labels['box'] = np.asarray(box, dtype=np.bool)

            if self.resize:
                box_rs = resize_frame(box, self.side_length)
                box_rs = np.nan_to_num(box_rs)
                box_rs = (box_rs > 0.5)
                self.echo.labels['box'] = box_rs

        if self.echo.labels['masks']:

            for i, seg_mask in enumerate(self.echo.labels['masks']):
                frame = list(seg_mask.keys())[0]
                seg = list(seg_mask.values())[0]

                seg = np.multiply(seg, mask)
                seg = seg[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

                if self.resize:
                    seg = resize_frame(seg, self.side_length)

                seg = np.nan_to_num(seg)
                seg = (seg > 0)
                self.echo.labels['masks'][i] = {frame: seg}

    def generate_foreground_mask(self, echo):
        """
		Genereate a mask of the moxing pixel separated from the background

		It uses the MOG (mixture of gaussian) model to separate foreground
		from background (1 for foreground, 0 for background)
		
		Parameters
		----------
		echo: Echo object
			echo that s preprocesse 
		Returns
		----------
		matrix_3d: 3D float array
			3d matrix of the echo video cropped and resized
		"""
        cap = cv2.VideoCapture(self.echo.file_path)
        fgbg = cv2.createBackgroundSubtractorMOG2()

        self.foreground = np.zeros(
            [self.echo_info["height"], self.echo_info["width"]])
        self.mask = np.zeros(
            [self.echo_info["height"], self.echo_info["width"]])

        if self.verbose:
            print("Generating foreground and mask..")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                fgmask = fgbg.apply(frame)
                self.foreground = self.foreground + fgmask
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.mask = self.mask + frame
            else:
                break

        cap.release()

        self.foreground[self.foreground > 0] = 1
        self.foreground = crop_outer_part(self.foreground, self.echo_info)

        count = Counter(self.mask.flatten().tolist())
        most_common = (count.most_common()[0])[0]

        if most_common != 0:
            self.mask[self.mask == most_common] = 0

        self.mask[self.mask > 0] = 1

        self.foreground = morphological_transformation_foreground(
            self.foreground)
        self.mask = morphological_transformation_mask(self.mask)
        if self.save_foreground:
            save_picture(
                os.path.join(DIR_PATH, '..', 'out', self.data_folder, 'images',
                             'masks', 'MOG'),
                self.echo_info["name"], self.foreground)

        if self.save_overlay:
            save_picture(
                os.path.join(DIR_PATH, '..', 'out', self.data_folder, 'images',
                             'overlay'),
                self.echo_info["name"], self.mask)

    def create_triangle_mask(self):
        """
		Genereate a trinagular mask of the video.

		Given the foreground mask and the overlay mask generates a triangular
		mask using simple heuristics
		
		"""

        if self.verbose:
            print("Creating triangle mask...")

        out_file_dir = os.path.join(DIR_PATH, '..', 'out', self.data_folder,
                                    'images', 'cropped')

        self.triangular_mask, self.left_v, self.right_v = opt_triangular_points(
            self.mask, self.foreground,
            self.validate_cropping, out_file_dir,
            self.echo_info["file_path"],
            self.echo_info['name'])

    def get_video_frame(self, frame_no, show=False):
        cap = cv2.VideoCapture(self.echo.get_info())

        current_frame_no = 0
        while True:
            ret, frame = cap.read()  # Read the frame
            if not ret:
                cap.release()
                raise ValueError('The specific frame_no is not valid!')
            if current_frame_no == frame_no:
                if show and ret:
                    time.sleep(1)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                cap.release()
                return frame
            current_frame_no += 1

    def show_video(self, file_path):
        cap = cv2.VideoCapture(file_path)

        time.sleep(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
