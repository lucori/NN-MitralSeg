import os
import cv2
import numpy as np
import shutil
import scipy.misc
import glob
import pickle5 as pickle
from video_processing import EchoProcess
import pickle
import json
from PIL import Image

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class DataMaster:
    def __init__(self, conf):
        self.conf = conf

        self.df = os.path.join(DIR_PATH, self.conf['Load_Save']['raw_data_folder'])
        classes = json.loads(self.conf.get('Load_Save', 'classes'))
        views = json.loads(self.conf.get('Load_Save', 'views'))
        verbose = self.conf['Video_Processing']['verbose']
        self.dt = DataCollection(self.df, classes, views, verbose)

    def load(self):
        if self.conf.getboolean('Load_Save', 'load_dataset'):
            self.dt = self.dt.load_pickle(
                self.conf['Load_Save']['pickle_name'])
        if self.conf.getboolean('Load_Save', 'load_echos_from_pickle'):
            self.dt = self.dt.load_pickle_multiple_file()
        if not self.dt.populated:
            self.dt.populate()
            processor = EchoProcess(**self.conf.get_par_video_processing())
            processor.process_dataset(self.dt)
            if self.conf.getboolean('Load_Save', 'save_dataset'):
                self.dt.save_pickle('test_dataset')


class DataCollection:
    """
	Class that contains and manage a collection of echo videos

	Parameters

	data_folder: string
		absolut path of the data folder that you want to use
	verbose: boolean
		set the amount of messages that you want to get
	
	Variables: 
		file_path_names: list 
			list the file path name of all the ehcos memorized
		echos; list 
			list of echo objects
		echos_info: list 
			list of info of the echos
		classes: array of two string 
			representing the possible diagnosis of echos
		views: array of two string 
			representing the possible views of echos
		populated: boolean 
			is set to True if the data collection has been loaded
	"""

    def __init__(self, verbose=False):
        self.data_folder = []
        self.file_path_names = []
        self.echos = []
        self.echos_infos = []
        self.classes = []
        self.populated = False
        self.verbose = verbose
        self.files_saved = []

    def __init__(self, df, classes, views, verbose=False):
        self.file_path_names = []
        self.echos = []
        self.echos_infos = []
        self.data_folder = df
        self.populated = False
        self.verbose = verbose
        self.classes = classes
        self.views = views
        self.files_saved = []

    def populate(self):
        """
		Loads all the echo files in the folder
		"""
        if os.path.isdir(self.data_folder):
            for dirpath, dirnames, filenames in os.walk(self.data_folder):
                for filename in [f for f in filenames if
                                 (f.endswith(".avi") or f.endswith(
                                     ".wmv")) and not f.startswith(".")]:
                    file_path = os.path.join(dirpath, filename)
                    statinfo = os.stat(file_path)
                    if statinfo.st_size != 0:
                        ec = Echo(file_path, self.data_folder, self.classes,
                                  self.views)
                        if not ec.exclude:
                            self.file_path_names.append(file_path)
                            self.echos.append(ec)
                            self.echos_infos.append(ec.get_info())
                    else:
                        print("File:", filename, "is zero bytes!")
            self.populated = True
        else:
            raise FileNotFoundError("The specified folder does not exist")

    def populate_dictionary(self, view):
        if os.path.isdir(self.data_folder):
            for dirpath, dirnames, filenames in os.walk(self.data_folder):
                for filename in [f for f in filenames if
                                 (f.endswith(".avi") or f.endswith(
                                     ".wmv")) and not f.startswith(".")]:
                    file_path = os.path.join(dirpath, filename)
                    statinfo = os.stat(file_path)
                    ec = Echo(file_path, self.data_folder, self.classes,
                              self.views)
                    if statinfo.st_size != 0 and not ec.exclude and ec.chamber_view in view:
                        self.files_saved.append(file_path)
                    elif statinfo.st_size != 0:
                        print("File:", filename, "is zero bytes!")
                    elif not ec.exclude:
                        print('View of the file is not considered.')
        else:
            raise FileNotFoundError("The specified folder does not exist")

    def __str__(self):
        print(self.echos_infos)

    def get_target(self, view):
        """
        returns the targets of all the echo
        Returns
        -------
        numpy array
        	array of boolean describing the target of each ech
        """
        y = np.array(
            [ech.diagnosis for ech in self.echos if ech.chamber_view == view])
        y = np.reshape(y, (-1, 1))

        return y

    def get_3dmatrix(self, view):
        """
        returns the list of 3d array of all the echo video
        Returns
        -------
        list of 3d arrays
        	list of 3d arrays that correspond to the echo video.
        	Note that is not given as a 4D array since the lenght of videos is not uniform.
        """
        return [ech.matrix3d for ech in self.echos if ech.chamber_view == view]

    def get_x_y(self, view):
        """
		returns the targets of all the echos

		Returns
		-------
		numpy array
			array of boolean describing the target of each echo

		"""
        return self.get_3dmatrix(view), self.get_target(view)

    def save_pickle(self, name):
        """
		Saves the data collection into a pickle file
		
		Parameters
		----------
		name : string
			name of the pickle file in which to save in
		"""
        out_file_dir = os.path.join(DIR_PATH, '..', 'out',
                                    os.path.basename(self.data_folder),
                                    'pickle')
        os.makedirs(out_file_dir, exist_ok=True)
        with open(os.path.join(out_file_dir, name + '.pkl'), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, name):
        """
		Loads the data collection from a pickle file
		
		Parameters
		----------
		name : string
			name of the pickle file that is used to load the data collection

		Returns
		----------
		DataCollection object
			data collection that was saved in the pickle file
		"""
        out_file_dir = os.path.join(DIR_PATH, '..', 'out',
                                    os.path.basename(self.data_folder),
                                    'pickle')
        os.makedirs(out_file_dir, exist_ok=True)
        out_file_name = os.path.join(out_file_dir, name + '.pkl')
        with open(out_file_name, 'rb') as output:
            try:
                if self.verbose:
                    print("loading from pickle file ", out_file_name)
                self = pickle.load(output)
                self.populated = True
                output.close()
            except EOFError:
                print("not found")
                pass
        return self

    def load_pickle_multiple_file(self):
        """
		Loads the data collection from a pickle file per echo video

		Returns
		----------
		DataCollection object
			data collection that was saved in the pickle file


		"""

        out_file_dir = os.path.join(DIR_PATH, '..', 'out',
                                    os.path.basename(self.data_folder),
                                    'pickle')
        if os.path.isdir(out_file_dir):
            for dirpath, dirnames, filenames in os.walk(out_file_dir):
                for filename in [f for f in filenames if f.endswith(".pkl")]:
                    print(filename)
                    file_path = os.path.join(dirpath, filename)
                    statinfo = os.stat(file_path)
                    if statinfo.st_size != 0:
                        with open(file_path, 'rb') as output:
                            try:
                                if self.verbose:
                                    print("loading from pickle file ",
                                          filename)
                                ec = pickle.load(output)
                                output.close()
                            except EOFError:
                                print("not found")
                                pass
                        self.echos.append(ec)
                        self.echos_infos.append(ec.get_info())
                        self.file_path_names.append(ec.file_path)
                    else:
                        print("File:", filename, "is zero bytes!")
            self.populated = True
        else:
            raise FileNotFoundError("The specified folder does not exist")
        return self


class Echo:
    """
    class that contains and manage an echo videos
    Parameters
    data_folder : string
    	absolut path of the data folder that you want to use
    file_path: string
    	absolut path of the echo file
    """

    def __init__(self, file_path, data_folder, classes, views):
        self.file_path = file_path
        self.data_folder = data_folder
        self.pickle_folder = None
        self.echo_name = None
        self.diagnosis = None
        self.chamber_view = None
        self.hospital = None
        self.height = None
        self.width = None
        self.no_frames = None
        self.fps = None
        self.duration = None
        self.matrix3d = []
        self.corrupt = None
        self.classes = classes
        self.views = views
        self.exclude = False
        self.labels = {'masks': [], 'box': None}
        self.name_data = "new_data"
        self.open()

    def open(self):
        """
        Opens the echo video according to the file path and set the meta information about it
        """
        cap = cv2.VideoCapture(self.file_path)
        file_name = os.path.splitext(os.path.basename(self.file_path))[0].upper()

        self.name_data = os.path.basename(os.path.dirname(os.path.dirname(self.file_path)))

        if not self.classes:
            self.diagnosis = "unknown"
            self.echo_name = file_name
        elif self.classes[0] in self.file_path:
            self.echo_name = self.classes[0] + file_name
            self.diagnosis = 1
        elif self.classes[1] in self.file_path:
            self.echo_name = self.classes[1] + file_name
            self.diagnosis = 0
        else:
            print("Unknown diagnosis in file:", self.file_path)
            self.diagnosis = "diagnosis_unknown"
            self.exclude = True

        self.width = int(cap.get(3))
        self.height = int(cap.get(4))
        self.fps = cap.get(5)
        if self.fps == 0:
            print("Not a valid video file!")
            self.corrupt = 1
            self.exclude = True
        else:
            self.corrupt = 0

        self.no_frames = int(cap.get(7))
        if not self.corrupt:
            self.duration = self.no_frames / self.fps
        else:
            self.duration = 'Nan'
            self.exclude = True

        cap.release()
        if not self.views:
            # no view provided
            self.chamber_view = "view_unknown"

        elif len(self.views) > 1:
            for view in self.views:
                if view.lower() in self.echo_name.lower():
                    self.chamber_view = view

            if not self.chamber_view:
                print("Unknown chamber-view in file:", self.file_path)
                self.chamber_view = -1
                self.exclude = True

        # store segmentation masks and box
        directory = os.path.dirname(self.file_path)
        masks = [f for f in
                 glob.glob(directory + "/**mask.png", recursive=True)]

        box = [f for f in glob.glob(directory + "/box.jpg", recursive=True)]

        if len(masks) == 0 or len(box) == 0:
            print(self.echo_name)
            print("No labels provided. Add ####_mask.png and/or box.jpg")
            return

        if len(masks) != 3:
            print("This folder ({}) does not contain 3 maskes.".format(file_name))

        img = np.asarray(Image.open(box[0]))
        img = (1 * (~((img[..., 0] >= 245) * (img[..., 1] >= 245) * (
                img[..., 2] >= 245)))).astype(np.uint8)

        self.labels['box'] = img

        for f in sorted(masks):

            # valve ground-truth as png as 4 color dimension and last is masking
            img = np.asarray(Image.open(f))
            frame = int(os.path.basename(f).split('_mask')[0])
            self.labels['masks'].append(
                {str(frame): (1 * (img[..., -1] == 255)).astype(np.uint8)})

    def get_info(self):
        """
        Returns a dictionary with all the metinformation of the ech
        Returns
        ----------
        echo_dict dict
        	dictionary with all the metainformation of the echo
        """
        echo_dict = {"name": self.echo_name,
                     "file_path": self.file_path,
                     "diagnosis": self.diagnosis,
                     "chamber_view": self.chamber_view,
                     "height": self.height,
                     "width": self.width,
                     "no_frames": self.no_frames,
                     "fps": self.fps,
                     "duration": self.duration,
                     "corrupt": self.corrupt}
        return echo_dict

    def set_3d_array(self, matrix):
        """
        Set the 3d array video of the echo

        Parameter
        matrix: 3D array
        	3D array that represents the echo video
        """
        self.matrix3d = matrix

    def save_frames(self):
        """
        Saves the provided 3D array as frames (.jpg)
        """
        echo_name = (os.path.splitext(self.echo_name))[0].upper()
        n_frames = self.matrix3d.shape[2]
        out_path = os.path.join(DIR_PATH, '..', 'out',
                                os.path.basename(self.data_folder), 'frames',
                                self.chamber_view,
                                echo_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        shutil.rmtree(out_path, ignore_errors=True)
        os.makedirs(out_path, exist_ok=True)

        ndarray = cv2.normalize(self.matrix3d, None, 0, 255, cv2.NORM_MINMAX)
        ndarray = ndarray.astype(np.uint8)

        for f in range(0, n_frames):
            frame = ndarray[:, :, f]
            Image.fromarray(frame).save(
                os.path.join(out_path, str(f).zfill(3) + ".jpg"))

    def save_pickle(self):
        """
        Saves the provided 3D array as a pickle file
        """
        echo_name = (os.path.splitext(self.echo_name))[0].upper()
        out_path = os.path.join(DIR_PATH, '..', 'data', 'in', 'processed', self.name_data)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(os.path.join(out_path, echo_name + '.pkl'), 'wb') as f:
            pickle.dump(self, f, protocol=-1)

        self.pickle_folder = out_path
