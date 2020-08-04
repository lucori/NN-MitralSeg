from configparser import ConfigParser
import os
from shutil import copyfile
import json
import numpy as np


class ConfigParserEcho(ConfigParser):

    def __init__(self):
        ConfigParser.__init__(self)

    def get_par_load_save(self):
        classes = json.loads(self.get('Load_Save', 'classes'))
        views = json.loads(self.get('Load_Save', 'views'))
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 self['Load_Save']['data_folder'])

        return data_path, classes, views

    def get_par_video_processing(self):
        param_dict = {
            "save_foreground": self.getboolean('Video_Processing',
                                               'save_foreground'),
            "save_overlay": self.getboolean('Video_Processing',
                                            'save_overlay'),
            "validate_cropping": self.getboolean('Video_Processing',
                                                 'validate_cropping'),
            "save_cropped_video": self.getboolean('Video_Processing',
                                                  'save_cropped_video'),
            "resize": self.getboolean('Video_Processing', 'resize'),
            "save_frames": self.getboolean('Video_Processing', 'save_frames'),
            "save_pickle_echo": self.getboolean('Video_Processing',
                                                'save_pickle_echo'),
            "verbose": self.getboolean('Video_Processing', 'verbose'),
            "side_length": self.getint('Video_Processing', 'side_length'),
            "crop_meta_data": self.getboolean('Video_Processing',
                                              'crop_meta_data'),
        }

        return param_dict

    def get_par_histogram(self):
        n_blocks = np.array(json.loads(self.get('Histogram', 'n_blocks')))
        n_bins = self.getint('Histogram', 'n_bins')
        use_density = self.getboolean('Histogram', 'use_density')
        verbose = self.getboolean('Histogram', 'verbose')

        return n_blocks, n_bins, use_density, verbose

    def copy_conf(self, file_name, time):
        out_conf_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'out',
            'used_configuration')
        os.makedirs(out_conf_dir, exist_ok=True)
        copyfile(file_name, os.path.join(out_conf_dir, time + '.ini'))
