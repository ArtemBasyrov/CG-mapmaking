import numpy as np
import h5py

class TOD(object):
    """a class to generate, save and load TOD"""

    def __init__(self, fname):
        super(TOD, self).__init__()
        self.tod_fname = fname
        self.tod = None
        self.NTOD = None


    def add_tod(self, tod):
        ''' add tod to the class and calculate NTOD '''
        self.tod = tod
        self.NTOD = len(tod)


    def read_tod_from_file(self):
        ''' read TOD from .h5 file '''
        with h5py.File(self.tod_fname, 'r') as file:
            self.tod = file['tod'][()]
            self.NTOD = len(self.tod)
            file.close()

        print(self.tod_fname, 'read in!')


    def save_tod_to_file(self):
        ''' save TOD to .h5 file '''
        h5file = h5py.File(self.tod_fname, 'w')
        h5file.create_dataset('tod', data=self.tod)
        h5file.close()

        self.tod = None # to empty memory

        print(self.tod_fname, 'has been created!')

        
