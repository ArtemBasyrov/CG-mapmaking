import numpy as np
from multiprocessing import Pool
import os
import h5py
from NPIPE_transfer_function import LFER4

import time

class trans_operator(object):
    ''' performs the transfer function operations '''

    def __init__(self, L=524288):
        super(trans_operator, self).__init__()
        self.L = L
        self.TF_path = 'path/to/precalculated/transfer/function/in/h5/format'
        self.transfer_func = None
        self.TF_form = 1

        # tod parameters
        self.tod_fname = None
        self.NTOD = None
        self.n = None


    def save_TF_to_h5(self, filename=None):
        ''' save the transfer function to the .h5 file '''

        if filename is None:
            filename = self.TF_path

        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('data', data=self.transfer_func)
        h5file.close()
        print(filename, 'has been created!')


    def read_TF_from_h5(self, filename=None):
        ''' read the transfer function from the .h5 file '''

        if filename is None:
            filename = self.TF_path

        with h5py.File(filename, 'r') as h5file:
            self.transfer_func = h5file['data'][()]
            h5file.close()


        print(filename, 'read in!')


    def calculate_TF(self, L):
        ''' calculate the given length of TF as LFER4 '''

        f_samp = 180.3737 # [Hz]
        freq = np.fft.fftfreq(L, d=1/f_samp)
        TF_npipe = LFER4(freq)

        return TF_npipe


    def calculate_filter_function(self, L):
        ''' calculate the filter function applied in the inverse oper '''

        f_samp = 180.3737 # [Hz]
        freq = np.fft.fftfreq(L, d=1/f_samp)

        f_Gauss = 65 # [Hz]
        f_c = 80 # [Hz]
        k = 0.9 
        f_max = f_c + k*(f_samp/2 - f_c)

        filt_func = np.ones(len(freq))
        f_test = np.abs(freq)

        filt_func[f_test >= f_max] = 0
        sel = (f_test > f_c) & (f_test < f_max)
        filt_func[sel] = np.cos(np.pi/2 * (f_test[sel]-f_c)/(f_max - f_c))**2

        return filt_func*np.exp(-0.5*(freq/f_Gauss)**2)


    def TF_oper(self, tod_fname):
        ''' calculate the application of a TF to an array x '''

        x_tran = self._TF_calculator(tod_fname, TF_form=1)

        # find parameters
        L_left = self.NTOD%self.L
        print('there are {0} elements left'.format(L_left))

        # calculate the remaining part of x
        with h5py.File(self.tod_fname, 'r') as file:
            x = file['tod'][self.n*self.L:]
            file.close()

        TF_left = self.calculate_TF(L=L_left)
        F_x = np.fft.fft(x, norm='ortho')
        F_x = F_x*TF_left
        x_tran[self.n*self.L:] = np.fft.ifft(F_x, norm='ortho').real

        del TF_left
        del F_x
        del x

        return x_tran


    def TF_transpose_oper(self, x):
        ''' calculate the application of a transpose of a TF to an array x '''         
        x_tran = self._TF_calculator(x, TF_form='T')

        # find parameters
        L_left = self.NTOD%self.L
        print('there are {0} elements left'.format(L_left))

        # calculate the remaining part of x
        with h5py.File(self.tod_fname, 'r') as file:
            x = file['tod'][self.n*self.L:]
            file.close()

        TF_left = self.calculate_TF(L=L_left)
        F_x = np.fft.fft(x, norm='ortho')
        F_x = F_x*np.flip(TF_left)
        x_tran[self.n*self.L:] = np.fft.ifft(F_x, norm='ortho').real

        del TF_left
        del F_x
        del x

        return x_tran


    def TF_inverse_oper(self, x):
        ''' calculate the application of an inverse of a TF to an array x '''
        x_tran = self._TF_calculator(x, TF_form=-1)

        # find parameters
        L_left = self.NTOD%self.L
        print('there are {0} elements left'.format(L_left))

        # calculate the remaining part of x
        with h5py.File(self.tod_fname, 'r') as file:
            x = file['tod'][self.n*self.L:]
            file.close()

        TF_left = self.calculate_TF(L=L_left)
        F_x = np.fft.fft(x, norm='ortho')
        filt_func = self.calculate_filter_function(L=L_left)
        F_x = F_x/TF_left*filt_func
        x_tran[self.n*self.L:] = np.fft.ifft(F_x, norm='ortho').real

        del TF_left
        del F_x
        del x

        return x_tran


    def TF_double_oper(self, x, sigma_path):
        # calculate the application of a TF and its transpose
        # to an array x under the assumption of N^-1 being diagonal
        # it will require to do a bit more than other functions

        # load or calculate transfer function
        if self.transfer_func is None or len(self.transfer_func) != self.L:
            if os.path.exists(self.TF_path):
                self.read_TF_from_h5(filename=self.TF_path)
            else:
                self.calculate_TF(L=self.L)


        # sigma_path save relevant arrays
        self.tod_fname = x
        self.sigma_path = sigma_path

        # find parameters
        with h5py.File(self.tod_fname, 'r') as file:
            tod = file['tod']

            self.NTOD = len(tod)
            self.n = int(self.NTOD/self.L)
            breakpoints = np.arange(2*self.n)*int(self.L/2)

            file.close()

        if breakpoints[-1] + self.L > self.NTOD:
            breakpoints = breakpoints[:-1]

        # the first step. Do (T^T * N^-1 * T) * x        
        start = time.time()
        pool = Pool(processes=64)
        res = pool.map(self._fourrier_proc1, breakpoints)
        pool.close()
        pool.join()
        print('The parallel part finished in', time.time() - start, 's')

        x_tran = np.zeros(self.NTOD)
        breakpoints[1:] += int(self.L/4)
        breakpoints = np.append(breakpoints, breakpoints[-1] + int(3/4*self.L))
        for i in range(len(breakpoints)-1):
            temp = res.pop(0)
            if i == 0:
                temp = temp[:int(3*self.L/4)]
            elif i == len(breakpoints)-2:
                temp = temp[int(self.L/4):]
            else:
                temp = temp[int(self.L/4):int(3*self.L/4)]
            x_tran[breakpoints[i]:breakpoints[i+1]] = temp

        # find parameters
        L_left = self.NTOD%self.L

        # calculate the remaining part of x
        with h5py.File(self.tod_fname, 'r') as file:
            x = file['tod'][self.n*self.L:]
            file.close()

        with h5py.File(self.sigma_path, 'r') as file:
            sigma = file['tod'][self.n*self.L:]
            file.close()

        last_el = self.n*self.L
        L_list = [2**18, 2**17, 2**16, 2**15, 2**14]
        for L_new in L_list:
            if L_left > L_new:
                self.read_TF_from_h5(filename='path/to/precalculated/TF_array_{0}_143.h5'.format(L_new))
                x_tran[last_el:last_el+L_new] = self._fft_proc_for_double_TF(x[:L_new], sigma[:L_new])
                x = x[L_new:]
                sigma = sigma[L_new:]
                L_left -= L_new
                last_el += L_new

        if L_left > 0:
            TF_left = self.calculate_TF(L=L_left)
            self.transfer_func = TF_left
            x_tran[last_el:] = self._fft_proc_for_double_TF(x, sigma)

        end = time.time()
        print('TF_double_oper finished with time:', end-start, 's')


        return x_tran 


    def _TF_calculator(self, tod_fname, TF_form):
        ''' an internal universal calculator for TF operations '''

        # load or calculate transfer function
        if self.transfer_func is None or len(self.transfer_func) != self.L:
            if os.path.exists(self.TF_path):
                self.read_TF_from_h5(filename=self.TF_path)
            else:
                self.calculate_TF(L=self.L)

        # save relevant arrays
        self.TF_form = TF_form
        self.tod_fname = tod_fname
        if self.TF_form == 'T': # save a bunch of operations
            self.transfer_func = np.conj(self.transfer_func)


        # find parameters
        with h5py.File(self.tod_fname, 'r') as file:
            tod = file['tod']
       
            self.NTOD = len(tod) 
            self.n = int(self.NTOD/self.L)
            breakpoints = np.arange(2*self.n)*int(self.L/2)

            file.close()
        #print('there are {0} segments with the total length of {1}'.format(self.n, self.n*self.L))

        if breakpoints[-1] + self.L > self.NTOD:
            breakpoints = breakpoints[:-1]

        start = time.time()
        pool = Pool(processes=64)
        res = pool.map(self._fourrier_proc, breakpoints)
        pool.close()
        pool.join()
        end = time.time()
        print('TF_calculator finished with time:', end-start, 's')
       
        x_tran = np.zeros(self.NTOD)
        breakpoints[1:] += int(self.L/4)
        breakpoints = np.append(breakpoints, breakpoints[-1] + int(3/4*self.L))
        for i in range(len(breakpoints)-1):
            temp = res.pop(0)
            if i == 0:
                temp = temp[:int(3*self.L/4)]
            elif i == len(breakpoints)-2:
                temp = temp[int(self.L/4):]
            else:
                temp = temp[int(self.L/4):int(3*self.L/4)]
            x_tran[breakpoints[i]:breakpoints[i+1]] = temp

        if self.TF_form == 'T': # return the array to how it was
            self.transfer_func = np.conj(self.transfer_func)

        return x_tran

    
    def _fourrier_proc(self, k):
        with h5py.File(self.tod_fname, 'r') as file:
            temp = file['tod'][k:k+self.L]
            file.close()

        F_x = np.fft.fft(temp, norm='ortho')

        # choose what to do: 1 = direct TF, -1 = inverse, T = transpose
        if self.TF_form == 1:
            F_x = F_x*self.transfer_func
        elif self.TF_form == -1:
            filt_func = self.calculate_filter_function(L=self.L)
            F_x = F_x/self.transfer_func*filt_func
        elif self.TF_form == 'T':
            F_x = F_x * self.transfer_func # already fliped before

        return np.fft.ifft(F_x, norm='ortho').real


    def _fourrier_proc1(self, k):
        with h5py.File(self.tod_fname, 'r') as file:
            temp = file['tod'][k:k+self.L]
            file.close()

        #sigma_path = '/mn/stornext/d5/data/artemba/other/sigma_noise_3e+2.h5'
        with h5py.File(self.sigma_path, 'r') as file:
            sigma = file['tod'][k:k+self.L]
            file.close()

        
        temp = self._fft_proc_for_double_TF(temp, sigma)
        

        return temp

    def _fft_proc_for_double_TF(self, x, sigma):
        F_x = np.fft.fft(x, norm='ortho')
        F_x = F_x*self.transfer_func
        temp = np.fft.ifft(F_x, norm='ortho').real

        temp = temp/sigma/sigma    

        transpose_TF = np.conj(self.transfer_func)
        F_x = np.fft.fft(temp, norm='ortho')
        F_x = F_x*transpose_TF
        temp = np.fft.ifft(F_x, norm='ortho').real

        return temp


