import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, identity, diags, load_npz
from scipy.sparse.linalg import cg, LinearOperator
import scipy
import h5py
from transfer_function_operator import trans_operator
from tod_mod import TOD
import sys

import time

class Driver(object):
    """drives the whole routine of reading and calculationg things"""

    def __init__(self, NSIDE: int):
        super(Driver, self).__init__()
        self.NSIDE = NSIDE
        self.NPIX = hp.pixelfunc.nside2npix(NSIDE)
        

        # pathes and arrays for tod relevant stuff
        self.tod_path = 'path/to/h5/tod/file'
        self.NTOD = None
        self.sigma_path = 'path/to/h5/sigma/white/noise/file'
        self.pix_path = 'path/to/h5/pointing/file'
        self.pix = None


    def read_pointing(self):
        ''' read pointing in pixels into memory '''

        # this function is very heavy and shouldn't be used in general
        # instead use create_pointing_matrix()

        with h5py.File(self.pix_path, 'r') as file:
            pix = file['pix'][()]
            file.close()

        self.pix = pix.astype(int)
        pix = None
        self.NTOD = len(self.pix)
        
        print(self.pix_path, 'read in! NTOD is', self.NTOD)


    def TF_operator(self, fname=None): 
        ''' apply transfer function once '''

        if fname is None:
            fname = self.tod_path

        L = 524288 # 2**19
        #L = 262144 # 2**18

        # call the operator class
        TF_oper = trans_operator(L)
        x_tran = TF_oper.TF_oper(fname) 
        TF_oper = None

        return x_tran


    def transpose_TF_operator(self, fname=None):
        ''' apply the transpose of the TF '''

        if fname is None:
            fname = self.tod_path

        L = 524288 # 2**19
        #L = 262144 # 2**18

        # call the operator class
        TF_oper = trans_operator(L)
        x_tran = TF_oper.TF_transpose_oper(fname)
        TF_oper = None

        return x_tran


    def double_TF_operator(self, fname):
        ''' apply TF twice '''

        L = 524288 # 2**19
        #L = 262144 # 2**18

        # call the operator class
        TF_oper = trans_operator(L)
        x_tran = TF_oper.TF_double_oper(fname, self.sigma_path)
        TF_oper = None

        return x_tran


    def inverse_TF_operator(self, fname=None):
        ''' apply the inverse of the TF '''

        if fname is None:
            fname = self.tod_path

        L = 524288 # 2**19
        #L = 262144 # 2**18

        # call the operator class
        TF_oper = trans_operator(L)
        x_tran = TF_oper.TF_inverse_oper(fname)
        TF_oper = None

        return x_tran


    def generate_tod(self):
        
        fname_woTF = 'path/to/save/tod/without/transfer/function'

        '''  
        # generating a polar beam (from map to tod)

        m_polar = np.zeros(self.NPIX)
        #m_polar[0:4] = np.ones(4)*100
        m_polar[37677201] = 100 # south pole in Ecliptic coordinates
        m_polar = hp.sphtfunc.smoothing(m_polar, fwhm=5.0/60*np.pi/180)
        '''

        ''' 
        # generating a cmb map from LCDM Planck 2015

        fname_LCDM = 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'
        LCDM = np.genfromtxt(fname_LCDM, skip_header=True)
        l = LCDM[:,0]
        cl = LCDM[:,1]
        factor = l*(l+1)/2/np.pi
        cl = cl/factor

        mp_cmb = hp.sphtfunc.synfast(cl, nside=self.NSIDE) # in microK
        mp_cmb = hp.sphtfunc.smoothing(mp_cmb, fwhm=7.2/60*np.pi/180)

        tod_obj = TOD(fname=fname_woTF)
        tod_obj.add_tod(self.P_matrix.dot(mp_cmb))
        tod_obj.save_tod_to_file()
        '''
        noise_obj = TOD(fname=self.sigma_path)
        noise_obj.read_tod_from_file()

        fname_wTF = self.tod_path
        tod_obj = TOD(fname=fname_wTF)
        x_tran = self.TF_operator(fname_woTF)
        tod_obj.add_tod(x_tran + np.random.normal(scale=noise_obj.tod))
        tod_obj.save_tod_to_file()
        noise_obj = None
        print('TOD has been generated!')


    def create_pointing_matrix(self, reduced=False):
        ''' create a pointing matrix as sparse matrix '''

        start = time.time()

        if self.pix is None:
            with h5py.File(self.pix_path, 'r') as file:
                pix = file['pix']

                if self.NTOD is None: 
                    self.NTOD = len(pix)

                # reduce the size of pointing matrix by not including 
                # not-observed pixels
                if reduced:
                    obs_pix, indeces = np.unique(pix, return_inverse=True)
                    self.NPIX_reduced = len(obs_pix)
                    self.pix_obs = obs_pix.astype(int)

                    #self.P_matrix = csr_matrix((np.ones(self.NTOD), (np.arange(self.NTOD), indeces)), shape=(self.NTOD, self.NPIX_reduced))
                    self.P_matrix = load_npz('path/to/npz/matrix')

                else:
                    #self.P_matrix = csr_matrix((np.ones(self.NTOD), (np.arange(self.NTOD), pix)), shape=(self.NTOD, self.NPIX))
                    self.P_matrix = load_npz('path/to/npz/matrix')
                file.close()

             

        else:
            self.P_matrix = csr_matrix((np.ones(self.NTOD), (np.arange(self.NTOD), self.pix)), shape=(self.NTOD, self.NPIX))

        self.pix = None
        print('Created pointing matrix')
        print('Time to create pointing matrix', time.time()-start, 's')


    def create_inverse_noise_matrix(self):
        ''' create an inverse noise matrix as sparce matrix '''

        start = time.time()
        
        scale = TOD(fname=self.sigma_path)
        scale.read_tod_from_file()

        self.N_inv = diags(1/scale.tod/scale.tod)
        scale = None
        print('Time to create N_inv:', time.time()-start, 's')


    def calculate_B_matrix(self):
        ''' calculate B matrix in Ax=B '''
                
        # read tod from file
        tod_obj = TOD(self.tod_path)
        tod_obj.read_tod_from_file() 

        scale = TOD(fname=self.sigma_path)
        scale.read_tod_from_file()
        tod = tod_obj.tod/scale.tod/scale.tod 
        scale = None
        
        tod_Bprecomp_path = 'tod_polar_Bprecomp.h5'
        tod_obj = TOD(tod_Bprecomp_path)
        tod_obj.add_tod(tod)
        tod_obj.save_tod_to_file()
        
        tod = self.transpose_TF_operator(tod_Bprecomp_path)
        self.B = self.P_matrix.T.dot(tod)
        return self.B


    def load_B_matrix(self, fname):
        ''' load B matrix from .h5 file '''
        tod_obj = TOD(fname)
        tod_obj.read_tod_from_file()
        self.B = tod_obj.tod.copy()
        tod_obj = None
        return self.B


    def A_dot_x(self, x):
        # operator which returns the result of the dot product
        # of matrix A = (P^T x T^T x N^-1 x T x P) [size:(NPIX,NPIX)] and x [size NPIX]

        temp = self.P_matrix.dot(x) # P x m
        
        # save to file and unload memory
        fname = 'A_comp_step1.h5'
        tod_obj = TOD(fname)
        tod_obj.add_tod(temp)
        tod_obj.save_tod_to_file()
        tod_obj = None; temp = None
        
        temp = self.double_TF_operator(fname) # (T^T x N^-1 x T) x (P x m)
        
        return self.P_matrix.T.dot(temp) # P^T x (T^T x N^-1 x T x P x m)


def conjugate_gradient(A, b, x0, M_inv, NPIX, pix_obs=[False], tol=1e-5, max_iter=1000):
    """
    Conjugate Gradient Method for solving the linear system Ax = b.

    :param A: The coefficient matrix (square, symmetric, positive-definite).
    :param b: The right-hand side vector.
    :param x0: Initial guess for the solution.
    :param M_inv: The inverse of the preconditioner matrix.
    :param NPIX: The number of pixels (required for reduced formats)
    :param pix_obs: Observed pixels (required for reduced formats)
    :param tol: Tolerance for convergence.
    :param max_iter: Maximum number of iterations.
    :return: The approximate solution x.
    """

    name_of_the_exp = 'cmb_the_whole_survey_143'

    x = x0
    r = b - A(x)
    d = M_inv.dot(r)
    delta = np.dot(r, d)
    delta0 = delta
    file = open('delta_{0}.txt'.format(name_of_the_exp), 'w')
    file.write('{0} {1}\n'.format(0, 1))
    file.close()

    # if the data is reduced
    if all(pix_obs):
        x_map = np.zeros(NPIX)
        r_map = np.zeros(NPIX)

    for k in range(max_iter):
        start1 = time.time()
        q = A(d)
        alpha = delta / np.dot(d, q)
        x += alpha * d
        r -= alpha * q

        s = M_inv.dot(r)
        delta_new = np.dot(r, s)

        file = open('delta_{0}.txt'.format(name_of_the_exp), 'a')
        file.write('{0} {1}\n'.format(k+1, delta_new/delta0))
        file.close()

        if delta_new < tol*tol*delta0:
            break
        beta = delta_new / delta
        print('alpha, beta, delta_new/delta0', alpha, beta, delta_new/delta0)
        d = s + beta * d
        delta = delta_new

        if k%3 == 0:
            if all(pix_obs):
                x_map[pix_obs] = x
                r_map[pix_obs] = r
            else:
                x_map = x
                r_map = r


            hp.fitsfunc.write_map('iter_{0}/CG_iter_{1}_map.fits'.format(name_of_the_exp, k), x_map, overwrite=True)
            hp.fitsfunc.write_map('iter_{0}/CG_iter_{1}_res.fits'.format(name_of_the_exp, k), r_map, overwrite=True) 

        print('Iteration {0} finished in {1} s\n'.format(k, time.time()-start1))

    if all(pix_obs):
        x_map[pix_obs] = x
        r_map[pix_obs] = r
    else:
        x_map = x
        r_map = r


    return x_map, r_map




def main():
    NSIDE = 2048
    reduced = False # use reduced=True for not full sky coverage
    d = Driver(NSIDE)
    d.create_pointing_matrix(reduced=reduced)

    d.generate_tod()

    # deconvolution experiment
    tod = d.inverse_TF_operator(d.tod_path)
    tod_obj = TOD('path/to/h5/tod/file/after/applying/the/inverse/of/the/TF')
    tod_obj.add_tod(tod)
    tod_obj.save_tod_to_file()
    

    '''
    d.calculate_B_matrix()
    tod_obj = TOD('/mn/stornext/d5/data/artemba/other/B_matrix_full_143.h5')
    tod_obj.add_tod(d.B)
    tod_obj.save_tod_to_file()
    '''
    d.load_B_matrix()

    if reduced:
        x0 = np.zeros(d.NPIX_reduced)

        # preconditioner
        M = load_npz('path/to/reduced/precond/matrix/in/npz/file')
    else:
        x0 = np.zeros(d.NPIX)

        # preconditioner
        M = load_npz('path/to/precond/matrix/in/npz/file')

    x_sol, r_sol = conjugate_gradient(d.A_dot_x, d.B, x0, M, d.NPIX)
    hp.fitsfunc.write_map('CG_map_final.fits', x_sol, overwrite=True)
    hp.fitsfunc.write_map('CG_res_final.fits', r_sol, overwrite=True)


if __name__ == '__main__':
    main()

