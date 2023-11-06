import numpy as np

fmod = 90.1875901876
fsamp = 2.0 * fmod

def kernel_four_tau(f, par):
  ''' sum of four lowpass filters '''
  w = 2.0*np.pi*f*1j
  return (par[0] / (1.0+w*par[3])
          + par[1] / (1.0+w*par[4])
          + par[2] / (1.0+w*par[5])
          + par[8] / (1.0+w*par[9]))


def lowpass(omega, tau):
  return 1 / (1.0 +omega*tau*1j)


def LFER4(f, par=None):
  ''' LFER4 transfer function '''
  # par[6] = tau_stray
  # par[7] = S_phase
  # par[0,1,2,8] = a1, a2, a3, a4
  # par[3,4,5,9] = tau1, tau2, tau3, tau4

  par = [0.491, 0.397, 0.0962, 6.64e-3, 6.64e-3, 26.4e-3, 2.02e-3, 0.00139, 0.0156, 336.e-3] # 143-5

  tau1 = 1.e3*100.e-9 # R1*C1
  tau3 = 10.e3*10.e-9 # R4*C4
  tau4 = 51.e3*1.e-6 # R2*C2
  zz3 = 510.e3 # R78
  zz4 = 1e-6 # C18
  zx1 = 18.7e3 # R9
  zx2 = 37.4e3 # R12
  fangmod = fmod * np.pi * 2
  nn = 5
  tau0 = par[6]
  sphase = par[7]
  zout = np.ones(f.shape[0], dtype=complex)
  norm = 1.0

  for i in range(f.shape[0]):
      ff = f[i]
      zbolo = kernel_four_tau(ff,par)
      omega = 2 * np.pi * ff
      tf =  0
      signe = -1
      for i1 in np.arange(1, nn+1):
        signe *= -1
        omegamod = (2.0*i1 - 1.0) * fangmod
        omegap = omega + omegamod
        omegam = - (omegamod - omega)

        # resonance low pass
        zfelp = lowpass(omegap, tau0)
        zfelm = lowpass(omegam, tau0)

        # electronic rejection filter
        zf1plu = (1.0 + 0.5*omegap*tau1*1j) / (1.0 + omegap*tau1*1j)
        zf1min = (1.0 + 0.5*omegam*tau1*1j) / (1.0 + omegam*tau1*1j)
        zfelp = zfelp * zf1plu
        zfelm = zfelm * zf1min

        # Sallen-Key high pass
        zSKplu = tau4*omegap*1j / (1.0 + omegap*tau4*1j)
        zSKmin = tau4*omegam*1j / (1.0 + omegam*tau4*1j)

        zfelp = zfelp * zSKplu * zSKplu
        zfelm = zfelm * zSKmin * zSKmin

        # sign reverse and gain
        zfelp *= -5.1
        zfelm *= -5.1

        # lowpass
        zfelp = zfelp * 1.5 * lowpass(omegap, tau3)
        zfelm = zfelm * 1.5 * lowpass(omegam, tau3)

        # third order equation

        zden3 = -1.0 * omegap**3 * zx1*zx1*zz3*zx2*zx2*1.0e-16*zz4
        zden2 = -1.0 * omegap*omegap*(zx1*zx2*zx2*zz3*1.e-16
                                      + zx1*zx1*zx2*zx2*1.e-16
                                      + zx1*zx2*zx2*zz3*zz4*1.e-8)
        zden1i = omegap*(zx1*zx2*zx2*1.e-8+zx2*zz3*zx1*zz4) + zden3
        zden1r = zx2*zx1 + zden2

        zfelp = zfelp * (2.0*zx2*zx1*zz3*zz4*omegap*1j) \
                / (zden1r + zden1i*1j)

        zden3 = -1.0 * omegam**3 * zx1*zx1*zz3*zx2*zx2*1.0e-16*zz4
        zden2 = -1.0 * omegam*omegam*(zx1*zx2*zx2*zz3*1.e-16
                                      + zx1*zx1*zx2*zx2*1.e-16
                                      + zx1*zx2*zx2*zz3*zz4*1.e-8)
        zden1i = omegam*(zx1*zx2*zx2*1.e-8+zx2*zz3*zx1*zz4) + zden3
        zden1r = zx2*zx1 + zden2

        zfelm = zfelm*(2.0*zx2*zx1*zz3*zz4*omegam*1j) \
                / (zden1r + zden1i*1j)

        # averaging effect
        arg = np.pi * omegap / (2*fangmod)
        zfelp = zfelp * (-1.0) * np.sin(arg) / arg
        arg = np.pi * omegam / (2*fangmod)
        zfelm = zfelm * (-1.0) * np.sin(arg) / arg

        zfelp = zfelp * (np.cos(sphase*omegap) + np.sin(sphase*omegap)*1j)
        zfelm = zfelm * (np.cos(sphase*omegam) + np.sin(sphase*omegam)*1j)
        tf = tf + (signe/(2.0*i1-1))*(zfelp + zfelm)
      if ff == 0:
        norm = tf
      zout[i] = tf/norm*zbolo

  return zout
