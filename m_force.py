import numpy as np
from poly7 import poly7 
from numba import jit 

@jit
def calc_force_sharp(natoms, pos, epsilon, sigma, nnbrs, whichnbr):

    force = np.zeros((natoms,3))

    for i in range (0,natoms):
        for j in range(0,nnbrs[i]):
            
            k = whichnbr[i,j]
            dx = pos[i,0]-pos[k,0]
            dy = pos[i,1]-pos[k,1]
            dz = pos[i,2]-pos[k,2]
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 

            #Forces
            force[i,0] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dx
            force[i,1] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dy
            force[i,2] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dz
        
    return force 


@jit
def calc_force_pol(natoms, pos, epsilon, sigma, nnbrs, whichnbr, rpoly, poly_coeffs):

    force = np.zeros((natoms,3))

    for i in range (0,natoms):
        for j in range(0,nnbrs[i]):
            
            k = whichnbr[i,j]
            dx = pos[i,0]-pos[k,0]
            dy = pos[i,1]-pos[k,1]
            dz = pos[i,2]-pos[k,2]
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 

            if dist < rpoly:
                #Forces
                force[i,0] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dx
                force[i,1] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dy
                force[i,2] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dz
            else:
                Polyder = poly_coeffs[1] + 2*poly_coeffs[2]*dist + 3*poly_coeffs[3]*dist**2 + 4*poly_coeffs[4]*dist**3 + 5*poly_coeffs[5]*dist**4 + 6*poly_coeffs[6]*dist**5 + 7*poly_coeffs[7]*dist**6
                force[i,0] += - Polyder * dx / dist
                force[i,1] += - Polyder * dy / dist
                force[i,2] += - Polyder * dz / dist

    return force

@jit
def calc_force_pol_pbc(natoms, pos, epsilon, sigma, nnbrs, whichnbr, rpoly, poly_coeffs, Lx, Ly, Lz):

    force = np.zeros((natoms,3))

    for i in range (0,natoms):
        for j in range(0,nnbrs[i]):
            
            k = whichnbr[i,j]
            dx = pos[i,0]-pos[k,0]
            dx = dx - Lx*np.round(dx / Lx)
            dy = pos[i,1]-pos[k,1]
            dy = dy - Ly*np.round(dy / Ly)
            dz = pos[i,2]-pos[k,2]
            dz = dz - Lz*np.round(dz / Lz)
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 

            if dist < rpoly:
                #Forces
                force[i,0] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dx
                force[i,1] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dy
                force[i,2] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dz
            else:
                Polyder = poly_coeffs[1] + 2*poly_coeffs[2]*dist + 3*poly_coeffs[3]*dist**2 + 4*poly_coeffs[4]*dist**3 + 5*poly_coeffs[5]*dist**4 + 6*poly_coeffs[6]*dist**5 + 7*poly_coeffs[7]*dist**6
                force[i,0] += - Polyder * dx / dist
                force[i,1] += - Polyder * dy / dist
                force[i,2] += - Polyder * dz / dist

    return force 


@jit
def calc_force_pol_pbc_cages(natoms, pos, epsilon, sigma, nnbrs, whichnbr, cutoff, rpoly, poly_coeffs, Lx, Ly, Lz):

    force = np.zeros((natoms,3))

    for i in range (0,natoms):
        for j in range(0,nnbrs[i]):
            
            k = whichnbr[i,j]
            dx = pos[i,0]-pos[k,0]
            dx = dx - Lx*np.round(dx / Lx)
            dy = pos[i,1]-pos[k,1]
            dy = dy - Ly*np.round(dy / Ly)
            dz = pos[i,2]-pos[k,2]
            dz = dz - Lz*np.round(dz / Lz)
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 

            if dist == 0:
                print("Error! Dist=0 --> error in nbrs ")

            if (dist < rpoly):
                #Forces
                force[i,0] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dx
                force[i,1] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dy
                force[i,2] += 24*epsilon*sigma**6 * (2*sigma**6 / dist**6 - 1) / dist**8 * dz
            elif (rpoly < dist < cutoff):
                Polyder = poly_coeffs[1] + 2*poly_coeffs[2]*dist + 3*poly_coeffs[3]*dist**2 + 4*poly_coeffs[4]*dist**3 + 5*poly_coeffs[5]*dist**4 + 6*poly_coeffs[6]*dist**5 + 7*poly_coeffs[7]*dist**6
                force[i,0] += - Polyder * dx / dist
                force[i,1] += - Polyder * dy / dist
                force[i,2] += - Polyder * dz / dist
            elif (dist > cutoff):
                continue 

    return force