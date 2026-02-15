import numpy as np
from numba import jit


def calc_epot_sharp(natoms,pos,epsilon,sigma,nnbrs,whichnbr):

    E_pot = 0

    for i in range (0,natoms):
        for j in range(0,nnbrs[i]):
            
            k = whichnbr[i,j]
            dx = pos[i,0]-pos[k,0]
            dy = pos[i,1]-pos[k,1]
            dz = pos[i,2]-pos[k,2]
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 

            #Potential energy using Lennard-Jones
            LJ = 4*epsilon*((sigma / dist)**(12)-(sigma / dist)**(6))
            E_pot += LJ
            
    E_pot = 0.5*E_pot
    
    return E_pot


@jit
def calc_epot_pol(natoms,pos,epsilon,sigma,nnbrs,whichnbr, rpoly, poly_coeffs):

    E_pot = 0

    for i in range (0,natoms):
        for j in range(0,nnbrs[i]):
            
            k = whichnbr[i,j]
            dx = pos[i,0]-pos[k,0]
            dy = pos[i,1]-pos[k,1]
            dz = pos[i,2]-pos[k,2]
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 
            if(dist < rpoly):
                #Potential energy using Lennard-Jones
                LJ = 4*epsilon*((sigma / dist)**(12)-(sigma / dist)**(6))
                E_pot += LJ
            else:
                Poly = poly_coeffs[0] + poly_coeffs[1]*dist + poly_coeffs[2]*dist**2 + poly_coeffs[3]*dist**3 + poly_coeffs[4]*dist**4 + poly_coeffs[5]*dist**5 + poly_coeffs[6]*dist**6 + poly_coeffs[7]*dist**7
                E_pot += Poly
    
    E_pot = 0.5*E_pot
    
    return E_pot



@jit 
def calc_epot_pol_pbc(natoms,pos,epsilon,sigma,nnbrs,whichnbr,rpoly, poly_coeffs,Lx,Ly,Lz):

    E_pot = 0

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

            if(dist < rpoly):
                #Potential energy using Lennard-Jones
                LJ = 4*epsilon*((sigma / dist)**(12)-(sigma / dist)**(6))
                E_pot += LJ
            else:
                Poly = poly_coeffs[0] + poly_coeffs[1]*dist + poly_coeffs[2]*dist**2 + poly_coeffs[3]*dist**3 + poly_coeffs[4]*dist**4 + poly_coeffs[5]*dist**5 + poly_coeffs[6]*dist**6 + poly_coeffs[7]*dist**7
                E_pot += Poly
    
    E_pot = 0.5*E_pot
    
    return E_pot



@jit 
def calc_epot_pol_pbc_cages(natoms,pos,epsilon,sigma,nnbrs,whichnbr,cutoff,rpoly,poly_coeffs,Lx,Ly,Lz):

    E_pot = 0

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

            if(dist < rpoly):
                #Potential energy using Lennard-Jones
                LJ = 4*epsilon*((sigma / dist)**(12)-(sigma / dist)**(6))
                E_pot += LJ
            elif (rpoly < dist < cutoff):
                Poly = poly_coeffs[0] + poly_coeffs[1]*dist + poly_coeffs[2]*dist**2 + poly_coeffs[3]*dist**3 + poly_coeffs[4]*dist**4 + poly_coeffs[5]*dist**5 + poly_coeffs[6]*dist**6 + poly_coeffs[7]*dist**7
                E_pot += Poly
            elif (dist > cutoff):
                continue 
    
    E_pot = 0.5*E_pot
    
    return E_pot