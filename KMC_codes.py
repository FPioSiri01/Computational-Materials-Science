import numpy as np
from numba import jit


def printXYZ(namef, height, Lx, Ly):
    fsnap=open(namef, 'w')
    fsnap.write(f"{np.sum(height+1)}\n\n")
    for x in range(Lx):
        for y in range(Ly):
            for z in range(height[x,y]+1):
                fsnap.write(f"{x} {y} {z}\n")
    fsnap.close()

#implement periodic boundary conditions
@jit
def pbcCorr(xi,Li):
    xi -= Li * np.floor(xi/Li)
    return int(xi)


#next 2 functions are the 2 parts of the counting neighbors procedure
@jit
def neighXYZ(x,y,z,height,L):
#focuses on a specific position and adds neighbors
    n = 0
    #right
    xright = pbcCorr(x+1,L)
    yright = y
    if(z <= height[xright,yright]):
        n += 1
    #left
    xleft = pbcCorr(x-1,L)
    yleft = y
    if(z <= height[xleft,yleft]):
        n += 1
    #up
    xup = x
    yup = pbcCorr(y+1,L)
    if(z <= height[xup,yup]):
        n += 1
    #down
    xdown = x
    ydown = pbcCorr(y-1,L)
    if(z <= height[xdown,ydown]):
        n += 1

    return n


@jit
def countNN(height,L):
#iterate evaluation of neighbors on the whole surface of the lattice
    n1 = np.zeros_like(height)

    for x in range(L):
        for y in range(L):
            z = height[x,y]
            n1[x,y]=neighXYZ(x,y,z,height,L)
    
    return n1


@jit
def findMove(rho,kdif,L):
    x = 0
    y = 0
    sumk = kdif[x,y]
    while (rho > sumk):
        if(x < L-1):
            x += 1
        else:
            x = 0
            y += 1

        sumk += kdif[x,y]

    return x, y
    
@jit
def get_rate_at(x, y, h_mat, temp_k, L, J0, J1):
    #Calcola il rate di diffusione per un singolo sito (x,y)
    nu = 1e13
    k_B = 1 / 11603
    z = h_mat[x, y]
    n1 = neighXYZ(x, y, z, h_mat, L)
    
    Eb = - (J0 + n1 * J1)
    k = 4 * nu * np.exp(-Eb / (k_B * temp_k))
    return k

def get_neighborhood(x, y, Lx, Ly):
    """Restituisce le coordinate di (x,y) e dei suoi 4 vicini con PBC."""
    coords = set()
    coords.add((x, y))
    coords.add((pbcCorr(x+1, Lx), y))
    coords.add((pbcCorr(x-1, Lx), y))
    coords.add((x, pbcCorr(y+1, Ly)))
    coords.add((x, pbcCorr(y-1, Ly)))
    return coords