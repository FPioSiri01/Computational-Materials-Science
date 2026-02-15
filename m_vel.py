import numpy as np

def calc_vel(cc,natoms,myseed,temp):
    np.random.seed(myseed)

    vel = np.zeros((natoms,3))
    #v2 = np.zeros(natoms)
    E_kin = 0
    k_B = 1 / 11603
    mass = 108 * 1.66*10**(-27) / 16

    
    for i in range(0,natoms):
        vel[i,0] = np.random.uniform(-cc,cc) 
        vel[i,1] = np.random.uniform(-cc,cc) 
        vel[i,2] = np.random.uniform(-cc,cc) 

    meanVx = np.mean(vel[:,0])
    meanVy = np.mean(vel[:,1])
    meanVz = np.mean(vel[:,2])

    # sottraggo moto del CM
    vtot2 = 0
    for i in range(0,natoms):
        vel[i,0] = vel[i,0] - meanVx
        vel[i,1] = vel[i,1] - meanVy
        vel[i,2] = vel[i,2] - meanVz

        vtot2 += vel[i,0]**2 + vel[i,1]**2 + vel[i,2]**2

    E_kin = 0.5 * mass * vtot2
    temp_new = 2 / (3*natoms*k_B) * E_kin

    vtot2 = 0
    for i in range(0,natoms):
        vel[i,0] = vel[i,0] * np.sqrt(temp / temp_new)
        vel[i,1] = vel[i,1] * np.sqrt(temp / temp_new)
        vel[i,2] = vel[i,2] * np.sqrt(temp / temp_new)

        vtot2 += vel[i,0]**2 + vel[i,1]**2 + vel[i,2]**2

    E_kin = 0.5 * mass * vtot2
    temp_fin = 2 / (3*natoms*k_B) * E_kin

    print("Check initial velocities:\n Temperature (T , T' , T'_check) ", temp,temp_new,temp_fin)
    
    return vel, E_kin