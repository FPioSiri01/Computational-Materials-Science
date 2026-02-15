import numpy as np
from numba import jit

@jit
def calc_nbrs(natoms,pos,cutoff):

    whichnbr = np.zeros((natoms,natoms), dtype=np.int64) 
    nnbrs = np.zeros((natoms,), dtype=np.int64)

    for i in range(0,natoms):
        for j in range(0,natoms):

            dxyz = pos[i,:]-pos[j,:]
            dist = np.sqrt(sum(dxyz**2))

            if i!=j and dist<=cutoff: 
                nnbrs[i] += 1
                whichnbr[i,nnbrs[i]-1] = j
                
    return nnbrs,whichnbr

"""""
def calc_nbrs(natoms,pos,cutoff):

    whichnbr = np.zeros((natoms,natoms), dtype=int) # we need int b/c it's a matrix with the labels of the neighbors
    nnbrs = np.zeros((natoms,), dtype=int)

    for i in range(0,natoms):
        for j in range(0,natoms):
            dx = pos[i,0]-pos[j,0]
            dy = pos[i,1]-pos[j,1]
            dz = pos[i,2]-pos[j,2]
            #d2 = dx*dx + dy*dy + dz*dz
            #dist = d2**(0.5) 
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if i!=j and dist<=cutoff: 
                nnbrs[i] += 1
                whichnbr[i,nnbrs[i]-1] = j
                
    return nnbrs,whichnbr
"""""

@jit
def calc_nbrs_pbc(natoms,pos,cutoff,Lx,Ly,Lz):

    whichnbr = np.zeros((natoms,natoms), dtype=np.int64) 
    nnbrs = np.zeros((natoms,), dtype=np.int64)

    for i in range(0,natoms):
        for j in range(0,natoms):
            
            dx = pos[i,0]-pos[j,0]
            dx = dx - Lx*np.round(dx / Lx)
            dy = pos[i,1]-pos[j,1]
            dy = dy - Ly*np.round(dy / Ly)
            dz = pos[i,2]-pos[j,2]
            dz = dz - Lz*np.round(dz / Lz)
            d2 = dx*dx + dy*dy + dz*dz
            dist = d2**(0.5) 

            if i!=j and dist<cutoff: 
                nnbrs[i] += 1
                whichnbr[i,nnbrs[i]-1] = j
                
    return nnbrs,whichnbr


@jit
def calc_nbrs_linked_cells(natoms, pos, rc, Lx, Ly, Lz):
    # 1. Definizione della griglia di celle
    # Il numero di celle deve garantire che la dimensione della cella sia >= rc
    n_cells_x = int(Lx // rc)
    n_cells_y = int(Ly // rc)
    n_cells_z = int(Lz // rc)
    
    # Gestione casi limite (piccoli box)
    if n_cells_x < 1: n_cells_x = 1
    if n_cells_y < 1: n_cells_y = 1
    if n_cells_z < 1: n_cells_z = 1
    
    cell_size_x = Lx / n_cells_x
    cell_size_y = Ly / n_cells_y
    cell_size_z = Lz / n_cells_z
    
    # 2. Strutture dati per le Linked Lists
    # head: contiene l'indice dell'ultimo atomo aggiunto alla cella (x,y,z)
    head = -1 * np.ones((n_cells_x, n_cells_y, n_cells_z), dtype=np.int32)
    # lscl (Linked Cell List): contiene l'indice dell'atomo precedente nella stessa cella
    lscl = -1 * np.ones(natoms, dtype=np.int32)
    
    # 3. Assegnazione degli atomi alle celle
    for i in range(natoms):
        # Applichiamo PBC per trovare l'indice corretto della cella
        x_pbc = pos[i, 0] - Lx * np.floor(pos[i, 0] / Lx)
        y_pbc = pos[i, 1] - Ly * np.floor(pos[i, 1] / Ly)
        z_pbc = pos[i, 2] - Lz * np.floor(pos[i, 2] / Lz)
        
        cx = int(x_pbc / cell_size_x)
        cy = int(y_pbc / cell_size_y)
        cz = int(z_pbc / cell_size_z)
        
        # Clamp per sicurezza numerica
        if cx >= n_cells_x: cx = n_cells_x - 1
        if cy >= n_cells_y: cy = n_cells_y - 1
        if cz >= n_cells_z: cz = n_cells_z - 1
        
        # Costruzione della lista collegata
        lscl[i] = head[cx, cy, cz]
        head[cx, cy, cz] = i
        
    # 4. Costruzione della Neighbor List
    # Stima massima dei vicini per pre-allocazione (es. 200 per FCC con r_cut tipico)
    max_nbrs = 200 
    whichnbr = np.zeros((natoms, max_nbrs), dtype=np.int32)
    nnbrs = np.zeros(natoms, dtype=np.int32)
    
    rc2 = rc * rc
    
    # Loop su tutte le celle
    for cx in range(n_cells_x):
        for cy in range(n_cells_y):
            for cz in range(n_cells_z):
                
                # Primo atomo nella cella corrente
                i = head[cx, cy, cz]
                
                # Scorre tutti gli atomi 'i' nella cella corrente
                while i != -1:
                    
                    # Loop sulle celle vicine (incluso se stessa e immagini periodiche)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                
                                # Indici cella vicina con PBC
                                ncx = (cx + dx + n_cells_x) % n_cells_x
                                ncy = (cy + dy + n_cells_y) % n_cells_y
                                ncz = (cz + dz + n_cells_z) % n_cells_z
                                
                                # Primo atomo nella cella vicina
                                j = head[ncx, ncy, ncz]
                                
                                # Scorre tutti gli atomi 'j' nella cella vicina
                                while j != -1:
                                    if i != j: # Evita auto-interazione
                                        
                                        # Calcolo distanza con PBC (Minimum Image Convention)
                                        d_x = pos[j, 0] - pos[i, 0]
                                        d_y = pos[j, 1] - pos[i, 1]
                                        d_z = pos[j, 2] - pos[i, 2]
                                        
                                        d_x = d_x - Lx * np.round(d_x / Lx)
                                        d_y = d_y - Ly * np.round(d_y / Ly)
                                        d_z = d_z - Lz * np.round(d_z / Lz)
                                        
                                        dist2 = d_x*d_x + d_y*d_y + d_z*d_z
                                        
                                        if dist2 < rc2:
                                            k = nnbrs[i]
                                            if k < max_nbrs:
                                                whichnbr[i, k] = j
                                                nnbrs[i] += 1
                                    
                                    # Prossimo atomo 'j' nella cella vicina
                                    j = lscl[j]
                                    
                    # Prossimo atomo 'i' nella cella corrente
                    i = lscl[i]
                    
    return nnbrs, whichnbr