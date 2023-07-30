import matplotlib.pyplot as plt 
import numpy as np 
import trimesh
from pdb import set_trace
# from scipy.sparse import coo_array


def init_verts(vertices,n):
    for i in range(n):
        for j in range(n):
            vertices[i*n+j] = np.array([i/n  ,j/n ,0])
            
    return vertices


def get_faces(n,faces):
    face_count = 0
    for i in range(n-1):
        for j in range(n-1):
            #first triangle
            f1 = i*n+j
            f2 = i*n+j+1
            f3 = ((i+1)*n)+j
            faces[face_count] = np.array([f1,f2,f3])
            face_count+=1
            # second triangle
            f4 = i*n+j+1
            f5 = ((i+1)*n)+j
            f6 = ((i+1)*n)+j+1
            faces[face_count] = np.array([f4,f6,f5])
            face_count+=1
    return faces




if __name__=='__main__':
    
    n = 50
    
    #initialize mesh segments
    num_verts = n*n
    vertices = np.zeros((n*n,3))
    vertices = init_verts(vertices,n)
    num_faces = (n-1)*(n-1)*2
    faces = np.zeros((num_faces,3))

    faces = get_faces(n,faces)

    # save mesh for visualisation
    mesh = trimesh.Trimesh(vertices, faces)
    # mesh.export('check.obj')
    
    #get boundary_verts
    all_edges = mesh.edges
    sorted_edges = np.sort(all_edges, axis=1)
    unique_edges, edge_counts = np.unique(sorted_edges, axis=0, return_counts=True)
    edges = unique_edges[edge_counts == 1]

    edge_vert_idx, idx_counts = np.unique(edges,return_counts=True)

    #rhs; 0 for laplace, forcing funciton for Poisson
    # f = lambda x: x.sum()
    # f= lambda x: x[0]**2  - x[1]**2
    f= lambda x: np.sin(10*x[0]) * np.sin(10*x[1])

    # A = coo_array((num_verts,num_verts), dtype=np.float32)
    A = np.zeros((num_verts,num_verts))
    M = np.zeros((num_verts,num_verts))
    F = np.array([f(v) for v in mesh.vertices]) #rhs
    u = np.zeros((num_verts,1)) #unknown function
    
      
    
    for i in range(num_faces):
        vert_list = faces[i]
        vert_list = np.array(vert_list, dtype=np.int32)
        v1,v2,v3 = vertices[vert_list,:]
        
        #transformation matrix Jaccobian
        J = np.zeros((3,2))
        J[:,0] = v2 - v1
        J[:,1] = v3 - v1
        
        J_inv = np.linalg.pinv(J).T #psuedo inverse
        
        #det
        det = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        
        #gradient of local basis function
        grad0 = np.array([-1,-1])
        grad1 = np.array([1,0])
        grad2 = np.array([0,1])

        grad = [grad0,grad1,grad2]        

        for j in range(3):
            idx1 = j 
            idx2= (j+1)%3
            
            v1 = vertices[vert_list,idx1]
            v2 = vertices[vert_list,idx2]
            
            gradi = grad[idx1]
            gradj = grad[idx2]
            
            integral = np.dot((J_inv @ gradi), (J_inv @ gradj)) * det * 0.5
            A[int(vert_list[idx1]), int(vert_list[idx2])] += integral
            A[int(vert_list[idx2]), int(vert_list[idx1])] += integral
            
            integral = np.dot((J_inv @ gradi), (J_inv @ gradi)) * det * 0.5
            A[int(vert_list[idx1]),int(vert_list[idx1])] += integral

            M[int(vert_list[idx1]), int(vert_list[idx2])] += det / 24.0
            M[int(vert_list[idx2]), int(vert_list[idx1])] += det / 24.0
            
            M[int(vert_list[idx1]), int(vert_list[idx1])] += det / 12.0


    rhs = M @ F
    
    #boundary conditions
    for i in range(num_verts):
        if i in edge_vert_idx:
            A[i] = np.zeros((1,num_verts))
            A[i,i] = 1.0
            
            #rhs 
            rhs[i] =  1.0#(vertices[i,0]*(vertices[i,0]-1) ) * vertices[i,1]*(vertices[i,1]-1)

    
    
    u_func = np.linalg.inv(A) @ rhs
    # colors = np.expand_dims(u,1)
    u_func = (u_func-u_func.min())/(u_func.max()-u_func.min())
    
    # cm = plt.get_cmap('gist_rainbow', lut=8)
    # colors = cm(u_func) * 255.0
    
    colors = np.repeat(np.expand_dims(u_func,-1),3,-1) * 255
    mesh_new = trimesh.Trimesh(mesh.vertices,mesh.faces,vertex_colors=colors[:,:3])
    mesh_new.export('export_mesh.obj')