import numpy as np
import plotly.graph_objects as go

def display_3d_points_go(X, x, img, fig): 
    
    # Plot a 3d set of points
    x_img = (x[:,:2].astype(int))
    rgb_txt = (img[x_img[:,1], x_img[:,0]])/255
    
    fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2],mode='markers',name='test',marker=dict(color=rgb_txt,size=2)))

    return


def display_3d_points_go2(X, fig):
    # Plot a 3d set of points
    fig.add_trace(
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', name='test'))

    return


def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, img_shape, fig, legend, scale=2):
    h, w = img_shape
    
    o = optical_center(P)
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',name=legend))
    
    return