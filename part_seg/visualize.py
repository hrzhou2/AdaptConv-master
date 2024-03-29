from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np

COLORS = ["tomato", "forestgreen", "royalblue", "gold", "cyan", "gray"]

def normalize(points: "numpy array (vertice_num, 3)"):
    center = np.mean(points, axis= 0)
    points = points - center
    max_d = np.sqrt(np.max(points @ (points.T)))
    points = points / max_d
    return points 
    
def visualize(points: "(vertice_num, 3)", labels: "(vertice_num, )", fig_name: str):
    points = np.array(points)
    labels = np.array(labels)

    points = normalize(points)
    eye = np.eye(3)
    bound_points = np.vstack((eye , eye * (-1)))
   
    x ,y ,z = points[:, 0], points[:, 1], points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection= "3d")
    ax.axis("off")
    
    colors = [COLORS[i % len(COLORS)] for i in labels]
    ax.scatter(x ,z, y, s= 3, c= colors, marker= "o")
    ax.scatter(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], s=0.01, c= "white")
    plt.savefig(fig_name)
    plt.close()


def test():
    pass

if __name__ == "__main__":
    test()