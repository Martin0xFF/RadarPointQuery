import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt

print("Loading Sample Radar Data")
dirs = glob.glob("RADAR_*/")
files = []
for d in dirs:
    files.append(glob.glob(d+"*"))
print(dirs)
for i in range(len(files[0])):
    clouds = []
    for j in range(0, len(files)):
        clouds.append(o3d.io.read_point_cloud(files[j][i]))
        
        labels = np.array(clouds[j].cluster_dbscan(eps=0.2, min_points=10))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        clouds[j].colors = o3d.utility.Vector3dVector(colors[:, :3])
        print(files[j][i]) 
    o3d.visualization.draw_geometries([clouds[0]],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

exit(0)
