import numpy as np
from sklearn.cluster import DBSCAN

class Track():
    def __init__(self, clusters, track_id):
        self.clusters = clusters # will be Txd nd array 
        self.id = track_id
        self.since_last = 0
        # should replace clusters with tracked states (centroid of cluster and x,y velocity)
        # Track with respect to ego frame
        centroids = np.mean(clusters, axis=0)
        self.state = np.array()
    
    def __repr__(self,):
        return f"id:{self.id} - last_update: {self.since_last} - centroid: {np.mean(self.clusters, axis=0)}"

class RTracker():
    '''
    Maintains a list of tracked objects
        
    '''
    def __init__(self,**kwargs):
        self.tracks = dict() # Keep track of track objects
        self.track_counter = 0
        self.eps = kwargs.get('db_eps', 4)
        self.min_samples = kwargs.get('db_min_sample', 1)

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
   
    def score_clusters(self, c1, c2):
        '''
        score the different clusters
            c1 - Txd cluster of points
            c2 - KxD cluster of points
        '''
        # centroid eucledian distance
        return np.linalg.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0))
        
    def associate(self, labels, points, tracks, threshold=1, timeout=4):
        '''
        labels - np.array of indices into the points Nxd array 
        points - Nxd new points

        Associate with currently present tracks,
            if association fails, just spawn cluster as new track
        '''
        
        unique_labels = range(max(labels))
        if len(self.tracks) == 0:
            # There are no tracks, iterate over clusters and spawn new if good 
            for label in unique_labels:  
                # TODO: Add Filtering on track based on heuristic
                self.tracks[self.track_counter] = Track(points[labels==label, ...], self.track_counter)
                self.track_counter += 1
        else:
            # Greedy Association
            # if score is too low, do not associate
            best_cluster_score = np.ones(len(unique_labels))*np.inf
            new_tracks = dict()
            dead_tracks = []
            
            for track_id in self.tracks: # Loop over each track 

                cl = self.tracks[track_id].clusters
                
                scores = []
                best_label = -1 # no label
                best_score = np.inf
                for label in unique_labels: # Loop over each cluster
                    cluster = points[labels==label, ...]
                    scores.append(self.score_clusters(cluster, self.tracks.get(track_id, None).clusters))
                    if scores[label] < best_score:
                        best_score = scores[label]
                        best_label = label
                    if scores[label] < best_cluster_score[label]:
                        best_cluster_score[label] = scores[label] # Keep track of this if best cluster is not found, spawn as new track

                if best_score < threshold:
                    self.tracks[track_id].since_last = 0
                    self.tracks[track_id].clusters = points[labels==best_label, ...] # update cluster
                    vis_track(cl, points[labels==best_label, ...], score=best_score, track_id=track_id)
                else:  
                    self.tracks[track_id].since_last += 1
                    if self.tracks[track_id].since_last > timeout:
                        dead_tracks.append(track_id)
                        

            for dead_track in dead_tracks:
                self.tracks.pop(dead_track) # remove track after failing for so long

            for label, best_score in zip(unique_labels, best_cluster_score):
                if best_score >= threshold:
                    print(f"New Track Created {self.track_counter}")
                    self.tracks[self.track_counter] = Track(points[labels==label, ...], self.track_counter)
                    self.track_counter += 1
        
    def cluster(self, points, use_pos=True):
        '''
        points - Nxd N points of d features
        use_pos - use only x and y to cluster, by default this is true
        '''
        return self.dbscan.fit(points).labels_

    def track(self, points): 
        '''
        Update tracks with new data
        '''

def vis_track(old_cluster, new_cluster, score=0, track_id=0):
    # Black removed and is used for noise instead.
    xy = old_cluster
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=(1, 0, 0),
        markeredgecolor="k",
        markersize=6,
    )

    xy = new_cluster
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=(0, 0, 1),
        markeredgecolor="k",
        markersize=6,
    )
    
    plt.title(f"Red: Old - Blue: New - Score: {score} - Track_id: {track_id}")
    plt.show()
 
def vis_cluster(labels, X):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = labels == k
    
        xy = X[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )
    
        xy = X[class_member_mask ]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
   
if __name__ == "__main__":
    '''
    Simple Test of tracker
    '''
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import RadarPointCloud
    import nuscenes as nusc
    import matplotlib.pyplot as plt
    nusc = NuScenes(version='v1.0-mini', dataroot="/home/martin/workspace/Autoronto/nuscenes_mini/RadarPointQuery/data", verbose=True)
    dataroot ="/home/martin/workspace/Autoronto/nuscenes_mini/RadarPointQuery/data" 
    city_scene = nusc.scene[0]
    fs_token = city_scene['first_sample_token']
    current_sample = nusc.get('sample', fs_token)
    rt = RTracker()
    for _ in range(10):

        radar_type = "RADAR_FRONT"
        radar_sample_data = nusc.get('sample_data', current_sample['data'][radar_type])
        radar_cloud = RadarPointCloud.from_file(dataroot+ "/" + radar_sample_data['filename'])
        radar_points = radar_cloud.points
   
        radar_xyz = radar_points[0:2, :].T

        X = radar_xyz
        labels = rt.cluster(X)
        
        rt.associate(labels, X, rt.tracks, threshold=3, timeout=4)

        # #############################################################################
        # Plot result
        
        vis_cluster(labels, X)
        current_sample = nusc.get('sample', current_sample['next'])

