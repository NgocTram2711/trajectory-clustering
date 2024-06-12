import os
import pickle

import geopandas as gpd
import holoviews as hv
import movingpandas as mpd
import numpy as np

from datetime import timedelta
from holoviews import dim
from shapely import LineString
from sklearn.cluster import DBSCAN

from src.utils import find_closest_segment


class Solution:
    GDF_COLUMNS = ["time", "track_fid", "track_seg_id", "track_seg_point_id", "geometry", "label"]

    def __init__(self, root, traj_collection):
        self.root = root
        self.traj_collection = traj_collection
        self.labels_ = []

    @property
    def file_name(self):
        return f"solution.pkl"

    @property
    def X(self):
        traces = list(self.traj_collection.get_data_as_points())
        return np.array([list(point[-1].coords)[0] for point in traces])

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.file_name))

    def save(self, **kwargs):
        print("SAVING SOLUTION...")
        with open(os.path.join(self.root, self.file_name), "wb") as f:
            pickle.dump(kwargs, f)

    @classmethod
    def get_plot(cls, traj_collection_clustered, clabel, mode):
        if mode == "points":
            pdf = traj_collection_clustered.to_point_gdf()
            return pdf.hvplot(
                "Longitude", "Latitude", geo=True, tiles="CartoLight",
                c="label", clim=(pdf["label"].min(), pdf["label"].max()),
                cmap="plasma", colorbar=True, clabel=clabel,
            )
        elif mode == "trajectories":
            return traj_collection_clustered.hvplot(
                c="label", tiles="CartoLight",
                cmap="plasma", colorbar=True, clabel=clabel,
            )
        return None


class Tca(Solution):
    def __init__(self, root, traj_collection, min_d, max_d, should_cluster=True):
        super().__init__(root, traj_collection)
        self.min_d = min_d
        self.max_d = max_d
        self.minutes = 10
        self.should_cluster = should_cluster

        self.clusters = None
        self.flows = None
        self.traj_collection_clustered = None
        self.solve()

    @property
    def file_name(self):
        return f"tca_{self.min_d}_{self.max_d}.pkl"

    def solve(self):
        if self._check_exists():
            print(f"LOADING TCA ({self.min_d}, {self.max_d}) SOLUTION...")
            with open(os.path.join(self.root, self.file_name), "rb") as f:
                solution = pickle.load(f)
                self.clusters = solution.get("clusters")
                self.flows = solution.get("flows")
                self.traj_collection_clustered = solution.get("traj_collection_clustered")
            return

        print(f"SOLVING TCA ({self.min_d}, {self.max_d})...")
        aggregator = mpd.TrajectoryCollectionAggregator(
            self.traj_collection.data,
            min_distance=self.min_d,
            max_distance=self.max_d,
            min_stop_duration=timedelta(minutes=self.minutes)
        )
        self.clusters = aggregator.get_clusters_gdf()
        self.flows = aggregator.get_flows_gdf()
        self.clean()
        if self.should_cluster:
            self.cluster_points()
        else:
            self.save(clusters=self.clusters, flows=self.flows)

    def clean(self):
        print("CLEANING TCA SOLUTION...")
        merged_flows = []
        visited = set()
    
        for i, row_i in self.flows.iterrows():
            if i in visited:
                continue
    
            line_i = row_i["geometry"]
            weight = row_i["weight"]
            obj_weight = row_i["obj_weight"]
    
            for j, row_j in self.flows.iterrows():
                line_j = row_j["geometry"]
                if i != j and j not in visited and line_i.equals(LineString(line_j.coords[::-1])):
                    weight += row_j["weight"]
                    obj_weight += row_j["obj_weight"]
                    visited.add(j)
    
            merged_flows.append({
                "geometry": line_i,
                "weight": weight,
                "obj_weight": obj_weight,
            })
    
        self.flows = gpd.GeoDataFrame(merged_flows)
        self.flows.sort_values("weight", ignore_index=True)

    def cluster_points(self):
        if self.traj_collection_clustered is not None:
            return

        print("CLUSTERING POINTS...")
        traces = []
        self.labels_ = []
        for point in self.traj_collection.get_data_as_points():
            weight = find_closest_segment(self.flows, point[-1])
            traces.append([*point, weight])
            self.labels_.append(weight)
        gdf = gpd.GeoDataFrame(
            traces,
            columns=Tca.GDF_COLUMNS,
            geometry="geometry",
            crs="epsg:4326"
        )
        gdf["max"] = gdf.groupby("track_fid")["label"].transform("max")
        gdf = gdf.sort_values("max", ascending=False).drop("max", axis=1)
        self.traj_collection_clustered = mpd.TrajectoryCollection(gdf, "track_fid", t="time")
        self.save(clusters=self.clusters, flows=self.flows, traj_collection_clustered=self.traj_collection_clustered)

    def plot(self, file_name, mode="flow"):
        print("PLOTTING TCA SOLUTION...")
        if mode == "flow":
            _plot = self.flows.hvplot(
                geo=True, c="weight", line_width=3,
                tiles="CartoLight",
                cmap="plasma", colorbar=True, clabel="Number of trajectories",
            ) * self.clusters.hvplot(
                geo=True, color="blue", alpha=dim("n").norm().clip(min=0.3),
                size=dim("n").norm().clip(min=0.2) * 28
            )
        else:
            _plot = self.get_plot(self.traj_collection_clustered, "Number of trajectories", mode)
        hv.save(_plot, file_name)


class Dbscan(Solution):
    def __init__(self, root, traj_collection, eps, min_samples):
        super().__init__(root, traj_collection)
        self.eps = eps
        self.min_samples = min_samples

        self.traj_collection_clustered = None
        self.solve()

    @property
    def file_name(self):
        return f"dbscan_{self.eps:.5f}_{self.min_samples}.pkl"

    def solve(self):
        if self._check_exists():
            print(f"LOADING DBSCAN ({self.eps:.5f}, {self.min_samples}) SOLUTION...")
            with open(os.path.join(self.root, self.file_name), "rb") as f:
                solution = pickle.load(f)
                self.traj_collection_clustered = solution.get("traj_collection_clustered")
            return

        print(f"SOLVING DBSCAN ({self.eps:.5f}, {self.min_samples})...")
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        db.fit(self.X)
        unique, counts = np.unique(db.labels_, return_counts=True)
        mapping = dict(zip(unique, counts))
        mapping[-1] = -1
        mp = np.vectorize(lambda el: mapping[el])
        self.labels_ = mp(db.labels_)
        traces = list(self.traj_collection.get_data_as_points())
        gdf = gpd.GeoDataFrame(
            [(*point, label) for point, label in zip(traces, self.labels_)],
            columns=Dbscan.GDF_COLUMNS,
            geometry="geometry",
            crs="epsg:4326"
        )
        self.traj_collection_clustered = mpd.TrajectoryCollection(gdf, "track_fid", t="time")
        self.save(traj_collection_clustered=self.traj_collection_clustered)

    def plot(self, file_name, mode="points"):
        print("PLOTTING DBSCAN SOLUTION...")
        _plot = self.get_plot(self.traj_collection_clustered, "Number of points", mode)
        hv.save(_plot, file_name)
