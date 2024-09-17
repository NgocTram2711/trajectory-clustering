import glob
import os
import geopandas as gpd
import holoviews as hv
import movingpandas as mpd
import pandas as pd
import pickle

from datetime import timedelta
from urllib.request import urlretrieve

from src.utils import contains_gpx_data


class OSMTraces:
    API_URL = "https://api.openstreetmap.org/api/0.6/trackpoints"

    def __init__(self, root, bbox, download=False):
        self.root = root
        self.bbox = bbox
        self.raw = gpd.GeoDataFrame()
        self.data = mpd.TrajectoryCollection([])

        if download:
            self.download()

        self.load()

    @property
    def trajectories(self):
        return self.data.trajectories

    def _check_exists(self, file_name=""):
        return os.path.exists(os.path.join(self.root, file_name))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.root)
        print("DOWNLOADING...")
        page = 0
        bbox = ",".join([str(b) for b in self.bbox])
        print(bbox)
        while True:
            file_name = os.path.join(self.root, f"tracks({page}).gpx")
            url = f"{self.API_URL}?bbox={bbox}&page={page}"
            print(url)
            urlretrieve(url, file_name)
            if not contains_gpx_data(file_name):
                os.remove(file_name)
                break
            page += 1
            if page == 10:
                break

    def load_raw_data(self):
        self.raw = gpd.GeoDataFrame(
            columns=["track_fid", "track_seg_id", "track_seg_point_id", "time", "geometry"],
            geometry="geometry"
        )
        search_pattern = os.path.join(self.root, "tracks(*).gpx")
        for file_name in glob.glob(search_pattern):
            init = 0 if len(self.raw.index) == 0 else self.raw["track_fid"].unique()[-1] + 1,
            new_data = gpd.read_file(file_name, layer="track_points")
            new_data.drop(columns=["ele", "course", "speed", "magvar", "geoidheight", "name", "cmt", "desc",
                                   "src", "url", "urlname", "sym", "type", "fix", "sat", "hdop", "vdop",
                                   "pdop", "ageofdgpsdata", "dgpsid"], inplace=True)
            new_data["track_fid"] += init
            new_data = new_data.groupby("track_fid")
            new_data = new_data.apply(lambda group: group.sort_values(by="time"))
            new_data = new_data.reset_index(drop=True)
            self.raw = pd.concat([self.raw, new_data])

    def load(self):
        print("LOADING DATA...")
        if self._check_exists("osm_traces.pkl"):
            with open(os.path.join(self.root, "osm_traces.pkl"), "rb") as f:
                self.data = pickle.load(f)
            return

        self.load_raw_data()
        self.data = mpd.TrajectoryCollection(self.raw, "track_fid", t="time")
        # print(f"LOADED {len(self.data)} TRAJECTORIES")
        self.data.add_distance(overwrite=True, units="m")
        # self.data.add_timedelta(overwrite=True)
        self.data.add_speed(name="speed", overwrite=True, units=("km", "h"))
        self.clean()

    def clean(self):
        if self._check_exists("osm_traces.pkl"):
            return

        print("CLEANING DATA...")
        self.data = mpd.DouglasPeuckerGeneralizer(self.data).generalize(tolerance=0.0001) #tolerance=0.0001 thuật toán được phép loại bỏ các điểm dữ liệu nhỏ hơn ngưỡng này
        new_data = []
        for i, track in enumerate(self.trajectories):
            for j, traj in enumerate(mpd.ObservationGapSplitter(track).split(gap=timedelta(minutes=30)).trajectories): #chia trajectory thành các phần nhỏ hơn nếu gap > 30 phút
                if len(traj.df.index) < 10: # nếu trajectory con ít hơn 10 điểm thì bỏ qua
                    continue
                new_data.append(traj)
        self.data = mpd.TrajectoryCollection(new_data)
        self.save()

    def get_data_as_points(self):
        for traj_id, traj in enumerate(self.trajectories):
            timestamps = traj.to_point_gdf().index
            data = traj.to_point_gdf().values
            yield from [(t, *d[:-2]) for t, d in zip(timestamps, data)]

    def save(self):
        if not self._check_exists("osm_traces.pkl"):
            print("SAVING DATA...")
            with open(os.path.join(self.root, "osm_traces.pkl"), "wb") as f: #ghi dữ liệu nhị phân
                pickle.dump(self.data, f)

    def plot(self, file_name):
        print(f"PLOTTING {len(self.data)} TRAJECTORIES...")
        plots = []
        for traj in self.data.trajectories:
            plots.append(traj.hvplot(
                c="speed", line_width=2, cmap="plasma", clabel="Speed", tiles="CartoLight" if len(plots) == 0 else None
            ))
        hv.save(hv.Overlay(plots), file_name)
