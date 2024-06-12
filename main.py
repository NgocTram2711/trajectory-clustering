import os
import holoviews as hv
import hvplot.pandas
import numpy as np

from holoviews import opts
from itertools import product

from src.evaluation import Evaluation
from src.solutions import Dbscan, Tca
from src.traces import OSMTraces


def main():
    bbox_subfolder = "_".join([str(b) for b in bbox]).replace(".", ",")
    images_folder = os.path.join("images", bbox_subfolder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    results_folder = os.path.join("results", bbox_subfolder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    osm_traces = OSMTraces(os.path.join("data", bbox_subfolder), bbox, download=True)
    # osm_traces.plot(f"{images_folder}/step_2.png")

    tca_grid = [(min_d, max_d) for min_d, max_d in product(range(50, 450, 50), range(50, 450, 50)) if max_d > min_d]
    tca_evaluation = Evaluation(results_folder, osm_traces, Tca, tca_grid, "tca.pkl")
    tca_evaluation.plot(f"{images_folder}/tca.png")
    best_index = np.argmax(tca_evaluation.scores, axis=0)[0]
    min_d, max_d = tca_evaluation.grid[best_index]
    tca = Tca(results_folder, osm_traces, min_d, max_d, should_cluster=False)
    tca.plot(f"{images_folder}/tca_{min_d}_{max_d}.png")
    tca.cluster_points()
    tca.plot(f"{images_folder}/tca_{min_d}_{max_d}_t.png", mode="trajectories")
    # tca.plot(f"{images_folder}/tca_{min_d}_{max_d}_p.png", mode="points")

    dbscan_grid = list(product(np.linspace(1e-4, 1e-3, 14), np.arange(5, 21, 5)))
    dbscan_evaluation = Evaluation(results_folder, osm_traces, Dbscan, dbscan_grid, "dbscan.pkl")
    dbscan_evaluation.plot(f"{images_folder}/dbscan.png", 15)
    best_index = np.argmax(dbscan_evaluation.scores, axis=0)[0]
    eps, min_samples = dbscan_evaluation.grid[best_index]
    dbscan = Dbscan(results_folder, osm_traces, eps, min_samples)
    dbscan.plot(f"{images_folder}/dbscan_{eps:.5f}_{min_samples}_t.png", mode="trajectories")
    # dbscan.plot(f"{images_folder}/dbscan_{eps:.5f}_{min_samples}_p.png", mode="points")


if __name__ == "__main__":
    x0, x1 = 14.7435, 14.7642
    y0, y1 = 44.9661, 44.9738
    bbox = (x0, y0, x1, y1)
    opts.defaults(opts.Overlay(frame_width=765, frame_height=522, fontscale=2))
    hv.extension("bokeh")
    main()
