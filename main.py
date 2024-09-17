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
    images_folder = os.path.join("images1", bbox_subfolder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    results_folder = os.path.join("results1", bbox_subfolder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    osm_traces = OSMTraces(os.path.join("data", bbox_subfolder), bbox, download=True)
    # osm_traces.plot(f"{images_folder}/step_2.png")

    tca_grid = [(min_d, max_d) for min_d, max_d in product(range(50, 450, 50), range(50, 450, 50)) if max_d > min_d]
    tca_evaluation = Evaluation(results_folder, osm_traces, Tca, tca_grid, "tca.pkl")
    tca_evaluation.plot(f"{images_folder}/tca.png")
    best_index = np.argmax(tca_evaluation.scores, axis=0)[0]
    print("tca_evaluation.scores shape", len(tca_evaluation.scores))
    print("tca_evaluation.scores", tca_evaluation.scores)
    # summax_index = np.argmax(np.sum(tca_evaluation.scores, axis=1))
    min_d, max_d = tca_evaluation.grid[best_index]
    print("min_d", min_d)
    print("max_d", max_d)
    # print("tca_evaluation.scores", tca_evaluation.scores[best_index])
    tca = Tca(results_folder, osm_traces, min_d, max_d, should_cluster=False)
    tca.plot(f"{images_folder}/tca_{min_d}_{max_d}.png")
    tca.cluster_points()
    tca.plot(f"{images_folder}/tca_{min_d}_{max_d}_t.png", mode="trajectories")
    # tca.plot(f"{images_folder}/tca_{min_d}_{max_d}_p.png", mode="points")

    dbscan_grid = list(product(np.linspace(1e-4, 1e-3, 14), np.arange(5, 21, 5)))
    dbscan_evaluation = Evaluation(results_folder, osm_traces, Dbscan, dbscan_grid, "dbscan.pkl")
    dbscan_evaluation.plot(f"{images_folder}/dbscan.png", 15)
    print("dbscan_evaluation.scores shape", len(dbscan_evaluation.scores))
    print("dbscan_evaluation.scores", dbscan_evaluation.scores)
    best_index = np.argmax(dbscan_evaluation.scores, axis=0)[0]
    # summax_index = np.argmax(np.sum(dbscan_evaluation.scores, axis=1))
    eps, min_samples = dbscan_evaluation.grid[best_index]
    dbscan = Dbscan(results_folder, osm_traces, eps, min_samples)
    dbscan.plot(f"{images_folder}/dbscan_{eps:.5f}_{min_samples}_t.png", mode="trajectories")
    # dbscan.plot(f"{images_folder}/dbscan_{eps:.5f}_{min_samples}_p.png", mode="points")


if __name__ == "__main__":
    # x0, x1 = 14.7435, 14.7642  [[[106.6963300879,10.784427208],[106.7080137904,10.784427208],[106.7080137904,10.789949774],[106.6963300879,10.789949774],[106.6963300879,10.784427208]]]
    # y0, y1 = 44.9661, 44.9738 106.69633,10.7844,106.7080,10.7899

    #[[[106.6926232751,10.7831414052],[106.7091349538,10.7831414052],[106.7091349538,10.7910353047],[106.6926232751,10.7910353047],[106.6926232751,10.7831414052]]]
    #106.6926,10.7831,106.7091,10.7910
#[[[106.6998890404,10.7882031027],[106.7189005379,10.7882031027],[106.7189005379,10.7989529073],[106.6998890404,10.7989529073],[106.6998890404,10.7882031027]]]
    #[[[106.705275255,10.7982993007],[106.7185253675,10.7982993007],[106.7185253675,10.8036108385],[106.705275255,10.8036108385],[106.705275255,10.7982993007]]]
    #106.7052,10.7982,106.71852,10.8036
    x0, x1 = 106.7052,106.7185210
    y0, y1 =  10.7982,10.8036
    bbox = (x0, y0, x1, y1) 
    opts.defaults(opts.Overlay(frame_width=765, frame_height=522, fontscale=2))
    hv.extension("bokeh")
    main() 
