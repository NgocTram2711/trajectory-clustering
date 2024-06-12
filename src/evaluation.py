import os
import pickle

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class Evaluation:
    def __init__(self, root, traj_collection, solution_class, grid, file_name):
        self.root = root
        self.traj_collection = traj_collection
        self.SolutionClass = solution_class
        self.grid = grid
        self.file_name = file_name
        self.scores = []

        self.grid_search()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.file_name))

    def save(self):
        print("SAVING EVALUATION...")
        with open(os.path.join(self.root, self.file_name), "wb") as f:
            pickle.dump({"grid": self.grid, "scores": self.scores}, f)

    def grid_search(self):
        if self._check_exists():
            print(f"LOADING RESULTS...")
            with open(os.path.join(self.root, self.file_name), "rb") as f:
                results = pickle.load(f)
                self.scores = results.get("scores")
                return

        scores = []
        for i, (p1, p2) in enumerate(self.grid):
            solution = self.SolutionClass(self.root, self.traj_collection, p1, p2)
            solution.solve()
            clusters = set(solution.labels_)
            if len(clusters) == 1 or (len(clusters) == 2 and -1 in clusters) or len(clusters) > 100:
                print("POOR SOLUTION")
                scores.append(None)
                continue
            scores.append((
                silhouette_score(solution.X, solution.labels_),
                calinski_harabasz_score(solution.X, solution.labels_)
            ))
            del solution
        self.scores = scores
        self.save()

    def plot(self, file_name, nticks=None):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 20
        plt.rcParams["axes.edgecolor"] = "#3d3d3d"
        plt.rcParams["axes.facecolor"] = "#fafaf8"
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["xtick.color"] = "#3d3d3d"
        plt.rcParams["ytick.labelsize"] = 16
        plt.rcParams["ytick.color"] = "#3d3d3d"

        if nticks is None:
            param_labels = [f"{p[0]}_{p[1]}" for p in self.grid]
        else:
            param_labels = [f"{p[0]:.5f}_{p[1]}" for p in self.grid]
        score1 = [s[0] for s in self.scores]
        score2 = [s[1] for s in self.scores]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(param_labels, score1, color="b")
        ax1.set_ylabel("Silhouette score", color="b", labelpad=15, fontdict={"style": "italic"})
        plt.xticks(rotation=45)
        if nticks is not None:
            ax1.set_xlim(-1, len(param_labels))
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=nticks, integer=True, prune="both"))

        ax2 = ax1.twinx()
        ax2.plot(param_labels, score2, color="r")
        ax2.set_ylabel("Calinski-Harabasz index", color="r", labelpad=20, fontdict={"style": "italic"})

        plt.tight_layout()
        plt.savefig(file_name)
