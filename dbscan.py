import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


class InteractiveClusterVisualizer:
    def __init__(self, epsilon=0.5, min_pts=5):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.data_points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.click_handler = self.fig.canvas.mpl_connect('button_press_event', self.handle_click)
        self.key_handler = self.fig.canvas.mpl_connect('key_press_event', self.handle_key)

        self.ax.set_title("DBSCAN Clustering Interactive Tool")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        plt.tight_layout()
        plt.show()

    def handle_click(self, event):
        if event.inaxes != self.ax:
            return

        num_points = 3
        for _ in range(num_points):
            x = event.xdata + random.uniform(-0.2, 0.2)
            y = event.ydata + random.uniform(-0.2, 0.2)
            self.data_points.append([x, y])
            self.ax.scatter(x, y, c='blue', edgecolor='black', alpha=0.8)

        self.fig.canvas.draw_idle()

    def handle_key(self, event):
        if event.key == 'enter':
            self.execute_clustering()

    def execute_clustering(self):
        if not self.data_points:
            print("No data points available for clustering!")
            return

        self.fig.canvas.mpl_disconnect(self.click_handler)
        self.fig.canvas.mpl_disconnect(self.key_handler)

        points_array = np.array(self.data_points)
        cluster_algorithm = CustomDBSCAN(eps=self.epsilon, min_samples=self.min_pts)
        cluster_labels = cluster_algorithm.fit(points_array)

        self.visualize_results(points_array, cluster_labels)

        cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_count = list(cluster_labels).count(-1)
        print(f"Identified clusters: {cluster_count}")
        print(f"Noise points detected: {noise_count}")

    def visualize_results(self, points, labels):
        self.ax.clear()
        unique_labels = set(labels)

        color_palette = plt.cm.tab10
        for label in unique_labels:
            if label == -1:
                color = [0.1, 0.1, 0.1, 1.0]
                marker = 'x'
            else:
                color = color_palette(label / len(unique_labels))
                marker = 'o'

            mask = (labels == label)
            self.ax.scatter(points[mask, 0], points[mask, 1],
                            color=color, marker=marker, s=70,
                            edgecolor='black',
                            label=f'Cluster {label}' if label != -1 else 'Noise')

        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("DBSCAN Clustering Result")
        self.fig.canvas.draw()


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, data):
        self.data = data
        n_points = data.shape[0]
        self.labels = np.zeros(n_points, dtype=int)
        current_cluster = 0

        for idx in range(n_points):
            if self.labels[idx] != 0:
                continue

            neighbors = self.find_neighbors(idx)

            if len(neighbors) < self.min_samples:
                self.labels[idx] = -1
                continue

            current_cluster += 1
            self.labels[idx] = current_cluster
            queue = deque(neighbors)

            while queue:
                current_idx = queue.popleft()

                if self.labels[current_idx] == -1:
                    self.labels[current_idx] = current_cluster

                if self.labels[current_idx] != 0:
                    continue

                self.labels[current_idx] = current_cluster
                new_neighbors = self.find_neighbors(current_idx)

                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)

        return self.labels

    def find_neighbors(self, point_idx):
        distances = np.linalg.norm(self.data - self.data[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()


def main():
    print("DBSCAN Clustering Interactive Tool")
    print("Instructions:")
    print("1. Click on the plot to add data points")
    print("2. Press ENTER to perform clustering")

    InteractiveClusterVisualizer(epsilon=0.5, min_pts=5)


if __name__ == '__main__':
    main()
