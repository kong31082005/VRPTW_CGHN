import matplotlib.pyplot as plt
import numpy as np

from algorithm.GeneticAlgorithm import GA
from algorithm.kmeans import Kmeans
from ultility.readDataFile import load_txt_dataset


# ============================================================
# CẤU HÌNH
# ============================================================

DATA_ID = "C101"
DATA_NUMBER_CUS = "100"

# K-Means
N_CLUSTER = 10
MAX_ITER = 1000
TIME_WEIGHT = 0.3

# GA
M = 0
INDIVIDUAL = 100
GENERATION = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
CONSERVE_RATE = 0.3

# Chọn biểu đồ muốn chạy
PLOT_KMEANS = False
PLOT_GA_CONVERGENCE = True


# ============================================================
# 1. VẼ K-MEANS
# ============================================================

def plot_clusters(
    data,
    labels,
    dataset_name
):
    depot = data[0]
    customers = data[1:]

    plt.figure(figsize=(8, 6))

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id

        plt.scatter(
            customers[mask, 0],
            customers[mask, 1],
            s=35,
            label=f"Cluster {cluster_id + 1}"
        )

    # Depot
    plt.scatter(
        depot[0],
        depot[1],
        marker="*",
        s=180,
        label="Depot"
    )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.title(
        f"Spatio-temporal K-Means Clustering - "
        f"{dataset_name}"
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. VẼ GA CONVERGENCE
# ============================================================

def plot_ga_convergence(
    ga_convergence,
    dataset_name
):
    if not ga_convergence:
        print("Không có dữ liệu GA convergence.")
        return

    generations = [
        item["generation"]
        for item in ga_convergence
    ]

    route_counts = [
        item["route_count"]
        for item in ga_convergence
    ]

    distances = [
        item["distance"]
        for item in ga_convergence
    ]

    # --------------------------------------------------------
    # 2.1. Route Count
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        generations,
        route_counts,
        linewidth=2
    )

    plt.xlabel("Generation")
    plt.ylabel("Best Number of Routes")

    plt.title(
        f"GA Convergence - Route Count - "
        f"{dataset_name}"
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 2.2. Distance
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        generations,
        distances,
        linewidth=2
    )

    plt.xlabel("Generation")
    plt.ylabel("Best Distance")

    plt.title(
        f"GA Convergence - Distance - "
        f"{dataset_name}"
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Đọc dữ liệu Solomon
    # --------------------------------------------------------

    url_data = (
        "data/txt/"
        + DATA_NUMBER_CUS
        + "/"
        + DATA_ID[:-2]
        + "/"
    )

    data, customers, vehicle_number, vehicle_capacity = (
        load_txt_dataset(
            url=url_data,
            name_of_id=DATA_ID + ".txt"
        )
    )

    print(f"Dataset: {DATA_ID}")
    print(f"Customers: {len(customers) - 1}")
    print(f"Vehicle limit: {vehicle_number}")
    print(f"Vehicle capacity: {vehicle_capacity}")

    # --------------------------------------------------------
    # 2. Chuẩn bị dữ liệu Spatio-temporal K-Means
    # --------------------------------------------------------

    customer_coords = np.delete(
        data,
        0,
        axis=0
    )

    time_features = np.array([
        (
            customer.readyTime
            + customer.dueTime
        ) / 2.0
        for customer in customers[1:]
    ])

    data_kmeans_spatiotemporal = np.column_stack((
        customer_coords,
        time_features
    ))

    # --------------------------------------------------------
    # 3. Chạy K-Means
    # --------------------------------------------------------

    kmeans = Kmeans(
        maxiter=MAX_ITER,
        n_cluster=N_CLUSTER
    )

    labels, _, _ = (
        kmeans.k_means_spatiotemporal(
            data_kmeans_spatiotemporal,
            time_weight=TIME_WEIGHT
        )
    )

    clusters = kmeans.data_to_cluster(
        labels
    )

    print(
        "Cluster sizes:",
        [len(cluster) for cluster in clusters]
    )

    # --------------------------------------------------------
    # 4. Vẽ K-Means
    # --------------------------------------------------------

    if PLOT_KMEANS:
        plot_clusters(
            data=data,
            labels=labels,
            dataset_name=DATA_ID
        )

    # --------------------------------------------------------
    # 5. Chạy GA và vẽ convergence
    # --------------------------------------------------------

    if PLOT_GA_CONVERGENCE:

        ga = GA(
            individual=INDIVIDUAL,
            generation=GENERATION,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            vehicle_capacity=vehicle_capacity,
            vehicle_number=vehicle_number,
            conserve_rate=CONSERVE_RATE,
            M=M,
            customers=customers
        )

        ga.fit_allClusters(
            clusters,
            use_recluster=False
        )

        plot_ga_convergence(
            ga.ga_convergence,
            DATA_ID
        )