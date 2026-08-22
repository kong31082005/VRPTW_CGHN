import numpy as np
from ultility.utilities import distance_cdist


# ===============================
# Thuật toán K-Means
# dùng để phân cụm khách hàng
# trước khi tối ưu tuyến
# ===============================

class Kmeans():

    def __init__(
        self,
        epsilon: float = 1e-5,
        maxiter: int = 1000,
        n_cluster: int = 3
    ):
        self.__epsilon = epsilon
        self.__maxiter = maxiter
        self.n_cluster = n_cluster

    # ===============================
    # Khởi tạo tâm cụm
    # ===============================
    def __init__cluster_center(
        self,
        X: np.ndarray,
        seed: int = 0
    ) -> np.ndarray:

        if seed > 0:
            np.random.seed(seed)

        return X[
            np.random.choice(
                X.shape[0],
                self.n_cluster,
                replace=False
            )
        ]

    # ===============================
    # Cập nhật tâm cụm
    # ===============================
    def update_cluster_center(
        self,
        X: np.ndarray,
        label: np.ndarray
    ) -> np.ndarray:

        cluster_centers = [
            np.mean(
                X[label == i],
                axis=0
            )
            for i in range(
                self.n_cluster
            )
        ]

        return np.array(
            cluster_centers
        )

    # ===============================
    # Cập nhật nhãn cụm
    # ===============================
    def update_membership_matrix(
        self,
        X: np.ndarray,
        V: np.ndarray
    ) -> np.ndarray:

        labels = np.argmin(
            distance_cdist(
                X,
                V
            ),
            axis=1
        )

        return np.array(labels)

    # ===============================
    # Chuyển label thành danh sách cụm
    # ===============================
    def label_to_cluster(
        self,
        labels: np.ndarray
    ):

        return [
            np.argwhere(
                labels == i
            ).flatten()
            for i in range(
                self.n_cluster
            )
        ]

    # ===============================
    # Kiểm tra hội tụ
    # ===============================
    def has_converged(
        self,
        centers,
        new_centers
    ):

        return set(
            tuple(a)
            for a in centers
        ) == set(
            tuple(a)
            for a in new_centers
        )

    # ===============================
    # K-Means tự cài đặt
    # ===============================
    def k_means(
        self,
        X: np.ndarray,
        seed: int = 42
    ) -> tuple:

        v = self.__init__cluster_center(
            X,
            seed
        )

        for step in range(
            self.__maxiter
        ):

            old_v = v.copy()

            u = self.update_membership_matrix(
                X,
                old_v
            )

            v = self.update_cluster_center(
                X,
                u
            )

            if self.has_converged(
                old_v,
                v
            ):
                break

        return (
            u,
            v,
            step + 1
        )

    # ===============================
    # K-Means cũ + sắp cụm
    # ===============================
    def k_means_sorted(
        self,
        X: np.ndarray,
        FirstPoint: np.ndarray,
        seed: int = 42
    ) -> tuple:

        u, v, step = self.k_means(
            X=X,
            seed=seed
        )

        v = self.sort_cluster_by_distance(
            V=v,
            FirstPoint=FirstPoint
        )

        u = self.update_membership_matrix(
            X,
            v
        )

        return (
            u,
            v,
            step
        )

    # ===============================
    # K-Means sklearn cũ
    # chỉ sử dụng tọa độ không gian
    # ===============================
    def k_means_lib_sorted(
        self,
        X: np.ndarray,
        FirstPoint: np.ndarray,
        seed: int = 42
    ) -> tuple:

        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.n_cluster,
            random_state=seed,
            n_init=10
        )

        kmeans.fit(X)

        centroid = (
            kmeans.cluster_centers_
        )

        v = self.sort_cluster_by_distance(
            V=centroid,
            FirstPoint=FirstPoint
        )

        u = self.update_membership_matrix(
            X,
            v
        )

        return (
            u,
            v,
            1
        )

    # ========================================================
    # K-Means không gian - thời gian
    #
    # Input X:
    # [
    #     x,
    #     y,
    #     time_feature
    # ]
    #
    # time_weight:
    # hệ số điều chỉnh mức ảnh hưởng
    # của đặc trưng thời gian.
    #
    # time_weight = 0:
    # bỏ ảnh hưởng thời gian.
    #
    # time_weight = 1:
    # giữ nguyên đặc trưng thời gian
    # sau khi chuẩn hóa.
    # ========================================================
    def k_means_spatiotemporal(
        self,
        X: np.ndarray,
        time_weight: float = 0.3,
        seed: int = 42
    ) -> tuple:

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler

        # ===============================
        # Kiểm tra dữ liệu đầu vào
        # ===============================
        if X.ndim != 2:
            raise ValueError(
                "Dữ liệu K-Means phải là ma trận 2 chiều."
            )

        if X.shape[1] != 3:
            raise ValueError(
                "K-Means không gian - thời gian "
                "yêu cầu dữ liệu [x, y, time]."
            )

        if not 0 <= time_weight <= 1:
            raise ValueError(
                "time_weight phải nằm trong khoảng [0, 1]."
            )

        # ===============================
        # Bước 1:
        # Chuẩn hóa x, y, time về [0, 1]
        # ===============================
        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        # ===============================
        # Bước 2:
        # Điều chỉnh mức ảnh hưởng
        # của thành phần thời gian
        # ===============================
        X_scaled[:, 2] = (
            X_scaled[:, 2]
            * time_weight
        )

        # ===============================
        # Bước 3:
        # Chạy K-Means
        # ===============================
        kmeans = KMeans(
            n_clusters=self.n_cluster,
            random_state=seed,
            n_init=10
        )

        labels = kmeans.fit_predict(
            X_scaled
        )

        centroids = (
            kmeans.cluster_centers_
        )

        return (
            labels,
            centroids,
            1
        )

    # ===============================
    # Chuyển nhãn K-Means
    # thành ID customer cho GA
    #
    # K-Means:
    # index 0 -> customer 1
    #
    # GA:
    # customer 1 -> id 1
    #
    # nên phải +1
    # ===============================
    def data_to_cluster(
        self,
        U: np.ndarray
    ):

        clusters = []

        for i in range(
            self.n_cluster
        ):

            indices = np.where(
                U == i
            )[0]

            customer_ids = (
                indices + 1
            )

            clusters.append(
                customer_ids
            )

        return clusters

    # ===============================
    # Sắp tâm cụm theo khoảng cách
    # ===============================
    def sort_cluster_by_distance(
        self,
        V: np.ndarray,
        FirstPoint: np.ndarray
    ) -> np.ndarray:

        sorted_V = []

        used_indices = set()

        current_point = (
            FirstPoint
        )

        for _ in range(
            len(V)
        ):

            distances = np.linalg.norm(
                current_point - V,
                axis=1
            )

            valid_indices = [
                i
                for i in range(len(V))
                if i not in used_indices
            ]

            nearest_index = (
                valid_indices[
                    np.argmin(
                        distances[
                            valid_indices
                        ]
                    )
                ]
            )

            sorted_V.append(
                V[nearest_index]
            )

            current_point = (
                V[nearest_index]
            )

            used_indices.add(
                nearest_index
            )

        return np.array(
            sorted_V
        )

    # ===============================
    # Hàm cũ - giữ để tránh ảnh hưởng
    # các phần khác của project
    # ===============================
    def sort_cluster_by_len(
        self,
        V: np.ndarray,
        FirstPoint: np.ndarray
    ) -> np.ndarray:

        V_len = [
            len(v)
            for v in V
        ]

        sorted_indices = (
            np.argsort(V_len)
        )

        sorted_V = np.zeros(
            (
                len(V),
                V.shape[1]
            )
        )

        for idx, val in enumerate(
            sorted_indices
        ):
            sorted_V[idx] = (
                V[val]
            )

        return sorted_V

    # ===============================
    # Elbow Method - bản tự cài đặt
    # ===============================
    def elbow_k_means(
        self,
        X: np.ndarray,
        C_scope
    ):

        import matplotlib.pyplot as plt

        distortions = []

        for k in C_scope:

            self.n_cluster = k

            u, v, step = self.k_means(
                X=X
            )

            distortions.append(
                np.sum(
                    np.min(
                        distance_cdist(
                            X,
                            v
                        ),
                        axis=1
                    ) ** 2
                )
            )

        plt.figure(
            figsize=(10, 10)
        )

        plt.plot(
            C_scope,
            distortions,
            "bx-"
        )

        plt.xlabel("k")
        plt.ylabel("Distortions")
        plt.grid(True)

        plt.show()

    # ===============================
    # Elbow Method - sklearn
    # ===============================
    def elbow_k_means_lib(
        self,
        X: np.ndarray,
        C_scope
    ):

        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        inertia = []

        for k in C_scope:

            kmeans = KMeans(
                n_clusters=k,
                random_state=0,
                n_init=10
            )

            kmeans.fit(X)

            inertia.append(
                kmeans.inertia_
            )

        plt.plot(
            C_scope,
            inertia
        )

        plt.xlabel(
            "Số lượng cụm (k)"
        )

        plt.ylabel(
            "Inertia"
        )

        plt.title(
            "Elbow Method"
        )

        plt.grid(True)

        plt.show()