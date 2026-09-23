import numpy as np

# Tính khoảng cách từ 1 điểm đến tất cả các điểm còn lại bằng euclidean


def distance_cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)

# Làm tròn số
def round_float(number: float, n: int = 2) -> float:
    if n == 0:
        return int(number)
    return round(number, n)
# Biểu diễn các điểm lên


def visualize_clusters(data, labels, centers):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.scatter(data[0, 0], data[0, 1], c='black',
                marker='o', s=200, label='Starter')
    # Xóa hàng đầu tiên
    data = np.delete(data, 0, 0)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1], label=f'Cluster {label}')
    # Convert centers to a NumPy array for proper indexing
    centers_array = np.array(centers)
    
    plt.scatter(centers_array[:, 0], centers_array[:, 1],
                c='red', marker='x', s=200, label='Centers')
    plt.legend()
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def write_excel_file(
    data_excels,
    data_files,
    data_name,
    run_time,
    title_names,
    fileio
):
    from xlsxwriter import Workbook
    import numpy as np

    workbook = Workbook(fileio)

    worksheet = workbook.add_worksheet(
        data_name + "GA"
    )

    title_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_size': 14
    })

    header_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center'
    })

    cell_format = workbook.add_format({
        'border': 1
    })

    total_columns = (
        len(data_files) * len(title_names)
    )

    char_data_end = (
        (total_columns // 26) * "A"
        + chr(
            ord('A')
            + total_columns % 26
        )
    )

    worksheet.merge_range(
        f'A1:{char_data_end}1',
        'Kết quả chạy thử bộ dữ liệu bằng GA',
        title_format
    )

    # ===============================
    # Tên từng dataset
    # ===============================
    for idx_d, data_file in enumerate(data_files):

        start_column = (
            idx_d * len(title_names) + 1
        )

        end_column = (
            (idx_d + 1) * len(title_names)
        )

        char_start = (
            (start_column // 26) * "A"
            + chr(
                ord('A')
                + start_column % 26
            )
        )

        char_end = (
            (end_column // 26) * "A"
            + chr(
                ord('A')
                + end_column % 26
            )
        )

        worksheet.merge_range(
            f'{char_start}2:{char_end}2',
            data_file[:-4],
            title_format
        )

    # ===============================
    # Header
    # ===============================
    titles = (
        ['Lần chạy']
        + title_names * len(data_files)
    )

    for idx_t, title in enumerate(titles):
        worksheet.write(
            2,
            idx_t,
            title,
            header_format
        )

    # ===============================
    # Chuẩn hóa dữ liệu
    # ===============================
    data = np.array(data_excels)

    data = np.reshape(
        data,
        (
            len(data_files),
            run_time,
            len(title_names)
        )
    )

    # ===============================
    # Xác định số lần chạy đã hoàn thành
    # ===============================
    completed_runs = 0

    for run in range(run_time):

        if np.any(
            ~np.isnan(data[:, run, :])
        ):
            completed_runs = run + 1

    # ===============================
    # Ghi các lần chạy đã hoàn thành
    # ===============================
    for run in range(completed_runs):

        row_values = [run + 1]

        for idx_d in range(len(data_files)):

            values = data[idx_d, run]

            for value in values:

                if np.isnan(value):
                    row_values.append("")
                else:
                    row_values.append(value)

        for column, value in enumerate(row_values):

            worksheet.write(
                run + 3,
                column,
                value,
                cell_format
            )

    # ===============================
    # Tính trung bình từng dataset
    # ===============================
    average_row = ["TB"]

    for idx_d in range(len(data_files)):

        dataset_data = data[idx_d]

        # Chỉ lấy những run đã hoàn thành
        valid_mask = ~np.isnan(
            dataset_data
        ).all(axis=1)

        valid_data = dataset_data[
            valid_mask
        ]

        if len(valid_data) > 0:

            mean_values = np.mean(
                valid_data,
                axis=0
            )

            average_row.extend(
                mean_values.tolist()
            )

        else:

            average_row.extend(
                [""] * len(title_names)
            )

    # ===============================
    # Ghi dòng trung bình
    # ===============================
    average_excel_row = (
        completed_runs + 3
    )

    for column, value in enumerate(average_row):

        worksheet.write(
            average_excel_row,
            column,
            value,
            cell_format
        )

    workbook.close()