import numpy as np
from collections import namedtuple 
import os

Customer = namedtuple('Customer',['id','xy_coord','demand','readyTime','dueTime','serviceTime'])

def load_txt_dataset(url=None, name_of_id=None):
    path = os.path.join(url, name_of_id)

    with open(path, 'r') as file:
        lines = file.readlines()

    # ==========================================
    # 1. Đọc thông tin VEHICLE
    # ==========================================
    vehicle_number = None
    vehicle_capacity = None

    for i, line in enumerate(lines):
        if "NUMBER" in line and "CAPACITY" in line:
            vehicle_data = lines[i + 1].strip().split()

            vehicle_number = int(vehicle_data[0])
            vehicle_capacity = int(vehicle_data[1])
            break

    if vehicle_number is None or vehicle_capacity is None:
        raise ValueError(
            f"Không đọc được NUMBER/CAPACITY từ {name_of_id}"
        )

    # ==========================================
    # 2. Tìm vị trí bắt đầu CUSTOMER
    # ==========================================
    customer_start = None

    for i, line in enumerate(lines):
        if "CUST NO." in line:
            customer_start = i + 1
            break

    if customer_start is None:
        raise ValueError(
            f"Không tìm thấy dữ liệu CUSTOMER trong {name_of_id}"
        )

    # ==========================================
    # 3. Đọc dữ liệu khách hàng
    # ==========================================
    customers = []
    cord_data = []

    customer_id = 0

    for line in lines[customer_start:]:
        data = line.strip().split()

        # Bỏ qua dòng trống
        if len(data) < 7:
            continue

        try:
            file_customer_id = int(data[0])
            xcoord = np.int64(data[1])
            ycoord = np.int64(data[2])
            demand = np.int64(data[3])
            ready_time = np.int64(data[4])
            due_date = np.int64(data[5])
            service_time = np.int64(data[6])
        except ValueError:
            continue

        # Solomon dùng ID 0, 1, 2, ...
        if file_customer_id != customer_id:
            raise ValueError(
                f"Customer ID không liên tục: "
                f"mong đợi {customer_id}, "
                f"nhận được {file_customer_id}"
            )

        cord_data.append(
            [xcoord, ycoord]
        )

        customers.append(
            Customer(
                file_customer_id,
                np.array([xcoord, ycoord]),
                demand,
                ready_time,
                due_date,
                service_time
            )
        )

        customer_id += 1

    # ==========================================
    # 4. Kiểm tra dữ liệu
    # ==========================================
    if len(customers) == 0:
        raise ValueError(
            f"Không đọc được khách hàng từ {name_of_id}"
        )

    # ==========================================
    # 5. Trả kết quả
    # ==========================================
    return (
        np.array(cord_data),
        customers,
        vehicle_number,
        vehicle_capacity
    )

