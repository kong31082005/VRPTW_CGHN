import pandas as pd
import numpy as np
from collections import namedtuple 
import os
# Warehouse = namedtuple('Warehouse',['xy_coord','dueDate'])
Customer = namedtuple('Customer',['id','xy_coord','demand','readyTime','dueTime','serviceTime'])

def load_csv_dataset(url=None,name_of_id=None, number_of_customer: int = None):
  # folder = file_name[0:2]
  # url = "Python\\Genetic_Algorithm\\data\\csv\\" + folder +"\\"+ name_of_id+ ".csv"

  path = os.path.join(url, name_of_id)
  f = pd.read_csv(path)
  print("dataset: ",name_of_id[0:4])
  print("Number of customer: ",number_of_customer)
  # data_kmeans = np.delete(np.array(f[['XCOORD.','YCOORD.']]),0,0)   
  data = np.array(f[['XCOORD.','YCOORD.']])
  # data = np.array(f[['XCOORD.','YCOORD.','DUE DATE']])

  demands = np.array(f['DEMAND'])
  ready_time = np.array(f['READY TIME'])
  due_date =  np.array(f['DUE DATE'])
  service_time =  np.array(f['SERVICE TIME'])

  customers = []
  for i in range(number_of_customer+1):
    # customers.append(Customer(i-1,data[i],demands[i],ready_time[i],due_date[i],service_time[i]))
    customers.append(Customer(i,data[i],demands[i],ready_time[i],due_date[i],service_time[i]))
  # warehouse = Warehouse(data[0],due_date[0])
  
  # return data,customers,warehouse
  return data[:number_of_customer+1],customers

# print(load_csv_dataset(url="data/csv/R1/",name_of_id="R102.csv",number_of_customer=25))

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

