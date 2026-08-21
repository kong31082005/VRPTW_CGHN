import numpy as np
from algorithm.AntColonyOptimization import ACO
from algorithm.GeneticAlgorithm import GA
from algorithm.ParticleSwarmOptimization import PSO
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
from ultility.utilities import round_float, write_excel_file
import signal
import logging
import os
import time
import sys

if __name__ == "__main__":
    _start_time_all = time.time()
    # ===============================Thiết lập thuật toán chạy===============================
    algorithms = ['GA']  # 'GA','PSO','ACO',
    # ===============================Thiết lập các thông số===============================
    # Thông số VRPTW
    VEHCICLE_CAPACITY = 200  # Trọng tải tối đa của 1 xe
    M = 0  # Sai số trong cửa sổ thời gian
    # Thông số chạy K-means
    N_CLUSTER = 10  # Khuyến nghị: Giảm từ 20 xuống 8 cụm cho 100 khách hàng để GA tối ưu tốt hơn
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100  # Số lượng khách hàng
    # Thông số chạy GA (Đã chỉnh lại tham số chuẩn để hội tụ tốt)
    INDIVIDUAL = 100  
    GENERATION = 100  
    CROSSOVER_RATE = 0.8  
    MUTATION_RATE = 0.15  
    CONSERVE_RATE = 0.3 
    # Thông số chạy PSO
    NUM_PARTICLES = 150  
    MAX_ITER_PSO = 150  
    W = 0.8  
    C1 = 0.15  
    C2 = 0.1  
    # Thông số chạy ACO
    ALFA = 2  
    BETA = 5  
    SIGM = 3  
    RO = 0.8  
    TH = 80  
    NUM_ANTS = 50  
    MAX_ITER_ACO = 100  
    # Thông số bộ dữ liệu chạy
    TITLE_NAMES = ['Route', 'Distance', 'Fitness', 'RunTime']
    DATA_ID = None  # File dữ liệu cụ thể
    DATA_NAME = "C1"  # Bộ dữ liệu
    DATA_NUMBER_CUS = "100" 
    RUN_TIMES = 10
    EXCEL_FILE = None  
    FILE_EXCEL_PATH = "result/"
    FILE_NAME = "_Test" 

    # ===============================Thiết lập gọi thuật toán tương ứng===============================
    def signal_handler(sig, frame):
        """Hàm xử lý khi nhận tín hiệu dừng chương trình"""
        print("\nChương trình bị dừng đột ngột, đang cố gắng lưu dữ liệu...")
        try:
            if (DATA_NAME != None):
                write_excel_file(data_excels=data_excels, data_files=data_files, data_name=DATA_NAME, run_time=RUN_TIMES,
                                algorithms=algorithms, title_names=TITLE_NAMES, fileio=EXCEL_FILE)
        except Exception as e:
            print(f"Không thể ghi file khi dừng khẩn cấp: {str(e)}")
        exit(0)

    if (DATA_ID != None):
        url_data = "data/txt/"+DATA_NUMBER_CUS+"/"+DATA_ID[:-2]+"/"
        data_files = [DATA_ID+".txt"]
        EXCEL_FILE = FILE_EXCEL_PATH + DATA_ID + FILE_NAME + ".xlsx"
    else:
        url_data = "data/txt/"+DATA_NUMBER_CUS+"/"+DATA_NAME+"/"
        data_files = sorted(
            [f for f in os.listdir(url_data) if f.endswith(('.txt'))])
        EXCEL_FILE = FILE_EXCEL_PATH + DATA_NAME + FILE_NAME + ".xlsx"

    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(filename='terminal_output.log', level=logging.INFO, filemode='w', encoding='utf-8')

    data_excels = np.zeros((len(algorithms), len(data_files), RUN_TIMES, len(TITLE_NAMES)))
    
    try:
        # Thiết lập chạy các thuật toán
        for idx_al, algorithm in enumerate(algorithms):
            len_data = len(data_files)
            logging.info(f"Bộ dữ liệu {DATA_NAME}: {data_files}")
            
            # Khởi tạo biến tích lũy tổng cho cả bộ dữ liệu
            run_time_data = 0
            route_count_data = 0
            distance_data = 0
            fitness_data = 0
            
            for idx_dat, data_file in enumerate(data_files):
                run_time_mean = 0
                route_count_mean = 0
                distance_mean = 0
                fitness_mean = 0
                
                _start_time = time.time()
                # Khởi tạo dữ liệu
                data, customers = load_txt_dataset(url=url_data, name_of_id=data_file)
                logging.info(f"Thời gian lấy dữ liệu: {round_float(time.time() - _start_time)}")

                warehouse = data[0]
                data_kmeans = np.delete(data, 0, 0)
            
                # chạy kmeans với thuật toán
                logging.info("#K-means =============================")
                kmeans = Kmeans(epsilon=EPSILON, maxiter=MAX_ITER, n_cluster=N_CLUSTER)

                U1, V1, step = kmeans.k_means_lib_sorted(data_kmeans, warehouse)
                cluster = kmeans.data_to_cluster(U1)
                logging.info(cluster)

                # --- VÒNG LẶP RUN_TIMES ĐÃ ĐƯỢC SỬA LẠI THỤT LỀ (INDENT) CHUẨN ---
                for run in range(RUN_TIMES):
                    match algorithm:
                        case "GA":
                            al = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                                    vehcicle_capacity=VEHCICLE_CAPACITY, conserve_rate=CONSERVE_RATE, M=M, customers=customers)
                        case "PSO":
                            al = PSO(num_particles=NUM_PARTICLES, max_iter=MAX_ITER_PSO,
                                    vehcicle_capacity=VEHCICLE_CAPACITY, M=M, w=W, c1=C1, c2=C2, customers=customers)
                        case "ACO":
                            al = ACO(num_ants=NUM_ANTS, max_iter=MAX_ITER_ACO, vehcicle_capacity=VEHCICLE_CAPACITY,
                                    M=M, alfa=ALFA, sigm=SIGM, beta=BETA, ro=RO, th=TH, customers=customers)
                    
                    logging.info(f"#{algorithm} Lần chạy {run+1} =============================")

                    # Gọi hàm tối ưu nằm TRONG vòng lặp run
                    best_fitness_global, best_route_global, best_distance_global, route_count_global, process_time = al.fit_allClusters(clusters=cluster)

                    run_time_mean += process_time
                    route_count_mean += route_count_global
                    distance_mean += best_distance_global
                    fitness_mean += best_fitness_global
                    
                    logging.info(f"Thời gian chạy {data_file[:-4]} lần {run+1}: {round_float(process_time)}")
                    logging.info(f"Fitness: {round_float(best_fitness_global)}") 
                    logging.info(f"Distance: {round_float(best_distance_global)}")
                    logging.info(f"Số lượng route: {route_count_global}")
                    logging.info(best_route_global)
                    
                    data_excels[idx_al][idx_dat][run] = np.array([route_count_global, round_float(best_distance_global), round_float(best_fitness_global), round_float(process_time)])
                    logging.info("===================================")
                
                # Tính toán giá trị trung bình cho file dữ liệu hiện tại
                avg_fitness = fitness_mean / RUN_TIMES
                avg_route = route_count_mean / RUN_TIMES
                avg_distance = distance_mean / RUN_TIMES
                avg_runtime = run_time_mean / RUN_TIMES

                # SỬA LỖI 1: Tích lũy vào biến tổng của toàn bộ dữ liệu (DATA)
                fitness_data += avg_fitness
                route_count_data += avg_route
                distance_data += avg_distance
                run_time_data += avg_runtime

                # Thống kê file dữ liệu hiện tại
                logging.info(f"#Thống kê {data_file[:-4]} =============================")
                logging.info(f"Số lượt chạy mỗi bộ dữ liệu: {RUN_TIMES}")
                logging.info(f"Fitness trung bình: {round_float(avg_fitness)}")
                logging.info(f"Số lượng route trung bình: {round_float(avg_route)}")
                logging.info(f"Thời gian di chuyển trung bình: {round_float(avg_distance)}")
                logging.info(f"Thời gian chạy trung bình: {round_float(avg_runtime)}")
                logging.info("====================================================================================================================")

            # Thống kê tổng quan cho cả DATA_NAME (Ví dụ: C1)
            logging.info("=====================================================================================================================================")
            logging.info(f"#Thống kê TOÀN BỘ BỘ DỮ LIỆU {DATA_NAME} =============================")
            logging.info(f"Số lượt chạy mỗi bộ dữ liệu: {RUN_TIMES}")
            logging.info(f"Fitness trung bình tổng: {round_float(fitness_data / len_data)}")
            logging.info(f"Số lượng route trung bình tổng: {round_float(route_count_data / len_data)}")
            logging.info(f"Thời gian di chuyển trung bình tổng: {round_float(distance_data / len_data)}")
            logging.info(f"Thời gian chạy trung bình tổng: {round_float(run_time_data / len_data)}")
            # SỬA LỖI: Sửa thành _start_time_all để đo chính xác thời gian chạy từ đầu chương trình
            logging.info(f"THÀNH CÔNG, tổng thời gian thực thi toàn hệ thống: {round_float(time.time() - _start_time_all)}")

    except Exception as e:
        print(f"Lỗi hệ thống trong quá trình thực thi: {str(e)}")    

    finally:
        # Khối lệnh luôn chạy để bảo toàn dữ liệu đầu ra Excel
        if (DATA_NAME != None):
            print("Đang xuất kết quả ra file Excel...")
            write_excel_file(data_excels=data_excels, data_files=data_files, data_name=DATA_NAME, run_time=RUN_TIMES,
                             algorithms=algorithms, title_names=TITLE_NAMES, fileio=EXCEL_FILE)
            print(f"Xuất file thành công: {EXCEL_FILE}")