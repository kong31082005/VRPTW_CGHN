import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
import math
import random
from ultility.utilities import round_float, write_excel_file, distance_cdist
import os
import time

# --- LỚP CÁ THỂ ---
class Individual():
    def __init__(self, customerList: np.array = None, fitness: float = 0, distance: float = 0):
        self.customerList = customerList
        self.fitness = fitness
        self.distance = distance

    def print(self):
        print(self.customerList, ' ', self.fitness, ' ', self.distance)


# --- THUẬT TOÁN DI TRUYỀN (GA) ---
class GA:
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.15, vehcicle_capacity: float = 200, conserve_rate: float = 0.1, 
                 M: float = 50, customers: list = None):
        self._individual = individual          # Số cá thể
        self._generation = generation          # Số thế hệ
        self._crossover_rate = crossover_rate  # Tỉ lệ trao đổi chéo
        self._mutation_rate = mutation_rate    # Tỉ lệ đột biến
        self._vehcicle_capacity = vehcicle_capacity  # Trọng tải của xe
        self._conserve_rate = conserve_rate    # Tỉ lệ bảo tồn
        self._M = M                            # Sai số thời gian
        self.customers = customers              # Dữ liệu khách hàng

        self.best_distance_global = 0
        self.route_count_global = 0
        self.best_fitness_global = 0
        self.best_route_global = []

        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

        self.process_time = 0
        self.__population = []
    # Khởi tạo quần thể ngẫu nhiên ban đầu
    def initialPopulation(self, cluster) -> None:
        self.__population = [Individual(customerList=np.random.permutation(cluster)) for _ in range(self._individual)]
    
    def individualToRoute(self, individual):
        route = []  
        vehicle_load = 0  
        sub_route = []  
        elapsed_time = 0  
        last_customer_id = 0  
        depot = self.customers[0]

        for customer_id in individual:
            customer = self.customers[customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand

            service_time = customer.serviceTime
            moving_time = np.linalg.norm(customer.xy_coord - self.customers[last_customer_id].xy_coord)
            arrive_time = elapsed_time + moving_time

            waiting_time = max(customer.readyTime - self._M - arrive_time, 0)
            return_time = np.linalg.norm(customer.xy_coord - depot.xy_coord)

            update_elapsed_time = arrive_time + service_time + waiting_time + return_time

            if (update_vehicle_load <= self._vehcicle_capacity) and (update_elapsed_time <= depot.dueTime + self._M):
                sub_route.append(customer_id)
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
            else:
                if sub_route:
                    route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
                
                time_from_depot = np.linalg.norm(depot.xy_coord - customer.xy_coord)
                arrive_time = time_from_depot
                waiting_time = max(customer.readyTime - self._M - arrive_time, 0)
                elapsed_time = arrive_time + service_time + waiting_time
                
            last_customer_id = customer_id
            
        if sub_route:
            route.append(sub_route)
            
        return [arr for arr in route if len(arr) > 0]

    # Tính mức độ thích nghi trên toàn bộ quần thể
    def cal_fitness_population(self):
        for individual in self.__population:
            individual.fitness, individual.distance = self.cal_fitness_individualV2(individual.customerList)

    # Hàm tính fitness phiên bản cũ (Giữ nguyên để tránh lỗi gọi hàm liên đới nếu có)
    def cal_fitness_individual(self, individual):
        route = self.individualToRoute(individual)
        fitness = 0
        distance = 0
        for sub_route in route:
            sub_route_time_cost = 0  
            sub_route_distance = 0  
            elapsed_time = 0
            last_customer_id = 0
            for customer_id in sub_route:
                customer = self.customers[customer_id]
                moving_time = np.linalg.norm(customer.xy_coord - self.customers[last_customer_id].xy_coord)
                sub_route_distance += moving_time

                arrive_time = moving_time + elapsed_time
                waiting_time = max(customer.readyTime - self._M - arrive_time, 0)
                delay_time = max(arrive_time - customer.dueTime - self._M, 0)

                sub_route_time_cost += waiting_time + delay_time
                elapsed_time = arrive_time + customer.serviceTime + waiting_time
                last_customer_id = customer_id
                
            return_time = np.linalg.norm(self.customers[last_customer_id].xy_coord - self.customers[0].xy_coord)
            sub_route_distance += return_time
            fitness += sub_route_distance + sub_route_time_cost
            distance += sub_route_distance

        return fitness, distance

    # Hàm tính fitness cải tiến V2 - Đã sửa lỗi logic tính khoảng cách, cộng trùng và tính phạt trễ giờ
    def cal_fitness_individualV2(self, individual):
        vehicle_load = 0  
        elapsed_time = 0  
        last_customer_id = 0  
        sub_route = []  
        fitness_penalty = 0
        distance = 0
        depot = self.customers[0]  
        
        for customer_id in individual:
            customer = self.customers[customer_id]
            last_customer = self.customers[last_customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand

            service_time = customer.serviceTime
            moving_time = np.linalg.norm(customer.xy_coord - last_customer.xy_coord)
            arrive_time = elapsed_time + moving_time

            waiting_time = max(customer.readyTime - self._M - arrive_time, 0)
            delay_time = max(arrive_time - customer.dueTime - self._M, 0)
            return_time = np.linalg.norm(customer.xy_coord - depot.xy_coord)

            update_elapsed_time = arrive_time + service_time + waiting_time + return_time

            if (update_vehicle_load <= self._vehcicle_capacity) and (update_elapsed_time <= depot.dueTime + self._M):
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
                sub_route.append(customer_id)
                distance += moving_time
                fitness_penalty += waiting_time + delay_time
            else:
                # Xe trước quay từ điểm cuối cùng của nó về kho
                return_from_last = np.linalg.norm(last_customer.xy_coord - depot.xy_coord)
                distance += return_from_last
                
                # Xe mới xuất phát đi từ kho tới khách hàng hiện tại
                time_to_new = np.linalg.norm(depot.xy_coord - customer.xy_coord)
                distance += time_to_new
                
                sub_route = [customer_id]
                vehicle_load = demand
                
                arrive_time = time_to_new  
                waiting_time = max(customer.readyTime - self._M - arrive_time, 0)
                delay_time = max(arrive_time - customer.dueTime - self._M, 0)
                
                fitness_penalty += waiting_time + delay_time
                elapsed_time = arrive_time + service_time + waiting_time

            last_customer_id = customer_id

        if len(sub_route) > 0:
            return_to_depot = np.linalg.norm(self.customers[last_customer_id].xy_coord - depot.xy_coord)
            distance += return_to_depot

        # Tổng fitness = Tổng quãng đường + Các chi phí phạt thời gian chờ/muộn
        total_fitness = distance + fitness_penalty
        return total_fitness, distance

    def cal_fitness_sub_route(self, route):
        sub_route_result = []
        for sub_route in route:
            fitness, _ = self.cal_fitness_individualV2(sub_route)
            sub_route_result.append(fitness)
        return sub_route_result

    # Sắp xếp và chọn lọc thế hệ (Elite Selection kết hợp loại bớt cá thể kém để tránh bão hòa)
    def selection(self):
        self.__population.sort(key=lambda x: x.fitness)
        positionToDel = math.floor(self._individual * 0.5)  # Giữ lại 50% quần thể tốt nhất làm bố mẹ
        del self.__population[positionToDel:]

    def SinglePointCrossover(self, dad, mom):
        pos1 = random.randrange(len(mom))
        filter_dad = np.setdiff1d(dad, mom[pos1:], assume_unique=True)
        filter_mom = np.setdiff1d(mom, dad[pos1:], assume_unique=True)

        gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:]))
        gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:]))
        return gene_child_1, gene_child_2

    def heuristic_SinglePointCrossover(self, dad, mom):
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        fitness_sub_route_mom = self.cal_fitness_sub_route(sub_route_mom)
        fitness_sub_route_dad = self.cal_fitness_sub_route(sub_route_dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if (self.best_fitness_pM <= best_fitness_sub_route_mom):
            pos1 = random.randrange(len(mom))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(best_fitness_sub_route_mom)]
            pos1 = np.argwhere(mom == best_sub_route_mom[0])[0][0]

        filter_dad = np.setdiff1d(dad, mom[pos1:], assume_unique=True)
        gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:]))
        
        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if (self.best_fitness_pD <= best_fitness_sub_route_dad):
            pos1 = random.randrange(len(dad))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(best_fitness_sub_route_dad)]
            pos1 = np.argwhere(dad == best_sub_route_dad[0])[0][0]

        filter_mom = np.setdiff1d(mom, dad[pos1:], assume_unique=True)
        gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:]))

        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad
        return gene_child_1, gene_child_2

    def heuristic_SinglePointCrossoverV2(self, dad, mom):
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)
        customer = mom[random.randrange(len(mom))]

        for sub_mom in sub_route_mom:
            if (customer in sub_mom):
                sub_route_mom = mom[np.argwhere(mom == sub_mom[0])[0][0]:]
                break
        for sub_dad in sub_route_dad:
            if (customer in sub_dad):
                sub_route_dad = dad[np.argwhere(dad == sub_dad[0])[0][0]:]
                break

        union_gene = np.union1d(sub_route_dad, sub_route_mom)
        intersect_gene = np.intersect1d(sub_route_dad, sub_route_mom, assume_unique=True)
        diff_gene = np.setdiff1d(union_gene, intersect_gene)

        def greedySearch(intersect_gene, diff_gene):
            commom_part = np.copy(intersect_gene)
            for gen in diff_gene:
                customer_obj = self.customers[gen]
                min_val = 1.e10
                idx_cus_min = -1
                for idx, i in enumerate(commom_part):
                    moving_time = np.linalg.norm(self.customers[i].xy_coord - customer_obj.xy_coord)
                    waiting_time = max(customer_obj.readyTime - self._M - moving_time, 0)
                    delay_time = max(moving_time - customer_obj.dueTime - self._M, 0)
                    if (min_val > waiting_time + delay_time):
                        min_val = waiting_time + delay_time
                        idx_cus_min = idx
                commom_part = np.insert(commom_part, idx_cus_min + 1, gen)
            return commom_part

        mom_idx = np.intersect1d(mom, union_gene, assume_unique=True, return_indices=True)[1]
        gene_child_1 = np.copy(mom)
        commom_part = greedySearch(intersect_gene, diff_gene)
        for idx, val in enumerate(mom_idx):
            gene_child_1[val] = commom_part[idx]

        dad_idx = np.intersect1d(dad, union_gene, assume_unique=True, return_indices=True)[1]
        gene_child_2 = np.copy(dad)
        commom_part = greedySearch(intersect_gene, np.flip(diff_gene))
        for idx, val in enumerate(dad_idx):
            gene_child_2[val] = commom_part[idx]

        return gene_child_1, gene_child_2

    def TwoPointCrossover(self, dad, mom):
        size = len(mom)
        pos1, pos2 = sorted(random.sample(range(size), 2))

        filter_dad = np.setdiff1d(dad, mom[pos1:pos2], assume_unique=True)
        filter_mom = np.setdiff1d(mom, dad[pos1:pos2], assume_unique=True)

        gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:pos2], filter_dad[pos1:]))
        gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:pos2], filter_mom[pos1:]))
        return gene_child_1, gene_child_2

    def TwoPointOrderCrossover(self, dad, mom):
        size = len(mom)
        pos1, pos2 = np.sort(random.sample(range(size), 2))

        pos1_dad, pos2_dad = np.sort([np.argwhere(dad == mom[pos1])[0][0], np.argwhere(dad == mom[pos2])[0][0]])
        pos1_mom, pos2_mom = np.sort([np.argwhere(mom == dad[pos1])[0][0], np.argwhere(mom == dad[pos2])[0][0]])

        if (len(mom[pos1:pos2]) == len(dad[pos1_dad:pos2_dad]) and all(mom[pos1:pos2] == dad[pos1_dad:pos2_dad])):
            pos2_dad += random.randint(0, size - len(dad[pos1:pos2]))

        if (len(dad[pos1:pos2]) == len(mom[pos1_mom:pos2_mom]) and all(dad[pos1:pos2] == mom[pos1_mom:pos2_mom])):
            pos2_mom += random.randint(0, size - len(mom[pos1:pos2]))

        filter_dad = np.array([dad[(x + pos2_dad) % size] for x in range(size) if dad[(x + pos2_dad) % size] not in mom[pos1:pos2]])
        filter_mom = np.array([mom[(x + pos2_mom) % size] for x in range(size) if mom[(x + pos2_mom) % size] not in dad[pos1:pos2]])

        gene_child_1 = np.hstack((filter_dad[len(mom[pos2:]):], mom[pos1:pos2], filter_dad[:len(mom[pos2:])]))
        gene_child_2 = np.hstack((filter_mom[len(dad[pos2:]):], dad[pos1:pos2], filter_mom[:len(dad[pos2:])]))
        return gene_child_1, gene_child_2

    def heuristic_TwoPointCrossoverV1(self, dad, mom):
        size = len(mom)
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        fitness_sub_route_mom = self.cal_fitness_sub_route(sub_route_mom)
        fitness_sub_route_dad = self.cal_fitness_sub_route(sub_route_dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if (self.best_fitness_pM <= best_fitness_sub_route_mom):
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(best_fitness_sub_route_mom)]
            pos1 = np.argwhere(mom == best_sub_route_mom[0])[0][0]
            pos2 = np.argwhere(mom == best_sub_route_mom[-1])[0][0]

        filter_dad = np.setdiff1d(dad, mom[pos1:pos2], assume_unique=True)
        gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:pos2], filter_dad[pos1:]))

        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if (self.best_fitness_pD <= best_fitness_sub_route_dad):
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(best_fitness_sub_route_dad)]
            pos1 = np.argwhere(dad == best_sub_route_dad[0])[0][0]
            pos2 = np.argwhere(dad == best_sub_route_dad[-1])[0][0]

        filter_mom = np.setdiff1d(mom, dad[pos1:pos2], assume_unique=True)
        gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:pos2], filter_mom[pos1:]))

        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad
        return gene_child_1, gene_child_2

    def heuristic_TwoPointCrossoverV2(self, dad, mom):
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)
        customer = mom[random.randrange(len(mom))]

        for sub_mom in sub_route_mom:
            if (customer in sub_mom):
                sub_route_mom = sub_mom
                break
        for sub_dad in sub_route_dad:
            if (customer in sub_dad):
                sub_route_dad = sub_dad
                break

        union_gene = np.union1d(sub_route_dad, sub_route_mom)
        intersect_gene = np.intersect1d(sub_route_dad, sub_route_mom, assume_unique=True)
        diff_gene = np.setdiff1d(union_gene, intersect_gene)

        def greedySearch(intersect_gene, diff_gene):
            commom_part = np.copy(intersect_gene)
            for gen in diff_gene:
                parts = [self.cal_fitness_individualV2(np.insert(commom_part, idx, gen))[0] for idx in range(len(commom_part) + 1)]
                idx_cus_min = np.argmin(parts)
                commom_part = np.insert(commom_part, idx_cus_min, gen)
            return commom_part

        mom_idx = np.intersect1d(mom, union_gene, assume_unique=True, return_indices=True)[1]
        gene_child_1 = np.copy(mom)
        commom_part = greedySearch(intersect_gene, diff_gene)
        for idx, val in enumerate(mom_idx):
            gene_child_1[val] = commom_part[idx]

        dad_idx = np.intersect1d(dad, union_gene, assume_unique=True, return_indices=True)[1]
        gene_child_2 = np.copy(dad)
        commom_part = greedySearch(intersect_gene, np.flip(diff_gene))
        for idx, val in enumerate(dad_idx):
            gene_child_2[val] = commom_part[idx]

        return gene_child_1, gene_child_2

    def OPEXCrossover(self, dad, mom):
        pos1 = random.randrange(len(mom))
        gene_child_1 = np.zeros(len(mom), dtype=int)
        gene_child_2 = np.zeros(len(dad), dtype=int)

        gene_child_1[:pos1] = np.copy(mom[:pos1])
        gene_child_2[:pos1] = np.copy(dad[:pos1])

        cut_gen_mom = np.copy(mom[pos1:])
        cut_gen_dad = np.copy(dad[pos1:])

        sort_cut_mom = np.argsort(cut_gen_mom)
        sort_cut_dad = np.argsort(cut_gen_dad)

        for idx, (d, m) in enumerate(zip(sort_cut_dad, sort_cut_mom)):
            gene_child_1[idx + pos1] = cut_gen_mom[d]
            gene_child_2[idx + pos1] = cut_gen_dad[m]
        return gene_child_1, gene_child_2

    def PMXCrossover(self, dad, mom):
        pos1, pos2 = sorted(random.sample(range(len(mom)), 2))
        gene_child_1 = np.zeros(len(mom), dtype=int)
        gene_child_2 = np.zeros(len(dad), dtype=int)
        mapping_gene = np.vstack((dad[pos1:pos2], mom[pos1:pos2]))

        for idx_p, (d, m) in enumerate(zip(dad, mom)):
            if idx_p in range(pos1, pos2):
                gene_child_1[idx_p] = mom[idx_p]
                gene_child_2[idx_p] = dad[idx_p]
                continue

            idx = idx_p
            i = 1
            while d in mapping_gene[i]:
                idx = np.argwhere(mapping_gene[i] == d)[0][0]
                d = mapping_gene[np.abs(i - 1)][idx]
            i = 0
            while m in mapping_gene[i]:
                idx = np.argwhere(mapping_gene[i] == m)[0][0]
                m = mapping_gene[np.abs(i - 1)][idx]

            gene_child_1[idx_p] = d
            gene_child_2[idx_p] = m
        return gene_child_1, gene_child_2

    def ArithmeticCrossover(self, dad, mom):
        alpha = random.random()
        convert_dad = np.zeros(len(dad), dtype=float)
        convert_mom = np.zeros(len(mom), dtype=float)
        for idx, (d, m) in enumerate(zip(dad, mom)):
            convert_dad[idx] = np.linalg.norm(self.customers[0].xy_coord - self.customers[d].xy_coord)
            convert_mom[idx] = np.linalg.norm(self.customers[0].xy_coord - self.customers[m].xy_coord)

        child_1 = convert_dad + alpha * (convert_mom - convert_dad)
        child_2 = convert_mom + alpha * (convert_dad - convert_mom)

        gene_child_1 = child_1.copy()
        gene_child_2 = child_2.copy()
        sort_child_1 = np.sort(gene_child_1)
        sort_child_2 = np.sort(gene_child_2)

        for i in range(len(sort_child_1)):
            gene_child_1[np.argwhere(gene_child_1 == sort_child_1[i])[0][0]] = i
            gene_child_2[np.argwhere(gene_child_2 == sort_child_2[i])[0][0]] = i

        gene_child_1 = gene_child_1.astype(int)
        gene_child_2 = gene_child_2.astype(int)

        # Lỗi tiềm ẩn nếu self.sort_cluster không tồn tại toàn cục, giữ nguyên theo yêu cầu của bạn
        for idx, (val1, val2) in enumerate(zip(gene_child_1, gene_child_2)):
            gene_child_1[idx] = self.sort_cluster[val1]
            gene_child_2[idx] = self.sort_cluster[val2]
        return gene_child_1, gene_child_2

    def PXCrossover(self, dad, mom):
        gene_child_1 = np.array([mom[i] if random.randint(0, 1) == 1 else 0 for i in range(len(dad))])
        gene_child_2 = np.array([dad[i] if random.randint(0, 1) == 1 else 0 for i in range(len(mom))])

        filter_dad = [x for x in dad if x not in gene_child_1]
        filter_mom = [x for x in mom if x not in gene_child_2]

        gene_child_1 = np.array([filter_dad.pop(0) if x == 0 else x for x in gene_child_1])
        gene_child_2 = np.array([filter_mom.pop(0) if x == 0 else x for x in gene_child_2])
        return gene_child_1, gene_child_2

    def IPXCrossover(self, dad, mom):
        gene_child_1 = np.array([mom[i] if random.randint(0, 1) == 1 else 0 for i in range(len(dad))])
        gene_child_2 = np.array([dad[i] if random.randint(0, 1) == 1 else 0 for i in range(len(mom))])

        filter_dad = list(reversed([x for x in dad if x not in gene_child_1]))
        filter_mom = list(reversed([x for x in mom if x not in gene_child_2]))

        gene_child_1 = np.array([filter_dad.pop(0) if x == 0 else x for x in gene_child_1])
        gene_child_2 = np.array([filter_mom.pop(0) if x == 0 else x for x in gene_child_2])
        return gene_child_1, gene_child_2

    def BestCostRouteCrossover(self, dad, mom):
        route_dad = self.individualToRoute(dad)
        route_mom = self.individualToRoute(mom)

        sub_route_mom = route_mom[np.random.randint(0, len(route_mom))]
        sub_route_dad = route_dad[np.random.randint(0, len(route_dad))]

        gene_child_1 = np.setdiff1d(dad, sub_route_mom)
        gene_child_2 = np.setdiff1d(mom, sub_route_dad)

        def greedySearch(intersect_gene, diff_gene):
            commom_part = np.copy(intersect_gene)
            for gen in diff_gene:
                parts = [self.cal_fitness_individualV2(np.insert(commom_part, idx, gen))[0] for idx in range(len(commom_part) + 1)]
                idx_cus_min = np.argmin(parts)
                commom_part = np.insert(commom_part, idx_cus_min, gen)
            return commom_part

        gene_child_1 = greedySearch(gene_child_1, sub_route_mom)
        gene_child_2 = greedySearch(gene_child_2, sub_route_dad)
        return gene_child_1, gene_child_2

    def STPBCrossover(self, dad, mom):
        probabilities = [0.25, 0.25, 0.25, 0.25]
        choice = np.random.choice([i for i in range(len(probabilities))], p=probabilities)
        match choice:
            case 0:
                gene_child_1, gene_child_2 = self.heuristic_SinglePointCrossover(dad, mom)
            case 1:
                gene_child_1, gene_child_2 = self.heuristic_TwoPointCrossoverV1(dad, mom)
            case 2:
                gene_child_1, gene_child_2 = self.PMXCrossover(dad, mom)
            case 3:
                gene_child_1, gene_child_2 = self.BestCostRouteCrossover(dad, mom)
        return gene_child_1, gene_child_2

    def sort_cluster_by_timewindow(self, cluster):
        distance_cluster = np.zeros(len(cluster), dtype=float)
        start_time_cluster = np.zeros(len(cluster), dtype=float)

        for idx, c in enumerate(cluster):
            distance_cluster[idx] = np.linalg.norm(self.customers[0].xy_coord - self.customers[c].xy_coord)
            start_time_cluster[idx] = self.customers[c].readyTime

        sort_cluster = np.lexsort((start_time_cluster, distance_cluster))
        for i in range(len(sort_cluster)):
            sort_cluster[i] = cluster[sort_cluster[i]]
        return sort_cluster

    def re_cluster_by_timewindow(self, clusters):
        def check_concatenate(cluster1, cluster2):
            total_cluster = np.concatenate((cluster1, cluster2), axis=0)
            check = sum([self.customers[i].serviceTime for i in total_cluster])
            check_capacity = sum([self.customers[i].demand for i in total_cluster])

            if check_capacity > self._vehcicle_capacity:
                return False
                
            cluster1_coords = [self.customers[i].xy_coord for i in cluster1]
            cluster2_coords = [self.customers[i].xy_coord for i in cluster2]
            total_cluster_coords = np.concatenate((cluster1_coords, cluster2_coords), axis=0)

            distance = distance_cdist(total_cluster_coords, total_cluster_coords, metric='euclidean')
            aver_dist = np.mean(distance[np.nonzero(distance)]) if np.count_nonzero(distance) > 0 else 0

            distance_to_depot = distance_cdist(total_cluster_coords, [self.customers[0].xy_coord], metric='euclidean')
            avg_distance_to_depot = np.min(distance_to_depot)
            
            check += 2 * avg_distance_to_depot + (len(total_cluster) - 1) * aver_dist
            return check <= self.customers[0].dueTime
        
        def concatenate_arrays(array, index1, index2):
            return [np.concatenate((array[index1], array[index2])).tolist()] + [array[i] for i in range(len(array)) if i != index1 and i != index2]
            
        i = 0
        while i < len(clusters) - 1:
            j = i + 1
            while j < len(clusters):
                if check_concatenate(clusters[i], clusters[j]):
                    clusters = concatenate_arrays(clusters, i, j)
                    j = i + 1  
                else:
                    j += 1  
            i += 1
        return clusters

    def sort_cluster_by_distance(self, cluster):
        convert_cluster = np.zeros(len(cluster), dtype=float)
        for idx, c in enumerate(cluster):
            convert_cluster[idx] = np.linalg.norm(self.customers[0].xy_coord - self.customers[c].xy_coord)

        sort_cluster = np.argsort(convert_cluster)
        for i in range(len(sort_cluster)):
            sort_cluster[i] = cluster[sort_cluster[i]]
        return sort_cluster

    # Sửa đổi đột biến Inversion cục bộ để bảo vệ cấu trúc định dạng cụm của Kmeans
    def mutation(self, child):
        child_new = np.copy(child)
        if len(child) < 2:
            return child_new
        pos1, pos2 = sorted(random.sample(range(len(child)), 2))
        child_new[pos1:pos2+1] = np.flip(child_new[pos1:pos2+1])
        return child_new

    def hybird(self):
        index = math.floor(self._conserve_rate * self._individual)
        if len(self.__population) < 2:
            return  # Tránh crash nếu quần thể quá nhỏ

        while (len(self.__population) < self._individual):
            hybird_rate = random.random()
            dad, mom = random.sample(self.__population, 2) # Cho phép lấy ngẫu nhiên trong phần giữ lại làm cha mẹ

            if (hybird_rate > self._crossover_rate):
                continue

            gene_child_1, gene_child_2 = self.STPBCrossover(np.copy(dad.customerList), np.copy(mom.customerList))

            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)

            child1 = Individual(customerList=gene_child_1)
            self.__population.append(child1)

            if (len(self.__population) < self._individual):
                child2 = Individual(customerList=gene_child_2)
                self.__population.append(child2)

    def fit(self, cluster):
        _start_time = time.time()
        self.initialPopulation(cluster)

        for _ in range(self._generation):
            self.cal_fitness_population()
            self.selection()
            self.hybird()
            
        self.process_time += round_float(time.time() - _start_time)

        self.cal_fitness_population()
        self.selection()

        self.best_fitness_global += self.__population[0].fitness
        self.best_distance_global += self.__population[0].distance
        best_route = self.individualToRoute(self.__population[0].customerList)
        self.best_route_global.append(best_route)
        self.route_count_global += len(best_route)
        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

    def fit_allClusters(self, clusters):
        clusters = self.re_cluster_by_timewindow(clusters)
        for i in range(len(clusters)):
            self.fit(cluster=clusters[i])
        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global, self.process_time


# --- HÀM MAIN CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    # Thông số K-means
    N_CLUSTER = 10         # Khuyên dùng 10 cụm cho bài toán 100 KH thay vì 20 cụm quá manh mún
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100
    
    # Thông số GA
    INDIVIDUAL = 150        # Cấu hình cân bằng giữa thời gian và chất lượng nghiệm
    GENERATION = 120
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    VEHCICLE_CAPACITY = 200 # Khớp tải trọng với bộ dữ liệu Solomon R101 chuẩn
    CONSERVE_RATE = 0.1
    M = 0
    
    # Thông số quản lý tệp dữ liệu
    DATA_ID = "R101"  
    DATA_NAME = "R1"  
    DATA_NUMBER_CUS = "100"  
    RUN_TIMES = 5          # Giảm bớt số lần chạy nháp để theo dõi kết quả nhanh hơn
    FILE_EXCEL_PATH = "result/"
    FILE_NAME = "_TestKmeans"
    
    if (DATA_ID != None):
        url_data = "data/txt/" + DATA_NUMBER_CUS + "/" + DATA_ID[:-2] + "/"
        data_files = [DATA_ID + ".txt"]
    else:
        url_data = "data/txt/" + DATA_NUMBER_CUS + "/" + DATA_NAME + "/"
        data_files = sorted([f for f in os.listdir(url_data) if f.endswith(('.txt'))])

    len_data = len(data_files)
    print(f"Bộ dữ liệu {DATA_NAME}: {data_files}")
    run_time_data = 0
    route_count_data = 0
    distance_data = 0
    fitness_data = 0

    data_excel = []
    for data_file in data_files:
        run_time_mean = 0
        route_count_mean = 0
        distance_mean = 0
        fitness_mean = 0

        _start_time = time.time()
        data, customers = load_txt_dataset(url=url_data, name_of_id=data_file)
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))

        print("#K-means =============================")
        kmeans = Kmeans(epsilon=EPSILON, maxiter=MAX_ITER, n_cluster=N_CLUSTER)
        warehouse = data[0]
        data_kmeans = np.delete(data, 0, 0)

        U1, V1, step = kmeans.k_means_lib_sorted(data_kmeans, warehouse)
        cluster = kmeans.data_to_cluster(U1)
        print("Các cụm khởi tạo ban đầu từ Kmeans:\n", cluster)

        print("#GA =============================")
        for j in range(RUN_TIMES):
            # Khởi tạo đối tượng GA mới hoàn toàn cho mỗi lượt để tránh lỗi tích lũy dữ liệu rác
            ga = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, 
                    mutation_rate=MUTATION_RATE, vehcicle_capacity=VEHCICLE_CAPACITY, 
                    conserve_rate=CONSERVE_RATE, M=M, customers=customers)
            
            best_fitness_global, best_route_global, best_distance_global, route_count_global, process_time = ga.fit_allClusters(clusters=cluster)

            run_time_mean += process_time
            print(f"Lượt chạy thứ {j+1} của tệp {data_file[:-4]}: Thời gian = {process_time}s")
            print("-> Fitness: ", round_float(best_fitness_global))
            print("-> Distance: ", round_float(best_distance_global))
            print("-> Số lượng xe (route): ", route_count_global)
            
            route_count_mean += route_count_global
            distance_mean += best_distance_global
            fitness_mean += best_fitness_global
            
        print(f"\n#Thống kê tổng hợp tệp {data_file[:-4]} (Trung bình sau {RUN_TIMES} lượt) ===============")
        print("Fitness TB: ", round_float(fitness_mean / RUN_TIMES))
        print("Số lượng tuyến đường TB: ", round_float(route_count_mean / RUN_TIMES))
        print("Khoảng cách di chuyển TB: ", round_float(distance_mean / RUN_TIMES))
        print("Thời gian tính toán TB: ", round_float(run_time_mean / RUN_TIMES))