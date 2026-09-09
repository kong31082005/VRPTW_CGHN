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
    def __init__(
        self,
        customerList: np.array = None,
        fitness: float = 0,
        distance: float = 0,
        vehicle_count: int = 0
    ):
        self.customerList = customerList
        self.fitness = fitness
        self.distance = distance
        self.vehicle_count = vehicle_count

    def print(self):
        print(self.customerList, ' ', self.fitness, ' ', self.distance)


# --- THUẬT TOÁN DI TRUYỀN (GA) ---
class GA:
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.15, vehcicle_capacity: float = None, vehicle_number: int = None,
                 conserve_rate: float = 0.1, M: float = 50, customers: list = None):
        self._individual = individual          # Số cá thể
        self._generation = generation          # Số thế hệ
        self._crossover_rate = crossover_rate  # Tỉ lệ trao đổi chéo
        self._mutation_rate = mutation_rate    # Tỉ lệ đột biến
        self._vehcicle_capacity = vehcicle_capacity  # Trọng tải của xe
        self._vehicle_number = vehicle_number
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
    # Xây dựng tuyến xe theo ràng buộc VRPTW Solomon: tải trọng, cửa sổ thời gian và thời gian quay về kho (Cong_1)
    def individualToRoute(self, individual):
        routes = []
        depot = self.customers[0]
        current_route = []
        vehicle_load = 0
        current_time = float(depot.readyTime)
        last_customer_id = 0

        depot = self.customers[0]

        for customer_id in individual:
            customer = self.customers[customer_id]

            # ==========================================
            # 1. Kiểm tra tải trọng
            # ==========================================
            new_vehicle_load = vehicle_load + customer.demand

            # ==========================================
            # 2. Tính thời gian di chuyển từ điểm trước
            # ==========================================
            travel_time = np.linalg.norm(
                self.customers[last_customer_id].xy_coord
                - customer.xy_coord
            )

            arrival_time = current_time + travel_time

            # ==========================================
            # 3. Nếu đến sớm thì phải chờ
            # ==========================================
            service_start_time = max(
                arrival_time,
                customer.readyTime
            )

            # ==========================================
            # 4. Kiểm tra cửa sổ thời gian của khách
            # ==========================================
            customer_time_feasible = (
                service_start_time <= customer.dueTime + self._M
            )

            # ==========================================
            # 5. Thời gian sau khi phục vụ khách
            # ==========================================
            finish_service_time = (
                service_start_time
                + customer.serviceTime
            )

            # ==========================================
            # 6. Kiểm tra còn quay về kho kịp không
            # ==========================================
            return_to_depot_time = np.linalg.norm(
                customer.xy_coord - depot.xy_coord
            )

            depot_time_feasible = (
                finish_service_time
                + return_to_depot_time
                <= depot.dueTime + self._M
            )

            # ==========================================
            # 7. Kiểm tra có thể thêm khách vào xe hiện tại không
            # ==========================================
            can_add_customer = (
                new_vehicle_load <= self._vehcicle_capacity
                and customer_time_feasible
                and depot_time_feasible
            )

            if can_add_customer:
                # Thêm khách vào tuyến hiện tại
                current_route.append(customer_id)

                vehicle_load = new_vehicle_load
                current_time = finish_service_time
                last_customer_id = customer_id

            else:
                # ==========================================
                # 8. Đóng tuyến hiện tại
                # ==========================================
                if current_route:
                    routes.append(current_route)

                # ==========================================
                # 9. Mở xe mới từ depot
                # ==========================================
                current_route = []
                vehicle_load = 0
                current_time = float(depot.readyTime)
                last_customer_id = 0

                # Tính lại cho khách hiện tại từ depot
                travel_time = np.linalg.norm(
                    depot.xy_coord
                    - customer.xy_coord
                )

                arrival_time = current_time + travel_time

                service_start_time = max(
                    arrival_time,
                    customer.readyTime
                )

                finish_service_time = (
                    service_start_time
                    + customer.serviceTime
                )

                return_to_depot_time = np.linalg.norm(
                    customer.xy_coord
                    - depot.xy_coord
                )

                # ==========================================
                # 10. Kiểm tra khách này tự đi một xe có hợp lệ không
                # ==========================================
                single_customer_feasible = (
                    customer.demand <= self._vehcicle_capacity
                    and service_start_time <= customer.dueTime + self._M
                    and finish_service_time + return_to_depot_time
                    <= depot.dueTime + self._M
                )

                if single_customer_feasible:
                    current_route = [customer_id]

                    vehicle_load = customer.demand
                    current_time = finish_service_time
                    last_customer_id = customer_id

                else:
                    # Đây là trường hợp dữ liệu hoặc nghiệm không hợp lệ
                    raise ValueError(
                        f"Khách hàng {customer_id} không thể phục vụ hợp lệ "
                        f"ngay cả khi dùng một xe riêng."
                    )

        # ==========================================
        # 11. Thêm tuyến cuối cùng
        # ==========================================
        if current_route:
            routes.append(current_route)

        return routes
    
    def diagnose_clusters(self, clusters):
        print("\n========== PHÂN TÍCH CLUSTER ==========")

        for i, cluster in enumerate(clusters):
            customers = [self.customers[int(c)] for c in cluster]

            ready_times = [c.readyTime for c in customers]
            due_times = [c.dueTime for c in customers]
            demands = [c.demand for c in customers]

            ready_min = min(ready_times)
            ready_max = max(ready_times)
            due_min = min(due_times)
            due_max = max(due_times)
            total_demand = sum(demands)

            print(
                f"Cluster {i + 1:02d} | "
                f"Customers={len(cluster):2d} | "
                f"Demand={total_demand:3d} | "
                f"Ready={ready_min:.0f}-{ready_max:.0f} | "
                f"Due={due_min:.0f}-{due_max:.0f}"
            )

        print("========================================\n")

    def diagnose_routes(self, routes):
        print("\n========== PHÂN TÍCH ROUTE ==========")

        depot = self.customers[0]

        for index, route in enumerate(routes, start=1):
            load = 0
            current_time = 0.0
            last_customer_id = 0

            for customer_id in route:
                customer = self.customers[customer_id]
                last_customer = self.customers[last_customer_id]

                load += customer.demand

                travel_time = np.linalg.norm(
                    customer.xy_coord - last_customer.xy_coord
                )

                arrival_time = current_time + travel_time
                service_start = max(arrival_time, customer.readyTime)

                current_time = (
                    service_start + customer.serviceTime
                )

                last_customer_id = customer_id

            return_time = np.linalg.norm(
                self.customers[last_customer_id].xy_coord
                - depot.xy_coord
            )

            finish_time = current_time + return_time

            print(
                f"Route {index:02d} | "
                f"Customers={len(route):2d} | "
                f"Load={load:3d}/{self._vehcicle_capacity} | "
                f"Finish={finish_time:.2f}/{depot.dueTime}"
            )

        print("=====================================\n")
    # ==================== Kiểm tra một tuyến có thỏa các ràng buộc VRPTW Solomon hay không ====================
    def is_route_feasible(self, route):
        depot = self.customers[0]

        vehicle_load = 0
        current_time = float(depot.readyTime)
        last_customer_id = 0

        for customer_id in route:
            customer = self.customers[customer_id]
            last_customer = self.customers[last_customer_id]

            # ============================
            # 1. Kiểm tra tải trọng xe
            # ============================
            vehicle_load += customer.demand

            if vehicle_load > self._vehcicle_capacity:
                return False

            # ============================
            # 2. Thời gian di chuyển
            # ============================
            travel_time = np.linalg.norm(
                customer.xy_coord - last_customer.xy_coord
            )

            arrival_time = current_time + travel_time

            # ============================
            # 3. Nếu đến sớm thì chờ
            # ============================
            service_start_time = max(
                arrival_time,
                customer.readyTime
            )

            # ============================
            # 4. Kiểm tra cửa sổ thời gian
            # ============================
            if service_start_time > customer.dueTime + self._M:
                return False

            # ============================
            # 5. Cập nhật thời gian sau phục vụ
            # ============================
            current_time = (
                service_start_time
                + customer.serviceTime
            )

            last_customer_id = customer_id

        # ============================
        # 6. Kiểm tra quay về kho
        # ============================
        if len(route) > 0:
            last_customer = self.customers[last_customer_id]

            return_time = np.linalg.norm(
                last_customer.xy_coord - depot.xy_coord
            )

            if current_time + return_time > depot.dueTime + self._M:
                return False

        return True

    # ==================== Kiểm tra toàn bộ nghiệm cuối ====================
    def validate_solution(self, routes):

        all_customers = []
        invalid_routes = []

        # ==========================================
        # 1. Kiểm tra từng route có hợp lệ không
        # ==========================================
        for idx, route in enumerate(routes):

            if not self.is_route_feasible(route):
                invalid_routes.append(idx + 1)

            for customer_id in route:
                all_customers.append(int(customer_id))

        # ==========================================
        # 2. Kiểm tra khách hàng bị trùng
        # ==========================================
        duplicate_customers = []

        unique_customers = set()

        for customer_id in all_customers:
            if customer_id in unique_customers:
                duplicate_customers.append(customer_id)
            else:
                unique_customers.add(customer_id)

        duplicate_customers = sorted(
            list(set(duplicate_customers))
        )

        # ==========================================
        # 3. Kiểm tra khách bị thiếu
        #
        # customers[0] là depot
        # customer thật từ 1 -> N
        # ==========================================
        expected_customers = set(
            range(1, len(self.customers))
        )

        actual_customers = set(all_customers)

        missing_customers = sorted(
            list(
                expected_customers
                - actual_customers
            )
        )

        # ==========================================
        # 4. Kiểm tra khách ngoài phạm vi
        # ==========================================
        extra_customers = sorted(
            list(
                actual_customers
                - expected_customers
            )
        )

        # ==========================================
        # 5. Kiểm tra tổng số lần phục vụ
        # ==========================================
        expected_count = (
            len(self.customers) - 1
        )

        actual_count = len(
            all_customers
        )
        # kiểm tra vehicle number
        vehicle_count = len(routes)

        vehicle_limit_feasible = (
            self._vehicle_number is None
            or vehicle_count <= self._vehicle_number
        )
        # ==========================================
        # 6. Kết luận
        # ==========================================
        is_valid = (
            len(invalid_routes) == 0
            and len(duplicate_customers) == 0
            and len(missing_customers) == 0
            and len(extra_customers) == 0
            and actual_count == expected_count
            and vehicle_limit_feasible
        )

        return {
            "is_valid": is_valid,
            "invalid_routes": invalid_routes,
            "duplicate_customers": duplicate_customers,
            "missing_customers": missing_customers,
            "extra_customers": extra_customers,
            "expected_customer_count": expected_count,
            "actual_customer_count": actual_count,
            "vehicle_count": vehicle_count,
            "vehicle_limit": self._vehicle_number,
            "vehicle_limit_feasible": vehicle_limit_feasible
        }
    # Tính tổng quãng đường của một tuyến 
    def calculate_route_distance(self, route):
        if len(route) == 0:
            return 0.0

        depot = self.customers[0]

        total_distance = 0.0
        last_customer_id = 0

        for customer_id in route:
            total_distance += np.linalg.norm(
                self.customers[last_customer_id].xy_coord
                - self.customers[customer_id].xy_coord
            )

            last_customer_id = customer_id

        # Khách cuối quay về kho
        total_distance += np.linalg.norm(
            self.customers[last_customer_id].xy_coord
            - depot.xy_coord
        )

        return total_distance

    # ==================== Tối ưu thứ tự khách trong một tuyến bằng 2-opt ====================
    def two_opt_route(self, route):
        if len(route) < 4:
            return list(route)

        best_route = list(route)
        best_distance = self.calculate_route_distance(best_route)

        improved = True

        while improved:
            improved = False

            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):

                    # Đảo ngược đoạn từ i đến j
                    new_route = (
                        best_route[:i]
                        + best_route[i:j + 1][::-1]
                        + best_route[j + 1:]
                    )

                    # Tuyến mới bắt buộc phải thỏa VRPTW
                    if not self.is_route_feasible(new_route):
                        continue

                    new_distance = self.calculate_route_distance(new_route)

                    # Chỉ nhận nếu quãng đường giảm
                    if new_distance < best_distance - 1e-9:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break

                if improved:
                    break

        return best_route

    def two_opt_all_routes(self, routes):
        return [self.two_opt_route(route) for route in routes]

    # Chuyển khách giữa các tuyến để giảm số lượng xe, loại bỏ hoàn toàn một tuyến nếu có thể.
    def relocate_routes(self, routes):

        routes = [list(route) for route in routes]

        improved = True

        while improved:
            improved = False

            # Ưu tiên xử lý tuyến ít khách trước
            route_order = sorted(
                range(len(routes)),
                key=lambda i: len(routes[i])
            )

            for source_idx in route_order:

                # Tuyến nguồn đã bị xóa
                if source_idx >= len(routes):
                    continue

                source_route = routes[source_idx]

                if len(source_route) == 0:
                    continue

                # Thử chuyển toàn bộ khách của tuyến nguồn
                temp_routes = [
                    list(route)
                    for route in routes
                ]

                customers_to_move = list(source_route)

                success = True

                for customer_id in customers_to_move:

                    best_target_idx = None
                    best_position = None
                    best_distance_increase = float("inf")

                    # ==========================================
                    # Thử chèn khách vào tất cả tuyến khác
                    # ==========================================
                    for target_idx, target_route in enumerate(temp_routes):

                        if target_idx == source_idx:
                            continue

                        # Thử tất cả vị trí chèn
                        for pos in range(len(target_route) + 1):

                            candidate_route = (
                                target_route[:pos]
                                + [customer_id]
                                + target_route[pos:]
                            )

                            # Tuyến mới phải hợp lệ
                            if not self.is_route_feasible(candidate_route):
                                continue

                            # Tính mức tăng quãng đường
                            old_distance = self.calculate_route_distance(
                                target_route
                            )

                            new_distance = self.calculate_route_distance(
                                candidate_route
                            )

                            distance_increase = (
                                new_distance - old_distance
                            )

                            # Chọn vị trí tăng quãng đường ít nhất
                            if distance_increase < best_distance_increase:
                                best_distance_increase = distance_increase
                                best_target_idx = target_idx
                                best_position = pos

                    # Không tìm được tuyến nào để chèn khách
                    if best_target_idx is None:
                        success = False
                        break

                    # Chèn khách vào tuyến tốt nhất
                    temp_routes[best_target_idx].insert(
                        best_position,
                        customer_id
                    )

                # ==========================================
                # Nếu chuyển được toàn bộ khách của tuyến nguồn
                # thì xóa tuyến đó
                # ==========================================
                if success:
                    del temp_routes[source_idx]

                    routes = temp_routes

                    improved = True
                    break

        return routes
    
    def best_insertion(self, customer_id, routes):
        best_route_idx = None
        best_position = None
        best_cost = float("inf")

        for route_idx, route in enumerate(routes):
            old_distance = self.calculate_route_distance(route)

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [customer_id] + route[pos:]

                if not self.is_route_feasible(new_route):
                    continue

                new_distance = self.calculate_route_distance(new_route)
                cost = new_distance - old_distance

                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = route_idx
                    best_position = pos

        return best_route_idx, best_position

    def get_best_insertions(self, customer_id, routes, max_options=5):
        options = []

        for route_idx, route in enumerate(routes):
            old_distance = self.calculate_route_distance(route)

            for position in range(len(route) + 1):
                new_route = (
                    route[:position]
                    + [customer_id]
                    + route[position:]
                )

                if not self.is_route_feasible(new_route):
                    continue

                new_distance = self.calculate_route_distance(new_route)
                increase = new_distance - old_distance

                options.append(
                    (increase, route_idx, position)
                )

        options.sort(key=lambda x: x[0])

        return options[:max_options]

    def backtrack_eliminate(self, customers_to_move, routes, customer_index=0, max_options=3, max_nodes=600, counter=None):
        if counter is None:
            counter = [0]

        if counter[0] >= max_nodes:
            return None

        counter[0] += 1

        # Đã chuyển hết khách
        if customer_index >= len(customers_to_move):
            return [list(route) for route in routes]

        customer_id = customers_to_move[customer_index]

        # ==========================================
        # Level 0: Direct insertion
        # ==========================================
        insertion_options = self.get_best_insertions(customer_id, routes, max_options=max_options)

        for _, route_idx, position in insertion_options:
            new_routes = [list(route) for route in routes]
            new_routes[route_idx].insert(position, customer_id)

            result = self.backtrack_eliminate(
                customers_to_move, new_routes, customer_index + 1,
                max_options, max_nodes, counter
            )

            if result is not None:
                return result

        # ==========================================
        # Level 1: 1-ejection
        # ==========================================
        ejected_routes = self.try_ejection_insert(customer_id, routes)

        if ejected_routes is not None:
            result = self.backtrack_eliminate(
                customers_to_move, ejected_routes, customer_index + 1,
                max_options, max_nodes, counter
            )

            if result is not None:
                return result

        # ==========================================
        # Level 2: 2-ejection
        # Chỉ dùng cho route nguồn nhỏ
        # ==========================================
        if len(customers_to_move) <= 5:
            double_ejected_routes = self.try_double_ejection_insert(customer_id, routes)

            if double_ejected_routes is not None:
                result = self.backtrack_eliminate(
                    customers_to_move, double_ejected_routes, customer_index + 1,
                    max_options, max_nodes, counter
                )

                if result is not None:
                    return result

        return None

    def try_ejection_insert(self, customer_id, routes):
        best_routes = None
        best_cost = float("inf")

        for target_idx, target_route in enumerate(routes):
            old_target_distance = self.calculate_route_distance(target_route)

            for eject_pos in range(len(target_route)):
                ejected_customer = target_route[eject_pos]
                reduced_route = target_route[:eject_pos] + target_route[eject_pos + 1:]

                for insert_pos in range(len(reduced_route) + 1):
                    new_target = (
                        reduced_route[:insert_pos]
                        + [customer_id]
                        + reduced_route[insert_pos:]
                    )

                    if not self.is_route_feasible(new_target):
                        continue

                    temp_routes = [list(route) for route in routes]
                    temp_routes[target_idx] = new_target

                    other_routes = [
                        route for i, route in enumerate(temp_routes)
                        if i != target_idx
                    ]

                    route_idx, position = self.best_insertion(
                        ejected_customer, other_routes
                    )

                    if route_idx is None:
                        continue

                    other_routes[route_idx].insert(position, ejected_customer)

                    candidate_routes = []
                    other_idx = 0

                    for i in range(len(temp_routes)):
                        if i == target_idx:
                            candidate_routes.append(new_target)
                        else:
                            candidate_routes.append(other_routes[other_idx])
                            other_idx += 1

                    new_target_distance = self.calculate_route_distance(new_target)
                    cost = new_target_distance - old_target_distance

                    if cost < best_cost:
                        best_cost = cost
                        best_routes = candidate_routes

        return best_routes

    def try_double_ejection_insert(self, customer_id, routes):
        routes = [list(route) for route in routes]

        # ==========================================
        # Thử từng route làm route đích
        # ==========================================
        for target_idx, target_route in enumerate(routes):

            if len(target_route) < 2:
                continue

            # ==========================================
            # Chọn 2 customer để eject
            # ==========================================
            for i in range(len(target_route) - 1):
                for j in range(i + 1, len(target_route)):

                    ejected_1 = target_route[i]
                    ejected_2 = target_route[j]

                    # Route sau khi lấy 2 customer ra
                    reduced_route = [
                        c for k, c in enumerate(target_route)
                        if k != i and k != j
                    ]

                    # ==========================================
                    # Thử mọi vị trí cho customer mới
                    # ==========================================
                    for insert_pos in range(len(reduced_route) + 1):

                        new_target = (
                            reduced_route[:insert_pos]
                            + [customer_id]
                            + reduced_route[insert_pos:]
                        )

                        if not self.is_route_feasible(new_target):
                            continue

                        candidate_routes = [
                            list(route)
                            for route in routes
                        ]

                        candidate_routes[target_idx] = new_target

                        # ==========================================
                        # Chèn lại hai customer bị eject
                        # Không cho chèn lại vào target route
                        # ==========================================
                        other_indices = [
                            k for k in range(len(candidate_routes))
                            if k != target_idx
                        ]

                        temp_routes = [
                            list(candidate_routes[k])
                            for k in other_indices
                        ]

                        # Thử cả hai thứ tự
                        ejection_orders = [
                            [ejected_1, ejected_2],
                            [ejected_2, ejected_1]
                        ]

                        for order in ejection_orders:

                            working_routes = [
                                list(route)
                                for route in temp_routes
                            ]

                            success = True

                            for ejected_customer in order:

                                route_idx, position = self.best_insertion(
                                    ejected_customer,
                                    working_routes
                                )

                                if route_idx is None:
                                    success = False
                                    break

                                working_routes[route_idx].insert(
                                    position,
                                    ejected_customer
                                )

                            if not success:
                                continue

                            # ==================================
                            # Ghép lại đúng vị trí các route
                            # ==================================
                            result_routes = [
                                list(route)
                                for route in candidate_routes
                            ]

                            for local_idx, original_idx in enumerate(other_indices):
                                result_routes[original_idx] = working_routes[local_idx]

                            # Kiểm tra an toàn cuối cùng
                            if all(
                                self.is_route_feasible(route)
                                for route in result_routes
                            ):
                                return result_routes

        return None
    
    def eliminate_routes(self, routes):
        routes = [list(route) for route in routes]
        improved = True

        while improved:
            improved = False

            route_order = sorted(
                range(len(routes)),
                key=lambda i: (
                    len(routes[i]),
                    sum(self.customers[c].demand for c in routes[i])
                )
            )

            for source_idx in route_order:
                if source_idx >= len(routes):
                    continue

                source_route = routes[source_idx]

                if not source_route:
                    continue

                # Chỉ search sâu với route nhỏ
                if len(source_route) > 6:
                    continue

                base_routes = [
                    list(route)
                    for i, route in enumerate(routes)
                    if i != source_idx
                ]

                hardest_first = sorted(
                    source_route,
                    key=lambda c: len(
                        self.get_best_insertions(
                            c,
                            base_routes,
                            max_options=20
                        )
                    )
                )

                customer_orders = [
                    hardest_first,
                    sorted(
                        source_route,
                        key=lambda c: self.customers[c].dueTime
                    ),
                    list(source_route)
                ]

                route_eliminated = False

                for customers_to_move in customer_orders:
                    result = self.backtrack_eliminate(
                        customers_to_move,
                        [list(route) for route in base_routes],
                        max_options=3,
                        max_nodes=600
                    )

                    if result is None:
                        continue

                    if not all(
                        self.is_route_feasible(route)
                        for route in result
                    ):
                        continue

                    routes = result
                    improved = True
                    route_eliminated = True
                    break

                if route_eliminated:
                    break

        return routes

    def optimize_route_order(self, route):
        route = list(route)

        if len(route) < 3:
            return route

        best_route = route
        best_distance = self.calculate_route_distance(route)
        improved = True

        while improved:
            improved = False

            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    new_route = list(best_route)
                    customer = new_route.pop(i)
                    new_route.insert(j, customer)

                    if not self.is_route_feasible(new_route):
                        continue

                    new_distance = self.calculate_route_distance(new_route)

                    if new_distance < best_distance - 1e-9:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break

                if improved:
                    break

        return best_route
    def optimize_all_route_orders(self, routes):
        return [
            self.optimize_route_order(route)
            for route in routes
        ]

    def two_opt_star(self, routes):
        routes = [list(route) for route in routes]
        improved = True

        while improved:
            improved = False

            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    route_1 = routes[i]
                    route_2 = routes[j]

                    old_distance = self.calculate_route_distance(route_1)
                    old_distance += self.calculate_route_distance(route_2)

                    best_route_1 = None
                    best_route_2 = None
                    best_distance = old_distance

                    for cut1 in range(1, len(route_1)):
                        for cut2 in range(1, len(route_2)):
                            new_route_1 = route_1[:cut1] + route_2[cut2:]
                            new_route_2 = route_2[:cut2] + route_1[cut1:]

                            if not new_route_1 or not new_route_2:
                                continue
                            if not self.is_route_feasible(new_route_1):
                                continue
                            if not self.is_route_feasible(new_route_2):
                                continue

                            new_distance = self.calculate_route_distance(new_route_1)
                            new_distance += self.calculate_route_distance(new_route_2)

                            if new_distance < best_distance - 1e-9:
                                best_distance = new_distance
                                best_route_1 = new_route_1
                                best_route_2 = new_route_2

                    if best_route_1 is not None:
                        routes[i] = best_route_1
                        routes[j] = best_route_2
                        improved = True
                        break

                if improved:
                    break

        return routes
    # Hoán đổi khách hàng giữa hai tuyến để cải thiện nghiệm
    def swap_between_routes(self, routes):
        routes = [list(route) for route in routes]

        improved = True

        while improved:
            improved = False

            # Duyệt từng cặp tuyến
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):

                    route_1 = routes[i]
                    route_2 = routes[j]

                    old_distance = (
                        self.calculate_route_distance(route_1)
                        + self.calculate_route_distance(route_2)
                    )

                    best_route_1 = None
                    best_route_2 = None
                    best_distance = old_distance

                    # ==========================================
                    # Thử đổi từng khách của route 1
                    # với từng khách của route 2
                    # ==========================================
                    for pos1 in range(len(route_1)):
                        for pos2 in range(len(route_2)):

                            new_route_1 = list(route_1)
                            new_route_2 = list(route_2)

                            # Hoán đổi hai khách
                            new_route_1[pos1], new_route_2[pos2] = (
                                new_route_2[pos2],
                                new_route_1[pos1]
                            )

                            # Hai tuyến mới đều phải hợp lệ
                            if not self.is_route_feasible(new_route_1):
                                continue

                            if not self.is_route_feasible(new_route_2):
                                continue

                            new_distance = (
                                self.calculate_route_distance(new_route_1)
                                + self.calculate_route_distance(new_route_2)
                            )

                            # Chỉ nhận nếu tổng quãng đường giảm
                            if new_distance < best_distance - 1e-9:
                                best_distance = new_distance
                                best_route_1 = new_route_1
                                best_route_2 = new_route_2

                    # Nếu tìm được phép đổi tốt hơn
                    if best_route_1 is not None:
                        routes[i] = best_route_1
                        routes[j] = best_route_2

                        improved = True
                        break

                if improved:
                    break

        return routes

    def relocate_for_distance(self, routes):
        routes = [list(route) for route in routes]
        improved = True

        while improved:
            improved = False

            for source_idx in range(len(routes)):
                if len(routes[source_idx]) <= 1:
                    continue

                for pos in range(len(routes[source_idx])):
                    customer_id = routes[source_idx][pos]

                    for target_idx in range(len(routes)):
                        if target_idx == source_idx:
                            continue

                        old_source = routes[source_idx]
                        old_target = routes[target_idx]
                        new_source = old_source[:pos] + old_source[pos + 1:]

                        if not self.is_route_feasible(new_source):
                            continue

                        old_distance = self.calculate_route_distance(old_source)
                        old_distance += self.calculate_route_distance(old_target)

                        for insert_pos in range(len(old_target) + 1):
                            new_target = old_target[:insert_pos] + [customer_id] + old_target[insert_pos:]

                            if not self.is_route_feasible(new_target):
                                continue

                            new_distance = self.calculate_route_distance(new_source)
                            new_distance += self.calculate_route_distance(new_target)

                            if new_distance < old_distance - 1e-9:
                                routes[source_idx] = new_source
                                routes[target_idx] = new_target
                                improved = True
                                break

                        if improved:
                            break

                    if improved:
                        break

                if improved:
                    break

        return routes
    # Tính độ thích nghi, quãng đường và số xe cho toàn bộ quần thể
    def cal_fitness_population(self):
        for individual in self.__population:
            individual.fitness, individual.distance = self.cal_fitness_individualV2(
                individual.customerList
            )

            routes = self.individualToRoute(individual.customerList)
            individual.vehicle_count = len(routes)

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

    # Tính fitness: quãng đường + phạt vi phạm thời gian, không phạt thời gian chờ
    # Tính fitness dựa trên các tuyến hợp lệ đã được tách bởi individualToRoute
    def cal_fitness_individualV2(self, individual):
        routes = self.individualToRoute(individual)

        total_distance = 0.0
        depot = self.customers[0]

        for sub_route in routes:
            if len(sub_route) == 0:
                continue

            last_customer_id = 0

            # Tính quãng đường đi qua từng khách trong tuyến
            for customer_id in sub_route:
                customer = self.customers[customer_id]
                last_customer = self.customers[last_customer_id]

                moving_distance = np.linalg.norm(
                    customer.xy_coord - last_customer.xy_coord
                )

                total_distance += moving_distance
                last_customer_id = customer_id

            # Khách cuối quay về kho
            last_customer = self.customers[last_customer_id]

            return_distance = np.linalg.norm(
                last_customer.xy_coord - depot.xy_coord
            )

            total_distance += return_distance

        # Vì individualToRoute đã đảm bảo ràng buộc,
        # fitness hiện tại chính là tổng quãng đường.
        total_fitness = total_distance

        return total_fitness, total_distance

    def cal_fitness_sub_route(self, route):
        sub_route_result = []
        for sub_route in route:
            fitness, _ = self.cal_fitness_individualV2(sub_route)
            sub_route_result.append(fitness)
        return sub_route_result

    # Chọn cá thể tốt: ưu tiên ít xe trước, sau đó mới ưu tiên quãng đường ngắn
    def selection(self):
        # Sắp xếp: ít xe hơn tốt hơn,
        # nếu cùng số xe thì quãng đường ngắn hơn tốt hơn
        self.__population.sort(
            key=lambda x: (
                x.vehicle_count,
                x.distance
            )
        )

        # Số cá thể tốt được giữ lại làm bố mẹ
        number_survivors = max(
            2,
            math.floor(self._individual * self._conserve_rate)
        )

        # Chỉ giữ lại các cá thể tốt nhất
        self.__population = self.__population[:number_survivors]

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

        working_clusters = [
            np.asarray(cluster, dtype=int)
            for cluster in clusters if len(cluster) > 0
        ]

        if len(working_clusters) <= 1:
            return working_clusters

        def get_cluster_info(cluster):
            coords = np.array(
                [self.customers[i].xy_coord for i in cluster],
                dtype=float
            )

            ready_times = np.array(
                [self.customers[i].readyTime for i in cluster],
                dtype=float
            )

            due_times = np.array(
                [self.customers[i].dueTime for i in cluster],
                dtype=float
            )

            time_midpoints = (ready_times + due_times) / 2.0

            spatial_center = np.mean(coords, axis=0)
            temporal_center = np.mean(time_midpoints)

            return spatial_center, temporal_center

        def cluster_distance(cluster1, cluster2):
            spatial1, temporal1 = get_cluster_info(cluster1)
            spatial2, temporal2 = get_cluster_info(cluster2)

            spatial_distance = np.linalg.norm(
                spatial1 - spatial2
            )

            temporal_distance = abs(
                temporal1 - temporal2
            )

            return spatial_distance, temporal_distance

        spatial_distances = []
        temporal_distances = []

        for i in range(len(working_clusters)):
            for j in range(i + 1, len(working_clusters)):
                spatial, temporal = cluster_distance(
                    working_clusters[i],
                    working_clusters[j]
                )

                spatial_distances.append(spatial)
                temporal_distances.append(temporal)

        if not spatial_distances:
            return working_clusters

        spatial_threshold = np.median(spatial_distances)
        temporal_threshold = np.median(temporal_distances)

        if spatial_threshold <= 0:
            spatial_threshold = 1.0

        if temporal_threshold <= 0:
            temporal_threshold = 1.0

        def merge_score(cluster1, cluster2):
            spatial, temporal = cluster_distance(
                cluster1,
                cluster2
            )

            spatial_score = spatial / spatial_threshold
            temporal_score = temporal / temporal_threshold

            return 0.5 * spatial_score + 0.5 * temporal_score

        initial_count = len(working_clusters)

        minimum_count = max(
            1,
            int(np.ceil(initial_count * 0.7))
        )

        while len(working_clusters) > minimum_count:
            best_pair = None
            best_score = float("inf")

            for i in range(len(working_clusters)):
                for j in range(i + 1, len(working_clusters)):
                    score = merge_score(
                        working_clusters[i],
                        working_clusters[j]
                    )

                    if score < best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None or best_score > 1.0:
                break

            i, j = best_pair

            merged = np.concatenate(
                (working_clusters[i], working_clusters[j])
            )

            working_clusters = [
                cluster
                for index, cluster in enumerate(working_clusters)
                if index != i and index != j
            ]

            working_clusters.append(merged)

        return working_clusters
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

    # Sinh thế hệ mới: lai ghép và đột biến được xử lý độc lập
    def hybird(self):
        if len(self.__population) < 2:
            return

        while len(self.__population) < self._individual:

            # Chọn ngẫu nhiên 2 cá thể bố mẹ
            dad, mom = random.sample(self.__population, 2)

            # ==========================================
            # 1. Lai ghép (Crossover - trao đổi gen)
            # ==========================================
            if random.random() <= self._crossover_rate:
                gene_child_1, gene_child_2 = self.STPBCrossover(
                    np.copy(dad.customerList),
                    np.copy(mom.customerList)
                )
            else:
                # Không lai ghép thì sao chép bố mẹ
                gene_child_1 = np.copy(dad.customerList)
                gene_child_2 = np.copy(mom.customerList)

            # ==========================================
            # 2. Đột biến con thứ nhất
            # ==========================================
            if random.random() <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)

            # ==========================================
            # 3. Đột biến con thứ hai
            # ==========================================
            if random.random() <= self._mutation_rate:
                gene_child_2 = self.mutation(gene_child_2)

            # ==========================================
            # 4. Thêm con vào quần thể
            # ==========================================
            child1 = Individual(customerList=gene_child_1)
            self.__population.append(child1)

            if len(self.__population) < self._individual:
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

    # ==================== GA + Relocate + Swap + Route Elimination ====================
    def fit_allClusters(self, clusters):
        print(f"Số cluster K-means: {len(clusters)}")

        print("\n--- TRƯỚC KHI GHÉP CLUSTER ---")
        self.diagnose_clusters(clusters)

        clusters = self.re_cluster_by_timewindow(clusters)

        print(f"Số cluster sau khi ghép: {len(clusters)}")
        print(
            "Kích thước cluster:",
            [len(cluster) for cluster in clusters]
        )

        print("\n--- SAU KHI GHÉP CLUSTER ---")
        self.diagnose_clusters(clusters)

        for i in range(len(clusters)):
            self.fit(cluster=clusters[i])

        # ==========================================
        # Bước 3: Gom tất cả các tuyến từ các cụm
        # thành một danh sách chung
        # ==========================================
        all_routes = []

        for cluster_routes in self.best_route_global:
            for route in cluster_routes:
                all_routes.append(list(route))

        print(f"[1] Sau GA: {len(all_routes)} xe")

        # ==========================================
        # Bước 4: Relocate
        # Chuyển khách giữa các tuyến nhằm
        # giảm số lượng xe
        # ==========================================
        improved_routes = self.relocate_routes(
            all_routes
        )

        print(f"[2] Sau Relocate: {len(improved_routes)} xe")
        # ==========================================
        # Bước 5: Swap
        # Hoán đổi khách giữa các tuyến nhằm
        # cải thiện cấu trúc và giảm quãng đường
        # ==========================================
        swapped_routes = self.swap_between_routes(
            improved_routes
        )

        print(f"[3] Sau Swap: {len(swapped_routes)} xe")

        ordered_routes = self.optimize_all_route_orders(
            swapped_routes
        )

        print(f"[4] Sau Optimize: {len(ordered_routes)} xe")
        # ==========================================
        # Bước 6: Route Elimination
        # Sau khi Swap thay đổi cấu trúc tuyến,
        # thử loại bỏ hoàn toàn một tuyến
        # ==========================================

        optimized_routes = self.eliminate_routes(ordered_routes)
        print(f"[5] Sau Eliminate: {len(optimized_routes)} xe")
        optimized_routes = self.optimize_all_route_orders(optimized_routes)

        optimized_routes = self.relocate_for_distance(optimized_routes)
        optimized_routes = self.optimize_all_route_orders(optimized_routes)

        before_star = sum(self.calculate_route_distance(r) for r in optimized_routes)

        optimized_routes = self.two_opt_star(optimized_routes)

        after_star = sum(self.calculate_route_distance(r) for r in optimized_routes)

        print(f"Trước 2-opt*: {len(optimized_routes)} xe | Distance = {before_star:.2f}")
        print(f"Sau 2-opt*:   {len(optimized_routes)} xe | Distance = {after_star:.2f}")

        validation = self.validate_solution(optimized_routes)

        self.diagnose_routes(optimized_routes)
        print("========== KIỂM TRA NGHIỆM ==========")
        print(f"Số xe sử dụng: "f"{validation['vehicle_count']}/"f"{validation['vehicle_limit']}")
        print(f"Ràng buộc số xe: "f"{validation['vehicle_limit_feasible']}")
        print(f"Hợp lệ toàn bộ: {validation['is_valid']}")
        print(f"Số khách dự kiến: {validation['expected_customer_count']}")
        print(f"Số khách thực tế: {validation['actual_customer_count']}")
        print(f"Route không hợp lệ: {validation['invalid_routes']}")
        print(f"Khách bị trùng: {validation['duplicate_customers']}")
        print(f"Khách bị thiếu: {validation['missing_customers']}")
        print(f"Khách ngoài phạm vi: {validation['extra_customers']}")
        print("======================================")
        # ==========================================
        # Bước 7: Tính lại tổng quãng đường
        # ==========================================
        total_distance = sum(
            self.calculate_route_distance(route)
            for route in optimized_routes
        )

        # ==========================================
        # Bước 8: Cập nhật kết quả cuối cùng
        # ==========================================
        self.best_route_global = optimized_routes
        self.route_count_global = len(optimized_routes)
        self.best_distance_global = total_distance
        self.best_fitness_global = total_distance

        # ==========================================
        # Bước 9: Trả kết quả
        # ==========================================
        return (
            self.best_fitness_global,
            self.best_route_global,
            self.best_distance_global,
            self.route_count_global,
            self.process_time
        )

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