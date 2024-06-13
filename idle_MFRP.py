from tkinter import *
import time, math, random, heapq
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import csv

SCALE = 0.5
FPS = 3
WINDOW = 100
LOOK_BACK = 20

window = Tk()
window.title("Oht_Simulator")
window.geometry("500x500")
canvas = Canvas(window, width = 1000, height = 1000, bg = "white")
canvas.pack()

global frame

def forcast_one_step(model):
    fc, conf = model.predict(n_periods=1, return_conf_int=True)
    return fc.tolist()[0], np.asarray(conf).tolist()[0]

def create_dataset(dataset, look_back):
    X, Y = [np.array([0 for i in range(look_back)]) for j in range(look_back)], [np.int64(0) for k in range(look_back)]

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def create_magged_oval(x, y, r, color):
    return canvas.create_oval(
        SCALE * x - r,
        SCALE * y - r,
        SCALE * x + r,
        SCALE * y + r,
        fill = color)
    
def create_magged_rectangle(x, y, width, height, color):
    return canvas.create_rectangle(
        SCALE * x - width / 2,
        SCALE * y - height / 2,
        SCALE * x + width / 2,
        SCALE * y + height / 2,
        fill = color)
    
def create_magged_line(x1, y1, x2, y2, color):
    return canvas.create_line(
        SCALE * x1,
        SCALE * y1,
        SCALE * x2,
        SCALE * y2,
        fill = color)
    
def create_magged_arc(x, y, r, start_angle, extent_angle, color):
    return canvas.create_arc(
        SCALE * (x - r),
        SCALE * (y - r),
        SCALE * (x + r),
        SCALE * (y + r),
        start = start_angle,
        extent = extent_angle,
        style = ARC,
        outline = color)

def magged_moveto(id, x, y, r):
    canvas.moveto(id, SCALE * x - r, SCALE * y - r)

class Sim:
    global vehicles
    vehicles = []

    global deliveries
    deliveries = []

    global MAX_DELIVERY_NUM
    MAX_DELIVERY_NUM = 1

    global frame
    frame = 0
    def __init__(self, m, p):
        self.m = m
        self.p = p

        self.df = pd.read_csv('./new3.csv')
        deliveries.append(('time', 'PU', 'DO', 'vehicle', 'pickup', 'done'))
        self.xlcount = 0

        m.draw_map()
        self.vehicle_generator(40)

        self.weight_initiator()

        while(True):
            global frame
            if frame % 10 == 0:
                p.table_updater(self.m)

            if frame % WINDOW == 0:
                p.idleer_DM(self.m)
            
            self.delivery_generator()
            for vhc in vehicles:
                vhc.step(self.m, self.p)
            
            window.update()
            
            frame += 1
            
            if frame > self.df.iloc[-1, 1] + 500:
                break

        with open("newDM3.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(deliveries)
        
    def vehicle_generator(self, num):
        occupied_nodes = set()
        non_node_num = 0
        #무작위 생성
        for i in range(num):
            #no vehicle on the road

            if len(occupied_nodes) == len(self.m.roadss):
                occupied_nodes.discard(vehicles[non_node_num].location)
                vehicles[non_node_num].random_move(self.m)
                non_node_num += 1

            while True:
                chosen_node = random.choice(list(self.m.roadss.keys()))
                if chosen_node not in occupied_nodes:
                    break
            
            x, y = chosen_node.coordinate
            vehicles.append(Vehicle(x, y, chosen_node))

            occupied_nodes.add(chosen_node)

    def delivery_generator(self):
        global frame
        while(self.df.iloc[self.xlcount, 1] <= frame and self.xlcount < len(self.df) - 1):
        #무작위 생성
            chosen_start = self.m.stations[self.df.iloc[self.xlcount, 2]]
            chosen_end = self.m.stations[self.df.iloc[self.xlcount, 3]]
            Delivery(chosen_start, chosen_end, 1, self.p, self.m)
            
            self.xlcount += 1

    def weight_initiator(self):
        for vehicle in vehicles:
            vehicle_one = vehicle
            break
        for roads in self.m.roadss.values():
            for road in roads.values():
                if type(road) == Curve:
                    road.weight = (road.distance(road.start.coordinate) / vehicle_one.curved_speed) * FPS

                elif type(road) == Line:
                    road.weight = (road.distance(road.start.coordinate) / vehicle_one.linear_speed) * FPS
            
        
class Map:
    def __init__(self):
        self.roadss = {}
        self.nodes = {}
        self.stations = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_station(self, station):
        self.add_node(station)
        self.stations[station.name] = station

    def add_road(self, road):
        if road.start not in self.roadss:
            self.roadss[road.start] = {}
        self.roadss[road.start][road.end] = road
        
    def draw_map(self):
        for node in self.nodes.values():
            create_magged_oval(node.coordinate[0], node.coordinate[1], 2, "black")
        
        for station in self.stations.values():
            create_magged_oval(station.coordinate[0], station.coordinate[1], 2, "blue")
        
        for roads in self.roadss.values():
            for road in roads.values():
                road.draw_road()

class Node:
    def __init__(self, name, coordinate):
        self.name = name
        self.coordinate = coordinate
        
class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
        self.weight = 0
    
    def draw_road(self):
        create_magged_line(self.start.coordinate[0], self.start.coordinate[1], self.end.coordinate[0], self.end.coordinate[1], "black")
    
    def distance(self, coordinate):
        return math.dist(coordinate, self.end.coordinate)
    
    def step_coordinate(self, coordinate, displacement):
        if self.distance(coordinate) < displacement:
            return displacement - self.distance(coordinate)
        else:
            denom = math.sqrt((self.end.coordinate[0] - coordinate[0])**2 + (self.end.coordinate[1] - coordinate[1])**2)
            new_x = coordinate[0] + (displacement * ((self.end.coordinate[0] - coordinate[0]) / denom))
            new_y = coordinate[1] + (displacement * ((self.end.coordinate[1] - coordinate[1]) / denom))
            return new_x, new_y

class Curve:
    def __init__(self, start, end, start_angle, extent_angle):
        self.start = start
        self.end = end
        self.start_angle = start_angle
        self.extent_angle = extent_angle

        start_angle = math.radians(start_angle)
        extent_angle = math.radians(extent_angle)

        if math.tan((-1)*start_angle) == math.tan(math.pi) or math.tan((-1)*start_angle) == math.tan(math.pi * -1):
            self.center_x = (start.coordinate[0] + end.coordinate[0]) / 2
        
        else:
            self.center_x = round(((start.coordinate[0] * math.tan(start_angle)) - (end.coordinate[0] * math.tan(extent_angle + start_angle)) - end.coordinate[1]) / \
                            (math.tan(start_angle) - math.tan(extent_angle + start_angle)), 5)
        if math.tan((-1)*start_angle) == math.tan(math.pi / 2) or math.tan((-1)*start_angle) == math.tan(math.pi / -2):
            self.center_y = round(math.tan((-1)*(start_angle + extent_angle)) * (self.center_x - end.coordinate[0]) + end.coordinate[1], 5)
        else:
            self.center_y = round(math.tan((-1)*start_angle) * (self.center_x - start.coordinate[0]) + start.coordinate[1], 5)
        self.radius = math.dist([self.center_x, self.center_y], start.coordinate)

        self.weight = 0
    
    def draw_road(self):
        x, y = self.start.coordinate
        p, q = self.end.coordinate

        top_left_x = self.center_x - self.radius
        top_left_y = self.center_y - self.radius
        bottom_right_x = self.center_x + self.radius
        bottom_right_y = self.center_y + self.radius

        create_magged_arc(self.center_x, self.center_y, self.radius, self.start_angle, self.extent_angle, "black")

    def distance(self, coordinate):
        point1 = coordinate
        point2 = self.end.coordinate

        radius = math.sqrt((self.start.coordinate[0] - self.center_x) ** 2 + (self.start.coordinate[1] - self.center_y) ** 2)

        theta_m = math.atan2(point1[1] - self.center_y, point1[0] - self.center_x)
        theta_p = math.atan2(point2[1] - self.center_y, point2[0] - self.center_x)
        arc_length_angle = theta_p - theta_m
        if arc_length_angle < 0:
            arc_length_angle += 2 * math.pi

        return arc_length_angle * radius

    def step_coordinate(self, coordinate, displacement):
        if self.distance(coordinate) < displacement:
            return displacement - self.distance(coordinate)
        else:
            x, y = coordinate

            start_radian = math.acos((x - self.center_x) / self.radius) 
            if y < self.center_y:
                start_radian = 2 * math.pi - start_radian
            start_angle = math.degrees(start_radian)

            end_angle = 180 / math.pi / self.radius * displacement + start_angle
            
            end_radian = math.radians(end_angle)

            new_x = self.center_x + self.radius * math.cos(end_radian)
            new_y = self.center_y + self.radius * math.sin(end_radian)

            return new_x, new_y


class Vehicle:
    width = 0.7
    height = 0.45
    linear_speed = 4
    curved_speed = 0.8
    acceleration = 1.6
    deceleration = 2.4
    capacity = 1
    safe_distance = 4

    weight_rambda = 0.5

    def __init__(self, x, y, location):
        self.id = create_magged_oval(x, y, 5, "red")
        self.speed = 0
        self.coordinate = [x, y]
        self.location = location
        self.state = "idle"
        self.route = []
        self.departure = None
        self.arrival = None
        self.start_frame = 0
        

    def accel_or_decel(self, map, policy):
        look_ahead = self.look_ahead(map, policy)

        if type(self.location) == Line:
            target_speed = self.linear_speed
        else:
            target_speed = self.curved_speed
        
        if (look_ahead == "crash") or \
            (look_ahead == "curve_ahead") or \
            (look_ahead == "almost_there") and self.speed > 0:
            self.speed = self.speed - (FPS * self.deceleration)
            if self.speed < 0:
                self.speed = 0
            elif look_ahead == "curve_ahead" and self.look_ahead != "almost_there" and self.speed < self.curved_speed:
                self.speed = self.curved_speed

        elif look_ahead == "clear" and self.speed < target_speed:
            self.speed = self.speed + (FPS * self.acceleration)
            if self.speed > target_speed:
                self.speed = target_speed
        else:
            pass

    def move(self, map, policy):
        init_speed = self.speed
        self.accel_or_decel(map, policy)
        displacement = (init_speed + self.speed) / 2 * FPS
        #displace by the path
        self.next_step(displacement, map, policy)
               
    def next_step(self, displacement, map, policy):
        if displacement == 0:
            return
        #일단 step
        step_result = self.location.step_coordinate(self.coordinate, displacement)
        if type(step_result) == float or type(step_result) == int:
            self.coordinate = self.location.end.coordinate
            displacement = step_result
        
        else:
            self.coordinate = step_result

        #node pass
        if self.coordinate == self.location.end.coordinate:
            self.passed(map, policy, self.location, frame)

            if len(self.route) == 0:
                self.location = self.location.end
                if self.state == "delivering":
                    self.done(map, policy)
                elif self.state == "assigned":
                    self.picking(policy, map)
                elif self.state == "relocating":
                    self.state = "idle"
                    self.speed = 0
                #elif self.state == "idle":
                    #self.route = policy.idleer_C(self.location, map)
                    #self.location = map.roadss[self.location][self.route[0]]

            else:
                self.next_step(displacement, map, policy)                
                        
        magged_moveto(self.id, self.coordinate[0], self.coordinate[1], 5)
    
    def picking(self, p, map):
        self.state = "delivering"
        self.speed = 0
        self.route = p.router(self.departure, self.arrival)
        self.route.pop(0)
        self.location = map.roadss[self.location][self.route[0]]
        self.pickup_time = frame

        print("p", self.departure.name, self.arrival.name)

    def done(self, map, p):
        deliveries.append((self.assign_time, self.departure.name, self.arrival.name, self.id, self.pickup_time, frame))
        self.assign_time = None
        self.pickup_time = None

        self.state = "idle"
        self.departure = None
        self.arrival = None 
        self.speed = 0
        #self.random_move(map)
        #self.route = p.idleer_C(self.location, map)
        #self.location = map.roadss[self.location][self.route[0]]
        print("d")

    def passed(self, map, p, road, frame):
        #시간 기록
        if self.start_frame != 0:
            road.weight = (1 - self.weight_rambda) * road.weight + (self.weight_rambda * FPS * (frame - self.start_frame))
        #경로 재탐색
        if len(self.route) > 1 and self.state != "idle":
            self.route = p.router(self.route[0], self.route[-1])
            self.location = map.roadss[self.route[0]][self.route[1]]

        #도착 시x
        elif len(self.route) == 1:
            pass

        elif self.state == "idle":
            self.location = map.roadss[self.route[0]][self.route[1]]
        
        self.start_frame = frame
        self.route.pop(0)


    def look_ahead(self, map, policy):
        #look ahead the whole path.

        self_deleted_vehicles = vehicles.copy()
        self_deleted_vehicles.remove(self)

        #이번 루트 또는 다음 route에 차가 있고, 거리가 가까울 경우
        for vehicle in self_deleted_vehicles:
            if (type(vehicle.location) == Node and (vehicle.location == self.location.end and self.location.distance(self.coordinate) < self.safe_distance) or \
                (type(vehicle.location) == Line and (vehicle.location == self.location and 0 < (self.location.distance(self.coordinate) - self.location.distance(vehicle.coordinate)) < self.safe_distance))):
                #앞 차 움직이기
                if vehicle.state == "idle":
                    vehicle.state = "relocating"
                    vehicle.route = policy.idleer(vehicle.location, map)
                    vehicle.location = map.roadss[vehicle.location][vehicle.route[0]]
                return "crash"
    
        # #route가 비었고, 남은 거리가 적음
        zero_decelerated_distance = self.speed * FPS / 2
        if len(self.route) == 1 and self.location.distance(self.coordinate) < zero_decelerated_distance:
            return "almost_there"
        
        # #다음 Route가 Curve이고 남은 거리 감안했을 때 줄여야할 경우
        curve_decelerated_distance = ((self.speed ** 2) - (self.curved_speed ** 2)) / (2 * FPS * self.deceleration)
        if len(self.route) > 1 and type(map.roadss[self.route[0]][self.route[1]]) == "Curve" and self.location.distance(self.coordinate) < curve_decelerated_distance:
            return "curve_ahead"
    
        # #straight일 경우
        return "clear"

    def step(self, map, policy):
        if self.state != "idle":
            self.move(map, policy)
        
    
    def random_move(self, map):
        chosen_road = random.choice(list(map.roadss[self.location].values()))    
        max_displacement = 0.25 * (FPS**2) * self.acceleration

        self.location = chosen_road
        self.coordinate = chosen_road.step_coordinate(chosen_road.start.coordinate, max_displacement)

        
        

class Policy:
    def __init__(self, map):
        self.dist = {}
        self.last_node = {}
        self.next_node = {}
        self.table_updater(map)
        #D & DD용
        self.demand_forecast = {}

        #D용

        #DM용
        df = pd.read_csv('new3.csv')
        self.df0 = {}
        self.model = {}
        self.X = {}
        self.Y = {}

        for station in map.stations.keys():
            self.df0[station] = df.loc[df['PULocationID'] == station]
            self.df0[station]['freq'] = self.df0[station]['tpep_pickup_datetime']//WINDOW
            self.df0[station] = self.df0[station].groupby('freq')['PULocationID'].count().reset_index()
        
            for i in range(0, self.df0[station]['freq'].max()+1):
                if self.df0[station].loc[self.df0[station]['freq'] == i].empty == True:
                    new = pd.DataFrame({'freq':i, 'PULocationID':0}, index = [i])
                    self.df0[station] = pd.concat([self.df0[station].iloc[:i], new, self.df0[station].iloc[i:]], ignore_index=True)

            dataset = self.df0[station]['PULocationID'].values.reshape(-1, 1)

            # Split the data into train and test sets

            self.X[station], self.Y[station] = create_dataset(dataset, LOOK_BACK)

            # Reshape input to be [samples, time steps, features]
            self.X[station] = np.reshape(self.X[station], (self.X[station].shape[0], 1, self.X[station].shape[1]))


            # Create and fit the LSTM network
            self.model[station] = Sequential()
            self.model[station].add(LSTM(4, input_shape=(1, LOOK_BACK)))
            self.model[station].add(Dense(1))
            self.model[station].compile(loss='mean_squared_error', optimizer='adam')

            datum = self.X[station][0].reshape(1, 1, LOOK_BACK)
            prediction = self.model[station].predict(datum)[0][0]
            Y = self.Y[station][0].reshape(1, 1)
            
            self.model[station].fit(datum, Y, epochs=5, batch_size=1, verbose=2)


# one point forcast 함수 정의, 신뢰구간도 함께 담아보기

    def assigner(self, delivery, map):
        # vehicles[0].route = [delivery.departure, delivery.arrival]
        departure = delivery.departure
        #traverse roads using breadth-first search starting from departure
        queue = [departure]
        visited = set()
        available_vehicles = []
        availables = []
            
        while queue and len(queue) > 0:
            end = queue.pop(0)
            if type(end) == Node:
                for end in map.roadss[end].keys():
                    for vehicle in vehicles:
                        if vehicle.location == end and vehicle.state == "idle" or vehicle.state == "relocating":
                            available_vehicles.append(vehicle)
                            availables.append(end)
                    if end not in visited:
                        queue.append(end)
                        visited.add(end)
                
                if available_vehicles:
                    break

            else:
                for roads in list(map.roadss.values()):
                    road = roads.get(end)
                    if road:
                        for vehicle in vehicles:
                            if vehicle.location == road and vehicle.state == "idle" or vehicle.state == "relocating":
                                available_vehicles.add(vehicle)
                                availables.append(road)
                        if road.start not in visited:
                            queue.append(road.start)
                            visited.add(road.start)
                    
                if available_vehicles:
                    break
                
        if not available_vehicles:
            return
        else:
            chosen_vehicle = random.choice(available_vehicles)

            #routing 필요
            if type(chosen_vehicle.location) == Node:
                chosen_vehicle.route = self.router(chosen_vehicle.location, departure)
            else:
                chosen_vehicle.route = self.router(chosen_vehicle.location.end, departure)

            chosen_vehicle.state = "assigned"
            chosen_vehicle.departure = departure
            chosen_vehicle.arrival = delivery.arrival
            chosen_vehicle.assign_time = frame

            if type(chosen_vehicle.location) == Node:
                chosen_vehicle.route.pop(0)
                if len(chosen_vehicle.route) > 0:
                    chosen_vehicle.location = map.roadss[chosen_vehicle.location][chosen_vehicle.route[0]]
                else:
                    chosen_vehicle.picking(self, map)

                           

    def table_updater(self, map):

        for road_i in map.roadss.keys():
            self.dist[road_i] = {}
            self.last_node[road_i] = {}
            self.next_node[road_i] = {}
            for road_j in map.roadss.keys():
                if road_i == road_j:
                    self.dist[road_i][road_j] = 0
                    self.last_node[road_i][road_j] = road_j
                elif map.roadss[road_i].get(road_j) != None:
                    self.dist[road_i][road_j] = map.roadss[road_i][road_j].weight
                    self.last_node[road_i][road_j] = road_j
                else:
                    self.dist[road_i][road_j] = math.inf
                    self.last_node[road_i][road_j] = None
    
        for node_k in map.roadss.keys():
            for node_i in map.roadss.keys():
                for node_j in map.roadss.keys():
                    if self.dist[node_i][node_j] > (self.dist[node_i][node_k] + self.dist[node_k][node_j]):
                        self.dist[node_i][node_j] = (self.dist[node_i][node_k] + self.dist[node_k][node_j])
                        self.last_node[node_i][node_j] = self.last_node[node_i][node_k]
                
    def router(self, departure, arrival):
        
        route = [departure]
        while departure != arrival:
            departure = self.last_node[departure][arrival]
            route.append(departure)

        return route

    #아무것도 안하기
    def idleer(self, node, map):
        unoccupied_road = []
        for road in map.roadss[node].values():
            unoccupied_road.append(road)
            for vehicle in vehicles:
                if vehicle.location == road or vehicle.location == road.end:
                    unoccupied_road.pop()
                    break
            
        if unoccupied_road:
            return [random.choice(unoccupied_road).end]
        else:
            return [random.choice(list(map.roadss[node].values())).end]
                
            
        
        return

    #외곽 순환
    def idleer_C(self, node, map):
        #외곽 순환

        outer_path = [107, 137, 170, 162, 229, 140, 262, 75, 43, 24, 151, 238, 239, 143, 50, 48, 68, 186, 90, 234]

        queue = [node]
        while queue:
            start = queue.pop(0)
            for end in map.roadss[start].keys():
                if end.name in outer_path:
                    break
                else:
                    queue.append(end)

        route = self.router(node, end)
        route.pop(0)

        ext_path = outer_path + outer_path[0:0]
        first_index = ext_path.index(end.name)
        for i in range(first_index + 1, len(ext_path)):
            route.append(map.stations[ext_path[i]])

        return route
                

#Deterministic하게
    def idleer_D():
        #Demand Forecasting
        pass

        #위치 이동

    def demand_forecaster(self, map):
        for station in map.stations.keys():
            try:
                datum = self.X[station][frame//WINDOW-1]
                datum = datum.reshape(1, 1, LOOK_BACK)
                
                prediction = self.model[station].predict(datum)[0][0]
                self.demand_forecast[map.stations[station]] = prediction

                Y = self.Y[station][frame//WINDOW-1].reshape(1, 1)
                self.model[station].fit(datum, Y, epochs=5, batch_size=1, verbose=2)
            except:
                pass

    #디맨드만 딥러닝으로
    def idleer_DM(self, map):
        #Demand Forecasting
        if frame > 0:
            self.demand_forecaster(map)

            #위치 이동
            locations = []
            idle_vehicle = []
            vehicle_location = {}

            for vehicle in vehicles:
                
                if vehicle.state == "idle":
                    idle_vehicle.append(vehicle)
                    available = []
                    if type(vehicle.location) == Node:
                        vehicle_location[vehicle] = vehicle.location
                    else:
                        vehicle_location[vehicle] = vehicle.location.end

                    for node in self.dist[vehicle_location[vehicle]].keys():
                        if self.dist[vehicle_location[vehicle]][node] < WINDOW:
                            available.append(node)
                    locations.append(available)
            

            from itertools import product

            def all_possible_lists(data):
                for element in product(*data):
                    yield list(element)

            maxi = -1
            current_sol = None
            for possible_list in all_possible_lists(locations):
                if self.eval_func(possible_list, map) > maxi:
                    maxi = self.eval_func(possible_list, map)
                    current_sol = possible_list
            
            if not current_sol:
                return
            else:
                for i, vehicle in enumerate(idle_vehicle):
                    vehicle.state = "relocating"
                    if type(vehicle.location) == Node:
                        vehicle.route = self.router(vehicle.location, current_sol[i])
                        if vehicle.route[0] == vehicle.location:
                            vehicle.route.pop(0)
                            vehicle.state = "idle"
                        else:
                            vehicle.location = map.roadss[vehicle.location][vehicle.route[0]]
                    else:
                        vehicle.route = self.router(vehicle.location.end, current_sol[i])

    def eval_func(self, sol, map):
        result = 0
        for start_node in sol:
            result += self.demand_forecast[start_node]
            for end_node in map.roadss[start_node].keys():
                result += self.demand_forecast[end_node]
        return result
    
    
    #딥러닝
    def idleer_L():
        #Random
        pass

class Delivery:
    def __init__(self, departure, arrival, weight, policy, map):
        self.departure = departure
        self.arrival = arrival
        self.weight = weight
        
        policy.assigner(self, map)
        

def main():
    
    m = Map()

    # a_1 = Node('a_1', [20, 40])
    # a_2 = Node('a_2', [40, 20])
    # a_3 = Node('a_3', [960, 20])
    # a_4 = Node('a_4', [980, 40])
    # a_5 = Node('a_5', [980, 500])
    # a_6 = Node('a_6', [980, 960])
    # a_7 = Node('a_7', [960, 980])
    # a_8 = Node('a_8', [40, 980])
    # a_9 = Node('a_9', [20, 960])
    # a_10 = Node('a_10', [20, 500])
    # m.add_station(a_1)
    # m.add_node(a_2)
    # m.add_node(a_3)
    # m.add_node(a_4)
    # m.add_node(a_5)
    # m.add_station(a_6)
    # m.add_node(a_7)
    # m.add_node(a_8)
    # m.add_node(a_9)
    # m.add_node(a_10)


    # b_1 = Node('b_1', [30, 40])
    # b_2 = Node('b_2', [40, 30])
    # b_3 = Node('b_3', [460, 30])
    # b_4 = Node('b_4', [470, 40])
    # b_5 = Node('b_5', [470, 500])
    # b_6 = Node('b_6', [470, 960])
    # b_7 = Node('b_7', [460, 970])
    # b_8 = Node('b_8', [40, 970])
    # b_9 = Node('b_9', [30, 960])
    # b_10 = Node('b_10', [30, 500])

    # m.add_station(b_1)
    # m.add_node(b_2)
    # m.add_node(b_3)
    # m.add_node(b_4)
    # m.add_node(b_5)
    # m.add_station(b_6)
    # m.add_node(b_7)
    # m.add_node(b_8)
    # m.add_node(b_9)
    # m.add_node(b_10)

    # c_1 = Node('c_1', [530, 40])
    # c_2 = Node('c_2', [540, 30])
    # c_3 = Node('c_3', [960, 30])
    # c_4 = Node('c_4', [970, 40])
    # c_5 = Node('c_5', [970, 500])
    # c_6 = Node('c_6', [970, 960])
    # c_7 = Node('c_7', [960, 970])
    # c_8 = Node('c_8', [540, 970])
    # c_9 = Node('c_9', [530, 960])
    # c_10 = Node('c_10', [530, 500])
    # m.add_station(c_1)
    # m.add_node(c_2)
    # m.add_node(c_3)
    # m.add_node(c_4)
    # m.add_node(c_5)
    # m.add_station(c_6)
    # m.add_node(c_7)
    # m.add_node(c_8)
    # m.add_node(c_9)
    # m.add_node(c_10)

    # m.add_road(Curve(a_1, a_2, 180, -90))
    # m.add_road(Line(a_2, a_3))
    # m.add_road(Curve(a_3, a_4, 90, -90))
    # m.add_road(Line(a_4, a_5))
    # m.add_road(Line(a_5, a_6))
    # m.add_road(Curve(a_6, a_7, 0, -90))
    # m.add_road(Line(a_7, a_8))
    # m.add_road(Curve(a_8, a_9, -90, -90))
    # m.add_road(Line(a_9, a_10))
    # m.add_road(Line(a_10, a_1))

    # m.add_road(Curve(b_1, b_2, 180, -90))
    # m.add_road(Line(b_2, b_3))
    # m.add_road(Curve(b_3, b_4, 90, -90))
    # m.add_road(Line(b_4, b_5))
    # m.add_road(Line(b_5, b_6))
    # m.add_road(Curve(b_6, b_7, 0, -90))
    # m.add_road(Line(b_7, b_8))
    # m.add_road(Curve(b_8, b_9, -90, -90))
    # m.add_road(Line(b_9, b_10))
    # m.add_road(Line(b_10, b_1))

    # m.add_road(Curve(c_1, c_2, 180, -90))
    # m.add_road(Line(c_2, c_3))
    # m.add_road(Curve(c_3, c_4, 90, -90))
    # m.add_road(Line(c_4, c_5))
    # m.add_road(Line(c_5, c_6))
    # m.add_road(Curve(c_6, c_7, 0, -90))
    # m.add_road(Line(c_7, c_8))  
    # m.add_road(Curve(c_8, c_9, -90, -90))
    # m.add_road(Line(c_9, c_10))
    # m.add_road(Line(c_10, c_1))

    # m.add_road(Curve(a_10, b_10, 180, -180))
    # m.add_road(Curve(b_5, c_10, 180, -180))
    # m.add_road(Curve(c_5, a_5, 180, -180))
    

    n_246 = Node(246, [200, 200])
    n_50 = Node(50, [300, 200])
    n_143 = Node(143, [400, 200])
    n_239 = Node(239, [500, 200])
    n_238 = Node(238, [600, 200])
    n_151 = Node(151, [700, 200])
    n_24 = Node(24, [800, 200])
    n_68 = Node(68, [200, 300])
    n_48 = Node(48, [300, 300])
    n_142 = Node(142, [400, 300])
    n_90 = Node(90, [175, 400])
    n_186 = Node(186, [225, 400])
    n_100 = Node(100, [275, 400])
    n_230 = Node(230, [325, 400])
#
    n_43 = Node(43, [600, 425])
    n_234 = Node(234, [175, 500])
    n_164 = Node(164, [250, 500])
    n_161 = Node(161, [325, 500])
    n_107 = Node(107, [175, 600])
    n_170 = Node(170, [250, 600])
    n_162 = Node(162, [325, 600])
    n_237 = Node(237, [450, 575])
    n_236 = Node(236, [550, 575])
    n_75 = Node(75, [750, 650])
    n_224 = Node(224, [175, 700])
    n_137 = Node(137, [225, 700])
    n_233 = Node(233, [275, 700])
    n_229 = Node(229, [325, 700])
    n_141 = Node(141, [450, 650])
    n_263 = Node(263, [550, 650])
    n_140 = Node(140, [450, 750])
    n_262 = Node(262, [550, 750])

    m.add_station(n_246)
    m.add_station(n_50)
    m.add_station(n_143)
    m.add_station(n_239)
    m.add_station(n_238)
    m.add_station(n_151)
    m.add_station(n_24)
    m.add_station(n_68)
    m.add_station(n_48)
    m.add_station(n_142)
    m.add_station(n_90)
    m.add_station(n_186)
    m.add_station(n_100)
    m.add_station(n_230)
    m.add_station(n_43)
    m.add_station(n_234)
    m.add_station(n_164)
    m.add_station(n_161)
    m.add_station(n_107)
    m.add_station(n_170)
    m.add_station(n_162)
    m.add_station(n_237)
    m.add_station(n_236)
    m.add_station(n_75)
    m.add_station(n_224)
    m.add_station(n_137)
    m.add_station(n_233)
    m.add_station(n_229)
    m.add_station(n_141)
    m.add_station(n_263)
    m.add_station(n_140)
    m.add_station(n_262)

    m.add_road(Line(n_68, n_246))
    m.add_road(Line(n_246, n_50))
    m.add_road(Line(n_143, n_50))
    m.add_road(Line(n_142, n_143))
    m.add_road(Line(n_239, n_143))
    m.add_road(Line(n_239, n_142))
    m.add_road(Line(n_238, n_239))
    m.add_road(Line(n_151, n_238))
    m.add_road(Line(n_24, n_151))
    m.add_road(Line(n_142, n_43))
    m.add_road(Line(n_43, n_239))
    m.add_road(Line(n_238, n_43))
    m.add_road(Line(n_43, n_151))
    m.add_road(Line(n_43, n_24))
    m.add_road(Line(n_48, n_68))
    m.add_road(Line(n_50, n_48))
    m.add_road(Line(n_142, n_48))
    m.add_road(Line(n_68, n_186))
    m.add_road(Line(n_90, n_68))
    m.add_road(Line(n_100, n_48))
    m.add_road(Line(n_48, n_230))
    m.add_road(Line(n_90, n_234))
    m.add_road(Line(n_186, n_164))
    m.add_road(Line(n_100, n_164))
    m.add_road(Line(n_161, n_230))
    m.add_road(Line(n_230, n_43))
    m.add_road(Line(n_43, n_161))
    m.add_road(Line(n_186, n_90))
    m.add_road(Line(n_186, n_100))
    m.add_road(Line(n_100, n_230))
    m.add_road(Line(n_164, n_234))
    m.add_road(Line(n_161, n_164))
    m.add_road(Line(n_234, n_107))
    m.add_road(Line(n_164, n_170))
    m.add_road(Line(n_162, n_161))
    m.add_road(Line(n_237, n_161))
    m.add_road(Line(n_237, n_162))
    m.add_road(Line(n_43, n_237))
    m.add_road(Line(n_236, n_43))
    m.add_road(Line(n_237, n_236))
    m.add_road(Line(n_236, n_75))
    m.add_road(Line(n_224, n_107))
    m.add_road(Line(n_107, n_170))
    m.add_road(Line(n_170, n_162))
    m.add_road(Line(n_107, n_137))
    m.add_road(Line(n_137, n_170))
    m.add_road(Line(n_170, n_233))
    m.add_road(Line(n_162, n_233))
    m.add_road(Line(n_162, n_229))
    m.add_road(Line(n_237, n_141))
    m.add_road(Line(n_229, n_141))
    m.add_road(Line(n_263, n_236))
    m.add_road(Line(n_75, n_43))
    m.add_road(Line(n_137, n_224))
    m.add_road(Line(n_233, n_137))
    m.add_road(Line(n_229, n_233))
    m.add_road(Line(n_229, n_140))
    m.add_road(Line(n_141, n_140))
    m.add_road(Line(n_141, n_263))
    m.add_road(Line(n_75, n_263))
    m.add_road(Line(n_140, n_262))
    m.add_road(Line(n_263, n_262))
    m.add_road(Line(n_262, n_75))


    p = Policy(m)

    sim = Sim(m, p)

    window.mainloop()

main()