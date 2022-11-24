import random
import copy
import time
import sys
import math
import tkinter  # //GUI模块
import threading
from functools import reduce
import pandas as pd

# 蚁群算法求解共享单车调度问题 VERSION 1.0
# 在不考虑自然调度的情况下，即某时刻的需求为静态，需要进行人工调度
# 以华南理工大学校园hello单车调度问题为例：
# 五山校区共有约600辆共享单车，全校分布约50个停车点（划分为21个停车区），其中有4个大型停车区
# 调度车辆共有2辆，满载容量为20（先不考虑调度人工成本）
# 默认货车从数组中的第一个站点发出

# 参数
(ALPHA, BETA, RHO, Q) = (1.0, 1.0, 0.5, 100)
# 站点数，蚁群
(city_num, ant_num) = (21,21)
# 输入
distance_x = [50,115,215,65,35,45,75,200,150,100,250,265,180,150,65,195,155,260,200,100,175]
distance_y = [45,135,95,130,100,160,460,65,100,60,150,215,250,380,530,530,525,485,375,205,145]

#bi为实际单车数-单车需求量的偏差值，bi>0供大于求->装上货车,bi<0供不应求->卸下货车
# 输入
bi=[0,5, 1, -18, 9, -28, -4, 1, 3, 5, -28, 9, 1, -6, 42, 10, -17, 8, -26, 12, 10]

service_bbb=[0,0.02, 0.004, 0.072, 0.036, 0.112, 0.016, 0.004, 0.012, 0.02, 0.112, 0.036, 0.004, 0.024, 0.167, 0.04, 0.068, 0.032, 0.104, 0.048, 0.04 ]

#MaxBike为货车最大负载量
MaxBike=40
# 站点距离和信息素
# 站点距离信息通过data_dist读入
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]

pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]


# ----------- 蚂蚁 -----------
class Ant(object):

    # 初始化
    def __init__(self, ID):

        self.ID = ID  # ID
        self.__clean_data()  # 初始化蚂蚁

    # 初始数据
    def __clean_data(self):

        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的站点
        self.CurrentBike=0 # 当前货车上的单车数
        self.not_visited_city = [True for i in range(city_num)] #站点是否已经访问过
        self.open_table_city = [True for i in range(city_num)]  # 探索站点的状态

        city_index = 0  # 初始出生点为第一个站点
        self.current_city = city_index
        self.path.append(city_index)
        self.not_visited_city[city_index] = False
        self.move_count = 1
        self.__calculate_open_table_city()

    #计算满足约束条件的备选列表
    def __calculate_open_table_city(self):
        for i in range(len(self.open_table_city)):
            if self.not_visited_city[i]==False:#过滤掉已经访问过的结点
                self.open_table_city[i]=False
            # #此处的约束可以尝试去除，即蚂蚁每到一个点，只需根据最短路径和用户满意度进行选择，而不是用车载容量进行判断
            # else:
            #     if (bi[i]>=0 and self.CurrentBike+bi[i]<=MaxBike) or (bi[i]<=0 and self.CurrentBike+bi[i]>=0):
            #         self.open_table_city[i]=True
            #     else:
            #         self.open_table_city[i] = False

    # 选择下一个站点
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个站点的概率
        total_prob = 0.0

        # 获取去下一个站点的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)

        # 轮盘选择站点
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 未从概率产生，顺序选择一个未访问站点
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break

        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明不可选择该站点
                next_city = random.randint(0, city_num - 1)

        # 返回下一个站点序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, city_num):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]

        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.not_visited_city[next_city] = False
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1
        # self.CurrentBike += bi[next_city]
        # 此处修改当前运载单车数量，应满足大于等于0，小于等于最大运载量
        if bi[next_city]>=0:
            if self.CurrentBike+bi[next_city]<=MaxBike:
                self.CurrentBike+=bi[next_city]
            else:
                self.CurrentBike=MaxBike
        else:
            if self.CurrentBike+bi[next_city]>=0:
                self.CurrentBike+=bi[next_city]
            else:
                self.CurrentBike=0
        # print(self.CurrentBike)

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有站点为止
        while self.move_count < city_num:
            # 移动到下一个站点
            self.__calculate_open_table_city()
            next_city = self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()


# ----------- TSP问题 -----------

class TSP(object):

    def __init__(self, root, width=800, height=800, n=city_num):

        # 创建画布
        self.root = root
        self.width = width
        self.height = height
        # 站点数目初始化为city_num
        self.n = n
        # tkinter.Canvas
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",  # 背景白色
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("SCUT-共享单车调度--(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLock()  # 线程锁

        self.__bindEvents()
        self.new()

        data = pd.read_excel("data_dist.xlsx", header=None)
        # 计算站点之间的距离
        for i in range(city_num):
            for j in range(city_num):
                distance_graph[i][j] = data[i][j]

    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quite)  # 退出程序
        self.root.bind("n", self.new)  # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索

    # 更改标题
    def title(self, s):

        self.root.title(s)

    # 初始化
    def new(self, evt=None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化站点节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            if i!=0:
                node = self.canvas.create_oval(x - self.__r,
                                           y - self.__r, x + self.__r, y + self.__r,
                                           fill="green",  # 填充绿色
                                           outline="#000000",  # 轮廓白色
                                           tags="node",
                                           )
            else: # 对出发点，标为红色
                node = self.canvas.create_oval(x - self.__r,
                                               y - self.__r, x + self.__r, y + self.__r,
                                               fill="#ff0000",  # 填充红色
                                               outline="#000000",  # 轮廓白色
                                               tags="node",
                                               )
            self.nodes2.append(node)
            # 显示坐标
            # self.canvas.create_text(x, y - 10,  # 使用create_text方法在坐标（302，77）处绘制文字
            #                         text='(' + str(x) + ',' + str(y) + ')',  # 所绘制文字的内容
            #                         fill='black'  # 所绘制文字的颜色为灰色
            #                         )
            self.canvas.create_text(x, y - 10,  # 使用create_text方法在坐标（302，77）处绘制文字
                                    text=str(i)+'('+str(bi[i])+')',  # 所绘制文字的内容
                                    fill='black'  # 所绘制文字的颜色为灰色
                                    )

        # 顺序连接站点
        # self.line(range(city_num))

        # 初始站点之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print(u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 开始搜索
    def search_path(self, evt=None):

        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            # 此处的运输成本参数，0.7为每公里运输成本，可修改
            print(u"迭代次数：", self.iter, u"最佳路径总距离：", float(self.best_ant.total_distance),U"运输成本：",0.7*float(self.best_ant.total_distance))
            print(u"最佳路径：")
            for i in range(len(self.best_ant.path)):
                print(self.best_ant.path[i],end=" ")
            print('\n')
            # 连线
            self.line(self.best_ant.path)
            # 设置标题
            self.title("SCUT-共享单车调度--(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1

    # 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        # 用户满意度service
        service=0
        for ant in self.ants:
            total_service=0
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻站点间留下信息素，与路径总距离反比
                # 计划在信息素浓度中加入用户满意度因素，分别赋予权重
                if bi[i]>=0:
                    if ant.CurrentBike+bi[i]<=MaxBike:
                        service=1
                    else:
                        service=(MaxBike-ant.CurrentBike)/bi[i]
                else:
                    if ant.CurrentBike+bi[i]>=0:
                        service=1
                    else:
                        service= ant.CurrentBike/(-bi[i])# 存在问题：此处的currentbike没有运行起来··
                        # 此处Q / ant.total_distance大约为10，给定(service_bbb[i]*service)*300大约为10
                temp_pheromone[start][end] += Q / ant.total_distance + (service_bbb[i]*service)*300 #需要调整合适的权重
                temp_pheromone[end][start] = temp_pheromone[start][end]
                total_service+=service_bbb[i]*service
                # total_service+=service
                # print(service)
                # break

            print(total_service)
        # 更新所有站点之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

    # 主循环
    def mainloop(self):
        self.root.mainloop()


# ----------- 程序的入口处 -----------

if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()
