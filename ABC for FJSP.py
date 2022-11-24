import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import datetime
import time
import sys
import json

class ABC():
    def __init__(self,popsize,MCN,limit):
        self.popsize = popsize                                          # 种群规模
        self.MCN = MCN                                                  # 最大迭代次数
        self.Limits = limit
        self.starttime = time.perf_counter()                            # 主程序开始时间
        self.Best_solution,self.Best_fit,self.Best_jst,\
        self.Best_jet,self.Best_jmu,self.begin_day,self.fixday_time,self.machine_fix = self.main()        # ABC主程序，返回种群最优值和最优解

# 处理OS段
    def handle(self,x):
        # 输入：可行解X
        # 输出：对工序部分处理后的列表，total_process行，每行存放[加工零件号，工序]
        piece_mark = np.zeros(data.jobs)     #统计工序的标志
        array = []                      #经过处理后的工序列表，total_process行，每行存放[加工零件号，工序]
        for i in range(data.total_process):
            piece_mark[int(x[i]-1)] += 1
            array.append((int(x[i]), int(piece_mark[int(x[i]-1)])))
        return array

# 按照产品类别序列生成可行解
    def initialize_pop(self,order_list):
        # 输入：产品类别序列
        # 输出：最早可用策略生成的排程方案、对应方案的完工时间
        pop = np.zeros(data.total_process*2)
        T_list111 = order_list
        OS_ = []
        for t in T_list111:
            OS_ = OS_ + data.product_sumlist[int(t)]
    # ES初始化
        # 生成OS段
        for j in range(len(OS_)):
            for o in range(data.operations):
                pop[j*data.operations+o] = OS_[j]
        # 测试用，将OS段随机打乱
        # np.random.shuffle(pop[:data.total_process])
        # 生成MS段
        self.early_init(pop)
        # self.rand_init(pop)
        # self.local_init(pop)
        # self.global_init(pop)
        fitness = self.calculate(pop)
        return pop,fitness

# 初始化随机生成产品类别序列
    def initialize(self):
        # 输出：产品类别序列、蜜源上限、对应产品类别生成可行解对应的完工时间
        order_pop = np.zeros((self.popsize,data.kinds))
        order_fitness = np.zeros(self.popsize)
        L = np.zeros(self.popsize)
        for i in range(self.popsize):
            order_pop[i] = [i for i in range(data.kinds)]
            np.random.shuffle(order_pop[i])
            ___,order_fitness[i] = self.initialize_pop(order_pop[i])
        return order_pop,L,order_fitness

# 最早可用策略生成MS段
    def early_init(self,x):
        # 输入：生成的OS
        # 处理：按照生成的OS段，按照最早可用策略生成MS段
        machine_starttime = np.zeros(data.machines+1)        # 每个机器上的开始时间
        machine_endtime = np.zeros(data.machines+1)          # 每个机器上的完成时间
        job_starttime = np.zeros((data.jobs,data.operations))
        job_endtime = np.zeros((data.jobs,data.operations))
        machine_state = np.zeros(data.machines+1)            # 每个机器的状态
        machine_processid = np.ones(data.machines+1)*data.kinds  # 记录机器上一次加工的产品品类
        m_all_list = [i+1 for i in range(data.machines)]
        array = self.handle(x)
        for p in range(data.total_process):
            j_index = array[p][0]-1
            o_index = array[p][1]-1
            m_list = data.machine_sel_list[j_index][o_index]    # 取出当前工件的的可用设备
            if m_list[0]!=111:# 需要修改
                m_starttime = [machine_endtime[i] for i in range(len(m_all_list)) if m_all_list[i] in m_list]       # 根据当前工件的可用设备取出其开始时间
                m_choose = [i for i in range(len(m_starttime)) if m_starttime[i] == min(m_starttime)]               # 从开始时间中取出最早可用设备集合
                m_use = np.random.choice(m_choose)                                                                  # 从最早可用设备集合中随机选择一台
                x[data.total_process + j_index*data.operations + o_index] = m_list[m_use]                                                                    # 该随机选择设备作为当前工件的加工设备
            else:
                m_use = 0
                x[data.total_process + j_index*data.operations + o_index] = m_list[0]
            MS_index = int(x[data.total_process + j_index*data.operations + o_index])-1      # 选中的设备号
            temp_leibie = [data.product_sumlist.index(j) for j in data.product_sumlist if x[p] in j]  # 计算当前加工工件属于哪一品类
            change_moduletime=0
            if MS_index != data.machines:
                if temp_leibie[0] == machine_processid[MS_index]:                               # 判断当前工件与上一工件是否同类   (若设备从静到动无需考虑换模时间)
                    change_moduletime = 0
                else:
                    if machine_processid[MS_index] == data.kinds:
                        change_moduletime = 0
                    else:
                        change_moduletime = data.data_changeoperation[o_index]
            machine_processid[MS_index] = temp_leibie[0]
            if o_index==0:  # 当前工序为该工件第一次加工
                if machine_state[MS_index] ==0:   # 当前机器未被使用过
                    job_starttime[j_index][o_index] = 0                                               # 则该工序开始时间=0
                    job_endtime[j_index][o_index] = data.machine_processtime[j_index][o_index][m_use]    # 该工序完成时间=开始时间+工序时长
                else:                                       # 当前机器被使用过
                    job_starttime[j_index][o_index] = machine_endtime[MS_index] + change_moduletime            # 则该工序开始时间=当前机器完工时间+换模时间
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
            else:               # 当前工序不是该工件第一次加工
                if machine_state[MS_index] ==0:   # 当前机器未被使用过
                    job_starttime[j_index][o_index] = job_endtime[j_index][o_index-1]                 # 则该工序开始时间=上一道工序完成时间
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
                    # 此处有个BUG，当虚拟工序加工时，会等待机器，因此考虑设置虚拟机器，用于加工所有的虚拟工序
                else:                                       # 当前机器被使用过                           # 则该工序开始时间=max(上一道工序完成时间，当前机器完工时间)
                    job_starttime[j_index][o_index] = max(job_endtime[j_index][o_index-1],machine_endtime[MS_index]+change_moduletime)
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
            machine_starttime[MS_index] = job_starttime[j_index][o_index]
            machine_endtime[MS_index] = job_endtime[j_index][o_index]
            machine_starttime[data.machines] = 0
            machine_endtime[data.machines] = 0
            machine_state[MS_index] = 1

# 对得到的最优产品类别序列进行贴近交期排程
    def generate_paicheng(self,order_list,begin_day):
        # 输入：产品类别序列、当月开始生产日期
        # 输出：排程方案、对应方案的完工时间、各工件开始和结束时间、各工件使用的设备
        pop = np.zeros(data.total_process*2)
        T_list111 = order_list
        OS_ = []
        for t in T_list111:
            OS_ = OS_ + data.product_sumlist[int(t)]
        # ES初始化
        # 生成OS段
        for j in range(len(OS_)):
            for o in range(data.operations):
                pop[j*data.operations+o] = OS_[j]
        # 生成MS段
        machine_starttime = np.zeros(data.machines+1)        # 每个机器上的开始时间
        machine_endtime = np.zeros(data.machines+1)          # 每个机器上的完成时间
        job_starttime = np.zeros((data.jobs,data.operations))
        job_endtime = np.zeros((data.jobs,data.operations))
        machine_used = np.zeros((data.jobs, data.operations))       # 每个工件工序使用的设备号
        machine_state = np.zeros(data.machines+1)            # 每个机器的状态
        machine_processid = np.ones(data.machines+1)*data.kinds  # 记录机器上一次加工的产品品类
        m_all_list = [i+1 for i in range(data.machines)]
        array = self.handle(pop)
        # 此处按照预期排程天数，根据交期提前排程，并且生成排程期间的维保的时间的对应设备
        fixday_time = []
        machine_fix = []
        for i in data.fix_day:
            if i >= begin_day:
                fixday_time.append([(i-begin_day)*1440,(i+1-begin_day)*1440])
                machine_fix.append(data.machinefix_list[data.fix_day.index(i)])
        # print("生产期间设备维护保养时间段为：",fixday_time)
        # print(machine_fix)
        fixdays = len(fixday_time)
        for p in range(data.total_process):
            j_index = array[p][0]-1
            o_index = array[p][1]-1
            m_list = list(data.machine_sel_list[j_index][o_index])    # 取出当前工件的的可用设备
            temp_leibie = [data.product_sumlist.index(j) for j in data.product_sumlist if pop[p] in j]  # 计算当前加工工件属于哪一品类
            while True:
                if not m_list:
                    pop[data.total_process + j_index*data.operations + o_index] = np.random.choice(data.machine_sel_list[j_index][o_index])
                    MS_index = int(pop[data.total_process + j_index*data.operations + o_index])-1
                    choose_index = list(data.machine_sel_list[j_index][o_index]).index(MS_index+1)
                    job_starttime[j_index][o_index] = fixday_time[fixday][1]
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][choose_index]
                    break
                if m_list[0]!=111:
                    m_starttime = [machine_endtime[i] for i in range(len(m_all_list)) if m_all_list[i] in m_list]       # 根据当前工件的可用设备取出其开始时间
                    m_choose = [i for i in range(len(m_starttime)) if m_starttime[i] == min(m_starttime)]               # 从开始时间中取出最早可用设备集合
                    m_use = np.random.choice(m_choose)                                                                  # 从最早可用设备集合中随机选择一台
                    pop[data.total_process + j_index*data.operations + o_index] = m_list[m_use]                         # 该随机选择设备作为当前工件的加工设备
                else:
                    m_use = 0
                    pop[data.total_process + j_index*data.operations + o_index] = m_list[m_use]
                    MS_index = int(pop[data.total_process + j_index*data.operations + o_index])-1
                    if machine_state[MS_index] ==0:   # 当前机器未被使用过
                        job_starttime[j_index][o_index] = job_endtime[j_index][o_index-1]                 # 则该工序开始时间=上一道工序完成时间
                        job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
                    else:                                       # 当前机器被使用过                           # 则该工序开始时间=max(上一道工序完成时间，当前机器完工时间)
                        job_starttime[j_index][o_index] = max(job_endtime[j_index][o_index-1],machine_endtime[MS_index])
                        job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
                    break
                # 选好一个加工设备，根据该工序开始时间判断，如果处于维保时间内，则判断该选择设备是否在维保设备集合中
                # 是，则从m_list中去掉该设备，重新进行一次选择
                MS_index = int(pop[data.total_process + j_index*data.operations + o_index])-1      # 选中的设备号
                change_moduletime=0
                if MS_index != data.machines:
                    if temp_leibie[0] == machine_processid[MS_index]:                               # 判断当前工件与上一工件是否同类   (若设备从静到动无需考虑换模时间)
                        change_moduletime = 0
                    else:
                        if machine_processid[MS_index] == data.kinds:
                            change_moduletime = 0
                        else:
                            change_moduletime = data.data_changeoperation[o_index]
                if o_index==0:  # 当前工序为该工件第一次加工
                    if machine_state[MS_index] ==0:   # 当前机器未被使用过
                        job_starttime[j_index][o_index] = 0                                               # 则该工序开始时间=0
                        job_endtime[j_index][o_index] = data.machine_processtime[j_index][o_index][m_use]    # 该工序完成时间=开始时间+工序时长
                    else:                                       # 当前机器被使用过
                        job_starttime[j_index][o_index] = machine_endtime[MS_index] + change_moduletime            # 则该工序开始时间=当前机器完工时间+换模时间
                        job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
                else:               # 当前工序不是该工件第一次加工
                    if machine_state[MS_index] ==0:   # 当前机器未被使用过
                        job_starttime[j_index][o_index] = job_endtime[j_index][o_index-1]                 # 则该工序开始时间=上一道工序完成时间
                        job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
                    else:                                       # 当前机器被使用过                           # 则该工序开始时间=max(上一道工序完成时间，当前机器完工时间)
                        job_starttime[j_index][o_index] = max(job_endtime[j_index][o_index-1],machine_endtime[MS_index]+change_moduletime)
                        job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_use]
                flag = 0
                for day in range(len(fixday_time)):
                    if job_starttime[j_index][o_index] in range(fixday_time[day][0],fixday_time[day][1]):
                        fixday = day
                        if MS_index+1 in machine_fix[day]:
                            m_list.remove(MS_index+1)
                            machine_endtime[MS_index] = fixday_time[day][1]
                        else:
                            flag += 1
                    else:
                        flag += 1
                if flag ==fixdays:
                    break
            # 每次选择这个不可用设备之后，由于时间一直没有更新，每次都会又选到该设备，但不可用且该工序时间开始时间仍为在设备维保区间内
            machine_used[j_index][o_index] = MS_index + 1
            machine_processid[MS_index] = temp_leibie[0]
            machine_starttime[MS_index] = job_starttime[j_index][o_index]
            machine_endtime[MS_index] = job_endtime[j_index][o_index]
            machine_starttime[data.machines] = 0
            machine_endtime[data.machines] = 0
            machine_state[MS_index] = 1
        max_processtime = 0
        for job in range(data.jobs):
            if max_processtime <= job_endtime[job][-1]:
                max_processtime = job_endtime[job][-1]
        print(f"单次排程总时长为：{max_processtime}分钟,约为{max_processtime/1440}天")
        return pop,max_processtime,job_starttime,job_endtime,machine_used,fixday_time,machine_fix

# 计算完工时间（不考虑设备维保和贴近交期）
    def calculate(self,x):
        # 输入:可行解X
        # 输出:目标函数值fitness（最大完工时间）
        machine_starttime = np.zeros(data.machines+1)        # 每个机器上的开始时间
        machine_endtime = np.zeros(data.machines+1)          # 每个机器上的完成时间
        job_starttime = np.zeros((data.jobs, data.operations))    # 每个工序的开始时间
        job_endtime = np.zeros((data.jobs, data.operations))      # 每个工序的完成时间
        machine_processid = np.ones(data.machines+1)*data.kinds  # 记录机器上一次加工的产品品类
        machine_state = np.zeros(data.machines+1)            # 每个机器的状态
        # 解码过程先判断该工序是否为第一道工序，再判断当前机器状态是否为第一次使用
        total_changetime =0
        array = self.handle(x)
        for i in range(data.total_process):                  # 循环OS段
            j_index = array[i][0]-1
            o_index = array[i][1]-1
            MS_index = int(x[data.total_process + j_index*data.operations + o_index])-1        # 当前OS上编码对应到MS段的机器号-1
            m_index = data.machine_sel_list[j_index][o_index].index(MS_index+1)                  # m_index对应machine_sel_list和machine_processtime中的存放位置
            temp_leibie = [data.product_sumlist.index(j) for j in data.product_sumlist if x[i] in j]  # 计算当前加工工件属于哪一品类
            change_moduletime=0
            if MS_index != data.machines:
                if temp_leibie[0] == machine_processid[MS_index]:                               # 判断当前工件与上一工件是否同类   (若设备从静到动无需考虑换模时间)
                    change_moduletime = 0
                else:
                    if machine_processid[MS_index] == data.kinds:
                        change_moduletime = 0
                    else:
                        change_moduletime = data.data_changeoperation[o_index]
            machine_processid[MS_index] = temp_leibie[0]
            total_changetime += change_moduletime
            if o_index==0:  # 当前工序为该工件第一次加工
                if machine_state[MS_index] ==0:   # 当前机器未被使用过
                    job_starttime[j_index][o_index] = 0                                               # 则该工序开始时间=0
                    job_endtime[j_index][o_index] = data.machine_processtime[j_index][o_index][m_index]    # 该工序完成时间=开始时间+工序时长
                else:                                       # 当前机器被使用过
                    job_starttime[j_index][o_index] = machine_endtime[MS_index] + change_moduletime            # 则该工序开始时间=当前机器完工时间+换模时间
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_index]
            else:               # 当前工序不是该工件第一次加工
                if machine_state[MS_index] ==0:   # 当前机器未被使用过
                    job_starttime[j_index][o_index] = job_endtime[j_index][o_index-1]                 # 则该工序开始时间=上一道工序完成时间
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_index]
                    # 此处有个BUG，当虚拟工序加工时，会等待机器，因此考虑设置虚拟机器，用于加工所有的虚拟工序
                else:                                       # 当前机器被使用过                           # 则该工序开始时间=max(上一道工序完成时间，当前机器完工时间)
                    job_starttime[j_index][o_index] = max(job_endtime[j_index][o_index-1],machine_endtime[MS_index]+change_moduletime)
                    job_endtime[j_index][o_index] = job_starttime[j_index][o_index] + data.machine_processtime[j_index][o_index][m_index]
            machine_starttime[MS_index] = job_starttime[j_index][o_index]
            machine_endtime[MS_index] = job_endtime[j_index][o_index]
            machine_starttime[data.machines] = 0
            machine_endtime[data.machines] = 0
            machine_state[MS_index] = 1
        return max(machine_endtime)
        # return total_changetime

# 新解的邻域搜索
    def cross_func(self,pop1,pop2):
        # 输入：两个可行解
        # 输出：按照POX交叉或者JBX交叉方式生成的新的可行解和对应完工时间
        x1=np.copy(pop1)
        x2=np.copy(pop2)
        jobset = [i+1 for i in range(data.kinds)]        # 生成所有工件集合
        np.random.shuffle(jobset)                       # 将工件顺序打乱
        rand_num = np.random.randint(1,data.kinds)       # 生成一个随机数用于划分工件集合
        jobset1 =jobset[:rand_num]                      # 将工件集切为两个没有交集的集合jobset1，jobset2
        # POX交叉
        index1,index2 = [0,0]                                                        # 用于索引remain中的保留位置
        change1 = [i for i in range(data.kinds) if pop1[i] in jobset1]       # x1个体要改变的位置，对于jobset2中的位置直接保留                                               # POX交叉
        change2 = [i for i in range(data.kinds) if pop2[i] in jobset1]   # x2个体要改变的位置，对于jobset2中的位置直接保留
        for i in range(data.kinds):                                      # 对OS编码进行循环
            if pop2[i] in jobset1:                                               # 如果当前OS段中
                x1[change1[index1]] = pop2[i]
                index1 += 1
            if pop1[i] in jobset1:
                x2[change2[index2]] = pop1[i]
                index2 += 1
        ___,fitnessx1 = self.initialize_pop(x1)
        ___,fitnessx2 = self.initialize_pop(x2)
        if fitnessx1<fitnessx2:
            return x1,fitnessx1
        else:
            return x2,fitnessx2

# 到达蜜源上限后的生成新的解
    def new_search(self,x):
        # 输入：当前到达蜜源上限的可行解
        # 输出：新的可行解和完工时间
        if np.random.rand() <0.5:
            swap_points = np.random.choice(np.arange(data.kinds), size=2, replace=False)
            x[swap_points[0]],x[swap_points[1]]=x[swap_points[1]],x[swap_points[0]]
        else:
            neibors_choose = np.random.choice(np.arange(data.kinds), size=3, replace=False)
            neibors_list = [x[neibors_choose[0]],x[neibors_choose[1]],x[neibors_choose[2]]]
            np.random.shuffle(neibors_list)
            x[neibors_choose[0]],x[neibors_choose[1]],x[neibors_choose[2]] = tuple(neibors_list)
        __,fitness = self.initialize_pop(x)
        return x,fitness

# 主函数
    def main(self):
    # 初始化种群
        pop,L,fitness = self.initialize()
        Best_chart = np.zeros(self.MCN)
        Xbest_chart = np.zeros((self.MCN,data.kinds))
        print(fitness)
    # 记录初代种群最优解和最优值
        Best_chart[0] = min(fitness)
        Xbest = pop[np.argmin(fitness)]
        Xbest_chart[0] = Xbest
    # 开始循环
        for it in range(1,self.MCN):
        # 雇佣蜂阶段
            for j in range(self.popsize):                           # 随机选取另一个个体与之进行交叉
                pop_list = [i for i in range(self.popsize)]
                pop_list.remove(j)                                  # 取一个不同于j的数k
                k = np.random.choice(pop_list)
                newbee,newbee_fitness = self.cross_func(pop[j],pop[k])
                if newbee_fitness < fitness[j]:          # 新个体
                    pop[j] = newbee
                    fitness[j] = newbee_fitness
                    L[j] = 0
                else:
                    L[j] +=1
        # 跟随蜂阶段
            F_val = 1/np.exp(fitness/1000)
            # F_val = 1/fitness
            idx = np.random.choice(np.arange(len(fitness)), size=len(fitness), replace=True, p=(F_val) / (F_val.sum()))
            # 轮盘赌选择后进一步搜索
            for j in idx:
                pop_list = [i for i in range(self.popsize)]
                pop_list.remove(j)                                  # 取一个不同于j的数k
                k = np.random.choice(pop_list)
                newbee,newbee_fitness = self.cross_func(pop[j],pop[k])
                if newbee_fitness < fitness[j]:          # 新个体
                    pop[j] = newbee
                    fitness[j] = newbee_fitness
                    L[j] = 0
                else:
                    L[j] +=1
        # 侦查蜂阶段
            for j in range(self.popsize):
                if L[j] >= self.Limits:
                    pop[j],fitness[j] = self.new_search(pop[j])
                    L[j] = 0
            print(pop)
            print(fitness)
        # 保存本次迭代的最优解和最优值
            if min(fitness) < Best_chart[it-1]:
                Best_chart[it] = min(fitness)
                Xbest = pop[np.argmin(fitness)]
                Xbest_chart[it] = Xbest
            else:
                Best_chart[it] = Best_chart[it-1]
                Xbest_chart[it] = Xbest_chart[it-1]
        # 显示算法进度
            finsh = "▓" * round(100*it/self.MCN)
            need_do = "-" * round(100*(self.MCN-it)/self.MCN)
            progress = ((it+1)/self.MCN) * 100
            dur = time.perf_counter() - self.starttime
            print("\r{:^3.1f}%[{}->{}]{:.2f}s".format(progress, finsh, need_do, dur), end="")
            print('\n')
            time.sleep(0.05)

    # 绘制收敛曲线
    #     plt.plot(np.arange(1,self.MCN+1,1),Best_chart,'b-')
    # 按照最优序列生成排程方案
        print("最优的产品类别序列为：",Xbest_chart[-1])
        print("对应排程时长为：",Best_chart[-1])
        expect_days = round(Best_chart[-1]/1440) + data.offset_begin
    #     expect_days = 51
        begin_day = calendar.monthrange(data.year,int(data.month_list[data.month_choose]))[1] - expect_days
        print("预期开始生产时间为：",begin_day)
        solution = np.zeros((self.popsize,data.total_process*2))                # 排程方案（OS+MS序列）
        solution_fit = np.zeros(self.popsize)                                   # 对应方案的完工时间
        job_starttime = np.zeros((self.popsize,data.jobs,data.operations))      # 各工件工序开始时间
        job_endtime = np.zeros((self.popsize,data.jobs,data.operations))        # 各工件工序完工时间
        machine_used = np.zeros((self.popsize,data.jobs,data.operations))       # 各工件工序使用的设备
    # 按照算法得到的最优产品类别序列，生成满足设备维保和贴近交期排程的方案（此处也可以使用局部搜索，获得更好的解）
        for i in range(self.popsize):
            solution[i],solution_fit[i],job_starttime[i],job_endtime[i],machine_used[i],fixday_time,machine_fix = self.generate_paicheng(Xbest_chart[-1],begin_day)
            # solution[i],solution_fit[i],job_starttime[i],job_endtime[i],machine_used[i],fixday_time,machine_fix = self.generate_paicheng([7,2,5,1,6,3,0,4],begin_day)
    # 保存最优排程方案
        Best_fit = min(solution_fit)
        Best_solution = solution[np.argmin(solution_fit)]
        Best_jst = job_starttime[np.argmin(solution_fit)]
        Best_jet = job_endtime[np.argmin(solution_fit)]
        Best_jmu = machine_used[np.argmin(solution_fit)]
        return Best_solution,Best_fit,Best_jst,Best_jet,Best_jmu,begin_day,fixday_time,machine_fix

class Data():
    def __init__(self,path,order_choose,month_choose,offset_begin):
        self.path = path
        self.month_choose = month_choose
        self.order_choose = order_choose
        self.offset_begin = offset_begin
        self.machine_sel_list,self.machine_processtime,self.operations_all_list,self.machines,self.product_sumlist,\
        self.data_changeoperation,self.machine_list,self.machinefix_list,self.fix_day,self.data_machinestoptime,\
        self.jobnums,self.year,self.month_list = self.read_data()
        self.operations = len(self.machine_sel_list[0])
        self.jobs = len(self.machine_sel_list)
        self.kinds = len(self.product_sumlist)                           # 工件总类别
        self.total_process = self.jobs * self.operations

# 读取dataset中数据
    def read_data(self):
        # 输入：文件路径
        # 输出：需要的数据
    # 读取产品信息
        data_products = pd.read_excel(self.path,sheet_name='产品信息')
        T_list =  np.unique(data_products['产品类别'])
        t_temp = data_products['产品类别'].tolist()
        p_temp = data_products['产品型号'].tolist()
        products_list=[]                    ### 产品类别信息
        for i in T_list:
            products_list.append([p_temp[it] for it in range(len(t_temp)) if t_temp[it] == i])
    # 读取订单信息
        data_orderinf1 = pd.read_excel(self.path,sheet_name=self.order_choose)
        order_dates = np.unique(data_orderinf1['订单日期'])
        day_begin=[]
        for temp_day in order_dates:
            day_begin.append(str(temp_day).split('T')[0])
        order_dates1 = np.unique(data_orderinf1['期望交期'])
        day_end = []
        month_list = []
        for temp_day1 in order_dates1:
            day_end.append(str(temp_day1).split('T')[0])
            month_list.append(str(temp_day1).split('T')[0].split('-')[1])
        # print(month_list)
        year = int(day_end[self.month_choose].split('-')[0])
        months = len(order_dates)                       # 共有几个月的订单
        readlist = ['订单编号','产品型号','数量','订单日期','期望交期']
        monthly_list = []
        for i in order_dates:               # 得到按月存放的订单信息
            monthly_list.append(data_orderinf1.loc[data_orderinf1['订单日期']== i,readlist])
        jobnum = monthly_list[self.month_choose]['数量'].tolist()
        jobnums = [0]
        for i in range(len(jobnum)):
            jobnums = jobnums + [jobnums[i]+jobnum[i]]
        product_num=[]
        product_sumnum = []
        for i in range(months):
            temp00=[]
            temp000=[]
            n_temp = monthly_list[i]['数量'].tolist()
            p_temp = monthly_list[i]['产品型号'].tolist()
            for j in range(len(T_list)):
                temp00.append([n_temp[it] for it in range((len(p_temp))) if p_temp[it] in products_list[j]])
                temp000.append(sum(temp00[j]))
            product_num.append(temp00)      ### 每月各订单中产品数量
            product_sumnum.append(temp000)  ### 每月各订单中各品类产品总数
    # 读取工序信息
        data_operations = pd.read_excel(self.path,sheet_name='产品类别工序')
        data_operations = data_operations.drop(columns=['类别序号'])             ### 各品类产品的加工工序
    # 读取设备信息和加工时间
        data_machine = pd.read_excel(self.path,sheet_name='设备信息')
        O_list = np.unique(data_machine['对应工序'])
        machine_list=[]                                                         ### 原始各工序对应加工机器
        mmm=[] # 按照对应工序划分的机器号
        for o in O_list:
            mmm.append(data_machine.index[data_machine['对应工序']==o].tolist())
            machine_list.append(data_machine.loc[data_machine['对应工序']==o,['设备编号','对应工序']])
        machines = max(max(mmm))+1                                             ### 所有机器数量
        data_machinetime = pd.read_excel(self.path,sheet_name='设备作业时间')     ### 原始各品类工序对应各机器加工时间
    # 读取设备维保信息
        data_machinefixinf = pd.read_excel(self.path,sheet_name='设备维保信息')
        F_list = np.unique(data_machinefixinf['停机维保日期信息'])
        F_list[0],F_list[2] = F_list[2],F_list[0]
        F_list[1],F_list[3] = F_list[3],F_list[1]
        machinefix_list = []# 存放各机器停机维保时间[]
        for f in F_list:
            machine_index = list(data_machinefixinf.loc[data_machinefixinf['停机维保日期信息']==f,['设备编号','停机维保日期信息']].index.values +1)
            machinefix_list.append(machine_index)
        if self.order_choose == '订单信息_2':
            machinefix_list = machinefix_list * 2
    # 读取设备空转停机时间
        data_machinestoptime = pd.read_excel(self.path,usecols=[0,4],sheet_name='设备稼动能耗信息')['间隔时间超过则停机小时数'].tolist()
        # print(data_machinestoptime)
    # 读取工序转换时间
        data_changeoperation = pd.read_excel(self.path,sheet_name='工序准备时间')['换型准备时间'].tolist()
    # 处理数据后得到机器选择列表和加工时长列表,以及当月订单的维保日期
        machine_sel,machine_process,operations_all_list,fix_day=self.process_data(day_begin,day_end,data_operations,data_machinetime,mmm,machines)
        machine_selpool = []
        machine_processtime = []
        for i in range(len(T_list)):
            machine_selpool = machine_selpool + [machine_sel[i]]*product_sumnum[self.month_choose][i]    # sum(product_num[0][i])             # product_num[0][0]表示第一个月订单中第一类产品的数量
            machine_processtime = machine_processtime + [machine_process[i]]*product_sumnum[self.month_choose][i]       # sum(product_num[0][i])
        # 应该按照当月订单各类产品数量往里面添加
        machine_selpool = tuple(machine_selpool)
        machine_processtime = tuple(machine_processtime)
    # 按照订单中产品数量生成分品类的产品编号
        list2 = [0]
        temp = 0
        for x in range(len(product_sumnum[self.month_choose])):
            temp =temp+product_sumnum[self.month_choose][x]
            list2.append(temp)
        target_list=[]
        for i in range(len(list2)-1):
            target_list.append([j+1 for j in range(list2[i],list2[i+1])])
    # 返回值分别为
        # 可用设备集合、可用设备集合对应加工时间、所有品类工件的工序、设备总数量、每类别工件的编号
        # 换模时间信息、按工序分开的设备集合、当月维护设备集合、当月维护日期、设备空转停机时长、当前订单交期年份、所有订单月份列表
        return machine_selpool,machine_processtime,tuple(operations_all_list),machines,target_list,\
               data_changeoperation,machine_list,tuple(machinefix_list),fix_day,\
               data_machinestoptime,tuple(jobnums),year,month_list
# 处理需要的数据
    def process_data(self,day_begin,day_end,operations,machinetimes,mmm,machine_num):
    # 计算开始加工年月日
        temp_syear = int(day_begin[self.month_choose].split('-')[0])
        temp_smonth = int(day_begin[self.month_choose].split('-')[1])
        temp_sday = int(day_begin[self.month_choose].split('-')[2])
        if temp_sday == calendar.monthrange(temp_syear,temp_smonth)[-1]:
            begin_date = int(self.offset_begin)
        else:
            begin_date = int(self.offset_begin + temp_sday)
        temp_eyear = int(day_end[self.month_choose].split('-')[0])
        temp_emonth = int(day_end[self.month_choose].split('-')[1])
        temp_eday = int(day_end[self.month_choose].split('-')[2])
        # print(temp_syear,temp_smonth,temp_sday,begin_date,'***',temp_eyear,temp_emonth,temp_eday)
    # 按照订单2记录当月维保日期，按照负数存放，即月底为0，往前1天则减1
        fix_day1 = []
        if self.order_choose=='订单信息_2':
            month_inf1 = calendar.monthrange(temp_syear,temp_smonth)
            month_week1 = []
            for i in range(month_inf1[1]):
                month_week1.append((month_inf1[0]+i)%7)
            count_sat1 = 0
            count_sun1 = 0
            for i in range(len(month_week1)):
                if month_week1[i]==5:
                    count_sat1 += 1
                    if count_sat1 ==2:
                        fix_day1.append(i+1-month_inf1[1])
                if month_week1[i]==6:
                    count_sun1 += 1
                    if count_sun1 ==2:
                        fix_day1.append(i+1-month_inf1[1])
            for j in range(month_inf1[1]-7,month_inf1[1]):
                if month_week1[j]==5:
                    fix_day1.append(j+1-month_inf1[1])
                if month_week1[j]==6:
                    fix_day1.append(j+1-month_inf1[1])
    # 按照订单1记录当月维保日期
        fix_day2 = []
        month_inf2 = calendar.monthrange(temp_eyear,temp_emonth)
        month_week2 = []
        for i in range(month_inf2[1]):
            month_week2.append((month_inf2[0]+i)%7)
        count_sat2 = 0
        count_sun2 = 0
        for i in range(len(month_week2)):
            if month_week2[i]==5:
                count_sat2 += 1
                if count_sat2 ==2:
                    fix_day2.append(i+1)
            if month_week2[i]==6:
                count_sun2 += 1
                if count_sun2 ==2:
                    fix_day2.append(i+1)
        for j in range(month_inf2[1]-7,month_inf2[1]):
            if month_week2[j]==5:
                fix_day2.append(j+1)
            if month_week2[j]==6:
                fix_day2.append(j+1)
        fix_day = fix_day1 +fix_day2
    # 对各类产品每一道工序支持的加工机器进行处理，取出支持加工机器集合，用于构造MS段done
        # 需要继续进行处理，由于原数据中存在干扰，需要利用'产品工序类别'中的数据进行判断，若该工序不存在，则跳过这一段的写入
        machine_selpool=[]
        machine_processtime=[]
        T_list = machinetimes.columns.values[1:] # 用于读入每一类产品的生产加工时间
        # 生成operations的行列名列表
        row_list = operations.index.values
        col_list = operations.columns.values
        count_row=0
        operations_all_list=[]
        for t in T_list:
            count_col=0
            temp1=[] # 用于存放每一类产品信息
            temp11=[] # 用于存放每一类产品信息
            for o in mmm: # mmm中按工序存放可行机器号[[1-10],[11,14],[...],...]用于循环，且实现按工序存放
                # 先判断该类产品的该类工序是否存在，若不存在则跳过添加
                temp2=[]
                temp22=[]
                if (operations.loc[[row_list[count_row]],[col_list[count_col]]]==1).bool():
                    for m in o: # 对每一道工序的机器进行循环，如果其中的数据不为nan，则添加进去
                        if np.isnan(machinetimes[t][m]) == False:
                            # 把所有机器号都放进来
                            temp2.append(machinetimes[t][m])
                            temp22.append(m+1)
                else:
                    temp2.append(0)
                    temp22.append(machine_num+1)
                temp1.append(temp2)
                temp11.append(temp22)
                count_col +=1
            operations_all_list.append(pd.to_numeric(operations.loc[row_list[count_row]]).tolist())
            machine_selpool.append(temp11) # machine_selpool中存放各类产品的每一道工序允许的加工机器号和对应的加工时间，生成MS段的时候可以用来选择对应工序的加工机器
            machine_processtime.append(temp1)
            count_row += 1
        return machine_selpool,machine_processtime,operations_all_list,fix_day
# 绘制甘特图
    def plot_gantt(self,job_starttime,job_endtime,machine_used,fitness11):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
        plt.figure(figsize=(16,9),dpi=100)
        colors = ['r', 'pink', 'orange', 'y', 'g', 'b', 'deeppink', 'purple']
        for i in range(self.jobs):
            t = [data.product_sumlist.index(j) for j in data.product_sumlist if i+1 in j]  # 计算当前加工工件属于哪一品类
            for j in range(self.operations):
                plt.barh(machine_used[i][j],job_endtime[i][j]-job_starttime[i][j],left=job_starttime[i][j],color=colors[t[0]])
                # plt.text(job_starttime[i][j],machine_used[i][j],f'{i+1}-{j+1}' ,fontsize=8, verticalalignment="center", horizontalalignment="left")
                # plt.text(job_starttime[i][j],machine_used[i][j],f'{i+1}-{j+1}__{int(job_protime[i][j]-job_starttime[i][j])}' ,fontsize=8, verticalalignment="center", horizontalalignment="left")
        days = int(fitness11//1440)
        for day in range(days):
            plt.axvline(1440*(day+1))
        plt.text(fitness11,1,f'Cmax={fitness11}',fontsize=8, verticalalignment="center", horizontalalignment="right")
        plt.grid(linestyle="--", alpha=0.5)
        plt.title("甘特图")
        plt.xlim(0,fitness11+10)
        plt.yticks(range(self.machines+1))
        plt.xlabel("加工时间 /min")
        plt.ylabel("机器")
        plt.show()
# 输出json文档
    def output_json(self,job_starttime,job_endtime,machine_used,fitness11,begin_day):
        order_id =1
        if self.order_choose == '订单信息_2':
            order_id = 2
        outputtime = time.strftime("%Y%m%d%H%M%S",time.localtime())
        shops_out =[]
        shop_inf =[]
        for m in range(self.machines):
            m_inf =[]
            m_position = np.argwhere(machine_used == m+1)
            for j_idx,o_idx in m_position:
                m_inf.append([job_starttime[j_idx][o_idx],job_endtime[j_idx][o_idx],j_idx])
                m_inf.sort()
            shop_inf.append(m_inf)
        # print(shop_inf)
        # 还需要解决的功能
        # 1、将工件号转换为对应的订单号和产品类型
        # 2、将维修保养时间段标记后放入,如果下一道工序的开始时间大于等于维保尾期
        for m in range(self.machines):
            datalist = []
            for i in range(len(shop_inf[m])):
                id_inf = self.get_orderid_productid(int(shop_inf[m][i][2])+1)
                # 计算
                data={
                    "start":self.calculate_time(begin_day,shop_inf[m][i][0]),
                    "end": self.calculate_time(begin_day,shop_inf[m][i][1]),
                    "val":[20,id_inf[0],id_inf[1],id_inf[2]]
                }
                datalist.append(data)
                if i <len(shop_inf[m])-1:
                    if shop_inf[m][i][1] != shop_inf[m][i+1][0]:
                        time_between = shop_inf[m][i+1][0]-shop_inf[m][i][1]
                        if time_between >= self.data_machinestoptime[m]*60:
                            data={
                                "start":self.calculate_time(begin_day,shop_inf[m][i][1]),
                                "end": self.calculate_time(begin_day,shop_inf[m][i+1][0]),
                                "val":[0]
                            }
                            datalist.append(data)
                            for day in range(len(Run_abc.fixday_time)):
                                if time_between >1340 and shop_inf[m][i+1][0] >= Run_abc.fixday_time[day][1]:
                                    if m+1 in Run_abc.machine_fix[day]:
                                        data={
                                            "start":self.calculate_time(begin_day,Run_abc.fixday_time[day][0]),
                                            "end": self.calculate_time(begin_day,Run_abc.fixday_time[day][1]),
                                            "val":[1]
                                        }
                                        datalist.append(data)
                        else:
                            data={
                                "start":self.calculate_time(begin_day,shop_inf[m][i][1]),
                                "end": self.calculate_time(begin_day,shop_inf[m][i+1][0]),
                                "val":[10]
                            }
                            datalist.append(data)
            shop = {
                'name':f"shop{m+1}",'data':datalist
            }
            shops_out.append(shop)
        path ="D:\董家伟7.25\Python\mypro\RESULTS\json\\" + f"{order_id}.1.ABC.{fitness11}.{outputtime}.json"
        with open(path,"w",encoding="utf-8") as file_in:
            json.dump(shops_out,file_in,ensure_ascii=False,indent=4)
# 换算时间
    def calculate_time(self,begin_day,time_minutes):
        # 输入：开始排程日期，对应工序（分钟）
        # 输出：json文档中要求的时间格式
        if int(self.month_list[self.month_choose]) == 1:
            if (begin_day-1)*1440+time_minutes <0:
                year = self.year-1
                month = 12
            else:
                year = self.year
                month = int(self.month_list[self.month_choose])
        else:
            year = self.year
            if (begin_day-1)*1440+time_minutes <0:
                month = int(self.month_list[self.month_choose])-1
            else:
                month = int(self.month_list[self.month_choose])
        # year = datetime.datetime.today().year
        days = (time_minutes //60) // 24
        hours = time_minutes // 60 -days*24
        minutes = time_minutes -days*24*60 -hours*60
        if begin_day<0:
            day = days + calendar.monthrange(year,month)[1]+begin_day
        else:
            day = begin_day + days
        # begin_time = f'%d%02d%02d000000' % (year,self.month_choose+1, begin_day)
        start = f'%d%02d%02d%02d%02d00' % (year,month, day, hours, minutes)
        return start
# 获取工件对应的订单编号的产品编号
    def get_orderid_productid(self,job):
        for i in range(len(self.jobnums)-1):
            if job in range(self.jobnums[i]+1,self.jobnums[i+1]+1):
                orderid = f'0455-00%03d' % (i+1+self.month_choose*28)
                productid = f'P00%02d' % (i+1)
                num = job - self.jobnums[i]
        return orderid,productid,num


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
# FJSP数据
    # month_choose为第几个月订单，dataset中[0,5]表示1-6月，offset_begin表示根据预期排程时长往前推的额外天数
    data = Data(path = 'D:\董家伟7.25\Python\mypro\dataset-v2.xlsx',order_choose='订单信息_1',month_choose=0,offset_begin=0)
# ABC调用
    Run_abc = ABC(10,20,5)
# 输出txt
    with open(rf'D:\董家伟7.25\Python\mypro\RESULTS\ABC-new0906-{Run_abc.starttime}.txt',mode='w',encoding='UTF-8') as filein:
        filein.write(f'Order_choose:{data.order_choose}\tMonth_choose:{data.month_choose+1}\t')
        filein.write(f'Start time:{Run_abc.starttime}\tEnd time:{time.perf_counter()}\n')
        filein.write(f'Parameter:\tMCN={Run_abc.MCN},\tpopsize={Run_abc.popsize},\tLimits={Run_abc.Limits}\n')
        filein.write(f'Best value={Run_abc.Best_fit}\n')
        filein.write(f'Start_time:\n{Run_abc.Best_jst}\n\n')
        filein.write(f'End_time:\n{Run_abc.Best_jet}\n\n')
        filein.write(f'Machine_used:\n{Run_abc.Best_jmu}')
# 输出json
    data.output_json(Run_abc.Best_jst,Run_abc.Best_jet,Run_abc.Best_jmu,Run_abc.Best_fit,Run_abc.begin_day)
# 绘制甘特图
    data.plot_gantt(Run_abc.Best_jst,Run_abc.Best_jet,Run_abc.Best_jmu,Run_abc.Best_fit)