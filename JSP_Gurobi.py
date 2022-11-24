from gurobipy import *
import matplotlib.pyplot as plt
# 读入data.txt，输出tij工序持续时间、rij各零件加工机器需求
def read_data(Dat):
    rij=[]
    tij=[]
    with open(Dat,'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        data2=[]
        for line in lines:
            data_line = line.strip("\n").split()
            # print(data_line)
            data2.append([int(i) for i in data_line])
    # print(data2)
    # 将data2中的数据提取出来到tij、rij中
    for data in data2:
        if len(data)>3:
            rrij=[]
            ttij=[]
            for k in range(len(data2[1])):
                if (k+1)%3==0:
                    rrij.append(data[k])
            for k in range(len(data2[1])):
                if k%3==0 and k!=0:
                    ttij.append(data[k])
            rij.append(rrij)
            tij.append(ttij)
    # print(rij)
    # print(tij)
    return tij,rij

def run_opt(tij,rij):
    try:
        M=1000
        m=Model("JSP-model")
        # 定义变量xij表示第i个零件的第j道工序开始时间
        xij=m.addMVar((6, 6), vtype=GRB.INTEGER,name="xij")
        # 定义0-1变量yijbd,等于1表示第i个零件的第j道工序在第b个零件的第d道工序之前
        yijbd=m.addMVar((6, 15), vtype=GRB.BINARY, name="yijbd")
        # 定义总时长
        obj_max = m.addVar(vtype=GRB.INTEGER, name="obj_max")
        # 定义约束条件
        # 零件工序顺序约束
        for i in range(6):
            for j in range(5):
                m.addConstr(xij[i][j]+tij[i][j]<=xij[i][j+1])

        # 加工机器约束
        for machine_num in range(1, 7):
            add_machineConstr(m, xij, yijbd, rij, tij, M, machine_num)

        # 目标函数转换为约束处理
        for i in range(6):
            m.addConstr(obj_max >= xij[i][5] + tij[i][5] )
        # 定义目标函数
        # 最小化完成时间
        m.setObjective(obj_max, sense=GRB.MINIMIZE)


        # 模型求解
        m.setParam(GRB.param.TimeLimit,60)
        m.setParam(GRB.param.TuneCriterion,2)
        m.optimize()

        m.write("JSP.lp")
        m.getObjective()
        stij=[]
        stij_temp = []
        for v in m.getVars()[:36]:
            stij_temp.append(v.x)
        if (len(stij_temp) % 6 == 0):
            for idx in range(0, len(stij_temp), 6):
                stij.append([stij_temp[idx], stij_temp[idx + 1],stij_temp[idx + 2],stij_temp[idx + 3],stij_temp[idx + 4],stij_temp[idx + 5]])
        else:
            print("IndexError: list index out of range.")

        # print(stij)
        # print(len(stij))
        print('Obj: %g' % m.objVal)

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')
    return stij

def add_machineConstr(m,xij,yijbd,rij,tij,M,machine_num):
    index = 0
    for ii in range(5):  # 控制外层零件循环
        for jj in range(6):  # 控制零件工序循环
            if rij[ii][jj] == machine_num:  # 选定加工机器号
                i = ii
                j = jj
                for bb in range(i + 1, 6):  # 控制向下层继续寻找零件
                    for dd in range(6):  # 控制零件工序循环
                        if rij[bb][dd] == machine_num:  # 找到相同加工机器号
                            b = bb
                            d = dd
                            m.addConstr(yijbd[machine_num-1][index] * M + (xij[i][j] - xij[b][d]) >= tij[b][d])
                            m.addConstr((1 - yijbd[machine_num-1][index]) * M + (xij[b][d] - xij[i][j]) >= tij[i][j])
                            index += 1

def plot_gantt(stij,tij,rij):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

    plt.figure(figsize=(20,8),dpi=200)
    colors = ['r', 'pink', 'orange', 'y', 'g', 'b', 'deeppink', 'purple', 'brown', 'black']
    for i in range(6):
        for j in range(6):
            plt.barh(rij[i][j],tij[i][j],left=stij[i][j],color=colors[i])
            plt.text(stij[i][j],rij[i][j],f'{i+1}-{j+1}\n{tij[i][j]}' ,fontsize=10, verticalalignment="center", horizontalalignment="left")
    plt.grid(linestyle="--", alpha=0.5)
    plt.title("甘特图")
    plt.xlabel("加工时间 /h")
    plt.ylabel("机器")
    plt.show()

def output_result(stij,tij):
    n = len(tij)
    l=len(tij[0])
    result=[]
    for num in range(1,7):
        res_temp=sort_result(n, l, num, tij, stij)
        res_temp= [f'M{num}：'] + res_temp
        result.append(res_temp)
    with open("result.txt",'w',encoding='utf-8') as outfile:
        for i in result:
            print(i)
            outfile.writelines(f'{i}\n')

def sort_result(n,l,num,tij,stij):
    temp1 = []
    temp2 = []
    for i in range(n):
        for j in range(l):
            if rij[i][j]==num:
                temp1.append(f'({i + 1},{i + 1}-{j + 1},{stij[i][j]},{stij[i][j] + tij[i][j]})')
                temp2.append(stij[i][j])
    machine1 = sorted(zip(temp1,temp2), key=lambda x: x[1])
    result_temp = zip(*machine1)
    result, sorted_item = [list(x) for x in result_temp]
    return result

if __name__ == '__main__':
    tij,rij=read_data("data.txt")
    stij=run_opt(tij,rij)
    plot_gantt(stij,tij,rij)
    output_result(stij, tij)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
