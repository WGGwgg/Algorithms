import copy
import time
import numpy as np
import matplotlib.pyplot as plt


class PSO():
    def __init__(self,MCN,popsize,c1,c2,func_sel):
        self.MCN = MCN                      # 算法迭代次数
        self.popsize = popsize              # 种群数量
        self.c1 = c1                        # 个体学习系数
        self.c2 = c2                        # 群体学习系数
        self.func_sel = func_sel            # 测试函数选择
        self.lb,self.ub,self.dim,self.v_lb,self.v_ub = self.get_boundary()

    # 获取测试函数边界、维度
    def get_boundary(self):
        dim = 30
        v_lb, v_ub = [-0.5, 0.5]
        if self.func_sel == 'F1':
            lb, ub = [-100, 100]
        elif self.func_sel == 'F2':
            lb, ub = [-10, 10]
        elif self.func_sel == 'F3':
            lb, ub = [-100, 100]
        elif self.func_sel == 'F4':
            lb, ub = [-100, 100]
        elif self.func_sel == 'F5':
            lb, ub = [-30, 30]
        elif self.func_sel == 'F6':
            lb, ub = [-100, 100]
        elif self.func_sel == 'F7':
            lb, ub = [-1.28, 1.28]
        elif self.func_sel == 'F8':
            lb, ub = [-500, 500]
        elif self.func_sel == 'F9':
            lb, ub = [-5.12, 5.12]
        elif self.func_sel == 'F10':
            lb, ub = [-32, 32]
        elif self.func_sel == 'F11':
            lb, ub = [-600, 600]
        elif self.func_sel == 'F12':
            lb, ub = [-50, 50]
        elif self.func_sel == 'F13':
            lb, ub = [-50, 50]
        elif self.func_sel == 'F14':
            lb, ub, dim = [-65, 65, 2]
        elif self.func_sel == 'F15':
            lb, ub, dim = [-5, 5, 4]
        elif self.func_sel == 'F16':
            lb, ub, dim = [-5, 5, 2]
        elif self.func_sel == 'F17':
            lb, ub, dim = [-5, 5, 2]
        elif self.func_sel == 'F18':
            lb, ub, dim = [-2, 2, 2]
        elif self.func_sel == 'F19':
            lb, ub, dim = [1, 3, 3]
        elif self.func_sel == 'F20':
            lb, ub, dim = [0, 1, 6]
        elif self.func_sel == 'F21':
            lb, ub, dim = [0, 10, 4]
        elif self.func_sel == 'F22':
            lb, ub, dim = [0, 10, 4]
        elif self.func_sel == 'F23':
            lb, ub, dim = [0, 10, 4]
        else:
            pass
        return lb, ub, dim, v_lb, v_ub

    # 初始化种群
    def initial_pop(self,ub,lb):
        pop = np.zeros((self.popsize,self.dim))
        for i in range(self.popsize):
            for j in range(self.dim):
                # pop[i][j] = lb + (ub - lb)*np.random.rand()
                pop[i][j] = 1
        return pop

    # 计算适应度
    def calculate(self,x):
        X = copy.deepcopy(x)
        dim = len(X)
        obj_val = 0
    # 定义F12、13中函数
        def y(x):
            return 1 + (x + 1) / 4

        def Ufun(x,a,k,m):
            y = 0
            for i in range(dim):
                if x[i]>a:
                    y += k*(x[i]-a)**m
                elif x[i]<-a:
                    y += k*(-x[i]-a)**m
                else:
                    y += 0
            # y = k*((x-a)**m)*(x>a) + k*((-x-a)**m)*(x<(-a)) + 0
            return y
    # 选择测试函数
        # 函数 Sphere Function
        if self.func_sel == 'F1':
            # obj_val = sum(X**2)
            obj_val = np.sum(X**2)

        # 函数 Schwefel's Problem 2.22
        elif self.func_sel == 'F2':
            # obj_val = sum(abs(X)) + math.prod(abs(X))
            obj_val = np.sum( np.abs(X)) + np.prod( np.abs(X))

        # 函数 Schwefel's Problem 1.2
        elif self.func_sel == 'F3':
            # for i in range(dim):
            #     obj_val += sum(X[:i+1])**2
            obj_val = np.sum( np.cumsum(X)**2)

        # 函数 Schwefel's Problem 2.21
        elif self.func_sel == 'F4':
            # obj_val = max(abs(X))
            obj_val = np.max( np.abs(X))

        # 函数 Rosenbrock's Function
        elif self.func_sel == 'F5':
            # obj_val = sum(100*(X[1:dim]-(X[:dim-1])**2)**2 + (X[:dim-1]-1)**2)
            f1 = 100 * (X[1:] - X[:-1] ** 2) ** 2
            f2 = (X[:-1] - 1) ** 2
            obj_val = np.sum(f1 + f2)

        # 函数 Step Function
        elif self.func_sel == 'F6':
            # obj_val = sum(abs(x+0.5)**2)
            obj_val = np.sum(np.floor(np.abs(X+0.5))**2)

        # 函数 Quartic Function i.e. Noise
        elif self.func_sel == 'F7':
            # obj_val = sum(np.multiply(np.arange(1,dim+1),(X**4))) + np.random.rand()
            i = np.arange(dim) + 1
            obj_val = np.sum(i*X**4) + np.random.uniform()

        # 函数 Schwefel's Problem 2.26
        elif self.func_sel == 'F8':
            # obj_val = sum(np.multiply(-X,np.sin(np.sqrt(abs(X)))))
            obj_val = -np.sum(X*np.sin(np.sqrt(np.abs(X))))

        # 函数 Rastrigin's Function
        elif self.func_sel == 'F9':
            # obj_val = sum(X**2-10*np.cos(2*np.pi*X)) + 10*dim
            obj_val = np.sum(X**2-10*np.cos(2*np.pi)+10)

        # 函数 Ackley's Function
        elif self.func_sel == 'F10':
            # obj_val = -20*np.exp(-0.2*np.sqrt(sum(x**2)/dim)) - np.exp(sum(np.cos(2*np.pi*X))/dim) + 20 + np.exp(1)
            f1 = -0.2 * np.sqrt(np.sum(X ** 2) / dim)
            f2 = np.sum(np.cos(2 * np.pi * X)) / dim
            obj_val = -20 * np.exp(f1) - np.exp(f2) + 20 + np.e

        # 函数 Griewank's Function
        elif self.func_sel == 'F11':
            # obj_val = sum(X**2)/4000 - np.prod(np.cos(X/np.sqrt(np.arange(1,dim+1)))) + 1
            i = np.arange(dim) + 1
            f1 = np.sum(X ** 2)
            f2 = np.prod(np.cos(X / np.sqrt(i)))
            obj_val = 1 / 4000 * f1 - f2 + 1

        # 函数 Penalized 1 Function
        elif self.func_sel == 'F12':
            obj_val = (np.pi/dim)*(10*((np.sin(np.pi*(1+(X[0]+1)/4)))*2)+sum((((X[:-1]+1)/4)**2)*(1+10*((np.sin(np.pi*(1+(X[1:]+1)/4))))**2))+((X[-1]+1)/4)**2)+Ufun(X, 10, 100, 4)
            # y1 = y(X[0])
            # yD = y(X[-1])
            # yi = y(X[:-1])
            # yi_1 = y(X[1:])
            # f1 = 10 * np.sin(np.pi * y1) ** 2
            # f2 = np.sum(((yi - 1) ** 2) * (1 + 10 * np.sin(np.pi * yi_1) ** 2))
            # f3 = (yD - 1) ** 2
            # obj_val = np.pi / dim * (f1 + f2 + f3) + Ufun(X, 10, 100, 4)

        # 函数 Penalized 2 Function
        elif self.func_sel == 'F13':
            obj_val = 0.1*((np.sin(3*np.pi*X[0]))**2 + sum((X[:-1]-1)**2*(1+(np.sin(3*np.pi*X[1:]))**2))+((X[dim-1]+1)**2)*(1+(np.sin(2*np.pi*X[-1]))**2))+Ufun(X, 5, 100, 4)
            # X1 = X[0]
            # XD = X[-1]
            # Xi = X[:-1]
            # Xi_1 = X[1:]
            # f1 = np.sin(3 * np.pi * X1) ** 2
            # f2 = np.sum((Xi - 1) ** 2 * (1 + np.sin(3 * np.pi * Xi_1) ** 2))
            # f3 = (XD - 1) ** 2 * (1 + np.sin(2 * np.pi * XD) ** 2)
            # obj_val = 0.1 * (f1 + f2 + f3) + Ufun(X, 5, 100, 4)

        # 函数 Shekel's Foxholes Function
        elif self.func_sel == 'F14':
            j = np.arange(25) + 1
            a1 = np.tile(np.array([-32, -16, 0, 16, 32]), 5)
            a2 = np.repeat(np.array([-32, -16, 0, 16, 32]), 5)
            X1 = X[0]
            X2 = X[1]
            f1 = j + (X1 - a1) ** 6 + (X2 - a2) ** 6
            obj_val = (1 / 500 + np.sum(1 / f1)) ** -1

        # 函数 Shekel's Foxholes Function
        elif self.func_sel == 'F15':
            a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
            b = np.array([4, 2, 1, 1 / 2, 1 / 4, 1 / 6, 1 / 8, 1 / 10, 1 / 12, 1 / 14, 1 / 16])
            X1 = X[0]
            X2 = X[1]
            X3 = X[2]
            X4 = X[3]
            f1 = X1 * (b ** 2 + b * X2)
            f2 = b ** 2 + b * X3 + X4
            obj_val = np.sum((a - f1 / f2) ** 2)

        # 函数 Six-Hump Camel-Back Function
        elif self.func_sel == 'F16':
            X1 = X[0]
            X2 = X[1]
            obj_val = 4*X1**2 - 2.1*X1**4 + X1**6/3 + X1*X2 - 4*X2**2 + 4*X2**4

        # 函数 Branin Function
        elif self.func_sel == 'F17':
            X1 = X[0]
            X2 = X[1]
            f1 = (X2 - 5.1 * X1 ** 2 / (4 * np.pi ** 2) + 5 * X1 / np.pi - 6) ** 2
            f2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(X1)
            obj_val = f1 + f2 + 10

        # 函数 Goldstein-Price Function
        elif self.func_sel == 'F18':
            X1 = X[0]
            X2 = X[1]
            f1 = 1 + (X1+X2+1)**2*(19-14*X1+3*X1**2-14*X2+6*X1*X2+3*X2**2)
            f2 = 30 + (2*X1-3*X2)**2*(18-32*X1+12*X1**2+48*X2-36*X1*X2+27*X2**2)
            obj_val = f1 * f2

        # 函数 Hartman-3
        elif self.func_sel == 'F19':
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[3.00, 10.0, 30.0],
                          [0.10, 10.0, 35.0],
                          [3.00, 10.0, 30.0],
                          [0.10, 10.0, 35.0]])
            P = np.array([[0.36890, 0.1170, 0.2673],
                          [0.46990, 0.4387, 0.7470],
                          [0.10910, 0.8732, 0.5547],
                          [0.03815, 0.5743, 0.8828]])
            for i in range(4):
                f1 = alpha[i]*np.exp(-np.sum(A[i]*(X-P[i])**2))
                obj_val = obj_val - f1

        # 函数 Hartman-6
        elif self.func_sel == 'F20':
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[10.0, 3.00, 17.0, 3.50, 1.70, 8.00],
                          [0.05, 10.0, 17.0, 0.10, 8.00, 14.0],
                          [3.00, 3.50, 1.70, 10.0, 17.0, 8.00],
                          [17.0, 8.00, 0.05, 10.0, 0.10, 14.0]])
            P = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                          [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                          [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                          [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
            for i in range(4):
                f1 = alpha[i]*np.exp(-np.sum(A[i]*(X-P[i])**2))
                obj_val = obj_val - f1

        # 函数 Shekel-5
        # F * (m=5) = -10.1532, F * (m=7) = -10.4029, F * (m=10) = -10.5364
        elif self.func_sel == 'F21':
            a = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                          [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
            c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
            for i in range(5):
                f1 = np.sum((X - a[:, i]) ** 2) + c[i]
                obj_val = obj_val - 1 / f1

        # 函数 Shekel-7
        elif self.func_sel == 'F22':
            a = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                          [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
            c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
            for i in range(7):
                f1 = np.sum((X - a[:, i]) ** 2) + c[i]
                obj_val = obj_val - 1 / f1
        # 函数 Shekel-10
        elif self.func_sel == 'F23':
            a = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                          [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
            c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
            for i in range(10):
                f1 = np.sum((X - a[:, i]) ** 2) + c[i]
                obj_val = obj_val - 1 / f1
        else:
            pass
    # # 将函数值转换为适应度
    #     if obj_val >= 0:
    #         fitness = 1 / (obj_val + 1)
    #     else:
    #         fitness = 1 + abs(obj_val)
    #     return fitness
        return obj_val

    # 检查边界
    def boundary_check(self,ub,lb,X):
        for i in range(self.dim):
            if X[i] > ub:
                X[i] = ub
            if X[i] < lb:
                X[i] = lb
        return X

    # 绘制收敛曲线
    def plot_curve(self,Best_chart):
        plt.plot(np.arange(1,self.MCN+1),Best_chart,'b--')
        plt.title('PSO-Benchmark Function {}'.format(self.func_sel))
        plt.xlabel('Iteration')
        plt.ylabel('Object Value')
        ax = plt.gca()
        plt.text(0.5,0.9,'Best Value = {:.3e}'.format(Best_chart[-1]),transform=ax.transAxes)
        plt.savefig('curve_{}.jpg'.format(self.func_sel))
        plt.close()

    # PSO主函数
    def pso_main(self):
        # 初始化粒子群
        pop_x = self.initial_pop(self.ub,self.lb)           # 初始化粒子群位置
        pop_v = self.initial_pop(self.v_ub,self.v_lb)       # 初始化粒子群速度
        fitness = np.zeros(self.popsize)
        # 计算适应度
        for i in range(self.popsize):
            fitness[i] = self.calculate(pop_x[i])
        # 记录个体历史最优位置
        pBest = pop_x
        pBestFitness = fitness
        # 记录种群历史最优位置
        gBest_index = np.argmin(fitness)
        gBset = pop_x[gBest_index]
        gBestFitness = fitness[gBest_index]
        # 记录迭代最优值
        Best_chart = np.zeros(self.MCN)
        Xbest_chart = np.zeros((self.MCN,self.dim))
        # 开始迭代
        print("**************PSO算法运行进度**************")
        starttime = time.perf_counter()
        for it in range(self.MCN):
            # 对每个粒子进行位置更新
            for i in range(self.popsize):
                # 速度更新
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                pop_v[i] = pop_v[i] + self.c1*r1*(pBest[i] - pop_x[i]) + self.c2*r2*(gBset - pop_x[i])
                pop_v[i] = self.boundary_check(self.v_ub,self.v_lb,pop_v[i])                    # 速度边界检查
            for i in range(self.popsize):
                # 位置更新
                pop_x[i] = pop_x[i] + pop_v[i]
                pop_x[i] = self.boundary_check(self.ub, self.lb, pop_x[i])                      # 位置边界检查
            # 对每个粒子计算新位置适应度
            for i in range(self.popsize):
                fitness[i] = self.calculate(pop_x[i])
                # 更新个体历史最优位置
                if fitness[i] < pBestFitness[i]:
                    pBest[i] = pop_x[i]
                    pBestFitness[i] = fitness[i]
            # 更新种群历史最优位置
            if min(fitness) < gBestFitness:
                gBest_index = np.argmin(fitness)
                gBset = pop_x[gBest_index]
                gBestFitness = fitness[gBest_index]
            # 记录当前迭代最优解和最优值
            Xbest_chart[it] = gBset
            Best_chart[it] = gBestFitness
            # 运行进度显示
            finsh = "▓" * round(100 * it / self.MCN)
            need_do = "-" * round(100*(self.MCN-it)/self.MCN)
            progress = ((it+1)/self.MCN) * 100
            dur = time.perf_counter() - starttime
            print("\r{:^3.1f}%[{}->{}]{:.2f}s".format(progress, finsh, need_do, dur), end="")
            time.sleep(0.05)
        # 打印收敛结果
        print('\n函数{0}的收敛最优值为：{1:.2e}'.format(self.func_sel,float(Best_chart[-1])))
        # 绘制收敛曲线
        # self.plot_curve(Best_chart)

if __name__ == '__main__':
    list_test = [f'F{i+1}' for i in range(9,13)]
    for i in list_test:
        Run = PSO(10, 10, 2.0, 2.0, func_sel=i)
        Run.pso_main()
    # Run = PSO(100,100,2.0,2.0,func_sel='F12')
    # Run.pso_main()