from cmath import exp
import sys
import matplotlib.pyplot as plt 
import random
import math
import matplotlib.animation as animation
from scipy import rand
import time
class GeneticAlgTSP:
    
    def __init__(self, filename):
        self.filename = filename
        file = open(filename,"r")
        list = file.readlines() #每一行数据写入到list中
        self.citys = []
        for line in list:
            tmp = line.strip().split()
            if tmp == '' or tmp == 'EOF':
                continue
            temp=[]
            temp.append(int(tmp[0]))
            temp.append(float(tmp[1]))
            temp.append(float(tmp[2]))
            self.citys.append(temp)           
        self.population = []
        lenth = len(self.citys)
        self.mutation = 0.3 #变异概率
        #随机生成
        for i in range(50):
            curr = random.sample(range(1, lenth+1), lenth)
            self.population.append(curr)
        tool = self.population[1][:]
        #贪心生成另外一半

        #贪心算法
        def distance(a, b):
            s = pow(self.citys[a-1][1]-self.citys[b-1][1], 2) + pow(self.citys[a-1][2]-self.citys[b-1][2], 2)
            return math.sqrt(s)
        def selection1(visited,gcur):
            max = 9999999
            flag = -1
            for i in range(len(gcur)):
                if gcur[i] not in visited:
                    for j in range(len(visited)):
                        tmp = distance(visited[j], gcur[i])
                        if tmp < max:
                            max = tmp
                            flag = i
            return flag

        for i in range(50):
            sta_city = random.randint(1,lenth-1)
            visited = []
            visited.append(sta_city)
            while len(visited) < lenth:
                index = selection1(visited,tool)
                visited.append(tool[index])
            self.population.append(visited)

    def distance(self, a, b):
        s = pow(self.citys[a-1][1]-self.citys[b-1][1],2) + pow(self.citys[a-1][2]-self.citys[b-1][2],2)
        return math.sqrt(s)

    def evaluate(self, path):
        s = 0
        for i in range (1,len(path)):
            s += self.distance(path[i-1], path[i])
        s += self.distance(1,path[len(path)-1])
        return s 

    #交叉下一代
    def reproduce_ox(self, parent1, parent2):
        length = len(parent1)
        sta,end = random.sample(range(1, length), 2)
        if end < sta:
            sta,end = end,sta  

        child = parent1[sta:end]
        child1 = []
        child2 = []
        for i in range(0, length):
            if parent2[i] not in child:
                if i<sta:
                    child1.append(parent2[i])
                else:
                    child2.append(parent2[i])
        child = child1 + child + child2
        return child
    def tournament_selection(self, population, tournament_size):
        # 随机选择锦标赛的参与者
        tournament = random.sample(population, tournament_size)
        # 选择最优的个体
        best_individual = min(tournament, key=self.evaluate)
        return best_individual
    #变异函数
    def mutate(self, path):
        lenth = len(path)
        charge = random.random()
        if charge < 0.4:
            a1,a2,a3 = random.sample(range(1, lenth-1), 3) 
            if a1>a2:
                a1,a2 = 2,a1
            if a2>a3:
                a2,a3 = a3,a2
            if a1>a2:
                a1,a2 = a2,a1
            tmp = path[0:a1]+path[a2:a3]+path[a1:a2]+path[a3:lenth]
        elif charge<0.6:
            i = random.randint(1,lenth-2)
            j = random.randint(2,lenth-1)
            #print(i,"<->",j)
            #print(path)
            if i != j:
                path[i],path[j] = path[j],path[i]
                tmp = path[:] 
                path[i],path[j] = path[j],path[i]
            else:
                tmp = path[:]
        else:
            k1,k2 = random.sample(range(1, lenth-1), 2)
            if k1 > k2:
                k1,k2 = k2,k1
            tmp = path[0:k1]+path[k1:k2][::-1]+path[k2:lenth]
            #print(len(tmp))
        return tmp
    def iterate(self, num):
        start_time = time.time()
        min = 99999999
        for var in self.population:
            if min > self.evaluate(var):
                min = self.evaluate(var)
        print(f'种群最小路径：{min}')
        print("----------------------------------------")
        ims = []
        fig1 = plt.figure(1)#用于生成动态图
        dis_change = []
        iteration = 0 #迭代次数
        while iteration < num:
            new_population = [] #新种群
            for count in range(0,10): #杂交10次
                #选择亲代
                ch1 = self.tournament_selection(self.population, 5)
                ch2 = self.tournament_selection(self.population, 5)
                #杂交出两个子代
                child1 = self.reproduce_ox(ch1, ch2)
                child2 = self.reproduce_ox(ch2, ch1)
                #有一定概率变异
                if random.random() < self.mutation :
                    child1 == self.mutate(child1)
                    child2 == self.mutate(child2)
                #为了防止子代全都一样，当父代相等时进行变异
                if child1 == child2:
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                #并入新种群
                new_population.append(child1)
                new_population.append(child2)
            flag=-1
            max=9999999
            #选取最优良的个体保存到下一代
            for i2 in range(0, len(self.population)):
                if self.evaluate(self.population[i2]) < max:
                    max = self.evaluate(self.population[i2])
                    flag=i2
            temp_one = self.population[flag][:]
            #显示当前迭代次数和最优值
            print(iteration, self.evaluate(temp_one))
            dis_change.append(max)
            self.population.clear()
            #更新
            self.population = new_population[0:19]
            new_population.clear()
            self.population.append(temp_one)
            iteration += 1
            #每隔一段时间进行采样
            if iteration%10==0:
                x1=[]
                y1=[]
                x1.append(self.citys[0][1])
                y1.append(self.citys[0][2])
                for var in temp_one:
                    x1.append(self.citys[var-1][1])
                    y1.append(self.citys[var-1][2])
                x1.append(self.citys[0][1])
                y1.append(self.citys[0][2]) 
                im = plt.plot(x1, y1, marker = '.', color = 'red',linewidth=1) 
                ims.append(im)
        elapsed_time = time.time() - start_time
        print(f'After {iteration} iterations, elapsed time: {elapsed_time/60:.2f} minutes')
        #结束杂交
        print("loop end")
        #找出最优解
        flag=-1
        max=9999999
        for i in range(0, len(self.population)):
            if self.evaluate(self.population[i]) < max:
                min = self.evaluate(self.population[i])
                flag = i 
        curr = self.population[flag]
        xo=[]
        yo=[]
        xo.append(self.citys[0][1])
        yo.append(self.citys[0][2])
        for var in curr:
            xo.append(self.citys[var-1][1])
            yo.append(self.citys[var-1][2])
        xo.append(self.citys[0][1])
        yo.append(self.citys[0][2])
        print(f'最短路径长度:{self.evaluate(curr)}')
        print(f'城市的访问次序：{curr}')

        fig2 = plt.figure(2)#用于显示最终的解
        plt.plot(xo, yo, marker = '.', color = 'red',linewidth=1) 
        #保存动态图
        ani = animation.ArtistAnimation(fig1, ims, interval=200, repeat_delay=1000)
        ani.save("TSP.gif", writer='pillow')

        fig3 = plt.figure(3)#用于显示总共费用的降低过程
        plt.title('the evolution of the cost')
        x_=[i for i in range(len(dis_change))]
        plt.plot(x_,dis_change)
        plt.show()

        print("End of all, and you got the best answer!")
        return self.evaluate(curr),elapsed_time