from GeneticAlgTSP import *

def main():
    filename = 'dj38.txt'

    num_iterations = 5000
    result=[]
    result_time=[]
    # for i in range(30):
    #     TSP = GeneticAlgTSP(filename)
    #     num,time=TSP.iterate(num_iterations)
    #     result.append(num)
    #     result_time.append(time)
    TSP = GeneticAlgTSP(filename)
    num,time=TSP.iterate(num_iterations)
    result.append(num)
    result_time.append(time)
    print("The interation  is",num_iterations)
    print("eacch case distance is")
    print(result)
    print("each case time is\n",result_time)
    print("The average distance is",sum(result)/len(result))
    print("The average time is",sum(result_time)/len(result_time))
    # a=[69.05011367797852, 65.33138966560364, 66.65562510490417, 69.88337349891663, 65.39635181427002, 65.6269884109497, 65.77681827545166, 67.2089295387268, 64.72636795043945, 66.2793116569519, 64.96787810325623, 59.687366247177124, 62.76708459854126, 62.91665530204773, 62.73270893096924, 63.67981457710266, 63.49577260017395, 62.94902276992798, 61.7575044631958, 62.2294921875, 63.06479525566101, 63.7184624671936, 62.887633085250854, 62.86031746864319, 62.4582793712616, 62.92140221595764]
    # print(sum(a)/len(a))
if __name__ == '__main__':
    main()