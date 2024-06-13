import re
import queue
import copy
def Judge(clause1, clause2):
    # 判断能否变量替换
    hash = []
    for i in range(1, len(clause1)):
        if clause1[i] not in variable and clause2[i] not in variable and clause1[i] == clause2[i]:
            continue
        elif clause1[i] not in variable and clause2[i] in variable:
            hash.append((clause2[i], clause1[i]))
        else:
            return False
    return hash
def resolve(KB, parent, assign, clause1_index, i, clause2_index, j, hash_result=None):
    cp1 = copy.deepcopy(KB[clause1_index])
    cp2 = copy.deepcopy(KB[clause2_index])
    del cp1[i]
    del cp2[j]
    if hash_result:
        for hash_pair in hash_result:
            for index, predicate in enumerate(cp2):
                while hash_pair[0] in predicate:
                    cp2[index][predicate.index(hash_pair[0])] = hash_pair[1]

    parent.append([clause1_index, i, clause2_index, j])
    assign.append(hash_result if hash_result else [])
    newkb = list(map(list, set(map(tuple, (cp1 + cp2)))))
    KB.append(newkb)
    if not newkb:
        return True
    return False


def mgu(KB, assign, parent):
    # 归结合一函数
    for clause1_index, clause1 in enumerate(KB):
        for clause2_index, clause2 in enumerate(KB):
            if clause1_index == clause2_index:
                continue
            for i, predicate1 in enumerate(clause1):
                for j, predicate2 in enumerate(clause2):
                    # 谓词相反
                    if (predicate1[0] == '~' + predicate2[0] or predicate2[0] == '~' + predicate1[0]) and len(predicate1) == len(predicate2):
                        if predicate1[1:] == predicate2[1:]:
                            if resolve(KB, parent, assign, clause1_index, i, clause2_index, j):
                                return
                        else:
                            hash_result = Judge(predicate1, predicate2)
                            if hash_result:
                                if resolve(KB, parent, assign, clause1_index, i, clause2_index, j, hash_result):
                                    return


def prun(n, KB, assign, parent):
    # 使用二叉树结构层序遍历剪枝
    prunkb = []
    q = queue.Queue()
    q.put(parent[-1])
    prunkb.append([KB[-1], parent[-1], assign[-1]])

    # 只有非知识库内的句子才会有变量替换
    while not q.empty():
        cur = q.get()
        # 大的先进队列(后推出来)，符合推理常理
        if cur[0] > cur[2]:
            if cur[0] >= n:
                prunkb.append([KB[cur[0]], parent[cur[0]], assign[cur[0]]])
                q.put(parent[cur[0]])
            if cur[2] >= n:
                prunkb.append([KB[cur[2]], parent[cur[2]], assign[cur[2]]])
                q.put(parent[cur[2]])
        else:
            if cur[2] >= n:
                prunkb.append([KB[cur[2]], parent[cur[2]], assign[cur[2]]])
                q.put(parent[cur[2]])
            if cur[0] >= n:
                prunkb.append([KB[cur[0]], parent[cur[0]], assign[cur[0]]])
                q.put(parent[cur[0]])
    return prunkb


def labeling(n, prunkb):
    # 重新标号，使用字典对应
    newindex = {i: None for i in range(n)}
    seen_indexes = set()
    for item in prunkb:
        indexes = item[1]
        # parent[0] and parent[2]
        for index in (indexes[0], indexes[2]):
            if index not in newindex and index not in seen_indexes:
                newindex[index] = None
                seen_indexes.add(index)
    newindex = sorted(newindex.keys())
    newindex = {x: newindex.index(x) + 1 for x in newindex}
    return newindex


def change2str(lst):
    # 变量替换
    # 初始化一个空字符串
    result = ""
    # 遍历列表中的元组
    for item in lst:
        # 将元组的第一个元素作为键，第二个元素作为值，拼接成字符串
        result += f"{item[0]}={item[1]},"
    # 去除最后一个逗号
    result = result.rstrip(",")
    # 返回结果字符串
    if result == "":
        return result
    return '{' + result + '}'


def restore(lst):
    # KB还原回正常形式
    # 初始化一个空字符串
    result = ""
    # 遍历列表中的元素
    for i, item in enumerate(lst):
        # 如果是第一个元素，添加开头的字符串
        if i == 0:
            result += item + '('
        else:
            result += item + ','
    # 返回结果字符串
    return result[:-1] + ')'


def int2str(kb, line, num):
    if len(kb[line]) == 1:
        return ''
    else:
        return chr(num + 97)


def oput(n, kb, prunkb, newindex):
    num = n
    for i, j in enumerate(prunkb):
        if i == len(prunkb) - 1:
            print(num + i + 1, f"R[{newindex[j[1][0]]},{newindex[j[1][2]]}] = []")
        else:
            #
            print(num + i + 1,
                  f"R[{newindex[j[1][0]]}{int2str(kb, j[1][0], j[1][1])},{newindex[j[1][2]]}{int2str(kb, j[1][2], j[1][3])}]{change2str(j[2])} = ",
                  end = '')
        if len(j[0]) == 0:
            return
        elif len(j[0]) == 1:
            print('('+restore(j[0][0])+','+')')
        else:
            print('(', end = '')
            for k in range(len(j[0])):
                if k is not len(j[0]) - 1:
                    print(restore(j[0][k]), end = ',')
                else:
                    print(restore(j[0][k]), end = '')
            print(')')


variable = ['x', 'y', 'z', 'u', 'v', 'w','xx','yy','zz','uu','vv','ww']


def final_resolve(input_str:str):
    input_str = input_str[5:].strip("{}")
    # 按逗号分割每个元组，并处理每个元组
    result = []
    matches = input_str.split('),(')
    matches[0] = matches[0][1:]
    matches[len(matches)-1] = matches[len(matches)-1][:-1]
    for index, match in enumerate(matches):
        print(index+1, ' ', '(', match, ')', sep = '')
        match = match.strip(',') # 去掉尾部的','
        result.append(match)

    KB = []
    for element in result:
        matches = (re.findall(r'~?\w+\(\w+,*\w*\)', element))
        KB.append(matches)
    for i in range(len(KB)):
        for j in range(len(KB[i])):
            KB[i][j] = KB[i][j].replace('(', ",").replace(')', '').split(',')

    # 记忆变量替换的列表assign和记录父子句的列表Parent
    n = len(KB)
    assign = [[] for _ in range(n)]
    parent = [[] for _ in range(n)]

    mgu(KB, assign, parent)
    
    prunkb = prun(n, KB, assign, parent)

    newindex = labeling(n, prunkb)

    prunkb = prunkb[::-1]
    oput(n, KB, prunkb, newindex)


    
def main():
    KB_str0 = "{(GradStudent(sue),),(~GradStudent(x),Student(x)),(~Student(x),HardWorker(x)),(~HardWorker(sue),)}"
    KB_str1 = "{(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),(L(tony,snow),),(~A(x),S(x),C(x)),(~C(y),~L(y,rain)),(L(z,snow),~S(z)),(~L(tony,u),~L(mike,u)),(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}"
    KB_str2 = "{(On(tony,mike),),(On(mike,john),),(Green(tony),),(~Green(john),),(~On(xx,yy),~Green(xx),Green(yy))}"
    print ('-----------Test0-------------')
    final_resolve(KB_str0)
   # print ('-----------Test1-------------')
   # final_resolve(KB_str1)
   # print ('-----------Test2-------------')
    #final_resolve(KB_str2)
if __name__ == '__main__':
    main()