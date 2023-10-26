# %%
import numpy as np
from sympy import *
from sympy.combinatorics import Permutation

# %%
class root_system_B():
    def __init__(self,n):
        self.rank = n
        self.dimension = self.dimension()
        self.simple_roots = self.simple_pos_roots()
        self.positive_roots = self.positive_roots()
        self.highest_root = self.highest_root()
        self.num_of_roots = self.num_of_roots()
        self.num_of_pos_root = self.num_of_pos_root()
        self.dynkin_diagram = self.dynkin_diagram()

    def dimension(self):
        return 2 * self.rank + 1
    
    def basic_root(self, i,j):
        n = self.rank
        root = [0] * n
        root[i-1] = 1
        root[j-1] = -1
        return root

    def root_to_matrix(self,root, x): 
        #x is a string for variable 
        x = Symbol(str(x))
        n = self.rank
        m = self.dimension
        mat = eye(m,m)
        total = sum(root)
        if total == 2:
            first = root.index(1) 
            second = root.index(1,first+1,n) 
            mat[first, m-1-second] =  x
            mat[second, m-1-first] = -x
        elif total == 1:
            idx = root.index(1)
            mat[idx,n] = x
            mat[n,m-1-idx] = -x
        elif total == 0:
            first = root.index(1)
            second = root.index(-1)
            #if first < second:
            mat[first, second] = x
            mat[m-1-second,m-1-first] = -x
            #else:
            #    mat[second,first] = x
            #    mat[m-1-first,m-1-second] = -x
        elif total == -1:
            idx = root.index(-1)
            mat[n,idx] = x
            mat[m-1-idx,n] = -x
        else:
            first = root.index(-1) 
            second = root.index(-1,first+1,n) 
            mat[m-1-second,first] =  x
            mat[m-1-first,second] = -x
        return mat
    
    def simple_root(self, i):
        #return the i-th simple positive root, indexing from 1.
        n = self.rank
        if i < n:
            return self.basic_root(i,i+1) 
        else:
            root = [0] * n
            root[n-1] = 1
            return root
        
    def simple_pos_roots(self):
        n = self.rank
        sim_pos_roots = {}
        for i in range(1,n+1):
            sim_pos_roots[i] = self.simple_root(i) 
        return sim_pos_roots

    def positive_roots(self):
        n = self.rank
        pos_roots = {}
        k = 0
        for i in range(n-1):
            for j in range(i+1,n):
                k += 1
                pos_roots[k] = self.basic_root(i+1,j+1)
                k += 1
                root = self.basic_root(i+1,j+1)
                root[j] = 1
                pos_roots[k] = root
        for i in range(n):
            k += 1
            root = [0] * n
            root[i] = 1
            pos_roots[k] = root     
        return pos_roots

    def highest_root(self):
        root = self.basic_root(1,2)
        root[1] = 1
        return root

    def num_of_roots(self):
        n = self.rank
        return 2 * (n ** 2)
    
    def num_of_pos_root(self):
        return self.num_of_roots // 2

    def dynkin_diagram(self):
        n = self.rank
        diag = '---'.join("0" for i in range(1, n)) 
        diag += '==>0\n'
        diag += '   '.join(str(i) for i in range(1, n+1))
        return diag


# %%
class unipotent_orbit_B():
    def __init__(self, partition):
        self.dim = sum(partition)
        self.partition = partition
        self.diagonal = self.diagonal_ele()
        self.diagonal_matrix = self.diagonal_ele_mat()
        self.dimension = self.G_dimension()

    def diagonal_ele(self):
        diagonal = []
        for i in self.partition:
            n = i - 1
            j = n
            while j >= -n:
                diagonal.append(j)
                j -= 2
        diagonal = sorted(diagonal, reverse = True)
        return diagonal
    
    def diagonal_ele_mat(self):
        t = Symbol('t')
        diagonal = self.diagonal
        d = []
        for p in diagonal:
            d.append(t**p)
        return diag(d, unpack = True)
    
    def G_dimension(self):
        diml = 0
        n = self.dim
        diagonal = self.diagonal
        for i in range(0,n):
            for j in range(i+1, n):
                if diagonal[i] - diagonal[j] >= 2:
                    diml += 1
                elif diagonal[i] - diagonal[j] == 1:
                    diml += 1/2
        if 0 in diagonal:
            k = diagonal.index(0)
        else:
            k = n//2
        return int((diml - k) //2)
    
    def unipotent(self,lvl):
        x = Symbol('*')
        n = self.dim
        mat = eye(n,n)
        diagonal = self.diagonal
        for i in range(n-1):
            for j in range(i+1,n):
                if i+j == n-1:
                    continue
                if diagonal[i] - diagonal[j] >= lvl:
                    mat[i,j] = x     
        return mat

# %%
class weyl_group_B():
    def __init__(self, rank):
        self.rank = rank
        self.generater = self.generater()
        self.group_order = self.group_order()

    def generater(self):
        n = self.rank
        gen = []
        for i in range(n):
            reflection = 'r' + str(i+1)
            gen.append(reflection)
        return gen
    
    def group_order(self):
        n = self.rank
        return factorial(n) * (2**n)

    def reduced_element(self, ele):
        ref = list(ele)
        num = ref[1::3]
        pt = 0
        while pt < len(num) - 1:
            if num[pt] == num[pt+1]:
                del num[pt]
                del num[pt]
                pt -= 1
            else:
                pt += 1
        reduced =[]
        for i in num:
            word = 'r' + str(i)
            reduced.append(word)
        if not reduced:
            return 'e'
        else:
            return '*'.join(reduced)
        
    def long_weyl_element(self, i):
        p = i
        n = self.rank
        long = []
        while p < n:
            word = 'r' + str(p)
            long.append(word)
            p += 1
        while p >= i:
            word = 'r' + str(p)
            long.append(word)
            p -= 1
        return '*'.join(long)

    def transpose_to_reflection(self,transpose):
        s = transpose[0]
        e = transpose[1]
        reflection = []
        p = e - 1
        while p > s:
            word = 'r' + str(p)
            reflection.append(word)
            p -= 1
        while p < e:
            word = 'r' + str(p)
            reflection.append(word)
            p += 1
        return '*'.join(reflection)
    
    def permutation_to_reflection(self, permutation):
        res = []
        for i in range(len(permutation)):
            if permutation[i] < 0:
                permutation[i] = -1 * permutation[i]
                res.append(self.long_weyl_element(permutation[i]))
            permutation[i] -= 1
        p = self.permutation_to_transposition(Permutation(permutation))
        for tran in p:
            tran_new = (tran[0]+1 ,tran[1]+1)
            res.append(self.transpose_to_reflection(tran_new))
        ref = '*'.join(res)
        ref = self.reduced_element(ref)
        return ref

    def matrix_to_reflection(self,mat):
        #You have to manually make sure the input is a permutation matrix.
        n = self.rank
        m = 2 * n + 1
        perm = [0] * n
        for i in range(n):
            col = list(mat.col(i))
            idx = col.index(1)
            if idx < n:
                perm[i] = idx + 1
            elif idx > n:
                perm[i] = -(m-idx)
        res = self.permutation_to_reflection(perm)
        return res

    def reflection_to_matrix(self, ele):
        n = self.rank
        m = 2 * n + 1
        matrix = eye(m,m)
        ele = self.reduced_element(ele)
        if ele == 'e':
            return matrix
        element = list(ele)
        reflections = element[1::3]
        for ref in reflections:
            r = int(ref)
            mat = eye(m,m)
            if r == n:
                mat[n,n] = -1
                mat[n-1,n-1] = 0
                mat[n-1,n+1] = 1
                mat[n+1,n-1] = 1
                mat[n+1,n+1] = 0
            else:
                mat[r-1,r-1] = 0
                mat[r-1,r] = 1
                mat[r,r-1] = 1
                mat[r,r] = 0
                mat[m-r,m-r] = 0
                mat[m-r-1,m-r-1] = 0
                mat[m-r-1,m-r] = 1
                mat[m-r,m-r-1] = 1
            matrix *= mat
        return matrix

    def element_order(self,ele):
        n = self.rank
        m = 2 * n + 1
        mat = self.reflection_to_matrix(ele)
        order = 1
        while mat != eye(m,m):
            mat *= self.reflection_to_matrix(ele)
            order += 1
        return order
    
    def permutation_to_transposition(self, perm):
        perm = perm.cyclic_form
        res = []
        for x in perm:
            nx = len(x)
            if nx == 2:
                res.append(x)
            elif nx > 2:
                for i in range(0,nx-1):
                    if x[i] < x[i+1]:
                        res.append((x[i],x[i+1]))
                    else:
                        res.append((x[i+1],x[i]))
        return res
    
    def element_length(self,ele):
        counter = 0
        ele = self.reduced_element(ele)
        if ele == 'e':
            return counter
        for i in ele:
            if i == 'r':
                counter += 1
        return counter


