# %%
import numpy as np
from sympy import *
from sympy.combinatorics import Permutation

# %%
class root_system_A():
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
        #This returns the dimension of the ambient euclidean space.
        return self.rank + 1

    def basic_root(self, i,j):
        #This returns the root e_i - e_j, indexing from 1.
        m = self.dimension
        root = [0] * m
        root[i-1] = 1
        root[j-1] = -1
        return root

    def root_to_matrix(self,root,x):
        #This express the corresponding root matrix in the Lie group GL_n (Not the lie algebra).
        x = Symbol(str(x))
        m = self.dimension
        mat = eye(m,m)
        i = root.index(1)
        j = root.index(-1)
        mat[i,j] = x
        return mat
    
    def simple_root(self, i):
        #return the i-th simple positive root, indexing from 1.
        return self.basic_root(i,i+1)

    def simple_pos_roots(self):
        #This returns the dictionary containing all the canonical positive simple roots. 
        #The size of the dict is equal to the rank of the root system. 
        n = self.rank
        sim_pos_roots = {}
        for i in range(1,n+1):
            sim_pos_roots[i] = self.simple_root(i) 
        return sim_pos_roots

    def positive_roots(self):
        #This returns the dictionary containing all the positive roots.
        n = self.rank
        pos_roots = {}
        k = 0
        for i in range(n):
            for j in range(i+1,n+1):
                k += 1
                pos_roots[k] = self.basic_root(i+1,j+1)
        return pos_roots

    def highest_root(self):
        #This returns the highest root
        return self.basic_root(1,self.rank)

    def num_of_roots(self):
        #This returns the number of roots, including the negative ones.
        n = self.rank
        return n * (n+1)
    def num_of_pos_root(self):
        #This returns the number of positive roots.
        return self.num_of_roots // 2

    def dynkin_diagram(self):
        #This returns the corresponding Dynkin diagram, with nodes labelled.
        n = self.rank
        diag = '---'.join("o" for i in range(1, n+1))
        diag += '\r\n'
        diag += '   '.join(str(i) for i in range(1, n+1))
        return diag

# %%
class unipotent_orbit_A():
    def __init__(self, partition):
        self.dim = sum(partition)
        self.partition = partition
        self.diagonal = self.diagonal_ele()
        self.diagonal_matrix = self.diagonal_ele_mat()
        self.dimension = self.G_dimension()

    def diagonal_ele(self):
        #This returns the diagonal element corresponding to the unipotent orbit, 
        #in list form with the powers in decreasing order.
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
        #This returns the diagonal element corresponding to the unipotent orbit in a diagonal matrix form.
        t = Symbol('t')
        diagonal = self.diagonal
        d = []
        for p in diagonal:
            d.append(t**p)
        return diag(d, unpack = True)
    
    def G_dimension(self):
        #This returns the Gelfand-Kirillov dimension of the given unipotent orbit.
        diml = 0
        n = self.dim
        diagonal = self.diagonal
        for i in range(0,n):
            for j in range(i+1, n):
                if diagonal[i] - diagonal[j] >= 2:
                    diml += 1
                elif diagonal[i] - diagonal[j] == 1:
                    diml += 1/2
        return diml
    
    def unipotent(self,lvl):
        #This returns a matrix representation of the filtration 
        #on the upper unipotent subgroup under the action of the diagonal element. 
        #The non-trivial entries are presented by * .
        x = Symbol('*')
        n = self.dim
        mat = eye(n,n)
        diagonal = self.diagonal
        for i in range(n-1):
            for j in range(i+1,n):
                if diagonal[i] - diagonal[j] >= lvl:
                    mat[i,j] = x     
        return mat

# %%
class weyl_group_A():
    def __init__(self, rank):
        self.rank = rank
        self.generater = self.generater()
        self.group_order = self.group_order()

    def generater(self):
        #This returns a set of generators for the weyl group, 
        #which are simple reflections with respect to each of the positive simple roots.
        n = self.rank
        gen = []
        for i in range(n):
            reflection = 'r' + str(i+1)
            gen.append(reflection)
        return gen
    
    def group_order(self):
        #This returns the order of the given weyl group.
        n = self.rank
        return factorial(n + 1)

    def reduced_element(self, ele):
        #This is a helper function to detect any duplicated simple reflections 
        #when joining multiple weyl group elements. 
        #Eg, r1*r1*r2 ---> r2.
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
        #This returns the long weyl group element (i,n+1-i) in simple reflections. 
        #It is useful in constructing word expressions for weyl group elements in B,C,D, but not so useful in A.
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
        #This returns the expression in simple reflections for a weyl group element which is a transpose.
        #Eg, (1,3) --->r1*r2*r1.
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
        #This returns an expression in simple reflections for a weyl group element which is a permutation.
        #The permutation is starting from indexing 1, which is different from the default sympy environment.
        #Eg, [2,3,1] ---> r1*r2, Note that this is not necessarily the shortest representation.
        res = []
        for i in range(len(permutation)):
            #if permutation[i] < 0:
                #permutation[i] = -1 * permutation[i]
                #res.append(self.long_weyl_element(permutation[i]))
            permutation[i] -= 1
        p = self.permutation_to_transposition(Permutation(permutation))
        for tran in p:
            tran_new = (tran[0]+1 ,tran[1]+1)
            res.append(self.transpose_to_reflection(tran_new))
        ref = '*'.join(res)
        ref = self.reduced_element(ref)
        return ref

    def matrix_to_reflection(self,mat):
        #This returns an expression in simple reflections for a permutation matrix. 
        #You have to manually make sure the input is a permutation matrix.
        n = self.rank + 1
        perm = [0] * n
        for i in range(n):
            col = list(mat.col(i))
            idx = col.index(1)
            perm[i] = idx + 1
        res = self.permutation_to_reflection(perm)
        return res

    def reflection_to_matrix(self, ele):
        #This returns a permutation matrix for a given reflection.
        n = self.rank + 1
        matrix = eye(n,n)
        ele = self.reduced_element(ele)
        if ele == 'e':
            return matrix
        element = list(ele)
        reflections = element[1::3]
        for ref in reflections:
            r = int(ref)
            mat = eye(n,n)
            mat[r-1,r-1] = 0
            mat[r-1,r] = 1
            mat[r,r-1] = 1
            mat[r,r] = 0
            matrix *= mat
        return matrix

    def element_order(self,ele):
        #This returns the order of a given weyl group element.
        #This function actually finds the order of the corresponding permutation matrix.
        n = self.rank + 1
        mat = self.reflection_to_matrix(ele)
        order = 1
        while mat != eye(n,n):
            mat *= self.reflection_to_matrix(ele)
            order += 1
        return order

    def permutation_to_transposition(self, perm):
        #This is a helper function that translates a permutation into transposition. 
        #The built in transpositions() function in sympy decomposes in a way that may creates unnecessarily longer words.
        #Eg, [2,3,1] ---> (1,3)(1,2) instead of (1,2)(2,3), resulting r2*r1*r2*r1 instead of r1*r2.
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
        #This returns the length of a given reflection.
        #If the word is reduced to none, identity is returned.
        counter = 0
        ele = self.reduced_element(ele)
        if ele == 'e':
            return counter
        for i in ele:
            if i == 'r':
                counter += 1
        return counter

# %%



