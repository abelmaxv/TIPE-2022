import numpy as np
import matplotlib.pyplot as plt
import random

#Implementation des processus de decision markovien (MDP) stationnaire 

class Proba():
    """Les probabilites du MDP sont representees par la liste des |A| matrices P_a """
    
    def __init__(self, list_proba )->None:
        self.P= list_proba
        
class Reward ():
    """Les recompenses sont representees par la liste des |A| matrices r_a"""

    def __init__(self,list_rewards)->None:
        self.R = list_rewards


class MDP ():
    """Les ensembles S (resp.A) sont caracterises par les entiers |S| (resp.|A|) """
    
    def __init__(self,s:int,a:int,p:Proba,r:Reward) -> None:

        assert len(p.P)== a and len(r.R)==a
        self.S = s
        self.A = a
        self.P = p.P
        self.R = r.R

       
    def value_iteration(self,gamma:float, eps:float)->tuple:
        """Algorithme d'iteration sur les valeurs. Les fonction valeur et les politique sont
        representees par des vecteurs de taille |S|"""

        V= np.array([0]*self.S)
        while True:
            newV = np.copy(V)
            for s in range(self.S):
                newV[s]=max([sum([self.P[a][s,s2]*(self.R[a][s,s2]+gamma*V[s2]) for s2 in range(self.S)]) for a in range(self.A)])
            if np.linalg.norm(newV-V)<=eps: 
                break
            V = newV
        pi = np.array([0]*self.S)
        for s in range(self.S):
            f = lambda a :  sum([self.P[a][s,s2]*(self.R[a][s,s2]+gamma*V[s2]) for s2 in range(self.S)])
            pi[s]=argmax(f,self.A)

            
        return V, pi




    def graph_Q(self,s:int,a:int,gamma:float, N)->None:
        """Pour tracer l'evolution de Q(s,a) au fil des itérations """
       
        V= np.array([0]*self.S)
        Q = []
        
        for i in range(N): 
            newV = np.copy(V)
            for s in range(self.S):
                newV[s]=max([sum([self.P[a][s,s2]*(self.R[a][s,s2]+gamma*V[s2]) for s2 in range(self.S)]) for a in range(self.A)])
            V = newV
            Q.append(sum([self.P[a][s,s2]*(self.R[a][s,s2]+gamma*V[s2]) for s2 in range(self.S)]))

        T = np.array(range(N))
        Q = np.array(Q)
        
    
        plt.plot(T,Q,label = 'Itération des valeurs')
        
    


    def policy_iteration(self,gamma:float, eps:float)->tuple:
        #Algotithme d'iteration sur politiques
       
        pi = np.array([0]*self.S)
        V=np.array([0]*self.S)

        while True:
            newpi = np.copy(pi)

            #prediction:
            
            while True:
                newV = np.copy(V)
                for s in range(self.S):
                    newV[s]= sum([self.P[pi[s]][s,s2]*(self.R[pi[s]][s,s2]+gamma*V[s2])  for s2 in range(self.S)])
                if np.linalg.norm(newV-V)<=eps: 
                    break
                V = newV
           
            #controle:
            
            for s in range(self.S):
                f = lambda a :  sum([self.P[a][s,s2]*(self.R[a][s,s2]+gamma*V[s2]) for s2 in range(self.S)])
                pi[s]=argmax(f,self.A)
            if np.linalg.norm(newpi-pi)==0:
                break
            pi = newpi
           
        return V , pi



def argmax(f, A:int)->int:
    """Détermine un élément de argmax f"""

    a = 0
    arg = f(a)
    for a2 in range(A):
        arg2 = f(a2)
        if arg2>arg:
            a = a2
            arg = arg2
    return a

def make_random_MDP(S:int,A:int,Rmax:float)-> MDP:
    """Genere un MDP aleatoire"""

    list_probas = []
    list_rewards = []
    for a in range (A):
        Pa = np.random.rand(S, S)
        Pa = Pa/Pa.sum(axis=1)[:,None]
        list_probas.append(Pa)
        Ra=Rmax*(np.random.rand(S,S))
        for s in range(S):
            for s2 in range(S):
                if random.random()<0.5:
                    Ra[s][s2]= -Ra[s][s2]
        list_rewards.append(Ra)
    P = Proba(list_probas)
    R = Reward(list_rewards)
    return MDP(S,A,P,R)
              


