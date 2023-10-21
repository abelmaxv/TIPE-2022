import numpy as np
from MDP_ens import *
import random
import matplotlib.pyplot as plt

class Agent():

    def __init__(self,s0:int, env:MDP)-> None:

        self.state = s0 
        self.env = env


    def SARSA (self, s0:int, sf:int, eps:float, gamma:float, N:int, lr = 0.01)->np.ndarray:
        """s0 est l'etat initial, sf est l'état final absorbant, epsilon induit la probabilité d'exploration
        (eps-greedy), N est le nombre d'episodes """
        

        Q = np.array([[0]*self.env.A]*self.env.S, dtype= np.float32)

        for i in range(N):
            self.state = s0
            if random.random() < eps:
                a = random.randint(0,self.env.A-1)
            else:
                a = 0
                Qa = Q[self.state][0]
                for b in range (self.env.A):
                    Q2 = Q[self.state][b]
                    if Q2>Qa:
                        a = b
                        Qa = Q2
            
            while self.state != sf: 
                s2 = etat_suivant(self.state,a,self.env)
                if random.random() < eps:
                    a2 = random.randint(0,self.env.A-1)
                else:
                    a2 = 0
                    Qa = Q[self.state][0]
                    for b in range (self.env.A):
                        Q2 = Q[self.state][b]
                        if Q2>Qa:
                            a2 = b
                            Qa=Q2
                r = self.env.R[a][self.state,s2]

                newQ = np.copy(Q)
                newQ[self.state,a]=Q[self.state,a]+lr*(r+gamma*Q[s2,a2] -Q[self.state,a])

                a = a2
                self.state = s2
                Q = newQ
                
            
        return Q

    
    def Q_learning (self, s0:int, sf:int, eps:float, gamma:float, N:int, lr= 0.01)->np.ndarray:
        """s0 est l'etat initial, sf est l'etat final absorbant, epsilon induit la probabilité d'exploration
        (eps-greedy), N est le nombre d'episodes """
        
        self.state = s0
        Q = np.array([[0]*self.env.A]*self.env.S, dtype=np.float32)

        # En commentaire: implementation softmax

        #A = np.array([[0]*self.env.A]*self.env.S) 
        #A stocke les taux d'apprentissage selon la methode uncertainty estimation
        for i in range(N):
            self.state = s0
            while self.state != sf:
                if random.random() < eps:
                    a = random.randint(0,self.env.A-1)
                else:
                    a = 0
                    Qa = Q[self.state][0]
                    for a2 in range (self.env.A):
                        Q2 = Q[self.state][a2]
                        if Q2>Qa:
                            a = a2
                            Qa=Q2
                #A[self.state,a]+=1
                s2 = etat_suivant(self.state,a,self.env)
                r= self.env.R[a][self.state,s2]
                newQ = np.copy(Q)            
                newQ[self.state,a]=Q[self.state,a]+(lr)*(r+gamma*max([Q[s2][a2] for a2 in range(self.env.A)])-Q[self.state][a])
                #newQ[self.state,a]=Q[self.state,a]+(1/A[self.state,a])*(r+gamma*max([Q[s2][a2] for a2 in range(self.env.A)])-Q[self.state][a])
                self.state = s2
            
                Q = newQ                
            
        return Q


    def graph_Q(self, s0:int, sf:int, eps:float, gamma:float,s2: int, a2:int, M:int, pas = 10)->None:
        """Trace Q(a,s) en fonction du nombre d'episodes"""

        #il faut que l'etat final soit associé à une recompense nulle    
        for a in range(self.env.A):
            for s in range(self.env.S):
                (self.env.R)[a][s,sf]=0

        Q_sa = []
        for i in range(0,M,pas):
            Q = self.Q_learning(s0,sf,eps,gamma,i)
            Q_sa.append(Q[s2,a2])
            print(i)

        T = np.array(range(0,M,pas))
        Q_sa = np.array(Q_sa)
        plt.plot(T,Q_sa, color ='red', label = 'Q-learning : epsilon = '+ str(eps))
    
    
    def graph_SARSA(self, s0:int, sf:int, eps:float, gamma:float,s2: int, a2:int, M:int, pas=10)->None:
        """Trace Q(a,s) en fonction du nombre d'episodes"""

        #il faut que l'état final soit associe à une recompense nulle    
        for a in range(self.env.A):
            for s in range(self.env.S):
                (self.env.R)[a][s,sf]=0

        Q_sa = []
        for i in range(0,M,pas):
            Q = self.SARSA(s0,sf,eps,gamma,i)
            Q_sa.append(Q[s2,a2])
            print(i)

        T = np.array(range(0,M,pas))
        Q_sa = np.array(Q_sa)
        plt.plot(T,Q_sa, color = 'blue', label = 'SARSA: epsilon = '+ str(eps))
        



def etat_suivant(s:int,a:int, env:MDP)->int:
    """Determine l'etat suivant lorsqu'on prend l'action a depuis l'etat s"""

    p = random.random()
    list_probas = [env.P[a][s,s2] for s2 in range(env.S)]
    for i in range(env.S):
        if p<list_probas[i]:
            return i
        else:
            p-=list_probas[i]


    
if __name__ == '__main__':

    S = 10
    A = 5
    Rmax = 100
    G = make_random_MDP(S,A,Rmax)
    agent = Agent(0,G)

    si = 0
    sf = 9
    eps = 0.3
    gamma = 0.9
    s = 2
    a = 3 
    iter  = 2000

 
    agent.graph_SARSA(si,sf,eps,gamma,s,a,iter)
    agent.graph_Q(si,sf,eps,gamma,s,a,iter )
    agent.env.graph_Q(s,a,gamma,iter)
    plt.xlabel('Itération')
    plt.ylabel('Q(2,3)')
    plt.legend()

    #agent.graph_Q(si,sf,0.1,gamma,s,a,iter)
    #agent.graph_Q(si,sf,0.5,gamma,s,a,iter)
    #agent.graph_Q(si,sf,0.9,gamma,s,a,iter)
    #plt.xlabel('Itération')
    #plt.ylabel('Q(2,3)')
    #plt.legend()
    #plt.title(label = 'Influence de epsilon')
    plt.show()
    


    




        
        
