# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations > 0:
            tempValues = self.values.copy()
            allStates = self.mdp.getStates()
            for state in allStates:
                allActionsForState = self.mdp.getPossibleActions(state)
                chanceNodeValues = []
                for action in allActionsForState:
                    finalStates = self.mdp.getTransitionStatesAndProbs(state, action)
                    weightedAvg = 0
                    for finalState in finalStates:
                        nextState = finalState[0]
                        probability = finalState[1]
                        Reward=self.mdp.getReward(state,action,nextState)
                        weightedAvg += (probability * (Reward + (self.discount * tempValues[nextState])))
                    chanceNodeValues.append(weightedAvg)
                if len(chanceNodeValues) != 0:
                    self.values[state] = max(chanceNodeValues)
            self.iterations -= 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        finalStates = self.mdp.getTransitionStatesAndProbs(state, action)
        weightedAvg = 0
        for finalState in finalStates:
            nextState = finalState[0]
            probability = finalState[1]
            Reward=self.mdp.getReward(state, action, nextState)
            weightedAvg += (probability * (Reward + (self.discount * self.values[nextState])))

        return weightedAvg

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        Terminal=self.mdp.isTerminal(state)
        if Terminal:
            return None
        allActionsForState = self.mdp.getPossibleActions(state)
        finalAction = ""
        maxSum = float("-inf")
        for action in allActionsForState:
            weightedAvg = self.computeQValueFromValues(state, action)
            if (maxSum == 0.0 and action == "") or weightedAvg >= maxSum:
                finalAction = action
                maxSum = weightedAvg

        return finalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        values = util.Counter()
        states = self.mdp.getStates()

        for i in range(self.iterations):
            currentState = states[i%len(states)]
            values = self.values.copy()
            possibleVals = []
            Terminal=self.mdp.isTerminal(currentState)

            if Terminal:
                self.values[currentState] = 0
   
            else:
                for action in self.mdp.getPossibleActions(currentState):
                    tempValue = 0
                                        
                    for t in self.mdp.getTransitionStatesAndProbs(currentState, action):
                        Reward=self.mdp.getReward(currentState, action, t[0])
                        tempValue += t[1]*(Reward+ self.discount * values[t[0]])
                    possibleVals.append(tempValue)

                self.values[currentState] = max(possibleVals)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #values = util.Counter()
        #predSets =[]
        #initialize an empty priority queue
        fringe = util.PriorityQueue()
        states=self.mdp.getStates()
        predecessors={}

        #For each state s do
        for tstate in self.mdp.getStates():
            previous=set()
            #find predecessors of state
            #predecessors = set()
            for state in states:
                actions=self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions=self.mdp.getTransitionStatesAndProbs(state,action)
                    for next,probability in transitions:
                        if probability!=0:
                            if tstate==next:
                                previous.add(state)

            predecessors[tstate]=previous
        for state in states:
            Terminal=self.mdp.isTerminal(state)
            if Terminal==False:
                current=self.values[state]
                qValues=[]
                actions=self.mdp.getPossibleActions(state)
                for action in actions:
                    tempvalue=self.computeQValueFromValues(state,action)
                    qValues=qValues+[tempvalue]
                maxQvalue=max(qValues)
                diff=current-maxQvalue
                if diff>0:
                    diff=diff*-1
                fringe.push(state,diff)

        for i in range(0,self.iterations):
            if fringe.isEmpty():
                break
            s=fringe.pop()
            Terminal=self.mdp.isTerminal(s)

            if not Terminal:
                values=[]
                for action in self.mdp.getPossibleActions(s):
                    value=0
                    for next,prob in self.mdp.getTransitionStatesAndProbs(s,action):
                        Reward=self.mdp.getReward(s,action,next)
                        value=value+(prob*(Reward+(self.discount*self.values[next])))
                    values.append(value)
                self.values[s]=max(values)

            for previous in predecessors[s]:
                current=self.values[previous]
                qValues=[]
                for action in self.mdp.getPossibleActions(previous):
                    qValues+=[self.computeQValueFromValues(previous,action)]
                maxQ=max(qValues)
                diff=abs((current-maxQ))
                if (diff>self.theta):
                    fringe.update(previous,-diff)
