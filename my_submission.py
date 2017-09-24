'''
2017 IFN680 Assignment   DUE: 25 September 2017

Student Name:     Helen Jeffrey
Student Number:   n9416528
Student eMail:    h.jeffrey@connect.qut.edu.au

'''

import numpy as np
import matplotlib.pyplot as plt

import pattern_utils
import population_search

import timeit

#------------------------------------------------------------------------------

class PatternPosePopulation(population_search.Population):
    '''
    
    '''
    def __init__(self, W, pat):
        '''
        Constructor. Simply pass the initial population to the parent
        class constructor.
        @param
          W : initial population
        '''
        self.pat = pat
        super().__init__(W)
    
    def evaluate(self):
        '''
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w 
            self.best_cost 
        @return 
           best cost of this generation            
        '''         
        
#        1. clip the particles so that they remain inside of the landscape boundaries.
        height, width = self.distance_image.shape[:2]
        np.clip(self.W[:, 0], 0, width-1, self.W[:,0]) # X
        np.clip(self.W[:, 1], 0, height-1, self.W[:,1]) # Y             
        
#        2. Calculate the cost of each particle        
        self.C = self.distance_image[self.W[:,1].astype(int), self.W[:,0].astype(int)]        
        
#        3. Evaluate cost of particle (Triangle)         
        evaluateResults = np.zeros(self.n)
        
        for i in range(self.n):            
            results = self.pat.evaluate(self.distance_image, self.W[i])            
            evaluateResults[i] = np.array(results[0])
            
        self.C = np.array(evaluateResults)                        
        
        # Determine which particle has the lowest cost
        i_min = self.C.argmin()
        
        # Get the cost of the minimum particle
        cost_min = self.C[i_min]        
        
        # If it is the best particle found so far, save it
        if cost_min < self.best_cost:
            self.best_w = self.W[i_min].copy()
            self.best_cost = cost_min

        # Return the minimum cost found
        return cost_min
    

    def mutate(self):
        '''
        Mutate each individual.
        The x and y coords should be mutated by adding with equal probability 
        -1, 0 or +1. That is, with probability 1/3 x is unchanged, with probability
        1/3 it is decremented by 1 and with the same probability it is 
        incremented by 1.
        The angle should be mutated by adding the equivalent of 1 degree in radians.
        The mutation for the scale coefficient is the same as for the x and y coords.
        @post:
          self.W has been mutated.
        '''
        
        assert self.W.shape==(self.n, 4)                       

#        1. Mutate the values: x, y, Scale - Add with equal probability -1,0 or +1 to the points of self.W
        mutations = np.random.choice([-1,0,1], 4*self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,4)          

#        2. Mutate theta (the angle) - Add the equivalent of 1 degree in radians.
        mutations[:,2] = np.random.choice([-0.0174533,0,0.0174533], self.n, replace=True, p = [1/3,1/3,1/3])        
        
#        3. Add Mutations to original pose
        self.W = self.W + mutations           
        
    def set_distance_image(self, distance_image):
        self.distance_image = distance_image

#------------------------------------------------------------------------------        

def initial_population(region, scale = 10, pop_size=20):
    '''
    initial population: exploit info from region   
    @param
        region : 
        scale : 
        pop_size : 
    
    '''         
    
    rmx, rMx, rmy, rMy = region
    W = np.concatenate( (
                 np.random.uniform(low=rmx,high=rMx, size=(pop_size,1)) ,
                 np.random.uniform(low=rmy,high=rMy, size=(pop_size,1)) ,
                 np.random.uniform(low=-np.pi,high=np.pi, size=(pop_size,1)) ,
                 np.ones((pop_size,1))*scale
                        ), axis=1)    
    return W

#------------------------------------------------------------------------------        
def test_particle_filter_search(population, genererations):
    '''
    Run the particle filter search on test image 1 or image 2 of the pattern_utils module
    @param
        population : population of shapes to be created
        genererations : The number of generations that the experiment will run for
    
    '''

    if True:
#         use image 1
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(True)
        ipat = 2 # index of the pattern to target
    else:
#        use image 2
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(True)
        ipat = 0 # index of the pattern to target
        
#     Narrow the initial search region
    pat = pat_list[ipat] #  (100,30, np.pi/3,40),
    
    xs, ys = pose_list[ipat][:2]
    region = (xs-20, xs+20, ys-20, ys+20)
    scale = pose_list[ipat][3]
        
    W = initial_population(region, scale , population)
    
    pop = PatternPosePopulation(W, pat)
    pop.set_distance_image(imd)
    
    pop.temperature = 5
    
#    Time how long it takes for particle_filter_search() to execute.
    start_time = timeit.default_timer()
    
    Lw, Lc = pop.particle_filter_search(genererations,log=True)
    
    executionTime = timeit.default_timer() - start_time    
    
    plotTitle = 'Population Size: ', population, ', Generations: ', genererations
    
    plt.plot(Lc)
    plt.title(plotTitle)
    plt.show()
        

#   TODO Occasional error raised when experiment ...
#   AttributeError: 'PatternPosePopulation' object has no attribute 'best_w'

    try: 
        pattern_utils.display_solution(pat_list, 
                      pose_list, 
                      pat,
                      pop.best_w)
    except:
        pass
    
#    TODO save solution to file use [start_time]_Solution.png name
                      
    pattern_utils.replay_search(pat_list, 
                      pose_list, 
                      pat,
                      Lw)

    report_particle_filter_search(start_time, 
                                  executionTime, 
                                  population, 
                                  genererations, 
                                  Lc) 
    
    
#------------------------------------------------------------------------------  

def report_particle_filter_search(startTime, executionTime, population, genererations, Lc):
    '''
    Create a report for the execution of the particle_filter_search
    Report captures (in csv format) :
    - test name,
    - timestamp,
    - execution time,  
    - population, 
    - genererations,  
    - best_cost for each generation    
        
    '''
#    Prepare file output       
    newRow = 'p' + str(population) + ' x ' + 'g' + str(genererations) + ',' + str(executionTime) + ',' + str(population) + ',' + str(genererations) + ', ' + str(Lc).strip('[]')
   
    fName = ('log_particle_filter_search_.csv')
    with open(fName,'a') as f:
        f.write(newRow + '\n')
        f.close()    
                    
#------------------------------------------------------------------------------        
if __name__=='__main__':    
    '''
    This script calls the test_particle_filter_search(population, genererations)
    computationalBudget, test population and testRepeat are specified     
    
    '''     
    computationalBudget = 1000     
    testRepeat = 1  
    testPopulations = (50, )     
    
#    testPopulations = (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500, 1000,)    
#    testPopulations = (1000, )#500, 400, 200, 100, 75, 50, 25, 20, 10, 5)
        
#    testRepeat - specifies the number of times a test is to be repeated
    
    
    for population in testPopulations:
        genererations = int(computationalBudget / population)
#        print('1. ', population, genererations)  
        
        t = 1
        while t <= testRepeat:    
            test_particle_filter_search(population, genererations)  
            t = t + 1          
            
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
#        
#    def test_2():
#        '''
#        Run the particle filter search on test image 2 of the pattern_utils module
#        
#        '''
#        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(False)
#        pat = pat_list[0]
#        
#        #region = (100,150,40,60)
#        xs, ys = pose_list[0][:2]
#        region = (xs-20, xs+20, ys-20, ys+20)
#        
#        W = initial_population_2(region, scale = 30, pop_size=40)
#        
#        pop = PatternPosePopulation(W, pat)
#        pop.set_distance_image(imd)
#        
#        pop.temperature = 5
#        
#        Lw, Lc = pop.particle_filter_search(40,log=True)
#        
#        plt.plot(Lc)
#        plt.title('Cost vs generation index')
#        plt.show()
#        
#        print(pop.best_w)
#        print(pop.best_cost)
#        
#        
#        
#        pattern_utils.display_solution(pat_list, 
#                          pose_list, 
#                          pat,
#                          pop.best_w)
#                          
#        pattern_utils.replay_search(pat_list, 
#                          pose_list, 
#                          pat,
#                          Lw)
#    
#    #------------------------------------------------------------------------------        
#        
    