import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import json 

# ----------------------------------- preprocess the dataset -------------------------------------
def preprocess(path, isdiscrete):    
    df = pd.read_csv(path)
    classes = pd.unique(df[df.columns[-1]])
    datasets = []
    target = df.columns[-1]
    for i in classes:
        clas = {}
        clas['name'] = i
        clas['data'] = df[df[target] == i]
        clas['data'] = clas['data'].drop(target,axis=1)
        clas['data'] = np.array(clas['data'])
        datasets.append(clas)
    
    domains = []
    for i in range(len(df.columns)):
        if isdiscrete[i] == 1:
            dom = list(pd.unique(df[df.columns[i]]) )
            domains.append(dom)
        else:
            domains.append([min(df[df.columns[i]]),max(df[df.columns[i]])])
    
    return (datasets, domains,df.columns)
        

# ------------------------------------ generate random genes ------------------------------------
def randomgenes(length, domains, isdiscrete):
    genes = []
    for i in range(length):
        gene = []

        #W
        gene.append(np.random.random())
        
        
        if isdiscrete[i] == 1:
            bits = []
            for x in range(len(domains[i])):
                r = np.random.randint(0,101)
                if r < 50:
                    bits.append(0)
                else:
                    bits.append(1)
            gene.append(bits)
            
            
            
        else:
            # continous attribute
            # lower
            lower = np.random.random()*(domains[i][1] - domains[i][0]) + domains[i][0]
            # upper
            upper = np.random.random()*(domains[i][1] - lower) + lower
            gene.append(lower)
            gene.append(upper)

        genes.append(gene)
    
    return genes
    



# ------------------------------ chromosome class ------------------------------------------
class chromosome:
    def __init__(self,length,classname,domains,isdiscrete,fitness, values=None):
        self.classname = classname
        self.fitness = fitness
        if values:
            self.genes = values.copy()
        else:
            self.genes = randomgenes(length,domains,isdiscrete)


# ---------------------------------- crossover -------------------------------------------
def crossover(chrom1, chrom2):
    
    # bit mask crossover
    bits = []
    for r in range(len(chrom1)):
        rand = np.random.randint(0,101)
        if rand < 65:
            bits.append(0)
        else:
            bits.append(1)
    offspring1 = chrom1.copy()
    offspring2 = chrom2.copy()
    
    for b in range(len(bits)):
        if bits[b] == 1:
            offspring1[b] = chrom2.copy()[b]
            offspring2[b] = chrom1.copy()[b]
    
    '''
    # one point crossover
    point1 = np.random.randint(0,len(chrom1)-1)
        
    offspring1 = chrom1.copy()
    offspring2 = chrom2.copy()
    for i in range(point1+1,len(chrom1)):
        offspring1[i] = chrom2[i]
        offspring2[i] = chrom1[i]
    '''
    return [offspring1, offspring2]


# ------------------------------------- mutation ------------------------------------------
def mutate(chrom, domains, isdiscrete):
    g = np.random.randint(0,len(chrom))
    sg = np.random.randint(0,3)
       
    # mutating weight
    if sg == 0:
        chrom[g][sg] = np.random.random()
        
    if sg == 1 and isdiscrete[g] == 0:
        chrom[g][sg] = np.random.random()*(chrom[g][2] - domains[g][0]) + domains[g][0]
    if sg == 2 and isdiscrete[g] == 0:
        chrom[g][sg] = np.random.random()*(domains[g][1] - chrom[g][1]) + chrom[g][1]
    if isdiscrete[g] == 1 and sg != 0:
        b = np.random.randint(0,len(chrom[g][1]))
        if chrom[g][1][b] == 0:
            chrom[g][1][b] = 1
            if np.array(chrom[g][1]).sum() == len(chrom[g][1]):
                chrom[g][0] = 0
        else:
            chrom[g][1][b] = 0
            
        




# ------------------------------- fitness function ------------------------------------
def fitness(chrom, limit, datasets, posindex, isdiscrete, domains):
    '''
        calculates accuracy*100 as fitness for each chromosome.
        INPUTS:
            - chrom: chromosome to calculate its fitness.
            
        OUTPUT:
            - Accuracy*100 on dataset
    '''
    flag = 0
    features = 0
    for k in range(len(chrom)):
        if chrom[k][0] >= limit:
            flag = 1
            features += 1
    if features == 0:
        return 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(datasets)):
        for j in range(len(datasets[i]['train'])):
            doesbelong = 1
            passed = 0
            for k in range(len(chrom)):
                if chrom[k][0] >= limit:
                    if isdiscrete[k] == 1:
                        if chrom[k][1][domains[k].index(datasets[i]['train'][j][k])] == 1:
                            passed += 1
                            
                    else:
                        if chrom[k][1] <= datasets[i]['train'][j][k] and chrom[k][2] >= datasets[i]['train'][j][k]:
                            passed += 1
                            
                            

            if passed == features and posindex == i:
                tp += 1
            elif passed == features and posindex != i:
                fp += 1
            elif passed != features and posindex == i:
                fn += 1
            elif passed != features and posindex != i:
                tn += 1
    return np.round(((tp+tn)/(tp+tn+fp+fn))*100, 2)
                            





# ----------------------------------- validate on unseen data ------------------------------------
def validate(chrom, limit, datasets, posindex, isdiscrete,domains):
    flag = 0
    features = 0
    for k in range(len(chrom)):
        if chrom[k][0] >= limit:
            flag = 1
            features += 1
    if flag == 0:
        return 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(datasets)):
        for j in range(len(datasets[i]['validate'])):
            doesbelong = 1
            passed = 0
            for k in range(len(chrom)):
                if chrom[k][0] >= limit:
                    if isdiscrete[k] == 1:
                        if chrom[k][1][domains[k].index(datasets[i]['train'][j][k])] == 1:
                            passed += 1
                            
                    else:
                        if chrom[k][1] <= datasets[i]['validate'][j][k] and chrom[k][2] >= datasets[i]['validate'][j][k]:
                            passed += 1
                            
                        
            if passed == features and posindex == i:
                tp += 1
            elif passed == features and posindex != i:
                fp += 1
            elif passed != features and posindex == i:
                fn += 1
            elif passed != features and posindex != i:
                tn += 1
    recall = 0
    if tp+fn != 0:
        recall = (tp / (tp+fn))
    precision = 0
    if tp+fp != 0:
        precision = (tp / (tp+fp))
    
    return np.round(((tp+tn)/(tp+tn+fp+fn))*100, 2)






# ---------------------------------- tournament selection ----------------------------
def tournament(pop, t):
    ind = np.random.randint(0,len(pop))
    best = pop[ind]
    for i in range(t):
        ind = np.random.randint(0,len(pop))
        if pop[ind].fitness > best.fitness:
            best = pop[ind]
    return best







# ------------------------------- print rule ---------------------------------------------
def printRule(chrom,att,isdiscrete,limit,classname,domains):
    rule = '----------------------\n'
    rule += 'rule for class '
    rule += classname
    rule += '\n'
    rule += 'fitness: '
    rule += str(chrom.fitness)
    rule += '\n'
    print()
    print('-----------------')
    print('rule for class',end=' ')
    print(classname)
    print('fitness:',end=' ')
    print(chrom.fitness)
    for r in range(len(chrom.genes)):
        if chrom.genes[r][0] >= limit:
            if isdiscrete[r] == 0:
                rule += str(np.round(chrom.genes[r][1],2))
                rule += ' '
                rule += '< '
                rule += att[r]
                rule += ' < '
                rule += str(np.round(chrom.genes[r][2],2))
                rule += ' and '
                print(np.round(chrom.genes[r][1],2), end = ' ')
                print('<',end = ' ')
                print(att[r],end=' ')
                print('<',end = ' ')
                print(np.round(chrom.genes[r][2],2),end = ' ')
            else:
                print(att[r],end = ' is in ')
                for x in range(len(chrom.genes[r][1])):
                    if chrom.genes[r][1][x] == 1:
                        print(domains[r][x],end=' , ')
                        
                    
            print('and', end = ' ')
    print()
    print('-----------------')
    rule = rule[:-4]
    rule += '-----------------\n'
    return rule



def rulesave(chrom,att,classname,limit,datasets,posindex,isdiscrete,domains):
    rule = '----------------------\n'
    rule += 'rule for class '
    rule += classname
    rule += '\n'
    rule += 'fitness: '
    rule += str(fitness(chrom, limit, datasets, posindex, isdiscrete,domains))
    rule += '\n'

    for r in range(len(chrom)):
        if chrom[r][0] >= limit:
            if isdiscrete[r] == 0:
                rule += str(np.round(chrom[r][1],2))
                rule += ' '
                rule += '< '
                rule += att[r]
                rule += ' < '
                rule += str(np.round(chrom[r][2],2))
                rule += ' and '
            else:
                rule += att[r]
                rule += ' is in ['
                for x in range(len(domains[r])):
                    if chrom[r][1][x] == 1:
                        rule += str(domains[r][x])
                        rule += ', '
                rule += ']'
                rule += ' and '
                
    rule = rule[:-4]
    rule += '\n-----------------\n'
    return rule
    




# ------------------------------------ MAIN ALGORITHM ----------------------------------
def GARule(datasets, domains, isdiscrete, mprob, popsize, maxgen, limit,attributes, t=3, fitnessthreshold = 90):
    '''
        INPUTS:
            - datasets: list of datasets for each calss - extracted by preprocess function
            - domains: domain of each feature - extracted by preprocess function
            - isdiscrete: a list of binary values showing type of each feature. 1 for discrete, 0 for continuous 
            - mprob: mutation probability
            - popsize: population size of genetic algorithm.
            - maxgen: maximum number of generations in genetic algorithm.
            - limit: a parameter which controls the presence of each feature in the rule. if for feature i, limit for i is equal or greater than limit parameter, then it will be included in the rule represented by the chromosome.
            - attributes: name of each feature in the dataset in order to convert the rules to human understandable language.
            - fitnessthreshold: if fitness of the rule is greater than this threshold, then add the rule to ruleset
            - t: t parameter of tornament selection function
        
        OUTPUT:
            - rule set extracted for each class
    '''
    
    report = []
    for z in range(len(datasets)): # one run for each class
        reportfit = 0
        reportval = 0
        print('run for class:', end=' ')
        print(datasets[z]['name'])
        rules = []
        model = []
        mean = []
        best = []
        worst = []
        pop = []
        validateMean = []
        validateBest = []
        totalvalidate = 0
        bestever = None
        worstfit = None
        totalfit = 0
        fits = []
        for i in range(popsize):
            indiv = chromosome(len(datasets[z]['train'][0]),datasets[z]['name'],domains,isdiscrete,0)
            indiv.fitness = fitness(indiv.genes, limit, datasets, z, isdiscrete,domains)
            fits.append(indiv.fitness)
            totalvalidate += validate(indiv.genes, limit, datasets, z, isdiscrete,domains)
            if fitness(indiv.genes, limit, datasets, z, isdiscrete, domains) >= fitnessthreshold:
                rep = True
                for r in rules:
                    if r == rulesave(indiv.genes, att, datasets[z]['name'], limit, datasets, z, isdiscrete,domains):
                        rep = False
                if rep:     
                    rules.append(rulesave(indiv.genes, att, datasets[z]['name'], limit, datasets, z, isdiscrete,domains))
                    model.append(list(indiv.genes))
                    reportfit += fitness(indiv.genes, limit, datasets, z, isdiscrete, domains)
                    reportval += validate(indiv.genes, limit, datasets, z, isdiscrete, domains)
           
            totalfit += indiv.fitness
            if bestever == None:
                bestever = indiv
            elif bestever.fitness < indiv.fitness:
                bestever = indiv
            if worstfit == None:
                worstfit = indiv.fitness
            elif worstfit > indiv.fitness:
                worstfit = indiv.fitness
            pop.append(indiv)
        
        # keep track of fitnesses
        mean.append(totalfit / len(pop))
        fits.sort()
        best.append(sum(fits[-3:])/3)
        worst.append(sum(fits[:3])/3)
        validateMean.append(totalvalidate / len(pop))
        validateBest.append(validate(bestever.genes, limit, datasets, z, isdiscrete,domains))
        
        
        
        for x in range(maxgen):    
            if x == np.floor(maxgen/2):
                mprob /= 2
            if x == np.floor(maxgen*0.7):
                mprob /= 2
            if x == np.floor(maxgen*0.85):
                mprob /= 2
            newgen = []
            newbest = bestever
            # selection
            for n in range(popsize):
                parent1 = tournament(pop, t)
                parent2 = tournament(pop, t)
                genes1,genes2 = crossover(parent1.genes, parent2.genes)
                offspring1 = chromosome(len(parent1.genes), datasets[z]['name'], domains, isdiscrete, fitness(genes1,limit,datasets,z,isdiscrete,domains),genes1)
                offspring2 = chromosome(len(parent1.genes), datasets[z]['name'], domains, isdiscrete, fitness(genes2,limit,datasets,z,isdiscrete,domains),genes2)
                if offspring1.fitness >= offspring2.fitness:    
                    newgen.append(offspring1)
                else:
                    newgen.append(offspring2)
                if newbest.fitness < offspring1.fitness:
                    newbest = offspring1
                if newbest.fitness < offspring2.fitness:
                    newbest = offspring2
            
            pop = []
            totalfit = 0
            totalvalidate = 0
            worstfit = None
            fits = []
            for n in range(popsize):
                w = newgen[n]
                mute = np.random.randint(0,101)
                if mute < mprob:
                    mutate(w.genes,domains,isdiscrete)
                    
                totalfit += w.fitness
                totalvalidate += validate(w.genes, limit, datasets, z, isdiscrete,domains)
                if worstfit == None:
                    worstfit = w.fitness
                elif worstfit > w.fitness:
                    worstfit = w.fitness
                w.fitness = fitness(w.genes,limit,datasets,z,isdiscrete,domains)
                if fitness(w.genes, limit, datasets, z, isdiscrete,domains) >= fitnessthreshold:
                    printRule(w, att, isdiscrete, limit, w.classname,domains)
                    rep = True
                    for r in rules:
                        if r == rulesave(w.genes, att, datasets[z]['name'], limit, datasets, z, isdiscrete,domains):
                            rep = False
                    if rep:     
                        rules.append(rulesave(w.genes, att, datasets[z]['name'], limit, datasets, z, isdiscrete,domains))
                        model.append(list(w.genes))
                        reportfit += fitness(w.genes, limit, datasets, z, isdiscrete, domains)
                        reportval += validate(w.genes, limit, datasets, z, isdiscrete, domains)
           
                fits.append(w.fitness)
                pop.append(w)
            
            pop.append(bestever)
            totalfit += bestever.fitness
            totalvalidate += validate(bestever.genes,limit,datasets,z,isdiscrete,domains)
            if newbest.fitness > bestever.fitness:
                bestever = newbest
                
            mean.append(totalfit / len(pop))
            fits.sort()
            best.append(sum(fits[-3:])/3)
            worst.append(sum(fits[:3])/3)
            validateMean.append(totalvalidate / len(pop))
            validateBest.append(validate(bestever.genes, limit, datasets, z, isdiscrete,domains))
            
        
        #visualize fitness
        plt.figure(figsize = (10,5),dpi=100)
        generations = np.arange(1,len(mean)+1)
        plt.plot(generations, best,'r')
        plt.plot(generations, mean,'b')
        plt.plot(generations, worst,'g')
        plt.title('fitness plot for class '+datasets[z]['name'])
        plt.show()
        
        plt.title('validation plot for class '+datasets[z]['name'])
        plt.plot(generations, best,'r')
        plt.plot(generations, validateMean,'k')
        plt.plot(generations,validateBest,'y')
        plt.show()        
        
        
        
        with open(datasets[z]['name']+' .txt','w') as f:
            for r in rules:
                f.write(r)


    
        
        
# ---------------------------------------- kfold -------------------------------------        
def kfold(datasets,k,rs):
    kf = KFold(n_splits=k,shuffle=True,random_state=rs)
    kdatasets = []
    for i in range(len(datasets)):
        kclass = {}
        kclass['name'] = datasets[i]['name']
        df = []
        for j, (train_index, test_index) in enumerate(kf.split(datasets[i]['data'])):
            train = []
            test = []
            for x in train_index:
                train.append(datasets[i]['data'][x])
            for y in test_index:
                test.append(datasets[i]['data'][y])
            df.append([np.array(train),np.array(test)])
        kclass['data'] = df
        kdatasets.append(kclass)
        
    kfolded = []    
    for kk in range(k):
        ds = []
        for i in range(len(datasets)):
            x = {}
            x['name'] = datasets[i]['name']
            x['train'] = kdatasets[i]['data'][kk][0]
            x['validate'] = kdatasets[i]['data'][kk][1]
            ds.append(x)
        kfolded.append(ds)   
    return kfolded




# --------------------------------------------- run kfold validation -------------------------------------------------
def kfoldvalidate(datasets, domains, isdiscrete, mprob, popsize, maxgen, limit,attributes, k, t=3,randomstate=101):
    kfolded = kfold(datasets,k,randomstate)
    results = []
    for d in kfolded:
        res = GARule(d,domains,isdiscrete,mprob,popsize,maxgen,limit,attributes,t)
        results.append(res)
    print(results)
        



# ------------------------------------------------- main ----------------------------------------------------------

# 1 if feature in position i is discrete, 0 if it's continuous
isdiscrete = [0,0,0,0,1]
# address of the dataset should be the first input of the below function.
ds, domains, att = preprocess('iris.csv', isdiscrete)
kfoldvalidate(ds, domains, isdiscrete=[0,0,0,0],mprob=90,popsize=100,maxgen=100,limit=0.3,attributes=att,k=10,t=3)
