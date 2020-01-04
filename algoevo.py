import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List
from itertools import permutations
import copy
import pickle


def loadObj(fileDir):
    with open(fileDir, 'rb') as f:
        return pickle.load(f)


def create_map(tspfile, sep=" ", index_col=0, header=None, names=None, skiprows=7, skipfooter=1):
    """Creates a pandas dataframe with index name equal to the number of that point and X and Y columns from tsp file"""
    if names is None:
        names = ["Y", "X"]
    return pd.read_csv(tspfile, sep=sep, index_col=index_col, header=header, names=names, skiprows=skiprows,
                       skipfooter=skipfooter, engine="python")


def calc_distance(path: List[float], sqrt=False) -> float:
    """Calculates distance/fitness"""
    distance = 0.0
    for i in range(0, len(path)):
        origin = path[i]
        if i + 1 < len(path):
            destination = path[i + 1]
        else:
            destination = path[0]  # If it is the last, distance to the beginning
        distance += origin.distance(destination, sqrt=sqrt)
    return distance


class City:
    """City object"""

    def __init__(self, name, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y

    def distance(self, city, sqrt=False):
        """Calculates distance to another city with or without sqrt"""
        disX = (self.x - city.x) ** 2
        disY = (self.y - city.y) ** 2
        if not sqrt:
            dist = disX + disY
        else:
            dist = np.sqrt(disX + disY)
        return dist

    def __repr__(self):
        return self.name


class Map:
    """Map object"""

    def __init__(self, tspfile):
        self.map = create_map(tspfile=tspfile)
        self.cities = self.create_cities()

    def plot_map(self):
        plt.figure(figsize=[8, 10])
        plt.scatter(self.map["X"], self.map["Y"])
        plt.show()

    def create_cities(self):
        """Creates a list of the cities objects"""
        return list(self.map.apply(lambda x: City(str(x.name), x["X"], x["Y"]), axis=1))


class Path:
    """Single Path Object"""

    def __init__(self, path: List):

        assert len(path) == len(set(path))  # Checking if there is no duplicate nodes

        self.path = path
        self.calculate_fitness()

    def calculate_fitness(self, sqrt=False):
        self.fitness = calc_distance(self.path, sqrt)
        return self.fitness

    def mutate_swap(self, rate=0.01, inPlace=True):
        """
        Selects two cities at  random  and  swap  their  positions in the path. This is repeated for a number of times dictated by the rate.
            
            rate: Float controling the number of mutations to apply
        """

        tempPath = self.path.copy()

        for _ in range(int(len(tempPath) * rate)):
            a = random.randint(0, len(tempPath) - 1)
            b = random.randint(0, len(tempPath) - 1)

            tempPath[a], tempPath[b] = tempPath[b], tempPath[a]

        if inPlace:
            self.path = tempPath
        else:
            return Path(tempPath)

    def mutate_greedy_swap(self, rate=0.01, attempts=10, inPlace=True):
        """   
        Finds the best two cities to swap after making a number of attempts. This is repeated for a number of times dictated by the rate. 
        
            rate: Float controling the number of mutations to apply
            attempts: Int controling number of attempts at each mutation.
        """

        tempPath = Path(self.path.copy())

        for mutation in range(int(len(tempPath.path) * rate)):
            currBest_a = 0
            currBest_b = 0
            currBest_score = tempPath.fitness

            for attempt in range(attempts):
                a = random.randint(0, len(tempPath.path) - 1)
                b = random.randint(0, len(tempPath.path) - 1)

                tryPath = copy.deepcopy(tempPath.path)
                tryPath[a], tryPath[b] = tempPath.path[b], tempPath.path[a]

                fitness = calc_distance(tryPath, sqrt=False)
                if fitness < currBest_score:
                    currBest_a = a
                    currBest_b = b
                    currBest_score = fitness

            tempPath.path[currBest_a], tempPath.path[currBest_b] = tempPath.path[currBest_b], tempPath.path[currBest_a]

        if inPlace:
            self.path = tempPath.path
        else:
            return tempPath

    def mutate_scramble(self, size=10, inPlace=True):
        """
        Selects a subset of the total path at random and then randomly rearranges it.
        
            size: Int controling the size of the sub-path to be rearranged.
        """

        if size > len(self.path) - 1: size = int(len(self.path) / 4)

        assert size > 1

        start = random.randint(0, len(self.path) - size)
        stop = start + size

        copy = self.path[start:stop]
        random.shuffle(copy)

        if inPlace:
            self.path[start:stop] = copy
        else:
            newPath = self.path.copy()
            newPath[start:stop] = copy
            return Path(newPath)

    def mutate_greedy_scramble(self, size=4, inPlace=True):
        """
        Selects a subset of the total path at random and then finds its optimal configuration.
        
            size: Int controling the size of the sub-path to be rearranged. Limited to be under 7 for performance
            considerations.
        """

        assert size < 10 and size > 1

        start = random.randint(0, len(self.path) - size)
        stop = start + size

        perms = list(set(permutations(self.path[start:stop])))  # enumerate all possible sub-paths

        fitnesses = [calc_distance(path) for path in perms]  # calc fitness of sub-paths

        if inPlace:
            self.path[start:stop] = perms[fitnesses.index(max(fitnesses))]  # replace with best sub-path
        else:
            newPath = self.path.copy()
            newPath[start:stop] = perms[fitnesses.index(max(fitnesses))]
            return Path(newPath)

    def crossover(self, mate):
        """
        Applies order crossover (OX) to a couple of paths and create offspring. Selects a sub-path of the first
        parent (self), creates a child with such sub-path at the same location and finally fills gaps with the second
        path (mate).
        
            mate: Path to be combined with self.
        """

        childPath_a = [-1 for x in range(len(self.path))]
        childPath_b = [-1 for x in range(len(self.path))]

        size = random.randint(int(len(self.path) * 0.2), int(len(self.path) * 0.8))

        start = random.randint(0, len(self.path) - size)
        stop = start + size

        splice = list(range(stop, len(self.path)))
        splice.extend(range(0, start))

        merge = list(range(stop, len(self.path)))
        merge.extend(range(0, stop))

        childPath_a[start:stop] = self.path[start:stop]  # Insert self sub-path into new Path
        childPath_b[start:stop] = mate.path[start:stop]  # Insert other parent sub-path into new Path

        # Loop through missing spots in new paths, fill in the order of parent
        for i in splice:

            for j in merge:
                if mate[j] not in childPath_a: break
            childPath_a[i] = mate[j]

            for j in merge:
                if self.path[j] not in childPath_b: break
            childPath_b[i] = self.path[j]

        return Path(childPath_a), Path(childPath_b)

    def __len__(self):
        return len(self.path)

    def __repr__(self):
        return str(self.path)

    def __getitem__(self, x):
        return self.path[x]

    def __iter__(self):
        return iter(self.path)

    """
    Changing comparison operations to be able to compare fitness by directly comparing paths and using sort methods
    """

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __ne__(self, other):
        return self.fitness != other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness


class Population:
    """Population object"""

    def __init__(self):
        self.population = []
        self.fitness = []

    def calculate_fitness(self, sqrt=False):
        """Calculates the distance for every path in population"""
        self.fitness = []
        for p in self.population:
            self.fitness.append(p.calculate_fitness(sqrt))

    def random_initial(self, size: int, map):
        print("Initializing random", size)
        for i in range(0, size):
            self.population.append(Path(random.sample(map.cities, len(map.cities))))

    def scramble_initial(self, size: int, map):
        print("Initializing greedy operators", size)
        for i in range(0, size):
            self.population.append(Path(random.sample(map.cities, len(map.cities))))
            self.population[i].mutate_greedy_scramble(size=8)
            self.population[i].mutate_greedy_swap(rate=0.01, attempts=1000, inPlace=False)

    def greedy_initial(self, size: int, map):
        """Selecting the closer one from two possible destinations"""
        n_cities = len(map.cities)
        print("Initializing simple greedy", size)
        for i in range(0, size):
            path = []
            to_test = map.cities.copy()
            initial = random.choice(to_test)
            path.append(initial)
            to_test.remove(initial)
            previous = initial
            while to_test:
                op_a = random.choice(to_test)
                op_b = random.choice(to_test)
                dist_a = previous.distance(op_a)
                dist_b = previous.distance(op_b)
                if dist_a < dist_b:
                    path.append(op_a)
                    to_test.remove(op_a)
                    previous = op_a
                else:
                    path.append(op_b)
                    to_test.remove(op_b)
                    previous = op_b
            assert len(path) == n_cities  # Check if all cities were placed
            self.population.append(Path(path))

    def currentBest_initial(self, size: int, recordFiles):
        print("Initializing with known records from previous runs", size)

        records = [loadObj(fileName) for fileName in recordFiles]

        while len(self.population) < int(size / 2):
            parent_1, parent_2 = [records[i] for i in np.random.choice(len(records), 2, replace=False)]
            self.population.extend(parent_1.crossover(parent_2))

        while len(self.population) < size:
            i = np.random.choice(len(records), 1)[0]
            self.population.append(records[i].mutate_swap(rate=0.05, inPlace=False))

    # Make population subscriptable
    def __getitem__(self, x):
        return self.population[x]

    def __setitem__(self, idx, value):
        self.population[idx] = value

    # Make population iterable
    def __iter__(self):
        return iter(self.population)


def natural_selection(pop_size=1000, generations=1000, test='Tournament', allTimeBest=0, recordFiles=[]):
    try:
        qatarmap = Map("qa194.tsp")
        counter = 0  # generation counter
        notImprovingCounter = 0  # counter for local minimun

        newRecords = []  # list storing new records file names

        # Initializing population
        pop = Population()
        if len(recordFiles) == 0:
            pop.greedy_initial(int(pop_size / 2), map=qatarmap)
            # pop.scramble_initial(int(pop_size/4), map=qatarmap)
            pop.random_initial(int(pop_size / 2), map=qatarmap)
        else:
            pop.currentBest_initial(int(pop_size * 0.75), recordFiles)
            pop.random_initial(int(pop_size / 4), map=qatarmap)

        pop.calculate_fitness(sqrt=False)
        currBest = pop.fitness[0]
        if allTimeBest == 0:
            allTimeBest = currBest

        print("\n STARTING \n", len(pop.population), "individuals for ", generations, "generations\n")

        # Main loop
        while counter < generations:

            # updating fitness and sorting
            pop.calculate_fitness(sqrt=False)
            pop.population.sort()
            pop.fitness.sort()
            fitns = np.array(pop.fitness)

            # update printing
            if counter % 10 == 0 and counter > 1:
                print("\n\tGeneration", counter)
                print("\t\ttotal population", fitns.shape[0])
                print("\t\tstarting fitness params", round(np.mean(fitns), 2), "+/-", round(np.std(fitns), 2))
                print("\t\tcurr best Solution:    ", round(pop.fitness[0], 2))
                if notImprovingCounter > 0:
                    print("\t\t\t no improvement for:", notImprovingCounter)

            newPop = []

            # Check if stuck            
            oldBest = currBest
            currBest = pop.fitness[0]
            if currBest == oldBest:
                notImprovingCounter += 1
            else:
                notImprovingCounter = 0

                # Check if new record, if true save it
            if currBest < allTimeBest:
                print("\nSaving new record")
                allTimeBest = currBest

                fileName = "record_at_" + str(int(round(pop.fitness[0]))) + ".pkl"
                with open(fileName, 'wb') as f:
                    pickle.dump(pop.population[0], f, -1)
                newRecords.append(fileName)

            # Select top 5% to survive / elitism
            newPop.extend(pop[:int(pop_size * 0.05)])

            # Select top 5% to mutate / elitism
            for individuo in pop[:int(pop_size * 0.05)]:
                newPop.append(individuo.mutate_greedy_swap(inPlace=False))

            # If stuck for more than 10 generations start acting
            if notImprovingCounter > 20:

                # Over 20 and under 50 introduce randomness to elite
                if notImprovingCounter < 35:
                    print("\t\t\t over 20 generations without change, mutate elite...")
                    for i in range(len(newPop)):
                        newPop[i].mutate_swap(rate=0.01, inPlace=True)
                    for i in range(int(pop_size * 0.1)):
                        pop.population[i].mutate_swap(rate=0.01, inPlace=True)

                # Over 40 and under 50 introduce randomness to elite
                elif 40 < notImprovingCounter < 50:
                    print("\t\t\t over 40 generations without change, mutate elite...")
                    for i in range(len(newPop)):
                        newPop[i].mutate_swap(rate=0.02, inPlace=True)
                    for i in range(int(pop_size * 0.3)):
                        pop.population[i].mutate_swap(rate=0.02, inPlace=True)

                # Over 75 heavily mutate population and reset notImprovingCounter
                if notImprovingCounter > 75:
                    print("\t\t\t over 75 generations without change, heavily mutating elite...")

                    for individuo in newPop:
                        individuo.mutate_swap(rate=0.35, inPlace=True)
                        individuo.mutate_scramble(size=20, inPlace=True)

                    for i in np.random.choice(len(pop.population), int(pop_size * 0.05), p=fitns / np.sum(fitns)):
                        pop[i].mutate_swap(rate=0.25, inPlace=True)

                    for i in range(int(pop_size * 0.6)):
                        pop.population[i].mutate_swap(rate=0.25, inPlace=True)
                        pop.population[i].mutate_scramble(size=20, inPlace=True)

                    notImprovingCounter = 0

            # TOURNAMENT - Choose 4 random individuals and select the best one in each round (25% of final new
            # population) Since in each round we'll compare 4 individuals, the total number has to be divisible by 4.
            # In case the population size isn't a multiple of 4, the remaining individuals will be mutated and added
            # to the next genereation.
            if test == 'Tournament':

                # the TOURNAMENT itself
                winners = []
                notUsed = np.arange(len(pop.population))
                for _ in range(int(pop_size / 4)):
                    tournIndexs = np.random.choice(notUsed, 4, replace=False)
                    winnerIndex = np.argmin(fitns[tournIndexs])
                    notUsed = np.setxor1d(notUsed, tournIndexs)
                    winners.append(pop.population[tournIndexs[winnerIndex]])

                # add extra individuals that didn't participate in the tournament
                if len(notUsed) == 1:
                    newPop.append(pop.population[notUsed[0]].mutate_swap(rate=0.04, inPlace=False))
                elif len(notUsed) > 1:
                    newPop.extend([pop.population[i].mutate_swap(rate=0.04, inPlace=False) for i in notUsed])

                # 40% new individuals by crossing over from winners
                for _ in range(int(pop_size * 0.2)):
                    parent_1, parent_2 = [winners[i] for i in np.random.choice(len(winners), 2, replace=False)]
                    child_1, child_2 = parent_1.crossover(parent_2)

                    newPop.extend([child_1, child_2])

                # 25% new individuals by adding mutated winners
                newPop.extend([w.mutate_swap(inPlace=False) for w in winners])

                # 2% new individuals by scramble from winners
                for i in np.random.choice(len(winners), int(pop_size * 0.02)):
                    newPop.append(winners[i].mutate_scramble(inPlace=False))

                # 14% new individuals by greedy scramble from winners
                for i in np.random.choice(len(winners), int(pop_size * 0.14)):
                    newPop.append(winners[i].mutate_greedy_scramble(inPlace=False))

                # 4% new individuals by random mutation from winners
                for i in np.random.choice(len(winners), int(pop_size * 0.04)):
                    newPop.append(winners[i].mutate_swap(rate=0.05, inPlace=False))

                # ~5% new individuals by greedy mutation from winners
                for i in np.random.choice(len(winners), pop_size - len(newPop)):
                    newPop.append(winners[i].mutate_greedy_swap(inPlace=False))

            # ROULETTE-WHEEL
            else:

                # Roulette-wheel 20% parents to produce 40% of new pop
                for _ in range(int(pop_size * 0.20)):
                    probs = fitns / np.sum(fitns)

                    a = np.random.choice(len(pop.population), p=probs)
                    parent_a = pop[a]

                    probs[a] = 0  # Avoid drawing parent_a again
                    probs = probs / np.sum(probs)  # correct prob distribution

                    b = np.random.choice(len(pop.population), p=probs)
                    parent_b = pop[b]

                    newPop.extend(parent_a.crossover(parent_b))

                # Roulette-wheel 5% for random mutation
                for i in np.random.choice(len(pop.population), int(pop_size * 0.05), p=fitns / np.sum(fitns)):
                    newPop.append(pop[i].mutate_swap(inPlace=False))

                # Roulette-wheel 20% for greedy mutation
                for i in np.random.choice(len(pop.population), int(pop_size * 0.20), p=fitns / np.sum(fitns)):
                    newPop.append(pop[i].mutate_greedy_swap(inPlace=False))

                # Roulette-wheel 5% for scramble
                for i in np.random.choice(len(pop.population), int(pop_size * 0.05), p=fitns / np.sum(fitns)):
                    newPop.append(pop[i].mutate_scramble(inPlace=False))

                # Roulette-wheel ~25% for greedy scramble
                for i in np.random.choice(len(pop.population), pop_size - len(newPop), p=fitns / np.sum(fitns)):
                    newPop.append(pop[i].mutate_greedy_scramble(inPlace=False))

            pop.population = newPop
            counter += 1

        print("\nENDING")
        print("\tpopulation fitness params", round(np.mean(fitns), 2), "+/-", round(np.std(fitns), 2))
        print("\tBest Solution:")
        pop.calculate_fitness(sqrt=True)
        pop.population.sort()
        print("\t", pop.fitness[0])
        print("\n", pop[0])
        return newRecords

    except KeyboardInterrupt:
        print("\nENDING")
        print("\tpopulation fitness params", round(np.mean(fitns), 2), "+/-", round(np.std(fitns), 2))
        print("\tBest Solution:")
        pop.calculate_fitness(sqrt=True)
        pop.population.sort()
        print("\t", pop.fitness[0])
        print("\n", pop[0])
        exit()


if __name__ == '__main__' and False:
    print("\n\n")
    print("############################")
    print("# Initialization / sorting #")
    print("############################")

    qatarmap = Map("qa194.tsp")

    rec_1 = [182, 194, 190, 189, 192, 191, 188, 193, 185, 170, 162, 155, 135, 121, 108, 92, 81, 70, 60, 69, 24, 26, 14,
             13, 16, 36, 62, 71,
             78, 88, 95, 97, 105, 112, 120, 133, 143, 160, 166, 171, 180, 178, 181, 177, 184, 175, 174, 173, 157, 141,
             122, 103, 93, 83, 79,
             77, 68, 67, 58, 31, 12, 10, 5, 3, 2, 1, 6, 23, 76, 87, 102, 113, 126, 140, 149, 163, 169, 179, 183, 187,
             186, 172, 164, 138, 119,
             91, 72, 25, 11, 7, 17, 21, 33, 57, 64, 45, 37, 51, 43, 48, 53, 52, 46, 44, 35, 32, 42, 49, 50, 55, 54, 41,
             38, 34, 27, 22, 18, 28,
             29, 39, 40, 47, 61, 66, 84, 110, 123, 124, 128, 115, 107, 106, 118, 136, 151, 159, 165, 168, 167, 158, 148,
             129, 116, 96, 74, 75,
             80, 82, 89, 94, 104, 127, 132, 130, 111, 98, 86, 85, 65, 20, 63, 90, 101, 109, 114, 134, 137, 125, 99, 59,
             8, 4, 9, 15, 19, 30, 56,
             73, 100, 117, 131, 147, 152, 153, 154, 150, 144, 139, 146, 142, 145, 156, 161, 176]

    rec_1_Path = Path([qatarmap.cities[x - 1] for x in rec_1])

    rec_2 = [61, 58, 53, 54, 55, 50, 49, 35, 12, 10, 15, 19, 30, 46, 56, 48, 52, 41, 44, 42, 32, 38, 43, 47, 40, 34, 27,
             28, 57, 81, 78, 80,
             89, 99, 94, 104, 127, 132, 134, 145, 156, 163, 174, 177, 170, 166, 135, 121, 117, 107, 97, 88, 70, 45, 39,
             37, 29, 33, 60, 69,
             74, 72, 26, 17, 14, 11, 8, 4, 2, 1, 6, 16, 25, 75, 93, 122, 141, 153, 157, 154, 169, 176, 161, 149, 140,
             130, 98, 86, 90, 82,
             76, 87, 102, 109, 114, 119, 144, 150, 139, 138, 137, 142, 146, 164, 172, 179, 173, 165, 159, 152, 131, 136,
             129, 133, 143, 160,
             171, 185, 193, 181, 178, 180, 167, 162, 155, 148, 151, 158, 168, 184, 188, 189, 191, 192, 190, 194, 182,
             186, 187, 183, 175,
             147, 118, 106, 92, 84, 66, 31, 9, 5, 3, 7, 13, 62, 101, 111, 125, 126, 113, 103, 91, 71, 36, 20, 65, 85,
             63, 59, 23, 24, 21,
             18, 22, 51, 73, 100, 110, 112, 116, 120, 123, 124, 128, 115, 108, 105, 96, 95, 83, 79, 77, 64, 68, 67]

    rec_2_Path = Path([qatarmap.cities[x - 1] for x in rec_2])

    print("\n", "rec_1_Path")
    print("\nvalue", rec_1_Path.calculate_fitness(sqrt=False))
    print("\nvalue", rec_1_Path.calculate_fitness(sqrt=True))

    print("\n", "rec_2_Path")
    print("\nvalue", rec_2_Path.calculate_fitness(sqrt=False))
    print("\nvalue", rec_2_Path.calculate_fitness(sqrt=True))
    exit()
    # qatarmap.plot_map()

    print("\n")
    print("Random Initialization")
    pop = Population()
    pop.random_initial(100, map=qatarmap)
    print("Pop size:", len(pop.population))
    pop.calculate_fitness(sqrt=True)
    print("Mean fitness:", round(np.mean(pop.fitness), 2))

    print("\n")
    print("Greedy Initialization - 1 vs 1")
    pop2 = Population()
    pop2.greedy_initial(100, map=qatarmap)
    print("Pop size:", len(pop2.population))
    pop2.calculate_fitness(sqrt=True)
    print("Mean fitness:", round(np.mean(pop2.fitness), 2))

    print("\n")
    print("Sorting")
    pop2.population.sort()
    print("Pop size:", len(pop2.population))
    pop2.calculate_fitness(sqrt=True)
    print("Mean fitness:", round(np.mean(pop2.fitness), 2))

    print("\n\n")
    print("#########################")
    print("# Mutations / Crossover #")
    print("#########################")

    print("\nPerform a simple swap of Path at index 2")
    fitness = pop[2].calculate_fitness()
    pop[2].mutate_swap()
    print("Change in fitness:", round((fitness - pop[2].calculate_fitness()) / fitness * 100, 2), "%")

    print("\nPerform a greedy swap of Path at index 2")
    fitness = pop[2].calculate_fitness()
    pop[2].mutate_greedy_swap()
    print("Change in fitness:", round((fitness - pop[2].calculate_fitness()) / fitness * 100, 2), "%")

    print("\nPerform a simple scramble of Path at index 2")
    fitness = pop[2].calculate_fitness()
    pop[2].mutate_scramble()
    print("Change in fitness:", round((fitness - pop[2].calculate_fitness()) / fitness * 100, 2), "%")

    print("\nPerform a greedy scramble of Path at index 2")
    fitness = pop[2].calculate_fitness()
    pop[2].mutate_greedy_scramble()
    print("Change in fitness:", round((fitness - pop[2].calculate_fitness()) / fitness * 100, 2), "%")

    print(
        "\nPerform a crossover between Path at index 2 and Path at index 1, replacing the Path at index 0 and at index 10 with the offspring")
    fitness = pop[2].calculate_fitness()
    pop[0], pop[10] = pop[2].crossover(pop[1])
    print("Change in fitness in relation to Path 2:", round((fitness - pop[0].calculate_fitness()) / fitness * 100, 2),
          "%")
    print("Change in fitness in relation to Path 2:", round((fitness - pop[10].calculate_fitness()) / fitness * 100, 2),
          "%\n")

if __name__ == '__main__':
    # print('Roulette')
    # Roulette = natural_selection(1000, 5000, 'Roulette')

    # We first run in start-up mode, that is no previous records are supplied and the population is initialized part
    # randomly and part greedly. After some sucessfull runs, the best scores are used to initialize a new population
    # enabling better results. The scores are saved on a best only with a initial threshold of 2.8 M In the actual
    # run, this process was repeated twice, first using solutions ~16k and then with solutions ~10k. Tourment proved
    # superior to roulette wheel (the two last lines were written a posteriori)

    """ ATTENTION: THIS WILL CREATE MULTIPLE INTERMEDIATE FILES """

    print('Tournament')

    recordFiles = natural_selection(100, 5000, 'Tournament', allTimeBest=2800000)

    recor = loadObj(recordFiles[-1])
    bestLast = recor.calculate_fitness(sqrt=False)
    print("\nbestLast", bestLast, "\n")
    toUse = -20 if len(recordFiles) >= 20 else len(recordFiles) * -1
    recordFiles = natural_selection(100, 100, 'Tournament', allTimeBest=bestLast, recordFiles=recordFiles[toUse:])

    recor = loadObj(recordFiles[-1])
    bestLast = recor.calculate_fitness(sqrt=False)
    print("\nbestLast", bestLast, "\n")
    toUse = -20 if len(recordFiles) >= 20 else len(recordFiles) * -1
    recordFiles = natural_selection(200, 100, 'Tournament', allTimeBest=bestLast, recordFiles=recordFiles[toUse:])

    recor = loadObj(recordFiles[-1])
    bestLast = recor.calculate_fitness(sqrt=False)
    """                                                                                       
    recor = loadObj("record_at_758729.0.pkl")
    print(recor)
    print("len", len(recor))
    print(recor.calculate_fitness(sqrt=False))
    print(recor.calculate_fitness(sqrt=True))
    """
