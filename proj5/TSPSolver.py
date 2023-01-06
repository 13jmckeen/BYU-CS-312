#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
from State import *
import heapq as hq
import itertools
import itertools
import copy



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario
		self.bssf = None
		self.costOfBestTourFound = None  # Cost of best tour found (optimal)
		self.storedStates = 0  # Max # of stored states at a given time
		self.BSSFUpdates = 0  # Number of BSSF updates
		self.statesCreated = 0  # Total # of states created
		self.statesPruned = 0  # Total # of states pruned


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	# -------------------------------------------------------------------------------------------------------------------------


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		pass

	# -------------------------------------------------------------------------------------------------------------------------



	def printMatrix(self, matrix):
		print('\n'.join([' '.join(['{:5}'.format(item) for item in row]) for row in matrix]))
		print('\n')

	def printConstMatrix(self, matrix):
		print("ORIGINAL COST MATRIX:")
		cities = self._scenario.getCities()
		header = "  "
		for i in range(len(matrix)):
			str = cities[i]._name
			city = cities[i]
			for j in range(len(matrix[i])):
				if (i == 0):
					header = header + " {:5}".format(cities[j]._name)
				s = city.costTo(cities[j]).__str__()  # .__str__() is needed to conver int to string
				str = str + " {:5}".format(s)  # " {:5}".format(string) is to format and space evenly
			if (i == 0):
				print(header)
			print(str)
		print()

	def printTheMatrix(self, matrix):
		cities = self._scenario.getCities()
		header = "  "
		for i in range(len(matrix)):
			str = cities[i]._name
			city = cities[i]
			for j in range(len(matrix[i])):
				if (i == 0):
					header = header + " {:5}".format(cities[j]._name)
				s = matrix[i][j].__str__()  # .__str__() is needed to conver int to string
				str = str + " {:5}".format(s)  # " {:5}".format(string) is to format and space evenly
			if (i == 0):
				print(header)
			print(str)
		print()


	def printCities(self, cities, string):
		print("Order of " + string)
		route = ""
		for index, city in enumerate(cities):
			if (index == len(cities) - 1):
				route = route + city._name
			else:
				route = route + city._name + " --> "
		print(route)

	def reduceList(self, cities, keep):
		k = keep
		n = len(cities)
		for i in range(0, n - k):
			cities.pop()
		return cities



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound(self, time_allowance=60.0):
		results = {}
		heap = []
		hq.heapify(heap)

		cities = self._scenario.getCities()
		self.printCities(cities, "cities:")
		costMatrix = self.createMatrix()  # creates 2d matrix of nxn where n is the length of cities array

		result = self.reduceMatrix(0,costMatrix)  # caclculates the lower cost bound and does reeduction matrix operation (rows than columns)
		# O (n^2)


		# Initial state
		s = State(cities[0]._name)
		s.matrix = result[0]
		s.lower_bound = result[1]
		s.markCity(cities[0]._index)
		s.index = cities[0]._index

		# Key of priority queue is an integer. We conver float using int().
		# We divide lower bound by the number of cities this will force the algorithm go down instead of side.
		# Key is lower cost bound divided by the depth.
		hq.heappush(heap, (int(result[1] / len(s.getVisited())), s))

		# We use a random algorithm to find a random initial BSSF that we will initially use.
		self.bssf = self.defaultRandomTour()["cost"]

		# We convert the array of Cities objects into an array of indices since the number of cities won’t change
		listInd = self.getCityIndexList(cities)
		cost = None
		tour = []
		foundTour = False

		# s*L = n! n^3 + log(max size of queue) = max size of queue = n! =>  O(n!n^3)
		# space is max size of queue (n!) n!n^2(space of matrix)
		start_time = time.time()
		while len(heap) != 0 and time.time() - start_time < time_allowance:
			wholeState = hq.heappop(heap) #We convert the first city in queue which is State 1
			state = wholeState[1]
			cost = state.lower_bound
			if cost > self.bssf:
				self.statesPruned += 1
				continue
			else:
				notVisited = self.getCitiesNotVisited(state.getVisited(), listInd)
				# We get the number of cities that the state has not visited O(n)

				if (len(notVisited) == 0):
					if (self.bssf > cost):
						lastCity = self.getCitites(state.getVisited())[-1]
						firstCity = self.getCitites(state.getVisited())[0]
						path = lastCity.costTo(firstCity)
						if (path != float("inf")): # it is a solution if there is a path between last and first cities
							foundTour = True
							self.BSSFUpdates += 1  # Number of BSSF updates
							self.bssf = cost # We update BSSF
							tour = self.getCitites(state.getVisited())
				else:
					self.modifyHeapQueue(wholeState, notVisited, heap)

		self.printCities(tour, "tour:")
		bssf = TSPSolution(tour) # creation of object containing our solution
		end_time = time.time()  # optimal

		if (len(tour) != 0):
			results['cost'] = bssf.cost if foundTour else self.bssf  # bssf.cost ???
			results['time'] = end_time - start_time
			results['count'] = self.BSSFUpdates  # number of intermediate solutions considered ???
			results['soln'] = bssf
			results['max'] = self.storedStates
			results['total'] = self.statesCreated
			results['pruned'] = self.statesPruned  # Total # of states pruned
			return results

		else:
			results['cost'] = self.bssf
			results['time'] = end_time - start_time
			results['count'] = self.BSSFUpdates  # number of intermediate solutions considered ???
			results['soln'] = bssf  # should we return the alretative ???
			results['max'] = self.storedStates
			results['total'] = self.statesCreated
			results['pruned'] = self.statesPruned  # Total # of states pruned
			return results

	def createMatrix(self):
		cities = self._scenario.getCities()
		rows = len(cities)
		cols = len(cities)

		costMatrix = [[]]
		costMatrix = [[0 for i in range(rows)] for j in range(cols)]

		for i in range(len(costMatrix)):
			city = cities[i]  # city A
			for j in range(len(costMatrix[i])):  # cities A B C D E F
				costMatrix[i][j] = city.costTo(
					cities[j])  # cost from(A) --> to(A), cost from(A) --> to(B), A --> C, A --> D, ...
		# self.printConstMatrix(costMatrix)

		return costMatrix

	def reduceMatrix(self, lowerBound, matrix):
		cities = self._scenario.getCities()
		bound_cost = lowerBound
		for i in range(len(matrix)):
			min_val = min(matrix[i])

			if (min_val == 0 or min_val == float("inf")):
				continue
			else:
				bound_cost += min_val
				for j in range(len(cities)):
					cell_val = matrix[i][j]
					if (cell_val != float("inf")):
						matrix[i][j] = cell_val - min_val
		for j in range(len(cities)):
			min_val = min([row[j] for row in matrix])
			if (min_val == 0 or min_val == float("inf")):
				continue
			else:
				bound_cost += min_val
				for i in range(len(matrix)):
					cell_val = matrix[i][j]
					if (cell_val != float("inf")):
						matrix[i][j] = cell_val - min_val
		# self.printMatrix(matrix)

		return matrix, bound_cost

	def modifyHeapQueue(self, wholeState, needToVisit, heap):
		cities = self._scenario.getCities()
		parentState = wholeState[1]
		for index in needToVisit:
			originalCost = copy.deepcopy(parentState.lower_bound)
			childMatrix = copy.deepcopy(parentState.getMatrix())
			if (childMatrix[parentState.index][index] == float("inf")):
				self.statesPruned += 1  # Pruned states not added to the queue or not counted becase state not creaated
				continue
			else:
				originalCost = originalCost + childMatrix[parentState.index][index]  # cost of path
				childMatrix = self.markRowsAndColsAndRefl(childMatrix, parentState.index, index)
				result = self.reduceMatrix(originalCost, childMatrix)  # created
				self.statesCreated += 1

				if (result[1] > self.bssf):
					self.statesPruned += 1
					continue
				else:
					s = State(cities[parentState.index]._name)
					s.matrix = result[0]
					s.lower_bound = result[1]
					s.buildName(cities[index]._name)
					s.visited = copy.deepcopy(parentState.visited)
					s.markCity(cities[index]._index)
					s.index = index
					# s.show()
					hq.heappush(heap, (
					int(result[1] / len(s.getVisited())), s))  # we add the state with the lower bound cost to the heap
					if (len(heap) > self.storedStates):
						self.storedStates = len(heap)

	def enumerate(sequence, start=0):
		n = start
		for elem in sequence:
			yield n, elem
			n += 1

	def getCitites(self, indices):
		tour = []
		cities = self._scenario.getCities()
		for i in indices:
			tour.append(cities[i])
		return tour

	def getCityIndexList(self, cities):
		indecies = []
		for city in cities:
			indecies.append(city._index)

		return indecies

	def getCitiesNotVisited(self, visited, cities):
		set_difference = set(visited).symmetric_difference(set(cities))
		list_difference = list(set_difference)
		return list_difference

	def markRowsAndColsAndRefl(self, matrix, row, col):
		matrix[col][row] = float("inf")
		# matrix[:,col] = float("inf") # matrix[:,col] means select rows from column col
		for i in range(len(matrix)):
			matrix[i][col] = float("inf")

		for j in range(len(matrix)):
			matrix[row][j] = float("inf")

		# self.printTheMatrix(matrix)
		return matrix


	# Get random tour; save value as BEST CASE, save the LIST
	# Make 2D array, input going & coming values (instantiate all values to infinity)
	# Get BEST LEAVING VALUE for each node (subtract from all leaving spots in matrix)
	# Get BEST ENTERING VALUE for each node (subtract from all entering spots in matrix)
	# This value is BEST CASE SCENARIO!
	# Pick start node
	# RECURSIVE FUNCTION: send a node & a copy of your matrix, and a copy of list of nodes visited in order
	# If no value leaving, but first node in list is same as last & list.length is same size as cities, return the list (value is now random tour value)
	# If no value leaving, return null
	# From Node, for loop through all values LEAVING your node (cross off LEAVING your node in matrix)
	#												(& cross off visiting the new node in matrix)
	# if any value is greater than the random tour? return null



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass


		



