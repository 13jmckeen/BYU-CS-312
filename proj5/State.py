class State:
    def __init__(self, name):
        self.matrix = [[]]  # the matrix we need to reduce or the one already reduced
        self.visited = []  # we store teh indecies of the cities
        self.lower_bound = 0  # the calculated matrix
        self.name = ""
        self.buildName(name)
        self.index = -1

    def show(self):
        print("State name: " + self.name)
        print("     Lower bound: %s" % self.lower_bound)
        print("     Visited: %s" % self.visited)
        print("     City index: %s" % self.index)
        print()

    def buildName(self, cityName):
        if (self.name == ""):
            self.name = "State(" + cityName + ")"
        else:
            self.name = self.name[:-1]  # removes the last character of string
            self.name = self.name + "-" + cityName + ")"

    def markCity(self, city):
        self.visited.append(city)

    def getVisited(self):
        return self.visited

    def getMatrix(self):
        return self.matrix

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.visited == other.visited and self.lower_bound == other.lower_bound and self.name == other.name and self.index == other.index

    def __lt__(self, other):  # for <
        if not isinstance(other, State):
            return False
        return self.lower_bound < other.lower_bound

    def __le__(self, other):  # for <=
        if not isinstance(other, State):
            return False
        return self.lower_bound <= other.lower_bound