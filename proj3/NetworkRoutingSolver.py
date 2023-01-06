#!/usr/bin/python3


from CS312Graph import *
import time


class NetworkRoutingSolver:
    def __init__( self ):
        pass

    def initializeNetwork( self, network ):
        assert( type(network) == CS312Graph )
        self.network = network

    def getShortestPath( self, destIndex ):
        self.dest = destIndex
        path_edges = []
        total_length = 0

        sourceNode = self.network.nodes[self.source]
        destinationNode = self.network.nodes[self.dest]
        previousNode = destinationNode
        while previousNode.prev is not None:
            edge = previousNode.prev
            path_edges.append( (edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge.length)) )
            total_length += edge.length
            previousNode = edge.src
        if previousNode != sourceNode:
            return {'cost':float("inf"), 'path':[]}
        return {'cost':total_length, 'path':path_edges}

    def computeShortestPaths( self, srcIndex, use_heap=False ):
        self.source = srcIndex
        t1 = time.time()

        if (use_heap == False):
            self.array_dijkstra(srcIndex)
        else:
            self.heap_dijkstra(srcIndex)

        t2 = time.time()
        return (t2-t1)

    # Dijkstra algorithm for an array
    def array_dijkstra(self, srcIndex):
        queue = self.make_array_queue()
        queue[srcIndex].dist = 0
        while len(queue) > 0: # Big O(n)
            node = self.array_delete_min(queue)
            for edge in node.neighbors:
                if edge.dest.dist > edge.src.dist + edge.length:
                    edge.dest.dist = edge.src.dist + edge.length
                    edge.dest.prev = edge


    # Dijkstra algorithm for a heap
    def heap_dijkstra(self, srcIndex):
        indiceNode = {}
        queue = []
        self.network.nodes[srcIndex].dist = 0
        self.make_heap_queue(queue, indiceNode)
        while len(queue) > 0: # Big O(n)
            node = self.heap_delete_min(queue, indiceNode)
            for edge in node.neighbors:
                if edge.dest.dist > edge.src.dist + edge.length:
                    edge.dest.dist = edge.src.dist + edge.length
                    edge.dest.prev = edge
                    self.decKey(queue, indiceNode, edge.dest)



    def make_array_queue(self): # Big O(n)
        return self.network.nodes.copy()

    def make_heap_queue(self, queue, node_indices): # Big O(n)
        for node in self.network.nodes:
            queue.append(node)
            self.bubbleUp(queue, len(queue)-1, node_indices)

    def heap_delete_min(self, queue, node_indices): # Big O(n)
        if len(queue) > 1:
            min_node = queue[0]
            queue[0] = queue.pop()
        else:
            return queue.pop()
        self.siftDown(queue, 0, node_indices)
        return min_node

    def array_delete_min(self, queue): # Big O(n)
        min_node = None
        min_index = -1
        for i in range(len(queue)):
            if min_node is None or min_node.dist > queue[i].dist:
                min_node = queue[i]
                min_index = i
        queue.pop(min_index)
        return min_node

    def decKey(self, queue, node_indices, node):
        self.bubbleUp(queue, node_indices[node], node_indices)

    # While the element is not the root or not the left element
    # If element is less than parent, swap the elements
    def bubbleUp(self, queue, index, node_indices): # Big O(log(n))
        parent_index = (index - 1) // 2
        node_indices[queue[index]] = index
        while index != 0 and queue[index].dist < queue[parent_index].dist:
            node_indices[queue[index]], node_indices[queue[parent_index]] = (parent_index, index)
            queue[index], queue[parent_index] = (queue[parent_index], queue[index])
            index = parent_index
            parent_index = (index - 1) // 2

    # If the current node has at a minimum 1 child, we get the index of min child,
    # swap values of current element
    def siftDown(self, queue, index, node_indices): # Big O(log(n))
        min_child_index = self.minChild(queue, index)
        node_indices[queue[index]] = index
        while min_child_index != 0 and queue[index].dist > queue[min_child_index].dist:
            node_indices[queue[index]], node_indices[queue[min_child_index]] = (min_child_index, index)
            queue[index], queue[min_child_index] = (queue[min_child_index], queue[index])
            index = min_child_index
            min_child_index = self.minChild(queue, index)

    # If the current node has only 1 child, we will return the index of the unique child
    def minChild(self, queue, i):
        if (i+1)*2 > len(queue):
            return 0
        elif (i+1)*2 == len(queue):
            return ((i+1)*2)-1
        else:
            b = (i+1)*2
            a = ((i+1)*2)-1
            return a if queue[a].dist < queue[b].dist else b



## TIME & SPACE COMPLEXITY FOR ARRAY VS HEAP
#
# Implement two versions of a priority queue class, one using an unsorted array (a python list) as the
# data structure and one using a heap:
#
#   For the array implementation, insert and decrease-key are simple O(1) operations, but deletemin will unavoidably be O(|V|).
#
#   For the heap implementation, all three operations (insert, delete-min, and decrease-key) must be
#   worst case O(log|V|). For your binary heap implementation, you may implement the binary heap
#   with an array, but remember that decrease-key will be O(|V|) unless you have a separate array (or
#   map) of object references into your binary heap, so that you can have fast access to an arbitrary
#   node. Thus, you must use the separate lookup map. Also, don't forget that you will need to adjust
#   this lookup array/map of references every time you swap elements in the heap.
