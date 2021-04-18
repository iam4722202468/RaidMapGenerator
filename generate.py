import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.algorithms.connectivity import minimum_st_edge_cut

colorArray = [ '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000' ]

class Node:
    name = ""
    children = None
    keys = None
    openings = None
    locks = None
    usable = None

    pos = None

    parent = None

    def __init__(self, name, posX, posY):
        self.name = name
        self.pos = (posX, posY)
        self.children = []
        self.keys = []
        self.openings = set()
        self.locks = {}
        self.usable = True
        
def randomOpenings(dir):
    o = set()

    if dir != None:
        o.add(dir)

    types = ['L', 'R', 'U', 'D']
    for type in types:
        if random.random() > 0.2:
            o.add(type)

    return o

def drawMapRecur(root, out):
    out[root.pos[1]*3+1][root.pos[0]*3+1] = 2
    out[root.pos[1]*3][root.pos[0]*3] = 1
    out[root.pos[1]*3+2][root.pos[0]*3+2] = 1
    out[root.pos[1]*3][root.pos[0]*3+2] = 1
    out[root.pos[1]*3+2][root.pos[0]*3] = 1

    # keep in mind this is y, x
    if 'L' not in root.openings:
        out[root.pos[1]*3+1][root.pos[0]*3] = 1
    else:
        findPos = (root.pos[0]-1, root.pos[1])

        for c in root.children:
            if c.pos == findPos:
                out[root.pos[1]*3+1][root.pos[0]*3] = 3
                out[root.pos[1]*3+1][root.pos[0]*3-1] = 3

    if 'R' not in root.openings:
        out[root.pos[1]*3+1][root.pos[0]*3+2] = 1
    else:
        findPos = (root.pos[0]+1, root.pos[1])
        for c in root.children:
            if c.pos == findPos:
                out[root.pos[1]*3+1][root.pos[0]*3+2] = 3
                out[root.pos[1]*3+1][root.pos[0]*3+3] = 3

    if 'U' not in root.openings:
        out[root.pos[1]*3][root.pos[0]*3+1] = 1
    else:
        findPos = (root.pos[0], root.pos[1]-1)
        for c in root.children:
            if c.pos == findPos:
                out[root.pos[1]*3][root.pos[0]*3+1] = 3
                out[root.pos[1]*3-1][root.pos[0]*3+1] = 3

    if 'D' not in root.openings:
        out[root.pos[1]*3+2][root.pos[0]*3+1] = 1
    else:
        findPos = (root.pos[0], root.pos[1]+1)
        for c in root.children:
            if c.pos == findPos:
                out[root.pos[1]*3+2][root.pos[0]*3+1] = 3
                out[root.pos[1]*3+3][root.pos[0]*3+1] = 3

    for x in root.children:
        drawMapRecur(x, out)

def drawMap(root, map):
    output = [[0 for i in range(len(map)*3)] for j in range(len(map)*3)]

    drawMapRecur(root, output)

    for x in output:
        for y in x:
            print(y, end=" ")

        print()

def findHiddenConnections(root, map):
    size = len(map)
    hidden = []

    for y in map:
        for x in y:
            if 'L' in x.openings and x.pos[0] -1 >= 0:
                n = map[x.pos[0]-1][x.pos[1]]
                if n not in x.children and x not in n.children and 'R' in n.openings:
                    # print("Loop found at", x.pos, 'L')
                    hidden.append((x.pos, 'L', x, n))

            if 'R' in x.openings and x.pos[0]+1 < size:
                n = map[x.pos[0]+1][x.pos[1]]

                if n not in x.children and x not in n.children and 'L' in n.openings:
                    # print("Loop found at", x.pos, 'R')
                    hidden.append((x.pos, 'R', x, n))

            if 'U' in x.openings and x.pos[1] - 1 >= 0:
                n = map[x.pos[0]][x.pos[1]-1]
                if n not in x.children and x not in n.children and 'D' in n.openings:
                    # print("Loop found at", x.pos, 'U')
                    hidden.append((x.pos, 'U', x, n))

            if 'D' in x.openings and x.pos[1]+1 < size:
                n = map[x.pos[0]][x.pos[1]+1]
                if n not in x.children and x not in n.children and 'U' in n.openings:
                    # print("Loop found at", x.pos, 'D')
                    hidden.append((x.pos, 'D', x, n))

    return hidden


def removeInvalid(size, target):
    toRemove = []

    for i in range(len(target)):
        if target[i][1] < 0 or target[i][2] < 0 or target[i][1] > size or target[i][2] > size:
            toRemove.append(target[i])

    for x in toRemove:
        target.remove(x)


def printTree(node, depth = 0):
    print(' '*depth*4, node.pos, node.openings)
    for x in node.children:
        printTree(x, depth + 1)

def getNext(node, nodes, nodeMap):
    searching = []

    if 'L' in node.openings:
        searching.append(('R', node.pos[0]-1, node.pos[1]))
    if 'R' in node.openings:
        searching.append(('L', node.pos[0]+1, node.pos[1]))
    if 'U' in node.openings:
        searching.append(('D', node.pos[0], node.pos[1]-1))
    if 'D' in node.openings:
        searching.append(('U', node.pos[0], node.pos[1]+1))
    
    removeInvalid(len(nodeMap)-1, searching)

    found = []

    for x in searching:
        n = nodeMap[x[1]][x[2]]
        if n in nodes:
            n.openings = randomOpenings(x[0])
            node.children.append(n)
            n.parent = node
            found.append(n)
            nodes.remove(n)

    return found

def removeConnections(map, toRemove):
    for x in toRemove:
        dir = x[1]
        pos = x[0]

        n = map[pos[0]][pos[1]]
        n.openings.remove(dir)

def findAllConnections(root):
    connections = []

    id = root.name

    for x in root.children:
        connections.append((id, x.name))
        connections += findAllConnections(x)

    return connections

def genRandomMap(size):
    nodeMap = []
    nodes = []

    for i in range(size):
        nodeMap.append([])
        for j in range(size):
            newNode = Node(str(i*size + j), i, j)
            nodeMap[-1].append(newNode)
            nodes.append(newNode)

    startNode = nodeMap[int(size/2)][int(size/2)]
    startNode.openings = randomOpenings(None)
    nodes.remove(startNode)

    addQueue = [startNode]

    while len(addQueue) > 0:
        currNode = addQueue.pop(0)
        toAdd = getNext(currNode, nodes, nodeMap)
        addQueue += toAdd

    return nodeMap, startNode

def getAllNodes(root):
    nodes = {}
    nodes[root.name] = root

    for x in root.children:
        nodes.update(getAllNodes(x))

    return nodes

def getDoorPlacements(G, src, tgt):
    return minimum_st_edge_cut(G, src, tgt)

def getPossibleKeyNodes(G, nodes, root, path, cutset):
    distanceLow = 2
    distanceHigh = 6

    nodeList = []

    for x in G.nodes:
        pl = -1

        for y in cutset:
            if y[0] in G.nodes:
                path = nx.shortest_path(G, source=x, target=y[0], method='dijkstra')
                if len(path) < pl or pl == -1:
                    pl = len(path)

            if y[1] in G.nodes:
                path = nx.shortest_path(G, source=x, target=y[1], method='dijkstra')
                if len(path) < pl or pl == -1:
                    pl = len(path)

        if len(nodes[x].keys) == 0 and x != root.name and pl >= distanceLow and pl <= distanceHigh:
            nodeList.append(nodes[x])

    return nodeList

def getColor():
    return colorArray.pop(0)

def findLongestPath(G, nodes, root):
    longestPath = []
    nodeVal = ''

    for node in G.nodes:
        if node != root.name and nodes[node].usable:
            path = nx.shortest_path(G, source=root.name, target=node, method='dijkstra')

            if len(path) > len(longestPath):
                longestPath = path
                nodeVal = node

    return nodeVal, longestPath

def findBossRoom(root, found):
    newCount = found
    newNode = root

    for x in root.children:
        if x in root.locks:
            newCount_, newNode_ = findBossRoom(x, found+1)
        else:
            newCount_, newNode_ = findBossRoom(x, found)

        if newCount_ > newCount:
            newCount = newCount_
            newNode = newNode_

    return newCount, newNode

def placeDoorsKeys(G, nodes, root):
    # Find longest path from root node
    gOrg = G
    G = G.copy()

    possibleKeyPlacements = G.nodes

    print("Root node is", root.name)

    while len(possibleKeyPlacements) > 0 and len(G.nodes) > 1:
        nodeVal, longestPath = findLongestPath(G, nodes, root)

        if nodeVal == '':
            break

        doorPlacements = getDoorPlacements(gOrg, root.name, nodeVal)

        print('Furthest node is', nodeVal, longestPath)

        # Remove unneded edges
        G.remove_edges_from(doorPlacements)

        # Remove nodes we can no longer reach
        removeNodes = []
        for x in G.nodes:
            try:
                path = nx.shortest_path(G, source=root.name, target=x, method='dijkstra')
            except nx.NetworkXNoPath:
                if x != root.name:
                    removeNodes.append(x)

        print('Removing', removeNodes)
        G.remove_nodes_from(removeNodes)

        # Find key placements
        possibleKeyPlacements = getPossibleKeyNodes(G, nodes, root, longestPath, doorPlacements)

        if len(possibleKeyPlacements) == 0:
            break

        chosenNode = random.choice(possibleKeyPlacements)
        print('Placing key in', chosenNode.name)

        color = getColor()
        chosenNode.keys.append(color)

        for x in doorPlacements:
            if len(doorPlacements) > 1:
                nodes[x[0]].usable = False
                nodes[x[1]].usable = False

            if nodes[x[1]] in nodes[x[0]].locks:
                print("ERROR", 'DOUBLE color. Try again')
                exit()
            if nodes[x[0]] in nodes[x[1]].locks:
                print("ERROR", 'DOUBLE color. Try again')
                exit()

            nodes[x[0]].locks[nodes[x[1]]] = color
            # print('Adding key', nodes[x[0]].name, nodes[x[1]].name, color)

        # print(doorPlacements)

map, root = genRandomMap(5)
printTree(root)
hidden = findHiddenConnections(root, map)
nodeList = getAllNodes(root)

if len(nodeList) == 1:
    print("Only one node exists")
    exit()

# print(len(nodeList))
# removeConnections(map, findHiddenConnections(root, map))

drawMap(root, map)

conn = findAllConnections(root)
print(conn)

G = nx.Graph()

for x in findHiddenConnections(root, map):
    conn.append((x[2].name, x[3].name))

G.add_edges_from(conn)
placeDoorsKeys(G, nodeList, root)

color_map = []
node_borders = []

for x in G.nodes:
    if len(nodeList[x].keys) > 0:
        color_map.append(nodeList[x].keys[0])
        node_borders.append(3)
    else:
        color_map.append("#000000")
        node_borders.append(1)

edge_colors = []
edge_weights = []

for x in G.edges:
    n1 = nodeList[x[0]]
    n2 = nodeList[x[1]]

    if n1 in n2.locks:
        # print('found key', x[1], x[0], n2.locks[n1])
        edge_colors.append(n2.locks[n1])
        edge_weights.append(3)
    elif n2 in n1.locks:
        # print('found key', x[0], x[1], n1.locks[n2])
        edge_colors.append(n1.locks[n2])
        edge_weights.append(3)
    else:
        edge_colors.append('#000000')
        edge_weights.append(1)

pos = nx.kamada_kawai_layout(G)

p = findBossRoom(root, 0)
print(p[0], p[1].name)

nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), linewidths=node_borders, node_color = '#ffffff', edgecolors = color_map, node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width = edge_weights, edge_color = edge_colors, arrows=False)

plt.show()
