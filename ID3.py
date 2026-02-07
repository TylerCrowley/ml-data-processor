import dataReader as datR
import math
from copy import copy


class Node:
    parent = None   # Parent node
    children = {}   # {attribute_value:node}
    nodeAttribute: datR.Attribute = datR.Attribute()   # The attribute which this node is testing for
    output = None


class DecisionTree:
    data_set: datR.Dataset
    root: Node


def removeFeature(dataset, feature):
    print("Remove " + feature.name)
    copySet = copy(dataset)
    for feat in copySet.features:
        if feat.name == feature.name:
            copySet.features.remove(feat)
    return copySet


def entropy(data, features):
    outcomes = {}   # {target_feature:instances}
    entropyTotal = 0
    for point in data:
        if point[features[-1]] in outcomes:
            outcomes[point[features[-1]]] += 1
        else:
            outcomes.update({point[features[-1]]: 1})
    for target in outcomes.values():
        entropyTotal += (target / len(data)) * math.log2((target / len(data)))
    return 0 - entropyTotal


def nextFeature(dataset):
    entropies = {}  # {feature:entropy}
    for feat in dataset.features[:-1]:  # All except target feature
        entropies.update({feat: 0})
        for val in feat.values:
            if any(d[feat] == val for d in dataset.data):
                print(feat)
                entropies[feat] += entropy([instance for instance in dataset.data if instance[feat] == val], dataset.features)
    print(entropies)
    return min(entropies, key=entropies.get)


def commonOutput(dataset):
    outputs = {}
    for point in dataset.data:
        out = point[dataset.features[-1]]
        if out in outputs:
            outputs[out] += 1
        else:
            outputs.update({out: 1})
    return max(outputs, key=outputs.get)


def buildTree(parentNode, currentData):
    if len(currentData.data) == 0:
        return
    newNode: Node = Node()
    newNode.parent = parentNode
    newNode.output = commonOutput(currentData)
    if len(currentData.features) > 1:
        newNode.nodeAttribute = nextFeature(currentData)
        if len(currentData.data) > 1:
            tempDataset: datR.Dataset = datR.Dataset()
            tempDataset.title = copy(currentData.title)
            tempDataset.features = removeFeature(currentData, newNode.nodeAttribute).features
            for val in newNode.nodeAttribute.values:
                tempDataset.data = copy([instance for instance in currentData.data if instance[newNode.nodeAttribute] == val])
                if len(tempDataset.data) > 0:
                    newNode.children.update({val: buildTree(newNode, tempDataset)})
    for key, child in newNode.children:
        if not child.nodeAttribute:
            newNode.children.pop(key)
    return newNode


def printTree(node, layer=0):
    print('-'*layer, end='')
    if layer == 0:
        print("ROOT")
    else:
        parentVal = [key for key, value in node.parent.children.items() if value == node]
        print(node.parent.nodeAttribute.name + "=" + parentVal[0] + ":" + node.output)
    for child in node.children.values():
        printTree(child, layer + 1)


def evaluate(point, root):
    if point[root.nodeAttribute] in root.children:
        return evaluate(point, root.children[point[root.nodeAttribute]])
    else:
        return root.output


def process(data, root, target):
    true_pn = 0
    false_pn = 0
    for point in data:
        if evaluate(point, root) == point[target]:
            true_pn += 1
        else:
            false_pn += 1
    return (true_pn / (true_pn + false_pn)) * 100


def ID3algorithm(training, testing):
    tree = DecisionTree()
    tree.root = buildTree(None, training)
    print(str(process(testing.data, tree.root, testing.features[-1])))
    print("done")
