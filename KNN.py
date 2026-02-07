import dataReader as datR
from operator import itemgetter


def KNearestNeighbors(instance: {}, training: datR.Dataset, k: int, x, y):
    if x % 100 == 0 or (x <= 100 and x % 10 == 0) or x <= 10:
        print("Testing item " + str(x) + "/" + str(y))
    model = training
    neighbors = {}
    index = 0
    for item in model.data:
        distance = 0
        for feat in model.features:
            if feat == model.targetFeature:
                continue
            if feat.continuous:
                distance += abs(float(item[feat.name]) - float(instance[feat.name])) * feat.distWeight
            else:
                if item[feat.name] != instance[feat.name]:
                    distance += 1 * feat.distWeight
        neighbors[index] = distance
        index += 1
    kNearest = dict(sorted(neighbors.items(), key=itemgetter(1), reverse=False)[:k])  # This line creates a new dict of only the knn
    targets = {}
    for vals in model.targetFeature.values:
        targets[vals] = 0
    for nbr in kNearest:
        targets[model.data[nbr][model.targetFeature.name]] += 1 / (neighbors[nbr] + 1)
    return max(targets, key=targets.get)
