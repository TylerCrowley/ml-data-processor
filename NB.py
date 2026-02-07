import dataReader as datR


def NaiveBayes(instance: {}, training: datR.Dataset, x, y):
    if x % 100 == 0 or (x <= 100 and x % 10 == 0) or x <= 10:
        print("Testing item " + str(x) + "/" + str(y))
    model = {}
    for item in training.data:
        if item[training.targetFeature.name] in model:
            model[item[training.targetFeature.name]].append(item)
        else:
            model.update({item[training.targetFeature.name]: [item]})
    k = len(training.features)
    probs = {}
    for key in model.keys():
        probs.update({key: 0})
        pc = 1
        for item in training.data:
            if item[training.targetFeature.name] == key:
                pc += 1
        probs[key] = float(pc / (len(training.data) + k))
    for key in model.keys():
        for atr in instance.keys():
            if atr == training.targetFeature:   # Don't use the target twice
                continue
            pc = 1
            for point in model[key]:
                if point[atr] == instance[atr]:
                    pc += 1
            probs[key] *= float(pc / (len(model[key]) + k))
    return max(probs, key=probs.get)
