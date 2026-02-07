import dataReader as datR


def GDse(instance: {}, weight: {}, target):
    guess = 0
    for feat in instance:
        if feat == target.name:
            continue
        guess += instance[feat]*weight[feat]
    return (instance[target.name] - guess)**2


def GradientDescent(dataset: datR.Dataset):
    sse = 0
    weights = {}
    costs = {}
    sse_best = 0
    weights_best = {}
    alpha = 2 * (10 ** -8)

    # Set initial weights to 1
    for feat in dataset.features:
        if feat == dataset.targetFeature:
            continue
        weights.update({feat.name: 1})
        costs.update({feat.name: 1})

    # Run once to get the starting sse
    weights_best = weights
    total_error = 0
    for item in dataset.data:
        total_error += GDse(item, weights, dataset.targetFeature)
    sse = total_error
    sse_best = sse

    # Now actually run GD
    iterations = 0
    max_iterations = 10000
    no_sse_change = 0
    max_no_change = 100000000000000000000000
    while iterations < max_iterations and no_sse_change < max_no_change:
        iterations += 1
        if iterations % 1000 == 0 or (iterations <= 1000 and iterations % 100 == 0):
            print("Testing item " + str(iterations) + "/" + str(max_iterations))
        for feat in dataset.features:
            if feat == dataset.targetFeature:
                continue
            weights[feat.name] += costs[feat.name]*alpha
            total_error = 0
            for item in dataset.data:
                total_error += GDse(item, weights, dataset.targetFeature)
            new_sse = total_error
            costs[feat.name] = -1 * (sse - new_sse)
        total_error = 0
        for item in dataset.data:
            total_error += GDse(item, weights, dataset.targetFeature)
        sse = total_error
        if sse < sse_best:
            no_sse_change = 0
            weights_best = weights
            sse_best = sse
        else:
            no_sse_change += 1

    return sse_best, weights_best

