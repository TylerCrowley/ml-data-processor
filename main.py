import dataReader as datR
import timeit
import ID3
import KNN
import NB
import GD


#  Begins data reporting
trainingData = datR.Dataset("Table7-1 numeric only.arff")
print("Dataset: " + trainingData.title)

for atr in trainingData.features:
    if atr.continuous:
        mn = min(range(len(trainingData.data)), key=lambda index: trainingData.data[index][atr.name])
        mx = max(range(len(trainingData.data)), key=lambda index: trainingData.data[index][atr.name])
        print(atr.name.capitalize() + ": Continuous, Min=" + str(trainingData.data[mn][atr.name]) + ", Max=" + str(trainingData.data[mx][atr.name]))
    else:
        print(atr.name.capitalize() + ": Discrete, Values=" + ','.join(atr.values))

startTest = timeit.default_timer()
sse, weights = GD.GradientDescent(trainingData)
stopTest = timeit.default_timer()

print("Sum of Squared Errors: ", str(sse))
print("Weights: ", weights)
print("Runtime: " + str(stopTest - startTest))
