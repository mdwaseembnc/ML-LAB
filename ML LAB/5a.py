import csv
import random
import math

# Load Data
def loadCsv(filename):
    dataset = []
    with open(filename, "rt") as file:
        lines = csv.reader(file)
        for row in lines:
            # Skip rows that are completely empty or contain invalid data
            if row and all(x != '' for x in row):  # Make sure no empty cells in the row
                try:
                    dataset.append([float(x) for x in row])  # Convert each entry to float
                except ValueError:
                    # If conversion fails, skip the row
                    print(f"Skipping invalid row: {row}")
                    continue
    return dataset

# Split the data into Training and Testing randomly
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

# Separate data by Class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

# Calculate Mean
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Calculate Standard Deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# Summarize the data
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

# Summarize Attributes by Class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances) 
    return summaries

# Calculate Gaussian Probability Density Function
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Calculate Class Probabilities
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

# Make a Prediction
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# Return a list of predictions for each test instance.
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

# Calculate accuracy ratio.
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

filename = 'DBetes.csv'  # Ensure the correct file path is provided
splitRatio = 0.70

# Load and split the dataset
dataset = loadCsv(filename)
trainingSet, testSet = splitDataset(dataset, splitRatio)
print(f'Split {len(dataset)} rows into train={len(trainingSet)} and test={len(testSet)} rows')

# Prepare model
summaries = summarizeByClass(trainingSet)

# Test model
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print(f'Accuracy: {accuracy}%')
