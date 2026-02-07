import shlex
import math


def stdDev(nums):
    total = 0
    length = 0
    for num in nums:
        total += num
        length += 1
    avg = total / length
    temp = 0
    for num in nums:
        temp += num - avg
    return math.sqrt((temp**2) / length), avg


class Attribute:
    def __init__(self):
        self.name = ""  # Name of the attribute
        self.continuous = False  # If it is discrete or continuous
        self.values = []  # If discrete, what are the possible values
        self.distWeight = 1


class Dataset:
    def __init__(self, file):
        self.title = ""  # Title of the dataset
        self.features = []  # List of the features
        self.data = []  # A list of datapoint dictionaries
        self.file = file

        arff_file = open(self.file)  # Open the file
        reading_data = False  # A bool to change the operation once we start reading actual data
        for file_line in arff_file:  # Read through the file
            file_line = file_line.rstrip()  # This removes any trailing whitespace and new line character
            file_line = file_line.lstrip()  # This removes any leading whitespace
            if not file_line.strip():  # Ignore any lines that are empty
                continue
            line = shlex.split(
                file_line)  # A special split function, which will a string into an array on white space, but will ignore quoted portions
            if line[0] == '%':  # Ignore comments
                continue

            if line[0] == "@data":  # Move to the next line and begin reading in data
                reading_data = True
                continue
            if not reading_data:  # Prep dataset
                if line[0] == "@relation":  # Set dataset title
                    self.title = line[1]
                elif line[0] == "@attribute":
                    feature = Attribute()
                    feature.name = line[1]
                    if line[2] == "numeric":  # It is a continuous value
                        feature.continuous = True
                    else:
                        enums = slice(1, -1)  # Remove the first and last char, the {}
                        feature.values = line[2][enums].split(',')
                    self.features.append(feature)
                else:
                    print("ERROR: Unknown Token")
                    exit()
            else:  # Read in data
                # DrKow: Made several fixes over these next few lines:
                data = file_line.split(',')  # Split the data into a list of it's values
                datapoint = {}  # A dictionary of feature:value
                for (value, attribute) in zip(data,
                                              self.features):  # Loops through both lists to assign them in a dictionary
                    if attribute.continuous:
                        value = float(value)
                    datapoint[attribute.name] = value
                self.data.append(datapoint)  # Add the data to the dataset
        self.targetFeature = self.features[-1]

    def Standardize(self):
        values = []
        for feat in self.features:
            if feat.continuous:
                for dat in self.data:
                    values.append(float(dat[feat.name]))
                standardDeviation, mean = stdDev(values)
                for dat in self.data:
                    dat[feat.name] = str((float(dat[feat.name]) - mean) / standardDeviation)

    def Normalize(self):
        values = []
        for feat in self.features:
            if feat.continuous:
                for dat in self.data:
                    values.append(float(dat[feat.name]))
                valMax = max(values)
                valMin = min(values)
                for dat in self.data:
                    dat[feat.name] = str((float(dat[feat.name]) - valMin) / (valMax - valMin))
