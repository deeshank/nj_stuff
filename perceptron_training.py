import glob
import os
import random
import sys
from twisted.internet import threads, reactor
import numpy
import texttable


__author__ = 'deepak'


class Utils(object):
    def __init__(self, data_file=None, labels_file=None,
                 training_set_file=None):
        if data_file:
            self.data_file = os.path.expanduser(data_file)
        else:
            self.data_file = None
        if labels_file:
            self.labels_file = os.path.expanduser(labels_file)
        else:
            self.labels_file = None

    def load_labels(self):
        label_file = os.path.expanduser(self.labels_file)
        f = open(label_file)
        dump = f.read()
        f.close()
        l = {}
        for line in dump.split('\n'):
            if line:
                l[int(line.split()[1])] = int(line.split()[0])
        labels = []
        for x in range(len(l.keys())):
            labels.append(l[x])
        return labels

    def parse_dataset(self):
        """
        Parse the input data file

        :param filename: filename of the data file to parse the records
        containing measures

        :return: dataset
        """
        f = open(self.data_file, "r")
        data_content = f.read()
        f.close()
        dataset = []
        for x in data_content.split("\n"):
            if x:
                dataset.append([float(y) for y in x.split()])
        return dataset

    @staticmethod
    def load_trained_labels(train_file):
        """
        Parse the trained label set

        :param inputfilename: filename of the trained data file containing the
        labels map for random record numbers

        :return: data_labels_map
        """
        f = open(train_file, "r")
        data_content = f.read()
        f.close()
        labels = {}
        lst = []
        for x in data_content.split("\n"):
            if x:
                key = int(x.split()[0])
                val = int(x.split()[1])
                if key in labels:
                    labels[key].append(val)
                else:
                    labels[key] = [val]
        train_set = []
        for x in labels.values():
            train_set += x
        return train_set

    def compute_balerr(self, results):
        if not self.labels_file:
            return "No Labels File"

        labels_file = os.path.expanduser(self.labels_file)
        f = open(labels_file, "r")
        data_content = f.read()
        f.close()
        labels = {}
        for x in data_content.split("\n"):
            if x:
                labels[int(x.split()[1])] = int(x.split()[0])
        matches = {}
        mismatches = {}
        for key in results.keys():
            if labels[key] == results[key]:
                matches[labels[key]] = matches.get(labels[key], 0) + 1
            else:
                mismatches[labels[key]] = mismatches.get(labels[key], 0) + 1

        total_labels = list(set(labels.values()))
        balerr = 0
        for x in total_labels:
            balerr += mismatches.get(x, 0.0) / float(
                matches.get(x, 0) + mismatches.get(x, 0))
        balerr = balerr / float(len(total_labels))
        return balerr


    @staticmethod
    def mul_by_const(list_a, constant):
        # return [constant * x for x in list_a]
        return numpy.array(list_a) * constant

    @staticmethod
    def add(list_a, list_b):
        # return [a + b for a, b in zip(list_a, list_b)]
        return numpy.array(list_a) + numpy.array(list_b)

    @staticmethod
    def dot_product(list_a, list_b):
        # return sum([a * b for a, b in zip(list_a, list_b)])
        return numpy.dot(numpy.array(list_a), numpy.array(list_b))


class Perceptron(object):
    def __init__(self, dataset, labels, eta=0.001):
        self.dataset = self.add_for_w0(dataset)
        self.labels = self.pre_process_labels(labels)
        self.w = [random.random()] * (len(self.dataset[0]))
        # self.w = numpy.random.random_sample((1, len(dataset[0])+1))
        self.w = Utils.mul_by_const(self.w, eta)
        self.eta = eta
        self.iterations = 0

    def compute_f(self):
        f = 0.0
        for i, ip in enumerate(self.dataset):
            a = Utils.dot_product(self.w, ip)
            f += (self.labels[i] - a) ** 2
        return f


    def compute_delf(self):
        df = [0] * len(self.dataset[0])
        for i, ip in enumerate(self.dataset):
            a = Utils.dot_product(self.w, ip)
            tmp = Utils.mul_by_const(ip, self.labels[i] - a)
            df = Utils.add(df, tmp)
        return df

    @staticmethod
    def add_for_w0(data):
        if type(data[0]) == list:
            return [x + [1] for x in data]
        elif type(data[0]) == tuple:
            data = [list(x) for x in data]
            return [x + [1] for x in data]
        else:
            return list(data) + [1]

    @staticmethod
    def pre_process_labels(labels):
        return [-1 if x == 0 else x for x in labels]

    def train(self):
        obj = self.compute_f()
        prev_obj = obj + 1000
        while prev_obj - obj > 0.001:
            prev_obj = obj
            delta_f = self.compute_delf()
            self.w = Utils.add(self.w, Utils.mul_by_const(delta_f, self.eta))
            obj = self.compute_f()
            self.iterations += 1
            sys.stdout.write("\r" + "Iteration Count : %s [%s]" % (self.iterations,prev_obj - obj))
            sys.stdout.flush()

    def classify(self, ip_vector):
        ip_vector = self.add_for_w0(ip_vector)
        if Utils.dot_product(self.w, ip_vector) > 0:
            return 1
        else:
            return 0

#
# dataset = [(0, 0), (0, 1), (1, 0), (1, 1), (10, 10), (10, 11), (11, 10),
# (11, 11)]
# labels = [0, 0, 0, 0, 1, 1, 1, 1]
# lblfile = "~/sample.labels"
#
# p = Perceptron(dataset, labels)
# p.train()
# print p.w
# print p.iterations
# r = {}
# for i, x in enumerate(dataset):
# r[i] = p.classify(x)
#
# util = Utils(labels_file=lblfile)
# print util.compute_balerr(r)

table = texttable.Texttable()
table.add_row(["Filename", "bal_err"])

source_path = "~/data/hill_valley"

filename = source_path.strip().rsplit("/")[-1]
data_set = source_path + "/" + filename + ".data"
train_files = source_path + "/" + filename + ".trainlabels.*"
labels = source_path + "/" + filename + ".labels"

# data_set = "~/data/micromass/micromass.data"
# labels = "~/data/micromass/micromass.labels"
# train_files = "~/data/micromass/micromass.trainlabels.*"

util = Utils(data_file=data_set, labels_file=labels)
data = util.parse_dataset()
labels = util.load_labels()

# p = Perceptron(data, labels, eta=0.000000001)
#p.train()

avg_err = 0.0
berr=0

training_sets = glob.glob(os.path.expanduser(train_files))
training_sets.sort()
for training_set in training_sets:
    train_set = []
    unclass_set = {}
    class_labels = []
    train_records = Utils.load_trained_labels(training_set)
    classified = range(len(data))
    unclassified = list(set(classified) - set(train_records))
    for rec in train_records:
        train_set.append(data[rec])
        class_labels.append(labels[rec])
    for rec in unclassified:
        unclass_set[rec]=data[rec]
    p = Perceptron(train_set, class_labels, eta=0.00000000000001)#0.000000001)
    p.train()
    res = {}
    for recno,record in unclass_set.items():
        res[recno] = p.classify(record)
    berr = util.compute_balerr(res)
    avg_err += berr
    table.add_row([training_set.rsplit("/")[-1], berr])

print ""
print table.draw()
print ""

print avg_err/float(len(training_sets))