import numpy


def load_data(file_path):
    f = open(file_path, 'rt')
    data = []
    for line in f:
        line = line.replace("\n", "")
        data.append(eval(line))
    data = numpy.array(data)
    data_class = data[:, 0]
    data_addr = data[:, 1:]

    return data_class, data_addr
