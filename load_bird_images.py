import os
import cv2
from matplotlib import pyplot
import numpy as np
import multiprocessing
from functools import partial

def remove_values_from_list(the_list, val):
   return [value for value in the_list if val not in value]

def read_bird(file, r, size):
    bird = cv2.imread(r + '//' + file)
    bird = cv2.resize(bird, size)
    return bird

def parallel_birds(r, f, size):
    pool = multiprocessing.Pool(processes=4)#set processes equal to num CPU cores
    func = partial(read_bird, r=r, size=size)
    birds = pool.map(func, f)
    pool.close()
    return birds

def load_birds(size):
    birds = []
    class_names = []
    class_number = 0
    class_numbers = []
    for r, d, f in os.walk('birds'):
        print('loading birds from:', r)
        if len(f) != 0:
            f = remove_values_from_list(f, '._')
            birds.extend(parallel_birds(r, f, size))
            class_names.extend(r for j in range(len(f)))
            class_numbers.extend(class_number for j in range(len(f)))
            class_number += 1
    birds = np.array(birds)
    class_numbers = np.array(class_numbers)
    class_names = np.array(class_names)
    return birds, class_numbers, class_names

if __name__ == '__main__':
    size = (200,200)
    birds, class_numbers, class_names = load_birds(size)
    print(class_numbers)
    print('birds shape =', np.shape(birds))
    print('birds class names =', np.shape(class_names))
    print('birds class numbers =', np.shape(class_numbers))
    for bird in birds:
        pyplot.imshow(bird)
        pyplot.show()