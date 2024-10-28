import numpy as np

dis = np.loadtxt('../data/cmu/distance.txt')
dis = sorted(dis)
np.savetxt('../data/cmu/dis_order.txt', dis)
