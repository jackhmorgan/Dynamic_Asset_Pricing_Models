from AssetPricing import GenerateBenchmarkModel
import numpy as np

utility_function = 'CRRA'
gamma = 10
size = 4

filename = f'benchmark_{utility_function}_{gamma}_{size}.txt'

bm = GenerateBenchmarkModel(utility_function=utility_function,
                            gamma=gamma,
                            size=size)
np.savetxt(filename, bm)