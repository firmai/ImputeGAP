# import algo_collection
# import numpy as np
#
# mat = np.loadtxt("airq_normal.txt")
# mat_miss = np.copy(mat)
# mat_miss[50:149, 0] = np.nan
#
# mat_rec = algo_collection.native_cdrec_param(mat_miss, 2, 1E-6, 100)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))
#
# mat_rec = algo_collection.native_stmvl(mat_miss)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))
#
# mat_rec = algo_collection.native_stmvl_param(mat_miss, 7, 0.85, 2.0)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))

