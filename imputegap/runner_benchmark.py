from imputegap.recovery.benchmark import Benchmark

# VARIABLES
reconstruction = False
save_dir = "./analysis"
nbr_run = 1

# SELECT YOUR DATASET(S) :
datasets_full = ["eeg-alcohol", "eeg-reading", "fmri-objectviewing", "fmri-stoptask", "chlorine", "drift"]
datasets_demo = ["drift", "chlorine"]

# SELECT YOUR OPTIMIZER :
optimiser_bayesian = {"optimizer": "bayesian", "options": {"n_calls": 2, "n_random_starts": 50, "acq_func": "gp_hedge", "metrics": "RMSE"}}
optimiser_greedy = {"optimizer": "greedy", "options": {"n_calls": 250, "metrics": "RMSE"}}
optimiser_pso = {"optimizer": "pso", "options": {"n_particles": 50, "iterations": 10, "metrics": "RMSE"}}
optimiser_sh = {"optimizer": "sh", "options": {"num_configs": 10, "num_iterations": 5, "metrics": "RMSE"}}
optimizers_demo = [optimiser_bayesian]

# SELECT YOUR ALGORITHM(S) :
algorithms_full = ["mean", "cdrec", "stmvl", "iim", "mrnn"]
algorithms_demo = ["cdrec", "iter_svd", "grouse", "dynammo", "rosl", "soft_imp", "spirit", "svt", "tkcm"]
algorithms_dl = ["brits", "deep_mvi", "mpin", "pristi"]
algorithms_all = ["mean", "cdrec", "stmvl", "iim", "mrnn", "iter_svd", "grouse", "dynammo", "rosl", "soft_imp", "spirit", "svt", "tkcm", "brits", "deep_mvi", "mpin", "pristi"]

# SELECT YOUR CONTAMINATION PATTERN(S) :
patterns_full = ["mcar", "mp", "blackout", "disjoint", "overlap", "gaussian"]
patterns_demo = ["mcar"]

# SELECT YOUR MISSING RATE(S) :
x_axis = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
x_axis_s = [0.2, 0.4, 0.6]

if not reconstruction:
    # START THE ANALYSIS
    list_results, sum_scores = Benchmark().eval(algorithms=algorithms_all, datasets=datasets_demo, patterns=patterns_demo, x_axis=x_axis, optimizers=optimizers_demo, save_dir=save_dir, runs=nbr_run)
else:
    test_plots = {'eegalcohol': {'mcar': {'mean': {'bayesian': {'0.05': {'scores': {'RMSE': 1.107394798606378, 'MAE': 0.9036474830477748, 'MI': 0.0, 'CORRELATION': 0.0}, 'times': {'contamination': 0.0005860328674316406, 'optimization': 0.0, 'imputation': 0.0002785921096801758, 'log_imputation': -8.19165637409532}}, '0.1': {'scores': {'RMSE': 0.8569349076796438, 'MAE': 0.6416542359734557, 'MI': 0.0, 'CORRELATION': 0.0}, 'times': {'contamination': 0.0009672641754150391, 'optimization': 0.0, 'imputation': 0.00025284290313720703, 'log_imputation': -8.301689083667116}}, '0.2': {'scores': {'RMSE': 0.9609255264919324, 'MAE': 0.756013835497571, 'MI': 0.0, 'CORRELATION': 0.0}, 'times': {'contamination': 0.001461029052734375, 'optimization': 0.0, 'imputation': 0.0001863241195678711, 'log_imputation': -8.608221145649537}}, '0.4': {'scores': {'RMSE': 1.0184989120725458, 'MAE': 0.8150966718352457, 'MI': 0.0, 'CORRELATION': 0.0}, 'times': {'contamination': 0.006487131118774414, 'optimization': 0.0, 'imputation': 0.00014793872833251953, 'log_imputation': -8.826370643510565}}, '0.6': {'scores': {'RMSE': 0.9997401940199045, 'MAE': 0.7985721718600829, 'MI': 0.0, 'CORRELATION': 0.0}, 'times': {'contamination': 0.016824841499328613, 'optimization': 0.0, 'imputation': 0.00016415119171142578, 'log_imputation': -8.715009900651088}}, '0.8': {'scores': {'RMSE': 0.9895691678332014, 'MAE': 0.7901674118013952, 'MI': 0.0, 'CORRELATION': 0.0}, 'times': {'contamination': 0.06057560443878174, 'optimization': 0.0, 'imputation': 0.00021696090698242188, 'log_imputation': -8.437072631391068}}}}, 'cdrec': {'bayesian': {'0.05': {'scores': {'RMSE': 0.27658600512073456, 'MAE': 0.20204444801773774, 'MI': 1.6287285825717355, 'CORRELATION': 0.9837210171556283}, 'times': {'contamination': 0.00024020671844482422, 'optimization': 0.6290972232818604, 'imputation': 0.011005878448486328, 'log_imputation': -4.509350224802846}}, '0.1': {'scores': {'RMSE': 0.2322153312143858, 'MAE': 0.1729082341483471, 'MI': 1.1990748751673153, 'CORRELATION': 0.9640732993793864}, 'times': {'contamination': 0.0020873546600341797, 'optimization': 0.6290972232818604, 'imputation': 0.011507630348205566, 'log_imputation': -4.474282360528691}}, '0.2': {'scores': {'RMSE': 0.21796283300762773, 'MAE': 0.16255811567403466, 'MI': 1.184724280002774, 'CORRELATION': 0.9737521039022545}, 'times': {'contamination': 0.003554224967956543, 'optimization': 0.6290972232818604, 'imputation': 0.01547861099243164, 'log_imputation': -4.1722778619303025}}, '0.4': {'scores': {'RMSE': 0.2852656711446442, 'MAE': 0.19577380664036, 'MI': 1.014828207927502, 'CORRELATION': 0.959485242427464}, 'times': {'contamination': 0.020360827445983887, 'optimization': 0.6290972232818604, 'imputation': 1.276460886001587, 'log_imputation': 0.07915447441603213}}, '0.6': {'scores': {'RMSE': 0.3360171448119046, 'MAE': 0.23184686418998596, 'MI': 0.8789374924043876, 'CORRELATION': 0.9418882413737133}, 'times': {'contamination': 0.08814132213592529, 'optimization': 0.6290972232818604, 'imputation': 3.2481226921081543, 'log_imputation': 0.6821764345232934}}, '0.8': {'scores': {'RMSE': 0.5558362531202891, 'MAE': 0.37446346030237454, 'MI': 0.5772409317426037, 'CORRELATION': 0.8478935496183876}, 'times': {'contamination': 0.17819464206695557, 'optimization': 0.6290972232818604, 'imputation': 26.945740580558777, 'log_imputation': 2.065192854396594}}}}}}}
    Benchmark().generate_plots(runs_plots_scores=test_plots, ticks=x_axis, subplot=False, save_dir=save_dir)
    Benchmark().generate_reports_txt(runs_plots_scores=test_plots, save_dir=save_dir, dataset="chlorine", run=0)
    Benchmark().generate_reports_excel(runs_plots_scores=test_plots, save_dir=save_dir, dataset="chlorine", run=0)