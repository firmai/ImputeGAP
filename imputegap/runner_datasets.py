from imputegap.recovery.explainer import Explainer
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

datasets = ["eeg-alcohol", "eeg-reading"]

for dataset in datasets:
    # small one
    data_n = TimeSeries()
    data_n.load_timeseries(data=utils.search_path(dataset), max_series=20, max_values=400, header=False)
    data_n.plot(input_data=data_n.data, max_series=20, save_path="./dataset/docs/" + dataset + "", display=False)
    data_n.plot(input_data=data_n.data, max_series=1, save_path="./dataset/docs/" + dataset + "", display=False)
    data_n.normalize(normalizer="min_max")
    data_n.plot(input_data=data_n.data, max_series=20, save_path="./dataset/docs/" + dataset + "", display=False)

    # 5x one
    data_n = TimeSeries()
    max_series = 3
    max_value = 500
    if dataset == "bafu":
        max_value = 10000
    elif dataset == "chlorine":
        max_value = 1000
    elif dataset == "eeg-alcohol":
        max_value = 256
    elif dataset == "eeg-reading":
        max_value = 1201
    elif dataset == "drift":
        max_value = 400

    data_n.load_timeseries(data=utils.search_path(dataset), max_series=max_series, max_values=max_value, header=False)
    data_n.plot(input_data=data_n.data, save_path="./dataset/docs/" + dataset + "", display=False)
    data_n.normalize(normalizer="min_max")
    data_n.plot(input_data=data_n.data, save_path="./dataset/docs/" + dataset + "", display=False)

    # full one
    data_n = TimeSeries()
    data_n.load_timeseries(data=utils.search_path(dataset), header=False)
    data_n.plot(input_data=data_n.data, save_path="./dataset/docs/" + dataset + "", display=False)

    categories, features = Explainer.load_configuration()
    characteristics, descriptions = Explainer.extractor_pycatch(data=data_n.data, features_categories=categories, features_list=features, do_catch24=False)

    p = "./dataset/docs/"+dataset+"/features_"+dataset+".txt"
    with open(p, 'w') as f:
        for desc in descriptions:
            key, category, description = desc
            if key in characteristics:
                value = characteristics[key]
                f.write(f"|{category}|{description}|{value}|\n")
            else:
                f.write(f"Warning: Key '{key}' not found in characteristics!\n")
    print(f"Table exported to {p}")
