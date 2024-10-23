from imputegap.recovery.explainer import Explainer
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

#datasets = ["eeg-test", "eeg", "chlorine", "drift", "climate"]

datasets = ["meteo","bafu"]


for dataset in datasets:
    # small one
    data_n = TimeSeries()
    data_n.load_timeseries(data=utils.search_path(dataset), max_series=20, max_values=400, header=False)

    data_n.plot(raw_data=data_n.data, title="02_"+dataset + " - raw data 20x400", max_series=20, save_path="./dataset/docs/" + dataset + "", display=False)
    data_n.plot(raw_data=data_n.data, title="03_"+dataset + " - raw data 01x400", max_series=1, save_path="./dataset/docs/" + dataset + "", display=False)

    data_n.normalize(normalizer="min_max")
    data_n.plot(raw_data=data_n.data, title="04_"+dataset + " - norm min_max data 01x400", max_series=20, save_path="./dataset/docs/" + dataset + "", display=False)


    # full one
    data_n = TimeSeries()
    data_n.load_timeseries(data=utils.search_path(dataset), header=False)

    data_n.plot(raw_data=data_n.data, title="01_"+dataset + " - raw data - NxM", save_path="./dataset/docs/" + dataset + "", display=False)

    categories, features = Explainer.load_configuration()
    characteristics, descriptions = Explainer.extract_features(data=data_n.data, features_categories=categories, features_list=features, do_catch24=False)


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
