<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# ImputeGAP - Datasets
ImputeGap brings a repository of highly curated time series datasets for missing values imputation. Those datasets contain
real-world time series from various of applications and which cover a wide range of characteristics and sizes. 


## Air-Quality

The air quality dataset brings a subset of air quality measurements collected from 36 monitoring stations in China from 2014 to 2015.

### Summary

| Data info       |                                                                                                                                        |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| codename   | airq                                                                                                                                   |
| name       | Air Quality                                                                                                                            |
| source     | Saverio De Vito (saverio.devito '@' enea.it), ENEA - National Agency for New Technologies, Energy and Sustainable Economic Development | 
| granularity        | hourly                                                                                                                                 |
| size | M=10 N=1000                                                                                                                            |

### Sample plots

![AIR-QUALITY dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/airq/01_airq_m.jpg)
![AIR-QUALITY dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/airq/03_airq_1.jpg)


<br /><hr /><br />



## BAFU

The BAFU dataset, kindly provided by the BundesAmt Für Umwelt (the Swiss Federal Office for the Environment)[https://www.bafu.admin.ch], contains water discharge time series collected from different Swiss rivers containing between 200k and 1.3 million values each and covers the time period from 1974 to 2015. The BAFU dataset appeared in [[2]](#ref2).

### Summary

| Data info          |                                             |
|--------------------|---------------------------------------------|
| codename   | BAFU<br/>bafu                               |
| name       | Hydrological data across multiple stations  |
| URL         | https://www.bafu.admin.ch/bafu/en/home.html |
| granularity        | 30 minutes                                  |
| range       | spans years 1974 to 2015                    |
| size | M=12 N=85203                                |



### Sample Plots

![BAFU dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/bafu/01_bafu-rawdata-NxM_graph.jpg)
![BAFU dataset - raw data 20x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/bafu/02_bafu-rawdata20x400_graph.jpg)
![BAFU dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/bafu/03_bafu-rawdata01x400_graph.jpg)


<br /><hr /><br />


## Chlorine

The Chlorine dataset originates from chlorine residual management aimed at ensuring the security of water distribution systems [Chlorine Residual Management for Water Distribution System Security](https://www.researchgate.net/publication/226930242_Chlorine_Residual_Management_for_Water_Distribution_System_Security), with data sourced from [US EPA Research](https://www.epa.gov/research).
It consists of 50 time series, each representing a distinct location, with 1,000 data points per series recorded at 5-minute intervals.


### Summary

| Dataset info          |                                                                                                                                                                                    |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codename   | chlorine                                                                                                                                                                           |
| name       | Chlorine data                                                                                                                                                                      |
| URL                | https://www.epa.gov/research                                                                                                                                                       |
| source             | United States Environmental Protection Agency, EPANET<br/>Prof. Jeanne M. VanBriesen                                                                                               |
| article            | Vanbriesen, Jeanne & Parks, Shannon & Helbling, Damian & Mccoy, Stacia. (2011). Chlorine Residual Management for Water Distribution System Security. 10.1007/978-1-4614-0189-6_11. | 
| granularity   | 5 minutes                                                                                                                                                                          |
| size | M=50 N=1000                                                                                                                                                                        |


### Sample Plots

![Chlorine dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/chlorine/01_chlorine-rawdata-NxM_graph.jpg)
![Chlorine dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/chlorine/03_chlorine-rawdata01x400_graph.jpg)

<br /><hr /><br />




## Climate

The Climate dataset is an aggregated and processed collection used for climate change attribution studies.
It contains observations data for 18 climate agents across 125 locations in North America [USC Melady Lab](https://viterbi-web.usc.edu/~liu32/data.html).
The dataset has a temporal granularity of 1 month, comprising 10 series with 5,000 values each.
This structure is particularly valuable for spatio-temporal modeling [Spatial-temporal causal modeling for climate change attribution](https://dl.acm.org/doi/10.1145/1557019.1557086), as it enables researchers to account for both spatial and temporal dependencies.

### Summary

| Dataset info          |                                                                         |
|--------------------|-------------------------------------------------------------------------|
| codename   | climate                                                                 |
| name       | Aggregated and Processed data collection for climate change attribution |
| Url                | https://viterbi-web.usc.edu/~liu32/data.html                            |
| item           | NA-1990-2002-Monthly.csv                                                |
| granularity   | 1 month                                                                 |
| size | M=10 N=5000                                                             |



### Sample Plots

![Climate dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/climate/01_climate-rawdata-NxM_graph.jpg)
![Climate dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/climate/03_climate-rawdata01x400_graph.jpg)


<br /><hr /><br />


## Drift
The Drift dataset comprises 13,910 measurements collected from 16 chemical sensors exposed to six different gases, with only batch 10 utilized for this dataset [Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270).
It includes information on the concentration levels to which the sensors were exposed during each measurement.
Data was collected over a 36-month period, from January 2008 to February 2011, at a gas delivery platform facility within the ChemoSignals Laboratory at the BioCircuits Institute, University of California, San Diego [On the calibration of sensor arrays for pattern recognition using the minimal number of experiments](https://www.sciencedirect.com/science/article/pii/S0169743913001937).
The dataset has a time granularity of 6 hours and consists of 100 time series, each containing 1,000 data points. 
### Summary

| Dataset info              |                                                                                                                                                                                        |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codename       | drift                                                                                                                                                                                  |
| name          | Gas Sensor Array Drift Dataset at Different Concentrations                                                                                                                             |
| source                 | Alexander Vergara (vergara '@' ucsd.edu)<br/>BioCircutis Institute<br/>University of California San DiegoSan Diego, California, USA                                                    |                                                                                                                                    |
| donator | Alexander Vergara (vergara '@' ucsd.edu)<br/>Jordi Fonollosa (fonollosa '@'ucsd.edu)<br/>Irene Rodriguez-Lujan (irrodriguezlujan '@' ucsd.edu)<br/>Ramon Huerta (rhuerta '@' ucsd.edu) |                   |
| URL                    | https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations                                                                                     |
| time granularity       | 6 hours                                                                                                                                                                                |
| dimensions     | M=100  N=1000                                                                                                                                                                          |
| remarks                | only batch 10 is taken from the dataset                                                                                                                                                |



### Sample Plots


![Drift dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/drift/01_drift-rawdata-NxM_graph.jpg)
![Drift dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/drift/03_drift-rawdata01x400_graph.jpg)


<br /><hr /><br />



## EEG-Alcohol

The EEG-Alcohol dataset, owned by Henri Begleiter [EEG dataset](https://kdd.ics.uci.edu/databases/eeg/eeg.data.html), is utilized in various studies such as [Statistical mechanics of neocortical interactions: Canonical momenta indicatorsof electroencephalography](https://link.aps.org/doi/10.1103/PhysRevE.55.4578).
It describes an EEG database composed of individuals with a genetic predisposition to alcoholism.
The dataset contains measurements from 64 electrodes placed on subject's scalps which were sampled at 256 Hz (3.9-msec epoch) for 1 second.
The dataset contains a total of 416 samples.
The specific subset used in ImputeGAP is the S2 match for trial 119, identified as `co3a0000458.rd`.
The dataset's dimensions are 64 series, each containing 256 values.
This dataset is primarily used for the analysis of medical and brain-related data, with a focus on detecting predictable patterns in brain wave activity.


### Summary

| Data info          | Values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name       | eeg-alcohol                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| codename   | co3a0000458.rd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| name       | EEG Database: Genetic Predisposition to Alcoholism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| URL                | https://kdd.ics.uci.edu/databases/eeg/eeg.data.html                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
 | specific URL       | http://kdd.ics.uci.edu/databases/eeg/eeg_full.tar                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                            |
| source             | UCI KDD Archive<br/>Henri Begleiter<br/>Neurodynamics Laboratory<br/>State University of New York Health Center<br/>Brooklyn, New York                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 
| Articles           | L. Ingber. (1997). Statistical mechanics of neocortical interactions: Canonical momenta indicators of electroencephalography. Physical Review E. Volume 55. Number 4. Pages 4578-4593.<br/><br/>L. Ingber. (1998). Statistical mechanics of neocortical interactions: Training and testing canonical momenta indicators of EEG. Mathematical Computer Modelling. Volume 27. Number 3. Pages 33-64.<br/><br/>J. G. Snodgrss and M. Vanderwart. (1980). "A standardized set of 260 pictures: norms for the naming agreement, familiarity, and visual complexity." Journal of Experimental Psychology: Human Learning and Memory. Volume 6. Pages 174-215. |
| Time granularity   | 1 second per measurement (3.9 ms epoch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| trials             | 120 trials                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| channels           | 64 channels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| samples            | 416 samples (368 post-stim samples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| time resolution    | 3.906 ms uV                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| specific trial     | S2 match, trial 119                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                       |
| size | M=64 N=256  electrodes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |



### Sample Plots

![EEG-ALCOHOL dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-alcohol/01_eeg-alcohol-rawdata-NxM_graph.jpg)
![EEG-ALCOHOL dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-alcohol/03_eeg-alcohol-rawdata01x400_graph.jpg)



<br /><hr /><br />





## EEG-READING

The EEG-Reading dataset, created by the DERCo, is a collection of EEG recordings obtained from participants engaged in text reading tasks [A Dataset for Human Behaviour in Reading Comprehension Using {EEG}](https://www.nature.com/articles/s41597-024-03915-8).
This corpus includes behavioral data from 500 participants, as well as EEG recordings from 22 healthy adult native English speakers.
The dataset features a time resolution of 1000 Hz, with time-locked recordings from -200 ms to 1000 ms relative to the stimulus onset.
The dataset consists of 564 epochs, although only one was selected for this specific EEG subset.
The extracted dataset contains 1201 values across 33 series.


### Summary

| Data info          |                                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| codename   | eeg-reading                                                                                         |
| name       | DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG                             |
| URL                | https://doi.org/10.17605/OSF.IO/RKQBU                                                               |
| specific URL       | https://osf.io/tu4zj                                                                               |
| source             | DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG                             |
| article            | https://www.nature.com/articles/s41597-024-03915-8<br/>Boi Mai Quach, Cathal Gurrin & Graham Healy  |
| time granularity   | 1000.0 Hz                                                                                           |
| t                  | -200.00 ...    1000.00 ms                                                                           |
| epoch              | 1 used on 564                                                                                       |
| size | M=33 N=1201                                                                                         |



### Sample Plots

![EEG-READING dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-reading/01_eeg-reading-rawdata-NxM_graph.jpg)
![EEG-READING dataset - raw data 20x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-reading/02_eeg-reading-rawdata20x400_graph.jpg)
![EEG-READING dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-reading/03_eeg-reading-rawdata01x400_graph.jpg)





<br /><hr /><br />






## Elctricity

The elctricity dataset has data on household energy consumption of 370 individual clients collected every minute between 2006 and 2010 in France (obtained from the UCI repository.


### Summary

| Dataset info          |                                                                                          |
|--------------------|------------------------------------------------------------------------------------------|
| codename   | electricity                                                                              |
| name       | ELECTRICITY                                                                              |
| source     | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014                  | 
| creator	 | Artur Trindade, artur.trindade '@' elergone.pt <br> Elergone, NORTE-07-0202-FEDER-038564 
| granularity        | 15 minutes                                                                               |
| size | M=20 N=5000                                                                              |



### Sample Plots

![ELECTRICITY dataset - raw data 20x5000](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/electricity/01_electricity_M.jpg)
![ELECTRICITY dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/electricity/03_electricity_1.jpg)





<br /><hr /><br />





## fMRI-Objectviewing

The fMRI-Objectviewing dataset was obtained from the OpenfMRI database, with the accession number ds000105. this dataset is an extraction of a fMRI scan of Visual object recognition. This scan measures neural responses, as reflected in hemodynamic changes, in six subjects (five female and one male). The stimuli consisted of gray-scale images depicting faces, houses, cats, bottles, scissors, shoes, chairs, and abstract (nonsense) patterns. Twelve time series datasets were collected for each subject. Each time series began and ended with 12 seconds of rest and included eight stimulus blocks, each lasting 24 seconds and corresponding to one of the stimulus categories. These blocks were separated by 12-second rest intervals. Stimuli were presented for 500 milliseconds with an interstimulus interval of 1500 milliseconds.

One hypothesis for converting this data into a two-dimensional time series involved using voxels as individual series, with their corresponding values serving as the data points. Based on this approach, the fMRI-OBJECTVIEWING dataset was extracted from the first run of subject 1. Voxels with values of 0 were removed, and the total number of voxels was reduced to 10,000 after dimensional flattening, resulting in a dataset consisting of 10,000 series, each containing 121 values.

The fMRI-OBJECTVIEWING dataset showcases brain activity in regions such as the inferior temporal cortex and the occipital lobe, illustrating distinct neural activation patterns associated with visual object recognition. 

### Summary

| Dataset info          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codename   | fmri-objectviewing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| name       | Visual object recognition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| URL                | https://www.openfmri.org/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
 | specific URL       | https://www.openfmri.org/dataset/ds000105/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| source             | OpenfMRI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| article            | Haxby, J.V., Gobbini, M.I., Furey, M.L., Ishai, A., Schouten, J.L., Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539):2425-30<br/>Hanson, S.J., Matsuka, T., Haxby, J.V. (2004). Combinatorial codes in ventral temporal lobe for object recognition: Haxby (2001) revisited: is there a "face" area? Neuroimage. 23(1):156-66<br/>O'Toole, A.J., Jiang, F., Abdi, H., Haxby, J.V. (2005). Partially distributed representations of objects and faces in ventral temporal cortex. J Cogn Neurosci, 17(4):580-90 |
| Time granularity   | 500ms                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| epoch              | 1 used on 36                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| size | M=121 N=10000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |



### Sample Plots

![fMRI-OBJECTVIEWING dataset - raw data 360x121](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-objectviewing/01_fmri-objectviewing-rawdata-NxM_plot.jpg)
![fMRI-OBJECTVIEWING dataset - raw data 20x121](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-objectviewing/02_fmri-objectviewing-rawdata20x121_plot.jpg)
![fMRI-OBJECTVIEWING dataset - raw data 01x121](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-objectviewing/03_fmri-objectviewing-rawdata01x121_plot.jpg)


<br /><hr /><br />



## fMRI-Stoptask

The fMRI-Stoptask dataset was obtained from the OpenfMRI database, with the accession number ds000007. This dataset is an extraction of a fMRI scan of Visual where subjects performed a stop-signal task with one of three response types: manual response, spoken letter naming, and spoken pseudo word naming.
Following the same conversion hypothesis as used for the object recognition dataset, the fMRI-Stoptask dataset was extracted from the first run of subject 1. Voxels with values of 0 were removed, and the total number of voxels was reduced to 10,000 after flattening the dimensions. This resulted in a dataset comprising 10,000 series, each containing 182 values.
The fMRI-Stoptask dataset will emphasize brain activity in regions such as the right inferior frontal gyrus and the basal ganglia, illustrating neural mechanisms of inhibition commonly associated with stop-signal tasks.


### Summary

| Dataset info          |                                                                                                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codename   | fmri-objectviewing                                                                                                                                                          |
| name       | Visual object recognition                                                                                                                                                   |
| URL                | https://www.openfmri.org/                                                                                                                                                   |
| specific URL       | https://www.openfmri.org/dataset/ds000007/                                                                                                                                  |
| source             | OpenfMRI                                                                                                                                                                    |
| article            | Xue, G., Aron, A.R., Poldrack, R.A. (2008). Common neural substrates for inhibition of spoken and manual responses. Cereb Cortex, 18(8):1923-32. doi: 10.1093/cercor/bhm220 |
| epoch              | 1 used on 120                                                                                                                                                               |
| size | M=182 N=10000                                                                                                                                                               |




### Sample Plots

![fMRI-STOPTASK dataset - raw data 20x182](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-stoptask/02_fmri-stoptask-rawdata20x182_plot.jpg)
![fMRI-STOPTASK dataset - raw data 01x182](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-stoptask/03_fmri-stoptask-rawdata01x182_plot.jpg)

<br /><hr /><br />


## Economy

This economy dataset is used for evaluating downstream forecasting models. It exhibits a seasonality of 7 and consists of 16 time series, each containing 931 values.


### Summary

| Dataset info          |                                                                                          |
|--------------------|------------------------------------------------------------------------------------------|
| codename   | forecast-economy                                                                         |
| name       | ECONOMY                                                                              |
| source     | https://zenodo.org/records/14023107                                                      | 
| size | M=16 N=931                                                                               |




### Sample Plots

![FORECAST-ECONOMY dataset - raw data M](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/forecast-economy/forecast-economy_M.jpg)
![FORECAST-ECONOMY dataset - raw data 1](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/forecast-economy/forecast-economy_1.jpg)




<br /><hr /><br />



## Meteo

The MeteoSwiss dataset, kindly provided by the Swiss Federal Office of Meteorology and Climatology [http://meteoswiss.admin.ch], contains weather time series recorded in different cities in Switzerland from 1980 to 2018. The MeteoSwiss dataset appeared in [[1]](#ref1).

### Summary

| Dataset info          |                                                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| codename   | meteo                                                                                                                       |
| name       | Meteo Suisse data                                                                                                           |
| source     | Federal Office of Meteorology and Climatology, MeteoSwiss<br/>Operation Center 1<br/>Postfach 257<br/>8058 Zürich-Flughafen | 
| granularity        | 10 minutes                                                                                                                  |
| size | M=20 N=10000                                                                                                                |
|                    | TBA                                                                                                                         |

### Sample Plots

![Meteo dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/meteo/01_meteo-rawdata-NxM_graph.jpg)
![Meteo dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/meteo/03_meteo-rawdata01x400_graph.jpg)




<br /><hr /><br />




## Motion

The motion dataset consists of time series data collected from accelerometer and gyroscope sensors, capturing attributes such as attitude, gravity, user acceleration, and rotation rate [[4]](#ref4). Recorded at a high sampling rate of 50Hz using an iPhone 6s placed in users' front pockets, the data reflects various human activities. While the motion time series are non-periodic, they display partial trend similarities.

### Summary

| Data info          |               |
|--------------------|---------------|
| codename   | motion        |
| name       | Motion        |
| source     |               | 
| granularity        |               |
| size | M=20 N=10000  |


### Sample Plots

![Motion dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/motion/01_motion_M.jpg)
![Motion dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/motion/03_motion_1.jpg)




<br /><hr /><br />




## Soccer

The soccer dataset, initially presented in the DEBS Challenge 2013 [[3]](#ref3), captures player positions during a football match. The data is collected from sensors placed near players' shoes and the goalkeeper's hands. With a high tracking frequency of 200Hz, it generates 15,000 position events per second. Soccer time series exhibit bursty behavior and contain numerous outliers.

### Summary

| Dataset info          |                                    |
|--------------------|------------------------------------|
| codename   | soccer                             |
| name       | Soccer                             |
| source     | Grand Challenges                   | 
| source     | https://debs.org/grand-challenges/ |
| size | M=10 N=501674                      |


### Sample Plots
![Soccer dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/soccer/01_soccer_M.jpg)
![Soccer dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/soccer/03_soccer_1.jpg)



<br /><hr /><br />




## Temperature

Add description here


### Summary

| Dataset info          |                                       |
|--------------------|---------------------------------------|
| codename   | temperature                           |
| name       | Temperature                           |
| URL     | http://www.cma.gov.cn                 | 
| source     | China Meteorological Administration   |
| granularity        | daily                                 |
| size | M=428 N=19358                         |


### Sample Plots

![Temperature dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/temperature/01_temperature_20.jpg)
![Temperature dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/temperature/03_temperature_1.jpg)



<br /><hr /><br />




## References

<a name="ref1"></a>
[1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

[2] Ines Arous, Mourad Khayati, Philippe Cudré-Mauroux, Ying Zhang, Martin L. Kersten, Svetlin Stalinlov: RecovDB: Accurate and Efficient Missing Blocks Recovery for Large Time Series. ICDE 2019: 1976-1979

[3] Christopher Mutschler, Holger Ziekow, and Zbigniew Jerzak. 2013. The DEBS  2013 grand challenge. In debs, 2013. 289–294

[4] Mohammad Malekzadeh, Richard G. Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile Sensor Data Anonymization. In Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI ’19). ACM,  New York, NY, USA, 49–58. https://doi.org/10.1145/3302505.3310068
