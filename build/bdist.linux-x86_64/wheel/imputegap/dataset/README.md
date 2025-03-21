<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# ImputeGAP - Datasets
ImputeGap uses several complete datasets containing different characteristics to test your implementations.


## AIR-QUALITY

This dataset, which has been sampled, defines the air quality for 10 series and 1000 values.

![AIR-QUALITY dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/airq/01_airq_m.jpg)
![AIR-QUALITY dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/airq/03_airq_1.jpg)

### Features
| Category       | Feature                                                           | Value                 |
|---------------|-------------------------------------------------------------------|-----------------------|
| Geometry      | 5-bin histogram mode                                              | 0.03961784235855337   |
| Geometry      | 10-bin histogram mode                                             | -1.091591313523553    |
| Geometry      | Proportion of high incremental changes in the series              | 0.6388638863886389    |
| Geometry      | Longest stretch of above-mean values                              | 85.0                  |
| Geometry      | Transition matrix column variance                                 | 0.005213419596354166  |
| Geometry      | Goodness of exponential fit to embedding distance distribution    | 0.06440850415658869   |
| Geometry      | Positive outlier timing                                           | -0.14859999999999995  |
| Geometry      | Negative outlier timing                                           | -0.12905              |
| Geometry      | Longest stretch of decreasing values                              | 11.0                  |
| Geometry      | Rescaled range fluctuation analysis (low-scale scaling)           | 0.52                  |
| Geometry      | Detrended fluctuation analysis (low-scale scaling)                | 0.44                  |
| Correlation   | First 1/e crossing of the ACF                                     | 16.478546570411197    |
| Correlation   | First minimum of the ACF                                          | 2                     |
| Correlation   | Histogram-based automutual information (lag 2, 5 bins)            | 0.0036854489313206226 |
| Correlation   | Time reversibility                                                | -0.005301249210534708 |
| Correlation   | First minimum of the AMI function                                 | 1.0                   |
| Correlation   | Change in autocorrelation timescale after incremental differencing| 0.02564102564102564   |
| Trend         | Wangs periodicity metric                                          | 2                     |
| Trend         | Entropy of successive pairs in symbolized series                  | 1.7205467135685308    |
| Trend         | Error of 3-point rolling mean forecast                            | 0.6999781913121067    |
| Transformation| Power in the lowest 20% of frequencies                            | 0.7325851700477723    |
| Transformation| Centroid frequency                                                | 0.050237870803258054  |


### Summary

| Data info          |                                                                                                                                        |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Dataset codename   | airq                                                                                                                                   |
| Dataset name       | Air Quality                                                                                                                            |
| Dataset source     | Saverio De Vito (saverio.devito '@' enea.it), ENEA - National Agency for New Technologies, Energy and Sustainable Economic Development | 
| Granularity        | hourly                                                                                                                                 |
| Dataset dimensions | M=10 N=1000                                                                                                                            |





<br /><hr /><br />





## BAFU

The BAFU dataset, kindly provided by the BundesAmt Für Umwelt (the Swiss Federal Office for the Environment)[https://www.bafu.admin.ch], contains water discharge time series collected from different Swiss rivers containing between 200k and 1.3 million values each and covers the time period from 1974 to 2015. The BAFU dataset appeared in [[2]](#ref2).

### Plots
The plots present a series of plots derived from the BAFU dataset, illustrating various aspects of the data and preprocessing steps.
BAFU dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
BAFU dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while BAFU dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, BAFU - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![BAFU dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/bafu/01_bafu-rawdata-NxM_graph.jpg)
![BAFU dataset - raw data 20x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/bafu/02_bafu-rawdata20x400_graph.jpg)
![BAFU dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/bafu/03_bafu-rawdata01x400_graph.jpg)

### Features

| Category       | Feature                                                            | Value                 |
|----------------|--------------------------------------------------------------------|-----------------------|
| Geometry       | 5-bin histogram mode                                               | 10.677826122412835    |
| Geometry       | 10-bin histogram mode                                              | 4.917292125971148     |
| Geometry       | Proportion of high incremental changes in the series               | 0.004720104456518018  |
| Geometry       | Longest stretch of above-mean values                               | 170406.0              |
| Geometry       | Transition matrix column variance                                  | 0.00591715976331361   |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.014639882988124284  |
| Geometry       | Positive outlier timing                                            | -0.6863011474556843   |
| Geometry       | Negative outlier timing                                            | 0.5013873729015802    |
| Geometry       | Longest stretch of decreasing values                               | 213.0                 |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.2                   |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.82                  |
| Correlation    | First 1/e crossing of the ACF                                      | 31.923359247023075    |
| Correlation    | First minimum of the ACF                                           | 60                    |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.0006100829700701747 |
| Correlation    | Time reversibility                                                 | 1.265206120624004e-06 |
| Correlation    | First minimum of the AMI function                                  | 40.0                  |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 2.542976299460889e-05 |
| Trend          | Wangs periodicity metric                                           | 3238                  |
| Trend          | Entropy of successive pairs in symbolized series                   | 1.1196301017972983    |
| Trend          | Error of 3-point rolling mean forecast                             | 0.27272853006237263   |
| Transformation | Power in the lowest 20% of frequencies                             | 0.9838423222377587    |
| Transformation | Centroid frequency                                                 | 0.03133275601505682   |


### Summary

| Data info          |                                             |
|--------------------|---------------------------------------------|
| Dataset codename   | BAFU<br/>bafu                               |
| Dataset name       | Hydrological data across multiple stations  |
| Url/source         | https://www.bafu.admin.ch/bafu/en/home.html |
| Granularity        | 30 minutes                                  |
| Observations       | spans years 1974 to 2015                    |
| Dataset dimensions | M=12 N=85203                                |




<br /><hr /><br />






## Chlorine

The Chlorine dataset originates from chlorine residual management aimed at ensuring the security of water distribution systems [Chlorine Residual Management for Water Distribution System Security](https://www.researchgate.net/publication/226930242_Chlorine_Residual_Management_for_Water_Distribution_System_Security), with data sourced from [US EPA Research](https://www.epa.gov/research).
It consists of 50 time series, each representing a distinct location, with 1,000 data points per series recorded at 5-minute intervals.
The dataset exhibits a cyclic pattern with recurring peaks, suggesting the presence of periodic characteristics.
This makes it particularly well-suited for time series imputation methods that are designed to detect and leverage seasonality in data.

### Plots
The plots present a series of plots derived from the Chlorine dataset, illustrating various aspects of the data and preprocessing steps.
Chlorine dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
Chlorine dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while Chlorine dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, Chlorine - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![Chlorine dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/chlorine/01_chlorine-rawdata-NxM_graph.jpg)
![Chlorine dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/chlorine/03_chlorine-rawdata01x400_graph.jpg)

### Features

| Category       | Feature                                                            | Value                 |
|----------------|--------------------------------------------------------------------|-----------------------|
| Geometry       | 5-bin histogram mode                                               | -0.7816940400450461   |
| Geometry       | 10-bin histogram mode                                              | -0.988389310201491    |
| Geometry       | Proportion of high incremental changes in the series               | 0.123702474049481     |
| Geometry       | Longest stretch of above-mean values                               | 970.0                 |
| Geometry       | Transition matrix column variance                                  | 0.06770833333333334   |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.24298075807561495   |
| Geometry       | Positive outlier timing                                            | -0.86082              |
| Geometry       | Negative outlier timing                                            | 0.3675600000000001    |
| Geometry       | Longest stretch of decreasing values                               | 124.0                 |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.24                  |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.32                  |
| Correlation    | First 1/e crossing of the ACF                                      | 1559.7693638923522    |
| Correlation    | First minimum of the ACF                                           | 62                    |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 1.150419457226549     |
| Correlation    | Time reversibility                                                 | -0.004961660957249964 |
| Correlation    | First minimum of the AMI function                                  | 40.0                  |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.004814305364511692  |
| Trend          | Wangs periodicity metric                                           | 122                   |
| Trend          | Entropy of successive pairs in symbolized series                   | 1.186724208570797     |
| Trend          | Error of 3-point rolling mean forecast                             | 0.08211532856608073   |
| Transformation | Power in the lowest 20% of frequencies                             | 0.9988700796218077    |
| Transformation | Centroid frequency                                                 | 0.0004793689962142944 |


### Summary

| Data info          |                                                                                                                                                                                    |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset codename   | chlorine                                                                                                                                                                           |
| Dataset name       | Chlorine data                                                                                                                                                                      |
| Url                | https://www.epa.gov/research                                                                                                                                                       |
| Source             | United States Environmental Protection Agency, EPANET<br/>Prof. Jeanne M. VanBriesen                                                                                               |
| Article            | Vanbriesen, Jeanne & Parks, Shannon & Helbling, Damian & Mccoy, Stacia. (2011). Chlorine Residual Management for Water Distribution System Security. 10.1007/978-1-4614-0189-6_11. | 
| Time granularity   | 5 minutes                                                                                                                                                                          |
| Dataset dimensions | M=50 N=1000                                                                                                                                                                        |




<br /><hr /><br />






## Climate

The Climate dataset is an aggregated and processed collection used for climate change attribution studies.
It contains observations data for 18 climate agents across 125 locations in North America [USC Melady Lab](https://viterbi-web.usc.edu/~liu32/data.html).
The dataset has a temporal granularity of 1 month, comprising 10 series with 5,000 values each.
This structure is particularly valuable for spatio-temporal modeling [Spatial-temporal causal modeling for climate change attribution](https://dl.acm.org/doi/10.1145/1557019.1557086), as it enables researchers to account for both spatial and temporal dependencies.
The dataset exhibits high variability, along with periodic or trend-like behavior and noise, making it suitable for advanced analytical techniques.

### Plots
The plots present a series of plots derived from the Climate dataset, illustrating various aspects of the data and preprocessing steps.
Climate dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
Climate dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while Climate dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, Climate - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![Climate dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/climate/01_climate-rawdata-NxM_graph.jpg)
![Climate dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/climate/03_climate-rawdata01x400_graph.jpg)


### Features

| Category       | Feature                                                            | Value                 |
|----------------|--------------------------------------------------------------------|-----------------------|
| Geometry       | 5-bin histogram mode                                               | 0.09325838586722446   |
| Geometry       | 10-bin histogram mode                                              | -0.4947305240048063   |
| Geometry       | Proportion of high incremental changes in the series               | 0.8884977699553991    |
| Geometry       | Longest stretch of above-mean values                               | 119.0                 |
| Geometry       | Transition matrix column variance                                  | 0.0002728894456633256 |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.14380405574828758   |
| Geometry       | Positive outlier timing                                            | 0.06807999999999992   |
| Geometry       | Negative outlier timing                                            | -0.20139999999999997  |
| Geometry       | Longest stretch of decreasing values                               | 10.0                  |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.7                   |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.64                  |
| Correlation    | First 1/e crossing of the ACF                                      | 3.3379314332379257    |
| Correlation    | First minimum of the ACF                                           | 6                     |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.08356735782752395   |
| Correlation    | Time reversibility                                                 | -0.019876160881000596 |
| Correlation    | First minimum of the AMI function                                  | 5.0                   |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.018867924528301886  |
| Trend          | Wangs periodicity metric                                           | 11                    |
| Trend          | Entropy of successive pairs in symbolized series                   | 1.8832855664772494    |
| Trend          | Error of 3-point rolling mean forecast                             | 0.7960326509462264    |
| Transformation | Power in the lowest 20% of frequencies                             | 0.7973876367136716    |
| Transformation | Centroid frequency                                                 | 0.4032451996154645    |


### Summary

| Data info          |                                                                         |
|--------------------|-------------------------------------------------------------------------|
| Dataset codename   | climate                                                                 |
| Dataset name       | Aggregated and Processed data collection for climate change attribution |
| Url                | https://viterbi-web.usc.edu/~liu32/data.html                            |
| Url item           | NA-1990-2002-Monthly.csv                                                |
| Time granularity   | 1 month                                                                 |
| Dataset dimensions | M=10 N=5000                                                             |





<br /><hr /><br />




## Drift
The Drift dataset comprises 13,910 measurements collected from 16 chemical sensors exposed to six different gases, with only batch 10 utilized for this dataset [Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270).
It includes information on the concentration levels to which the sensors were exposed during each measurement.
Data was collected over a 36-month period, from January 2008 to February 2011, at a gas delivery platform facility within the ChemoSignals Laboratory at the BioCircuits Institute, University of California, San Diego [On the calibration of sensor arrays for pattern recognition using the minimal number of experiments](https://www.sciencedirect.com/science/article/pii/S0169743913001937).
The dataset has a time granularity of 6 hours and consists of 100 time series, each containing 1,000 data points. This dataset is particularly valuable for testing algorithms designed to handle drift and outliers.

### Plots
The plots present a series of plots derived from the Drift dataset, illustrating various aspects of the data and preprocessing steps.
Drift dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
Drift dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while Drift dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, Drift - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![Drift dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/drift/01_drift-rawdata-NxM_graph.jpg)
![Drift dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/drift/03_drift-rawdata01x400_graph.jpg)

### Features


| Category       | Feature                                                            | Value                 |
|----------------|--------------------------------------------------------------------|-----------------------|
| Geometry       | 5-bin histogram mode                                               | -1.2468173605295707   |
| Geometry       | 10-bin histogram mode                                              | 0.8254787327262303    |
| Geometry       | Proportion of high incremental changes in the series               | 0.965619656196562     |
| Geometry       | Longest stretch of above-mean values                               | 92.0                  |
| Geometry       | Transition matrix column variance                                  | 0.0009604922097481461 |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.03029724105804939   |
| Geometry       | Positive outlier timing                                            | 0.006040000000000045  |
| Geometry       | Negative outlier timing                                            | 0.008399999999999963  |
| Geometry       | Longest stretch of decreasing values                               | 7.0                   |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.64                  |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.84                  |
| Correlation    | First 1/e crossing of the ACF                                      | 0.5531405331698586    |
| Correlation    | First minimum of the ACF                                           | 1                     |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.0030914312919998453 |
| Correlation    | Time reversibility                                                 | -0.007087613186347406 |
| Correlation    | First minimum of the AMI function                                  | 4.0                   |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 1.0                   |
| Trend          | Wangs periodicity metric                                           | 3                     |
| Trend          | Entropy of successive pairs in symbolized series                   | 2.188715240115428     |
| Trend          | Error of 3-point rolling mean forecast                             | 1.1957075487620428    |
| Transformation | Power in the lowest 20% of frequencies                             | 0.17090456872484874   |
| Transformation | Centroid frequency                                                 | 1.8841598396202843    |

### Summary

| Data info              |                                                                                                                                                                                        |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset codename       | drift                                                                                                                                                                                  |
| Dataset names          | Gas Sensor Array Drift Dataset at Different Concentrations                                                                                                                             |
| Source                 | Alexander Vergara (vergara '@' ucsd.edu)<br/>BioCircutis Institute<br/>University of California San DiegoSan Diego, California, USA                                                    |                                                                                                                                    |
| Donors of the Dataset: | Alexander Vergara (vergara '@' ucsd.edu)<br/>Jordi Fonollosa (fonollosa '@'ucsd.edu)<br/>Irene Rodriguez-Lujan (irrodriguezlujan '@' ucsd.edu)<br/>Ramon Huerta (rhuerta '@' ucsd.edu) |                                                                                                                                                                                    |
| Time granularity       | 6 hours                                                                                                                                                                                |
| Dataset dimensions     | M=100  N=1000                                                                                                                                                                          |
| Remarks                | only batch 10 is taken from the dataset                                                                                                                                                |
| Url                    | https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations                                                                                     |









<br /><hr /><br />








## EEG-ALCOHOL

The **EEG-ALCOHOL** dataset, owned by Henri Begleiter [EEG dataset](https://kdd.ics.uci.edu/databases/eeg/eeg.data.html), is utilized in various studies such as [Statistical mechanics of neocortical interactions: Canonical momenta indicatorsof electroencephalography](https://link.aps.org/doi/10.1103/PhysRevE.55.4578).
It describes an EEG database composed of individuals with a genetic predisposition to alcoholism.
The dataset contains measurements from 64 electrodes placed on subject's scalps which were sampled at 256 Hz (3.9-msec epoch) for 1 second.
The dataset contains a total of 416 samples.
The specific subset used in ImputeGAP is the S2 match for trial 119, identified as `co3a0000458.rd`.
The dataset's dimensions are 64 series, each containing 256 values.
This dataset is primarily used for the analysis of medical and brain-related data, with a focus on detecting predictable patterns in brain wave activity.

### Plots
The plots present a series of plots derived from the EEG-ALCOHOL dataset, illustrating various aspects of the data and preprocessing steps.
EEG-ALCOHOL dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
EEG-ALCOHOL dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while EEG-ALCOHOL dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, EEG-ALCOHOL - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![EEG-ALCOHOL dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-alcohol/01_eeg-alcohol-rawdata-NxM_graph.jpg)
![EEG-ALCOHOL dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-alcohol/03_eeg-alcohol-rawdata01x400_graph.jpg)



### Features

| Category       | Feature                                                            | Value                |
|----------------|--------------------------------------------------------------------|----------------------|
| Geometry       | 5-bin histogram mode                                               | 1.0100240549492727   |
| Geometry       | 10-bin histogram mode                                              | 0.48777445966067257  |
| Geometry       | Proportion of high incremental changes in the series               | 0.9014222059451871   |
| Geometry       | Longest stretch of above-mean values                               | 272.0                |
| Geometry       | Transition matrix column variance                                  | 0.002397955611566198 |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.2767301848754665   |
| Geometry       | Positive outlier timing                                            | 0.004302978515625    |
| Geometry       | Negative outlier timing                                            | -0.015869140625      |
| Geometry       | Longest stretch of decreasing values                               | 13.0                 |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.48                 |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.34                 |
| Correlation    | First 1/e crossing of the ACF                                      | 40.477031355783346   |
| Correlation    | First minimum of the ACF                                           | 5                    |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.25175574018411473  |
| Correlation    | Time reversibility                                                 | 0.014249236968944982 |
| Correlation    | First minimum of the AMI function                                  | 4.0                  |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.025423728813559324 |
| Trend          | Wangs periodicity metric                                           | 8                    |
| Trend          | Entropy of successive pairs in symbolized series                   | 1.633147167561509    |
| Trend          | Error of 3-point rolling mean forecast                             | 0.5702585744085695   |
| Transformation | Power in the lowest 20% of frequencies                             | 0.876724240984972    |
| Transformation | Centroid frequency                                                 | 0.02684466378800049  |




### Summary

| Data info          | Values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset name       | eeg-alcohol                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Dataset codename   | co3a0000458.rd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Dataset name       | EEG Database: Genetic Predisposition to Alcoholism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Url                | https://kdd.ics.uci.edu/databases/eeg/eeg.data.html                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
 | Specific URL       | http://kdd.ics.uci.edu/databases/eeg/eeg_full.tar                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                            |
| Source             | UCI KDD Archive<br/>Henri Begleiter<br/>Neurodynamics Laboratory<br/>State University of New York Health Center<br/>Brooklyn, New York                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 
| Articles           | L. Ingber. (1997). Statistical mechanics of neocortical interactions: Canonical momenta indicators of electroencephalography. Physical Review E. Volume 55. Number 4. Pages 4578-4593.<br/><br/>L. Ingber. (1998). Statistical mechanics of neocortical interactions: Training and testing canonical momenta indicators of EEG. Mathematical Computer Modelling. Volume 27. Number 3. Pages 33-64.<br/><br/>J. G. Snodgrss and M. Vanderwart. (1980). "A standardized set of 260 pictures: norms for the naming agreement, familiarity, and visual complexity." Journal of Experimental Psychology: Human Learning and Memory. Volume 6. Pages 174-215. |
| Time granularity   | 1 second per measurement (3.9 ms epoch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Trials             | 120 trials                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Channels           | 64 channels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Samples            | 416 samples (368 post-stim samples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Time resolution    | 3.906 ms uV                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Specific trial     | S2 match, trial 119                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                       |
| Dataset dimensions | M=64 N=256  electrodes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |






<br /><hr /><br />





## EEG-READING

The **EEG-READING** dataset, created by the DERCo, is a collection of EEG recordings obtained from participants engaged in text reading tasks [A Dataset for Human Behaviour in Reading Comprehension Using {EEG}](https://www.nature.com/articles/s41597-024-03915-8).
This corpus includes behavioral data from 500 participants, as well as EEG recordings from 22 healthy adult native English speakers.
The dataset features a time resolution of 1000 Hz, with time-locked recordings from -200 ms to 1000 ms relative to the stimulus onset.
The dataset consists of 564 epochs, although only one was selected for this specific EEG subset.
The extracted dataset contains 1201 values across 33 series.
The goal of this dataset is to provide an alternative perspective on neuroscience-related datasets and to enable comparisons of results from different research studies utilizing the same technology.

### Plots
The plots present a series of plots derived from the EEG-READING dataset, illustrating various aspects of the data and preprocessing steps.
EEG-READING dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
EEG-READING dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while EEG-READING dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, EEG-READING - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![EEG-READING dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-reading/01_eeg-reading-rawdata-NxM_graph.jpg)
![EEG-READING dataset - raw data 20x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-reading/02_eeg-reading-rawdata20x400_graph.jpg)
![EEG-READING dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/eeg-reading/03_eeg-reading-rawdata01x400_graph.jpg)



### Features


| Category       | Feature                                                           | Value                 |
|----------------|-------------------------------------------------------------------|-----------------------|
| Geometry       | 5-bin histogram mode                                              | -0.9695404504750353   |
| Geometry       | 10-bin histogram mode                                             | 0.8830459332128981    |
| Geometry       | Proportion of high incremental changes in the series              | 0.03237110802974915   |
| Geometry       | Longest stretch of above-mean values                              | 918.0                 |
| Geometry       | Transition matrix column variance                                 | 0.010398644752018455  |
| Geometry       | Goodness of exponential fit to embedding distance distribution    | 0.254572789901192     |
| Geometry       | Positive outlier timing                                           | 0.9746760449755458    |
| Geometry       | Negative outlier timing                                           | 0.9568144002420209    |
| Geometry       | Longest stretch of decreasing values                              | 60.0                  |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)           | 0.18                  |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                | 0.12                  |
| Correlation    | First 1/e crossing of the ACF                                     | 9.848740939179107     |
| Correlation    | First minimum of the ACF                                          | 15                    |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)            | 0.06581564166675209   |
| Correlation    | Time reversibility                                                | 0.0027164281133996667 |
| Correlation    | First minimum of the AMI function                                 | 14.0                  |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.02413793103448276   |
| Trend          | Wangs periodicity metric                                          | 27                    |
| Trend          | Entropy of successive pairs in symbolized series                  | 1.2800853946906863    |
| Trend          | Error of 3-point rolling mean forecast                            | 0.28083302897988865   |
| Transformation | Power in the lowest 20% of frequencies                            | 0.999395333262392     |
| Transformation | Centroid frequency                                                | 0.07526093240564423   |


### Summary

| Data info          |                                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| Dataset codename   | eeg-reading                                                                                         |
| Dataset name       | DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG                             |
| Url                | https://doi.org/10.17605/OSF.IO/RKQBU                                                               |
 | Specific URL       | https://osf.io/tu4zj                                                                               |
| Source             | DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG                             |
| Article            | https://www.nature.com/articles/s41597-024-03915-8<br/>Boi Mai Quach, Cathal Gurrin & Graham Healy  |
| Time granularity   | 1000.0 Hz                                                                                           |
| t                  | -200.00 ...    1000.00 ms                                                                           |
| Epoch              | 1 used on 564                                                                                       |
| Dataset dimensions | M=33 N=1201                                                                                         |




<br /><hr /><br />






## ELECTRICITY

This dataset records the electricity consumption of 370 individual points or clients. The data has already been normalized and reduced to a certain size.


![ELECTRICITY dataset - raw data 20x5000](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/electricity/01_electricity_M.jpg)
![ELECTRICITY dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/electricity/03_electricity_1.jpg)



### Features
| Category       | Feature                                                           | Value                 |
|---------------|-------------------------------------------------------------------|-----------------------|
| Geometry      | 5-bin histogram mode                                              | -0.4685485265493501   |
| Geometry      | 10-bin histogram mode                                             | 0.07152377965335521   |
| Geometry      | Proportion of high incremental changes in the series              | 0.8447984479844799    |
| Geometry      | Longest stretch of above-mean values                              | 404.0                 |
| Geometry      | Transition matrix column variance                                 | 0.0020904341971590534 |
| Geometry      | Goodness of exponential fit to embedding distance distribution    | 0.10998958730308796   |
| Geometry      | Positive outlier timing                                           | 0.00039999999999995595|
| Geometry      | Negative outlier timing                                           | 0.11043000000000003   |
| Geometry      | Longest stretch of decreasing values                              | 18.0                  |
| Geometry      | Rescaled range fluctuation analysis (low-scale scaling)           | 0.82                  |
| Geometry      | Detrended fluctuation analysis (low-scale scaling)                | 0.78                  |
| Correlation   | First 1/e crossing of the ACF                                     | 3.9130160412663173    |
| Correlation   | First minimum of the ACF                                          | 8                     |
| Correlation   | Histogram-based automutual information (lag 2, 5 bins)            | 0.12322769215345583   |
| Correlation   | Time reversibility                                                | 0.05865528561858635   |
| Correlation   | First minimum of the AMI function                                 | 7.0                   |
| Correlation   | Change in autocorrelation timescale after incremental differencing| 0.03125               |
| Trend         | Wangs periodicity metric                                          | 11                    |
| Trend         | Entropy of successive pairs in symbolized series                  | 1.766700916944433     |
| Trend         | Error of 3-point rolling mean forecast                            | 0.7001850121310101    |
| Transformation| Power in the lowest 20% of frequencies                            | 0.8771182134792012    |
| Transformation| Centroid frequency                                                | 0.2614957874348976    |



### Summary

| Data info          |                                                                                          |
|--------------------|------------------------------------------------------------------------------------------|
| Dataset codename   | electricity                                                                              |
| Dataset name       | ELECTRICITY                                                                              |
| Dataset source     | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014                  | 
| Dataset creator	 | Artur Trindade, artur.trindade '@' elergone.pt <br> Elergone, NORTE-07-0202-FEDER-038564 
| Granularity        | 15 minutes                                                                               |
| Dataset dimensions | M=20 N=5000                                                                              |


<br /><hr /><br />





## fMRI-OBJECTVIEWING

The **fMRI-OBJECTVIEWING** dataset was obtained from the OpenfMRI database, with the accession number ds000105. this dataset is an extraction of a fMRI scan of Visual object recognition. This scan measures neural responses, as reflected in hemodynamic changes, in six subjects (five female and one male). The stimuli consisted of gray-scale images depicting faces, houses, cats, bottles, scissors, shoes, chairs, and abstract (nonsense) patterns. Twelve time series datasets were collected for each subject. Each time series began and ended with 12 seconds of rest and included eight stimulus blocks, each lasting 24 seconds and corresponding to one of the stimulus categories. These blocks were separated by 12-second rest intervals. Stimuli were presented for 500 milliseconds with an interstimulus interval of 1500 milliseconds.

One hypothesis for converting this data into a two-dimensional time series involved using voxels as individual series, with their corresponding values serving as the data points. Based on this approach, the **fMRI-OBJECTVIEWING** dataset was extracted from the first run of subject 1. Voxels with values of 0 were removed, and the total number of voxels was reduced to 10,000 after dimensional flattening, resulting in a dataset consisting of 10,000 series, each containing 121 values.

The **fMRI-OBJECTVIEWING** dataset will showcase brain activity in regions such as the inferior temporal cortex and the occipital lobe, illustrating distinct neural activation patterns associated with visual object recognition. 

### Plots
The plots present a series of plots derived from the fMRI-OBJECTVIEWING dataset, illustrating various aspects of the data and preprocessing steps.
fMRI-OBJECTVIEWING dataset - raw data 360x121 shows the full raw dataset, consisting of NxM time series.
fMRI-OBJECTVIEWING dataset - raw data 20x121 provides a subset of the data, limited to 20 time series over 121 time steps, while fMRI-OBJECTVIEWING dataset - raw data 01x121 focuses on a single time series extracted from the dataset.
Finally, fMRI-OBJECTVIEWING - normalized 20x121 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x121 subset.

![fMRI-OBJECTVIEWING dataset - raw data 360x121](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-objectviewing/01_fmri-objectviewing-rawdata-NxM_plot.jpg)
![fMRI-OBJECTVIEWING dataset - raw data 20x121](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-objectviewing/02_fmri-objectviewing-rawdata20x121_plot.jpg)
![fMRI-OBJECTVIEWING dataset - raw data 01x121](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-objectviewing/03_fmri-objectviewing-rawdata01x121_plot.jpg)


### Features
| Category       | Feature                                                            | Value                    |
|----------------|--------------------------------------------------------------------|--------------------------|
| Correlation    | First 1/e crossing of the ACF                                      | 690.0367168745234        |
| Correlation    | First minimum of the ACF                                           | 1                        |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.5218313413315354       |
| Correlation    | Time reversibility                                                 | -0.0002451758280741301   |
| Correlation    | First minimum of the AMI function                                  | 3.0                      |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.0005353319057815846    |
| Geometry       | 5-bin histogram mode                                               | -0.12774354563568313     |
| Geometry       | 10-bin histogram mode                                              | -0.45252300323190653     |
| Geometry       | Proportion of high incremental changes in the series               | 0.6172318005463854       |
| Geometry       | Longest stretch of above-mean values                               | 1711.0                   |
| Geometry       | Transition matrix column variance                                  | 0.017013232514177693     |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.235589448407061        |
| Geometry       | Positive outlier timing                                            | 0.018067033976124858     |
| Geometry       | Negative outlier timing                                            | -0.07835169880624426     |
| Geometry       | Longest stretch of decreasing values                               | 8.0                      |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.88                     |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.8                      |
| Transformation | Power in the lowest 20% of frequencies                             | 0.9776606094833528       |
| Transformation | Centroid frequency                                                 | 0.00086286419318573      |
| Trend          | Wangs periodicity metric                                           | 120                      |
| Trend          | Entropy of successive pairs in symbolized series                   | 1.6230003735065441       |
| Trend          | Error of 3-point rolling mean forecast                             | 0.19307148327953894      |


### Summary

| Data info          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset codename   | fmri-objectviewing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Dataset name       | Visual object recognition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Url                | https://www.openfmri.org/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
 | Specific URL       | https://www.openfmri.org/dataset/ds000105/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Source             | OpenfMRI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Article            | Haxby, J.V., Gobbini, M.I., Furey, M.L., Ishai, A., Schouten, J.L., Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539):2425-30<br/>Hanson, S.J., Matsuka, T., Haxby, J.V. (2004). Combinatorial codes in ventral temporal lobe for object recognition: Haxby (2001) revisited: is there a "face" area? Neuroimage. 23(1):156-66<br/>O'Toole, A.J., Jiang, F., Abdi, H., Haxby, J.V. (2005). Partially distributed representations of objects and faces in ventral temporal cortex. J Cogn Neurosci, 17(4):580-90 |
| Time granularity   | 500ms                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Epoch              | 1 used on 36                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Dataset dimensions | M=121 N=10000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |






<br /><hr /><br />









## fMRI-STOPTASK

The **fMRI-STOPTASK** dataset was obtained from the OpenfMRI database, with the accession number ds000007. This dataset is an extraction of a fMRI scan of Visual where subjects performed a stop-signal task with one of three response types: manual response, spoken letter naming, and spoken pseudo word naming.
Following the same conversion hypothesis as used for the object recognition dataset, the **fMRI-STOPTASK** dataset was extracted from the first run of subject 1. Voxels with values of 0 were removed, and the total number of voxels was reduced to 10,000 after flattening the dimensions. This resulted in a dataset comprising 10,000 series, each containing 182 values.
The **fMRI-STOPTASK** dataset will emphasize brain activity in regions such as the right inferior frontal gyrus and the basal ganglia, illustrating neural mechanisms of inhibition commonly associated with stop-signal tasks.

### Plots
The plots present a series of plots derived from the fMRI-STOPTASK dataset, illustrating various aspects of the data and preprocessing steps.
fMRI-STOPTASK dataset - raw data 360x182 shows the full raw dataset, consisting of NxM time series.
fMRI-STOPTASK dataset - raw data 20x182 provides a subset of the data, limited to 20 time series over 182 time steps, while fMRI-STOPTASK dataset - raw data 01x182 focuses on a single time series extracted from the dataset.
Finally, fMRI-STOPTASK - normalized 20x182 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x182 subset.

![fMRI-STOPTASK dataset - raw data 20x182](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-stoptask/02_fmri-stoptask-rawdata20x182_plot.jpg)
![fMRI-STOPTASK dataset - raw data 01x182](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/fmri-stoptask/03_fmri-stoptask-rawdata01x182_plot.jpg)


### Features
| Category       | Feature                                                            | Value                    |
|----------------|--------------------------------------------------------------------|--------------------------|
| Correlation    | First 1/e crossing of the ACF                                      | 0.7309996478226994       |
| Correlation    | First minimum of the ACF                                           | 3                        |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.006610621171669942     |
| Correlation    | Time reversibility                                                 | -0.002092377602975993    |
| Correlation    | First minimum of the AMI function                                  | 2.0                      |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.0001343724805159903    |
| Geometry       | 5-bin histogram mode                                               | -0.9444773627468062      |
| Geometry       | 10-bin histogram mode                                              | -0.5643072060839007      |
| Geometry       | Proportion of high incremental changes in the series               | 0.948808742502175        |
| Geometry       | Longest stretch of above-mean values                               | 31.0                     |
| Geometry       | Transition matrix column variance                                  | 0.036458333333333336     |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.13675364536909898      |
| Geometry       | Positive outlier timing                                            | 0.22446581196581206      |
| Geometry       | Negative outlier timing                                            | -0.11759768009768012     |
| Geometry       | Longest stretch of decreasing values                               | 8.0                      |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.42                     |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.86                     |
| Transformation | Power in the lowest 20% of frequencies                             | 0.305435624949023        |
| Transformation | Centroid frequency                                                 | 1.3206615845703813       |
| Trend          | Wangs periodicity metric                                           | 15                       |
| Trend          | Entropy of successive pairs in symbolized series                   | 2.182136704207034        |
| Trend          | Error of 3-point rolling mean forecast                             | 1.0768460198485337       |


### Summary

| Data info          |                                                                                                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset codename   | fmri-objectviewing                                                                                                                                                          |
| Dataset name       | Visual object recognition                                                                                                                                                   |
| Url                | https://www.openfmri.org/                                                                                                                                                   |
 | Specific URL       | https://www.openfmri.org/dataset/ds000007/                                                                                                                                  |
| Source             | OpenfMRI                                                                                                                                                                    |
| Article            | Xue, G., Aron, A.R., Poldrack, R.A. (2008). Common neural substrates for inhibition of spoken and manual responses. Cereb Cortex, 18(8):1923-32. doi: 10.1093/cercor/bhm220 |
| Epoch              | 1 used on 120                                                                                                                                                               |
| Dataset dimensions | M=182 N=10000                                                                                                                                                               |







<br /><hr /><br />


## FORECAST-ECONOMY

This economic dataset is used for testing downstream forecasting. It exhibits a seasonality of 7 and consists of 16 time series, each containing 931 values.


![FORECAST-ECONOMY dataset - raw data M](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/forecast-economy/forecast-economy_M.jpg)
![FORECAST-ECONOMY dataset - raw data 1](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/forecast-economy/forecast-economy_1.jpg)



### Features
| Category       | Feature                                                           | Value                 |
|---------------|-------------------------------------------------------------------|-----------------------|
| Geometry      | 5-bin histogram mode                                              | -0.5710874806115164   |
| Geometry      | 10-bin histogram mode                                             | -0.9082987200476134   |
| Geometry      | Proportion of high incremental changes in the series              | 0.7816717019133937    |
| Geometry      | Longest stretch of above-mean values                              | 357.0                 |
| Geometry      | Transition matrix column variance                                 | 0.011316872427983538  |
| Geometry      | Goodness of exponential fit to embedding distance distribution    | 0.12664898312226522   |
| Geometry      | Positive outlier timing                                           | 0.18958109559613323   |
| Geometry      | Negative outlier timing                                           | -0.2299274973147154   |
| Geometry      | Longest stretch of decreasing values                              | 7.0                   |
| Geometry      | Rescaled range fluctuation analysis (low-scale scaling)           | 0.3                   |
| Geometry      | Detrended fluctuation analysis (low-scale scaling)                | 0.22                  |
| Correlation   | First 1/e crossing of the ACF                                     | 124.60446764082629    |
| Correlation   | First minimum of the ACF                                          | 1                     |
| Correlation   | Histogram-based automutual information (lag 2, 5 bins)            | 0.22074051585149523   |
| Correlation   | Time reversibility                                                | 0.28049126008447584   |
| Correlation   | First minimum of the AMI function                                 | 5.0                   |
| Correlation   | Change in autocorrelation timescale after incremental differencing| 0.0012224938875305623 |
| Trend         | Wangs periodicity metric                                          | 6                     |
| Trend         | Entropy of successive pairs in symbolized series                  | 1.8906454432766748    |
| Trend         | Error of 3-point rolling mean forecast                            | 0.7191953107910503    |
| Transformation| Power in the lowest 20% of frequencies                            | 0.6678786769493903    |
| Transformation| Centroid frequency                                                | 0.009203884727314454  |




### Summary

| Data info          |                                                                                          |
|--------------------|------------------------------------------------------------------------------------------|
| Dataset codename   | forecast-economy                                                                         |
| Dataset name       | ECONOMY                                                                              |
| Dataset source     | https://zenodo.org/records/14023107                                                      | 
| Dataset dimensions | M=16 N=931                                                                               |





<br /><hr /><br />







## Meteo

The MeteoSwiss dataset, kindly provided by the Swiss Federal Office of Meteorology and Climatology [http://meteoswiss.admin.ch], contains weather time series recorded in different cities in Switzerland from 1980 to 2018. The MeteoSwiss dataset appeared in [[1]](#ref1).

### Plots
The plots present a series of plots derived from the Meteo dataset, illustrating various aspects of the data and preprocessing steps.
Meteo dataset - raw data 64x256 shows the full raw dataset, consisting of NxM time series.
Meteo dataset - raw data 20x400 provides a subset of the data, limited to 20 time series over 400 time steps, while Meteo dataset - raw data 01x400 focuses on a single time series extracted from the dataset.
Finally, Meteo - normalized 20x400 demonstrates the impact of "MIN-MAX" normalization on the raw data, applied to the same 20x400 subset.

![Meteo dataset - raw data 64x256](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/meteo/01_meteo-rawdata-NxM_graph.jpg)
![Meteo dataset - raw data 01x400](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/meteo/03_meteo-rawdata01x400_graph.jpg)

### Features

| Category       | Feature                                                            | Value                  |
|----------------|--------------------------------------------------------------------|------------------------|
| Geometry       | 5-bin histogram mode                                               | 0.8059092680204243     |
| Geometry       | 10-bin histogram mode                                              | -0.9508567159430887    |
| Geometry       | Proportion of high incremental changes in the series               | 0.5634428172140861     |
| Geometry       | Longest stretch of above-mean values                               | 4048.0                 |
| Geometry       | Transition matrix column variance                                  | 0.003505040957781977   |
| Geometry       | Goodness of exponential fit to embedding distance distribution     | 0.2534374512649519     |
| Geometry       | Positive outlier timing                                            | -0.142285              |
| Geometry       | Negative outlier timing                                            | 0.095275               |
| Geometry       | Longest stretch of decreasing values                               | 28.0                   |
| Geometry       | Rescaled range fluctuation analysis (low-scale scaling)            | 0.8                    |
| Geometry       | Detrended fluctuation analysis (low-scale scaling)                 | 0.7                    |
| Correlation    | First 1/e crossing of the ACF                                      | 754.468963708409       |
| Correlation    | First minimum of the ACF                                           | 13                     |
| Correlation    | Histogram-based automutual information (lag 2, 5 bins)             | 0.6302168700823286     |
| Correlation    | Time reversibility                                                 | 0.00026558821164784945 |
| Correlation    | First minimum of the AMI function                                  | 12.0                   |
| Correlation    | Change in autocorrelation timescale after incremental differencing | 0.0023030861354214646  |
| Trend          | Wangs periodicity metric                                           | 23                     |
| Trend          | Entropy of successive pairs in symbolized series                   | 1.4082420876782868     |
| Trend          | Error of 3-point rolling mean forecast                             | 0.3127146465659483     |
| Transformation | Power in the lowest 20% of frequencies                             | 0.9795268689123781     |
| Transformation | Centroid frequency                                                 | 0.0007669903939428711  |


### Summary

| Data info          |                                                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Dataset codename   | meteo                                                                                                                       |
| Dataset name       | Meteo Suisse data                                                                                                           |
| Dataset source     | Federal Office of Meteorology and Climatology, MeteoSwiss<br/>Operation Center 1<br/>Postfach 257<br/>8058 Zürich-Flughafen | 
| Granularity        | 10 minutes                                                                                                                  |
| Dataset dimensions | M=20 N=10000                                                                                                                |
|                    | TBA                                                                                                                         |

#### Dataset description

##### Stations
| stn       | Name       | Parameter  | Data source               | Longitude/Latitude | Coordinates [km] | Elevation [m]  |
|-----------|------------|------------|---------------------------|--------------------|------------------|----------------|
| ZHUST     | Aatal Höhe | erssurs0   | Kanton Zürich; Tiefbauamt | 8°45'/47°21'       | 698612/244769    | 490            |

##### Parameters

|            | Unit   | Description                      |
|------------|--------|----------------------------------|
| erssurs0   | mm     | null                             |
| merssurs0  | Code   | Mutation information on erssurs0 |

Observation interval for hourly values, unless otherwise indicated in the parameter description: HH  = (HH-1):41 - HH:40
Example: 13 = observation period 12:41 to 13:40

<br /><hr /><br />



<br /><hr /><br />






## Motion

This dataset consists of time series data collected from accelerometer and gyroscope sensors, capturing attributes such as attitude, gravity, user acceleration, and rotation rate [[4]](#ref4). Recorded at a high sampling rate of 50Hz using an iPhone 6s placed in users' front pockets, the data reflects various human activities. While the motion time series are non-periodic, they display partial trend similarities.

![Motion dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/motion/01_motion_M.jpg)
![Motion dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/motion/03_motion_1.jpg)

### Features
| Category       | Feature                                                           | Value                 |
|---------------|-------------------------------------------------------------------|-----------------------|
| Geometry      | 5-bin histogram mode                                              | 0.8059092680204243    |
| Geometry      | 10-bin histogram mode                                             | -0.9508567159430887   |
| Geometry      | Proportion of high incremental changes in the series              | 0.5634428172140861    |
| Geometry      | Longest stretch of above-mean values                              | 4048.0                |
| Geometry      | Transition matrix column variance                                 | 0.003505040957781977  |
| Geometry      | Goodness of exponential fit to embedding distance distribution    | 0.2534374512649519    |
| Geometry      | Positive outlier timing                                           | -0.142285             |
| Geometry      | Negative outlier timing                                           | 0.095275              |
| Geometry      | Longest stretch of decreasing values                              | 28.0                  |
| Geometry      | Rescaled range fluctuation analysis (low-scale scaling)           | 0.8                   |
| Geometry      | Detrended fluctuation analysis (low-scale scaling)                | 0.7                   |
| Correlation   | First 1/e crossing of the ACF                                     | 754.468963708409      |
| Correlation   | First minimum of the ACF                                          | 13                    |
| Correlation   | Histogram-based automutual information (lag 2, 5 bins)            | 0.6302168700823286    |
| Correlation   | Time reversibility                                                | 0.00026558821164784945|
| Correlation   | First minimum of the AMI function                                 | 12.0                  |
| Correlation   | Change in autocorrelation timescale after incremental differencing| 0.0023030861354214646 |
| Trend         | Wangs periodicity metric                                          | 23                    |
| Trend         | Entropy of successive pairs in symbolized series                  | 1.4082420876782868    |
| Trend         | Error of 3-point rolling mean forecast                            | 0.3127146465659483    |
| Transformation| Power in the lowest 20% of frequencies                            | 0.9795268689123781    |
| Transformation| Centroid frequency                                                | 0.0007669903939428711 |



### Summary

| Data info          |               |
|--------------------|---------------|
| Dataset codename   | motion        |
| Dataset name       | Motion        |
| Dataset source     |               | 
| Granularity        |               |
| Dataset dimensions | M=20 N=10000  |

<br /><hr /><br />




## Soccer

This dataset, initially presented in the DEBS Challenge 2013 [[3]](#ref3), captures player positions during a football match. The data is collected from sensors placed near players' shoes and the goalkeeper's hands. With a high tracking frequency of 200Hz, it generates 15,000 position events per second. Soccer time series exhibit bursty behavior and contain numerous outliers.

![Soccer dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/soccer/01_soccer_M.jpg)
![Soccer dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/soccer/03_soccer_1.jpg)

### Features
| Category       | Feature                                                           | Value                 |
|---------------|-------------------------------------------------------------------|-----------------------|
| Geometry      | 5-bin histogram mode                                              | 0.09084722786947164   |
| Geometry      | 10-bin histogram mode                                             | -0.2583118928950434   |
| Geometry      | Proportion of high incremental changes in the series              | 0.0011092863312203406 |
| Geometry      | Longest stretch of above-mean values                              | 71757.0               |
| Geometry      | Transition matrix column variance                                 | 0.006802721088435373  |
| Geometry      | Goodness of exponential fit to embedding distance distribution    | 0.3417367925024475    |
| Geometry      | Positive outlier timing                                           | 0.1377957597962023    |
| Geometry      | Negative outlier timing                                           | 0.05898850648030396   |
| Geometry      | Longest stretch of decreasing values                              | 1096.0                |
| Geometry      | Rescaled range fluctuation analysis (low-scale scaling)           | 0.48                  |
| Geometry      | Detrended fluctuation analysis (low-scale scaling)                | 0.46                  |
| Correlation   | First 1/e crossing of the ACF                                     | 17792.5919437391      |
| Correlation   | First minimum of the ACF                                          | 5221                  |
| Correlation   | Histogram-based automutual information (lag 2, 5 bins)            | 1.1086843892176654    |
| Correlation   | Time reversibility                                                | 6.552378315312122e-06 |
| Correlation   | First minimum of the AMI function                                 | 40.0                  |
| Correlation   | Change in autocorrelation timescale after incremental differencing| 0.0006084989404959624 |
| Trend         | Wangs periodicity metric                                          | 11198                 |
| Trend         | Entropy of successive pairs in symbolized series                  | 1.1035630861406998    |
| Trend         | Error of 3-point rolling mean forecast                            | 0.01179174759304637   |
| Transformation| Power in the lowest 20% of frequencies                            | 0.9999572395164824    |
| Transformation| Centroid frequency                                                | 0.0001370695723550248 |



### Summary

| Data info          |                                    |
|--------------------|------------------------------------|
| Dataset codename   | soccer                             |
| Dataset name       | Soccer                             |
| Dataset source     | Grand Challenges                   | 
| Dataset source     | https://debs.org/grand-challenges/ |
| Dataset dimensions | M=10 N=501674                      |



<br /><hr /><br />






## Temperature

![Temperature dataset - raw data](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/temperature/01_temperature_20.jpg)
![Temperature dataset - one series](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/temperature/03_temperature_1.jpg)

### Features
| Category       | Feature                                                           | Value                 |
|---------------|-------------------------------------------------------------------|-----------------------|
| Geometry      | 5-bin histogram mode                                              | 22.045650551725167    |
| Geometry      | 10-bin histogram mode                                             | 8.743492676833597     |
| Geometry      | Proportion of high incremental changes in the series              | 0.7958665687091343    |
| Geometry      | Longest stretch of above-mean values                              | 21931.0               |
| Geometry      | Transition matrix column variance                                 | 0.0008670367268468369 |
| Geometry      | Goodness of exponential fit to embedding distance distribution    | 0.0037844314057919114 |
| Geometry      | Positive outlier timing                                           | 0.4030755233654515    |
| Geometry      | Negative outlier timing                                           | -0.572629720089644    |
| Geometry      | Longest stretch of decreasing values                              | 15.0                  |
| Geometry      | Rescaled range fluctuation analysis (low-scale scaling)           | 0.4                   |
| Geometry      | Detrended fluctuation analysis (low-scale scaling)                | 0.38                  |
| Correlation   | First 1/e crossing of the ACF                                     | 81.7405995576158      |
| Correlation   | First minimum of the ACF                                          | 183                   |
| Correlation   | Histogram-based automutual information (lag 2, 5 bins)            | 1.439425719697347e-06 |
| Correlation   | Time reversibility                                                | 0.005588686797775345  |
| Correlation   | First minimum of the AMI function                                 | 40.0                  |
| Correlation   | Change in autocorrelation timescale after incremental differencing| 0.00847457627118644   |
| Trend         | Wangs periodicity metric                                          | 365                   |
| Trend         | Entropy of successive pairs in symbolized series                  | 1.4530196005684877    |
| Trend         | Error of 3-point rolling mean forecast                            | 0.3145083996285715    |
| Transformation| Power in the lowest 20% of frequencies                            | 0.9542560901714987    |
| Transformation| Centroid frequency                                                | 0.017202605837583908  |



### Summary

| Data info          |                                       |
|--------------------|---------------------------------------|
| Dataset codename   | temperature                           |
| Dataset name       | Temperature                           |
| Dataset source     | http://www.cma.gov.cn                 | 
| Dataset source     | China Meteorological Administration   |
| Granularity        | daily                                 |
| Dataset dimensions | M=428 N=19358                         |




<br /><hr /><br />




## References

<a name="ref1"></a>
[1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

[2] Ines Arous, Mourad Khayati, Philippe Cudré-Mauroux, Ying Zhang, Martin L. Kersten, Svetlin Stalinlov: RecovDB: Accurate and Efficient Missing Blocks Recovery for Large Time Series. ICDE 2019: 1976-1979

[3] Christopher Mutschler, Holger Ziekow, and Zbigniew Jerzak. 2013. The DEBS  2013 grand challenge. In debs, 2013. 289–294

[4] Mohammad Malekzadeh, Richard G. Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile Sensor Data Anonymization. In Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI ’19). ACM,  New York, NY, USA, 49–58. https://doi.org/10.1145/3302505.3310068
