# MLoPs for ML Trigger at 40 MHz

## Introduction

CMS experiment at CERN records data in order of 40 Terabytes per second. This level information cannot be stored into storage systems. The amount data is in such scales that even online Analysis of it on CPU and GPU farms is not possible. CMS records data at a frequency of 40 MHz (40 million event every second). To solve this problem there exists an elaborate trigger system that filter the data from the stagering rate of 40 MHz as small as 1kHz. This is majorly done in two stages:

1. Level 1 Trigger (L1T): This is the first trigger that operates mostly on Field Programmable Gate Array (FPGAs) and is responsible for reducing the rate from 40 MHz to 100 kHz.
2. High Level Trigger (HLT): This is the second trigger that operates on CPU and GPU farms and use a complete physics analysis (reconstruction) to make the trigger decision. HLT reduces the data rate from 100 kHz to 1 kHz.

![](https://codimd.web.cern.ch/uploads/upload_5f3e7294f0b11652a2772904d721d8d2.jpg)

The L1T consists of combination of many different trigger subsystems. One such system is the AXOL1TL which stands for Anomaly eXtraction Online L1 Trigger Lightweight in L1 Global Trigger. AXOL1TL is a completely ML based trigger that is designed to trigger events that are unbiased from the physical intuitions used to build the other trigger systems such that it can detect new exotic signature. More details regarding the ML side of the trigger and its associated development can be found here.

The CMS detector goes through recalibration every few months. This makes it necessary to deploy and develop models everytime the detector is recalibrated. Moreover machine learning models can often start to diverge when the incoming input statistics changes due to progressive changes in detector condition. All these factors reduce the life of any ML based trigger and hence such triggers need frequent redeployment. In this report we will try to cover onto some MLoPs and infrastructures that we developed over the last one year to ease the task of model redeployment and eventually pave the path towards continual learning and full autonomous trigger systems.

## Recipe for a ML Trigger at 40 MHz

Before we dewelve deep into the infrastructed needed to develop such systems we need to understand the overall high level process of building and deploying such a trigger.

![](https://codimd.web.cern.ch/uploads/upload_04c8d2ddb1891a84b312355e85aedb6e.png)

1. Dataset Curation: For new training run curation new data is absolutely necessary give that the data distribution of the detector changes after every new calibration. The data curation consists of two steps:
    * Zerobias: The zerobias (ZB) refers to the dataset that is directly collected from the detector. This dataset does not go through any form of trigger or physics selection making it completely independent from the trigger components and unbiased.
    * MonteCarlo: To test the performance of the trigger various exotic physics processes are simulated using montecarlo (MC) simulation. This set is used only during validation and not for training. MC is difficult to acquire due to the associated computational cost. Hence we treat MC slightly differently when it comes to MLOps. 
2. Model Training: Model Training step consists of treating the acquire ZB and MC to make it ML friendly. Following which the model is trained following a fixed recipe that we developed through separate testing. The models are trained in PyTorch with quantisation aware training. The hyperparameters are optimised using [Optuna](https://github.com/optuna/optuna) and [ASHA](https://arxiv.org/abs/1810.05934) and the overall distributed setup is managed SLURM and RAY.
3. Deployment: The model deployment stage consists of taking the trained weights from PyTorch and building the model using HLS4ML that converts into Verilog. This is done on a hardware machine. After this set the efficiencies are matched for the harware model and the original one. Given a green signal from the team and the trigger committee we deploy it. This step has been kept intentionally manual in nature to have a degree of freedom on the trigger.
4. Divergence Tracker: The final step can be considered as an entirely separate system too. The task of divergence tracker or Scouting Analyser is to continually stream data from the detector and compare the model output statistics over time. This system helps in knowing when the model starts to perform poorly and a new deployment is needed.

During my Technical Studentship I mainly worked on developing steps 1(refactory),2 (complete refactory and redevelopment) and 4 (built from scratch). The deployment  part is still mostly manual but given the most time consuming steps which being step 1 and 2 are automated now the overall time to build a new trigger and have it deployment has reduced from a few weeks to a couple of days. This report will skip step 1 and 3 of this pipeline given it has a lot of internal components and code related to the experiment that is not exactly approved to be made public.

## Model Training
![](https://codimd.web.cern.ch/uploads/upload_d4b9e3cdc2bc1206cbe2cd9c7bab66a3.png)


The code associated with the above setup can be found [here](https://github.com/dc250601/TechStudent24/tree/main/AXOL1TL-Training).

The model training schematic is shown as above. The overall training process along with the hyperparameter tuning can be broken done into a few successive steps. 

1. For all sorts of model training we have a configuration.yml that we use to set up everything.
2.  The [data_util](https://github.com/dc250601/TechStudent24/tree/main/AXOL1TL-Training/axo/data_util) subrepository handles the data processing and prepares it for training. The raw data is read from the CERN storage, then processed and stored as a temporary H5 file that will be used for training.
3.  Once the data preparation is complete and a suitable training method is found from the config the associated [recipe](https://github.com/dc250601/TechStudent24/tree/main/AXOL1TL-Training/axo/recipies) is launched. This recipe calls a number of RAY servers that starts the training process.
4.  The RAY cluster begins the hyper-parameter optimisation sampling of which is controlled by Optuna and the early stopping criterions are managed by ASHA. This RAY cluster can be setup in two ways.
    -  Using SLURM to schedule large scale search and training runs. The entry point to these codes are through [1](https://github.com/dc250601/TechStudent24/blob/main/AXOL1TL-Training/axo/job_slurm_cvae.sh) and [2](https://github.com/dc250601/TechStudent24/blob/main/AXOL1TL-Training/axo/job_slurm_lvae.sh)
    -  Using standalone compute nodes. The entry point in this case is simply the recipe file itself.
5. After completion of the training run and selection of the best model the pipline moves toward creating various plots and physics analysis. Once all the analysis is complete the pipeline starts creating a HTML report giving the details of the entire run. This report is handy and help with rapidly deciding on the performance of the model. The entire code for this step is [here](https://github.com/dc250601/TechStudent24/blob/main/AXOL1TL-Training/axo/utilities/display.py).
6. Finally the pipeline starts to put all these together into a H5 file. This includes all the configs, the training data, MonteCarlo, model history and the plots too. The H5 is stored for future use and is also picked up by the deployment stage.

## Divergence Tracker
![](https://codimd.web.cern.ch/uploads/upload_9f1ea72bf22ac163f87658bc159e4560.png)

The diagram shows the schematic to the Scouting Analyser System. This is the same system that is used to track divergences in the online anomaly detection performance.

### Processing Pipeline

Input FIFO: The data for our pipeline comes from the L1 Scouting System. The L1 Scouting system streams a subpart of the complete data that passes through the L1-Trigger and stores is into tape for future analysis. Before passing it to tape the system temporarily stores it on CERN storage disk that can be accessed readily. Given the massive rate of this data ~1Tb evert hour when the system runs. The system only stores the data for anywhere from 24 to 48 hours on the disk. Hence this system acts like a FIFO channel.

#### Stage 0 (Data Ingestion)

The purpose of Stage 0 is to Ntuplise the raw root files into suitable format after reading it from the temporary storage disk. Moreover the first stage directly writes the Ntulpes to `/dev/shm/StoreStage0/` to prevent IO bottlenecks down the line. This stage works with 64 threads all of which access a unique root file as per the FIFO schema. The benifit for using 64 threads in parallel for the ntuplising part is two fold:
1. The read write speed of a single thread on the CERN files system remains capped making it necessary to use multiple threads to prevent information bottleneck and make it possible to transfer ~1Tb data per hour.
2. The Ntuplising step is extremely slow, although written in C++ the step handles a major chunk of data pre-processing. Moreover this step filters a lot of redundant event level information making it the slowest and the heavies step in the entire pipeline.
Each '.root' file produced in this step contains ~70 million events and 23.31s of data taking or 1 Lumisection of data.

#### Stage 1 Data Preprocessing (64 Threads)
- Input: ROOT files from `/dev/shm/StoreStage0/`
- Output: Zarr format files in `/dev/shm/StoreStage1/`
 This step is run by 32 parallel threads all of which access the same root file from `/dev/shm/StoreStage0/`. This stage reads and extract the hardware values of the L1 trigger objects (Muons, E/Gamma, Jets, Energy Sums) and preprocesse them. Along with this it also stores the associated meta-deta for every event that it processes for future use. All 32 threads write parallely to 32 different zarr files into `/dev/shm/StoreStage1/`.
 
 
#### Stage 2: Model Inference (16 Threads + 1 GPU)
- Input: Zarr files from Stage 1
- Output: Zarr files with AXOL1TL scores in `/dev/shm/StoreStage2/`
This step runs on 16 CPU threads all of which share a single H100 GPU. Each of this threads load the quantized QKeras model. Following which it reads from the zarr files created in stage 1 and runs inference on it. The anomaly scores calculated from this step are stored into zarr format again in `/dev/shm/StoreStage2/`.

#### Stage 3: Histogram Generation (8 Threads)
- Input: Score files from Stage 2
- Output: Histogram shards in `/dev/shm/StoreStage3/`
This stage runs on a single thread where it reads the zarr files from the previous step containing the anomaly scores the events and creates with bins [0, 20000]  and width 1. Following which it converts the histogram to sparse format (only non-zero bins stored) and stores it into a text encoded file with the suffix of `.axo_shards`. The meta data for the range of events associated with the histogram is also store this shard

#### Stage 4: Aggregation & Storage
- Input: Histogram shards from Stage 3
- Output: Final histograms in `/eos/project/c/cms-axol1tl/Pipeline/Scouting/ScoreStream`
This is again a process running on a single thread. This process scans the memory to find all the available shards and merges the 32 shards that belonged to same lumisection. This also detects and removes duplicate shards that might have creept up due to racing conditions or failures in the inference step. The 32 shards of a single lumisection is merged into a histogram that is stored into a sparse format with text encoding with a '.axo' suffix along with associated UTC timestamps.

#### Tracker: Resource Monitoring
The tracker monitors the memory and disk usage across the different stages

### 50 Hz Rate Monitoring

One important purpose of the above system is monitoring of raw rate at 50 Hz of pure rate. Raw rate can be though of number of events that the trigger is allowed to pass through the event. The Pure rate can be though of the number of events that AXOL1TL uniquely pass and these events do not belong to any other trigger in the L1 Trigger System. Given that there is a lot of overlap between different triggers it is not uncommon to find to have a very high raw rate (number of event passing through the trigger) for a nominal pure rate (number of *unique* event passing through the trigger). Even though the pure rate of 50 Hz has a sufficiently large associated raw rate to it still due to changing detector conditions the thresholds change quite a bit given the total L1 rate is close to a 100kHz which is orders of magnitude larger than this rate. 50 Hz Rate and its thresholds are regularly monintored and to have these numbers as close as possible to real trigger conditions is important for the proper working of the trigger. This system gives a window where the 50 Hz rate numbers and their associated thresholds can be easily calculated by tapping the output of StoreStage2.

### Retrain Trigger

Continual learning is something that we started our discussion with. For continual learning to work it needs a trigger which activates the model retrain and re-deployment procedures. This can be easily done in this system by monitoring the statistics produced by this system. Moreover the retrain trigger has to curate the zerobias for re-training the trigger this is something the StoreStage1 where the processed files.

### Large Scale Statistics
When working with anomaly detection looking into the tails of the anomaly distribution. To investigate this statistically rare region one often needs a lot of statistics that can take a lot of time to gather. Using the outputs of this system statistics of ~1 to ~10 Billion points are easily achivevable making studies on the tails of the anomaly distribution more effective.

![](https://codimd.web.cern.ch/uploads/upload_d6075cf7ae12a8c84407614651e02e23.png)
The above heatmap does the distribution of the anomaly score on the x-axis and its distribution over different lumisections or time steps.

![](https://codimd.web.cern.ch/uploads/upload_b02cab90618f34fd57f2f726cfd128ee.png)
The above plot shows the variation of the raw rate with different lumisections.