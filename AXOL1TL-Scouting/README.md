# MLoPs for ML Trigger at 40 MHz

## Introduction

The CMS experiment at CERN records data at a rate of approximately 40 Terabytes per second. This level of information cannot be stored in storage systems. The amount of data is on such a scale that even online analysis of it on CPU and GPU farms is not possible. CMS records data at a frequency of 40 MHz (40 million events every second). To solve this problem, an elaborate trigger system exists that filters the data from the staggering rate of 40 MHz down to as small as 1kHz. This is mainly done in two stages:

1. Level 1 Trigger (L1T): This is the first trigger that operates mostly on Field Programmable Gate Array (FPGAs) and is responsible for reducing the rate from 40 MHz to 100 kHz.
2. High Level Trigger (HLT): This is the second trigger that operates on CPU and GPU farms and uses a complete physics analysis (reconstruction) to make the trigger decision. HLT reduces the data rate from 100 kHz to 1 kHz.

![](https://codimd.web.cern.ch/uploads/upload_5f3e7294f0b11652a2772904d721d8d2.jpg)

The L1T comprises a combination of various trigger subsystems. One such system is the AXOL1TL, which stands for Anomaly eXtraction Online L1 Trigger Lightweight in the L1 Global Trigger. AXOL1TL is a completely ML-based trigger that is designed to trigger events that are unbiased from the physical intuitions used to build the other trigger systems, such that it can detect new exotic signatures. For more details regarding the ML side of the trigger and its associated development, please refer to this link.

The CMS detector goes through recalibration every few months. This necessitates the deployment and development of models each time the detector is recalibrated. Moreover, machine learning models can often start to diverge when the incoming input statistics change due to progressive changes in detector condition. All these factors reduce the lifespan of any ML-based trigger, and hence, such triggers require frequent redeployment. In this report, we will cover some MLoPs and infrastructures that we have developed over the last year to ease the task of model redeployment and eventually pave the way towards continual learning and fully autonomous trigger systems.

## Recipe for an ML Trigger at 40 MHz

Before we delve deep into the infrastructure needed to develop such systems, we need to understand the overall high-level process of building and deploying such a trigger.

![](https://codimd.web.cern.ch/uploads/upload_04c8d2ddb1891a84b312355e85aedb6e.png)

1. Dataset Curation: For new training run curation, new data is necessary, given that the data distribution of the detector changes after every new calibration. The data curation consists of two steps:
    * Zerobias: The zerobias (ZB) refers to the dataset that is directly collected from the detector. This dataset undergoes no trigger or physics selection, making it completely independent of the trigger components and unbiased.
    * Monte Carlo: To test the performance of the trigger, various exotic physics processes are simulated using the Monte Carlo (MC) simulation. This set is used only during validation and not for training. MC is difficult to acquire due to the associated computational cost. Hence, we treat MC slightly differently when it comes to MLOps. 
2. Model Training: Model Training step consists of treating the acquired ZB and MC to make it ML-friendly. Following this, the model is trained following a fixed recipe that we developed through separate testing. The models are trained in PyTorch with quantisation-aware training. The hyperparameters are optimized using [Optuna](https://github.com/optuna/optuna) and [ASHA](https://arxiv.org/abs/1810.05934), and the overall distributed setup is managed by SLURM and RAY.
3. Deployment: The model deployment stage consists of taking the trained weights from PyTorch and building the model using HLS4ML, which converts into Verilog. This is done on a hardware machine. After this, the efficiencies are matched for the hardware model and the original one. Given a green signal from the team and the trigger committee, we deploy it. This step has been intentionally kept manual to maintain a degree of freedom in triggering.
4. Divergence Tracker: The final step can be considered as an entirely separate system, too. The task of the divergence tracker, also known as the Scouting Analyser, is to continually stream data from the detector and compare the model output statistics over time. This system helps in knowing when the model starts to perform poorly and a new deployment is needed.

During my Technical Studentship, I primarily worked on developing steps 1 (refactoring), 2 (complete refactoring and redevelopment), and 4 (built from scratch). The deployment part is still mostly manual, but given that the most time-consuming steps, being steps 1 and 2, are now automated, the overall time to build a new trigger and have it deployed has reduced from a few weeks to a couple of days. This report will skip steps 1 and 3 of the pipeline, as it contains numerous internal components and code related to the experiment that have not yet been approved for public release.

## Model Training
![](https://codimd.web.cern.ch/uploads/upload_d4b9e3cdc2bc1206cbe2cd9c7bab66a3.png)


The code associated with the above setup can be found [here](https://github.com/dc250601/TechStudent24/tree/main/AXOL1TL-Training).

The model training schematic is shown above. The overall training process, along with hyperparameter tuning, can be broken down into a few successive steps. 

1. For all sorts of model training, we have a configuration.yml that we use to set up everything.
2. The [data_util](https://github.com/dc250601/TechStudent24/tree/main/AXOL1TL-Training/axo/data_util) subrepository handles the data processing and prepares it for training. The raw data is read from the CERN storage, then processed and stored as a temporary H5 file that will be used for training.
3. Once the data preparation is complete and a suitable training method is found from the config, the associated [recipe](https://github.com/dc250601/TechStudent24/tree/main/AXOL1TL-Training/axo/recipies) is launched. This recipe calls several RAY servers that start the training process.
4. The RAY cluster begins the hyper-parameter optimisation sampling, which Optuna controls, and the early stopping criteria are managed by ASHA. This RAY cluster can be set up in two ways.
    - Using SLURM to schedule large-scale search and training runs. The entry point to these codes are through [1](https://github.com/dc250601/TechStudent24/blob/main/AXOL1TL-Training/axo/job_slurm_cvae.sh) and [2](https://github.com/dc250601/TechStudent24/blob/main/AXOL1TL-Training/axo/job_slurm_lvae.sh)
    - Using standalone compute nodes. The entry point in this case is simply the recipe file itself.
5. After completion of the training run and selection of the best model, the pipeline moves toward creating various plots and physics analysis. Once all the analysis is complete, the pipeline begins creating an HTML report that provides details of the entire run. This report is handy and helps with rapidly assessing the model's performance. The entire code for this step is [here](https://github.com/dc250601/TechStudent24/blob/main/AXOL1TL-Training/axo/utilities/display.py).
6. Finally, the pipeline starts to put all these together into an H5 file. This includes all configurations, training data, Monte Carlo results, model history, and plots. The H5 is stored for future use and is also picked up by the deployment stage.

## Divergence Tracker
![](https://codimd.web.cern.ch/uploads/upload_9f1ea72bf22ac163f87658bc159e4560.png)

The diagram shows the schematic of the Scouting Analyser System. This is the same system used to track divergences in online anomaly detection performance.

### Processing Pipeline

Input FIFO: The data for our pipeline comes from the L1 Scouting System. The L1 Scouting system streams a subset of the complete data that passes through the L1 Trigger and stores it on tape for future analysis. Before passing it to tape, the system temporarily stores it on the CERN storage disk, which can be accessed readily. Given the massive rate of this data, ~1 TB every hour when the system is running. The system stores data for only 24 to 48 hours on the disk. Hence, this system acts like a FIFO channel.

#### Stage 0 (Data Ingestion)

The purpose of Stage 0 is to Ntuple the raw root files into a suitable format after reading them from the temporary storage disk. Moreover the first stage directly writes the Ntulpes to `/dev/shm/StoreStage0/` to prevent IO bottlenecks down the line. This stage operates with 64 threads, all of which access a unique root file according to the FIFO schema. The benefit of using 64 threads in parallel for the ntuplising part is two-fold:
1. The read write speed of a single thread on the CERN files system remains capped making it necessary to use multiple threads to prevent information bottleneck and make it possible to transfer ~1Tb data per hour.
2. The Ntuplising step is extremely slow, although written in C++, the step handles a major chunk of data pre-processing. Moreover, this step filters much redundant event-level information, making it the slowest and the heaviest step in the entire pipeline.
Each '.root' file produced in this step contains ~70 million events and 23.31 seconds of data taking, or 1 Lumisection of data.

#### Stage 1 Data Pre-processing (64 Threads)
- Input: ROOT files from `/dev/shm/StoreStage0/`
- Output: Zarr format files in `/dev/shm/StoreStage1/`
 This step is executed by 32 parallel threads, all of which access the same root file located in `/dev/shm/StoreStage0/`. This stage reads and extracts the hardware values of the L1 trigger objects (Muons, E/Gamma, Jets, Energy Sums) and pre-processes them. Additionally, it stores the associated metadata for every event it processes for future use. All 32 threads write in parallel to 32 different zarr files into `/dev/shm/StoreStage1/`.
 
 
#### Stage 2: Model Inference (16 Threads + 1 GPU)
- Input: Zarr files from Stage 1
- Output: Zarr files with AXOL1TL scores in `/dev/shm/StoreStage2/`
This step runs on 16 CPU threads, all of which share a single H100 GPU. Each of these threads loads the quantized Keras model. Following this, it reads from the zarr files created in stage 1 and runs inference on them. The anomaly scores calculated in this step are stored in Zarr format again in `/dev/shm/StoreStage2/`.

#### Stage 3: Histogram Generation (8 Threads)
- Input: Score files from Stage 2
- Output: Histogram shards in `/dev/shm/StoreStage3/`
This stage operates on a single thread, where it reads the Zarr files from the previous step that contain the anomaly scores and events, and creates bins with a range of [0, 20000] and a width of 1. Following this, it converts the histogram to sparse format (only non-zero bins are stored) and stores it in a text-encoded file with the suffix `.axo_shards`. The metadata for the range of events associated with the histogram is also stored in this shard.

#### Stage 4: Aggregation & Storage
- Input: Histogram shards from Stage 3
- Output: Final histograms in `/eos/project/c/cms-axol1tl/Pipeline/Scouting/ScoreStream`
This is again a process running on a single thread. This process scans the memory to find all the available shards and merges the 32 shards that belonged to the same lumisection. This also detects and removes duplicate shards that may have occurred due to racing conditions or failures in the inference step. The 32 shards of a single lumisection are merged into a histogram, which is stored in a sparse format with text encoding, using a '.axo' suffix, along with associated UTC timestamps.

#### Tracker: Resource Monitoring
The tracker monitors the memory and disk usage across the different stages

### 50 Hz Rate Monitoring

One important purpose of the above system is to monitor the raw rate at 50 Hz, which is the pure rate. The raw rate is the number of events that the trigger is allowed to pass through. The Pure rate can be though of the number of events that AXOL1TL uniquely pass and these events do not belong to any other trigger in the L1 Trigger System. Given that there is a lot of overlap between different triggers, it is not uncommon to find a very high raw rate (number of events passing through the trigger) for a nominal pure rate (number of *unique* events passing through the trigger). Even though the pure rate of 50 Hz has a sufficiently large associated raw rate, the thresholds to change significantly with undetected conditions, given that the L1 rate is close to kHz, which is orders of magnitude larger than this rate. The 50 Hz Rate and its thresholds are regularly monitored, and having these numbers as close as possible to real trigger conditions is important for the proper functioning of the trigger. This system gives a window where the 50 Hz rate numbers and their associated thresholds can be easily calculated by tapping the output of StoreStage2.

### Retrain Trigger

Continual learning is something that we started our discussion with. For continual learning to be effective, it requires a trigger that activates the model retraining and redeployment procedures. This can be easily done within this system by monitoring the statistics it produces. Moreover, the retrain trigger has to curate the zero-bias for retraining the trigger. This is something that StoreStage1 handles, where the processed files are stored.

### Large Scale Statistics
When working with anomaly detection, look into the tails of the anomaly distribution. To investigate this statistically rare region, one often needs many statistics, which can take a considerable amount of time to gather. Using the outputs of this system, statistics of ~1 to ~10 billion points are easily achievable, making studies on the tails of the anomaly distribution more effective.

![](https://codimd.web.cern.ch/uploads/upload_d6075cf7ae12a8c84407614651e02e23.png)
The above heatmap illustrates the distribution of the anomaly score on the x-axis and its variation across different lumisections or time steps.

![](https://codimd.web.cern.ch/uploads/upload_b02cab90618f34fd57f2f726cfd128ee.png)
The above plot shows the variation of the raw rate with different lumisections.