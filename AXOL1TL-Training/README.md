
# Anomaly Detection at 40 MHz

# Introduction

The Compact Muon Solenoid (CMS) Experiment at CERN produces massive amounts of data every second ($ ($ \sim \mathcal{O}$(40) TB/s), most of which is not practically stored or even analyzedanalyzed entirely. To solve this issue, several filtration steps or triggers have been developed that operate hierarchically to filter the amount of incoming data and reduce the number of events (collision events) from 40 MHz to a manageable 1 kHz. Two triggering steps achieve this:
 - Level 1 Trigger (L1T): This is the first trigger that operates mainly on Field Programmable Gate Array (FPGAs) and is responsible for reducing the rate from 40 MHz to 100 kHz. 
 - High Level Trigger (HLT): This is the second trigger that operates on CPU and GPU farms and uses a complete physics analysis (reconstruction) to make the trigger decision. HLT reduces the data rate from 100 kHz to 1 kHz.


![](https://codimd.web.cern.ch/uploads/upload_51ae471109dd49c440b017b6dbe1a1aa.jpg)

Given the massive amount of data, a rough resolution (portion of the entire data) of the events is used to make a decision for the L1T, making physics reconstruction algorithms impossible. Hence, L1T is mostly a cut-based trigger where thresholds to physics observables are set as criteria for trigger conditions. Even though L1T works exceptionally well and has been heavily developed and tuned for quite some time, it still remains heavily biased by the physical intuitions/assumptions that are used to build the L1T.

The Standard Model (SM) of physics, being one of the most successful theories in modern physics, still fails to acknowledge a lot of phenomena such as the existence of gravity as a separate entity different from the other fundamental forces, CP Violation, the existence of dark matter, etc. All these discrepancies have led to theories that try to incorporate these phenomena. For the verification of such theoretical Beyond Standard Model (BSM), the discovery of new BSM particles is absolutely necessary. Given BSM is an uncharted territory that might or might not follow the physical assumptions of the existing L1 Trigger, calls for the development of a new unbiased trigger completely from scratch that can efficiently do the job of filtering data and remain completelycompletely based on statistical properties rather than physics-based intuitions.

## AXOL1TL

Anomaly eXtraction Online L1 Trigger Lightweight in L1 Global Trigger (AXOL1TL) is a trigger based on an anomaly detection algorithm that is deployed as a part of the L1T, which takes in a subset of the Level-1 (L1) objects (measurables) and tries to assign an anomaly score for every event. Events are selected on the basis of their anomaly score according to fixed-rate budgets.

### Working of the existing baseline

![](https://codimd.web.cern.ch/uploads/upload_3cd179da8e14cdd8bde2fb5f2e4a7f2c.png)

The above diagram shows the training and deployment of the baseline (v4) model. In this report, we discuss how we can improve on this architecture to make the trigger more accurate (in detection), robust (to changing detector conditions), and interpretable (for analysis).

The above method exhibits a few problems:
1. The loss (used to train) the original Variational AutoEncoder (VAE) for anomaly detection (AD) does not align well with the final optimisation goal of the model, given that the headless (without Decoder) model no longer represents an autoencoder. To use the latent vectors (their L2 norm in this case) for anomaly detection can be challenging and unpredictable.
2. VAEs do not guarantee any form of explicit clustering in the embedding space where similar samples strictly lie close to each other. Reconstruction error of a point might hint toward it being anomalous, but the same is not always valid for points that are close to each other (or have similar L2 norm).
3. Taking the L2 norm of the latents explicitly assumes that the anomalous samples will always be strictly away from the center, but that might not always be true.

To address these problems, we modify the VAE. The modification will be discussed and developed over the course of the following few sections.

### Self-Supervised Learning, metric and clustering.

Self-Supervised Learning (SSL) is a ML paradigm that focuses on the production of information-rich feature vectors from a dataset without explicitly using any labels. To achieve this, SSL models are trained on a surrogate task where pseudo labels are used from training and the computation of loss. For this study, we primarily focus on the VICReg, a well-celebrated SSL technique.

#### VICReg

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) work by training on the surrogate task of generating embedding spaces that are invariant to specific augmentation spaces. This is achieved by using a siamese-like setup where two views of the same event are used, but both of them have been augmented by different degrees. The Euclidean distance in the embedding space between these two views is then minimised using gradient descent.

![](https://codimd.web.cern.ch/uploads/upload_b4cf6c75203f0b7c827e481f2518c359.png)
VICReg Training Procedure [Bardes et al.](https://arxiv.org/pdf/2105.04906)

Along with the MSE loss that acts across the two projection vectors Z, to ensure stability from collapse and the non-triviality of embedding space, we have the Covariance and the Variance regularisation terms. The covariance component prevents the embedding dimensions from capturing the same information, whereas the variance component prevents all embedding vectors in a batch from collapsing to a single point in the representation space.


#### Metric

When working with latent spaces we are often very interested in defining a metric on the same. The definition of metric or distance on embedding space gives us an idea of the exact nature of the embedding space. Metric on the latent space of an autoencoder is often ill defined due to its erratic nature. Even though VAEs have solved this problem to a certain extent by making the space continuous through variational sampling where two similar samples have similar embeddings due to the variational step which makes local structures easier to define but the metric definition on a global level still remains ill-defined. VICReg solves this by introducing augmentations that often bring inductive biases within the system that make the global structures clearer.

To get a stronger intuition on this idea we will take a real like example. We consider a cylindrical system ($r\cos{t}$,$r\sin{t}$,$h$) where the rotation across the axis (of the cylinder) is invariant to our use cases. We can think of it as a coordinate change. We fix the radius to 1 and the dimension of embedding space to 2 for easier visualization. The input dataset looks like:![](https://codimd.web.cern.ch/uploads/upload_524434a78c63737bdc87fe6a67a24582.png)
When we train a VAE and a VICReg encoder (same architecture) on this dataset we see the following latent space.![](https://codimd.web.cern.ch/uploads/upload_03236af6f7bce7ab16e94df853048b28.png)
In the above diagram we chose $Z \in \{-0.5,\,-0.25,\,0,\,0.25,\,0.5\}$ hence the 5 different clusters. 

As already evident VICReg does a fantastic job of getting rid of the angular component from within the data. We can use the MSE distance between the points in this case for our theta invariant task. The same is not true for VAE, it being a information preserving system cannot get rid of the angular component, making the concept of distance on out latent space ambiguous. One thing to exercise caution in this entire argument is that the distance is only relevant with respect to proper augmentations. With only gaussian blurring we won't be able to define a metric even with our VICReg setup. Removing the rotation from the augmentation space makes out metric for VICReg latent ill-defined for our task at hand where rotation along the axis does not change the system. Below we show the nature of the latent space of VICReg without the rotation augmentation![](https://codimd.web.cern.ch/uploads/upload_54df15c45a3341c5ad9123e3eedc7db9.png)

Using different augmentation strategies, we can easily encode various types of invariances within our system making VICReg an ideal choice to build an embedding space. The metric in this space being the Euclidean distance between two points that shows how the two separate events are different from each other. Moreover, within each cluster of point in the VICReg space we can expect samples that are physically invariant to one another (samples that have different representation on $R*n$ but have similar meaning physically for instance Lorentz boosted samples, rotated samples etc.). Building systems that can cluster our anomalous samples together is our first step in the task of building our unified anomaly detection framework.


## Anomaly Detection, Density and Clusters

VAEs are one of the best candidate models for anomaly detection (AD). VAEs work well AD since it fails to reconstruct the points that lie in low density region which the model has a hard time learning. The below figure represent the working of a VAE in the context of anomaly detection when trained on a dataset of 5 gaussians that are overlapping on a 2D space.
![](https://codimd.web.cern.ch/uploads/upload_ab828337b29d3e37f8a7807d6a013ab7.png)
As we can see when using the standard reconstruction loss, we can correctly identify the 5 cluster centers along with the distribution whereas the surrounding region has a higher reconstruction loss given that there are lesser number of pointers there. Although the $||z||_2$ (on the right) or $\mathbf{z}^\top \mathbf{z}$ setup without the Decoder (following the v4 architecture of AXOL1TL) works well in the vicinity of the cluster and in finding the global region of high density but fails to capture the details of the distributions.

**Out-of-the-box-distribution** VAEs as demonstrated can have good anomaly detection capacities till they face the Out-of-the-box-distribution problem in which case the VAE learns a little too well about the dataset and ends up reconstructing everything (including the anomalies) perfectly making it a bad choice for anomaly detection in those cases. This problem can get further aggrevated when the training dataset itself contains the anomalous datapoints (true in our case).

On the other hand even though we have established the paradigm of latent space creation through VICReg and making tight clusters of points we have not yet established on how we can address the problem of finding the clusters where our anomalies or rare samples lies.

## Mahalanobis Distance, Gaussian Mixtures and Anomaly detection
Self-Supervised methods such as VICReg due to their inherent nature of minimizing invariance loss over the embedding space has a tendency to create clusters. We can broadly classify the entire dataset D to be composed of three components:

1. Clusters of Nomalies: By normalies we understand the physics processes/signals that occur frequently and can be readily found using the exisiting physics based triggeres. Thes nomalies are defined on the basis of their stastistical importance and such clusters are coonsidered to have massive number of elements and strong distribution peaks. These clusters are importanct for the Standard Model studies but given that here we are trying to look for exotic signatures that are statistically outnumbered we call these standard physics processes our nomalies. We are not very interested in probing these clusters in this work given the massive amount of work that has been already done studying such processes. Lets denote them by the notation $N_i$, we can have many such physics processes all of which can be categorised into their own clusters as $N_0,N_1...N_n$
2. Cluster of Anomalies:  Given the pretext that the anomalies follow a different form of distribution pattern within the dataset given they originate from exotic physics processes we can consider these anomalies to be contained within their own clusters just like the nomalies except these cluster centers are separated from $N_i$. Given we have the assumption that these anomalies belong to exotic physics signatures hence they must be present in clusters. Absense of clusters for anomalies mean they can never be detected using our framework using certainty pointing towards a deeper problem such as poor detector resolution or poor choice of augmentation/architecture. Lets call these $A_0,..A_1,...A_a$
3. Stray points: For the sake of completion we can consider the remaining points to be sprayed across the embedding space. These points are generally statistical noise and can be anything from hadronic noises to issues in the detector. We can call all these points as $S_0,S1,....S_s$ 
4. 
Given that there does not exist a rigid definition distinguishing $S$ and $A$, and that in the absence of a well-defined embedding space or identifiable anomalies, $S$ may consist of shallow distributions or even isolated points—which are inherently difficult to model—we absorb all $S$ into $A$.


We can now say $D = \left( \bigcup_{i \in n} N_i \right) \times \left( \bigcup_{j \in J} A_j \right)$

Now, finding $\bigcup_{i \in I} N_i$ is straightforward, as these correspond to some of the most well-studied processes in high-energy physics (HEP). However, we deliberately abstain from invoking physics intuition, since our goal is to construct an unbiased system. Instead, we adopt the statistical definition of $N_i$ — treating them as the most well-populated distributions.

We take a hyper-parameter $k \in \mathrm{N}$ assuming that we have $K$ clusters of nomalies. Now we fit a gaussian mixture model (GMM) on the latent space with $k$ gaussians making sure that all our standard physics signatures get absorbed within these $k$ clusters. Now for any point $p$ on the embedding space we can find how anomalous it is by finding how far it is from these cluster centers.

Had value of $k$ been 1. We could had used the Mahalanobis distance for finding the distance of $p$ from the cluster center. For finding the distance of a point $p$ from $k$ gaussians one can use methods like [these](https://ieeexplore.ieee.org/document/818035) . Given such methods are extremely difficult to implement on FPGA board (where we want to deploy our model) we skip it. Instead we try to computer $D_i$ for $p$ where we define $D_i$ using the mahalanobis distance as

$$
D_i(\mathbf{p}) = \sqrt{(\mathbf{p} - \boldsymbol{\mu_i})^{T} \, \mathbf{\Sigma_i}^{-1} \, (\mathbf{p} - \boldsymbol{\mu_i})}
$$

**where:**

- **$\mathbf{p}$** — point or event in the embedding space whose anomaly score is being evaluated.  
- **$D_i(\mathbf{p})$** — Mahalanobis distance of point $\mathbf{p}$ from the $i^{\text{th}}$ cluster center.  
- **$\boldsymbol{\mu_i}$** — mean of the $i^{\text{th}}$ cluster in the embedding space.  
- **$\mathbf{\Sigma_i}$** — covariance matrix of the $i^{\text{th}}$ gaussian on the embedding space.n   

Now given that we are considering our embedding space to be dominated by K cluster, we can use an aggregation method to combine $D_0,D_1,...,D_K$ to get a single composite anomaly metric. One can choose different such metrics such as arithmetic mean, min-max, geometric mean etc. Here in this work, we found geometric mean outperforms the other aggregation metrics for the task at hand.
Hence we can write the anomaly score as:
$$
D(p) = \left( \prod_{i=0}^{k-1} D_i \right)^{\frac{1}{k}}
$$


![](https://codimd.web.cern.ch/uploads/upload_c891b920f606a57002313bccdfa549f6.png)

The above diagram shows the simple implementation of the model. When evaluated and compared with the exisiting baseline (v4 of AXOL1TL) over a number of monte-carlo signals that simulate possible exotic signatures we see an improvement over the board hinting strongly towards the success of the model.

| Process                                                           | Ours | V4 |
|-------------------------------------------------------------------|---------|--------|
| GluGluHToBB_M-125                                                 | 83.27   | 31.76  |
| GluGluHToGG_M-125                                                 | 8.76    | 4.01   |
| GluGluHToGG_M-90                                                  | 3.89    | 2.98   |
| GluGluHToTauTau                                                   | 3.23    | 1.99   |
| GluGluHTo2B2WtoLNu2Q                                              | 41.26   | 26.64  |
| HHToBB                                                            | 80.01   | 58.04  |
| HHHTo4B2Tau                                                       | 74.97   | 55.97  |
| VBFHToTauTau                                                      | 12.27   | 11.84  |
| VBFHTo2B                                                          | 20.96   | 13.81  |
| WToTauTo3Mu                                                       | 0.12    | 0.11   |
| ttHto2B                                                           | 81.47   | 65.25  |
| ttHto2C                                                           | 83.70   | 67.83  |

The above table shows the efficiency of detecting the above monte-carlo signal when the trigger is operated at a working frequency of 1kHz.

## VAEs with VICReg

We again go back to our discussion on VAEs as the above method though interpretable and better performing than the baseline (AXOL1TL v4) has a few shortcomings:
1. **Mahalanobis Distance for Gaussian Mixtures**: The use of GMMs along with mahalanobis distance is not well defined when it comes to defining a single composite score/distance/metric. The hacky way of using an aggregation function cannot be analytically justified and might lead to unexpected results.
2.**Computational Bottlenecks**: The best performing aggregation method for this use case is geometric mean. We arrived at this result after extensive experimentation with different aggregation methods such as arithmetic mean, thresholding, taking min or max etc. Implementation of geometric mean is extremely expensive on FPGA boards making this model good in theory but difficult to deploy in production.

The question now is whether one can design an estimator (MLP type) that can do the GMM clustering and the subsequent distance estimation. Given that we are estimating the distance from the highly populated cluster we can say that a VAE when trained on this embeddings can learn to reconstruct this strong cluster centres with more ease compared to the weaker anomalous ones. This is very different compared to using an VAE directly on top of the data which can be better understood by going back to the  toy **cylinder problem**. The intial distribution of the data on the cylinder though taken uniform (for our ease) can be though of combination of clusters. When the density estimation using VAEs is done on the input space alone there is a strong chance that the model learns the entire distribution given that the anomalies not well separated on the input space. Moreover, due to the high dimensionality the input space has a wider and more uniform distribution compared to the latent space where we see the distribution collapsing to clusters since the invariance conditions are enforced using the VICReg loss. The uniform/well-spread distribution on the input space makes it easier for the VAE to learn the entire input space defeating our intention of it only to learn the statistically significant regions. Whereas when trained on the embedding space the VAE has a hard time learning to recontruct small anomalous regions that are separated from the area of nomalies and are significantly outnumbered. The plot on the left shows our original input dataset and the one on the right shows the nature of the embedding space. 
| Input Space | Latent Space |
|---|---|
| ![](https://codimd.web.cern.ch/uploads/upload_9a4712b99bd87ea78f91e908b6e0d6e9.png) | ![](https://codimd.web.cern.ch/uploads/upload_1be4340ddf67e9ad247f50c6567e537e.png) |
The distribution on the left (input space) is therefore easier for the VAE to generalize than the one on the right (embedding space). We can enforce this idea further by training a shallow VAE for the density estimation since embeddings derived from VICReg are disentangled, simplified and smaller in dimensionality making a shallow VAE sufficient to learn the nomalies Moreover using this setup we also successfully managed to incorporate some degree of physics information (that we can control through the augmentation) and detector information into the system by forcing the invariance. This lets us to the next step where we add a VAE on top of the VICReg embedding. We follow the given recipe:

1. Train VICReg with a suitable encoder on the raw data using physics inspired and context motivated augmentations.
2. Strip away the encoder of the VICReg and freeze the weights.
3. Extract the embeddings of the VICReg space.
4. Train a simple VAE on embedding space.
5. Remove the variational part of the VAE along with the resampling step and convert it to an AE.
6. Use the reconstruction loss between the input to the VAE (now AE) and output as the anomaly score.

![](https://codimd.web.cern.ch/uploads/upload_f8184d332663ce6ecc661e90049d00f9.png)


The above setup makes it very easy to deploy the above model onto the FPGA board given it is a stack of MLPs. Moreover it outperforms the GMM+VICReg setup since the VAE backbox now manages everything. The comparison of efficiency in detecting exotic monte-carlo signals with respect to the last deployment (vanilla v4) is shown below

The efficiency numbers for the V5 (ours) and V4 (old base line) are as follows
| Signal Name                    | VICReg + GMM (ours) | V4 (Baseline)     | VICReg + VAE(ours, deployed) |
|-------------------------------|--------------------:|-------:|------------------------------:|
| GluGluHToBB_M-125              | 83.27               | 31.76  | 95.93771                       |
| GluGluHToGG_M-125              | 8.76                | 4.01   | 37.77188                       |
| GluGluHToGG_M-90               | 3.89                | 2.98   | 15.854626                      |
| GluGluHToTauTau                | 3.23                | 1.99   | 4.831854                       |
| GluGluHTo2B2WtoLNu2Q           | 41.26               | 26.64  | 53.692781                      |
| HHToBB                         | 80.01               | 58.04  | 84.302641                      |
| HHHTo4B2Tau                    | 74.97               | 55.97  | 81.811932                      |
| VBFHToTauTau                   | 12.27               | 11.84  | 26.036048                      |
| VBFHTo2B                       | 20.96               | 13.81  | 27.53418                       |
| WToTauTo3Mu                    | 0.12                | 0.11   | 0.270025                       |
| ttHto2B                        | 81.47               | 65.25  | 86.183296                      |
| ttHto2C                        | 83.7                | 67.83  | 88.221684                      |

The efficiency vs Rate plot given below. For the same budget (trigger rate independent of other triggers) both the v4 and v5 can be compared using the efficiency at the AXOL1TL medium rate.

![](https://codimd.web.cern.ch/uploads/upload_d1658051797d329cdd1c036305dc287e.png)

For more details on the Model Architecture and performance see
1. https://cds.cern.ch/record/2904695/files/DP2024_059.pdf
2. [Model Report](https://mlflow-deploy-mlflow.app.cern.ch/#/experiments/564166212411965545/runs/cfdddb53e6dc4fc6a243ae980d66348a/artifacts)

Given the scale of the project and the scale engineering associated with deploying the trigger a lot of were done on the MLOPs side too which can be referred here.