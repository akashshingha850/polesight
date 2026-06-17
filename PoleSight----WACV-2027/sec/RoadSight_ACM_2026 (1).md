59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 

1 

2 

3 

4 

5 

6 7 

8 

9 

10 

11 12 13 

14 

73 **ACM Reference Format:** 74 Akash Shingha Bappy, Siva Ariram, Timo Mäenpää, and Juha Röning. 2026. 75 ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersec76 tion Detection on Edge Devices. In _Proceedings of 34th ACM International Conference on Multimedia (MM ’26)._ ACM, New York, NY, USA, 12 pages. 77 https://doi.org/XXXXXXX.XXXXXXX 78 79 80 **1 Introduction** 81 Road intersection detection from aerial imagery is emerging in mod82 ern geospatial applications, including intelligent transportation sys83 tems, disaster response, and urban infrastructure monitoring, where 84 precise identification can improve safety and efficiency [2]. Inter85 sections are particularly safety-critical: over 50% of severe crashes 86 occur at or near intersections, making precise detection essential for 87 improving traffic safety and navigation [45]. To address these safety 88 protocol, with robustness validated via challenges, intelligent systems that can identify intersections, assess 89 their conditions, and support smart traffic management and navi90 gation. The development of such systems requires robust computer 91 vision models capable of accurate real-time detection across diverse 92 environmental conditions, including seasonal variations, different 93 lighting conditions, and weather changes that significantly affect 94 intersection visibility and characteristics [22]. The proliferation of 95 UAVs has enabled cost-effective and high-resolution data collection 96 for training such models, though challenges remain in handling 97 environmental variability and ensuring model generalization [16]. 98 → **Object detection** ; _Vision for_ This work uses RGB imagery from standard UAV cameras without 99 → _Transportation_ . additional sensors such as LiDAR, focusing on object detection 100 frameworks suitable for real-time deployment. 101 The motivation stems from the scarcity of specialized datasets 102 for road intersections in UAV imagery, hindering the development 103 of robust deep learning models for real-world scenarios [7]. Dif104 ficulties include manual annotation of diverse intersection types 105 across seasons, computational constraints for edge inference, and 106 the need for high accuracy in complex terrains. The significance lies 107 in paving the way for terrain-aware mapping systems that could 108 integrate intersection detection with 3D modeling, thereby reduc109 ing reliance on expensive surveying methods. Prior efforts remain 110 limited by dataset scale, environmental diversity, and their predom111 inant focus on general roads or vehicles rather than intersections 112 [30, 41]. Notably, existing datasets lack seasonal diversity, limiting 113 model generalization across environmental conditions. The goal 114 of this work is to create a multi-seasonal UAV intersection dataset 115 116 

15 

16 

17 18 

19 

38 39 

40 

41 

42 

43 44 45 46 

47 

48 49 

# **ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersection Detection on Edge Devices** 

Akash Shingha Bappy 

Biomimetics and Intelligent Systems Group (BISG) University of Oulu, Oulu, Finland akash.bappy@oulu.fi 

## Timo Mäenpää 

Biomimetics and Intelligent Systems Group (BISG) University of Oulu, Oulu, Finland timo.maenpaa@oulu.fi 

## Siva Ariram 

Biomimetics and Intelligent Systems Group (BISG) University of Oulu, Oulu, Finland siva.ariram@oulu.fi 

## Juha Röning 

Biomimetics and Intelligent Systems Group (BISG) University of Oulu, Oulu, Finland 

## **Abstract** 

17 Road intersections are critical nodes in transportation networks, 18 and their accurate detection from aerial imagery is essential for ap19 plications such as autonomous navigation, urban planning, and traf20 fic management. However, existing aerial object detection datasets 21 often lack intersection-specific annotations, seasonal diversity, and 22 UAV-centric design, limiting model robustness and real-time deploy23 ment on edge devices. To address these gaps, we introduce ROAD24 SIGHT (Road-Oriented Aerial Dataset for Intersection Detection on 25 Edge Hardware Platform), a UAV-captured dataset comprising high26 resolution RGB images collected under both summer and winter 27 conditions. The dataset includes 969 images (907 with target classes 28 and 62 background samples) and 1,355 expert-annotated instances 29 in two classes: roundabout and intersection. We establish baseline 30 performance by benchmarking multiple state-of-the-art detectors 31 under a unified training protocol, with robustness validated via 32 cross-validation. The results demonstrate strong detection accu33 racy and a favorable accuracy–efficiency trade-off, while enabling 34 real-time performance on resource-constrained edge devices. These 35 findings establish ROADSIGHT as a novel as well as foundational 36 dataset for season-aware UAV intersection detection, while outlin37 ing key limitations to guide future research. The dataset along with 38 supplementary materials are publicly available at this link. 

## **CCS Concepts** 

• **Computing methodologies** → **Object detection** ; _Vision for robotics_ ; • **Applied computing** → _Transportation_ . 

## **Keywords** 

Road intersection detection, Aerial imagery, UAV dataset, YOLOv11, Edge computing, Real-time detection, Computer vision. 

50 Permission to make digital or hard copies of all or part of this work for personal or **Unpublished working draft. Not for distribution.** classroom use is granted without fee provided that copies are not made or distributed 51 for profit or commercial advantage and that copies bear this notice and the full citation 52 on the first page. Copyrights for components of this work owned by others than the 53 author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, orrepublish, to post on servers or to redistribute to lists, requires prior specific permission 54 and/or a fee. Request permissions from permissions@acm.org. 55 _MM ’26, Rio de Janeiro, Brazil_ 56 © 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 57 https://doi.org/XXXXXXX.XXXXXXX 58 2026-06-17 09:20. Page 1 of 1–12. 

juha.roning@oulu.fi 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

Bappy et al. 

175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 

117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 

140 

141 

142 

143 

144 

145 

146 

147 

148 

149 150 

**Figure 1: ROADSIGHT dataset approach overview: addressing intersection detection challenges through systematic dataset creation and optimized model evaluation for real-time UAV deployment.** and evaluate edge-optimized object detection models for real-time but limited by static satellite origins, lacking UAV-specific perspecdeployment, as shown in Figure 1. Therefore, we propose a novel tives and seasonal data [43]. In contrast, our dataset uniquely targets dataset of UAV-captured images annotated for road intersections, road intersections with UAV-captured RGB images in winter and collected from localized test areas in Finland, under winter and sumsummer conditions, providing annotations for specific types (roundmer conditions, to capture seasonal variations. The methodology abouts or intersections), which enhances novelty in environmental involves data collection, preprocessing, training, and evaluation robustness and integration with real-time detection models, unlike using different YOLO (You Only Look Once) as well as RT-DETR the broader but less specialized focus of these works. (Real-Time Detection Transformer) models, followed by model optiSpecific road-related datasets address infrastructure elements but mization for edge deployment [26, 33, 38]. This method emphasizes vary in scope and platform. The UAVDT dataset includes 100 video advantages such as improved mean average precision (mAP) in clips with over 80,000 frames for vehicle detection and tracking, varying conditions, reduced inference time on edge devices with advantageous for motion analysis in traffic scenarios but limited by limited computational capability for enhanced geo-spatial analysis. its focus on vehicles and sparse intersection annotations [9]. The Roundabout Aerial Images dataset by Puertas et al. [32] contains 15,474 images from roundabouts with 985,260 instances, strong **2 Related Work** for traffic density analysis and high instance count; however, it is restricted to roundabouts and lacks seasonal variations. Another Datasets and methods for object detection in aerial imagery, parwork on satellite-based road intersection detection uses 14,692 ticularly for road infrastructure and UAV-based applications, are images to train models for route planning, benefiting urban applireviewed to contextualize our novel dataset and its evaluation. They cations with advantages in scalability from satellite data but lack of are grouped into general aerial object detection datasets and specific resolution and UAV viewpoint or environmental adaptations [10]. road-related datasets, highlighting their contributions, advantages, In contrast, our method differentiates by combining multi-season disadvantages, and differences from our approach. General aerial UAV imagery with intersection-specific annotations, evaluated for object detection datasets provide benchmarks for tasks like vehicle edge-optimized performance, offering superior adaptability and or small object identification but often lack targeted annotations specificity not found in these vehicle- or satellite-centric datasets. for road intersections. For instance, the DOTA dataset comprises In addition to real-world UAV datasets, several synthetic datasets 2,806 high-resolution aerial images with 188,282 instances across 15 have emerged that provide annotated data for aerial scene undercategories, including vehicles and bridges and roundabouts [39]. Its39]. Its]. Its standing. Such as, Syndrone [35], MidAir [12], and DDOS [23] used advantage is the large scale and multi-sensor sourcing, supporting simulation environments to generate large-scale synthetic imagery advanced detectors like rotated CNNs; however, its disadvantages with annotations for tasks such as depth estimation, obstacle avoidby not including intersection-specific classes and limited environance, and semantic segmentation. These datasets allow controlled mental diversity, leading to poor generalization in seasonal variaenvironments but lack real-world complexities such as sensor noise, tions. Similarly, the VisDrone dataset offers 10,209 images and 263 geographic diversity, and fine-grained intersection annotations, and videos from drones, with 2.5 million annotations for detection and are not designed for edge deployment. Our work, in contrast, captracking, excelling in real-world drone scenarios with advantages tures true seasonal variation (e.g., snow-covered and sunlit roads), in video sequences for dynamic analysis [44].44].]. Yet, it focuses on focuses on road intersections and benchmarks real-time edge perpedestrians and vehicles, neglecting intersections, and suffers from formance, thereby establishing itself as a complementary and novel occlusion issues in dense scenes. The iSAID dataset extends DOTA dataset for UAV-based intersection detection through its practical with instance segmentation, annotating 655,451 objects in 2,806 imemphasis on real-data fidelity and deployment constraints. 

> 151 Datasets and methods for object detection in aerial imagery, par- 

> 152 ticularly for road infrastructure and UAV-based applications, are 

> 153 reviewed to contextualize our novel dataset and its evaluation. They 

> 154 are grouped into general aerial object detection datasets and specific 

> 155 road-related datasets, highlighting their contributions, advantages, 

> 156 disadvantages, and differences from our approach. General aerial 

> 157 object detection datasets provide benchmarks for tasks like vehicle 

> 158 or small object identification but often lack targeted annotations 

> 159 for road intersections. For instance, the DOTA dataset comprises 

> 160 2,806 high-resolution aerial images with 188,282 instances across 15 

> 161 categories, including vehicles and bridges and roundabouts [39]. Its39]. Its]. Its 

> 162 advantage is the large scale and multi-sensor sourcing, supporting 

> 163 advanced detectors like rotated CNNs; however, its disadvantages 

> 164 by not including intersection-specific classes and limited environ- 

> 165 mental diversity, leading to poor generalization in seasonal varia- 

> 166 tions. Similarly, the VisDrone dataset offers 10,209 images and 263 

> 167 videos from drones, with 2.5 million annotations for detection and 

> 168 tracking, excelling in real-world drone scenarios with advantages 

> 169 in video sequences for dynamic analysis [44].44].]. Yet, it focuses on 

> 170 pedestrians and vehicles, neglecting intersections, and suffers from 

> 171 occlusion issues in dense scenes. The iSAID dataset extends DOTA 

> 172 with instance segmentation, annotating 655,451 objects in 2,806 im- 

> 173 ages across 15 categories, benefiting precise boundary delineation 

174 

2026-06-17 09:20. Page 2 of 1–12. 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersection Detection on Edge Devices 

> 233 **3 Dataset Construction** 234 

291 

292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 

241 

242 243 

244 

245 

246 

247 

248 

249 

250 

251 

252 

253 

254 

255 

256 

257 

258 

259 

260 

261 

274 

286 

287 288 

**==> picture [205 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
tq 292<br>**----- End of picture text -----**<br>


This section describes the construction of the dataset, including 235 UAV-based data collection, preprocessing, expert annotation, and 236 formatting for machine learning. The methodology ensures re237 producibility, privacy compliance, and robustness across seasonal 238 variations, addressing gaps in prior aerial datasets. 239 

## **3.1 Data Collection and Preprocessing** 

240 

Data collection was performed using a DJI Phantom 4 RTK drone equipped with a 20-megapixel RGB camera capable of capturing images at a resolution of 5472x3648 pixels. Flights were conducted at 60–90 m altitude to balance spatial coverage and resolution [4]. Flights targeted intersections (including 4-leg crossroads and 3-leg T/Y junctions) and roundabouts. A total of around 1500 raw images were captured initially during summer and winter to include diverse conditions such as snow, varying lighting, and weather. 

**Figure 2: Example annotated images showing bounding boxes for roundabouts (class 0) and intersections (class 1).** 

Flights targeted intersections (including 4-leg crossroads and 3-leg optimizing model parameters, the validation set facilitates hyperpaT/Y junctions) and roundabouts. A total of around 1500 raw images rameter tuning and early stopping, and the test set provides a final, were captured initially during summer and winter to include diverse unbiased estimate of model performance. A split ratio of 70:15:15 conditions such as snow, varying lighting, and weather. was chosen, following widely adopted conventions in computer From the initial image set, a manual filtering process was applied vision and machine learning benchmarks, where 70-80% is allocated to remove irrelevant samples (e.g., images without any class or for training and the remainder is split between validation and testimproperly exposed). However, 62 background samples (e.g., iming [20]. The split was performed using scikit-learn’s train _test ages without any classes of interest) were intentionally retained to _split function, initially separating 70% of the data for training, improve model robustness against false positives [29]. This resulted followed by an equal division of the remaining 30% between valiin a curated dataset of 969 relevant images containing clearly visidation and testing sets. This approach ensured strict independence ble intersections. No color correction was applied to preserve raw between subsets, preventing data leakage. The use of stratified aerial characteristics and images were resized to 960x640 (3:2 aspect random sampling with a fixed random seed (42) guarantees reproratio) for a balance between detail preservation and computational ducible results across experimental runs. As a result, the dataset efficiency, following common practices [34]. was divided into subsets of 678, 145 and 146 images for training, validation and testing respectively as detailed in Appendix E. **3.2 Data Annotation** Annotations were performed using the CVAT (Computer Vision **3.4 Dataset Formatting** Annotation Tool) application [37], which was deployed on a local37], which was deployed on a local], which was deployed on a local The dataset was formatted following the requirements of the YOLO machine to ensure data privacy and retain full oversight of the family of models, ensuring direct compatibility with modern trainannotation process. Two classes were defined: “roundabout” (class ing pipelines. Each image is stored in the .jpg format, and each 0) and “intersection” (class 1, encompassing T/Y and 4-leg types). corresponding label is saved as a .txt file with the same base Bounding boxes were drawn around each instance of these classes in filename. Labels follow the YOLO convention, where each line in the images following the unified boundary definitions as mentioned the file corresponds to one annotated object using the structure: in Appendix A. The annotation process was carried out, following a class_id x_center y_center width height. Here, all coordistandardized protocol and reviewed by another annotator to ensure nates are normalized to the range [0 _,_ 1] with respect to the image consistency and accuracy [3], following common quality-control3], following common quality-control], following common quality-control dimensions. For example, 0 0.5 0.5 0.25 0.25 would indicate practices used in large-scale datasets [11, 25]. To ensure consistency,11, 25]. To ensure consistency,, 25]. To ensure consistency, 25]. To ensure consistency,]. To ensure consistency, an object of class ID 0 located at the center of the image with width an annotation guideline was followed prior to labeling, and a twoand height equal to 25% of the image size [14]. 

262 Annotations were performed using the CVAT (Computer Vision 263 Annotation Tool) application [37], which was deployed on a local37], which was deployed on a local], which was deployed on a local 264 machine to ensure data privacy and retain full oversight of the 265 annotation process. Two classes were defined: “roundabout” (class 266 0) and “intersection” (class 1, encompassing T/Y and 4-leg types). 267 Bounding boxes were drawn around each instance of these classes in 268 the images following the unified boundary definitions as mentioned 269 in Appendix A. The annotation process was carried out, following a 270 standardized protocol and reviewed by another annotator to ensure 271 consistency and accuracy [3], following common quality-control3], following common quality-control], following common quality-control 272 practices used in large-scale datasets [11, 25]. To ensure consistency,11, 25]. To ensure consistency,, 25]. To ensure consistency, 25]. To ensure consistency,]. To ensure consistency, 273 an annotation guideline was followed prior to labeling, and a two274 pass review process was implemented as detailed in Appendix B. 

The dataset is organized into train/, val/, and test/ directories, each containing images/ (RGB files) and labels/ (YOLO annotations), as illustrated in Appendix F. To enable training, a configuration file (data.yaml) defines dataset paths and class information. It specifies the locations of the train, val, and test sets, along with the number and names of classes. The complete configuration is provided in Appendix G, which ensures interoperability with different YOLO versions. In practice, such standardized formatting reduces preprocessing overhead, supports reproducibility, and aligns with best practices in dataset sharing [40]. 

275 The annotation protocol was designed according to a data min276 imization principle: in addition to the raw images required for 277 intersection detection, we only store geometry level labels for road 278 features (bounding boxes and class IDs for intersections or round279 abouts) and collect no extra person or vehicle specific attributes or 280 identifiers. The workflow also omits additional per-instance meta281 data such as weather or visibility flags, per-instance occlusion tags, 282 or amodal bounding boxes, as detailed in Appendices C, D; aligning 283 with general data protection regulation (GDPR) recommendations 284 to limit processing to what is necessary for the declared research 285 purpose. Some of the annotated examples are shown in Figure 2. 

## **4 Dataset Statistics** 

This section presents key statistics of the dataset, with Table 1 summarizing image count, annotations, classes, formats, and resolution. Subsequent subsections detail class-wise and bounding box statistics, including width, height, area, and aspect ratio distributions, 

## **3.3 Data Splitting** 

Proper dataset splitting prevents overfitting and ensure unbiased evaluation of model generalization [13]. The training set is used for 2026-06-17 09:20. Page 3 of 1–12. 

289 

290 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

Bappy et al. 

407 

408 409 410 

352 

411 412 

413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 

432 

433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 

388 

400 

401 402 403 404 405 

349 **Table 1: Key attributes of the ROADSIGHT dataset, including** 350 **image statistics, annotation details, and technical specifica-** 351 **tions. Here “KB” stands for kilobytes.** 

**Table 3: ROADSIGHT dataset split distribution, detailing image counts and class-specific instance allocations across training, validation, and testing sets.** 

|353<br>354<br>355<br>356<br>357<br>358<br>359<br>360<br>361<br>362<br>363<br>364<br>365<br>366<br>367|Attribute<br>Value|
|---|---|
||<br>Total images<br>969 (907 with classes, 62 background)<br>Total annotations<br>1,355<br>Classes<br>2 (Roundabout, Intersection)<br>Image format<br>JPEG<br>Annotation format<br>YOLO (normalized coordinates)<br>Image resolution<br>960×640 pixels<br>Aspect ratio<br>3:2 (1.5:1)<br>Color space<br>RGB (Red, Green, Blue)<br>Color channels<br>3<br>Bit depth<br>8-bit per channel<br>Image size (KB)<br>68–599 (Avg: 182)|
|||



|||||**Instances**||
|---|---|---|---|---|---|
||**Split**<br>Training<br>Validation|**Image**<br>678<br>145|**Intersection**<br>540<br>112|**Roundabout**<br>402<br>89|**Total**<br>942<br>201|
||Testing|146|118|94|212|
||**Total**|**969**|**770**|**585**|**1,355**|



## **4.2 Data Split Distribution** 

3 8-bit per channel 68–599 (Avg: 182) unbiased model training and evaluation. **5 Algorithmic Analysis Mean ± Std Median Range** 0.413 ± 0.173 0.450 0.027–0.749 0.488 ± 0.212 0.437 0.092–0.950 0.223 ± 0.162 0.172 0.008–0.593 0.959 ± 0.549 0.792 0.095–3.315 0.210 ± 0.158 0.163 0.013–0.908 0.210 ± 0.114 0.191 0.026–0.737 0.055 ± 0.078 0.029 0.002–0.501 **5.1 Experimental Setup** 1.104 ± 0.673 0.990 0.082–5.031 

The dataset is partitioned into training, validation, and testing sets using a 70:15:15 ratio, as detailed in Section 3.3. Table 3 provides a breakdown of image counts and annotated instances across these splits, ensuring balanced class representation. Intersections account for 55.7-57.3% of instances, while roundabouts represent 42.7-44.3%, with consistent proportions maintained across all subsets to support unbiased model training and evaluation. **5 Algorithmic Analysis Range** This section presents a comprehensive evaluation of modern YOLO 0.027–0.749 and DETR architectures on the dataset, aiming to establish robust 0.092–0.950 benchmarks for road intersection detection. The analysis encom0.008–0.593 passes experimental setup, model selection rationale, performance 0.095–3.315 across multiple models, and edge deployment. Trade-offs between detection accuracy and computational efficiency are systematically 0.013–0.908 assessed, providing detailed metrics and feasibility insights. 0.026–0.737 0.002–0.501 **5.1 Experimental Setup** 0.082–5.031 Experiments were conducted on both a high-end workstation and a resource-constrained edge device. Model training and initial evaluation were performed on a workstation equipped with an NVIDIA GeForce RTX 2080 Ti GPU (11GB GDDR6, 4352 CUDA cores), an Intel Core i7-8700 CPU (3.20 GHz, 6 cores/12 threads), and 64 GB RAM (DDR4). The software stack included Python 3.10.12 for scripting, PyTorch 2.8.0 for deep learning, CUDA 12.8 for GPU acceleration, Ultralytics YOLO v8.3.189 for model implementation, WandB 0.21.3 for experiment tracking, and Ubuntu 22.04 as the operating system. For deployment, inference was conducted on an NVIDIA Jetson Orin Nano edge device equipped with an ARM Cortex-A78AE CPU, a 1024-core integrated GPU, 8 GB LPDDR5 memory, and delivering 40 TOPS of AI performance in a 15 W power mode. 

> 368 **Table 2: Bounding box statistics for Roundabout (class 0) and** 

> 369 **Intersection (class 1) classes showing width, height, area, and** 

> 370 **aspect ratio distributions with coordinates (0–1 range).** 

371 372 373 

|**Class**<br>0<br>1|**Metric**<br>Width<br>Height<br>Area<br>Ratio<br>Width<br>Height<br>Area<br>Ratio|ished<br> <br>**Mean ± Std**<br>**Median**<br>**Range**<br>0.413 ± 0.173<br>0.450<br>0.027–0.749<br>0.488 ± 0.212<br>0.437<br>0.092–0.950<br>0.223 ± 0.162<br>0.172<br>0.008–0.593<br>0.959 ± 0.549<br>0.792<br>0.095–3.315<br>0.210 ± 0.158<br>0.163<br>0.013–0.908<br>0.210 ± 0.114<br>0.191<br>0.026–0.737<br>0.055 ± 0.078<br>0.029<br>0.002–0.501<br>1.104 ± 0.673<br>0.990<br>0.082–5.031|
|---|---|---|



374 375 376 

377 378 379 380 381 382 

383 384 385 followed by a split-wise analysis of the training, validation, and 386 test sets with image and instance counts to assess class balance. 387 

## **4.1 Class and Bounding Box Statistics** 

389 390 The dataset exhibits distinct bounding box characteristics across 391 the two classes, as detailed in Table 2. Roundabouts generally oc392 cupy larger portions of the image, with mean width (0.413 ± 0.173), 393 height (0.488 ± 0.212), and area (0.223 ± 0.162) significantly higher 394 than intersections (width: 0.210 ± 0.158, height: 0.210 ± 0.114, area: 395 0.055 ± 0.078). This reflects the typically larger physical footprint of 396 roundabouts compared to intersections. Aspect ratios show moder397 ate variability for both classes, with roundabouts averaging 0.959 ± 398 0.549 and intersections 1.104 ± 0.673, indicating a tendency toward 399 more square-like shapes for roundabouts and slightly elongated 400 forms for intersections. The ranges highlight the diversity within 401 each class, with intersections displaying greater variability in width 402 and aspect ratio, due to the inclusion of various intersection ge403 ometries (3-leg and 4-leg). These statistics underscore the dataset’s 404 representation of real-world road features and inform model train405 ing by highlighting class-specific scale differences. 

## **5.2 Model Selection and Training parameters** 

The YOLO family was selected as the primary detection framework due to its native compatibility with the dataset format and proven effectiveness in real-time object detection tasks. To establish robust benchmarks, we evaluated YOLOv8, YOLOv11, and YOLOv12 across nano, small, medium, large, and extra-large scales to analyze the trade-off between computational efficiency and detection accuracy for edge deployment. Batch sizes of 16 were used across experiments, with some larger models limited to batch size 8 due to GPU memory constraints. However, models such as YOLOv12l and 

406 

2026-06-17 09:20. Page 4 of 1–12. 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersection Detection on Edge Devices 

523 

524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 

507 

508 

509 

510 

511 

512 

513 

514 

515 

516 

517 

518 519 

520 

465 **Table 4: Performance comparison of YOLO (v8, v11, v12) and RT-DETR variants across different model scales and batch sizes,** 466 **showing mAP, precision, recall, F1-score, and inference speed (time in milliseconds per image).** 

||**hi AP ii ll F1- d if d**|**(ti i illid  i)**|
|---|---|---|
|466|**sowng m, precson, reca, score, an nerence spee**|**me n msecons per mage.**|
|467|||
|468|**Model**<br>**Batch Size**<br>**mAP**50<br>**mAP**50:95|**Precision**<br>**Recall**<br>**F1 Score**<br>**Speed(ms)**|
|469|||
||YOLOv8n<br>16<br>0.969<br>0.635|0.978<br>0.956<br>0.967<br>4.653|
|470|||
||YOLOv8s<br>16<br>0.974<br>0.661|0.995<br>0.941<br>0.968<br>4.885|
|471|||
||YOLOv8m<br>16<br>0.963<br>0.657|0.984<br>0.921<br>0.951<br>7.366|
|472|||
||YOLOv8l<br>16<br>0.965<br>0.629|0.978<br>0.939<br>0.958<br>12.707|
|473|||
||YOLOv8x<br>8<br>0.966<br>0.636|0.992<br>0.926<br>0.958<br>18.371|
|474|||
||YOLOv11n<br>16<br>0.975<br>0.637|0.979<br>0.940<br>0.959<br>6.870|
|475|||
||YOLOv11s<br>16<br>0.966<br>0.676|0.958<br>0.937<br>0.947<br>6.668|
|476|||
||YOLOv11m<br>16<br>0.949<br>0.610|0.970<br>0.882<br>0.924<br>8.707|
|477|||
||YOLOv11l<br>16<br>0.968<br>0.640|0.982<br>0.922<br>0.951<br>12.192|
|478<br>479<br>480<br>481<br>482<br>483<br>484<br>485<br>486<br>487<br>488<br>489<br>490<br>491<br>492<br>493<br>494<br>495<br>496<br>497<br>498<br>499<br>500<br>501<br>502<br>503<br>504<br>505|Unpublished working draft.<br>Not for distribution.<br>YOLOv11x<br>8<br>0.957<br>0.608<br>0.993<br>0.901<br>0.945<br>18.507<br>YOLOv12n<br>16<br>0.970<br>0.648<br>0.980<br>0.932<br>0.955<br>10.304<br>YOLOv12s<br>16<br>0.961<br>0.649<br>0.967<br>0.949<br>0.958<br>10.461<br>YOLOv12m<br>8<br>0.968<br>0.614<br>0.953<br>0.947<br>0.950<br>11.169<br>RT-DETR-l<br>8<br>0.966<br>0.599<br>0.948<br>0.964<br>0.956<br>10.600<br>RT-DETR-x<br>8<br>0.919<br>0.567<br>0.904<br>0.860<br>0.882<br>13.900<br>YOLOv12x were excluded from selected experiments due to exces-<br>sive memory demands that are impractical for edge deployment. In<br>addition to the YOLO family, we included real-time Detection Trans-<br>former (RT-DETR) variants (RT-DETR-l and RT-DETR-x) to further<br>validate the dataset under a complementary architectural paradigm.<br>DETR style models rely on transformer-based encoder-decoder<br>mechanisms and bipartite matching loss, providing a contrast to<br>anchor-based, one-stage detectors such as YOLO [6]. To ensure<br>a fair comparison focused on dataset characteristics rather than<br>optimizer tuning, RT-DETR models were trained using the same<br>core optimization settings as YOLO.<br>Training was conducted for 100 epochs with an early stopping<br>patience of 15 epochs to prevent overftting. The optimizer was set<br>to automatic selection, with an initial learning rate of 0.01, fnal<br>learning rate of 0.01, momentum of 0.937, and weight decay of<br>0.0005. Automatic Mixed Precision (AMP) training was enabled to<br>improve memory efciency and training speed [27]. Loss-function<br>weighting was optimized for the dataset’s class distribution, with<br>classifcation loss weight of 0.5 and bounding-box regression loss<br>weight of 7.5 [21]. Data augmentation techniques included geo-<br>YOLOv11s (batch size 16) provides the strongest overall accuracy-<br>efciency trade-of, achieving the highest mAP50:95(0.676) with low<br>inference time (6.67ms). Nano models are faster (4.65–6.87ms) but<br>less consistent at stricter IoU thresholds, while large and extra-large<br>models such as YOLOv8x and YOLOv11x are substantially slower<br>(18.37–18.51ms) without improving mAP50:95 over YOLOv11s. Al-<br>though YOLOv11n reaches a higher mAP50(0.975), its lower mAP50:95<br>(0.637) indicates weaker localization. RT-DETR remains competitive<br>but does not surpass YOLOv11s in combined accuracy and speed.<br>Figure 3 provides complementary evidence where the precision-<br>recall and confdence-based plots (Figure 3A–D) show stable class-<br>wise behavior across threshold settings, with strong precision and<br>recall maintained for both roundabout and intersection categories.<br>The confusion matrix (Figure 3E) indicates minimal cross-class<br>confusion, supporting reliable discrimination between the two la-<br>bels. Qualitative detections (Figure 3F) from summer and winter<br>scenes further demonstrate robustness under seasonal and visibil-<br>ity changes, including snow-covered conditions. In addition, 5-fold<br>cross-validation confrms that these results are stable across data<br>splits (average: 0.950 precision, 0.925 recall, 0.937 F1-score, 0.962||
|506<br>|metric transformations (rotation, translation, scaling, shearing),|mAP50, and 0.669 mAP50:95), with fold-wise details in Appendix I.|



Training was conducted for 100 epochs with an early stopping patience of 15 epochs to prevent overfitting. The optimizer was set to automatic selection, with an initial learning rate of 0.01, final learning rate of 0.01, momentum of 0.937, and weight decay of 0.0005. Automatic Mixed Precision (AMP) training was enabled to improve memory efficiency and training speed [27]. Loss-function weighting was optimized for the dataset’s class distribution, with classification loss weight of 0.5 and bounding-box regression loss weight of 7.5 [21]. Data augmentation techniques included geometric transformations (rotation, translation, scaling, shearing), photometric adjustments (hue, saturation, brightness variations) and advanced augmentation strategies such as mosaic, mixup, copypaste, and random erasing. These settings are tuned for aerial road imagery across seasonal and lighting variation [24, 42]. 

## **5.4 Edge Device Benchmarking** 

To evaluate the practical deployment feasibility of the trained YOLOv11s model on resource-constrained edge devices, comprehensive benchmarking was conducted on the NVIDIA Jetson Orin Nano platform. The evaluation focused on analyzing the trade-offs between inference speed, detection accuracy, and memory consumption across different model formats such as Pytorch [31], TorchScript [8], ONNX [5], TensorRT [18], Tensorflow [1], MNN [19], NCNN [28] and quantization precisions such as 32-bit floating point (FP32), 16-bit floating point (FP16), and 8-bit integer (INT8) [17]. Table 5 presents the performance comparison of various export formats and quantization levels, providing insights into optimal performance for real-time detection. 

Standard object-detection metrics were employed, including mAP at IoU thresholds of 0.5 (mAP50) and 0.5 to 0.95 (mAP50:95), along with precision, recall, and F1-score; formal definitions are provided in Appendix H [15, 36]. Detailed implementation settings, including architecture-specific configurations, augmentation hyperparameters, and run-averaging protocol, are provided in Appendix J. 

## **5.3 Performance Comparison and Analysis** 

Table 4 compares YOLO (v8, v11, v12) and RT-DETR variants using mAP50, mAP50:95, precision, recall, F1-score, and inference time. 2026-06-17 09:20. Page 5 of 1–12. 

521 

522 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

Bappy et al. 

639 

640 with mAP95 dropping to 0.653, while FP16 delivered only mini641 mal accuracy degradation compared to the PyTorch FP32 baseline 642 (0.684) with a significant 55% speed improvement from 32.6ms. 643 —— roundaboutintersection e2| [— roundaboutintersection Other frameworks, such as Tensorflow, ONNX, MNN and NCNN 644 — all classes 1.00 at 0.822 — ailclasses 0.97 at 0.000 were substantially slower, making them unsuitable for real-time 645 applications. Therefore, TensorRT FP16 was the preferred choice 646 for real-time intersection detection, balancing detection accuracy 647 Confusion Matrix for reliable UAV navigation with inference speeds above 60 FPS for 648 roundabout 1 resource-constrained edge deployment. 649 - 650 \ § intersection 18) **6 Limitations and Future Work** 651 652 Although this dataset advances UAV-based intersection detection, 653 \ background 13 * important limitations remain. The dataset is still modest in size and 654 4 5 2 ° geographically concentrated in localized Finnish regions, which 655 3 § g may reduce generalization to different road layouts, urban mor656 08 10 he phologies, and traffic environments. Seasonal coverage is currently 657 F centered on summer and winter, with limited representation of 658 adverse visibility conditions such as heavy rain, fog, and active 659 ee rounded i SPS, snowfall. In addition, the class schema is intentionally coarse, using 660 only two categories, where multiple intersection geometries are 661 Be| Oecad mergedthe present study also does not yet cover the full design space ofinto a single class. While strong baselines are reported, 662663 detector families, training strategies, and deployment settings. 664 Future work will address these gaps through coordinated dataset, 665 modeling, and deployment extensions. Data collection will be ex666 panded across broader geographic regions, road standards, and envi667 **analysis of the best-performing** ronmental conditions, with richer metadata for visibility and occlu668 **model, including: precision–recall** sion. Annotation taxonomy will be refined toward finer-grained in669 tersection categories, potentially with hierarchical labels for down670 stream mapping and planning use cases. On the modeling side, 671 upcoming studies will evaluate a wider range of detector and 672 transformer-based approaches, together with transfer learning, 673 domain adaptation, semi-supervised learning, and more system674 **Orin Nano; reporting model size** atic hyperparameter and ablation analyses. Deployment studies 675 will also be extended to additional edge hardware and accelera676 tion toolchains, with emphasis on portability, energy-performance 677 **Size mAP** 50 **mAP** 50:95 **Speed** trade-offs, and operational reliability in real UAV pipelines. 678 679 18.3 0.971 0.684 32.6 **7 Conclusion** 680 18.3 0.971 0.686 31.9 This work introduces ROADSIGHT, a dedicated multi-season UAV 681 18.3 0.971 0.684 33.9 36.4 0.973 0.665 37.8 dataset for road intersection detection, and establishes a repro682 36.4 0.973 0.664 26.9 ducible benchmark pipeline for training and evaluating modern 683 18.1 0.974 0.666 342.5 object detectors in this domain. By focusing on intersection-centric 684 38.2 0.973 0.665 24.1 annotations and seasonal variability, the dataset addresses a practi685 **21.7 0.973 0.664 14.5** cal gap between generic aerial benchmarks and real-world trans686 12.2 0.975 0.653 10.9 portation intelligence needs. The study demonstrates that robust 687 36.2 0.973 0.665 368.8 intersection detection from RGB UAV imagery is feasible for real688 9.3 0.974 0.663 212.1 time, resource-aware applications when data design, model selec689 18.2 0.973 0.666 326.5 tion, and deployment constraints are jointly considered. Beyond 690 benchmark reporting, the main contribution is a foundation for 691 future research on resilient road-scene understanding across chang692 ing environments. This enables broader opportunities in intelligent 693 traffic monitoring, emergency response support, infrastructure anal694 ysis, and autonomy-oriented geospatial systems. 695 696 

615 616 **Table 5: Performance comparison of YOLOv11s across var-** 617 **ious formats on Jetson Orin Nano; reporting model size** 618 **(megabytes), accuracy and inference speed (ms per image).** 619 620 **Format Size mAP** 50 **mAP** 50:95 **Speed** 621 622 PyTorch (FP32) 18.3 0.971 0.684 32.6 623 PyTorch (FP16) 18.3 0.971 0.686 31.9 PyTorch (INT8) 18.3 0.971 0.684 33.9 624 TorchScript (FP32) 36.4 0.973 0.665 37.8 625 TorchScript (FP16) 36.4 0.973 0.664 26.9 626 ONNX (FP16) 18.1 0.974 0.666 342.5 627 TensorRT (FP32) 38.2 0.973 0.665 24.1 628 **TensorRT (FP16) 21.7 0.973 0.664 14.5** 629 TensorRT (INT8) 12.2 0.975 0.653 10.9 630 TensorFlow (FP32) 36.2 0.973 0.665 368.8 631 MNN (INT8) 9.3 0.974 0.663 212.1 632 NCNN (FP16) 18.2 0.973 0.666 326.5 633 634 635 Benchmark results show that TensorRT with FP16 quantization 636 provides the optimal configuration for real-time UAV deployment, 637 

581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 

accuracy with mAP50:95 of 0.664. Although the INT8 variant offered faster inference at 10.9ms (92 FPS), it compromised accuracy with mAP95 dropping to 0.653, while FP16 delivered only minimal accuracy degradation compared to the PyTorch FP32 baseline (0.684) with a significant 55% speed improvement from 32.6ms. Other frameworks, such as Tensorflow, ONNX, MNN and NCNN were substantially slower, making them unsuitable for real-time applications. Therefore, TensorRT FP16 was the preferred choice for real-time intersection detection, balancing detection accuracy for reliable UAV navigation with inference speeds above 60 FPS for resource-constrained edge deployment. 

609 610 **Figure 3: Performance analysis of the best-performing** 611612613614 **trainedcurve (A), Precision–confidence curve (B), recall–confidencecurve (C), F1–confidence curve (D), Confusion matrix (E) andqualitative detection examples (F) under different weather.YOLOv11s model, including: precision–recall** 

Benchmark results show that TensorRT with FP16 quantization provides the optimal configuration for real-time UAV deployment, achieving 14.5ms inference speed (69 FPS) while maintaining high 

638 

2026-06-17 09:20. Page 6 of 1–12. 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersection Detection on Edge Devices 

755 

756 

757 

758 

759 

760 

761 

762 

763 

764 

765 

766 

767 

768 

769 

770 

771 

772 

773 

774 

775 

776 

777 

778 

779 780 781 

782 783 784 785 786 787 788 789 

790 791 

792 793 794 

795 

796 797 

798 799 

800 

801 

802 

803 804 

805 806 

807 808 809 

810 811 812 

## **Acknowledgments** 

- [18] Eunjin Jeong, Jangryul Kim, and Soonhoi Ha. 2022. TensorRT-Based Framework and Optimization Methodology for Deep Learning Inference on Jetson Boards. _ACM Trans. Embed. Comput. Syst._ 21, 5, Article 51 (Oct. 2022), 26 pages. doi:10. 1145/3508391 

697 698 The authors acknowledge the Biomimetics and Intelligent Systems 699 Group (BISG), University of Oulu, for providing a strong research 700 environment and support for this work. Additionally, the authors 701 also thank VTT Technical Research Centre of Finland Ltd, including 702 the Innovative Air Mobility (IAM) group, for collaboration and 703 support; this work was carried out within the Drolo II - Business 704 From Urban Airspace (RRF) project. 

- [19] Xiaotang Jiang, Huan Wang, Yiliu Chen, Ziqi Wu, Lichuan Wang, Bin Zou, Yafeng Yang, Zongyang Cui, Yu Cai, Tianhang Yu, Chengfei Lv, and Zhihua Wu. 2020. MNN: A Universal and Efficient Inference Engine. In _MLSys_ . 

- [20] V. Roshan Joseph. 2022. Optimal ratio for data splitting. _Statistical Analysis and Data Mining: The ASA Data Science Journal_ 15, 4 (2022), 531–538. arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/sam.11583 doi:10.1002/sam. 11583 

- [21] Rahima Khanam, Tahreem Asghar, and Muhammad Hussain. 2025. Comparative Performance Evaluation of YOLOv5, YOLOv8, and YOLOv11 for Solar Panel Defect Detection. _Solar_ 5, 1 (2025). doi:10.3390/solar5010006 

705 

> 706 **References** 

- 707 [1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, 708 Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, San709 jay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, 

- 710 Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike 711 Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul 712 Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals,Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng.Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 713 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. 714 https://www.tensorflow.org/ Software available from tensorflow.org. 

   - [22] Jouko Kinnari, Francesco Verdoja, and Ville Kyrki. 2022. Season-invariant GNSSdenied visual localization for UAVs. _IEEE Robotics and Automation Letters_ 7, 4 (Oct. 2022), 10232–10239. doi:10.1109/LRA.2022.3191038 arXiv:2110.01967 [cs]. 

- Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, [23] Benedikt Kolbeinsson and Krystian Mikolajczyk. 2024. DDOS: The Drone Depth 

- 710 Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike and Obstacle Segmentation Dataset. In _Proceedings of the IEEE/CVF Conference_ 711 Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul _on Computer Vision and Pattern Recognition (CVPR) Workshops_ . 7328–7337. 712 Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals,Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng.Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. [24] Yong-Suk Lee, Maheshkumar Prakash Patil, Jeong Gyu Kim, Yong Bae Seo, Dong-Hyun Ahn, and Gun-Do Kim. 2025. Hyperparameter Optimization for Tomato 713 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Leaf Disease Recognition Based on YOLOv11m. _Plants_ 14, 5 (2025). doi:10.3390/ 714 https://www.tensorflow.org/ Software available from tensorflow.org. plants14050653 [2] Tanzina Afrin, Nita Yodo, Arup Dey, and Lucy G. Aragon. 2024. Advancements in [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, 

- 715 UAV-Enabled Intelligent Transportation Systems: A Three-Layered Framework James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. 716 and Future Directions. _Applied Sciences_ 14, 20 (2024). doi:10.3390/app14209455 2015. Microsoft COCO: Common Objects in Context. arXiv:1405.0312 [cs.CV] 717 [3] Azim Ahmadzadeh, Rohan Adhyapak, Armin Iraji, Kartik Chaurasiya, V Aparna, https://arxiv.org/abs/1405.0312 and Petrus C. Martens. 2025. A Guide for Manual Annotation of Scientific [26] Wenyu Lv, Shangliang Xu, Yian Zhao, Guanzhong Wang, Jinman Wei, Cheng Cui, 

- 718 Imagery: How to Prepare for Large Projects. arXiv:2508.14801 [cs.LG] https: Yuning Du, Qingqing Dang, and Yi Liu. 2023. DETRs Beat YOLOs on Real-time 719 //arxiv.org/abs/2508.14801 Object Detection. arXiv:2304.08069 [cs.CV] 720 [4] Siva Ariram, Veikko Pekkala, Timo Mäenpää, Antti Tikänmaki, and Juha Röning. [27] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich 2024. UAV-based Intelligent Information Systems on Winter Road Safety for Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh 

- 721 Autonomous Vehicles. In _2024 IEEE Smart World Congress (SWC)_ . 1892–1898. Venkatesh, and Hao Wu. 2018. Mixed Precision Training. doi:10.48550/arXiv. 722 doi:10.1109/SWC62898.2024.00290 1710.03740 arXiv:1710.03740 [cs]. [5] Junjie Bai, Fang Lu, Ke Zhang, et al. 2019. ONNX: Open Neural Network Exchange. [28] Hui Ni and The ncnn contributors. 2017. _ncnn_ . https://github.com/Tencent/ncnn 

- 723 https://github.com/onnx/onnx. [29] Alba Nogueira-Rodríguez, Daniel Glez-Peña, Miguel Reboiro-Jato, and Hugo 724 [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander López-Fernández. 2023. Negative Samples for Improving Object Detection—A 725 Kirillov, and Sergey Zagoruyko. 2020. End-to-End Object Detection with TransCase Study in AI-Assisted Colonoscopy for Polyp Detection. _Diagnostics_ 13, 5 formers. _CoRR_ abs/2005.12872 (2020). arXiv:2005.12872 https://arxiv.org/abs/ (2023). doi:10.3390/diagnostics13050966 

- 726 2005.12872 [30] Lubna Obaid, Khaled Hamad, Rami Al-Ruzouq, Saleh Abu Dabous, Karim Ismail, 727 [7] Bingyan Cui, Zhen Liu, and Qifeng Yang. 2025. UAV-YOLO12: A Multi-Scale and Emran Alotaibi. 2025. State-of-the-art review of unmanned aerial vehicles 728 Road Segmentation Model for UAV Remote Sensing Imagery.doi:10.3390/drones9080533 _Drones_ 9, 8 (2025). (UAVs)progress, applications, challenges, and opportunities.and artificial intelligence (AI) for traffic and safety _Transportation Research_ analyses: Recent 729 [8] Zachary DeVito. 2022. Torchscript: Optimized execution of pytorch programs. _Interdisciplinary Perspectives_ 33 (2025), 101591. doi:10.1016/j.trip.2025.101591 

Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul [24] Hyun Ahn, and Gun-Do Kim. 2025. Leaf Disease Recognition Based on YOLOv11m. Software available from tensorflow.org. plants14050653 [25] _Applied Sciences_ 14, 20 (2024). doi:10.3390/app14209455 2015. https://arxiv.org/abs/1405.0312 2025. A Guide for Manual Annotation of Scientific [26] arXiv:2508.14801 [cs.LG] https: Yuning Du, Qingqing Dang, and Yi Liu. 2023. Object Detection. arXiv:2304.08069 [cs.CV] [27] Paulius Micikevicius, Sharan Narang, Information Systems on Winter Road Safety for _2024 IEEE Smart World Congress (SWC)_ . 1892–1898. Venkatesh, and Hao Wu. 2018. 1710.03740 arXiv:1710.03740 [cs]. . 2019. ONNX: Open Neural Network Exchange. 2019. ONNX: Open Neural Network Exchange. [28] Hui Ni and The ncnn contributors. 2017. [29] Alba Nogueira-Rodríguez, Daniel López-Fernández. 2023. End-to-End Object Detection with TransarXiv:2005.12872 https://arxiv.org/abs/ (2023). doi:10.3390/diagnostics13050966 [30] UAV-YOLO12: A Multi-Scale and Emran Alotaibi. 2025. _Drones_ 9, 8 (2025). and artificial intelligence (AI) Torchscript: Optimized execution of pytorch programs. _Interdisciplinary Perspectives_ [31] The Unmanned Aerial arXiv:1804.00518 [cs.CV] Chintala. 2019. Library. arXiv:1912.01703 [cs.LG] [32] Sánchez-Soriano. 2022. _Data_ 7, 4 (2022). doi:10.3390/data7040047 The Pascal Visual Object Classes (VOC) Challenge. [33] _Vision_ 2 303–338. doi:10. 

- 729 [8] Zachary DeVito. 2022. Torchscript: Optimized execution of pytorch programs. 730 _Retrieved January_ (2022). 

   - [31] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv:1912.01703 [cs.LG] https://arxiv.org/abs/1912.01703 

- 730 

- [9] Dawei Du, Yuankai Qi, Hongyang Yu, Yifan Yang, Kaiwen Duan, Guorong Li, 

- 731 Weigang Zhang, Qingming Huang, and Qi Tian. 2018. The Unmanned Aerial 732 Vehicle Benchmark: Object Detection and Tracking. arXiv:1804.00518 [cs.CV] https://arxiv.org/abs/1804.00518 

- 733 [10] Fatmaelzahraa Eltaher, Ayman Taha, Jane Courtney, and Susan Mckeever. 2022. 

- 734 Using Satellite Images Datasets for Road Intersection Detection in Route Planning. 735 _Articles_ (Oct. 2022). doi:10.21427/vv5x-bb77 

   - [32] Enrique Puertas, Gonzalo De-Las-Heras, Javier Fernández-Andrés, and Javier Sánchez-Soriano. 2022. Dataset: Roundabout Aerial Images for Vehicle Detection. _Data_ 7, 4 (2022). doi:10.3390/data7040047 

- 736 [11] Andrew Zisserman. 2010.Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, andThe Pascal Visual Object Classes (VOC) Challenge. 737 _International Journal of Computer Vision_ 88, 2 (01 Jun 2010), 303–338. doi:10. 1007/s11263-009-0275-4 

   - [33] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016. You Only Look Once: Unified, Real-Time Object Detection. doi:10.48550/arXiv.1506.02640 arXiv:1506.02640 [cs]. 

   - [34] Joseph Redmon and Ali Farhadi. 2018. YOLOv3: An Incremental Improvement. arXiv:1804.02767 [cs.CV] https://arxiv.org/abs/1804.02767 

- 738 

- [12] Michael Fonder and Marc Van Droogenbroeck. 2019. Mid-Air: A Multi-Modal 

- 739 Dataset for Extremely Low Altitude Drone Flights. In _Proceedings of the IEEE/CVF_ 740 _Conference on Computer Vision and Pattern Recognition (CVPR) Workshops_ . 

   - [35] Giulia Rizzoli, Francesco Barbato, Matteo Caligiuri, and Pietro Zanuttigh. 2023. SynDrone - Multi-Modal UAV Dataset for Urban Scenarios. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops_ . 2210–2220. 

- 741 [13] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. _Deep Learning_ . MIT Press. http://www.deeplearningbook.org. 

- 742 [14] Hira Lal Gope, Hidekazu Fukai, Fahim Mahafuz Ruhad, and Shohag Barman. 2024. 743 Comparative analysis of YOLO models for green coffee bean detection and defect 744 classification. _Scientific Reports_ 14, 1 (Nov. 2024), 28946. doi:10.1038/s41598-02478598-7 

   - [36] Ranjan Sapkota, Marco Flores-Calero, Rizwan Qureshi, Chetan Badgujar, Upesh Nepal, Alwin Poulose, Peter Zeno, Uday Bhanu Prakash Vaddevolu, Sheheryar Khan, Maged Shoman, Hong Yan, and Manoj Karkee. 2025. YOLO advances to its genesis: a decadal and comprehensive review of the You Only Look Once (YOLO) series. _Artificial Intelligence Review_ 58, 9 (June 2025), 274. doi:10.1007/s10462025-11253-3 

- 745 [15] Zijian He, Kang Wang, Tian Fang, Lei Su, Rui Chen, and Xihong Fei. 2024. Com746 prehensive Performance Evaluation of YOLOv11, YOLOv10, YOLOv9, YOLOv8 and YOLOv5 on Object Detection of Power Equipment. arXiv:2411.18871 [cs.CV] 

- 747 https://arxiv.org/abs/2411.18871 

   - [37] Boris Sekachev, Nikita Manovich, Maxim Zhiltsov, Andrey Zhavoronkov, Dmitry Kalinin, Ben Hoff, TOsmanov, Dmitry Kruchinin, Artyom Zankevich, DmitriySidnev, Maksim Markelov, Johannes222, Mathis Chenuet, A-Andre, Telenachos, Aleksandr Melnikov, Jijoong Kim, Liron Ilouz, Nikita Glazov, Priya4607, Rush Tehrani, Seungwon Jeong, Vladimir Skubriev, Sebastian Yonekura, Vugia Truong, Zliang7, Lizhming, and Tritin Truong. 2020. opencv/cvat: v1.1.0. doi:10.5281/ ZENODO.4009388 

- 748 [16] Xing Hu, Siyuan Chen, Qianqian Duan, Choon Ki Ahn, Huiliang Shang, and 749 Dawei Zhang. 2025. Domain Adaptation for Big Data in Agricultural Image Analysis: A Comprehensive Review. arXiv:2506.05972 [cs.CV] https://arxiv.org/ 

- 750 abs/2506.05972 

- 751 [17] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew 752 Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.Howard, Hartwig Adam, and Dmitry Kalenichenko. 2017. Quantization and 753 arXiv:1712.05877 [cs.LG] https://arxiv.org/abs/1712.05877 

- [38] Shahil Sharma, Siddarth Singotam, Abhinav Kayastha, Omid Jafari, Aki Happonen, Jukka-Pekka Skön, Jukka Heikkonen, and Rajeev Kanth. 2025. YOLO 

> 754 2026-06-17 09:20. Page 7 of 1–12. 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

Bappy et al. 

- 813 for Urban Traffic: Insights from Helsinki Port Surveillance. In _Soft Computing:_ 814 _Theories and Applications_ , Rajesh Kumar, Ajit Kumar Verma, Om Prakash Verma, and Jitendra Rajpurohit (Eds.). Springer Nature Singapore, Singapore, 13–23. 

- 815 [39] Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai 816 Datcu, Marcello Pelillo, and Liangpei Zhang. 2018. DOTA: A Large-Scale Dataset 817 for Object Detection in Aerial Images. In _2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 3974–3983. doi:10.1109/CVPR.2018.00418 

||||||
|---|---|---|---|---|
|813||for Urban Trafc: Insights from Helsinki Port Surveillance. In_Soft Computing:_||871|
|814||_Theories and Applications_, Rajesh Kumar, Ajit Kumar Verma, Om Prakash Verma,||872|
|815|[39]|and Jitendra Rajpurohit (Eds.). Springer Nature Singapore, Singapore, 13–23.<br> Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai||873|
|816||Datcu, Marcello Pelillo, and Liangpei Zhang. 2018. DOTA: A Large-Scale Dataset||874|
|817||for Object Detection in Aerial Images. In_2018 IEEE/CVF Conference on Computer_<br>_Vision and Pattern Recognition_. 3974–3983. doi:10.1109/CVPR.2018.00418||875|
|818|[40]|Wenjuan Yang, Xuhui Zhang, Bing Ma, Yanqun Wang, Yujia Wu, Jianxing Yan,||876|
|819||Yongwei Liu, Chao Zhang, Jicheng Wan, Yue Wang, Mengyao Huang, Yuyang Li,||877|
|820||and Dian Zhao. 2023. An open dataset for intelligent recognition and classifcation<br>of abnormal condition in longwall mining. _Scientifc Data_10, 1 (June 2023), 416.||878|
|821||doi:10.1038/s41597-023-02322-9 Publisher: Nature Publishing Group.||879|
|822|[41]|Kai Ye, Haidi Tang, Bowen Liu, Pingyang Dai, Liujuan Cao, and Rongrong Ji. 2025.||880|
|||More Clear, More Flexible, More Precise: A Comprehensive Oriented Object|||
|823||Detection benchmark for UAV. doi:10.48550/arXiv.2504.20032 arXiv:2504.20032||881|
|824||[cs].||882|
|825|[42]|Gui Yu and Xinglin Zhou. 2023. An Improved YOLOv5 Crack Detection Method<br>Combined with a Bottleneck Transformer. _Mathematics_ 11, 10 (2023). doi:10.||883|
|826<br>827<br>828<br>829<br>830<br>831<br>832<br>833<br>834<br>835<br>836<br>837<br>838<br>839<br>840<br>841<br>842<br>843<br>844<br>845<br>846<br>847<br>848<br>849<br>850<br>851<br>852<br>853|[43] <br>[44] <br>[45]|Unpublished working draft.<br>Not for distribution.<br>3390/math11102377<br> Syed Waqas Zamir, Aditya Arora, Akshita Gupta, Salman Khan, Guolei Sun, Fa-<br>had Shahbaz Khan, Fan Zhu, Ling Shao, Gui-Song Xia, and Xiang Bai. 2019.<br>iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images.<br>arXiv:1905.12886 [cs.CV] https://arxiv.org/abs/1905.12886<br> Pengfei Zhu, Longyin Wen, Dawei Du, Xiao Bian, Heng Fan, Qinghua Hu, and<br>Haibin Ling. 2022. Detection and Tracking Meet Drones Challenge. _IEEE Trans-_<br>_actions on Pattern Analysis and Machine Intelligence_ 44, 11 (2022), 7380–7399.<br>doi:10.1109/TPAMI.2021.3119563<br> Shuya Zong, Sikai Chen, Majed Alinizzi, and Samuel Labi. 2022. Leveraging<br>UAV Capabilities for Vehicle Tracking and Collision Risk Assessment at Road<br>Intersections. _Sustainability_14, 7 (2022). doi:10.3390/su14074034||884<br>885<br>886<br>887<br>888<br>889<br>890<br>891<br>892<br>893<br>894<br>895<br>896<br>897<br>898<br>899<br>900<br>901<br>902<br>903<br>904<br>905<br>906<br>907<br>908<br>909<br>910<br>911|
|854||||912|
|855||||913|
|856||||914|
|857||||915|
|858||||916|
|859||||917|
|860||||918|
|861||||919|
|862||||920|
|863||||921|
|864||||922|
|865||||923|
|866||||924|
|867||||925|
|868||||926|
|869||||927|
|870|||2026-06-17 09:20 Pae 8 of 1–12|928|



2026-06-17 09:20. Page 8 of 1–12. 

ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersection Detection on Edge Devices 

987 

988 

989 990 991 992 

993 

994 995 996 997 998 999 

> 942 **Date:** 17 May 2026 (modified: 29 May 2026) 1000 943 comprises 969 high-resolution images, capturing both summer and **Relevance:** 1 — The paper has significant dataset-based contribu1001 944 winter conditions to account for seasonal variability. Furthermore, tions, but also some notable algorithmic contributions 1002 945 the paper provides a comprehensive algorithmic analysis, bench- **Rating:** 5 — Marginally below acceptance threshold 1003 946 marking modern detection architectures like YOLOv8, YOLOv11, **Confidence:** 4 — The reviewer is confident but not absolutely 1004 947 YOLOv12, and RT-DETR, and evaluates their deployment feasibility certain that the evaluation is correct 1005 948 on an edge device (NVIDIA Jetson Orin Nano). 1006 

> 949 **Review** 1007 950 **Quality and Clarity** The paper presents ROADSIGHT, a multi-season UAV dataset for 1008 

> 951 The paper is well-structured and clearly written. The methodologies 1009 road intersection and roundabout detection on edge devices. The 

> 952 for data collection, preprocessing, and annotation are transparent dataset contains 969 aerial RGB images with 1,355 annotations 1010 

> 953 and systematic. The inclusion of a detailed performance comparison collected in summer and winter conditions. The authors benchmark 1011 

> 954 across varying quantization levels and formats (e.g., demonstrat1012 YOLOv8/v11/v12 and RT-DETR models. The work aims to support 

> 955 ing that TensorRT with FP16 quantization provides the optimal 1013 robust UAV-based transportation and edge AI applications. 

> 956 configuration) adds practical value for real-world deployment. 1014 

> 957 **Pros** 1015 958 1016 **Originality and Significance** • Provides extensive benchmark results across YOLOv8/v11/v12 

> 959960 While intersection detection is not a new problem, the specific and RT-DETR models under unified training settings. 10171018 961962 focus on UAV-captured multi-season imagery coupled with edge-deployment benchmarking provides a neat, application-orientedperspective. However, the contribution leans more toward a special• Strong practical relevance for UAV navigation, intelligenttransportation, and edge multimedia systems. 10191020 

> 963 ized use case rather than a foundational community dataset. The **Cons** 1021 

> 964965966 limited scale and scope restrict its overall significance for broadermultimedia and computer vision research. • with modern aerial vision benchmarks.Dataset size is still relatively small ( _<_ 1000 images) compared 102210231024 

> 967 **Pros** • Geographic diversity is limited to localized Finnish regions, 1025 968 which may reduce generalization. 1026 969 • **Practical Focus:** Strong motivation targeting real-time de• The class taxonomy is coarse, merging multiple intersection 1027 

> ployment on resource-constrained edge hardware, which is types into a single “intersection” class. 

1027 1028 1029 1030 

1031 1032 

1033 1034 

1041 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

## 929 **8 Review 1 — Reviewer CCcd1** 

- 930 **Title:** Review of ROADSIGHT: A Multi-Season UAV Dataset for 

> 931 Intersection Detection 

932 

- **Date:** 17 May 2026 (modified: 29 May 2026) 

> 933 **Relevance:** 1 — The paper has significant dataset-based contribu- 

- 934 tions, but also some notable algorithmic contributions 

- 935 **Rating:** 5 — Marginally below acceptance threshold 

- 936 **Confidence:** 4 — The reviewer is confident but not absolutely 937 certain that the evaluation is correct 

   - (e.g., T-junctions, Y-junctions, and 4-leg crossroads) into a single “intersection” class. 

   - **Geographic Constraint:** The data was collected exclusively in localized Finnish regions. This lack of geographic diversity severely limits the models’ adaptability to different urban morphologies and international road layouts. 

   - **Algorithmic Novelty:** The work relies entirely on existing off-the-shelf detectors and does not propose any novel network architectures or unique training paradigms. 

- 938 

939 

## **Review** 

- 940 

941 The paper present ROADSIGHT, a dataset tailored for road inter942 section and roundabout detection using UAV imagery. The dataset 943 comprises 969 high-resolution images, capturing both summer and 944 winter conditions to account for seasonal variability. Furthermore, 945 the paper provides a comprehensive algorithmic analysis, bench946 marking modern detection architectures like YOLOv8, YOLOv11, 947 YOLOv12, and RT-DETR, and evaluates their deployment feasibility on an edge device (NVIDIA Jetson Orin Nano). 

- 969 • **Practical Focus:** Strong motivation targeting real-time de970 ployment on resource-constrained edge hardware, which is highly relevant for UAV applications. 

971 

- 972 • **Comprehensive Benchmarking:** Provides solid baseline 973 evaluations using a wide array of state-of-the-art object detectors and deployment frameworks. 

- 974 975 • **Seasonal Variation:** The inclusion of both winter (snow976 covered) and summer conditions addresses a practical gap in existing aerial datasets. 

977 

978 

## **Cons** 

979 

- 980 • **Modest Scale:** With only 969 images and 1,355 instances, 981 the dataset is quite small compared to contemporary bench982 marks, which may be insufficient to train models that gener983 alize well in the wild. 984 • **Coarse Taxonomy:** The classification schema is overly sim985 plified, merging complex and varied intersection geometries 

## **9 Review 2 — Reviewer W2YP** 

**Title:** Review: ROADSIGHT: A Multi-Season UAV Dataset for RealTime Road Intersection Detection on Edge Devices 

- The work mainly benchmarks existing detectors and does not introduce methodological or architectural innovations. 

## **10 Review 3 — Reviewer DTh3** 

**Title:** A UAV-based detection dataset for edge devices **Date:** 15 May 2026 (modified: 29 May 2026) 

**Relevance:** 1 — The paper has significant dataset-based contribu1035 tions, but also some notable algorithmic contributions 1036 **Rating:** 5 — Marginally below acceptance threshold 1037 **Confidence:** 4 — The reviewer is confident but not absolutely 1038 certain that the evaluation is correct 1039 1040 

## **Review** 

The dataset focuses only on UAV-based detection of two road-layout 1042 classes, intersections and roundabouts, collected from localized 1043 

- 2026-06-17 09:20. Page 9 of 1–12. 

986 

1044 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

Bappy et al. 

> 1045 regions in Finland. While the paper includes seasonal variation and 

> 1046 edge-device benchmarking, the dataset’s limited taxonomy, modest 

> 1047 scale, and geographically constrained collection reduce its potential 

> 1048 value as a broadly useful community benchmark. In its current 

> 1049 form, the dataset seems more like a specialized application dataset 

> 1050 than a resource that would enable a wide range of future research. 

> 1051 For a dataset-track paper, I would expect either substantially 

> 1052 broader coverage, a more challenging or underexplored annotation 

> 1053 taxonomy, stronger evidence of community need, or more extensive 

> 1054 benchmarking that reveals new research challenges. This paper 

> 1055 only partially satisfies these expectations. 

> 1056 Overall, while the dataset is somewhat narrow, it is clearly con- 

> 1057 structed, practically motivated, and accompanied by reasonable 

> 1058 baseline and deployment experiments. I therefore view it as just 

> 1059 below the acceptance threshold, though with limited expected im1060 

|||||
|---|---|---|---|
|1045|regions in Finland. While the paper includes seasonal variation and||1103|
|1046|edge-device benchmarking, the dataset’s limited taxonomy, modest||1104|
|1047|scale, and geographically constrained collection reduce its potential||1105|
|1048|value as a broadly useful community benchmark. In its current||1106|
|1049|form, the dataset seems more like a specialized application dataset||1107|
|1050|than a resource that would enable a wide range of future research.||1108|
|1051|For a dataset-track paper, I would expect either substantially||1109|
|1052|broader coverage, a more challenging or underexplored annotation||1110|
|1053|taxonomy, stronger evidence of community need, or more extensive||1111|
|1054|benchmarking that reveals new research challenges. This paper||1112|
|1055|only partially satisfes these expectations.||1113|
|1056|Overall, while the dataset is somewhat narrow, it is clearly con-||1114|
|1057|structed, practically motivated, and accompanied by reasonable||1115|
|1058<br>1059<br>1060<br>1061<br>1062<br>1063<br>1064<br>1065<br>1066<br>1067<br>1068<br>1069<br>1070<br>1071<br>1072<br>1073<br>1074<br>1075<br>1076<br>1077<br>1078<br>1079<br>1080<br>1081<br>1082<br>1083<br>1084<br>1085|Unpublished working draft.<br>Not for distribution.<br>baseline and deployment experiments. I therefore view it as just<br>below the acceptance threshold, though with limited expected im-<br>pact.<br>**11**<br>**Review 4 — Reviewer GMzG**<br>**Title:**Dataset for Road Intersection Detection<br>**Date:**15 May 2026 (modifed: 29 May 2026)<br>**Relevance:**2 — The main contributions of this paper are exclu-<br>sively dataset-based<br>**Rating:**4 — Ok but not good enough – rejection<br>**Confdence:**5 — The reviewer is absolutely certain that the evalu-<br>ation is correct and very familiar with the relevant literature<br>**Review**<br>This paper presents a UAV dataset for road intersection detection.<br>It includes 969 high-resolution images captured during both sum-<br>mer and winter, with 1,355 manual annotations across two classes:<br>roundabout and intersection. The authors provide detection bench-<br>marks using YOLOv8, YOLOv11, YOLOv12, and RT-DETR models,<br>and validate real-time performance on the NVIDIA Jetson Orin<br>Nano.<br>• This dataset contains only 969 images, which is relatively<br>small for modern detection models.<br>• The annotations are simple. Only two high-level classes are<br>used. Intersection sub-types (T-junction, Y-junction, cross-<br>shaped) are not diferentiated.<br>• All data were collected from a limited region in Finland,||1116<br>1117<br>1118<br>1119<br>1120<br>1121<br>1122<br>1123<br>1124<br>1125<br>1126<br>1127<br>1128<br>1129<br>1130<br>1131<br>1132<br>1133<br>1134<br>1135<br>1136<br>1137<br>1138<br>1139<br>1140<br>1141<br>1142<br>1143|
|1086|which may limit the model’s generalizability to other road||1144|
|1087|structures and urban layouts.||1145|
|1088|• No cross-dataset evaluation or domain adaptation exper-||1146|
|1089|iments have been conducted. Additionally, robustness on||1147|
|1090|small or occluded intersections has not been analyzed.||1148|
|1091|**12**<br>**PAPER WITH 1000 SAMPLES**||1149|
|1092|||1150|
|1093|• #266 | A Spatial Relationship Aware Dataset for Robotics||1151|
|1094|| 1000 samples | 6 classes | 6-7k instances||1152|
|1095|• #270 | MCOD: The First Challenging Benchmark for Mul-||1153|
|1096|tispectral Camoufaged Object Detection | 1500 images | 1||1154|
|1097|class | 1500 instances||1155|
|1098|||1156|
|1099|||1157|
|1100|||1158|
|1101|||1159|
|1102||2026-06-17 09:20 Pae 10 of 1–12|1160|



2026-06-17 09:20. Page 10 of 1–12. 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

ROADSIGHT: A Multi-Season UAV Dataset for Real-Time Road Intersection Detection on Edge Devices 

> 1161 **13 RESPONSE TO REVIEWERS** 1219 1162 1220 1163 1221 1164 1222 1165 1223 1166 1224 1167 1225 1168 1226 1169 1227 1170 1228 1171 1229 1172 1230 1173 1231 1174 1232 1175 1233 1176 1234 1177 1235 1178 1236 1179 1237 1180 1238 1181 1239 1182 1240 1183 1241 1184 1242 1185 1243 1186 1244 1187 1245 1188 1246 1189 1247 1190 1248 1191 1249 1192 1250 1193 1251 1194 1252 1195 1253 1196 1254 1197 1255 1198 1256 1199 1257 1200 1258 1201 1259 1202 1260 1203 1261 1204 1262 1205 1263 1206 1264 1207 1265 1208 1266 1209 1267 1210 1268 1211 1269 1212 1270 1213 1271 1214 1272 1215 1273 1216 1274 1217 1275 

> 1218 2026-06-17 09:20. Page 11 of 1–12. 1276 

MM ’26, November 10–14,2026, Rio de Janeiro, Brazil 

Bappy et al. 

1335 

1336 1337 1338 

1339 1340 

1341 

1320 1321 

1322 

1323 

1324 

1325 1326 1327 

1328 

1329 

1330 

(intersection box area: mean 0.055, median 0.029, min 0.002 of image area; Sec. Class and BBox Statistics). A dedicated scale/occlusion breakdown will be added to the dataset card and pursued in the planned occlusion-metadata release (App. D). 

> 1277 We thank all four reviewers for their careful, constructive feed- 

> 1278 back, and for recognizing the paper’s clarity, edge-deployment 

> 1279 focus, comprehensive benchmarking, and seasonal coverage. Four 

> 1280 concerns recur; we address these first (C1-C4), then reply to reviewer- 

> 1281 specific points. 

Reviewer DTh3 (community value / impact). The public release (images, labels, dataset card, training configs, reproducible splits, fixed seed) is designed for direct reuse and extension. We will strengthen the statement of community need and intended downstream uses (mapping, navigation, transport monitoring). 

> 1282 C1. Dataset scale (all reviewers). We respectfully note that dataset- 

> 1283 track contribution is driven by the gap addressed and annota- 

> 1284 tion/benchmark quality, not image count alone. ROADSIGHT is, 

> 1285 to our knowledge, the first UAV dataset to jointly provide (i) in- 

> 1286 tersection and roundabout-specific target annotations, (ii) true 

> 1287 multi-season capture including snow-covered winter, and (iii) edge- 

> 1288 deployment benchmarking. Prior aerial datasets address at most one 

> 1289 or two of these: general aerial benchmarks (DOTA[1], VisDrone[2], 

> 1290 iSAID[3]) and UAV traffic datasets (UAVDT[4], UTUAV[5]) anno- 

> 1291 tate vehicles rather than intersections, while the Roundabout Aerial 

> 1292 dataset[6] is single-class and season-invariant. On scale specifi- 

> 1293 cally, a directly comparable ACM MM’25 dataset paper, the Spatial- 

> 1294 Relationship-Aware robotics dataset[7], was accepted with 1,000 

> 1295 images, close to our 969, showing the community values focused 

> 1296 datasets of this size when they fill a clear gap. Our 1,355 expert- 

> 1297 verified instances with two-pass review yield stable results, evi- 

> 1298 denced by 5-fold cross-validation with low variance. Broader ge- 

> 1299 ographic and environmental expansion is already planned in our 

> 1300 Limitations section. 

1342 stream uses (mapping, navigation, transport monitoring). 1343 target annotations, (ii) true We believe these clarifications address the core concerns within 1344 the paper’s intended dataset-track scope, and we will incorporate 1345 every revision above in the camera-ready version. 1346 References [1] Xia et al. DOTA. CVPR 2018. doi: 10.1109/CVPR.2018.004 **1** 3478 [2] Du et al. VisDrone-DET2019. ICCVW 2019. doi: 10.1109/IC1348 CVW.2019.00030. [3] Zamir et al. iSAID. CVPRW 2019., pp. 281349 and season-invariant. On scale specifi37. [4] Du et al. UAVDT: object detection and tracking. ECCV 1350 2018. doi:10.1007/978-3-030-01249-623[5] _𝐿𝑒𝑝𝑖𝑛𝑒𝑡𝑎𝑙.𝑈𝑇𝑈𝐴𝑉.𝑑𝑜𝑖_ : 1351 10 _._ 3390/ _𝑑𝑟𝑜𝑛𝑒𝑠_ 10010015[6] _𝑃𝑢𝑒𝑟𝑡𝑎𝑠𝑒𝑡𝑎𝑙.𝑅𝑜𝑢𝑛𝑑𝑎𝑏𝑜𝑢𝑡𝑎𝑒𝑟𝑖𝑎𝑙𝑖𝑚𝑎𝑔𝑒𝑠.𝑑𝑜𝑖_ : 1352 10 _._ 3390/ _𝑑𝑎𝑡𝑎_ 7040047[7] _𝑊𝑎𝑛𝑔𝑒𝑡𝑎𝑙.𝑆𝑝𝑎𝑡𝑖𝑎𝑙𝑅𝑒𝑙𝑎𝑡𝑖𝑜𝑛𝑠ℎ𝑖𝑝𝐴𝑤𝑎𝑟𝑒𝐷𝑎𝑡𝑎𝑠𝑒𝑡𝑓𝑜𝑟𝑅𝑜𝑏𝑜𝑡𝑖𝑐𝑠.𝐴𝐶𝑀𝑀𝑀_ 1353 10 _._ 1145/3746027 _._ 3758293[8] _𝐿𝑖𝑒𝑡𝑎𝑙.𝑀𝐶𝑂𝐷.𝐴𝐶𝑀𝑀𝑀_[′] 25 _.𝑑𝑜𝑖_ : 10 _._ 1145/37460271354 _._ two-pass review yield stable results, evi1355 1356 1357 1358 1359 1360 1361 1362 1363 1364 We already report geometry diversity 1365 1366 1367 1368 1369 in Finland (all reviewers). We 1370 1371 1372 1373 1374 1375 1376 1377 1378 1379 1380 1381 1382 1383 1384 1385 1386 1387 this and frame domain adaptation as 1388 1389 1390 1391 1392 

> 1301 C2. Coarse two-class taxonomy (all reviewers). Merging T-, Y-, 

> 1302 and 4-leg junctions into one "intersection" class was a deliberate, 

> 1303 documented choice (Sec. Data Annotation; App. A), motivated by 

> 1304 reliable separability from aerial RGB at 60-90 m altitude and consis- 

> 1305 tent inter-annotator agreement. A minimal class schema is accepted 

> 1306 practice at this venue: MCOD[8] (ACM MM’25) released 1,500 im- 

> 1307 ages with a single class. We already report geometry diversity 

> 1308 quantitatively: intersection boxes show markedly higher width and 

> 1309 aspect-ratio variance than roundabouts (Sec. Class and BBox Statis- 

> 1310 tics), reflecting the underlying sub-types. Finer hierarchical labels 

> 1311 (T/Y/4-leg) are identified as future work (Sec. Limitations). 

> 1312 C3. Geographic concentration in Finland (all reviewers). We 

> 1313 acknowledge this and already flag it as the primary limitation (Sec. 

> 1314 Limitations). We stress that the Finnish setting directly serves our 

> 1315 season-aware goal: it is what yields genuine snow-covered winter 

> 1316 imagery and paired summer-winter captures of the same road types, 

> 1317 the seasonal variation that existing aerial datasets explicitly lack 

> 1318 (Sec. Related Work). The geographic scope and our plan to expand 

> 1319 collection across broader regions, road standards, and conditions 

> 1320 are already stated in the Introduction and Future Work. 

C4. Algorithmic novelty (CCcd1, W2YP, GMzG). The paper is positioned primarily as a dataset and benchmark contribution, not an architecture paper. Its methodological contribution is the unified, reproducible protocol across YOLOv8/v11/v12 and RT-DETR with a 3-precision edge study on Jetson Orin Nano (Sec. Edge Benchmarking): TensorRT-FP16 yields a 55 

Reviewer GMzG (cross-dataset eval; small/occluded robustness). No public UAV dataset shares our intersection taxonomy and season coverage, so a like-for-like cross-dataset transfer is not currently well defined; we will state this and frame domain adaptation as planned future work (Sec. Limitations). On small/occluded cases: we retained 62 background images to suppress false positives, and our statistics already show many small intersections are represented 

1331 1332 1333 

1334 

2026-06-17 09:20. Page 12 of 1–12. 

