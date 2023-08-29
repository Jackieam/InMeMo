### Step1: Feature Extraction
Please extract the image features of train and val of Pascal-5<sup>i</sup>.
#### The pixel-level retriever:
#### val
```
python tools/feature_extractor_detection.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val
```
#### train
```
python tools/feature_extractor_detection.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level train
```
### Step2: Calculate Similarity
We need to generate json file for trn set and val set separately.
#### val
```
python tools/calculate_similariity_detection.py features_vit-laion2b_pixel-level val train
```
#### trn
```
python tools/calculate_similariity_detection.py features_vit-laion2b_pixel-level train train
```
