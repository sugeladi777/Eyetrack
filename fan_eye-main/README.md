# fan_eye

The repository is based on [fan_training](https://github.com/hhj1897/fan_training).

# Install
## Step 1: Create Conda Environment
```
# python=3.8 and 3.9 have been tested
conda create -n fan_eye python=3.9
conda activate fan_eye
```
## Step 2: Install Pytorch
Find the suitable version from [Previous Pytorch Versions](https://pytorch.org/get-started/previous-versions/).
```
# pytorch==1.12.1 with cudatoolkit=11.3 has been tested
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
## Step 3: Install Other Tool Packages
```
pip install -r requirements.txt
```

## Step 4: Install Face Detector
We use [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md) as our face detector.
```
# mediapipe==0.10.14 has been tested
pip install mediapipe
```

# Train
```
# Train face landmark predictor
bash run_train.sh

# Train eye landmark predictor (using eye-region images)
bash run_train_eye.sh

# Train eye landmark predictor (using full-face images)
bash run_train_eyefromface.sh
```

# Convert Pytorch Model to Onnx Model
```
python temp_ckpt_to_pth_v[x].py     # x = 3 or 4
```

# Test
```
cd landmark_detector
$work_dir=<work_dir>
python main.py \
    --ckpt ../pretrained/mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx \
    --data "/home/lichengkai/Eyetrack/image" \
    --out "/home/lichengkai/Eyetrack/landmark" \
    --savelmk --usefilter
```

# Tips
* fan4 may be too lightweight to support 106-p facial landmark localization.
* eye-region cropping is important for great performance.
* alpha blending may worsen the performance (need further confirmation)

---

## Face Detector (abandoned)
We use [this repository](https://github.com/hhj1897/face_detection) as our face detector.
```
conda activate ibug

git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull            # Or manually download the .pth files
pip install -e .
```

## Face Detector (abandoned)
We use [dlib19.21](https://github.com/davisking/dlib) as our face detector.

python main.py --ckpt ../pretrained/mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx --data $work_dir/image --out $work_dir/landmark --savelmk --usefilter