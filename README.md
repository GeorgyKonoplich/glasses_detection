# glasses_detection

## Requirements
* keras
* python 3.6
* opencv 
* dlib 
* numpy
* albumentations
* sklearn

## Usage
### 1. Download preprocessed celebA dataset [here](https://drive.google.com/file/d/1tCNojxRwQWWdS8oKM1tQI0T-wZTFljzc/view)
```
├── dataset
   └── with_glasses
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
   ├── without_glasses
       ├── zzz.jpg
       ├── www.png
       └── ...
```

### 2. Train
* python main.py --epoch 10

## Author
Georgy Konoplich
