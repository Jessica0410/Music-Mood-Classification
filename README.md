# Music Mood Classification System (MMCS)
This project is my final year project and I uploaded here hoping that this can help you. Please do not copy and paste straight away.

## Project Briefing
This project aims at developing a music mood classification system using audio features only by CNN. The overall accuracy of this system reached more than 0.90. There two classifiers, the first one is using 1D input data format and the second is 2D input data format. 1D classifer reached overall 0.91 accuracy and 2D classifier reached overall 0.96 accuracy. And finally, I did a simple testing user interface.

## Programming Languages used:
1. Python
   - Audio processing: Pydub, Ipython
   - Feature extraction: Librosa
   - Data processing: Numpy, Pandas, Sklearn
   - Classification: Tensorflow
   - Model Evaluation: Matplotlib
   - Web development: Django, Json
2. R Language
   - EDA: Tidyverse
3. HTML/CSS/Javascript

### Datasets
#### Dataset 1: 4Q Audio Emotion Dataset (Russell)
- 900 ~30 second clips gathered from AllMusic API
- The files are organized in 4 folders (Q1 to Q4)
- Two metadata csv files with annotations and extra metadata (I didn't use this)
- A total of 225 music pieces are determined for each class in the database to have an equal number of samples in each class.

Source: [http://mir.dei.uc.pt/downloads.html](http://mir.dei.uc.pt/downloads.html)

**If you use it, please cite the following article(s):** 

PDF Panda R., Malheiro R. & Paiva R. P. (2018). "Novel audio features for music emotion recognition". 
IEEE Transactions on Affective Computing (IEEE early access). DOI: 10.1109/TAFFC.2018.2820691.

PDFPanda R., Malheiro R., Paiva R. P. (2018). "Musical Texture and Expressivity Features for Music Emotion Recognition". 
19th International Society for Music Information Retrieval Conference -- ISMIR 2018, Paris, France.

#### Dataset 2: Turkish Music Emotion Dataset
- 400 ~30 second from Turkish verbal and non-verbal music from diffrent genre
- The files are classified into 4 classes: Happy, Angry, Sad and Relaxed
- A total of 100 music pieces are determined for each class in the database to have an equal number of samples in each class

source: [https://www.kaggle.com/datasets/blaler/turkish-music-emotion-dataset](https://www.kaggle.com/datasets/blaler/turkish-music-emotion-dataset)

**If you use it, please cite the following article(s):** 

Bilal Er, M., & Aydilek, I. B. (2019). Music emotion recognition by using chroma spectrogram and deep visual features. 
Journal of Computational Intelligent Systems, 12(2), 1622–1634. International Journal of Computational Intelligence Systems, 
DOI: https://doi.org/10.2991/ijcis.d.191216.001

### Audio Preprocessing
We first check the length of all the audio, remove all the audios less than 25.5s and cut all the audio clips to 25.5s

### Feature Extraction
For 1D data input, I extracted tempo, key, scale, mean and variance of mfcc, root mean square, mel-spectrogram, tonnetz, chroma features, zero crossing rate, spectral roll-off, spectral centroid overall 57 input features and store them in an excel file.

For 2D data input, I extracted mel, mfcc, spectrogram only, as a few features can achieve high accuracy for 2D classifier. And stored them in .npz file.

### Data Preparation
Before training, we need to preprocess the data. We need to split features and labels, train/val/test set split, features and labels encoding in numeric way(1D features), resize the features(2D features), features scaling for each set and finally reshaped the data which fit the classifiers. 

### Classification
#### 1D classifier
I designed 5 layers and 4 layers CNN, consisting of 3(2) conv1D and maxpooling layers and 2 fully connected layers.
Dataset1 made the model overfit, and I tried following methods to tackle the problems:
- Add regularizer
- Add dropout
- Reduce number of features
- Simplify the network
They all didn't work. So, I changed dataset to dataset2. The the performance of model trained by Dataset2 improved a lot. (about 80% accuracy)

#### 2D classifier
I designed 5 layers CNN, consisting of 3 conv2D and maxpooling layers and 2 fully connected layers. 
Dataset1 still made the model overfit but better than using 1D classifier. I didn't try to solve the overfit problem here, instead, I changed to dataset2 straight away. 
I used spectrogram, mfccs and mel-spectrogram respectively and tried different resizing size. If we resized to small size like 64 x 64, the model will suffer from unfitting problems, as infomation lost a lot, but if it's too large, like 1000 x 1000 it will be very time consuming and requires large storage. So after different tries, I used 200 x 200 for spectrogram, 600 x 120 for mfcc and 400 x 300 for mel-spectrogram. And each model



