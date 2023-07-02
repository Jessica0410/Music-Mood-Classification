# Music Mood Classification System (MMCS)
This project is my final year project and I uploaded here hoping that this can help you. Please do not copy and paste straight away.

## Project Briefing
This project aims at developing a music mood classification system using audio features only by CNN. The overall accuracy of this system reached more than 0.90. There two classifiers, the first one is using 1D input data format and the second is 2D input data format. 1D classifer reached overall 0.91 accuracy and 2D classifier reached overall 0.96 accuracy. And finally, I did a simple testing user interface.

## Programming Languages Used:
1. Python
   - Audio processing: Pydub, Ipython
   - Feature extraction: Librosa
   - Data processing: Numpy, Pandas, Sklearn
   - Classification: Tensorflow
   - Model Evaluation: Matplotlib
   - Web development: Django
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

**Related file:**
- [MMC Feature Extraction 1D](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Feature%20Extraction%201D.ipynb)
- [MMC Feature Extraction 2D](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Feature%20Extraction%202D.ipynb)

### Data Preparation
Before training, we need to preprocess the data. We need to split features and labels, train/val/test set split, features and labels encoding in numeric way(1D features), resize the features(2D features), features scaling for each set and finally reshaped the data which fit the classifiers. 

### Classification & Model Evaluation
#### 1D classifier
I designed 5 layers and 4 layers CNN, consisting of 3(2) conv1D and maxpooling layers and 2 fully connected layers.
Dataset1 made the model overfit, and I tried following methods to tackle the problems:
- Add regularizer
- Add dropout
- Reduce number of features
- Simplify the network
They all didn't work. So, I changed dataset to dataset2. The the performance of model trained by Dataset2 improved a lot. (82% accuracy)

**Related file:**
[MMC Classification 1D](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Classification%201D.ipynb)

#### 2D classifier
I designed 5 layers CNN, consisting of 3 conv2D and maxpooling layers and 2 fully connected layers. 
Dataset1 still made the model overfit but better than using 1D classifier. I didn't try to solve the overfit problem here, instead, I changed to dataset2 straight away. 
I used spectrogram, mfccs and mel-spectrogram to train the model respectively and tried different resizing size. If we resized to small size like 64 x 64, the model will suffer from unfitting problems, as infomation lost a lot, but if it's too large, like 1000 x 1000 it will be very time consuming and requires large storage. So after different tries, I used 300 x 300 for spectrogram, 600 x 120 for mfcc and 400 x 300 for mel-spectrogram. Finally I emsmebled three models by voting for majority. The model achieved 80% overall accuracy.

**Related file:**
[MMC Classification 2D](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Classification%202D.ipynb)

### EDA and Audio Feature Visualization
In order to find the reason why the dataset2 is better than dataset1, I used Boxplot and Histogram in R to show the different mean distribution of 1D features for different mood classes. And also I visualized the audio features using python Matplotlib. The EDA results shows that the value distribution of features for each mood class in dataset2 are distinct. So that's the reason why dataset2 is better.
**Related file:**
- [EDA dataset1](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/EDA1.html)
- [EDA dataset2](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/EDA2.html)
- [Features Visualization](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Feature%20Visualization.ipynb)

### Data Augmentation
Although the results using dataset2 reached 80% accuracy which meet my expectation. But there still some problems with this dataset. 
The training process for both 1D and 2D classifers actually is very unstable, and the difference between validation accuracy and testing accuracy is large (eg. val_acc: 0.78 test_acc: 0.90), which may lead to the less robust of the system. Below are one example of it. And this is due to the insufficient data in dataset.

<img width="470" alt="Screen Shot 2023-06-25 at 13 05 43" src="https://github.com/Jessica0410/Music-Mood-Classification/assets/69900031/f6fb7a50-8a05-450b-87e8-55f83276735f">

So, I did audio data augmentation using two techniques: adding noise and time shifting. And my dataset was enlarged from 400 to 1200. And we put all the augmented data into training and validation set. The results become better. Which leads to stable training process, higher model accuracy and less difference between val_acc and test_acc.  Below is one example of training accuracy:

<img width="467" alt="Screen Shot 2023-06-25 at 13 11 29" src="https://github.com/Jessica0410/Music-Mood-Classification/assets/69900031/85e7a327-6ac2-4e37-90cd-37616c1f476b">

The model using augmented data reached 94% model accuracy (1D classifier) and 93% accuracy (2D classifier).
**Related file:**
- [MMC Audio Augmentation](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Audio%20Augmentation.ipynb)
- [MMC Classification with Augmented Data 1D](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Classification%20With%20Augmented%20Data%201D.ipynb)
- [MMC Classification with Augmented Data 2D](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/MMC%20Classification%20With%20Augmented%20Data%202D.ipynb)


### Model Testing
#### Python File Testing
For testing the model, you can enter any song you like and model_num (1 or 2) to decide which classifier you want to use, 1 for 1D classifer and 2 for 2D classifier. Run all relate code and type in filename and model num to get the predicted mood. Noted, the song must be downloaded and stored in the folder [SampleAudio](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc/templates/SampleAudio) under [templates](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc/templates) of [mmc](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc) folder. 

**Related file:**
[MMC Model Testing](https://github.com/Jessica0410/Music-Mood-Classification/blob/main/Model%20Testing.ipynb)

#### Web based UI for testing
If you want to a more easily used and aesthetic testing interface, you first change the directory to [mmc](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc) (depends on where you store in your computer) and enter *python manage.py runserver* in terminal to launch the website. Noted, the song must be downloaded and stored in the folder [SampleAudio](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc/templates/SampleAudio) under [templates](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc/templates) of [mmc](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc) folder. And you **must** upload song from [SampleAudio](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc/templates/SampleAudio) Folder.

**User interface is like below:**
<img width="1440" alt="Screen Shot 2023-06-25 at 13 29 53" src="https://github.com/Jessica0410/Music-Mood-Classification/assets/69900031/2d1a0476-9836-4421-a905-bf3b9a31165e">
<img width="1440" alt="Screen Shot 2023-06-25 at 13 31 06" src="https://github.com/Jessica0410/Music-Mood-Classification/assets/69900031/a0a39c09-78de-49d1-a02c-161e3b6bb40c">
<img width="1439" alt="Screen Shot 2023-06-25 at 13 31 38" src="https://github.com/Jessica0410/Music-Mood-Classification/assets/69900031/fa463fbe-013c-4cf8-8a92-2b53b0f31d84">

**Related folder**:
[mmc](https://github.com/Jessica0410/Music-Mood-Classification/tree/main/mmc)









