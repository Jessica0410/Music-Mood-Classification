# Music Mood Classification System (MMCS)
This project is my final year project and I uploaded here hoping that this can help you. Please do not copy and paste straight away.

## Project Briefing
This project aims at developing a music mood classification system using audio features only by CNN. The overall accuracy of this system
reached more than 0.90. There two classifiers, the first one is using 1D input data format and the second is 2D input data format.

### Datasets

#### Dataset 1: 4Q Audio Emotion Dataset (Russell)
- 900 ~30 second clips gathered from AllMusic API
- The files are organized in 4 folders (Q1 to Q4) 255 for each folder
- Two metadata csv files with annotations and extra metadata (I didn't use this)

Source: [http://mir.dei.uc.pt/downloads.html](http://mir.dei.uc.pt/downloads.html)

**If you use it, please cite the following article(s):** 

PDF Panda R., Malheiro R. & Paiva R. P. (2018). "Novel audio features for music emotion recognition". 
IEEE Transactions on Affective Computing (IEEE early access). DOI: 10.1109/TAFFC.2018.2820691.

PDFPanda R., Malheiro R., Paiva R. P. (2018). "Musical Texture and Expressivity Features for Music Emotion Recognition". 
19th International Society for Music Information Retrieval Conference -- ISMIR 2018, Paris, France.

#### Dataset 2: Turkish Music Emotion Dataset
- 400 ~30 second from Turkish verbal and non-verbal music from diffrent genre.
- The files are classified into 4 classes: Happy, Angry, Sad and Relaxed with each 100 files.

