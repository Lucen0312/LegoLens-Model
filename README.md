# LegoLens-Model
Had to create this again due to merge conflicts from PC and macbook.
Github Copilot was used in limited way in terms of debugging.
AI was only used for some errors that weren't able to be found online.
## Process of creating this model
Used mobilenet to do image recognition but discovered that object detection is far better than image classification for my usage.
Used Tensorflow's Object Detection API but discovered that it was deprecated.
Used KerasCV and Keras 3 with YOLOv8 which are both modern solutions for object detection fine tuning.

## Dataset
Dataset were annotated using labelImg which annotates images by PascalVOC format.

## To-do
- [x] Use ultralytics instead of Keras for fine tuning which is YOLOv8's official provider.
- [x] Add 4 more classes to be proven as POC with class of 5. (Proof of Concept)
- [x] Increase data with 50 images per set and apply data augmentation.
- [x] Increase epoch to 30 from 10 for higher accuracy.
