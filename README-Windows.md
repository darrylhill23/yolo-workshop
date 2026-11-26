# Training YOLO Model

We are going to train a YOLO model.

## Installation

### 1. Download the ZIP file

Go to 

`https://github.com/darrylhill23/yolo-workshop`

Click Code → Download ZIP.

Extract the ZIP file.

Double-click setup.bat.




### 3. Training Steps

1. Find something distinctive you want to track. It should have a distinctive pattern and/or colour. Record a video showing it in various places at various angles using

`python record.py`

Press escape when done. Aim for about 40 seconds to a minute.

2. Chop it into frames.

Run `python extract-frames.py`. This will extract frames at 1 second intervals and put them into a **candidates** folder.

3. Label this training data. Run

`python label.py`

If the particular frame does not contain the object you are tracking, hit 'n'. Otherwise, draw a bounding box around your object (click and drag your mouse) and hit enter. This will create a **dataset** folder with some sub-folders.

4. Split into training data and validation data.

`python val-split.py`

5. Train your model. In data.yaml, change the label to whatever you like. In train-model.py, change device to "cpu" if you do not have a nvidia gpu. Then run

`python train-model.py`

On a cpu, this can take a while. A gpu should be quite fast. 

6. Test your model. Run `python3 cam-test-model.py`. It should be tracking your object, though possibly not very well. 

7. We are going to record more data. Rename 'candidates' to 'candidates-old'. Rename 'dataset' to 'dataset-old'.

8. Record a longer video (60 seconds or more) using `python3 record.py`.

9. Run `python extract-frames.py`. This will refill the candidates folder.

9. Instead of manually labelling everything, run 

`python label-with-inference.py`

It will display the frames with the models guess of where the bounding box should go. If the frame is properly labeled, hit enter. If not, hit 'n'. The frames for which you select 'n' will be left for you to manually label. 

Once that is complete, run `python label.py` to label the remaining data. 

10. Again, split into training data and validation data.

`python val-split.py`

11. Merge the old data, run

`python merge-old-data.py`

12. Retrain your model. 

`python train-model.py`

13. Test it out. It should be working better. 

`python cam-test-model.py`

