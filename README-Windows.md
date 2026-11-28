# Training YOLO Model

We are going to train a YOLO model.

## Installation

Go to 

`https://github.com/darrylhill23/yolo-workshop`

Click Code → Download ZIP.

Extract the ZIP file.

Double-click setup.bat. This will take a while to download the necessary Python libraries. Once it is down, it should give you a message and the terminal should remain open. This terminal is where you will type your commands.




## Training Your Model

### Record Your Object

Find something distinctive you want to track. It should have a distinctive pattern and/or colour. Run the command:

`python record.py`

The webcam should turn on and begin recording you. Show the object in various poses, and record some short segments where the object is not visible. Aim for about 40 seconds to a minute (much longer and you will have a lot of images to label). Press escape when done. If you are happy with the recording, proceed to the next step. Otherwise run `python record.py` again.

### Chop Your Recording into Frames

Type the command

`python extract-frames.py`. 

This will extract frames from your video at 1 second intervals and add them to a **candidates** folder. If you have already done this step, there will be other images in **candidates**, which is perfectly fine. 

### Label Your Training Data

This is probably the most tedious part of being a data scientist. The images will be displayed for you, and you will draw a bounding box around the object. You are essentially telling the AI what you want it to return if it sees your image. In this case, it should return a rectangle that "bounds" your object. 

Run the command:

`python label.py`

A window will open which will show your frames one at a time. If the particular frame does not contain the object you are tracking, just press enter. Otherwise, draw a bounding box around your object (click and drag your mouse) and hit enter. If you mess it up, try again. If you accidentally draw a bounding box and you want to cancel it, press "delete" or "backspace".

This will create a **dataset** folder with some sub-folders, where it will put the images a long with the label (the bounding box you drew). If **dataset** already exists, it will add more data to it.

### Split Data Into Training and Validation

Initially all your data is in the training data folder, but you need some validation data (data that is used to see if the model is working). Any time you have added data, afterwards run the command:

`python val-split.py`

This will ensure about 20% of your data is reserved for validation. If you do not do this, the model will not train correctly. 

### Train Your Model

If you want to change the name of the object, you can edit the data.yaml file.
You may also wish to make some changes to `train-model.py` (you don't have to, but you can if you like). Here are some things you can do:

1. If you have a discrete GPU, change "cpu" to 0. Your model will train much faster.
2. The training will take about 5-10 minutes. If it looks like it will take longer, you can decrease the number of epochs (though it might not work as well). It is currently set to 150 - you may try 100 if the training is taking too long.

Once you are happy with the setup, run  

`python train-model.py`

On a cpu, this can take a while, 5-10 minutes. A gpu should be quite fast, a few minutes at most. 

### Test Your Model. 

Once it is done training, and it saved your model, you can try it and see how well it trained. Run 

`python cam-test-model.py` 

It may take a little longer as it loads the AI model. Once the screen appears, you can show your object to the camera and see if it is tracking correctly. If you turned down the epochs too much, it may not be working very well. 

If it is working well you can try the game. Run 

`python ufo_attack.py` 

Try to avoid the UFOs and their attacks. 

### Advanced Training

If you chose a high contrast object, your model is probably working quite well. But maybe you chose an object that is harder to identify and requires more training. We are going to record more data, but then use the model to help label the data. Rename the 'candidates' folder to 'candidates-old'. Rename 'dataset' to 'dataset-old'.

Record a longer video (60 seconds or more), againg by running

`python3 record.py`

Extract the frames as before, run:

`python extract-frames.py`. 

Now, instead of manually labelling everything, we will use the model to help. Run 

`python label-with-inference.py`

This will display your frames, as well as the bounding box the model guessed. If the bounding box looks good to you, hit enter. If it does not look good, hit 'n'. The frames for which you select 'n' will be left for you to manually label. 

Once that is complete, you can label the data that the model got wrong. Run 

`python label.py` 

to label the remaining data the old-fashioned way. 

Once you are done with that, you will want to merge it with the old data. Run

`python merge-old-data.py`

Once again, you will want some of it to be validation data. Run

`python val-split.py`

Retrain your model:

`python train-model.py`

Test it out. It should be working better. 

`python cam-test-model.py`

