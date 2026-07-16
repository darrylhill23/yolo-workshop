# Training YOLO Model

Short description of what this project does.

## Installation

### 1. Clone the repository

git clone https://github.com/darrylhill23/yolo-workshop
cd yolo-workshop

### 2. Make a Python virtual environment

Navigate to the yolo-workshop folder.

#### On Windows Powershell:

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

#### On Windows Terminal:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

#### On Linux:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#### On Mac

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 3. Training Steps

1. Find something distinctive you want to track. It should have a distinctive pattern and/or colour. Record a video showing it in various places at various angles using

python3 record.py

Press escape when done. Aim for about 40 seconds to a minute.

2. Chop it into frames.

Run python3 extract-frames.py. This will extract frames at 1 second intervals and put them into a candidates folder.

3. Label this training data. Run

python3 label.py

If the particular frame does not contain the object you are tracking, hit 'n'. Otherwise, draw a bounding box around your object.

4. Split into training data and validation data.

python3 val-split.py

5. Train your model. In data.yaml, change the label to whatever you like. In train-model.py, change device to "cpu" if you do not have a nvidia gpu. Then run

python3 train-model.py

On a cpu, this can take a while. A gpu should be quite fast. 

6. Test your model. Run python3 cam-test-model.py. It should be tracking your object, though possibly not very well. 

7. We are going to record more data. Rename 'candidates' to 'candidates-old'. Rename 'dataset' to 'dataset-old'.

8. Record a longer video (60 seconds or more) using python3 record.py.

9. Run python3 extract-frames.py. This will refill the candidates folder.

9. Instead of manually labelling everything, run 

python3 label-with-inference.py

If the frame is properly labeled, hit enter. If not, hit 'n'. This will leave the improperly labeled data in candiates. 

Once that is complete, run python3 label.py to label the remaining data. 

10. Again, split into training data and validation data.

python3 val-split.py

11. Merge the old data, run

python3 merge-old-data.py

12. Retrain your model. 

python3 train-model.py

13. Test it out. It should be working better. 

python3 cam-test-model.py

