# CS5260-PromptPilot

## Dataset

<!-- [Download UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) -->

[Download filtered DIV2K dataset](https://drive.google.com/drive/folders/1q01buDiVBR-d9cPTFvBsktBlbrfWEwUv?usp=drive_link)

## How to use the Tool

### Run

`pip install streamlit` <br>
Assuming your current working path is the project folder, in terminal window, type `streamlit run tool_app/app.py`

### Functionality

Tab `Reward Network Annotation` <br>

1. input the absolute path of any dataset path you want to annotate <br>
2. input your nickname <br>

Tab `Reward Annotation Training` <br>

1. put all annotators' csv files in one folder (by default, `user_marks`, but you could use any other names). Different annotator is put in separate files.
2. input the foler path, and press Enter.
3. pre-processed result will show in the table.
4. click the model training button.
5. after finish training and saving the model, the test result will show in the table.

Tab `🤖 Video Agent 🤖`

1. click the button to upload an image.
2. type user prompt
3. wait for a while, the inference video will show below (currently mock data)
