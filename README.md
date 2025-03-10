# UCSD MentalHealth Bot

<a href='https://junyuelin.github.io/UCSD-MentalHealth-Bot/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://drive.google.com/file/d/1drG4iNLO4BLdvc93x-t30T9NeCIvjBtO/view'><img src='https://img.shields.io/badge/Report-PDF-blue'> <a href='https://drive.google.com/file/d/1ZVkDBPWi54AnFdzfzN2dQqMrwsYt-Xt_/view'><img src='https://img.shields.io/badge/Poster-PDF-red'>

This is a mental health chatbot targeted at UCSD students, primarily to inform students of resources available to them as well as provide a nonjudgemental space for them to reach out for help. 

# Getting Started 

To build the environment, you will need to take these steps:
```
conda create --n mentalHealthBot python = 3.10
conda activate mentalHealthBot
pip install -r requirements.txt
```

You will also need to create a .env file in the project folder and set the environment variable to your own OpenAI API key:
```
OPENAI_API_KEY = YOUR_KEY
```

To interact with our chatbot application you will run the following:
```
cd scripts
streamlit run app.py
```

This will open up the chatbot application locally and allow you to talk with it. You should have the following on your screen:

![image](https://github.com/user-attachments/assets/3e04efad-f5ff-4977-a8b7-daea9f9375d9)


When talking to the chatbot, it is expected to direct users to UCSD mental health resources, as well as provide emotional validation and support. Resources shown will be real and accurate (as of Winter 2025), and the chatbot will not be able to assists in non-mental health topics (ex. cooking, coding, doing homework, etc.). If the chatbot receives a message flagged as a crisis, users will be directed to emergency resources. 

# Model Comparison

To compare our chatbot to our baseline, you can run the following:
```
# assuming you are still pathed to scripts
cd ..
cd test
python evaluate.py
```

This will write the results of both our model with guardrails and our baseline model of just RAG to output_responses.csv.
