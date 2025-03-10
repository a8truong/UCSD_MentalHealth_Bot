# UCSD MentalHealth Bot

This is a mental health chatbot targeted at UCSD students, primarily to inform students of resources available to them as well as provide a nonjudgemental space for them to reach out for help. 

To run the code, conda create --n envName python = 3.10, and then pip install -r requirements.txt and create a .env file in the project folder and set environment variable OPENAI_API_KEY = Your own OpenAI API key. 

With the conda environment activated, cd scripts and run streamlit run app.py. This will open up the chatbot application locally and allow you to talk with it. 

When talking to the chatbot, it is expected to direct users to UCSD mental health resources, as well as provide emotional validation and support. Resources shown will be real and accurate (as of Winter 2025), and the chatbot will not be able to assists in non-mental health topics (ex. cooking, coding, doing homework, etc.). If the chatbot receives a message flagged as a crisis, users will be directed to emergency resources. 

For more information, check out our website [here](https://junyuelin.github.io/UCSD-MentalHealth-Bot/). Our [report](https://docs.google.com/document/d/1TzYtUBsK8DJXdEISnxdlLfpRIt-m98zRv6dpxGGuuN8/edit?usp=drive_link) will contain more in-depth information on our methods.
