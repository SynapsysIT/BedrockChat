# Easy DocuBot

This project is an easy setup for a "Talk to your docs" ChatBot using documents stored on AWS S3 and AWS Bedrock Models and Knowledge Base. LangChain was used to orchestrate the chain that uses your documents.

The use case as example here is a ChatBot to ask questions about the possibility of getting government aid in Paris for house rennovation.  

Clone the repository to your local machine.

Install the required Python packages. You can do this by running the following command in your terminal:

```
pip install -r requirements.txt
```

You need to create an .env file on your local repo and setup this three env variables:

```
KNOWLEDGE_BASE_ID
CREDENTIALS_PROFILE_NAME
MODEL_ID
```


Run the application. You can do this by running the following command in your terminal on the root of the project:

```
flask run 
```

The application will start running on your local machine, and you can access it by navigating to http://localhost:5000 in your web browser.