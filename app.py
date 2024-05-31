from flask import Flask, render_template, request, jsonify
from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os


def ask_document(**kwargs):
    load_dotenv()
    knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID")
    credentials_profile_name = os.getenv("CREDENTIALS_PROFILE_NAME")
    model_id = os.getenv("MODEL_ID")

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id,
        credentials_profile_name=credentials_profile_name,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 8}},
    )

    model_kwargs_claude = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_gen_len": 600,
    }

    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs=model_kwargs_claude,
        credentials_profile_name=credentials_profile_name,
    )

    system_prompt = (
        "Vous êtes un assistant qui aide les utilisateurs à trouver des aides publiques pour leur logement. Vous recevrez une série de documents comme contexte et devrez répondre à la question de l'utilisateur sur la base de ses données, qui seront également fournies. Idéalement, vous devriez être en mesure de trouver une aide que l'utilisateur peut recevoir, mais si vous avez un doute, n'inventez rien, dites que vous n'êtes pas sur. "
        "Vous ne devez pas répéter les informations de l'utilisateur. Au maximum, dites bonjour + nom"
        "Essayez d'être pragmatique et cherchez l'aide spécifique que l'utilisateur peut recevoir dans sa situation (nombre de personnes, salaire, lieu)."
        "Info sur l'utilisateur: "
        f"Nom: {kwargs['user_name']}"
        f"Ville: {kwargs['city']}"
        f"Plage de salaire: {kwargs['salary_range']}"
        f"Nombre de personnes dans la maison: {kwargs['numer_of_people_on_house']}"
        f"Type de travaux envisagés: {kwargs['work_types']}"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    query = f"{kwargs['query']} ville:{kwargs['city']} salaire:{kwargs['salary_range']} nombre_personnes:{kwargs['numer_of_people_on_house']} type_travaux:{kwargs['work_types']}"
    response = chain.invoke({"input": query})

    return response["answer"]


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask_document", methods=["POST"])
def ask_document_route():
    print("Started ask_document_route")
    query = request.form["query"].strip('"')
    user_name = request.form["user_name"].strip('"')
    city = request.form["city"].strip('"')
    salary_range = request.form["salary_range"].strip('"')
    numer_of_people_on_house = request.form["number_of_people_in_house"].strip('"')
    work_types = request.form["work_types"].strip('"')


    args_dict = {
        "query": query,
        "user_name": user_name,
        "city": city,
        "salary_range": salary_range,
        "numer_of_people_on_house": numer_of_people_on_house,
        "work_types": work_types,
    }

    response = ask_document(**args_dict)

    print(response)

    response = {"response": response}

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
