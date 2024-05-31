from flask import Flask, render_template, request, jsonify
from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os


def get_income_category(num_people, income, location):
    # Define the base thresholds for each category for both regions
    income=int(income)
    num_people=int(num_people)
    thresholds_hors_idf = {
        1: {'tres_modestes': 17009, 'modestes': 21805, 'intermediaires': 30549},
        2: {'tres_modestes': 24875, 'modestes': 31889, 'intermediaires': 44907},
        3: {'tres_modestes': 29917, 'modestes': 38349, 'intermediaires': 54071},
        4: {'tres_modestes': 34948, 'modestes': 44802, 'intermediaires': 63235},
        5: {'tres_modestes': 40002, 'modestes': 51281, 'intermediaires': 72400},
    }
    
    thresholds_idf = {
        1: {'tres_modestes': 23541, 'modestes': 28657, 'intermediaires': 40018},
        2: {'tres_modestes': 34551, 'modestes': 42058, 'intermediaires': 58827},
        3: {'tres_modestes': 41493, 'modestes': 50513, 'intermediaires': 70382},
        4: {'tres_modestes': 48447, 'modestes': 58981, 'intermediaires': 82839},
        5: {'tres_modestes': 55427, 'modestes': 67457, 'intermediaires': 94844},
    }
    
    # Additional amount per supplementary person
    additional_amounts_hors_idf = {'tres_modestes': 5045, 'modestes': 6462, 'intermediaires': 9165}
    additional_amounts_idf = {'tres_modestes': 6970, 'modestes': 8486, 'intermediaires': 12006}
    
    # Select the appropriate thresholds based on location
    if location == 'ile_de_france':
        thresholds = thresholds_idf
        additional_amounts = additional_amounts_idf
    else:
        thresholds = thresholds_hors_idf
        additional_amounts = additional_amounts_hors_idf
    
    # Calculate thresholds for households with more than 5 people
    if num_people > 5:
        base_thresholds = thresholds[5]
        extra_people = num_people - 5
        for category in base_thresholds:
            base_thresholds[category] += extra_people * additional_amounts[category]
        thresholds[num_people] = base_thresholds
    
    # Get the thresholds for the given number of people
    current_thresholds = thresholds.get(num_people, thresholds[5])
    
    # Determine the income category
    if income <= current_thresholds['tres_modestes']:
        return 'Menages aux revenus tres modestes'
    elif income <= current_thresholds['modestes']:
        return 'Menages aux revenus modestes'
    elif income <= current_thresholds['intermediaires']:
        return 'Menages aux revenus intermediaires'
    else:
        return 'Menages aux revenus superieurs'

    

def ask_document(**kwargs):

    revenue_class = get_income_category(kwargs['numer_of_people_on_house'], kwargs['salary_range'], kwargs['city'])

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
        "temperature": 0.2,
        "top_p": 0.9,
    }

    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs=model_kwargs_claude,
        credentials_profile_name=credentials_profile_name,
    )

    system_prompt = (
        "Vous êtes un assistant qui aide les utilisateurs à trouver des aides publiques pour leur logement. Vous recevrez une série de documents comme contexte et devrez répondre à la question de l'utilisateur sur la base de ses données, qui seront également fournies. Idéalement, vous devriez être en mesure de trouver une aide que l'utilisateur peut recevoir, mais si vous avez un doute, n'inventez rien, dites que vous n'êtes pas sur. "
        "Vous ne devez pas répéter les informations de l'utilisateur. Au maximum, dites bonjour + nom"
        f"Essayez d'être pragmatique et cherchez l'aide spécifique que l'utilisateur à demander ({kwargs['query']}) et que il puisse recevoir dans sa situation de plafond de revenu {revenue_class}"
        "Ce sont les types de plafonds de revenu :"
        "Menages aux revenus tres modestes, Menages aux revenus modestes, Menages aux revenus intermediaires, Menages aux revenus superieurs"
        "Soyez aussi succinct que possible. Et pour chaque type de travail demandé par l'utilisateur, fournissez-le sur une nouvelle ligne avec des puces (*), par exemple :"
        "Isolation: ..."
        "Info sur l'utilisateur: "
        f"Nom: {kwargs['user_name']}"
        f"Plafond revenu: {revenue_class}"
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
    query = f"{kwargs['query']} type_travaux:{kwargs['work_types']} Plafond revenu:{revenue_class}"
    response = chain.invoke({"input": query})

    return response["answer"]


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask_document", methods=["POST"])
def ask_document_route():
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

    response = {"response": response}

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
