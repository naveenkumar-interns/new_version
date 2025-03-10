
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from datetime import datetime
import pymongo
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List

import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader



load_dotenv()

GOOGLE_API_KEY = "AIzaSyDR5hSTYjo6jbiTpHw8AEKZsuRVEEFcAJk"
hf_token = "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"
MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

model_1 = "gemini-1.5-flash"
model_2 = "gemini-2.0-pro-exp-02-05"
model_3 = "gemini-2.0-flash-lite"




client = pymongo.MongoClient(MONGO_URI)
db = client["jacksonHardwareDB"]
collection = db["inventory"]
user_client = pymongo.MongoClient("mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0")
user_db = user_client["chatbot"]
chat_history_collection = user_db["chats"]



app = Flask(__name__)
CORS(app)


llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)



def get_data(url):

    loader = WebBaseLoader(url)
    docs = loader.load()

    cleaned_docs = []  
    for doc in docs:
        content = doc.page_content
        # Remove excessive whitespace and newlines
        content = re.sub(r'\s+', ' ', content).strip()
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        # Remove repetitive headers and footers
        content = re.sub(r'Visit Our Hardware Store Today.*', '', content, flags=re.DOTALL)
        
        # Update the page_content within the doc object
        doc.page_content = content
        cleaned_docs.append(doc)

    return cleaned_docs


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=GOOGLE_API_KEY)

vector_store = InMemoryVectorStore(embeddings)

web_list = ["https://www.jacksonshardware.com/contact-us","https://www.jacksonshardware.com/JacksonsHardwareStory","https://www.jacksonshardware.com/san-rafael-ca","https://www.jacksonshardware.com/about","https://www.jacksonshardware.com/faqs","https://www.jacksonshardware.com/careers"]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits = text_splitter.split_documents(get_data(web_list))

# Index chunks
_ = vector_store.add_documents(documents=all_splits)







embedding_cache ={}

def generate_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-l6-v2")
    response = embeddings.embed_query(text)
    embedding_cache[text]=response
    return response


def convert_to_json(data):
    result = []
    forai = []
    for product in data:
        # Filter out unnecessary keys from metadata
        product_info = {
        'id': product.get('id'),
        'title': product.get('title'),
        'description': product.get('description'),
        'product_type': product.get('product_type'),
        'link': product.get('link'),
        'image_list': product.get('image_list'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity'),
        'vendor': product.get('vendor')
        }
        result.append(product_info)

    print(result)

    return result,forai


def get_product_search(query):
    results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "embeddings",
        "numCandidates": 100,
        "limit": 8,
        # "index": "vector_search_index",
        "index": "vx",
        }}
    ])
    return convert_to_json(results)

def analyze_intent(query):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",  # Assuming this is a valid model; adjust if needed
        temperature=0.7,
        max_tokens=60,
        timeout=None,
        max_retries=2,
        google_api_key="AIzaSyDR5hSTYjo6jbiTpHw8AEKZsuRVEEFcAJk"
    )
    try:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a senior data analyst. Analyze the user query and categorize the intent as:
        1. **'product'** (asking about products),
        2. **'website/company'** (asking about the website, company, or FAQs related to services, policies, or operations),
        3. **'general'** (queries unrelated to products, website, or company).

        Return only the category name (e.g., "product", "website/company", or "general"). Do not include preambles, explanations, or additional text.

        **Examples:**
        - Query: "What shoes do you have?" → product
        - Query: "Where can I find promotions going on?" → website/company
        - Query: "Can we come and check the products?" → website/company
        - Query: "How does your website work?" → website/company
        - Query: "Why is the sky blue?" → general
        - Query: "Tell me about your company" → website/company
        - Query: "Any deals on laptops?" → product
        - Query: "What’s your return policy?" → website/company
        - Query: "Why we connect?" → general

        **FAQs to Categorize as 'website/company':**
        - Query: "Do you have a showroom?" → website/company
        - Query: "How do I apply for a job at Jackson’s Hardware?" → website/company
        - Query: "Do you offer parts and repairs?" → website/company
        - Query: "Do you have a contact phone number?" → website/company
        - Query: "Where can I find promotions going on?" → website/company
        - Query: "What if I return an item due to it being defective?" → website/company
        - Query: "Do you deliver?" → website/company
        - Query: "Do you repair screens?" → website/company
        - Query: "Can I rent a power washer from Jackson’s Hardware?" → website/company
        - Query: "What credit cards do you accept?" → website/company
        - Query: "Is it possible to special order an item?" → website/company
        - Query: "Can we come and check the products?" → website/company
        - Query: "What are Jackson’s Hardware store hours?" → website/company
        """
            ),
            ("human", "{query}")
        ])

        chain = prompt | llm
        response = chain.invoke({"query": query})
        return response.content.strip()
    except Exception as e:
        print(f"Error in analyze_intent: {str(e)}")
        raise

def research_intent(chat_history):
    llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
            """You are a senior research assistant. 
            Analyze the chat history to track the user's current topic and predict their request.
              Accumulate filters (e.g., specifications) until the topic changes, then reset context. 
              Respond with only a phrase summarizing the current request, prioritizing the latest input. No explanations or additional text.

                Examples:
                1. User: I need a heater.
                Bot: Heater
                User: I need a 220V one.
                Bot: Heater 220V
                User: I need it in black.
                Bot: Heater 220V Black
                User: Can you list tables?
                Bot: Table

                2. User: Show me smartphones.
                Bot: Smartphone
                User: I need one with 128GB storage.
                Bot: Smartphone 128GB
                User: Show me refrigerators.
                Bot: Refrigerator

                Analyze the conversation and return the summarizing word or phrase."""
        ),
        ("human", "{chat_history}")
    ])

        chain = prompt | llm

        response = chain.invoke({"chat_history": chat_history})

        print("\n",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in research_intent: {str(e)}")
        raise

def prioritize_products(user_intent, products):

    llm = ChatGoogleGenerativeAI(
    model=model_3,
    temperature=0.7,
    max_tokens=50000,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

    input_str = f"User asks for : '{user_intent}'\n Products we have: {json.dumps(products, indent=2)}"
    try:
        prompt = """ Role: You are a Product Prioritization Expert specializing in ranking products based on user intent, price constraints, and relevance. 
            Your task is to filter, reorder, and return the most relevant products that match the user's intent and budget, colour , product type and other features.

            Rules for Prioritization:
            1. **Match User Intent**: 
            - Prioritize products that contain keywords from the user's intent in the title, description, or product type.
            - Stronger keyword matches (e.g., exact matches in the title) should rank higher.

            2. **Apply Price Constraints**:
            - If a price limit is specified (e.g., "under $30"), exclude products exceeding this threshold.
            - If no price limit is provided, ignore this rule.

            3. **Sort Order**:
            - First, sort by **intent relevance** (strongest keyword matches first).
            - Then, sort by **price** (low to high) within products of equal relevance.

            4. **Output Format**:
            - Return a JSON array of the relevant products, excluding remove unrelated items from the list; do not alter input data values, only filter and reorder.



        Examples:
        Example 1
        Intent: 'waterproof gloves under $20'
        Products:
        [
        {"id": 1, "title": "Waterproof Gloves", "price": "19.99", "inventory_quantity": 5, "description": "Waterproof"},
        {"id": 2, "title": "Leather Gloves", "price": "25.00", "inventory_quantity": 3, "description": "Durable"}
        ]
        Output:
        [
        {"id": 1, "title": "Waterproof Gloves", "price": "19.99", "inventory_quantity": 5, "description": "Waterproof"}
        ]

        Example 2
        Intent: 'touchscreen gloves'
        Products:
        [
        {"id": 4, "title": "Touchscreen Gloves", "price": "29.99", "inventory_quantity": 2, "description": "Touchscreen"},
        {"id": 5, "title": "Work Gloves", "price": "15.00", "inventory_quantity": 4, "description": "Rugged"}
        ]
        Output:
        [
        {"id": 4, "title": "Touchscreen Gloves", "price": "29.99", "inventory_quantity": 2, "description": "Touchscreen"},
        {"id": 5, "title": "Work Gloves", "price": "15.00", "inventory_quantity": 4, "description": "Rugged"}
        ]

        Task Execution:
        Now, apply these rules to the following product dataset and return the top 8 most relevant products in sorted JSON format:

        """ + input_str



        # Format the input string correctly and pass it as the 'input' variable
  
        response = llm.invoke(prompt)
        prompt = ""
  
        print("AI product result :",response.content.replace("\n", "").replace("```json", "").replace("```", "").strip())

        return json.loads(response.content.replace("\n", "").replace("```json", "").replace("```", "").strip())
    
    except Exception as e:
        print(f"Error in prioritize_products: {str(e)}")
        raise


def get_response(input_text,related_products,user_intent):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Jackson Hardware Store's AI assistant. Your role is to help customers find tools, hardware, or equipment,
          suggest relevant products based on their needs, and provide key details like brand, features, or availability. 
          Respond in 1-2 short, direct sentences (max 20 tokens) with no technical formatting, explanations, or symbols. 
          avoid preambles. and talk in a friendly manner.
          Actual user intention: {user_intent}
          Use related products from: {related_products}."""
    ),
    ("human", "{input}"),
])

        chain = prompt | llm

        response = chain.invoke({"input": input_text, "related_products":related_products,"user_intent":user_intent})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def General_QA(query):
    try:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are Jackson Hardware Store's friendly AI assistant. 
                Answer any user query—about products, the store, or general topics—in 1-2 short, warm sentences. 
                Avoid preambles, keep it relevant to the store when possible, 
                and redirect off-topic queries with a helpful nudge.

                don't, say "sure" first

                Examples:
                Query: "What time do you open?" → "We’re open at 8 AM—come by soon!"
                Query: "How’s the weather today?" → "Not sure about the weather, but we’ve got tarps if it rains!"
                Query: "Where are you located?" → "We’re in San Rafael—stop by and see us!"
                Query: "What’s a good gift idea?" → "A tool set from us makes a solid gift!"
                Query: "Do you have a showroom?" → "Yes, we do. You can visit us at 435 Du Bois St., San Rafael, CA 94901."
                Query: "How do I apply for a job at Jackson’s Hardware?" → "Please visit our About Us page to apply."
                Query: "Do you offer parts and repairs?" → "Yes, we do. Please check our Tool Repair page."
                Query: "Do you have a contact phone number?" → "Yes, we can be reached at 415.870.4083."
                Query: "Where can I find promotions going on?" → "Please visit our Sales & Events page to see promotions and events."
                Query: "What if I return an item due to it being defective?" → "Please return the item to the store and report to any of our sales associates."
                Query: "Do you deliver?" → "We certainly do. Please visit our Delivery page to get more information."
                Query: "Do you repair screens?" → "Yes, and our typical turnaround is 2 weeks."
                Query: "Can I rent a power washer from Jackson’s Hardware?" → "Customers can rent a power washer and many other items from Jackson’s Hardware. Please check our Tool Rental page."
                Query: "What credit cards do you accept?" → "Jackson’s Hardware accepts Mastercard, VISA, Discover, and American Express."
                Query: "Is it possible to special order an item?" → "Yes. If you do not find an item at Jackson’s Hardware, we can order that for you. We can also order items directly from the manufacturers featured in our stores. On average, special orders are available within 2 business weeks."
                Query: "Can we come and check the products?" → "Yes, you can."
                Query: "What are Jackson’s Hardware store hours?" → "Our hours are Mon-Fri: 6:00 AM to 6:00 PM; Saturday: 7:00 AM to 5:00 PM; and Sunday we are closed."
                """
            ),
            ("human", "{input}")
        ])

        chain = prompt | llm

        response = chain.invoke({"input": query})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def Store_QA(query):

    global vector_store

    retrieved_docs = vector_store.similarity_search(query)

    try:        

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly, expert agent at Jackson's Hardware, a 100% employee-owned store in San Rafael, CA,
          since 1964, offering 80,000+ top-tier hardware, tools, plumbing, paint, and equipment items for DIYers and pros. 
          Promote our industry-best products and service, praising customers’ great choice in shopping with us. 
          Answer in 1-2 short sentences (max 20 tokens), using jacksonshardware.com links if relevant. 
          Don’t mention competitors or unrelated resources;
         if unsure, direct to (415) 454-3740, office@jacksonshardware.com, or 62 Woodland Ave., San Rafael, CA 94901 (include for location or contact queries). 
         Avoid guesses, ask follow-ups if needed, and keep it accurate.
        Use the following pieces of context to answer the user's question:

        ----------------

        {context} 

    
        note : output should be a single line response.
        """
    ),
    ("human", "{input}")
])


        chain = prompt | llm

        response = chain.invoke({"input": query,"context":retrieved_docs})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 
   
@app.route('/chat', methods=['POST'])
def chat_product_search():
    try:
        message = request.json
        email = message.get('email')
        if email is None:
            return jsonify({'error': 'email is required'}), 400
        
        query = analyze_intent(message.get('content')).lower()
        prioritize_products_response = None

        if query == "general":
            ai_response = General_QA(message.get('content'))

        elif query == "website/company":
            ai_response = Store_QA(message.get('content'))

        else:

            query = {"Email": email} if email else {"Email": "guest_69dd2db7-11bf-49cc-934c-14fa2811bb4c"}
            chat_history = list(chat_history_collection.find(query))
            # Extract just sender and text from chat history
            chat_history = [{'sender': msg['sender'], 'text': msg['text']} 
                    for chat_doc in chat_history 
                    for msg in chat_doc.get('messages', [])]
            

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            chat_history.append({'sender': message.get('sender'), 'text': message.get('content')})

            message.update({
                'timestamp': datetime.now().isoformat()
            })

            print(chat_history)

            research_intent_response = research_intent(chat_history)
            chat_history = []

            # print("\n\nchat_history : ", chat_history)
            print("\n\nresearch_intent_response : ", research_intent_response)

            
            related_product = get_product_search(research_intent_response)

            prioritize_products_response = prioritize_products(research_intent_response,related_product)
            related_product = ""

            # print("\n\nprioritize_products_response : ", prioritize_products_response)

            ai_response = get_response(input_text = message['content'], user_intent = research_intent_response,related_products=prioritize_products_response)
    
        
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'related_products_for_query':prioritize_products_response
        }        
        ai_response = ""
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            "error_response" : str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "working"})

if __name__ == "__main__":
    app.run(debug=False)
