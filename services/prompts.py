from langchain_core.prompts import PromptTemplate
from services.models import GreetingClassifier
from langchain_core.output_parsers import PydanticOutputParser

def detectGreeting():
    parser = PydanticOutputParser(pydantic_object=GreetingClassifier)
    
    detectPrompt = PromptTemplate(
        template="""
        You are a greeting classifier. Your task is to determine if the input is a greeting or not.
        Common greetings include: hello, hi, hey, good morning, good afternoon, good evening, etc.
        
        Input: {prompt}
        
        {format_instructions}
        
        Respond with ONLY the JSON object, no other text or explanation.
        Example: {{"result": "yes"}} or {{"result": "no"}}
        """,
        input_variables=["prompt"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return detectPrompt, parser

def systemPrompt():
    systemPrompt = PromptTemplate(
        template="""
        You are an Q/A assistant. use tools when needed to answer the user's question accurately
        If a tool is required call it. If not answer directly
        Keep answer concise and to the point """,
        input_variables=[]
    )
    return systemPrompt

def userPrompt(question, context, conversation_history):
    userPrompt = f"""
    Question: {question}
        
    Context: {context}

    conversation history: {conversation_history}
    Use the context and conversation history to answer the question, if the context is not relevant to the question, provide the answer to the question  and inform the user that answer is not based on the context or knowledge base
    make sure the answer is concise and to the point no additional mentioning of context or knowledge base
    
    """
    return userPrompt
