import os
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import FakeEmbeddings

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

embedding = FakeEmbeddings(size=384)
db = Chroma(persist_directory="vectordb", embedding_function=embedding)
retriever = db.as_retriever()

def ask(question):
    prompt = f"""You are a helpful assistant for TTU (Tatung University of Technology) in Taiwan.
TTU is located at No.40, Sec. 3, Zhongshan N. Rd., Taipei City 104, Taiwan.
TTU offers programs in Engineering, Management, and Design.
TTU phone: +886-2-2182-2928
TTU is a private teaching and research university.
Always answer in English.
If you don't know, say "I don't have that information."

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    return response.content
