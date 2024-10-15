
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Replace book.pdf with any pdf of your choice
loader = UnstructuredPDFLoader("Yohannanof_report.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

# Choose any query of your choice
query = "Who is Rich Dad?"
docs = docsearch.get_relevant_documents(query)
# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
output = chain.run(input_documents=docs, question=query)
print(output)