import os

# Google AI Studioなどで取得したAPIキーを設定
os.environ["GOOGLE_API_KEY"] = "AIzaSyDwBTWkfOKiiRnj4svbtn9FeATKrGaEiOA"

# 1. ドキュメント読み込み用
from langchain_community.document_loaders import PyMuPDFLoader

# 2. テキスト分割用
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 3. Embeddingとベクトルストア用
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 4. LLMとチェーン用
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

############################################

def load_document(directory_path: str):
    """指定されたディレクトリ内のすべてのPDFファイルを読み込む"""
    all_docs = []
    pdf_files = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)

    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"{pdf_path} の読み込みに失敗しました - {str(e)}")

    return all_docs

docs = load_document("./data")

############################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n\n\n", "\n\n", "\n", "。", "、"],
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(docs)

############################################

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

############################################

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

prompt_template = """あなたは内閣府の「人々のつながりに関する基礎調査」の専門アナリストです。
この調査は、日本の孤独・孤立の実態を把握するために実施された全国調査です。

以下の文脈を使用して、質問に対して正確で分かりやすい回答を提供してください。
調査の内容、質問項目、統計データなどについて、具体的かつ専門的に回答してください。

回答する際の注意点：
- 調査の目的や方法論について言及する場合は正確に
- 統計データや数値がある場合は明確に提示
- 質問番号や選択肢がある場合は具体的に参照
- 専門用語は分かりやすく説明

文脈に答えが見つからない場合は、「提供された調査資料には該当する情報が見つかりません」と回答してください。

文脈:
{context}

質問: {question}

回答:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

############################################

qa_chain.invoke({"query": "調査の回答期限はいつまでですか？"})