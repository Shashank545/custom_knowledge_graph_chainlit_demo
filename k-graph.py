import logging
import sys
#
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (SimpleDirectoryReader,
                         ServiceContext,
                         KnowledgeGraphIndex)


from llama_index.settings import Settings

# from llama_index.core import LLMPredictor
#
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pyvis.network import Network


HF_TOKEN = "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"
llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN
)

embed_model = LangchainEmbedding(
  HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN,model_name="thenlper/gte-large")
)

documents = SimpleDirectoryReader("../Project_Vertex_AI/").load_data()
print(len(documents))


#setup the service context

service_context = Settings.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

#setup the storage context

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

#Construct the Knowlege Graph Undex
index = KnowledgeGraphIndex.from_documents( documents=documents,
                                           max_triplets_per_chunk=3,
                                           service_context=service_context,
                                           storage_context=storage_context,
                                          include_embeddings=True)


query = "What is the net value potential in Retail & CPG?"
query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=5,)
#
message_template =f"""<|system|>Please check if the following pieces of context has any mention of the  keywords provided in the Question.If not then don't know the answer, just say that you don't know.Stop there.Please donot try to make up an answer.</s>
<|user|>
Question: {query}
Helpful Answer:
</s>"""
#
response = query_engine.query(message_template)
#
print(response.response.split("<|assistant|>")[-1].strip())




