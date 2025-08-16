import os
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


DEFAULT_INDEX_DIR = os.getenv("INDEX_DIR", os.path.join("data", "index"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class RAGConfig:
    dataset_path: str
    index_dir: str = DEFAULT_INDEX_DIR
    embed_model: str = EMBED_MODEL
    chat_model: str = CHAT_MODEL


class FitnessRAG:
    def __init__(self, cfg: Optional[RAGConfig] = None) -> None:
        self.cfg = cfg or RAGConfig(dataset_path=self._auto_dataset_path())
        os.makedirs(self.cfg.index_dir, exist_ok=True)

    def _auto_dataset_path(self) -> str:
        candidates = [
            "megaGymDataset.csv",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "megaGymDataset.csv"),
            os.path.join(os.path.dirname(__file__), "megaGymDataset.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        # fall back; will error later during build
        return candidates[0]

    def _make_docs(self, df: pd.DataFrame) -> List[Document]:
        docs: List[Document] = []
        for _, row in df.iterrows():
            title = str(row.get("Title") or row.get("title") or row.get("Exercise Name") or row.get("name") or "Unknown Exercise").strip()
            body_part = str(row.get("BodyPart") or row.get("body_part") or row.get("Body Part") or row.get("target") or "").strip()
            equipment = str(row.get("Equipment") or row.get("equipment") or "").strip()
            level = str(row.get("Level") or row.get("level") or row.get("Difficulty") or "").strip()
            type_ = str(row.get("Type") or row.get("type") or "").strip()
            mechanics = str(row.get("Mechanics") or row.get("mechanics") or "").strip()
            primary = str(row.get("PrimaryMuscles") or row.get("primary_muscles") or row.get("Primary Muscles") or "").strip()
            secondary = str(row.get("SecondaryMuscles") or row.get("secondary_muscles") or row.get("Secondary Muscles") or "").strip()
            notes = str(row.get("Instructions") or row.get("Description") or row.get("notes") or row.get("Instructions (short)") or "").strip()

            text = f"Title: {title}\nBody Part: {body_part}\nEquipment: {equipment}\nLevel: {level}\nType: {type_}\nMechanics: {mechanics}\nPrimary Muscles: {primary}\nSecondary Muscles: {secondary}\nNotes: {notes}"
            docs.append(Document(page_content=text, metadata={"title": title, "body_part": body_part}))
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(docs)

    def build_or_refresh_index(self) -> str:
        df = pd.read_csv(self.cfg.dataset_path)
        df.columns = [str(c).strip() for c in df.columns]
        docs = self._make_docs(df)
        embeddings = OpenAIEmbeddings(model=self.cfg.embed_model)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(self.cfg.index_dir)
        return self.cfg.index_dir

    def _load_vectorstore(self) -> FAISS:
        embeddings = OpenAIEmbeddings(model=self.cfg.embed_model)
        return FAISS.load_local(self.cfg.index_dir, embeddings, allow_dangerous_deserialization=True)

    def qa(self, question: str, k: int = 6) -> str:
        vs = self._load_vectorstore()
        # Use MMR to diversify retrieved docs and fetch more candidates
        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(20, k * 3)})

        llm = ChatOpenAI(model=self.cfg.chat_model, temperature=0)

        prompt = PromptTemplate.from_template(
            """
            You are a helpful, safety-first fitness assistant.
            Use the provided context to answer the user's question about exercises, equipment alternatives, form and safety tips, or suitable substitutions. If the question references days like Monday or a user's plan, infer that it relates to general workout structure and equipment categories rather than calendar specifics.

            If the exact answer is not directly in the context, provide concise, reasonable, and safe general guidance grounded in common fitness best practices, and suggest related exercises or alternatives that fit the user's constraints (e.g., bodyweight vs. dumbbell). Do not simply say you don't know; offer helpful options while being clear if the context didn't have an exact match.

            Keep answers concise (3-6 bullet points when giving tips).

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

        answer = qa_chain.invoke({"query": question})
        if isinstance(answer, dict) and "result" in answer:
            return answer["result"]
        return str(answer)
