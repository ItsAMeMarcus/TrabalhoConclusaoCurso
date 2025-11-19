from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector # Importa o tipo Vetor
from app.data.database import Base

class Tcc(Base):
    __tablename__ = "tccs"
    
    id = Column(Integer, primary_key=True, index=True)
    titulo = Column(String(255), nullable=False)
    autor = Column(String(100))
    ano = Column(Integer)
    url_pdf = Column(String(255))
    orientador = Column(String(255))
    
    # Relação: Um TCC tem muitos chunks
    chunks = relationship("Chunk", back_populates="tcc", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    tcc_id = Column(Integer, ForeignKey("tccs.id"), nullable=False)
    texto_chunk = Column(Text, nullable=False)
    numero_chunk = Column(Integer, nullable=False)
    
    # Define a coluna de embedding com a dimensão correta
    embedding = Column(Vector(768)) 
    
    # Relação: Um Chunk pertence a um TCC
    tcc = relationship("Tcc", back_populates="chunks")

