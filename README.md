# 🧠 Chat RAG con LangChain, LangGraph y Streamlit

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) que permite al usuario subir documentos PDF y hacer preguntas sobre su contenido en lenguaje natural. Utiliza modelos LLM locales vía [Ollama](https://ollama.com), embeddings personalizados, y una arquitectura basada en `LangGraph`.

---

## 🚀 ¿Qué hace este proyecto?

- Permite subir un PDF y transformarlo en fragmentos indexados.
- Genera embeddings con `nomic-embed-text`.
- Responde preguntas usando un modelo local (como `llama3.2:3b`).
- Usa recuperación de contexto relevante del PDF.
- Usa Streamlit como interfaz conversacional.

---

## 🛠️ Requisitos

- Python 3.10+
- Ollama instalado y corriendo 
- Modelos descargados:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text


  USO:
  instala dependencias
  uv add requirements.txt

  ejecuta la app con
  uv run uv streamlit rag7.py





Ejemplos de consultas y respuestas esperadas

 `¿Qué es SellMoreTrips.IA?`                
 SellMoreTrips.IA es una plataforma impulsada por IA que ayuda a agentes de viaje a automatizar sus ventas. 
| `¿SellMoreTrips reemplaza a los agentes?` |
 No, es una herramienta de apoyo que los potencia, no los sustituye.    
`¿Qué beneficios ofrece SellMoreTrips?`   
Automatización, cotizaciones instantáneas, atención 24/7 y personalización de recomendaciones.             
 `¿Cuándo murió Albert Einstein?`
 No tengo información sobre eso en este documento.                                                          


