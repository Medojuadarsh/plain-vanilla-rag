import os
import json
import re
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv
from retrieve import retrieve_context

# Load environment variables from .env file
load_dotenv()

# LLM Client wrapper that detects available keys/libraries and supports fallback
class LLMClient:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.provider = None
        
        # Check Gemini API Key (support both GEMINI_API_KEY and GOOGLE_API_KEY)
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        # Determine if the key is a valid key and not a placeholder or empty
        is_placeholder = False
        if self.gemini_key:
            k_strip = self.gemini_key.strip()
            if k_strip == "" or k_strip.startswith("your_") or "placeholder" in k_strip.lower() or k_strip == "dummy":
                is_placeholder = True
                
        if self.gemini_key and not is_placeholder:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.provider = "gemini"
                self.model = genai.GenerativeModel(self.model_name)
                print(f"[LLM] Initialized Gemini model: {self.model_name}")
            except ImportError:
                print("[LLM Warning] API key found but 'google-generativeai' package not installed.")
                
        if not self.provider:
            print("[LLM Info] No valid API keys detected or running offline. Running in SIMULATION/MOCK mode.")
            self.provider = "mock"

    def generate(self, prompt: str) -> str:
        if self.provider == "gemini":
            try:
                # Set response_mime_type to application/json for structured output
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return response.text
            except Exception as e:
                print(f"[LLM Error] Gemini call failed: {e}. Trying fallback without JSON config...")
                try:
                    response = self.model.generate_content(prompt)
                    return response.text
                except Exception as ex:
                    print(f"[LLM Error] Gemini fallback call failed: {ex}")
                    return self._mock_respond(prompt)
                    
        else:
            return self._mock_respond(prompt)

    def _mock_respond(self, prompt: str) -> str:
        """
        Simulated answering logic when offline or API key is not present.
        Extracts key sentences from the prompt context.
        """
        # Find context block in prompt
        context_match = re.search(r"Context:\s*---\s*(.*?)\s*---\s*Question:", prompt, re.DOTALL)
        question_match = re.search(r"Question:\s*(.*?)\s*\n", prompt)
        
        context_text = context_match.group(1) if context_match else ""
        question_text = question_match.group(1) if question_match else ""
        
        # Simple rule-based answer generation from context
        # In a real environment, this is just a fallback.
        if not context_text or "DPM 2025" not in context_text and "Navy" not in context_text and "DFPDS" not in context_text:
            return json.dumps({
                "answer": "I cannot answer this question based on the provided corpus.",
                "source": "",
                "section": ""
            })
            
        # Parse first chunk to simulate a valid grounded answer
        # Find lines of chunk info
        chunks = context_text.split("Chunk ID:")
        if len(chunks) > 1:
            first_chunk = chunks[1]
            doc_match = re.search(r"Document:\s*([^\n]+)", first_chunk)
            sect_match = re.search(r"Section:\s*([^\n]+)", first_chunk)
            text_match = re.search(r"Text:\s*(.*?)(?=\n\n|\Z)", first_chunk, re.DOTALL)
            
            doc = doc_match.group(1).strip() if doc_match else ""
            sect = sect_match.group(1).strip() if sect_match else ""
            text = text_match.group(1).strip() if text_match else ""
            
            # Formulate simulated answer from first 2 sentences of chunk text
            sentences = [s.strip() for s in re.split(r'\.|\n', text) if len(s.strip()) > 15]
            ans_body = sentences[0] + ". " + (sentences[1] + "." if len(sentences) > 1 else "")
            
            return json.dumps({
                "answer": f"[Simulated Response] {ans_body}",
                "source": doc,
                "section": sect
            })
            
        return json.dumps({
            "answer": "I cannot answer this question based on the provided corpus.",
            "source": "",
            "section": ""
        })

def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    # Format retrieved chunks
    context_str = ""
    for idx, c in enumerate(chunks):
        context_str += f"Chunk ID: {c['chunk_id']}\n"
        context_str += f"Document: {c['document']}\n"
        context_str += f"Section: {c['section']}\n"
        context_str += f"Text: {c['text']}\n"
        context_str += "\n" + "="*40 + "\n\n"
        
    prompt = f"""You are a highly precise procurement and policy compliance assistant.
Your task is to answer the user's question using ONLY the provided document chunks (context).

Guidelines:
1. Grounding: Rely ONLY on the clear facts mentioned in the context. Do not assume or extrapolate.
2. Unanswerable: If the context does not contain the information needed to answer the question, or if there is not enough detail, return the answer as: "I cannot answer this question based on the provided corpus."
3. Formatting: You MUST return your output in the following JSON format ONLY:
{{
  "answer": "The detailed answer grounded in the context, including citations to sections.",
  "source": "The exact document filename (e.g., RegsNavyI.pdf) used to answer the question. If the question is unanswerable, return an empty string.",
  "section": "The exact section name (e.g., Rule 10 or Preface) from the source document used to answer the question. If the question is unanswerable, return an empty string."
}}

Context:
---
{context_str}
---

Question: {question}

JSON Answer:"""
    return prompt

def ask_question(question: str, index_path: str = "index.pkl", top_k: int = 5, model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
    """
    RAG Pipeline query step: Retrieves context and queries LLM.
    """
    # 1. Retrieve relevant chunks
    chunks = retrieve_context(question, index_path=index_path, top_k=top_k)
    
    if not chunks:
        return {
            "answer": "I cannot answer this question based on the provided corpus.",
            "source": "",
            "section": ""
        }
        
    # 2. Build prompt
    prompt = build_rag_prompt(question, chunks)
    
    # 3. Call LLM
    llm = LLMClient(model_name=model_name)
    response_raw = llm.generate(prompt)
    
    # 4. Parse JSON response
    try:
        # Strip potential markdown formatting (```json ... ```)
        cleaned_json = response_raw.strip()
        if cleaned_json.startswith("```"):
            cleaned_json = re.sub(r"^```(?:json)?\n", "", cleaned_json)
            cleaned_json = re.sub(r"\n```$", "", cleaned_json)
        cleaned_json = cleaned_json.strip()
        
        result = json.loads(cleaned_json)
        
        # Basic validation of keys
        if "answer" not in result or "source" not in result or "section" not in result:
            raise KeyError("Missing required keys in LLM JSON output")
            
        return result
    except Exception as e:
        print(f"[Warning] Failed to parse LLM response as JSON: {e}. Raw response:\n{response_raw}")
        
        # Fallback parsing
        # If the LLM returned raw text, we try to extract answer and cite the top chunk
        ans = response_raw.strip()
        top_chunk = chunks[0]
        
        # Check if the answer looks like a refusal
        if "cannot answer" in ans.lower() or "not mentioned" in ans.lower() or "no information" in ans.lower():
            return {
                "answer": "I cannot answer this question based on the provided corpus.",
                "source": "",
                "section": ""
            }
            
        return {
            "answer": ans,
            "source": top_chunk["document"],
            "section": top_chunk["section"]
        }

def main():
    parser = argparse.ArgumentParser(description="Query the full RAG system")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--index", type=str, default="index.pkl", help="Path to index file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Gemini model name")
    
    args = parser.parse_args()
    
    result = ask_question(args.question, index_path=args.index, top_k=args.top_k, model_name=args.model)
    
    print("\n" + "="*20 + " RAG SYSTEM ANSWER " + "="*20)
    print(f"Question: {args.question}")
    print(f"Answer:   {result['answer']}")
    print(f"Source:   {result['source'] or 'N/A'}")
    print(f"Section:  {result['section'] or 'N/A'}")
    print("="*59 + "\n")

if __name__ == "__main__":
    main()
