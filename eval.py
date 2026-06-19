import os
import argparse
from typing import List, Dict, Any
from retrieve import retrieve_context
from query import ask_question

# Define our validation dataset with ground truth document and section
EVAL_DATASET = [
    {
        "id": 1,
        "question": "When did the Delegation of Financial Powers Rules, 2024 come into force?",
        "gt_document": "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf",
        "gt_section": "Rule 1 Short title and commencement-",
        "is_answerable": True
    },
    {
        "id": 2,
        "question": "In which authority do the residuary financial powers vest under DFPR 2024?",
        "gt_document": "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf",
        "gt_section": "Rule 6 Residuary financial powers-",
        "is_answerable": True
    },
    {
        "id": 3,
        "question": "What is the financial limit for open or limited tender contracts that can be sanctioned by the Secretary of a department?",
        "gt_document": "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf",
        "gt_section": "Rule 11 Indents, contracts and purchases-",
        "is_answerable": True
    },
    {
        "id": 4,
        "question": "Up to what amount can a Department of Government of India waive recovery of overpayment made to a Government servant with the concurrence of Financial Advisers?",
        "gt_document": "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf",
        "gt_section": "Rule 15 Waiver of recovery of overpayment made to Government servants–",
        "is_answerable": True
    },
    {
        "id": 5,
        "question": "What is the exact reimbursement rate for private vehicle travel in Paris under the 2026 guidelines?",
        "gt_document": "",
        "gt_section": "",
        "is_answerable": False
    }
]

def evaluate_retrieval(index_path: str, top_k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    Evaluates retrieval Recall@K against the ground truth evaluation dataset.
    """
    print(f"\n--- Evaluating Retrieval Quality (Index: {index_path}) ---")
    
    hits = {k: 0 for k in top_k_values}
    total_evaluable = 0
    
    for item in EVAL_DATASET:
        if not item["is_answerable"]:
            continue
        total_evaluable += 1
        
        question = item["question"]
        gt_doc = item["gt_document"]
        gt_sect = item["gt_section"]
        
        # Retrieve chunks
        try:
            retrieved = retrieve_context(question, index_path=index_path, top_k=max(top_k_values))
        except Exception as e:
            print(f"Error during retrieval for '{question}': {e}")
            continue
            
        # Check recall
        for k in top_k_values:
            retrieved_k = retrieved[:k]
            # A hit is defined as retrieving the correct document and section
            is_hit = any(
                c["document"] == gt_doc and c["section"].strip() == gt_sect.strip()
                for c in retrieved_k
            )
            if is_hit:
                hits[k] += 1
                
    # Calculate Recall@K
    recall_results = {}
    print("\nRetrieval Recall Results:")
    for k in top_k_values:
        recall = (hits[k] / total_evaluable) * 100 if total_evaluable > 0 else 0
        recall_results[f"Recall@{k}"] = recall
        print(f"  Recall@{k}: {recall:.2f}% ({hits[k]}/{total_evaluable} correct)")
        
    return recall_results

def evaluate_generation(index_path: str, model_name: str) -> Dict[str, float]:
    """
    Evaluates answer quality, citation correctness, and refusal rate.
    """
    print(f"\n--- Evaluating Generation Quality (Model: {model_name}) ---")
    
    citation_correct = 0
    refusal_correct = 0
    total_answerable = 0
    total_unanswerable = 0
    
    for item in EVAL_DATASET:
        question = item["question"]
        gt_doc = item["gt_document"]
        gt_sect = item["gt_section"]
        is_ans = item["is_answerable"]
        
        if is_ans:
            total_answerable += 1
        else:
            total_unanswerable += 1
            
        try:
            res = ask_question(question, index_path=index_path, top_k=5, model_name=model_name)
            pred_ans = res["answer"]
            pred_doc = res["source"]
            pred_sect = res["section"]
            
            if is_ans:
                # Check citation accuracy
                doc_match = pred_doc == gt_doc
                sect_match = pred_sect.strip() == gt_sect.strip()
                
                print(f"\nQ: {question}")
                print(f"  GT Cit:  {gt_doc} | {gt_sect}")
                print(f"  Pred Cit: {pred_doc} | {pred_sect}")
                print(f"  Answer:   {pred_ans}")
                
                if doc_match and sect_match:
                    citation_correct += 1
            else:
                # Check unanswerable refusal handling
                refusal_match = "cannot answer" in pred_ans.lower()
                print(f"\nUnanswerable Q: {question}")
                print(f"  Pred refusal check: {'PASSED' if refusal_match else 'FAILED'}")
                print(f"  Answer: {pred_ans}")
                
                if refusal_match:
                    refusal_correct += 1
                    
        except Exception as e:
            print(f"Error during generation for '{question}': {e}")
            
    citation_acc = (citation_correct / total_answerable) * 100 if total_answerable > 0 else 0
    refusal_acc = (refusal_correct / total_unanswerable) * 100 if total_unanswerable > 0 else 0
    
    print("\nGeneration Metrics:")
    print(f"  Citation Accuracy: {citation_acc:.2f}% ({citation_correct}/{total_answerable} correct)")
    print(f"  Refusal Accuracy:  {refusal_acc:.2f}% ({refusal_correct}/{total_unanswerable} correct)")
    
    return {
        "citation_accuracy": citation_acc,
        "refusal_accuracy": refusal_acc
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument("--index", type=str, default="index.pkl", help="Path to index file")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Gemini model name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.index):
        print(f"Error: Index file not found at {args.index}. Please run ingest.py first.")
        return
        
    evaluate_retrieval(args.index)
    evaluate_generation(args.index, args.model)

if __name__ == "__main__":
    main()
