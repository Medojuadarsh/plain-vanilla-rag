import os
import pandas as pd
import argparse
from query import ask_question

def main():
    parser = argparse.ArgumentParser(description="Run the RAG system over the test dataset")
    parser.add_argument("--test_csv", type=str, default=os.path.join("Data", "test.csv"), help="Path to test.csv")
    parser.add_argument("--index", type=str, default="index.pkl", help="Path to index file")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to write predictions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process for quick testing")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Gemini model name")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve per question")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_csv):
        print(f"Error: Test CSV file not found at {args.test_csv}")
        return
        
    if not os.path.exists(args.index):
        print(f"Error: Index file not found at {args.index}. Please run ingest.py first.")
        return
        
    print(f"Loading questions from {args.test_csv}...")
    df = pd.read_csv(args.test_csv)
    total_q = len(df)
    print(f"Total questions loaded: {total_q}")
    
    if args.limit:
        print(f"Limiting execution to first {args.limit} questions...")
        df = df.head(args.limit)
        
    results = []
    
    for idx, row in df.iterrows():
        qid = row['id']
        question = row['question']
        print(f"[{idx+1}/{len(df)}] Processing ID: {qid}...")
        
        try:
            res = ask_question(question, index_path=args.index, top_k=args.top_k, model_name=args.model)
            results.append({
                'id': qid,
                'prediction': res['answer'],
                'pred_source': res['source'],
                'pred_section': res['section']
            })
        except Exception as e:
            print(f"[Error] Failed to process ID {qid}: {e}")
            results.append({
                'id': qid,
                'prediction': "Error processing question.",
                'pred_source': "",
                'pred_section': ""
            })
            
    # Write to output csv
    out_df = pd.DataFrame(results)
    
    # Ensure correct column ordering
    out_df = out_df[['id', 'prediction', 'pred_source', 'pred_section']]
    
    out_df.to_csv(args.output_csv, index=False)
    print(f"Pipeline run completed! Predictions saved to: {args.output_csv}")
    
    # Print some statistics
    unanswerable_cnt = sum(out_df['prediction'].str.contains("cannot answer", case=False, na=False))
    print(f"Summary Statistics:")
    print(f"  - Total processed: {len(out_df)}")
    print(f"  - Unanswerable flags raised: {unanswerable_cnt}")
    print(f"  - Grounded answers generated: {len(out_df) - unanswerable_cnt}")

if __name__ == "__main__":
    main()
