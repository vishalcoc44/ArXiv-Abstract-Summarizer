from rouge_score import rouge_scorer
import pandas as pd # Assuming you stored outputs in a CSV

# Load your summarization results
df_summaries = pd.read_csv("summary_outputs.csv") # Or your file
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Focus on ROUGE-L

results_rouge = []

for index, row in df_summaries.iterrows():
    doc_id = row['doc_id']
    ref_summary = str(row['reference_summary_1']) # Ensure it's a string

    # Fine-tuned Gemma
    ft_summary = str(row['finetuned_gemma_summary'])
    if pd.notna(ft_summary) and ft_summary.strip(): # Check for non-empty summary
        scores_ft = scorer.score(ref_summary, ft_summary)
        results_rouge.append({'id': doc_id, 'model': 'finetuned_gemma', 'rougeL_f1': scores_ft['rougeL'].fmeasure})
    else:
        results_rouge.append({'id': doc_id, 'model': 'finetuned_gemma', 'rougeL_f1': 0.0}) # Or None

    # Base Gemma
    base_summary = str(row['base_gemma_summary'])
    if pd.notna(base_summary) and base_summary.strip():
        scores_base = scorer.score(ref_summary, base_summary)
        results_rouge.append({'id': doc_id, 'model': 'base_gemma', 'rougeL_f1': scores_base['rougeL'].fmeasure})
    else:
        results_rouge.append({'id': doc_id, 'model': 'base_gemma', 'rougeL_f1': 0.0})


    # Gemini API
    gemini_summary = str(row['gemini_api_summary'])
    if pd.notna(gemini_summary) and gemini_summary.strip():
        scores_gemini = scorer.score(ref_summary, gemini_summary)
        results_rouge.append({'id': doc_id, 'model': 'gemini_api', 'rougeL_f1': scores_gemini['rougeL'].fmeasure})
    else:
        results_rouge.append({'id': doc_id, 'model': 'gemini_api', 'rougeL_f1': 0.0})


df_rouge_results = pd.DataFrame(results_rouge)
print("ROUGE-L Results:")
print(df_rouge_results.groupby('model')['rougeL_f1'].mean())
# df_rouge_results.to_csv("rouge_scores.csv", index=False)