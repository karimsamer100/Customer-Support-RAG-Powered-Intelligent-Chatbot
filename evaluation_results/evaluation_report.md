# RAG Evaluation Report

## Overview
This evaluation measures retrieval and generation quality on a sampled set of support questions.

## Evaluation Setup
- sample size: 5
- top_k: 5
- vector store: FAISS
- embedding model: sentence-transformers/all-MiniLM-L6-v2
- generation model (LLM_MODEL env): deepseek-v3.2

## Metrics
- Retrieval success rate
- Average top similarity score
- Average latency
- BLEU-like unigram overlap
- ROUGE-L (LCS / reference length)

## Results Summary
| Metric | Value |
|---|---:|
| sample_size | 5 |
| successful_evaluations | 5 |
| avg_latency_seconds | 0.6525 |
| avg_top_score | 1.0000 |
| avg_retrieved_count | 2.20 |
| avg_simple_bleu_score | 0.5121 |
| avg_rouge_l_score | 0.8211 |
| retrieval_success_rate | 1.0000 |

## Sample Results
(Up to 5 samples)

### Query 1
- Query: U guys are not telling me anything different but same thing which has been repeated n times. Useless
- Generated answer: Based on similar support cases, here is the most relevant available guidance:

We can't access your account over Twitter. You can choose to write back to our e-mail. We will revert at the earliest.
- Top score: 1.0000
- Latency: 0.83s

### Query 2
- Query: For my return amazon has issued refund amount on 29th oct 2017 but this is still not credited into my bank account.
- Generated answer: Based on similar support cases, here is the most relevant available guidance:

Usually it will take 3-5 business days for the refund to be reflected in your bank account as per the bank's regulations.
- Top score: 1.0000
- Latency: 0.60s

### Query 3
- Query: I ended the chat for my sanity....any help would be appreciated
- Generated answer: Based on similar support cases, here is the most relevant available guidance:

I'm sorry the chat wasn't helpful! Usually after the chat, there's an option to leave feedback. When these are completed (1/2)
- Top score: 1.0000
- Latency: 0.62s

### Query 4
- Query: How about a 'where IS my package?' button ? £40 lost. 😡 [SUPPORT_LINK]
- Generated answer: Based on similar support cases, here is the most relevant available guidance:

Hi, sorry to hear that, contact us here: [SUPPORT_LINK] we'll help
- Top score: 1.0000
- Latency: 0.63s

### Query 5
- Query: order returned on 3rd Oct, no update till date such a pathetic service. Everytime need 2 call help line no [SUPPORT_LINK]
- Generated answer: Based on similar support cases, here is the most relevant available guidance:

Sorry for the miss. The refunds are usually issued once they reach the seller/fulfillment center which might not have (1/2)
- Top score: 1.0000
- Latency: 0.59s

## Notes and Limitations
- This evaluation is simplified for academic/demo purposes.
- BLEU/ROUGE are approximate and do not fully measure helpfulness.
- Human review is recommended for production-quality answers.
- Evaluation uses historical support questions from the vector store; treat this as a functionality and quality sanity check.
