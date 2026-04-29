1. preprocessing.py
   - load amazon_final_data.csv
   - clean question/answer
   - keep original + cleaned text
   - remove links, mentions, agent signatures, weird encoding
   - save processed_amazon_support.csv

2. eda.py أو جزء EDA في نفس الملف مؤقتًا
   - dataset size
   - nulls / duplicates
   - length stats
   - common issue keywords

3. embedder.py
   - بعدين نستخدم embedding model جاهز
   - ينفع API عادي، مش لازم تنزله local

4. build_index.py
   - build vector DB

5. rag_pipeline.py
   - retrieve top-k
   - send context to LLM
   - get final answer