# Support Portal Integration

The chatbot is exposed through REST API endpoints suitable for integration into a support portal.

Key endpoints:

- GET /health
- POST /ask
- POST /feedback
- GET /metrics
- GET /dashboard

The repository includes a small demo portal in `support_portal_demo/index.html` which shows how a web page can call `/ask` and `/feedback`.

Example curl for `/ask`:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: demo-key" \
  -d '{"question":"Where is my order?","top_k":5}'
```

Example curl for `/feedback`:

```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -H "x-api-key: demo-key" \
  -d '{"question":"Where is my order?","answer":"Demo","rating":5,"comment":"Helpful"}'
```
