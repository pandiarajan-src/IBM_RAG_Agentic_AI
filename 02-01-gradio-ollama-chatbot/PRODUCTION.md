# Taking the Support Bot to Production

**Analogy:** the lab is a food truck - one cook, one window, cash box under the counter. Production is a restaurant chain: more cooks, a host at the door, health inspections, and real bookkeeping. Same recipe, different operation.

## 1. What changes

```
Lab:         Browser -> Gradio -> Ollama (laptop) -> SQLite file

Production:  Browser -> CDN/WAF -> Load balancer (TLS, SSO)
                          -> React or Gradio UI
                          -> FastAPI service (auth, rate limit, guardrails, logging)
                          -> Model server pool (GPU) -> Postgres + object store
```

## 2. Deployment guide (small internal rollout)

1. **Build and push:** `docker build -t registry.example.com/supportbot:1.0.0 .` then push. Tag by version, never `latest`.
2. **Configure by environment:** `OLLAMA_HOST`, `OLLAMA_MODEL`, `SUPPORTBOT_DB`. No secrets in the image.
3. **Run Ollama on a GPU host** with `OLLAMA_HOST=0.0.0.0`, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=-1` (keep model warm). Keep it on a private network - Ollama has no authentication.
4. **Put a reverse proxy in front** (nginx/Caddy/Traefik) for TLS. For streaming, disable buffering (`proxy_buffering off;`) and raise read timeouts. If served under a sub-path, set `GRADIO_ROOT_PATH`.
5. **Add login:** `demo.launch(auth=...)` is a stopgap. For SSO, mount Gradio inside FastAPI (`gr.mount_gradio_app(app, demo, path="/", auth_dependency=...)`) behind your OIDC provider.
6. **Lock down the UI before real customers see it:** remove the Tuning controls and the Conversation log tab (see section 3, rows 6-7).
7. **Health checks and restart policy:** probe `/` on the app and `/api/tags` on Ollama.
8. **Roll out** to a pilot group, watch the metrics in section 4, then widen.

## 3. Challenges when you scale up

| # | Challenge | Why it bites | What to do |
|---|---|---|---|
| 1 | **Concurrency** | One model answers one request at a time per slot; user 20 waits behind 19 others | Tune `default_concurrency_limit` and `OLLAMA_NUM_PARALLEL`; show queue position; move to vLLM/TGI for batching |
| 2 | **GPU cost and cold starts** | Loading a model takes 10-60 s; idle GPUs burn money | Keep models warm; autoscale on queue depth, not CPU; pick the smallest model that passes your evals |
| 3 | **SQLite under load** | One writer at a time; file lives on one container's disk | Move to Postgres; until then enable WAL mode and one shared volume |
| 4 | **Multiple app replicas** | Gradio keeps session state in memory, so a user bounced to another replica loses their stream | Sticky sessions at the load balancer, or keep the UI stateless and store history server-side |
| 5 | **Auth and tenancy** | The lab has no login; anyone with the URL can chat | SSO at the proxy or FastAPI layer; tie `session_id` to a real user id |
| 6 | **Exposed controls** | Customers can edit the system prompt and pick any model | Remove `additional_inputs` for end users; keep them in an internal admin build |
| 7 | **Exposed data** | The log tab and its auto-generated API (`/recent_turns`) show every customer's conversation | Remove from the public app, or set `api_visibility="private"` and put it behind an admin role |
| 8 | **PII in logs** | Customers paste account numbers and addresses | Redact before writing; set retention; encrypt at rest; document it for compliance |
| 9 | **Prompt injection and abuse** | "Ignore your instructions and..." | Input/output guardrails, rate limiting per user, max message length, never give the bot tools it does not need |
| 10 | **Hallucination** | The bot invents refund policies | Ground answers in your docs with RAG (ChromaDB lab), cite sources, add a human-handoff path |
| 11 | **Unbounded history** | Long chats overflow the context window and slow every turn | Trim or summarise history; cap turns per session |
| 12 | **Streaming through proxies** | Buffers and 60 s idle timeouts break token streams | Disable proxy buffering, raise timeouts, test through the real ingress |
| 13 | **Observability** | "It felt slow" is not a bug report | Log time-to-first-token, tokens/s, error rate, queue wait; trace each request id end to end |
| 14 | **Quality regressions** | A model or prompt change silently makes answers worse | A golden set of questions run in CI; track thumbs up/down |
| 15 | **Model licence and supply chain** | llama3.2 is not MIT/Apache; unpinned deps drift | Legal review of the model licence; pin versions (done); scan images |
| 16 | **Outgrowing Gradio** | Brand design, mobile UX, complex flows | Keep `supportbot/llm.py` behind FastAPI and swap the front end for React - the back end does not change |

## 4. Minimum production checklist

- [ ] TLS and SSO in front of the app; Ollama not reachable from the internet
- [ ] Tuning controls and log tab removed from the customer build
- [ ] Postgres instead of SQLite; PII redaction and retention policy
- [ ] Rate limits, message length cap, history cap
- [ ] Dashboards: time-to-first-token, queue wait, error rate, GPU memory
- [ ] Golden-question eval running in CI
- [ ] Model licence approved
- [ ] Load test at 2x expected peak
- [ ] Human-handoff path when the bot does not know
