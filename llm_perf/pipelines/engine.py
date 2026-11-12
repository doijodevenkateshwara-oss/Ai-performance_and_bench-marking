"""Declarative pipeline engine — executes retrieve/rerank/generate/agent stages."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from ..config import PipelineStage
from ..metrics import record_stage, stage_span
from ..providers import RequestSpec, call_llm


class StageContext:
    def __init__(self, question: str, row: Optional[dict], scenario: str,
                 target_params: dict):
        self.question = question
        self.row = row or {}
        self.scenario = scenario
        self.params = target_params
        self.docs: list[dict] = []          # retrieved docs {text, score, ...}
        self.context_texts: list[str] = []
        self.output_text = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}


def _llm_ref_params(llm: dict, default_provider: str = "openai") -> dict:
    return llm or {}


def run_stage_generate(stage: PipelineStage, ctx: StageContext) -> None:
    """Render template + call LLM."""
    from jinja2 import Template
    llm_cfg = stage.llm or {}
    provider = llm_cfg.get("provider", ctx.params.get("provider", "openai"))
    model = llm_cfg.get("model", ctx.params.get("model", "gpt-4o"))
    template_path = stage.template
    if template_path and os.path.exists(template_path):
        template = Template(open(template_path).read())
    else:
        template = Template(stage.template or "{{question}}\n\nContext:\n{{context}}")
    context = "\n\n".join(ctx.context_texts)
    prompt = template.render(question=ctx.question, context=context,
                             row=ctx.row, **ctx.row)
    spec = RequestSpec(
        provider=provider, prompt=prompt,
        params={**llm_cfg.get("params", {}), "model": model},
        stream=False, api_key=_resolve_key(llm_cfg),
        base_url=llm_cfg.get("base_url"), timeout_s=llm_cfg.get("timeout_s", 120.0),
        extra={"azure": llm_cfg.get("azure") or {}, "aws": llm_cfg.get("aws") or {},
               "grpc": llm_cfg.get("grpc") or {}},
    )
    with stage_span("generate", ctx.scenario, provider=provider, model=model):
        start = time.monotonic()
        resp = call_llm(spec)
        dur = time.monotonic() - start
        record_stage(ctx.scenario, "generate", dur, provider=provider)
        ctx.output_text = resp.text
        ctx.usage["prompt_tokens"] += resp.prompt_tokens
        ctx.usage["completion_tokens"] += resp.completion_tokens
        if (resp.raw or {}).get("error") if isinstance(resp.raw, dict) else False:
            ctx.output_text = ""
            raise RuntimeError(f"generate stage failed: {resp.raw.get('error')}")


def run_stage_retrieve(stage: PipelineStage, ctx: StageContext) -> None:
    vs = stage.vector_store or {}
    vtype = vs.get("type", "http")
    top_k = stage.top_k
    with stage_span("retrieve", ctx.scenario, store=vtype):
        start = time.monotonic()
        docs = _retrieve(vtype, vs, ctx.question, top_k)
        dur = time.monotonic() - start
        record_stage(ctx.scenario, "retrieve", dur, store=vtype)
        ctx.docs = docs
        ctx.context_texts = [d.get("text", "") for d in docs]


def _retrieve(vtype: str, cfg: dict, query: str, top_k: int) -> list[dict]:
    if vtype == "http":
        return _retrieve_http(cfg, query, top_k)
    if vtype == "json":
        return _retrieve_json(cfg, query, top_k)
    if vtype == "chroma":
        return _retrieve_chroma(cfg, query, top_k)
    raise ValueError(f"unsupported vector_store type: {vtype}")


def _retrieve_http(cfg: dict, query: str, top_k: int) -> list[dict]:
    import requests
    url = cfg["url"]
    body = {"query": query, "top_k": top_k, **cfg.get("params", {})}
    headers = dict(cfg.get("headers", {}))
    key_env = cfg.get("api_key_env")
    if key_env and os.environ.get(key_env):
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    resp = requests.post(url, json=body, headers=headers,
                         timeout=cfg.get("timeout_s", 30))
    resp.raise_for_status()
    data = resp.json()
    # normalize: list of docs or {results|documents|chunks: [...]}
    items = data if isinstance(data, list) else (
        data.get("results") or data.get("documents") or data.get("chunks") or [])
    docs = []
    for it in items[:top_k]:
        if isinstance(it, str):
            docs.append({"text": it})
        elif isinstance(it, dict):
            text = it.get("text") or it.get("content") or it.get("page_content") or str(it)
            docs.append({"text": text, "score": it.get("score"), "raw": it})
    return docs


def _retrieve_json(cfg: dict, query: str, top_k: int) -> list[dict]:
    """Local JSON/JSONL doc store with keyword-overlap scoring (no embeddings)."""
    import json as _json
    path = cfg["path"]
    docs = []
    if path.endswith(".jsonl"):
        for line in open(path):
            if line.strip():
                docs.append(_json.loads(line))
    else:
        data = _json.load(open(path))
        docs = data if isinstance(data, list) else data.get("documents", [])
    qterms = set(_tokenize(query))
    scored = []
    for d in docs:
        text = d.get("text") or d.get("content") or (d if isinstance(d, str) else "")
        terms = set(_tokenize(str(text)))
        score = len(qterms & terms) / max(len(qterms), 1)
        scored.append((score, {"text": str(text), "score": score, "raw": d}))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:top_k]]


def _tokenize(s: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]{3,}", s.lower())


def _retrieve_chroma(cfg: dict, query: str, top_k: int) -> list[dict]:
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("chroma vector store requires `pip install chromadb` "
                           "(or use type: json/http)")
    client = chromadb.PersistentClient(path=cfg.get("path", "./chroma_db"))
    col = client.get_collection(cfg.get("collection", "llm_perf"))
    res = col.query(query_texts=[query], n_results=top_k)
    docs = []
    for i, doc in enumerate(res.get("documents", [[]])[0]):
        meta = (res.get("metadatas") or [[]])[0]
        meta = meta[i] if i < len(meta) else {}
        dist = (res.get("distances") or [[]])[0]
        score = 1.0 / (1.0 + dist[i]) if i < len(dist) else None
        docs.append({"text": doc, "score": score, "raw": meta})
    return docs


def run_stage_rerank(stage: PipelineStage, ctx: StageContext) -> None:
    if not ctx.docs:
        return
    method = stage.method or "llm"
    with stage_span("rerank", ctx.scenario, method=method):
        start = time.monotonic()
        if method == "llm":
            ctx.docs = _rerank_llm(stage, ctx)
        else:
            ctx.docs = _rerank_cross_encoder(stage, ctx)
        ctx.context_texts = [d.get("text", "") for d in ctx.docs]
        record_stage(ctx.scenario, "rerank", time.monotonic() - start, method=method)


def _rerank_llm(stage: PipelineStage, ctx: StageContext) -> list[dict]:
    """Pointwise LLM rerank: score each doc 0-10, keep order."""
    llm_cfg = stage.llm or {}
    provider = llm_cfg.get("provider", "openai")
    model = llm_cfg.get("model", "gpt-4o-mini")
    scored = []
    for d in ctx.docs:
        prompt = (f"Rate the relevance of the passage to the question from 0 to 10. "
                  f"Reply with ONLY the number.\n\nQuestion: {ctx.question}\n\n"
                  f"Passage: {d.get('text','')[:2000]}")
        spec = RequestSpec(provider=provider, prompt=prompt,
                           params={**llm_cfg.get("params", {}), "model": model,
                                   "max_tokens": 8, "temperature": 0},
                           api_key=_resolve_key(llm_cfg),
                           base_url=llm_cfg.get("base_url"), timeout_s=60)
        resp = call_llm(spec)
        try:
            score = float((resp.text or "0").strip().split()[0])
        except Exception:
            score = 0.0
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def _rerank_cross_encoder(stage: PipelineStage, ctx: StageContext) -> list[dict]:
    raise NotImplementedError("cross_encoder rerank requires sentence-transformers; "
                              "use method: llm or install the extra")


def run_stage_agent(stage: PipelineStage, ctx: StageContext) -> None:
    raise NotImplementedError(
        "agent stage lands in Phase 2b — use llm_api/pipeline(generate) targets for now")


_STAGE_FNS = {
    "retrieve": run_stage_retrieve,
    "rerank": run_stage_rerank,
    "generate": run_stage_generate,
    "agent": run_stage_agent,
}


def run_pipeline(stages: list[PipelineStage], question: str, row: Optional[dict],
                 scenario: str, target_params: dict) -> StageContext:
    ctx = StageContext(question, row, scenario, target_params)
    for st in stages:
        fn = _STAGE_FNS.get(st.stage)
        if fn is None:
            raise ValueError(f"unknown stage: {st.stage}")
        fn(st, ctx)
    return ctx


def _resolve_key(llm_cfg: dict) -> Optional[str]:
    env_name = llm_cfg.get("api_key_env")
    if env_name:
        return os.environ.get(env_name)
    # common defaults per provider
    defaults = {"openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}
    provider = llm_cfg.get("provider", "openai")
    env = defaults.get(provider)
    return os.environ.get(env) if env else None
