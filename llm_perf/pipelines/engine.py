"""Declarative pipeline engine — executes retrieve/rerank/generate/agent stages.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json
import os
import time

from ..config import PipelineStage
from ..metrics import estimate_cost, record_stage, stage_span
from ..providers import RequestSpec, call_llm


class StageContext:
    """Mutable per-request state threaded through pipeline stages."""

    def __init__(
        self,
        question: str,
        row: dict | None,
        scenario: str,
        target_params: dict,
        pricing=None,
    ):
        """StageContext(question, row, scenario, target_params, pricing)."""
        self.question = question
        self.row = row or {}
        self.scenario = scenario
        self.params = target_params
        self.pricing = pricing
        self.docs: list[dict] = []  # retrieved docs {text, score, ...}
        self.context_texts: list[str] = []
        self.output_text = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.stage_metrics: dict[str, float] = {}
        self.agent_eval: dict = {}  # populated by the agent stage


def run_stage_generate(stage: PipelineStage, ctx: StageContext) -> None:
    """Render template + call LLM."""
    from jinja2 import Template

    llm_cfg = stage.llm or {}
    provider = llm_cfg.get("provider", ctx.params.get("provider", "openai"))
    model = llm_cfg.get("model", ctx.params.get("model", "gpt-4o"))
    template_path = stage.template
    if template_path and os.path.exists(template_path):
        with open(template_path) as f:
            template = Template(f.read())
    else:
        template = Template(stage.template or "{{question}}\n\nContext:\n{{context}}")
    context = "\n\n".join(ctx.context_texts)
    # row fields expand into template vars, but reserved names (question,
    # context, row) are supplied explicitly — a dataset column literally
    # named e.g. "question" must not collide with the explicit kwarg.
    row_vars = {k: v for k, v in ctx.row.items() if k not in ("question", "context", "row")}
    prompt = template.render(question=ctx.question, context=context, row=ctx.row, **row_vars)
    spec = RequestSpec(
        provider=provider,
        prompt=prompt,
        params={**llm_cfg.get("params", {}), "model": model},
        stream=False,
        api_key=_resolve_key(llm_cfg),
        base_url=llm_cfg.get("base_url"),
        timeout_s=llm_cfg.get("timeout_s", 120.0),
        extra={
            "azure": llm_cfg.get("azure") or {},
            "aws": llm_cfg.get("aws") or {},
            "grpc": llm_cfg.get("grpc") or {},
        },
    )
    with stage_span("generate", ctx.scenario, provider=provider, model=model):
        start = time.monotonic()
        resp = call_llm(spec)
        dur = time.monotonic() - start
        record_stage(ctx.scenario, "generate", dur, provider=provider)
        ctx.output_text = resp.text
        ctx.usage["prompt_tokens"] += resp.prompt_tokens
        ctx.usage["completion_tokens"] += resp.completion_tokens
        err = resp.raw.get("error") if isinstance(resp.raw, dict) else None
        if err:
            ctx.output_text = ""
            raise RuntimeError(f"generate stage failed: {err}")


def run_stage_retrieve(stage: PipelineStage, ctx: StageContext) -> None:
    """Execute the retrieve stage: fetch docs into ctx."""
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
    """Dispatch to the configured vector-store backend."""
    if vtype == "http":
        return _retrieve_http(cfg, query, top_k)
    if vtype == "json":
        return _retrieve_json(cfg, query, top_k)
    if vtype == "chroma":
        return _retrieve_chroma(cfg, query, top_k)
    raise ValueError(f"unsupported vector_store type: {vtype}")


def _retrieve_http(cfg: dict, query: str, top_k: int) -> list[dict]:
    """POST the query to an HTTP vector store; normalize doc list."""
    from ..security import read_response_text, shared_checked_session

    url = cfg["url"]
    body = {"query": query, "top_k": top_k, **cfg.get("params", {})}
    headers = dict(cfg.get("headers", {}))
    key_env = cfg.get("api_key_env")
    if key_env and os.environ.get(key_env):
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    # stream + capped read: the endpoint is untrusted, don't buffer unbounded
    resp = shared_checked_session().post(
        url,
        json=body,
        headers=headers,
        timeout=cfg.get("timeout_s", 30),
        stream=True,
    )
    resp.raise_for_status()
    data = json.loads(read_response_text(resp))
    # normalize: list of docs or {results|documents|chunks: [...]}
    items = (
        data
        if isinstance(data, list)
        else (data.get("results") or data.get("documents") or data.get("chunks") or [])
    )
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
        with open(path) as f:
            for line in f:
                if line.strip():
                    docs.append(_json.loads(line))
    else:
        with open(path) as f:
            data = _json.load(f)
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
    """Lowercase alphanumeric tokens (len>=3) for keyword scoring."""
    import re

    return re.findall(r"[a-z0-9]{3,}", s.lower())


def _retrieve_chroma(cfg: dict, query: str, top_k: int) -> list[dict]:
    """Query a local Chroma collection for top-k docs."""
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("chroma vector store requires `pip install chromadb` " "(or use type: json/http)")
    client = chromadb.PersistentClient(path=cfg.get("path", "./chroma_db"))
    col = client.get_collection(cfg.get("collection", "llm_perf"))
    res = col.query(query_texts=[query], n_results=top_k)
    docs = []
    # `or [[]]` guards both a missing key and an empty documents list
    for i, doc in enumerate((res.get("documents") or [[]])[0]):
        meta = (res.get("metadatas") or [[]])[0]
        meta = meta[i] if i < len(meta) else {}
        dist = (res.get("distances") or [[]])[0]
        score = 1.0 / (1.0 + dist[i]) if i < len(dist) else None
        docs.append({"text": doc, "score": score, "raw": meta})
    return docs


def run_stage_rerank(stage: PipelineStage, ctx: StageContext) -> None:
    """Execute the rerank stage over retrieved docs."""
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
        prompt = (
            f"Rate the relevance of the passage to the question from 0 to 10. "
            f"Reply with ONLY the number.\n\nQuestion: {ctx.question}\n\n"
            f"Passage: {d.get('text','')[:2000]}"
        )
        spec = RequestSpec(
            provider=provider,
            prompt=prompt,
            params={**llm_cfg.get("params", {}), "model": model, "max_tokens": 8, "temperature": 0},
            api_key=_resolve_key(llm_cfg),
            base_url=llm_cfg.get("base_url"),
            timeout_s=60,
        )
        resp = call_llm(spec)
        try:
            score = float((resp.text or "0").strip().split()[0])
        except Exception:
            score = 0.0
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def _rerank_cross_encoder(stage: PipelineStage, ctx: StageContext) -> list[dict]:
    """Cross-encoder rerank (requires optional deps; not implemented)."""
    raise NotImplementedError(
        "cross_encoder rerank requires sentence-transformers; " "use method: llm or install the extra"
    )


def _build_agent_tools(tool_cfgs: list[dict]):
    """Convert stage tool configs into OpenAI tool schemas + executors."""
    schemas = []
    executors = {}
    for t in tool_cfgs:
        ttype = t.get("type")
        name = t.get("name") or t.get("function", "").split(":")[-1]
        if not name and ttype == "function":
            # derive a usable tool name from the import path's callable
            name = (t.get("import_path") or "").split(":")[-1].split(".")[-1]
        if not name:
            continue
        desc = t.get("description", f"Tool {name}")
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "Tool input / query"},
                            **(t.get("parameters") or {}),
                        },
                        "required": ["input"],
                    },
                },
            }
        )
        if ttype == "http":
            executors[name] = ("http", t)
        elif ttype == "function":
            executors[name] = ("function", t)
        elif ttype == "retrieve":
            executors[name] = ("retrieve", t)
    return schemas, executors


def _execute_tool(executor, name: str, args: dict, ctx: StageContext):
    """Run one tool; returns (result_text, duration_s)."""
    import json as _json

    start = time.monotonic()
    if executor is None:
        return f"error: unknown tool {name!r}", time.monotonic() - start
    kind, cfg = executor
    try:
        if kind == "http":
            from ..security import read_response_text, shared_checked_session

            q = args.get("input") or args.get("query") or _json.dumps(args)
            body = dict(cfg.get("body_template") or {})
            body.setdefault("query", q)
            body.setdefault("input", q)
            headers = dict(cfg.get("headers") or {})
            key_env = cfg.get("api_key_env")
            if key_env and os.environ.get(key_env):
                headers["Authorization"] = f"Bearer {os.environ[key_env]}"
            # checked session re-validates redirects; capped read bounds memory
            resp = shared_checked_session().request(
                cfg.get("method", "POST"),
                cfg["url"],
                json=body,
                headers=headers,
                timeout=cfg.get("timeout_s", 30),
                stream=True,
            )
            return read_response_text(resp)[:8000], time.monotonic() - start
        if kind == "function":
            from ..security import python_execution_allowed

            if not python_execution_allowed(f"agent tool {name!r} import {cfg.get('import_path')}"):
                return (
                    "error: python execution disabled by security.python_execution=off",
                    time.monotonic() - start,
                )
            import importlib
            import inspect

            mod_path, _, attr = cfg["import_path"].partition(":")
            mod = importlib.import_module(mod_path)
            fn = mod
            for part in attr.split("."):
                fn = getattr(fn, part)
            if not callable(fn):
                return str(fn)[:8000], time.monotonic() - start
            # kwargs first (agent args are a dict); fall back to the scalar
            # "input"/first-value positional for single-arg functions
            try:
                result = fn(**args) if args else fn(args.get("input", ""))
            except TypeError:
                result = fn(args.get("input") or next(iter(args.values()), ""))
            if inspect.isawaitable(result):
                import asyncio

                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
            return str(result)[:8000], time.monotonic() - start
        if kind == "retrieve":
            docs = _retrieve(
                cfg.get("vector_store", {}).get("type", "http"),
                cfg.get("vector_store", {}),
                args.get("input", ""),
                int(cfg.get("top_k", 5)),
            )
            return "\n\n".join(d.get("text", "") for d in docs)[:8000], time.monotonic() - start
    except Exception as e:
        return f"error: tool {name} failed: {e}", time.monotonic() - start
    return "", time.monotonic() - start


def _agent_signature(name: str, args: dict) -> str:
    """Stable signature for loop detection on tool name+args."""
    try:
        canon = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        canon = str(args)
    return f"{name}:{canon}"


def _evaluate_agent_success(stage: PipelineStage, ctx: StageContext, ev: dict) -> bool:
    """Decide task success from termination + optional answer matching."""
    cfg = stage.success
    ok_terminated = ev["terminated"] == "final_answer" or (
        ev["terminated"] == "max_iterations" and cfg is not None and cfg.on_max_iterations == "pass"
    )
    if not ok_terminated or ev["final_empty"]:
        return False
    if cfg is not None:
        if cfg.require_tool_use and ev["tool_calls"] == 0:
            return False
        if cfg.expected_field:
            expected = ctx.row.get(cfg.expected_field)
            if expected is None:
                return False
            out = ctx.output_text or ""
            exp = str(expected)
            if cfg.match == "exact":
                ok = out.strip() == exp.strip()
            elif cfg.match == "regex":
                import re

                ok = bool(re.search(exp, out))
            else:  # contains
                ok = exp.lower() in out.lower()
            if not ok:
                return False
    return True


def _new_agent_eval() -> dict:
    """Fresh agent-evaluation counters dict."""
    return {
        "steps": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "unknown_tools": 0,
        "arg_errors": 0,
        "repeated_calls": 0,
        "loops": 0,
        "max_iter_reached": False,
        "terminated": "error",
        "final_empty": True,
    }


def _agent_llm_call(
    ctx: StageContext,
    llm_cfg: dict,
    provider: str,
    params: dict,
    messages: list,
    i: int,
):
    """One agent LLM step inside a stage span; returns NormalizedResponse."""
    model = params.get("model")
    spec = RequestSpec(
        provider=provider,
        prompt=ctx.question,
        messages=list(messages),
        params=params,
        stream=False,
        api_key=_resolve_key(llm_cfg),
        base_url=llm_cfg.get("base_url"),
        timeout_s=llm_cfg.get("timeout_s", 120.0),
        extra={
            "azure": llm_cfg.get("azure") or {},
            "aws": llm_cfg.get("aws") or {},
            "grpc": llm_cfg.get("grpc") or {},
        },
    )
    with stage_span("agent_step", ctx.scenario, step=i, provider=provider, model=model):
        start = time.monotonic()
        resp = call_llm(spec)
        dur = time.monotonic() - start
    record_stage(ctx.scenario, "agent_step", dur, provider=provider, step=i)
    return resp


def _agent_tool_round(
    tool_calls: list,
    executors: dict,
    ctx: StageContext,
    ev: dict,
    sig_counts: dict,
    loop_flagged: set,
    loop_threshold: int,
    messages: list,
) -> None:
    """Execute all tool calls from one assistant turn; append tool messages."""
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"input": fn.get("arguments") or ""}
            ev["arg_errors"] += 1
        if not isinstance(args, dict):
            # model returned a JSON array/scalar — wrap it as tool input
            args = {"input": args if isinstance(args, str) else json.dumps(args)}
        ev["tool_calls"] += 1
        sig = _agent_signature(name, args)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        if sig_counts[sig] > loop_threshold and sig not in loop_flagged:
            loop_flagged.add(sig)
            ev["loops"] += 1
            ev["repeated_calls"] += sig_counts[sig]
        if name not in executors:
            ev["unknown_tools"] += 1
        result_text, tool_dur = _execute_tool(executors.get(name), name, args, ctx)
        if (result_text or "").startswith("error:"):
            ev["tool_errors"] += 1
        record_stage(ctx.scenario, "agent_tool", tool_dur, tool=name)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": result_text[:8000],
            }
        )


def run_stage_agent(stage: PipelineStage, ctx: StageContext) -> None:
    """Tool-use agent loop (OpenAI-compatible chat completions)."""
    llm_cfg = stage.llm or {}
    provider = llm_cfg.get("provider", "openai")
    if provider not in ("openai", "azure"):
        raise NotImplementedError(
            f"agent stage supports openai/azure providers for tool calling, got {provider!r}"
        )
    tools_schema, executors = _build_agent_tools(stage.tools or [])
    model = llm_cfg.get("model", "gpt-4o")
    max_iter = stage.max_iterations or 6
    loop_threshold = stage.loop_repeat_threshold or 3

    messages = [{"role": "user", "content": ctx.question}]
    params = {**llm_cfg.get("params", {}), "model": model}
    if tools_schema:
        params["tools"] = tools_schema
        params["tool_choice"] = "auto"
    total_in = total_out = 0
    steps = 0
    cost_usd = 0.0

    ev = _new_agent_eval()
    sig_counts: dict[str, int] = {}
    loop_flagged: set[str] = set()

    for i in range(max_iter):
        resp = _agent_llm_call(ctx, llm_cfg, provider, params, messages, i)
        total_in += resp.prompt_tokens
        total_out += resp.completion_tokens
        steps += 1
        ev["steps"] = steps
        cost_usd += estimate_cost(resp.prompt_tokens, resp.completion_tokens, ctx.pricing)
        if stage.max_cost_usd is not None and cost_usd > stage.max_cost_usd:
            ev["terminated"] = "cost_cap"
            break

        err = (resp.raw or {}).get("error") if isinstance(resp.raw, dict) else None
        if err:
            raise RuntimeError(f"agent step {i} failed: {err}")

        tool_calls = resp.tool_calls or []
        if not tool_calls:
            ctx.output_text = resp.text
            ev["terminated"] = "final_answer"
            ev["final_empty"] = not (resp.text or "").strip()
            break

        messages.append({"role": "assistant", "content": resp.text or None, "tool_calls": tool_calls})
        _agent_tool_round(tool_calls, executors, ctx, ev, sig_counts, loop_flagged, loop_threshold, messages)
    else:
        ev["terminated"] = "max_iterations"
        ev["max_iter_reached"] = True
        ctx.output_text = ctx.output_text or "(max agent iterations reached)"
        ev["final_empty"] = not (ctx.output_text or "").strip()
    if ev["terminated"] == "cost_cap":
        ev["final_empty"] = not (ctx.output_text or "").strip()
    _finalize_agent_eval(stage, ctx, ev, steps, cost_usd, total_in, total_out)


def _finalize_agent_eval(
    stage: PipelineStage,
    ctx: StageContext,
    ev: dict,
    steps: int,
    cost_usd: float,
    total_in: int,
    total_out: int,
) -> None:
    """Score agent success and publish eval metrics/stage metrics."""
    ev["success"] = _evaluate_agent_success(stage, ctx, ev)
    ctx.usage["prompt_tokens"] += total_in
    ctx.usage["completion_tokens"] += total_out
    ctx.agent_eval = ev
    ctx.stage_metrics["agent_steps"] = steps
    ctx.stage_metrics["agent_tool_calls"] = ev["tool_calls"]
    ctx.stage_metrics["agent_loops"] = ev["loops"]
    ctx.stage_metrics["agent_success"] = 1.0 if ev["success"] else 0.0
    ctx.stage_metrics["agent_cost_usd"] = cost_usd


_STAGE_FNS = {
    "retrieve": run_stage_retrieve,
    "rerank": run_stage_rerank,
    "generate": run_stage_generate,
    "agent": run_stage_agent,
}


def run_pipeline(
    stages: list[PipelineStage],
    question: str,
    row: dict | None,
    scenario: str,
    target_params: dict,
    pricing=None,
) -> StageContext:
    """Execute stages in order against a fresh StageContext."""
    ctx = StageContext(question, row, scenario, target_params, pricing=pricing)
    for st in stages:
        fn = _STAGE_FNS.get(st.stage)
        if fn is None:
            raise ValueError(f"unknown stage: {st.stage}")
        fn(st, ctx)
    return ctx


def _resolve_key(llm_cfg: dict) -> str | None:
    """Resolve the API key env var for a stage's llm config."""
    env_name = llm_cfg.get("api_key_env")
    if env_name:
        return os.environ.get(env_name)
    # common defaults per provider
    defaults = {"openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    provider = llm_cfg.get("provider", "openai")
    env = defaults.get(provider)
    return os.environ.get(env) if env else None
