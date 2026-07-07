# Known Limitations & Post-v0.1 Improvements

This file tracks things that were deliberately simplified, deferred, or
left incomplete in code already written — not part of the 10-phase build
plan, just an honest running list so nothing gets silently forgotten.

Each entry: what's simplified, why it was acceptable for now, what a real
fix would look like.

---

## config.py

**`embedding.provider` only accepts `"huggingface"`.**
Schema has the field, validation hard-rejects anything else. Intentional
for v0.1 (local-only, zero-network guarantee). Post-v0.1: add
`OllamaEmbedder`, `OpenAIEmbedder`, etc. as new `Embedder` implementations
— see embedding.py entry below.

**`retrieval.call_depth` is hardcoded to `1`.**
Field exists in the schema but any value other than `1` is rejected.
Forward-compat placeholder only, per the frozen spec. Making this
adjustable means real multi-hop graph expansion work, not a config change.

---

## embedding.py

**Real inference runs on the torch backend, not ONNX Runtime, despite the class name.**
`HuggingFaceONNXEmbedder` was originally wired to `SentenceTransformer(model,
backend="onnx")`, which routes through `optimum`'s ONNX exporter. On real
testing (not just this environment — an actual user machine with real
network access) this failed with:

```
ImportError: cannot import name '_attention_scale' from
'torch.onnx.symbolic_opset14'
```

`optimum`'s exporter reaches into a private, version-specific torch
internal API that doesn't exist across all torch releases — this is a
genuine incompatibility in the `torch`/`optimum`/`transformers` version
triangle, not a bug in Rylox's code. Chasing an exact pinned combination
of all three that's confirmed to work is a real rabbit hole (these
packages break compatibility with each other frequently), so the decision
was to revert to the default (torch) backend now — stable and verified
working — rather than ship something fragile just to say "ONNX."

`onnxruntime` is still a declared dependency and still checked by
`doctor`, but nothing currently routes inference through it. Revisit this
once a known-good version triple is found, or once `optimum`'s exporter
patches around this itself in a later release.

**No batching/rate limits considered.**
Large repos will need batched `embed()` calls with a sane batch size —
not designed yet.

---

## chunking.py

**Docstring extraction is manual string-stripping, not `ast.literal_eval`.**
Deliberate: `ast.literal_eval` requires a fully valid Python statement and
would throw on exactly the malformed-file cases the chunker needs to
survive. Current approach is more code but doesn't have that failure mode.
Known gap: doesn't handle implicit string concatenation as a docstring
(`"a" "b"` back-to-back) — vanishingly rare in practice, not tested.

**Nested `def` inside another `def` never gets a `parent_class`.**
A function nested inside another function is chunked as a plain
`"function"`, with no tracking of the enclosing function. Only class
ancestry is tracked. Acceptable for v0.1's granularity (function/method/
class only); revisit if closures/factories turn out to matter for
retrieval quality later.

**Only UTF-8 (with or without BOM) is supported.**
A file with a PEP 263 encoding cookie declaring e.g. `latin-1` will fail
to decode and get skipped as malformed (correct per spec §12, but the
*reason* it's skipped is an encoding gap, not actual malformation — worth
a clearer error message distinguishing the two later).

**Lambdas and comprehensions are never chunked.**
By design — spec §2 scope is function/method/class only. Noting it here
so it's a documented decision, not an oversight if someone asks.

---

## cache.py

**Schema version mismatch is a hard failure, no migration path.**
Any `schema_version` other than the current one raises `IndexCorruptError`
telling the user to `rylox clean`. Fine while the schema is young; once
the index format stabilizes, a real migration path (or at least a
version-to-version upgrade script) will be worth building instead of
forcing a full re-index on every schema bump.

**No file locking.**
Two `rylox index` processes running against the same repo simultaneously
could race — the atomic rename means the file is never *corrupted*, but
one process's work can be silently clobbered by the other's. Not handled;
acceptable for a single-user local tool, worth reconsidering if `serve`
mode (deferred, §14) ever makes concurrent access realistic.

**Entire index loaded into memory on every read.**
Fine at v0.1's expected repo scale. Would need a real on-disk index
structure (not one JSON blob) if repo sizes get large enough for this to
matter.

---

## indexer.py

**`.gitignore` support is simplified `fnmatch`, not the real gitignore spec.**
No `!` negation, no nested-directory precedence rules, no `**` semantics
beyond what `fnmatch` gives for free. Handles the common cases (ignoring
`node_modules`, `build/`, `*.log`) correctly. A real implementation would
use a proper gitignore-matching library rather than hand-rolled patterns.

**File-size ceiling is a hardcoded constant, not configurable.**
Spec §12 calls for "a *configurable* per-file size ceiling" — currently
`MAX_FILE_SIZE_BYTES = 5MB` is a constant in `indexer.py`, not exposed via
`rylox.toml`. Should move into `BudgetConfig` or a new config section.

**Directory walk doesn't prune ignored directories early.**
`repo.rglob("*.py")` walks the entire tree first, then filters by ignore
pattern per-file. On a repo with a huge `node_modules` or `vendor` tree,
this means fully traversing directories that get thrown away anyway.
Works correctly, just not efficiently — a real fix walks directory-by-
directory and skips descending into an ignored directory at all.

**Every file is re-hashed on every run, even unchanged ones.**
Correct, but reads and hashes the full byte content of every tracked file
on every `index` run to detect changes — no fast pre-check using mtime/size
before falling back to a content hash. Fine at v0.1 scale; worth revisiting
if `index` becomes noticeably slow on large repos.

**Symlink cycles aren't explicitly handled.**
`path.resolve()` on a symlink loop would raise `OSError` at the OS level
rather than being caught with a clear Rylox-specific message. Not
observed in testing, not specifically guarded against.
