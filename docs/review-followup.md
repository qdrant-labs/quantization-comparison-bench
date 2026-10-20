# Review follow-up

This tracks the 22 review threads behind the experimental setup. It records how
each request is handled; it does not claim VM measurements have run or that a
Google Docs thread has been resolved.

| Comment IDs | Request | Disposition and evidence |
| --- | --- | --- |
| `AAACJDs--JA`, `AAACJDs--Gg`, `AAACJDs--GI`, `AAACIY_WTNk` | English data, several topics and more than one embedding model. | Four English tasks: FiQA, SciFact, NFCorpus and Quora. Supplied Cohere English v3 vectors (1,024d) and local BGE-small-en-v1.5 vectors (384d) share IDs. See [data](data.md); `qbench readiness --require-all` checks actual preparation, including missing artifacts. Language is not an experiment axis. |
| `AAACIY_WTNg` | Reuse supplied BEIR embeddings and relevance judgments; prepare before VM time. | The Cohere path imports frozen source vectors with locked file checksums and uses existing qrels. No paid embedding API is needed. `prepare-all` stages raw data and Cohere; `--with-bge` completes the second model. |
| `AAACIY_WTNs` | Include a run where original vectors exceed RAM. | The [VM runbook](vm.md) and pressure profile check the raw vector footprint against the effective Qdrant cap, hold the cap fixed across methods, and collect memory/I/O evidence. A normal 32-GiB VM run alone does not satisfy this requirement. Actual pressure results remain a VM task. |
| `AAACIY_WTNo` | Start with representative configurations; keep full results in the repository and explain constraints. | The primary configuration has one representative quantizer per family plus float32 HNSW. Full development sweeps and ablations are separate. The current guide selects against quality, latency and memory budgets and points to the repository for full tables. |
| `AAACIY_WTNc`, `AAACBs49y48` | Fix graph construction, tune search effort with oversampling, and isolate optimization effects. | Construction settings remain fixed. `configs/tuning.yaml` combines `hnsw_ef`, rescoring and oversampling on development queries; `qbench freeze` records quality-qualified settings for held-out execution. `configs/ablations.yaml` keeps search effort fixed to isolate rescoring/oversampling. See [protocol](protocol.md). |
| `AAACIRRfySc`, `AAACIRRfySQ`, `AAACApfTf3w` | Improve the decision process and avoid a simplistic method-ranking visual. | The current guide's second method-picking image is replaced with a text decision workflow: define budgets, measure baselines, evaluate candidates, tune on development data, check all budgets and freeze for held-out testing. The older single-curve proposal is superseded. The vector-byte chart is preserved with a scope caption. |
| `AAACIRec6v8` | Distinguish dimensions, quantized search copies and storage datatypes. | Current guide adds the three-way distinction. The separate `configs/datatype.yaml` compares stored Turbo4 with float32 plus a quantized copy; rescoring compressed stored vectors does not recover float32 precision. |
| `AAACHZkpflc`, `AAACBs49y4g` | Distinguish embedding quantization from model-weight methods. | Current guide explicitly separates stored embeddings from GPTQ/AWQ/QLoRA weight compression. These are not added as Qdrant benchmark methods. |
| `AAACHZkpflU`, `AAACEznlWao` | Explain Qdrant methods, how they work, when to choose them, and the testing process. | Existing method explanations are retained; the selection and measurement sections now describe concrete development/held-out steps. Final conclusions require measured data and author-owned analysis. |
| `AAACApfTf30` | Earlier duplicate of the methods/mechanisms/selection outline. | This thread was already resolved. Its requested structure is retained. |
| `AAACEzij1Fw`, `AAACBsVGLdo` | Earlier research notes on neural quantization, trained rotations and KV-cache/PagedAttention. | Superseded by the agreed Qdrant vector-search scope. The earlier draft is labeled historical; these are not presented as shipped Qdrant benchmark options. |
| `AAACBsVGLho`, `AAACEjS74S0` | Implementation and interactive TurboQuant references. | The current guide retains the [implementation explanation](https://ivanpleshkov.dev/blog/turboquant/) and [interactive walkthrough](https://arkaung.github.io/interactive-turboquant/). |

The same pass corrected 1.5-bit compression from 24× to approximately 21×,
clarified that quantized search copies retain original vectors, and separated
exact-neighbor recall from relevance nDCG.

## Completion evidence

Before measurement, preserve `data/readiness.json`, the source and embedding
manifests, repository commit, lockfile and server image digest. Run the synthetic
smoke test and a real-data smoke test against the pinned server. A source
download, a generated model cache, and a validated final vector artifact are
different stages.

On the VM, execute development tuning, freeze settings, run the held-out matrix
across three index builds and five measured passes, and collect resource and
pressure evidence. Keep full raw results and configuration provenance. Label
the primary seeded 150-query test sample; prepared artifacts retain all official
test queries for an additional full-test run. Controlled cold-cache results
need a verified cache-reset procedure and I/O evidence.

The setup and review changes make those measurements reproducible; final
quality/latency/memory comparisons and publication review remain dependent on
the resulting data.
