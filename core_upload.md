Chose files that define the pipeline spine (config → builder → pipeline), registries/factories, QC/safety/dataset-card layers, and safe HTTP, guided by the docs and canonical tree. Included one CLI glue module plus five cross-cutting tests covering plan wiring, middleware behavior, registry/plugin loading, QC gating, and HTTP safety. Prioritized architectural wiring and extension points over leaf implementations.

#	Path	Category	Why it’s core
1	agents.md	Doc	Operating rules and invariants for AI assistants
2	llms.md	Doc	Architecture map and module responsibilities
3	project_files.md	Doc	Canonical file tree to validate paths exist
4	src/repocapsule/core/config.py	Core	Config models; single source of run specs and defaults
5	src/repocapsule/core/interfaces.py	Core	Protocols/types for sources, sinks, scorers, hooks
6	src/repocapsule/core/registries.py	Core	Registries for sources/sinks/handlers/QC scorers
7	src/repocapsule/core/factories_sources.py	Core	Builds sources and bytes handlers from specs
8	src/repocapsule/core/factories_sinks.py	Core	Sink factories and output path wiring
9	src/repocapsule/core/factories_qc.py	Core	QC/safety scorer factory wiring
10	src/repocapsule/core/builder.py	Core	Config → PipelinePlan/Runtime assembly and overrides
11	src/repocapsule/core/pipeline.py	Core	Pipeline engine, middleware wiring, runtime loop
12	src/repocapsule/core/decode.py	Core	Bytes → normalized text decoding helpers
13	src/repocapsule/core/chunk.py	Core	ChunkPolicy and text/code chunking logic
14	src/repocapsule/core/records.py	Core	Record construction, metadata schema, summaries
15	src/repocapsule/core/qc_controller.py	Core	Inline/advisory QC controller and signal handling
16	src/repocapsule/core/qc_post.py	Core	Post-hoc QC driver and sidecar export
17	src/repocapsule/core/dataset_card.py	Core	Dataset card fragments, merge/render helpers
18	src/repocapsule/core/safe_http.py	Core	Hardened HTTP client with redirect/IP safeguards
19	src/repocapsule/core/plugins.py	Core	Entry-point plugin discovery and registry wiring
20	src/repocapsule/cli/runner.py	CLI	CLI/lib glue that builds configs and runs engines
21	src/repocapsule/core/factories_context.py	Core	
22	src/repocapsule/core/hooks.py	Core	

Swap suggestions:

src/repocapsule/core/language_id.py (swap with 19 or 16) if working on language tagging/metadata.
src/repocapsule/core/qc_utils.py (swap with 15) for deeper QC heuristics and near-dup logic.
src/repocapsule/core/concurrency.py (swap with 11 or 22) when tuning executor selection and submit windows.
src/repocapsule/core/convert.py (swap with 12 or 13) if focusing on decode→chunk→record helpers outside the pipeline.
src/repocapsule/core/factories_context.py (swap with 7 or 18) for repo context and SafeHttpClient construction details.

* `builder.py`
* `chunk.py`
* `dataset_card.py`
* `decode.py`
* `factories_context.py`
* `factories_qc.py`
* `factories_sinks.py`
* `factories_sources.py`
* `hooks.py`
* `interfaces.py`
* `pipeline.py`
* `plugins.py`
* `qc_controller.py`
* `qc_post.py`
* `records.py`
* `registries.py`
* `runner.py`
* `safe_http.py`