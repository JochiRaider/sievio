# Migration Guide: RepoCapsule → Sievio

Sievio is the new name for the project previously called RepoCapsule. This release is a **hard break**—there are no `repocapsule` compatibility shims for imports or CLI entry points.

## What changed
- Python distribution/package renamed to `sievio`; imports now use `sievio` (`from sievio import SievioConfig`, etc.).
- Console scripts renamed to `sievio` and `sievio-qc`; command shapes are unchanged.
- Plugin entry point group is now `sievio.plugins`.
- Default logger name and user-agent strings use `sievio` and point at `https://github.com/jochiraider/sievio`.
- Environment variable toggles use the new prefix (for example, `SIEVIO_EVTX_RECOVER` replaces `REPOCAPSULE_EVTX_RECOVER`).

## How to migrate your code
1. Update dependencies: replace any `repocapsule` requirement or `pip install` command with `sievio` (and adjust extras accordingly, e.g., `sievio[qc]`).
2. Rewrite imports and type names: `RepocapsuleConfig` → `SievioConfig`, module paths `repocapsule.*` → `sievio.*`.
3. Switch CLIs: calls like `repocapsule run ...` should be updated to `sievio run ...` (same flags/subcommands). The `repocapsule-qc` alias is now `sievio-qc`.
4. Update plugins: if you expose entry points, move them under the `sievio.plugins` group.
5. Refresh configs/logging: any custom `logger_name` or user-agent strings that referenced the old name should be updated; defaults already point to `sievio`.

If you still need the old name for downstream consumers, pin them to the last `repocapsule` release. New releases will use `sievio` exclusively.
