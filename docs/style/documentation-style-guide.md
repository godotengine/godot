# Documentation Style Guide

## Purpose

Keep documentation accurate, concise, and verifiable against code.

## Usage

| Step | Action | Reference |
| --- | --- | --- |
| Scope review | Read all Markdown files in the target docs area before editing. | `docs/index.md` |
| Implementation verification | Verify behavior claims against `modules/gaussian_splatting/` before publishing. | `modules/gaussian_splatting/register_types.cpp:69` |
| Style application | Prefer direct task-based writing and compact tables for inventories. | `docs/style/documentation-style-guide.md` |
| Evidence capture | Add `file:line` references for implementation claims. | `modules/gaussian_splatting/register_types.cpp:75` |
| Link validation | Run repository link checks after docs edits. | `scripts/docs/check_links.py:133` |

## API

| Rule area | Rule | Source |
| --- | --- | --- |
| Class names | Document class names only when they are registered with `GDREGISTER_CLASS` or `GDREGISTER_ABSTRACT_CLASS`. | `modules/gaussian_splatting/register_types.cpp:75` |
| Singleton names | Document singleton names exactly as registered. | `modules/gaussian_splatting/register_types.cpp:101` |
| Method/property exposure | Document methods/properties only when exposed in `_bind_methods()` and `ADD_PROPERTY`. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:84` |
| Loader API exposure | Document loader methods only when bound through `ClassDB::bind_method`. | `modules/gaussian_splatting/io/ply_loader.cpp:60` |
| C++ docs source | Keep C++ API docs aligned with Doxygen input patterns. | `docs/Doxyfile:14` |
| GDScript docs source | Keep function docs in `##` comments directly above `func` signatures. | `scripts/extract_gdscript_docs.py:34` |
| Link paths | Use relative paths that resolve from the source file directory and stay inside repository boundaries. | `scripts/docs/check_links.py:99` |

## Examples

```bash
rg -n "GDREGISTER_CLASS\(|GDREGISTER_ABSTRACT_CLASS\(" modules/gaussian_splatting/register_types.cpp
python3 scripts/docs/check_links.py docs/style
```

## Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| Documented class/singleton does not exist | Registration changed. | Re-audit `register_types.cpp` before publishing. |
| Documented method is missing at runtime | Binding removed or renamed. | Re-check `_bind_methods()` and property bindings. |
| Heading anchor link fails | Anchor text does not match checker slug normalization. | Rename heading or update anchor target. |
| Relative link is reported missing | Path resolved from wrong source directory. | Recompute path relative to the current document and rerun checker. |
