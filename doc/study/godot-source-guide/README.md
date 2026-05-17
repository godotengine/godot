# Godot source guide

This folder contains the split version of the interactive Godot source-reading guide.

- `index.html`: document structure and long-form guide content.
- `architecture-map.html`: independent interactive node map for the Godot source architecture.
- `styles.css`: layout, diagrams, trace rows, tables, and responsive rules.
- `data.js`: structured data for directory cards, startup path, source paths, module tables, beginner guide cards, concept explanations, and the architecture map graph.
- `app.js`: rendering, filtering, beginner guide toggling, search, concept keyword linking, favorites, concept deep links, and scroll progress behavior.
- `source-map.js`: SVG rendering, zoom/pan, search, filtering, deep links, and node detail behavior for the architecture map.
- `validate-source-map.mjs`: local integrity checks for map nodes, relationships, concept IDs, HTML anchors, and source anchors.

The page is intentionally static and can be opened directly in a browser. When adding a new data-driven interactive section, put data in `data.js` and DOM behavior in `app.js`; keep explanatory prose in `index.html`.

The front-page file counts use reproducible Git-tracked-file commands from the repository root. The headline code-file count is `git ls-files | rg '\.(c|cc|cpp|h|hpp|hh|m|mm|py)$' | Measure-Object`, which currently reports 8,294 C/C++/Objective-C++/Python files including `thirdparty`. Directory cards use `git ls-files <top-level-dir>` for tracked files and the same extension filter for code files; build artifacts, `.git`, `bin`, and `__pycache__` are not counted.

Concept explanations are data-driven. Add a new item to the `concepts` array in `data.js` with a stable `id`, visible `title`, `aliases`, beginner explanation, implementation notes, common confusions, source entry points, and related concept ids. `article` can be a plain string or a structured block array. Block arrays support headings, paragraphs, lists, code blocks, tables, flow diagrams, and callouts. `app.js` automatically scans article text and marks matching aliases as clickable keywords, so individual paragraphs do not need manual links.

Main-guide beginner explanations are also data-driven. Add or update entries in the `beginnerGuides` object in `data.js`; each key should match the visible h3/h4 subsection heading in `index.html`. `app.js` inserts these cards automatically and the sidebar toggle controls their visibility.

The architecture map is data-driven through `sourceMapGraph` in `data.js`. Keep node IDs stable, prefer reusing an existing concept `id`, and give every node at least one verified source anchor. Use `defaultVisible: false` on secondary high-fan-out edges so the default map stays readable; those edges still appear when the connected node is selected. After editing the map, run:

```bash
node doc/study/godot-source-guide/validate-source-map.mjs
```
