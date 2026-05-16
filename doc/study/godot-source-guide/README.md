# Godot source guide

This folder contains the split version of the interactive Godot source-reading guide.

- `index.html`: document structure and long-form guide content.
- `styles.css`: layout, diagrams, trace rows, tables, and responsive rules.
- `data.js`: structured data for directory cards, startup path, source paths, module tables, and concept explanations.
- `app.js`: rendering, filtering, level switching, search, concept keyword linking, favorites, and scroll progress behavior.

The page is intentionally static and can be opened directly in a browser. When adding a new data-driven interactive section, put data in `data.js` and DOM behavior in `app.js`; keep explanatory prose in `index.html`.

Concept explanations are data-driven. Add a new item to the `concepts` array in `data.js` with a stable `id`, visible `title`, `aliases`, beginner explanation, implementation notes, common confusions, source entry points, and related concept ids. `article` can be a plain string or a structured block array. Block arrays support headings, paragraphs, lists, code blocks, tables, flow diagrams, and callouts. `app.js` automatically scans article text and marks matching aliases as clickable keywords, so individual paragraphs do not need manual links.
