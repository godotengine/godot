# Screenshot Capture Specification

## Purpose

Define a repeatable, human-executed process for capturing documentation screenshots from the editor.

## Why this exists

Text-only automation cannot produce authoritative editor UI screenshots. Captures must be taken in a real editor session and reviewed for consistency.

When authoritative editor screenshots are not yet available, use a clearly labeled diagram or benchmark artifact rather than a mock editor frame. Replace those interim visuals when the real capture pass is complete.

## Capture baseline

Use these defaults unless a page requires an explicit override:

- Resolution: `1920x1080`
- UI scale: default (`100%`)
- Theme: default editor theme
- Window state: maximized, no overlapping debug windows unless documented
- Scene state: deterministic camera pose and visible inspector selection

## File naming and storage

- Store images in `docs/assets/images/`.
- Naming format: `<topic>-<context>-<state>.webp`.
- Example: `quickstart-node-inspector-initial.webp`.

## Figure contract

- Every figure must have concise alt text.
- Every figure must include a caption that explains the action or decision it supports.
- Leave standard markdown images unwrapped so the docs site can expand them through `glightbox`.
- Diagrams and benchmark artifacts must identify themselves honestly in captions; do not present them as editor screenshots.

## Per-shot metadata (required)

For each screenshot, record:

1. Target docs page path.
2. Scene/resource loaded.
3. Exact interaction steps before capture.
4. Expected visible UI elements to verify.

## Capture workflow

1. Open target scene/resource.
2. Apply deterministic state setup (camera, selection, toggles).
3. Capture screenshot.
4. Optimize image size and confirm readability.
5. Insert image into target docs page with concise caption and alt text.
6. Run docs media budget check:
   - `python3 scripts/check_docs_media_budget.py`

## Quality gates

- No sensitive/local machine information in UI.
- Captions describe what decision/action the screenshot supports.
- Images remain legible at standard docs-site layout width.
- Media budget checks pass.

## Priority pages for initial capture pass

- `docs/getting-started/quick-start.md`
- `docs/user/quickstart.md`
- `docs/workflows/importing.md`
- `docs/workflows/GSPLATWORLD_BAKE.md`
- `docs/user/manual/runtime-behavior.md`
- `docs/user/manual/lighting-behavior.md`
- `docs/troubleshooting/recurring-issues.md`
- `docs/architecture/overview.md`
- `docs/reference/compatibility-matrix.md`
- `docs/testing/benchmark-suite.md`
