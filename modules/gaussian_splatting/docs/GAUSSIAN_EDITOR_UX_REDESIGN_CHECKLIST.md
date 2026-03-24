# Gaussian Editor UX Redesign Checklist

This branch starts the Gaussian splatting editor UX move toward Godot-native asset flow:

- `GaussianSplatAsset` owns import and asset-authoring settings.
- `GaussianSplatNode3D` owns per-instance/runtime settings in the inspector.
- The bottom panel has been removed from the primary editing workflow in `GaussianEditorPlugin`.

## Ownership Model

### Asset and import settings

Keep these in the `GaussianSplatAsset` flow and its import/reimport path:

- source file and import options
- compression / reduction / normalization choices
- thumbnail generation metadata
- asset-level statistics and cached preview data
- reimport actions

Rule of thumb: if changing a value affects every instance that references the same asset, it belongs here.

### Instance and runtime settings

Keep these in the `GaussianSplatNode3D` inspector:

- assigned `GaussianSplatAsset`
- quality/runtime overrides
- debug visualization toggles
- runtime preview behavior
- per-node overrides that do not rewrite the shared asset

Rule of thumb: if the setting only changes one placed node, it belongs here.

### Bottom panel

The Gaussian bottom panel is no longer part of the normal editor flow.

- Import/reimport and preview now route through asset inspector + import dialog + resource preview pipeline.
- Any future batch/global tooling should be introduced as a separate, explicit workflow.

## TODO

### Completed in this branch

- [x] Added a module-level `EditorResourcePreviewGenerator` for `GaussianSplatAsset`.
- [x] Registered and unregistered the preview generator from `GaussianEditorPlugin`.
- [x] Added an interactive `GaussianAssetPreviewControl` for the asset inspector.
- [x] Preserved asset metadata summary and `Reimport...` action in the asset inspector.
- [x] Added safe fallback behavior when the interactive preview cannot be shown.
- [x] Kept preview generation anchored in existing thumbnail infrastructure for filesystem previews.
- [x] Enabled 3D viewport drag-and-drop acceptance for `GaussianSplatAsset`.
- [x] Added `GaussianSplatNode3D` instantiation on viewport drop with undo/redo integration.
- [x] Removed the Gaussian bottom panel from `GaussianEditorPlugin` primary workflow.

### Follow-up optional improvements

- [ ] Verify direct raw `.ply`/`.spz` drag semantics in the viewport and decide whether to support them explicitly.
- [ ] Replace the proxy inspector preview with a deeper renderer-backed preview if coupling stays manageable.
- [ ] Tighten asset inspector layout and styling once the workflow is stable.
- [ ] Add automated tests or scripted editor QA for preview and reimport behavior.

## Manual QA Checklist

Run these steps in the editor on a clean project copy:

1. Import a `.ply` file into the project.
2. Import a `.spz` file into the project.
3. Confirm the Filesystem dock shows a preview thumbnail for the imported Gaussian asset.
4. Open the imported asset in the inspector.
5. Confirm the asset inspector shows an interactive preview area.
6. Drag inside the preview and verify orbit rotation works.
7. Scroll inside the preview and verify zoom works.
8. Select an asset state that cannot build the interactive preview and confirm the fallback thumbnail or empty state appears safely.
9. Press `Reimport...` from the asset inspector and confirm the import settings dialog opens with the saved configuration.
10. Reimport the asset and confirm the metadata/thumbnail state refreshes.
11. Drag the imported Gaussian asset into the 3D viewport.
12. Confirm the drop creates a `GaussianSplatNode3D`.
13. Verify the created node has the expected asset assigned.
14. Use undo after the drop and confirm the node creation is reverted.
15. Use redo after undo and confirm the node is restored.
16. Modify the asset, reimport it, and confirm existing scene instances pick up the change or refresh as expected.
17. Trigger any hot-reload path available on this branch and confirm the editor does not lose the asset link or crash.

## Risk Notes

- The current asset preview is a proxy visualization plus thumbnail fallback, not full renderer-viewport parity yet.
- Drag-and-drop currently targets imported `GaussianSplatAsset` flow in the viewport path; direct raw file-drop semantics should be tested explicitly on this branch.
- Any future move from proxy preview to renderer-backed preview should be treated as a coupling risk and validated against editor stability.
- Bottom-panel removal should be sequenced after drag/drop and node-inspector behavior are fully stable.

## Handoff Notes

If you continue this redesign, keep the next implementation steps narrow:

- land viewport drag/drop separately from inspector polish
- land bottom-panel removal separately from instance/asset inspector cleanup
- avoid moving import ownership into node inspection
- keep fallback preview paths intact until the interactive preview is proven stable
