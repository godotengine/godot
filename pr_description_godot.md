# feat(accessibility): Add true tree/grid hierarchy, AccessKit expansion support, tooltip localization, and SceneTreeDock focus delegation

### Tested Environment & System Details

* **Godot Version:** Godot v4.8.dev (`master` branch, commit `db88f30a008b60535f7dfee9b6533e0a9aee6082`)
* **OS / Platform:** Microsoft Windows 11 Pro 64-bit (`v10.0.26200`, Build `26200`, `x86_64`)
* **Accessibility Assistive Technology Tested:**
  * **NVDA** (NonVisual Desktop Access version `2026.1`)
  * **Windows Narrator** (Windows 11 Build `26200`)
  * **AccessKit Subsystem** (Windows UI Automation / UIA backend driver)

---

## Summary

This Pull Request brings a major, essential accessibility transformation to Godot Engine's GUI controls and Editor:

1. **Complete Scene Tree & `Tree` Accessibility Overhaul (`[GUI]`, `[Drivers/AccessKit]`)**:
   * Replaces incorrect generic roles with standard `ROLE_TREE_ITEM`, `ROLE_ROW`, `ROLE_GRID`, `ROLE_GRID_CELL`, and `ROLE_GROUP`.
   * Introduces **real hierarchical depth tracking** (`Level 1`, `Level 2`, etc.), 1-indexed sibling set metrics (`X of Y`), and child counts upon expansion (`Expanded (N items)`). Previously, Godot exposed the Scene Tree as a completely flat structure, leaving blind developers unable to ascertain node depth or parent-child nesting.
2. **AccessKit Modern Expansion Support (`[Drivers/AccessKit]`, `[GUI]`)**:
   * Integrates AccessKit's expansion API (`ACTION_EXPAND`, `ACTION_COLLAPSE`, `update_set_list_item_expanded`). Previously nonexistent in Godot official, screen readers can now explicitly announce collapsed/expanded states and toggle branch visibility directly.
3. **Control Tooltips Accessibility & Localization (`[GUI]`, `[Core]`)**:
   * Exposes control tooltips across all GUI controls (`Control`, `Tree`, `ItemList`, `PopupMenu`, `TabBar`) translated via `atr(data.tooltip)`. Visually impaired users no longer lose critical contextual help and button descriptions.
4. **Editor Focus Loss Prevention (`[Editor]`)**:
   * Implements `grab_dock_focus()` on `EditorDock` and `SceneTreeDock`. Keyboard focus is no longer dropped into empty containers when filtering, deleting nodes, or switching scenes; focus gracefully delegates to the filter box or root creation buttons, and restores focus to selected nodes upon menu dismissal.

---

## Motivation & Detailed Rationale

### 1. Why Scene Tree Accessibility Needed a Complete Transformation
In official Godot `master`, the `Tree` control and `SceneTreeDock` were virtually unnavigable for screen reader users (NVDA, JAWS, Narrator, AccessKit/AT-SPI):
- **Incorrect Item Roles**: Tree items were mapped to generic/unusable roles, preventing screen readers from treating them as interactive tree elements.
- **Loss of Hierarchy**: The Scene Tree was exposed as a completely flat list. Blind developers had no way of knowing whether a node was a root, a child of a CanvasItem, or a deeply nested sub-child. Understanding scene structure was impossible.
- **The Solution**: This PR refactors `Tree` accessibility updates to compute real parent-child depth (`child_level = parent_level + 1`). Screen readers now clearly announce node levels ("Level 1", "Level 2") and position in set ("2 of 5"), unlocking true hierarchical navigation.

### 2. Modern AccessKit Expansion Integration & Child Item Counts
Standard Godot lacked support for AccessKit's node expansion state:
- Screen readers could not determine if a parent node was collapsed or expanded, nor how many children were revealed upon expanding.
- This PR integrates AccessKit's `update_set_list_item_expanded`, `ACTION_EXPAND`, and `ACTION_COLLAPSE`, and appends visible child counts ("Expanded (N items)"), enabling screen readers to announce expansion state and child counts automatically.

### 3. Tooltips Accessibility: Essential Context for Blind Users
When a blind developer navigates Godot's editor controls, they rely heavily on tooltips for guidance:
- In official Godot, tooltips were either completely hidden from accessibility or passed as raw untranslated strings.
- This PR routes all control tooltips through `atr(...)` in `NOTIFICATION_ACCESSIBILITY_UPDATE`, ensuring blind developers receive vital contextual help in their active editor language.

### 4. Preventing Focus Loss in Editor Docks
Keyboard focus in `SceneTreeDock` was frequently lost when deleting nodes, reordering elements, or working on empty/filtered scenes:
- Implementing `grab_dock_focus()` ensures focus delegates cleanly to the filter LineEdit or Add/Instantiate buttons.
- `_scene_tree_context_menu_closed()` guarantees keyboard focus returns directly to the target node when context menus close.

---

## Detailed Subsystem Breakdown

### 1. `[GUI]` (Graphical User Interface)
* **Control Tooltips (`scene/gui/control.cpp`)**: Dispatches localized tooltips via `atr(data.tooltip)` in `NOTIFICATION_ACCESSIBILITY_UPDATE`.
* **Tree Control (`scene/gui/tree.cpp`, `scene/gui/tree.h`)**:
  * Implemented `set_accessibility_as_grid()` for table/grid accessibility mode.
  * Replaced flat structure with hierarchical `_accessibility_update_item()` depth tracking (`Level X`), sibling set counts (`X of Y`), child count announcements on expansion, active descendant focus tracking, and cell/button localization.
  * Created child `ROLE_GROUP` container elements (`accessibility_group_element`) under parent tree items to encapsulate child nodes cleanly for screen readers.
* **Hierarchy Cleanup & Drivers (`scene/gui/item_list.cpp`, `scene/gui/popup_menu.cpp`, `scene/gui/code_edit.cpp`, `drivers/accesskit/`)**:
  * Fixed AccessKit 0-based position in set indexing so screen readers announce `1 of N` for initial list items through `N of N`.
  * Added persistent `level` tracking in AccessKit driver so tree depth levels persist across accessibility tree rebuilds.

### 2. `[Editor]` (Editor Docks & Workflows)
* **Dock Focus Delegation (`editor/docks/editor_dock.h`, `editor/docks/scene_tree_dock.cpp`, `.h`)**:
  * Added `virtual void grab_dock_focus()` to `EditorDock`.
  * Overrode `SceneTreeDock::grab_dock_focus()` to delegate focus to `filter` or `button_add`/`button_instance`.
  * Added `_scene_tree_context_menu_closed()` to restore focus to selected nodes.

### 3. `[Display / Accessibility Server]` (`servers/display/`)
* Pure virtual declarations and stubs for `element_set_parent`, `update_clear_children`, `update_set_author_id`, `update_set_expanded`, `update_set_checked_state`, and `update_set_selected_state`.
* Registered `ROLE_GRID`, `ROLE_GRID_CELL`, and `ROLE_GROUP` in `AccessibilityServerEnums`.

### 4. `[Drivers / AccessKit]` (`drivers/accesskit/`)
* Mapped `ROLE_GRID`, `ROLE_GRID_CELL`, and `ROLE_GROUP`.
* Integrated AccessKit node expansion state, 1-indexed position in set, and guarded level assignment for levels > 0.

---

## Long-Term Commitment & Maintainer Note

> **Note to Godot Maintainers**:
> Without these changes, Godot 4 is technically usable for simple tasks, but true game development is extremely difficult for blind developers because the Scene Tree hierarchy was effectively invisible. This PR takes a massive step forward in removing those barriers for one of the most critical parts of the editor.
>
> I am fully committed to being a long-term, regular contributor to Godot Engine's accessibility subsystem. I have further accessibility improvements planned for upcoming PRs and welcome any feedback or adjustments from core maintainers to ensure this meets Godot's highest standards!

---

## How to Test

1. **Scene Tree Navigation**:
   * Enable NVDA or Narrator and navigate `SceneTreeDock` with arrow keys.
   * **Expected**: Screen reader announces Node name, depth level ("Level 1", "Level 2"), expansion state with child count ("Expanded (3 items)", "Collapsed (3 items)"), and position in set ("1 of 4").
2. **Expansion Actions**:
   * Collapse/expand a node via screen reader shortcut or keyboard.
   * **Expected**: Expansion state and child count are announced immediately.
3. **Tooltip Verification**:
   * Focus buttons or filter inputs in the editor.
   * **Expected**: Localized tooltips are announced cleanly.
4. **Focus Stability**:
   * Delete or filter nodes in `SceneTreeDock`.
   * **Expected**: Keyboard focus delegates to filter input or creation buttons instead of dropping.
