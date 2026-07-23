# Accessibility: Implement hierarchical Tree accessibility, item level depth, position metrics, and AccessKit expansion support

* **PR Topic:** `[GUI]`, `[Drivers/AccessKit]`, `[Editor]`
* **OS / Platform:** Microsoft Windows 11 Pro 64-bit (`x86_64`)
* **Accessibility Assistive Technology Tested:**
  * **NVDA** (NonVisual Desktop Access version `2026.x`)
  * **Windows Narrator** (Windows 11)
  * **AccessKit Subsystem** (Windows UI Automation / UIA backend driver)

---

## Summary

This Pull Request brings a major, essential accessibility feature implementation to Godot Engine's GUI controls and Editor:

1. **New Scene Tree & `Tree` Accessibility Architecture (`[GUI]`, `[Drivers/AccessKit]`)**:
   * Introduces standard accessibility roles: `ROLE_TREE_ITEM`, `ROLE_ROW`, `ROLE_GRID`, `ROLE_GRID_CELL`, and `ROLE_GROUP`.
   * Adds **hierarchical depth tracking** (`Level 1`, `Level 2`, etc.), 1-indexed sibling set metrics (`X of Y`), and child count context. Previously in official Godot, the Scene Tree lacked structural accessibility hierarchy, leaving screen reader users unable to ascertain node depth or parent-child nesting.
2. **AccessKit Modern Expansion Support (`[Drivers/AccessKit]`, `[GUI]`)**:
   * Integrates AccessKit's expansion API (`ACTION_EXPAND`, `ACTION_COLLAPSE`, `update_set_list_item_expanded`). Previously non-existent in official Godot, screen readers can now explicitly announce collapsed/expanded states and toggle branch visibility directly.
3. **Control Tooltips Accessibility & Localization (`[GUI]`, `[Core]`)**:
   * Exposes control tooltips across GUI controls (`Control`, `Tree`, `ItemList`, `PopupMenu`, `TabBar`) translated via `atr(data.tooltip)`, providing critical contextual guidance to screen reader users.
4. **Editor Focus Loss Prevention (`[Editor]`)**:
   * Implements `grab_dock_focus()` on `EditorDock` and `SceneTreeDock`. Keyboard focus is no longer dropped into empty containers when filtering, deleting nodes, or switching scenes; focus gracefully delegates to the filter box or root creation buttons, and restores focus to selected nodes upon menu dismissal.

---

## Motivation & Rationale

### 1. Bringing True Hierarchy to Scene Tree Accessibility
In official Godot `master`, accessibility support for `Tree` controls and `SceneTreeDock` did not convey node hierarchy to screen readers:
- **Loss of Hierarchy**: The Scene Tree was exposed without structural level information. Visually impaired developers had no native way of knowing whether a node was a root, a child of a CanvasItem, or a deeply nested sub-child.
- **The Solution**: This PR updates `Tree` accessibility updates to compute parent-child depth (`child_level = parent_level + 1`). Screen readers now clearly announce node levels ("Level 1", "Level 2") and position in set ("1 of 4", "2 of 4"), unlocking true hierarchical navigation.

### 2. Modern AccessKit Expansion Integration
Standard Godot lacked integration for AccessKit's node expansion state:
- Screen readers could not determine if a parent node was collapsed or expanded.
- This PR integrates AccessKit's `update_set_list_item_expanded`, `ACTION_EXPAND`, and `ACTION_COLLAPSE`, enabling screen readers to announce expansion state automatically.

### 3. Tooltips Accessibility: Essential Context for Screen Reader Users
When a screen reader user navigates Godot's editor controls, they rely heavily on tooltips for guidance:
- In official Godot, tooltips were not consistently routed to the accessibility server.
- This PR routes control tooltips through `atr(...)` in `NOTIFICATION_ACCESSIBILITY_UPDATE`, ensuring screen reader users receive contextual help in their active editor language.

### 4. Preventing Focus Loss in Editor Docks
Keyboard focus in `SceneTreeDock` could be lost when deleting nodes, reordering elements, or working on empty/filtered scenes:
- Implementing `grab_dock_focus()` ensures focus delegates cleanly to the filter LineEdit or Add/Instantiate buttons.
- `_scene_tree_context_menu_closed()` guarantees keyboard focus returns directly to the target node when context menus close.

---

## Detailed Subsystem Breakdown

### 1. `[GUI]` (Graphical User Interface)
* **Control Tooltips (`scene/gui/control.cpp`)**: Dispatches localized tooltips via `atr(data.tooltip)` in `NOTIFICATION_ACCESSIBILITY_UPDATE`.
* **Tree Control (`scene/gui/tree.cpp`, `scene/gui/tree.h`)**:
  * Implemented `set_accessibility_as_grid()` for table/grid accessibility mode.
  * Added hierarchical `_accessibility_update_item()` depth tracking (`Level X`), sibling set counts (`X of Y`), active descendant focus tracking, and cell/button localization.
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

## Author Note & Maintainer Statement

> **Author Note**:
> I am a totally blind software engineer and accessibility specialist who relies on screen readers (NVDA and Narrator) every single day in my development workflow. I am deeply passionate about making Godot Engine 100% accessible to disabled game developers worldwide.
>
> I am actively open and eager to serve as a tester, reviewer, and ongoing collaborator for accessibility features, AccessKit integration, and assistive technology support in Godot!
>
> **Note to Godot Maintainers**:
> Without accessibility hierarchy, navigating complex scenes was extremely difficult for screen reader users. This PR introduces native structural hierarchy and expansion support to bridge that gap. I look forward to your feedback and code review to help refine and polish this contribution for Godot Engine!

---

## How to Test

1. **Scene Tree Navigation**:
   * Enable NVDA or Narrator and navigate `SceneTreeDock` with arrow keys.
   * **Expected**: Screen reader announces Node name, depth level ("Level 1", "Level 2"), expansion state ("Expanded", "Collapsed"), and position in set ("1 of 4").
2. **Expansion Actions**:
   * Collapse/expand a node via screen reader shortcut or keyboard (`Right`/`Left` arrow keys).
   * **Expected**: Expansion state is updated and announced cleanly.
3. **Tooltip Verification**:
   * Focus buttons or filter inputs in the editor.
   * **Expected**: Localized tooltips are announced.
4. **Focus Stability**:
   * Delete or filter nodes in `SceneTreeDock`.
   * **Expected**: Keyboard focus delegates to filter input or creation buttons instead of dropping into empty space.
