# Confirmed local source map

- `editor/SCsub`: editor source collection and subordinate `SConscript` calls; now includes `editor/slime_ai/SCsub`.
- `editor/register_editor_types.cpp::register_editor_types`: built-in editor plugin registration via `EditorPlugins::add_by_type<T>()`. `EditorNode` creates these in `editor/editor_node.cpp` near the `EditorPlugins::create()` loop.
- `editor/plugins/editor_plugin.h`: `EditorPlugin` base, `add_dock(EditorDock *)`, scene save/close notifications, and `get_undo_redo()`.
- `editor/editor_node.h::get_edited_scene`, `get_editor_selection`, `is_scene_unsaved`: current editor scene, selection, and unsaved status. `editor/editor_interface.cpp::get_open_scenes`, `get_unsaved_scenes` are public wrappers.
- `editor/editor_data.h::get_edited_scene_root`, `get_current_edited_scene_history_id`, `is_scene_changed`: edited roots/history. `is_scene_changed` updates its last checked version and is unsuitable as a pure read-only dirty query. Use `EditorNode::is_scene_unsaved` and a content revision for P02.
- `editor/editor_undo_redo_manager.h`: `create_action_for_history`, do/undo methods, reference ownership, and `commit_action(bool p_execute)`; `commit_action` defaults to executing do actions. Account for that when applying a transaction exactly once.
- `platform/windows/os_windows.cpp::execute_with_pipe`: returns `stdio`, `stderr`, `pid`; `drivers/windows/file_access_windows_pipe.cpp::get_length` uses `PeekNamedPipe`. `OS::kill` and `is_process_running` are available for child cleanup.
- `tests/SCsub`: auto-discovers nested test `.cpp` files and builds force-link header. `tests/test_main.cpp` uses doctest and the `--test` entry point. A nonzero matching case count is required for test evidence.

The implementation is split by responsibility: `slime_ai_editor_plugin` owns the dock and editor lifecycle; `slime_ai_service_client` owns private pipe transport; `slime_ai_project_inspector`, `slime_ai_scene_inspector`, `slime_ai_object_inspector`, and `slime_ai_api_describer` own read-only observations; `slime_ai_scene_commands`, `slime_ai_scene_transaction`, and `slime_ai_journal` own the allowlist, preview/grants/undo, and durable operation records. `editor/slime_ai/SCsub` enters only the editor build. `tools/slime_ai/agent_service` contains the offline fake child.

These symbols were inspected at local HEAD, not inferred from the remote plan. The final editor-only build and native tests verified the integration.
