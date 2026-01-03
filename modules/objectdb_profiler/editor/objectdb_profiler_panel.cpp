/**************************************************************************/
/*  objectdb_profiler_panel.cpp                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "objectdb_profiler_panel.h"

#include "../snapshot_collector.h"
#include "data_viewers/class_view.h"
#include "data_viewers/node_view.h"
#include "data_viewers/object_view.h"
#include "data_viewers/refcounted_view.h"
#include "data_viewers/summary_view.h"

#include "core/config/project_settings.h"
#include "core/os/time.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/docks/inspector_dock.h"
#include "editor/editor_node.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"

// ObjectDB snapshots are very large. In remote_debugger_peer.cpp, the max in_buf and out_buf size is 8mb.
// Snapshots are typically larger than that, so we send them 6mb at a time. Leaving 2mb for other data.
const int SNAPSHOT_CHUNK_SIZE = 6 << 20;

void ObjectDBProfilerPanel::_request_object_snapshot() {
	take_snapshot->set_disabled(true);
	take_snapshot->set_text(TTRC("Generating Snapshot"));
	// Pause the game while the snapshot is taken so the state of the game isn't modified as we capture the snapshot.
	if (EditorDebuggerNode::get_singleton()->get_current_debugger()->is_breaked()) {
		requested_break_for_snapshot = false;
		_begin_object_snapshot();
	} else {
		awaiting_debug_break = true;
		requested_break_for_snapshot = true; // We only need to resume the game if we are the ones who paused it.
		EditorDebuggerNode::get_singleton()->debug_break();
	}
}

void ObjectDBProfilerPanel::_on_debug_breaked(bool p_reallydid, bool p_can_debug, const String &p_reason, bool p_has_stackdump, int p_thread_id) {
	if (p_reallydid && awaiting_debug_break) {
		awaiting_debug_break = false;
		_begin_object_snapshot();
	}
}

void ObjectDBProfilerPanel::_begin_object_snapshot() {
	Array args = { next_request_id++, SnapshotCollector::get_godot_version_string() };
	EditorDebuggerNode::get_singleton()->get_current_debugger()->send_message("snapshot:request_prepare_snapshot", args);
}

bool ObjectDBProfilerPanel::handle_debug_message(const String &p_message, const Array &p_data, int p_index) {
	if (p_message == "snapshot:snapshot_prepared") {
		int request_id = p_data[0];
		int total_size = p_data[1];
		partial_snapshots[request_id] = PartialSnapshot();
		partial_snapshots[request_id].total_size = total_size;
		Array args = { request_id, 0, SNAPSHOT_CHUNK_SIZE };
		take_snapshot->set_text(vformat(TTR("Receiving Snapshot (0/%s MiB)"), _to_mb(total_size)));
		EditorDebuggerNode::get_singleton()->get_current_debugger()->send_message("snapshot:request_snapshot_chunk", args);
		return true;
	}
	if (p_message == "snapshot:snapshot_chunk") {
		int request_id = p_data[0];
		PartialSnapshot &chunk = partial_snapshots[request_id];
		chunk.data.append_array(p_data[1]);
		take_snapshot->set_text(vformat(TTR("Receiving Snapshot (%s/%s MiB)"), _to_mb(chunk.data.size()), _to_mb(chunk.total_size)));
		if (chunk.data.size() != chunk.total_size) {
			Array args = { request_id, chunk.data.size(), chunk.data.size() + SNAPSHOT_CHUNK_SIZE };
			EditorDebuggerNode::get_singleton()->get_current_debugger()->send_message("snapshot:request_snapshot_chunk", args);
			return true;
		}

		take_snapshot->set_text(TTRC("Visualizing Snapshot"));
		// Wait a frame just so the button has a chance to update its text so the user knows what's going on.
		get_tree()->connect("process_frame", callable_mp(this, &ObjectDBProfilerPanel::receive_snapshot).bind(request_id), CONNECT_ONE_SHOT);
		return true;
	}
	return false;
}

void ObjectDBProfilerPanel::receive_snapshot(int request_id) {
	const Vector<uint8_t> &in_data = partial_snapshots[request_id].data;
	Ref<DirAccess> snapshot_dir = _get_and_create_snapshot_storage_dir();
	if (snapshot_dir.is_valid()) {
		Error err;
		String base_snapshot_file_name = Time::get_singleton()->get_datetime_string_from_system(false).replace_char('T', '_').replace_char(':', '-');
		String snapshot_file_name = base_snapshot_file_name;
		String current_dir = snapshot_dir->get_current_dir();
		String joined_dir = current_dir.path_join(snapshot_file_name) + ".odb_snapshot";

		for (int i = 2; FileAccess::exists(joined_dir); i++) {
			snapshot_file_name = base_snapshot_file_name + '_' + String::chr('0' + i);
			joined_dir = current_dir.path_join(snapshot_file_name) + ".odb_snapshot";
		}

		Ref<FileAccess> file = FileAccess::open(joined_dir, FileAccess::WRITE, &err);
		if (err == OK) {
			file->store_buffer(in_data);
			file->close(); // RAII could do this typically, but we want to read the file in _show_selected_snapshot, so we have to finalize the write before that.

			_add_snapshot_button(snapshot_file_name, joined_dir);
			snapshot_list->deselect_all();
			snapshot_list->set_selected(snapshot_list->get_root()->get_first_child());
			snapshot_list->ensure_cursor_is_visible();
			_show_selected_snapshot();
		} else {
			ERR_PRINT("Could not persist ObjectDB Snapshot: " + String(error_names[err]));
		}
	}
	partial_snapshots.erase(request_id);
	if (requested_break_for_snapshot) {
		EditorDebuggerNode::get_singleton()->debug_continue();
	}
	take_snapshot->set_disabled(false);
	take_snapshot->set_text("Take ObjectDB Snapshot");
}

Ref<DirAccess> ObjectDBProfilerPanel::_get_and_create_snapshot_storage_dir() {
	String profiles_dir = "user://";
	Ref<DirAccess> da = DirAccess::open(profiles_dir);
	ERR_FAIL_COND_V_MSG(da.is_null(), nullptr, vformat("Could not open 'user://' directory: '%s'.", profiles_dir));
	Error err = da->change_dir("objectdb_snapshots");
	if (err != OK) {
		Error err_mk = da->make_dir("objectdb_snapshots");
		Error err_ch = da->change_dir("objectdb_snapshots");
		ERR_FAIL_COND_V_MSG(err_mk != OK || err_ch != OK, nullptr, "Could not create ObjectDB Snapshots directory: " + da->get_current_dir());
	}
	return da;
}

TreeItem *ObjectDBProfilerPanel::_add_snapshot_button(const String &p_snapshot_file_name, const String &p_full_file_path) {
	TreeItem *item = snapshot_list->create_item(snapshot_list->get_root());
	item->set_text(0, p_snapshot_file_name);
	item->set_metadata(0, p_full_file_path);
	item->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_DISABLED);
	item->move_before(snapshot_list->get_root()->get_first_child());
	_update_diff_items();
	_update_enabled_diff_items();
	return item;
}

void ObjectDBProfilerPanel::_show_selected_snapshot() {
	if (snapshot_list->get_selected()->get_text(0) == (String)diff_button->get_selected_metadata()) {
		for (int i = 0; i < diff_button->get_item_count(); i++) {
			if (diff_button->get_item_text(i) == current_snapshot->name) {
				diff_button->select(i);
				break;
			}
		}
	}
	show_snapshot(snapshot_list->get_selected()->get_text(0), diff_button->get_selected_metadata());
	_update_enabled_diff_items();
}

void ObjectDBProfilerPanel::_on_snapshot_deselected() {
	snapshot_list->deselect_all();
	diff_button->select(0);
	clear_snapshot();
	_update_enabled_diff_items();
}

Ref<GameStateSnapshot> ObjectDBProfilerPanel::get_snapshot(const String &p_snapshot_file_name) {
	if (snapshot_cache.has(p_snapshot_file_name)) {
		return snapshot_cache.get(p_snapshot_file_name);
	}

	Ref<DirAccess> snapshot_dir = _get_and_create_snapshot_storage_dir();
	ERR_FAIL_COND_V_MSG(snapshot_dir.is_null(), nullptr, "Could not access ObjectDB Snapshot directory");

	String full_file_path = snapshot_dir->get_current_dir().path_join(p_snapshot_file_name) + ".odb_snapshot";

	Error err;
	Ref<FileAccess> snapshot_file = FileAccess::open(full_file_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V_MSG(err != OK, nullptr, "Could not open ObjectDB Snapshot file: " + full_file_path);

	Vector<uint8_t> content = snapshot_file->get_buffer(snapshot_file->get_length()); // We want to split on newlines, so normalize them.
	ERR_FAIL_COND_V_MSG(content.is_empty(), nullptr, "ObjectDB Snapshot file is empty: " + full_file_path);

	Ref<GameStateSnapshot> snapshot = GameStateSnapshot::create_ref(p_snapshot_file_name, content);
	if (snapshot.is_valid()) {
		snapshot_cache.insert(p_snapshot_file_name, snapshot);
	}

	return snapshot;
}

void ObjectDBProfilerPanel::show_snapshot(const String &p_snapshot_file_name, const String &p_snapshot_diff_file_name) {
	clear_snapshot(false);

	current_snapshot = get_snapshot(p_snapshot_file_name);
	if (!p_snapshot_diff_file_name.is_empty()) {
		diff_snapshot = get_snapshot(p_snapshot_diff_file_name);
	}

	_update_view_tabs();
	_view_tab_changed(view_tabs->get_current_tab());
}

void ObjectDBProfilerPanel::_view_tab_changed(int p_tab_idx) {
	// Populating tabs only on tab changed because we're handling a lot of data,
	// and the editor freezes for a while if we try to populate every tab at once.
	SnapshotView *view = cast_to<SnapshotView>(view_tabs->get_current_tab_control());
	GameStateSnapshot *snapshot = current_snapshot.ptr();
	GameStateSnapshot *diff = diff_snapshot.ptr();
	if (snapshot != nullptr && !view->is_showing_snapshot(snapshot, diff)) {
		view->show_snapshot(snapshot, diff);
	}
}

void ObjectDBProfilerPanel::clear_snapshot(bool p_update_view_tabs) {
	for (SnapshotView *view : views) {
		view->clear_snapshot();
	}

	const Object *edited_object = InspectorDock::get_inspector_singleton()->get_edited_object();
	if (Object::cast_to<SnapshotDataObject>(edited_object)) {
		EditorNode::get_singleton()->push_item(nullptr);
	}

	current_snapshot.unref();
	diff_snapshot.unref();

	if (p_update_view_tabs) {
		_update_view_tabs();
	}
}

void ObjectDBProfilerPanel::set_enabled(bool p_enabled) {
	take_snapshot->set_text(TTRC("Take ObjectDB Snapshot"));
	take_snapshot->set_disabled(!p_enabled);
}

void ObjectDBProfilerPanel::_snapshot_rmb(const Vector2 &p_pos, MouseButton p_button) {
	if (p_button != MouseButton::RIGHT) {
		return;
	}
	rmb_menu->clear(false);

	rmb_menu->add_icon_item(get_editor_theme_icon(SNAME("Rename")), TTRC("Rename"), OdbProfilerMenuOptions::ODB_MENU_RENAME);
	rmb_menu->add_icon_item(get_editor_theme_icon(SNAME("Folder")), TTRC("Show in File Manager"), OdbProfilerMenuOptions::ODB_MENU_SHOW_IN_FOLDER);
	rmb_menu->add_icon_item(get_editor_theme_icon(SNAME("Remove")), TTRC("Delete"), OdbProfilerMenuOptions::ODB_MENU_DELETE);

	rmb_menu->set_position(snapshot_list->get_screen_position() + p_pos);
	rmb_menu->reset_size();
	rmb_menu->popup();
}

void ObjectDBProfilerPanel::_rmb_menu_pressed(int p_tool, bool p_confirm_override) {
	String file_path = snapshot_list->get_selected()->get_metadata(0);
	String global_path = ProjectSettings::get_singleton()->globalize_path(file_path);
	switch (rmb_menu->get_item_id(p_tool)) {
		case OdbProfilerMenuOptions::ODB_MENU_SHOW_IN_FOLDER: {
			OS::get_singleton()->shell_show_in_file_manager(global_path, true);
			break;
		}
		case OdbProfilerMenuOptions::ODB_MENU_DELETE: {
			DirAccess::remove_file_or_error(global_path);
			snapshot_list->get_root()->remove_child(snapshot_list->get_selected());
			if (snapshot_list->get_root()->get_child_count() > 0) {
				snapshot_list->set_selected(snapshot_list->get_root()->get_first_child());
			} else {
				// If we deleted the last snapshot, jump back to the summary tab and clear everything out.
				clear_snapshot();
			}
			_update_diff_items();
			break;
		}
		case OdbProfilerMenuOptions::ODB_MENU_RENAME: {
			snapshot_list->edit_selected(true);
			break;
		}
	}
}

void ObjectDBProfilerPanel::_edit_snapshot_name() {
	String new_snapshot_name = snapshot_list->get_selected()->get_text(0);
	String full_file_with_path = snapshot_list->get_selected()->get_metadata(0);
	Vector<String> full_path_parts = full_file_with_path.rsplit("/", false, 1);
	String full_file_path = full_path_parts[0];
	String file_name = full_path_parts[1];
	String old_snapshot_name = file_name.split(".")[0];
	String new_full_file_path = full_file_path.path_join(new_snapshot_name) + ".odb_snapshot";

	bool name_taken = false;
	for (int i = 0; i < snapshot_list->get_root()->get_child_count(); i++) {
		TreeItem *item = snapshot_list->get_root()->get_child(i);
		if (item != snapshot_list->get_selected()) {
			if (item->get_text(0) == new_snapshot_name) {
				name_taken = true;
				break;
			}
		}
	}

	if (name_taken || new_snapshot_name.contains_char(':') || new_snapshot_name.contains_char('\\') || new_snapshot_name.contains_char('/') || new_snapshot_name.begins_with(".") || new_snapshot_name.is_empty()) {
		EditorNode::get_singleton()->show_warning(TTRC("Invalid snapshot name."));
		snapshot_list->get_selected()->set_text(0, old_snapshot_name);
		return;
	}

	Error err = DirAccess::rename_absolute(full_file_with_path, new_full_file_path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTRC("Snapshot rename failed"));
		snapshot_list->get_selected()->set_text(0, old_snapshot_name);
	} else {
		snapshot_list->get_selected()->set_metadata(0, new_full_file_path);
	}

	_update_diff_items();
	_show_selected_snapshot();
}

ObjectDBProfilerPanel::ObjectDBProfilerPanel() {
	set_name(TTRC("ObjectDB Profiler"));

	snapshot_cache = LRUCache<String, Ref<GameStateSnapshot>>(SNAPSHOT_CACHE_MAX_SIZE);

	EditorDebuggerNode::get_singleton()->get_current_debugger()->connect("breaked", callable_mp(this, &ObjectDBProfilerPanel::_on_debug_breaked));

	HSplitContainer *root_container = memnew(HSplitContainer);
	root_container->set_anchors_preset(Control::LayoutPreset::PRESET_FULL_RECT);
	root_container->set_v_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	root_container->set_h_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	root_container->set_split_offset(300 * EDSCALE);
	add_child(root_container);

	VBoxContainer *snapshot_column = memnew(VBoxContainer);
	root_container->add_child(snapshot_column);

	take_snapshot = memnew(Button(TTRC("Take ObjectDB Snapshot")));
	snapshot_column->add_child(take_snapshot);
	take_snapshot->connect(SceneStringName(pressed), callable_mp(this, &ObjectDBProfilerPanel::_request_object_snapshot));

	snapshot_list = memnew(Tree);
	snapshot_list->create_item();
	snapshot_list->set_hide_folding(true);
	snapshot_column->add_child(snapshot_list);
	snapshot_list->set_select_mode(Tree::SelectMode::SELECT_ROW);
	snapshot_list->set_hide_root(true);
	snapshot_list->set_columns(1);
	snapshot_list->set_column_titles_visible(true);
	snapshot_list->set_column_title(0, "Snapshots");
	snapshot_list->set_column_expand(0, true);
	snapshot_list->set_column_clip_content(0, true);
	snapshot_list->connect(SceneStringName(item_selected), callable_mp(this, &ObjectDBProfilerPanel::_show_selected_snapshot));
	snapshot_list->connect("nothing_selected", callable_mp(this, &ObjectDBProfilerPanel::_on_snapshot_deselected));
	snapshot_list->connect("item_edited", callable_mp(this, &ObjectDBProfilerPanel::_edit_snapshot_name));
	snapshot_list->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	snapshot_list->set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	snapshot_list->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	snapshot_list->set_theme_type_variation("TreeSecondary");

	snapshot_list->set_allow_rmb_select(true);
	snapshot_list->connect("item_mouse_selected", callable_mp(this, &ObjectDBProfilerPanel::_snapshot_rmb));

	rmb_menu = memnew(PopupMenu);
	add_child(rmb_menu);
	rmb_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ObjectDBProfilerPanel::_rmb_menu_pressed).bind(false));

	HBoxContainer *diff_button_and_label = memnew(HBoxContainer);
	diff_button_and_label->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	snapshot_column->add_child(diff_button_and_label);
	Label *diff_against = memnew(Label(TTRC("Diff Against:")));
	diff_button_and_label->add_child(diff_against);

	diff_button = memnew(OptionButton);
	diff_button->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	diff_button->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	diff_button->connect(SceneStringName(item_selected), callable_mp(this, &ObjectDBProfilerPanel::_show_selected_snapshot).unbind(1));
	diff_button_and_label->add_child(diff_button);

	// Tabs of various views right for each snapshot.
	view_tabs = memnew(TabContainer);
	view_tabs->set_theme_type_variation("TabContainerInner");
	root_container->add_child(view_tabs);
	view_tabs->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	view_tabs->set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	view_tabs->connect("tab_changed", callable_mp(this, &ObjectDBProfilerPanel::_view_tab_changed));

	add_view(memnew(SnapshotSummaryView));
	add_view(memnew(SnapshotClassView));
	add_view(memnew(SnapshotObjectView));
	add_view(memnew(SnapshotNodeView));
	add_view(memnew(SnapshotRefCountedView));

	set_enabled(false);

	// Load all the snapshot names from disk.
	Ref<DirAccess> snapshot_dir = _get_and_create_snapshot_storage_dir();
	if (snapshot_dir.is_valid()) {
		for (const String &file_name : snapshot_dir->get_files()) {
			Vector<String> name_parts = file_name.split(".");
			ERR_CONTINUE_MSG(name_parts.size() != 2 || name_parts[1] != "odb_snapshot", "ObjectDB snapshot file did not have .odb_snapshot extension. Skipping: " + file_name);
			_add_snapshot_button(name_parts[0], snapshot_dir->get_current_dir().path_join(file_name));
		}
	}
}

void ObjectDBProfilerPanel::add_view(SnapshotView *p_to_add) {
	views.push_back(p_to_add);
	view_tabs->add_child(p_to_add);
	_update_view_tabs();
}

void ObjectDBProfilerPanel::_update_view_tabs() {
	bool has_snapshot = current_snapshot.is_valid();
	for (int i = 1; i < view_tabs->get_tab_count(); i++) {
		view_tabs->set_tab_disabled(i, !has_snapshot);
	}

	if (!has_snapshot) {
		view_tabs->set_current_tab(0);
	}
}

void ObjectDBProfilerPanel::_update_diff_items() {
	diff_button->clear();
	diff_button->add_item(TTRC("None"), 0);
	diff_button->set_item_metadata(0, String());
	diff_button->set_item_auto_translate_mode(0, Node::AUTO_TRANSLATE_MODE_ALWAYS);

	for (int i = 0; i < snapshot_list->get_root()->get_child_count(); i++) {
		String name = snapshot_list->get_root()->get_child(i)->get_text(0);
		diff_button->add_item(name);
		diff_button->set_item_metadata(i + 1, name);
	}
}

void ObjectDBProfilerPanel::_update_enabled_diff_items() {
	TreeItem *selected_snapshot = snapshot_list->get_selected();
	if (selected_snapshot == nullptr) {
		diff_button->set_disabled(true);
		return;
	}

	diff_button->set_disabled(false);

	String snapshot_name = selected_snapshot->get_text(0);
	for (int i = 0; i < diff_button->get_item_count(); i++) {
		diff_button->set_item_disabled(i, diff_button->get_item_text(i) == snapshot_name);
	}
}

String ObjectDBProfilerPanel::_to_mb(int p_x) {
	return String::num((double)p_x / (double)(1 << 20), 2);
}
