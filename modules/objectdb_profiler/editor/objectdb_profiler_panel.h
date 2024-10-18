/**************************************************************************/
/*  objectdb_profiler_panel.h                                             */
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

#ifndef OBJECTDB_PROFILER_PANEL_H
#define OBJECTDB_PROFILER_PANEL_H

#include "core/io/dir_access.h"
#include "core/templates/lru.h"
#include "data_viewers/snapshot_view.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"
#include "snapshot_data.h"

const int SNAPSHOT_CACHE_MAX_SIZE = 10;

enum RC_MENU_OPERATIONS {
	RENAME,
	SHOW_IN_FOLDER,
	DELETE,
};

struct PartialSnapshot {
	int total_size;
	Vector<uint8_t> data;
};

// UI loaded by the debugger
class ObjectDBProfilerPanel : public Control {
	GDCLASS(ObjectDBProfilerPanel, Control);

protected:
	int next_request_id = 0;

	Tree *snapshot_list;
	Button *take_snapshot;
	TabContainer *view_tabs;
	PopupMenu *rmb_menu;
	OptionButton *diff_button;
	HashMap<int, String> diff_options;
	HashMap<int, PartialSnapshot> partial_snapshots;

	List<SnapshotView *> views;
	Ref<GameStateSnapshotRef> current_snapshot;
	Ref<GameStateSnapshotRef> diff_snapshot;
	LRUCache<String, Ref<GameStateSnapshotRef>> snapshot_cache;

	void _request_object_snapshot();
	void _show_selected_snapshot();
	Ref<DirAccess> _get_and_create_snapshot_storage_dir();
	TreeItem *_add_snapshot_button(const String &p_snapshot_file_name, const String &p_full_file_path);
	void _snapshot_rmb(const Vector2 &p_pos, MouseButton p_button);
	void _rmb_menu_pressed(int p_tool, bool p_confirm_override);
	void _apply_diff(int p_item_idx);
	void _update_diff_items();
	void _update_enabled_diff_items();
	void _edit_snapshot_name();
	void _view_tab_changed(int p_tab_idx);
	String _to_mb(int x);

public:
	ObjectDBProfilerPanel();

	void receive_snapshot(int request_id);
	void show_snapshot(const String &p_snapshot_file_name, const String &p_snapshot_diff_file_name);
	void clear_snapshot();
	Ref<GameStateSnapshotRef> get_snapshot(const String &p_snapshot_file_name);
	void set_enabled(bool enabled);
	void add_view(SnapshotView *p_to_add);

	bool handle_debug_message(const String &p_message, const Array &p_data, int p_index);
};

#endif // OBJECTDB_PROFILER_PANEL_H
