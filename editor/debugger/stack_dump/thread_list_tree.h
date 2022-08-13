/*************************************************************************/
/*  thread_list_tree.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef THREAD_LIST_TREE_H
#define THREAD_LIST_TREE_H

#include "scene/gui/tree.h"
#include "thread_info.h"

namespace editor::dbg::sd {

class View : public Tree {
	GDCLASS(View, Tree)

	using Field = ThreadInfo::Field;
	using DebugThreadID = ThreadInfo::DebugThreadID;

	// ReSharper disable once CppHidingFunction as this is normal behavior for GDCLASS objects.
	void _notification(int p_what);

	// Debug Thread ID for currently selected thread.
	DebugThreadID _current;

	// Columns defined for the tree, not necessarily all visible.  Indexed by Field enum above.
	struct ColumnDeclaration;
	ColumnDeclaration *_field_info;

	// Index from tree column to field id.
	Vector<Field> _field_index;

	// Popup menu to select shown columns.
	PopupMenu *_column_title_context_menu = nullptr;

	// Hard coded color preferences.
	static const Color _color_by_severity[];

	// Tooltip text by status code. Indexed by FieldInfo::Status.
	static const String _tooltip_by_status[];

	// Icons
	Ref<Texture2D> _play_start_icon;
	Ref<Texture2D> _crashed_icon;

	int _get_column_index(Field p_field) const;
	int _is_field_visible(Field p_field, int &r_column) const;
	PopupMenu *_build_column_title_context_menu() const;
	String _format_frame_text(const Dictionary &p_stack_frame_info);
	String _format_stack_text(const TypedArray<Dictionary> &p_stack_info);
	void _rebuild();
	void _remove_stack_frames(TreeItem &p_top, const ThreadInfo &p_thread);
	void _build_stack_dump_internal(const Ref<ThreadInfo> &p_thread);
	void _update_status_for_threads();
	void _notify_frame_selected(const ThreadInfo &p_thread, int p_frame, const Dictionary &p_frame_info);

	// Signal handlers.
	void _on_columm_title_context_menu_id_pressed(int p_id);
	void _on_item_selected();
	void _on_column_title_clicked(int p_column, int p_mouse_button_index);

protected:
	static void _bind_methods();

public:
	// Column indexes in the tree view that we use to store meta information in each tree item.
	enum class Meta {
		STACK = 0,
		THREAD,
		FRAME,
		NUM_META_COLUMNS
	};

	Field get_field(int field_index) {
		return static_cast<Field>(field_index);
	}

	void add_row(Ref<ThreadInfo> &p_thread, int p_at_position = -1);
	void build_stack_dump(const Ref<ThreadInfo> &p_thread);
	void update_status(const ThreadInfo &p_thread);
	void update_info(const ThreadInfo &p_thread);

	void set_current(const ThreadInfo &p_thread);

	bool is_pin_column_visible() const;
	bool is_pinned(const ThreadInfo &p_thread) const;

	static Ref<ThreadInfo> get_meta_thread(TreeItem &p_item);
	static void set_meta_thread(TreeItem &p_item, const Ref<ThreadInfo> &p_thread);

	static int get_meta_frame(TreeItem &p_item);
	static void set_meta_frame(TreeItem &p_item, int p_frame);

	static Dictionary get_meta_stack(TreeItem &p_item);
	static void set_meta_stack(TreeItem &p_item, const Dictionary &p_stack_frame_info);

	View();
	~View();
};

} // namespace editor::dbg::sd

#endif // THREAD_LIST_TREE_H
