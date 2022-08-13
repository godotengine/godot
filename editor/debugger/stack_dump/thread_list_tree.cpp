/*************************************************************************/
/*  thread_list_tree.cpp                                                 */
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

#include "thread_list_tree.h"
#include "thread_info.h"

namespace editor::dbg::sd {

using Field = ThreadInfo::Field;

struct View::ColumnDeclaration {
	String title;
	String long_title;
	bool visible = false;
	int column_index = -1;

	ColumnDeclaration(const char *p_title, const char *p_long_title, bool p_visible) {
		title = p_title;
		long_title = p_long_title;
		visible = p_visible;
	}
};

// These are indexed by severity code in the debugger, but let's not bother including all that implementation detail.
static const int SEVERITY_NUM_VALUES = 6;
const Color View::_color_by_severity[SEVERITY_NUM_VALUES] = {
	Color(0.8, 0.8, 0.8, 1.0),
	Color(1.0, 1.0, 1.0, 1.0),
	Color(1.0, 1.0, 0.0, 1.0),
	Color(1.0, 0.89, 0.027, 1.0),
	Color(1.0, 0.78, 0.054, 1.0),
	Color(1.0, 0.0, 0.0, 1.0)
};

const String View::_tooltip_by_status[static_cast<int>(ThreadInfo::Status::NUM_VALUES)] = {
	"Thread is blocked and not available for debugging",
	"Thread is running",
	"Thread is paused after break on another thread",
	"Thread hit break or error during step execution",
	"Thread is paused for debugging",
	"Thread has encountered an error",
	"Thread has exited"
};

// Hard coded sizing.
enum Pixels {
	PIXEL_ICON_SIZE = 24,
	PIXEL_H_SEPARATION = 6,
	PIXEL_H_CHILD_MARGIN = 18,
	PIXEL_ID_WIDTH = 48
};

PopupMenu *View::_build_column_title_context_menu() const {
	PopupMenu *menu = memnew(PopupMenu());
	menu->set_allow_search(true);
	for (Field field : { Field::NAME, Field::CATEGORY, Field::LANGUAGE, Field::DEBUG_ID }) {
		menu->add_check_item(_field_info[static_cast<int>(field)].long_title, static_cast<int>(field));
		const int index = menu->get_item_index(static_cast<int>(field));
		menu->set_item_as_checkable(index, true);
		menu->set_item_checked(index, _field_info[static_cast<int>(field)].visible);
	}
	return menu;
}

inline int View::_get_column_index(Field p_field) const {
	return _field_info[static_cast<int>(p_field)].column_index;
}

inline int View::_is_field_visible(Field p_field, int &r_column) const {
	r_column = _field_info[static_cast<int>(p_field)].column_index;
	return (r_column > -1);
}

String View::_format_frame_text(const Dictionary &p_stack_frame_info) {
	return vformat("%d - %s:%d - at function: %s",
			p_stack_frame_info["frame"],
			p_stack_frame_info["file"],
			p_stack_frame_info["line"],
			p_stack_frame_info["function"]);
}

String View::_format_stack_text(const TypedArray<Dictionary> &p_stack_info) {
	Vector<String> lines;
	for (int frame = 0; frame < p_stack_info.size(); ++frame) {
		lines.append(_format_frame_text(p_stack_info[frame]));
	}
	return String("\n").join(lines);
}

void View::set_meta_thread(TreeItem &p_item, const Ref<ThreadInfo> &p_thread) {
	p_item.set_metadata(static_cast<int>(Meta::THREAD), p_thread);
}

// ReSharper disable once CppMemberFunctionMayBeConst because it actually modifies the tree through the non-const argument p_top.
void View::_remove_stack_frames(TreeItem &p_top, const ThreadInfo &p_thread) {
	if (p_thread.get_debug_thread_id() == _current &&
			nullptr != p_top.get_first_child() &&
			_field_index.size() > 2) {
		// Move focus off of any child stack frames.
		// REVISIT: Do we need to search for a selectable field in row selection mode?
		p_top.select(2);
	}
	for (TreeItem *remove_me = p_top.get_first_child(); remove_me != nullptr; remove_me = p_top.get_first_child()) {
		p_top.remove_child(remove_me);
	}
}

// Warning: this is just the stack item build we also need to do every time we rebuild the view, so it does not imply
// any user interaction.
void View::_build_stack_dump_internal(const Ref<ThreadInfo> &p_thread) {
	TreeItem &top = *p_thread->tree_item;
	_remove_stack_frames(top, **p_thread);

	const TypedArray<Dictionary> &stack = p_thread->stack_dump_info;
	if (stack.is_empty()) {
		set_meta_stack(top, Dictionary());
		return;
	}
	set_meta_stack(top, stack[0]);
	int stack_column;
	if (_is_field_visible(Field::STACK, stack_column)) {
		top.set_text(stack_column, _format_frame_text(stack[0]));
		top.set_tooltip_text(stack_column, vformat("%s\n%s", p_thread->reason, _format_stack_text(stack)));
	}
	const String empty;
	for (int frame = 1; frame < stack.size(); ++frame) {
		TreeItem &line = *create_item(&top);
		set_meta_stack(line, stack[frame]);
		set_meta_thread(line, p_thread);
		set_meta_frame(line, frame);
		for (int scan = 0; scan < get_columns(); ++scan) {
			line.set_selectable(scan, false);
		}
		int column;
		if (_is_field_visible(Field::STATUS, column)) {
			line.set_icon(column, Ref<Texture2D>());
			line.set_icon_max_width(column, PIXEL_ICON_SIZE);
		}
		if (_is_field_visible(Field::ID, column)) {
			line.set_text_alignment(column, HORIZONTAL_ALIGNMENT_CENTER);
			line.set_text(column, empty);
		}
		if (_is_field_visible(Field::DEBUG_ID, column)) {
			line.set_text_alignment(column, HORIZONTAL_ALIGNMENT_CENTER);
			line.set_text(column, empty);
		}
		if (_is_field_visible(Field::CATEGORY, column)) {
			line.set_text_alignment(column, HORIZONTAL_ALIGNMENT_CENTER);
			line.set_text(column, empty);
		}
		if (_is_field_visible(Field::STACK, column)) {
			line.set_selectable(column, true);
			line.set_text(column, _format_frame_text(stack[frame]));
		}
	}
}

void View::build_stack_dump(const Ref<ThreadInfo> &p_thread) {
	if (nullptr == p_thread->tree_item) {
		return;
	}

	_build_stack_dump_internal(p_thread);

	// When first received, the current thread's stack is the debug focus.
	if (!p_thread->stack_dump_info.is_empty()) {
		p_thread->tree_item->set_collapsed(false);
		if (p_thread->get_debug_thread_id() == _current) {
			_notify_frame_selected(**p_thread, 0, p_thread->stack_dump_info[0]);
		}
	}
}

Ref<ThreadInfo> View::get_meta_thread(TreeItem &p_item) {
	return static_cast<Ref<ThreadInfo>>(p_item.get_metadata(static_cast<int>(Meta::THREAD)));
}

void View::_rebuild() {
	// Pass 0: tear down but keep source data.
	Vector<Ref<ThreadInfo>> source_data;
	if (get_root() != nullptr) {
		for (TreeItem *item = get_root()->get_first_child(); item != nullptr; item = item->get_next()) {
			Ref<ThreadInfo> thread = get_meta_thread(*item);
			// We are about to destroy this tree item in clear() below.
			thread->tree_item = nullptr;
			source_data.append(thread);
		}
	}
	clear();
	_field_index.clear();

	// Pass 1: measure columns.
	int column = 0;
	for (int field_index = 0; field_index < static_cast<int>(Field::NUM_VALUES); ++field_index) {
		if (_field_info[field_index].visible) {
			++column;
		}
	}
	if (column < static_cast<int>(Meta::NUM_META_COLUMNS)) {
		column = static_cast<int>(Meta::NUM_META_COLUMNS);
	}
	set_columns(column);

	// Pass 2: build.
	column = 0;
	for (int field_index = 0; field_index < static_cast<int>(Field::NUM_VALUES); ++field_index) {
		if (!_field_info[field_index].visible) {
			_field_info[field_index].column_index = -1;
			continue;
		}
		Field field = get_field(field_index);
		switch (field) {
			case Field::STATUS: {
				// Extra horizontal separation in size calculation is due to an error in child size calculation in Tree control.
				set_column_custom_minimum_width(column, (PIXEL_ICON_SIZE + PIXEL_H_SEPARATION) * 2 + PIXEL_H_CHILD_MARGIN + PIXEL_H_SEPARATION);
				set_column_expand(column, false);
				break;
			}
			case Field::ID: {
				set_column_custom_minimum_width(column, PIXEL_ID_WIDTH);
				set_column_expand(column, false);
				break;
			}
			case Field::DEBUG_ID:
			case Field::NAME:
			case Field::LANGUAGE:
			case Field::CATEGORY: {
				set_column_expand(column, false);
				break;
			}
			default:
				// Other fields use default formatting.
				break;
		}
		set_column_title(column, _field_info[field_index].title);
		_field_info[field_index].column_index = column;
		_field_index.append(field);
		column++;
	}

	// Create root.
	create_item();

	// Recreate existing rows.
	for (Ref<ThreadInfo> &thread : source_data) {
		add_row(thread);
		update_status(**thread);
		update_info(**thread);
		if (!thread->stack_dump_info.is_empty()) {
			_build_stack_dump_internal(thread);
		}
	}
}

void View::_update_status_for_threads() {
	if (get_root() != nullptr) {
		for (TreeItem *item = get_root()->get_first_child(); item != nullptr; item = item->get_next()) {
			Ref<ThreadInfo> thread = get_meta_thread(*item);
			update_status(**thread);
		}
	}
}

void View::_notify_frame_selected(const ThreadInfo &p_thread, int p_frame, const Dictionary &p_frame_info) {
	emit_signal(p_thread.can_debug ? "thread_frame_selected" : "thread_nodebug_frame_selected",
			p_thread.get_debug_thread_id(),
			p_frame,
			p_frame_info);
}

void View::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_crashed_icon = get_theme_icon("ErrorSign", "EditorIcons");
			_play_start_icon = get_theme_icon("PlayStart", "EditorIcons");
			_column_title_context_menu = _build_column_title_context_menu();
			add_child(_column_title_context_menu);
			const Error ret = _column_title_context_menu->connect("id_pressed", callable_mp(this, &View::_on_columm_title_context_menu_id_pressed));
			ERR_FAIL_COND_MSG(ret != OK, "failed to connect required signal from debugger");

			// Initialize column mapping.
			_rebuild();
			break;
		}
		case NOTIFICATION_EXIT_TREE: {
			if (_column_title_context_menu != nullptr) {
				memdelete(_column_title_context_menu);
				_column_title_context_menu = nullptr;
			}
			// Disconnect from signals to undo circularities.
			if (_column_title_context_menu != nullptr) {
				_column_title_context_menu->disconnect("id_pressed", callable_mp(this, &View::_on_columm_title_context_menu_id_pressed));
			}
			disconnect("item_selected", callable_mp(this, &View::_on_item_selected));
			disconnect("column_title_clicked", callable_mp(this, &View::_on_column_title_clicked));
			break;
		}
		case NOTIFICATION_POSTINITIALIZE: {
			set_select_mode(SELECT_ROW);
			set_allow_reselect(true);
			set_column_titles_visible(true);
			set_hide_root(true);
			set_anchors_preset(PRESET_FULL_RECT);
			set_h_size_flags(SIZE_EXPAND_FILL);
			set_v_size_flags(SIZE_EXPAND_FILL);

			Error ret = connect("item_selected", callable_mp(this, &View::_on_item_selected));
			ERR_FAIL_COND_MSG(ret != OK, "failed to connect required signal 'item_selected' from Tree control");

			ret = connect("column_title_clicked", callable_mp(this, &View::_on_column_title_clicked));
			ERR_FAIL_COND_MSG(ret != OK, "failed to connect required signal 'column_title_clicked' from Tree control");

			break;
		}
		default:
			break;
	}
}

int View::get_meta_frame(TreeItem &p_item) {
	return static_cast<int>(p_item.get_metadata(static_cast<int>(Meta::FRAME)));
}

Dictionary View::get_meta_stack(TreeItem &p_item) {
	return static_cast<Dictionary>(p_item.get_metadata(static_cast<int>(Meta::STACK)));
}

void View::_on_item_selected() {
	TreeItem *row = get_selected();
	if (nullptr == row) {
		return;
	}
	Ref<ThreadInfo> thread = get_meta_thread(*row);
	_current = thread->get_debug_thread_id();
	_update_status_for_threads();
	_notify_frame_selected(**thread, get_meta_frame(*row), get_meta_stack(*row));
}

void View::_on_column_title_clicked(int p_column, int p_mouse_button_index) {
	switch (p_mouse_button_index) {
		case static_cast<int>(MouseButton::RIGHT): {
			_column_title_context_menu->set_visible(true);
			return;
		}
		case static_cast<int>(MouseButton::LEFT): {
			if (p_column >= _field_index.size()) {
				// Deferred late.
				return;
			}
			const Field field = _field_index[p_column];
			switch (field) {
				case Field::ID:
				case Field::DEBUG_ID:
				case Field::NAME:
				case Field::LANGUAGE:
					emit_signal("sort_requested", static_cast<int>(field));
					break;
				default:
					// Other fields do not sort.
					break;
			}
			break;
		}
		default:
			break;
	}
}

void View::_on_columm_title_context_menu_id_pressed(int p_id) {
	if (nullptr == _column_title_context_menu) {
		// Deferred late.
		return;
	}
	const int index = _column_title_context_menu->get_item_index(p_id);
	_column_title_context_menu->toggle_item_checked(index);
	_field_info[p_id].visible = _column_title_context_menu->is_item_checked(index);
	_rebuild();
}

void View::_bind_methods() {
	ADD_SIGNAL(MethodInfo("thread_frame_selected", PropertyInfo(Variant::PACKED_BYTE_ARRAY, "tid"), PropertyInfo(Variant::INT, "frame")));
	ADD_SIGNAL(MethodInfo("thread_nodebug_frame_selected", PropertyInfo(Variant::PACKED_BYTE_ARRAY, "tid"), PropertyInfo(Variant::INT, "frame")));
	ADD_SIGNAL(MethodInfo("sort_requested", PropertyInfo(Variant::INT, "field_index")));
}

void View::set_current(const ThreadInfo &p_thread) {
	_current = p_thread.get_debug_thread_id();
	if (_field_index.size() < 3) {
		// Not enough fields shown, should not happen.
		return;
	}
	// REVISIT: Do we need to search for a selectable field in row selection mode?
	p_thread.tree_item->select(2);
}

void View::set_meta_frame(TreeItem &p_item, int p_frame) {
	return p_item.set_metadata(static_cast<int>(Meta::FRAME), p_frame);
}

void View::set_meta_stack(TreeItem &p_item, const Dictionary &p_stack_frame_info) {
	p_item.set_metadata(static_cast<int>(Meta::STACK), p_stack_frame_info);
}

void View::add_row(Ref<ThreadInfo> &p_thread, int p_at_position) {
	TreeItem &item = *create_item(get_root(), p_at_position);
	set_meta_thread(item, p_thread);
	set_meta_frame(item, 0);
	set_meta_stack(item, Dictionary());

	int column;
	if (_is_field_visible(Field::STATUS, column)) {
		item.set_cell_mode(column, TreeItem::CELL_MODE_CHECK);
		item.set_editable(column, true);
		item.set_icon_max_width(column, PIXEL_ICON_SIZE);
		item.set_selectable(column, false);
		item.set_icon(column, nullptr);
	}
	if (_is_field_visible(Field::ID, column)) {
		item.set_text_alignment(column, HORIZONTAL_ALIGNMENT_CENTER);
		item.set_text(column, vformat("%d", p_thread->thread_number));
	}
	if (_is_field_visible(Field::DEBUG_ID, column)) {
		item.set_text_alignment(column, HORIZONTAL_ALIGNMENT_CENTER);
		item.set_text(column, p_thread->get_debug_thread_id_hex());
	}
	if (_is_field_visible(Field::CATEGORY, column)) {
		item.set_text_alignment(column, HORIZONTAL_ALIGNMENT_CENTER);
		item.set_text(column, p_thread->is_main_thread ? "Main" : "Worker");
	}
	if (_is_field_visible(Field::NAME, column)) {
		item.set_text(column, "");
	}
	if (_is_field_visible(Field::LANGUAGE, column)) {
		item.set_text(column, "");
	}
	if (_is_field_visible(Field::STACK, column)) {
		if (p_thread->stack_dump_info.size() > 0) {
			build_stack_dump(p_thread);
		} else {
			item.set_text(column, "");
			item.set_tooltip_text(column, vformat("%s\n%s", p_thread->reason, p_thread->has_stack_dump ? "(no stack info received)" : "(stack info unavailable)"));
			item.set_collapsed(true);
		}
	}

	// WARNING: Memory ownership passed to caller via thread info object.
	p_thread->tree_item = &item;
}

void View::update_status(const ThreadInfo &p_thread) {
	using Status = ThreadInfo::Status;
	int status_column;
	if (!_is_field_visible(Field::STATUS, status_column)) {
		return;
	}

	// Sanitize this value because we don't include the headers needed to know the valid range.
	int severity = p_thread.severity_code;
	if (severity < 0) {
		severity = 0;
	} else if (severity >= SEVERITY_NUM_VALUES) {
		severity = SEVERITY_NUM_VALUES - 1;
	}

	if (p_thread.get_debug_thread_id() == _current) {
		p_thread.tree_item->set_icon_modulate(status_column, _color_by_severity[severity]);
	} else {
		p_thread.tree_item->set_icon_modulate(status_column, _color_by_severity[severity] * 0.5);
	}

	switch (p_thread.status) {
		case Status::PAUSED:
		case Status::ALERT:
		case Status::BREAKPOINT: {
			p_thread.tree_item->set_icon(status_column, p_thread.can_debug ? _play_start_icon : _crashed_icon);
			break;
		}
		case Status::CRASHED:
			p_thread.tree_item->set_icon(status_column, _crashed_icon);
			break;
		default:
			p_thread.tree_item->set_icon(status_column, nullptr);
	}
	p_thread.tree_item->set_tooltip_text(status_column, _tooltip_by_status[static_cast<int>(p_thread.status)]);
}

// ReSharper disable once CppMemberFunctionMayBeConst because it actually modifies the tree through p_thread.tree_item.
void View::update_info(const ThreadInfo &p_thread) {
	if (nullptr == p_thread.tree_item) {
		return;
	}
	int column;
	if (_is_field_visible(Field::NAME, column)) {
		p_thread.tree_item->set_text(column, p_thread.thread_name);
	}
	if (_is_field_visible(Field::LANGUAGE, column)) {
		p_thread.tree_item->set_text(column, p_thread.language);
	}
}

bool View::is_pin_column_visible() const {
	return _get_column_index(Field::STATUS) > -1;
}

bool View::is_pinned(const ThreadInfo &p_thread) const {
	if (nullptr == p_thread.tree_item) {
		return false;
	}
	int column;
	if (!_is_field_visible(Field::STATUS, column)) {
		return false;
	}
	return p_thread.tree_item->is_checked(column);
}

View::View() {
	_field_info = new ColumnDeclaration[static_cast<int>(Field::NUM_VALUES)]{
		ColumnDeclaration("Pin State", "Pin and State", true),
		ColumnDeclaration("ID", "ID", true),
		ColumnDeclaration("Name", "Name", true),
		ColumnDeclaration("Where", "Where", true),
		ColumnDeclaration("Lang.", "Language", false),
		ColumnDeclaration("Category", "Category", false),
		ColumnDeclaration("Debug ID", "Debug ID", false)
	};
}

View::~View() {
	delete[] _field_info;
}

} // namespace editor::dbg::sd
