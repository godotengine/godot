/*************************************************************************/
/*  script_watches.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "script_watches.h"

#include "core/math/expression.h"
#include "editor/script_editor_debugger.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tool_button.h"

class ScriptWatch : public HBoxContainer {

	GDCLASS(ScriptWatch, HBoxContainer);

	void _expression_entered(const String &p_new_expression) {
		expression_line->release_focus();
	}

	void _expression_changed(const String &p_new_expression) {
		emit_signal("expression_changed", this);
	}

	void _lock_pressed(bool p_locked) {
		emit_signal("lock_changed", this, p_locked);
	}

	void _track_pressed(bool p_tracking) {
		emit_signal("tracking_changed", this, p_tracking);
	}

	void _remove_pressed() {
		emit_signal("watch_removed", this);
	}

	void _expression_lost_focus() {
		emit_signal("focus_lost", this);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("_expression_entered", "new_expression"), &ScriptWatch::_expression_entered);
		ClassDB::bind_method(D_METHOD("_expression_changed", "new_expression"), &ScriptWatch::_expression_changed);
		ClassDB::bind_method(D_METHOD("_lock_pressed"), &ScriptWatch::_lock_pressed);
		ClassDB::bind_method(D_METHOD("_track_pressed"), &ScriptWatch::_track_pressed);
		ClassDB::bind_method(D_METHOD("_remove_pressed"), &ScriptWatch::_remove_pressed);
		ClassDB::bind_method(D_METHOD("_expression_lost_focus"), &ScriptWatch::_expression_lost_focus);

		ADD_SIGNAL(MethodInfo("expression_changed", PropertyInfo(Variant::OBJECT, "watch")));
		ADD_SIGNAL(MethodInfo("lock_changed", PropertyInfo(Variant::OBJECT, "watch"), PropertyInfo(Variant::BOOL, "locked")));
		ADD_SIGNAL(MethodInfo("tracking_changed", PropertyInfo(Variant::OBJECT, "watch"), PropertyInfo(Variant::BOOL, "tracking")));
		ADD_SIGNAL(MethodInfo("watch_removed", PropertyInfo(Variant::OBJECT, "watch")));
		ADD_SIGNAL(MethodInfo("focus_lost", PropertyInfo(Variant::OBJECT, "watch")));
	}

	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_ENTER_TREE:
				remove_button->set_icon(get_icon("Remove", "EditorIcons"));
				lock_toggle->add_icon_override("checked", get_icon("Lock", "EditorIcons"));
				lock_toggle->add_icon_override("unchecked", get_icon("Unlock", "EditorIcons"));
		}
	}

public:
	Label *id_label;
	LineEdit *expression_line;
	RichTextLabel *result_label;
	CheckBox *lock_toggle;
	CheckBox *track_toggle;
	ToolButton *remove_button;
	int id;

	void set_id(int p_watch_id) {
		id = p_watch_id;
		id_label->set_text(("#" + itos(p_watch_id).lpad(2, "0")));
	}

	void set_result(const String &result, bool success) {
		if (success) {
			result_label->set_text(result);
		} else {
			result_label->clear();
			result_label->push_color(get_color("error_color", "Editor"));
			result_label->add_text(result);
			result_label->pop();
		}
	}

	void disable(bool p_disable) {
		Control::FocusMode focus;
		if (p_disable) {
			focus = FOCUS_NONE;
		} else {
			focus = FOCUS_ALL;
		}

		expression_line->set_editable(!p_disable);
		expression_line->set_focus_mode(focus);
		lock_toggle->set_disabled(p_disable);
		track_toggle->set_disabled(p_disable);
	}

	ScriptWatch(int p_watch_id) {
		id_label = memnew(Label);
		id_label->set_text(("#" + itos(p_watch_id).lpad(2, "0")));
		add_child(id_label);

		expression_line = memnew(LineEdit);
		expression_line->set_placeholder("Enter expression...");
		expression_line->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		expression_line->connect("text_entered", this, "_expression_entered");
		expression_line->connect("text_changed", this, "_expression_changed");
		expression_line->connect("focus_exited", this, "_expression_lost_focus");
		add_child(expression_line);

		result_label = memnew(RichTextLabel);
		result_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		result_label->set_scroll_active(false);
		add_child(result_label);

		lock_toggle = memnew(CheckBox);
		lock_toggle->set_focus_mode(FOCUS_NONE);
		lock_toggle->connect("toggled", this, "_lock_pressed");
		add_child(lock_toggle);

		track_toggle = memnew(CheckBox);
		track_toggle->set_focus_mode(FOCUS_NONE);
		track_toggle->set_visible(false); // Feature disabled temporarily
		track_toggle->connect("toggled", this, "_track_pressed");
		add_child(track_toggle);

		remove_button = memnew(ToolButton);
		remove_button->set_icon(get_icon("Error", "EditorIcons"));
		remove_button->connect("pressed", this, "_remove_pressed");
		add_child(remove_button);

		this->id = p_watch_id;
	}
};

void ScriptWatches::_watch_changed(Object *p_watch) {

	ScriptWatch *w = cast_to<ScriptWatch>(p_watch);
	const String &new_expression = w->expression_line->get_text();

	if (!new_expression.empty()) {
		if (p_watch == watches[watches.size() - 1]) {
			add_watch(new_expression);

			const int length = w->expression_line->get_text().length();
			w->expression_line->set_cursor_position(length);

			emit_signal("watch_added", "");
		}
	}
}

void ScriptWatches::_lock_changed(Object *p_watch, bool p_locked) {
	ScriptWatch *w = cast_to<ScriptWatch>(p_watch);

	emit_signal("lock_updated", w->id - 1, p_locked);
}

void ScriptWatches::_tracking_changed(Object *p_watch, bool p_tracking) {
	ScriptWatch *w = cast_to<ScriptWatch>(p_watch);

	emit_signal("tracking_updated", w->id - 1, p_tracking);
}

void ScriptWatches::_watch_removed(Object *p_watch) {
	const int num_watches = watches.size();
	ScriptWatch *w = cast_to<ScriptWatch>(p_watch);

	if (watches[num_watches - 1] != w) {
		_remove_watch(w);
		emit_signal("watch_removed", w->id - 1);
	}
}

void ScriptWatches::_lost_focus(Object *p_watch) {
	if (p_watch != _get_watch_ghost()) {
		ScriptWatch *w = cast_to<ScriptWatch>(p_watch);
		const String &expr = w->expression_line->get_text();

		emit_signal("expression_updated", w->id - 1, expr);
		if (w->lock_toggle->is_pressed()) {
			w->lock_toggle->set_pressed(false);
			_lock_changed(p_watch, false);
		}
	}
}

void ScriptWatches::_breaked(bool p_breaked, bool p_can_debug) {
	for (int i = 0; i < get_watch_count(); ++i) {
		watches[i]->lock_toggle->set_disabled(!p_breaked);
	}
}

void ScriptWatches::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_watch_changed", "watch"), &ScriptWatches::_watch_changed);
	ClassDB::bind_method(D_METHOD("_lock_changed", "watch"), &ScriptWatches::_lock_changed);
	ClassDB::bind_method(D_METHOD("_tracking_changed", "watch"), &ScriptWatches::_tracking_changed);
	ClassDB::bind_method(D_METHOD("_watch_removed", "watch"), &ScriptWatches::_watch_removed);
	ClassDB::bind_method(D_METHOD("_breaked", "p_breaked", "p_can_debug"), &ScriptWatches::_breaked);
	ClassDB::bind_method(D_METHOD("_lost_focus", "p_watch"), &ScriptWatches::_lost_focus);

	ADD_SIGNAL(MethodInfo("watch_added"));
	ADD_SIGNAL(MethodInfo("expression_updated", PropertyInfo(Variant::INT, "watch_index"), PropertyInfo(Variant::STRING, "new_expression")));
	ADD_SIGNAL(MethodInfo("lock_updated", PropertyInfo(Variant::INT, "watch_index"), PropertyInfo(Variant::BOOL, "locked")));
	ADD_SIGNAL(MethodInfo("tracking_updated", PropertyInfo(Variant::INT, "watch_index"), PropertyInfo(Variant::BOOL, "tracking")));
	ADD_SIGNAL(MethodInfo("watch_removed", PropertyInfo(Variant::INT, "watch_index")));
}

ScriptWatch *ScriptWatches::_get_watch_ghost() const {
	ERR_EXPLAIN("No watch ghost found!");
	ERR_FAIL_COND_V(watches.size() == 0, NULL);
	return watches[watches.size() - 1];
}

void ScriptWatches::_create_watch_ghost() {
	ScriptWatch *watch = memnew(ScriptWatch(watches.size() + 1));
	watch->connect("watch_removed", this, "_watch_removed");
	watch->connect("expression_changed", this, "_watch_changed");
	watch->connect("lock_changed", this, "_lock_changed");
	watch->connect("tracking_changed", this, "_tracking_changed");
	watch->connect("focus_lost", this, "_lost_focus");
	main_vbox->add_child(watch);
	watches.push_back(watch);

	watch->lock_toggle->set_disabled(true);
	watch->track_toggle->set_disabled(true);
}

void ScriptWatches::_remove_watch(ScriptWatch *p_watch) {
	int id = p_watch->id;

	watches.remove(id - 1);
	memdelete(p_watch);
	_update_watch_ids(id - 1);
}

void ScriptWatches::_update_watch_ids(int p_from_index) {
	const int size = watches.size();

	for (int i = p_from_index; i < size; i++) {
		watches[i]->set_id(i + 1);
	}
}

void ScriptWatches::add_watch(const String &p_expression) {
	const int index = get_watch_count();

	_create_watch_ghost();
	watches[index]->lock_toggle->set_disabled(false);
	watches[index]->track_toggle->set_disabled(false);
	watches[index]->expression_line->set_text(p_expression);
}

void ScriptWatches::update_watch_result(int p_index, const String &p_result, bool p_success) {
	ERR_FAIL_INDEX(p_index, watches.size());
	ScriptWatch *watch = watches[p_index];

	watch->set_result(p_result, p_success);
}

void ScriptWatches::remove_watch(int p_index) {
	ERR_FAIL_INDEX(p_index, watches.size());
	ScriptWatch *watch = watches[p_index];

	_remove_watch(watch);
}

int ScriptWatches::get_watch_count() const {
	// Minus one because there's always one extra "ghost" watch
	return watches.size() - 1;
}

String ScriptWatches::get_watch_expression(int p_index) {
	ERR_FAIL_INDEX_V(p_index, watches.size(), String());

	return watches[p_index]->expression_line->get_text();
}

void ScriptWatches::disable(bool p_disable) {
	for (int i = 0; i < get_watch_count(); i++) {
		watches[i]->disable(p_disable);
	}

	ScriptWatch *const w = _get_watch_ghost();

	Control::FocusMode focus;
	if (p_disable) {
		focus = FOCUS_NONE;
	} else {
		focus = FOCUS_ALL;
	}
	w->expression_line->set_editable(!p_disable);
	w->expression_line->set_focus_mode(focus);
}

ScriptWatches::ScriptWatches(ScriptEditorDebugger *p_debugger) {
	main_vbox = memnew(VBoxContainer);
	main_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	_create_watch_ghost();
	add_child(main_vbox);

	p_debugger->call_deferred("connect", "breaked", this, "_breaked");
}
