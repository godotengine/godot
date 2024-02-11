/**************************************************************************/
/*  run_instances_dialog.cpp                                              */
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

#include "run_instances_dialog.h"

#include "core/config/project_settings.h"
#include "editor/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/spin_box.h"
#include "scene/main/timer.h"

RunInstancesDialog *RunInstancesDialog::singleton = nullptr;

void RunInstancesDialog::_fetch_main_args() {
	if (!main_args_edit->has_focus()) { // Only set the text if the user is not currently editing it.
		main_args_edit->set_text(GLOBAL_GET("editor/run/main_run_args"));
	}
}

void RunInstancesDialog::_start_main_timer() {
	main_apply_timer->start();
}

void RunInstancesDialog::_start_instance_timer() {
	instance_apply_timer->start();
}

void RunInstancesDialog::_refresh_argument_count() {
	while (argument_container->get_child_count() > 0) {
		memdelete(argument_container->get_child(0));
	}

	override_list.resize(instance_count->get_value());
	argument_list.resize_zeroed(instance_count->get_value());

	for (int i = 0; i < argument_list.size(); i++) {
		VBoxContainer *instance_vb = memnew(VBoxContainer);
		argument_container->add_child(instance_vb);

		HBoxContainer *hbox = memnew(HBoxContainer);
		instance_vb->add_child(hbox);

		Label *l = memnew(Label);
		hbox->add_child(l);
		l->set_text(vformat(TTR("Instance %d"), i + 1));

		CheckBox *cb = memnew(CheckBox);
		hbox->add_child(cb);
		cb->set_text(TTR("Override Main Run Args"));
		cb->set_tooltip_text(TTR("If disabled, the instance arguments will be appended after the Main Run Args."));
		cb->set_pressed(override_list[i]);
		cb->set_h_size_flags(Control::SIZE_SHRINK_END | Control::SIZE_EXPAND);
		cb->connect(SNAME("toggled"), callable_mp(this, &RunInstancesDialog::_start_instance_timer).unbind(1));
		instance_vb->set_meta(SNAME("override"), cb);

		LineEdit *le = memnew(LineEdit);
		instance_vb->add_child(le);
		le->set_text(argument_list[i]);
		le->connect(SNAME("text_changed"), callable_mp(this, &RunInstancesDialog::_start_instance_timer).unbind(1));
		instance_vb->set_meta(SNAME("args"), le);
	}
}

void RunInstancesDialog::_save_main_args() {
	ProjectSettings::get_singleton()->set_setting("editor/run/main_run_args", main_args_edit->get_text());
	ProjectSettings::get_singleton()->save();
}

void RunInstancesDialog::_save_arguments() {
	override_list.clear();
	override_list.resize(argument_container->get_child_count());
	argument_list.clear();
	argument_list.resize(argument_container->get_child_count());

	String *w = argument_list.ptrw();
	for (int i = 0; i < argument_container->get_child_count(); i++) {
		const Node *instance_vb = argument_container->get_child(i);

		CheckBox *check_box = Object::cast_to<CheckBox>(instance_vb->get_meta(SNAME("override")));
		ERR_FAIL_NULL(check_box);
		override_list[i] = check_box->is_pressed();

		LineEdit *edit = Object::cast_to<LineEdit>(instance_vb->get_meta(SNAME("args")));
		ERR_FAIL_NULL(edit);
		w[i] = edit->get_text();
	}

	EditorSettings::get_singleton()->set_project_metadata("debug_options", "multiple_instances_enabled", enable_multiple_instances_checkbox->is_pressed());
	EditorSettings::get_singleton()->set_project_metadata("debug_options", "multiple_instances_overrides", override_list);
	EditorSettings::get_singleton()->set_project_metadata("debug_options", "multiple_instances_arguments", argument_list);
}

Vector<String> RunInstancesDialog::_split_cmdline_args(const String &p_arg_string) const {
	Vector<String> split_args;
	int arg_start = 0;
	bool is_quoted = false;
	char32_t quote_char = '-';
	char32_t arg_char;
	int arg_length;
	for (int i = 0; i < p_arg_string.length(); i++) {
		arg_char = p_arg_string[i];
		if (arg_char == '\"' || arg_char == '\'') {
			if (i == 0 || p_arg_string[i - 1] != '\\') {
				if (is_quoted) {
					if (arg_char == quote_char) {
						is_quoted = false;
						quote_char = '-';
					}
				} else {
					is_quoted = true;
					quote_char = arg_char;
				}
			}
		} else if (!is_quoted && arg_char == ' ') {
			arg_length = i - arg_start;
			if (arg_length > 0) {
				split_args.push_back(p_arg_string.substr(arg_start, arg_length));
			}
			arg_start = i + 1;
		}
	}
	arg_length = p_arg_string.length() - arg_start;
	if (arg_length > 0) {
		split_args.push_back(p_arg_string.substr(arg_start, arg_length));
	}
	return split_args;
}

int RunInstancesDialog::get_instance_count() const {
	if (enable_multiple_instances_checkbox->is_pressed()) {
		return instance_count->get_value();
	} else {
		return 1;
	}
}

void RunInstancesDialog::get_argument_list_for_instance(int p_idx, List<String> &r_list) const {
	bool override_args = override_list[p_idx];
	bool use_multiple_instances = enable_multiple_instances_checkbox->is_pressed();
	String raw_custom_args;

	if (use_multiple_instances && override_args) {
		raw_custom_args = argument_list[p_idx];
	} else {
		raw_custom_args = main_args_edit->get_text();
	}

	String exec = OS::get_singleton()->get_executable_path();

	if (!raw_custom_args.is_empty()) {
		// Allow the user to specify a command to run, similar to Steam's launch options.
		// In this case, Godot will no longer be run directly; it's up to the underlying command
		// to run it. For instance, this can be used on Linux to force a running project
		// to use Optimus using `prime-run` or similar.
		// Example: `prime-run %command% --time-scale 0.5`
		const int placeholder_pos = raw_custom_args.find("%command%");

		Vector<String> custom_args;

		if (placeholder_pos != -1) {
			// Prepend executable-specific custom arguments.
			// If nothing is placed before `%command%`, behave as if no placeholder was specified.
			Vector<String> exec_args = _split_cmdline_args(raw_custom_args.substr(0, placeholder_pos));
			if (exec_args.size() > 0) {
				exec = exec_args[0];
				exec_args.remove_at(0);

				// Append the Godot executable name before we append executable arguments
				// (since the order is reversed when using `push_front()`).
				r_list.push_front(OS::get_singleton()->get_executable_path());
			}

			for (int i = exec_args.size() - 1; i >= 0; i--) {
				// Iterate backwards as we're pushing items in the reverse order.
				r_list.push_front(exec_args[i].replace(" ", "%20"));
			}

			// Append Godot-specific custom arguments.
			custom_args = _split_cmdline_args(raw_custom_args.substr(placeholder_pos + String("%command%").size()));
			for (int i = 0; i < custom_args.size(); i++) {
				r_list.push_back(custom_args[i].replace(" ", "%20"));
			}
		} else {
			// Append Godot-specific custom arguments.
			custom_args = _split_cmdline_args(raw_custom_args);
			for (int i = 0; i < custom_args.size(); i++) {
				r_list.push_back(custom_args[i].replace(" ", "%20"));
			}
		}
	}

	if (use_multiple_instances && !override_args) {
		r_list.push_back(argument_list[p_idx]);
	}
}

RunInstancesDialog::RunInstancesDialog() {
	singleton = this;
	set_min_size(Vector2i(0, 600 * EDSCALE));

	main_apply_timer = memnew(Timer);
	main_apply_timer->set_wait_time(0.5);
	main_apply_timer->set_one_shot(true);
	add_child(main_apply_timer);
	main_apply_timer->connect("timeout", callable_mp(this, &RunInstancesDialog::_save_main_args));

	instance_apply_timer = memnew(Timer);
	instance_apply_timer->set_wait_time(0.5);
	instance_apply_timer->set_one_shot(true);
	add_child(instance_apply_timer);
	instance_apply_timer->connect("timeout", callable_mp(this, &RunInstancesDialog::_save_arguments));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	{
		Label *l = memnew(Label);
		main_vb->add_child(l);
		l->set_text(TTR("Main Run Args:"));
	}

	main_args_edit = memnew(LineEdit);
	main_vb->add_child(main_args_edit);
	_fetch_main_args();
	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &RunInstancesDialog::_fetch_main_args));
	main_args_edit->connect("text_changed", callable_mp(this, &RunInstancesDialog::_start_main_timer).unbind(1));

	enable_multiple_instances_checkbox = memnew(CheckBox);
	enable_multiple_instances_checkbox->set_text(TTR("Enable Multiple Instances"));
	enable_multiple_instances_checkbox->set_pressed(EditorSettings::get_singleton()->get_project_metadata("debug_options", "multiple_instances_enabled", false));
	main_vb->add_child(enable_multiple_instances_checkbox);
	enable_multiple_instances_checkbox->connect("pressed", callable_mp(this, &RunInstancesDialog::_start_instance_timer));

	override_list = EditorSettings::get_singleton()->get_project_metadata("debug_options", "multiple_instances_overrides", varray(false, false, false, false));
	argument_list = EditorSettings::get_singleton()->get_project_metadata("debug_options", "multiple_instances_arguments", PackedStringArray{ "", "", "", "" });

	instance_count = memnew(SpinBox);
	instance_count->set_min(1);
	instance_count->set_max(20);
	instance_count->set_value(argument_list.size());
	main_vb->add_child(instance_count);
	instance_count->connect("value_changed", callable_mp(this, &RunInstancesDialog::_start_instance_timer).unbind(1));
	instance_count->connect("value_changed", callable_mp(this, &RunInstancesDialog::_refresh_argument_count).unbind(1));
	enable_multiple_instances_checkbox->connect("toggled", callable_mp(instance_count, &SpinBox::set_editable));

	{
		Label *l = memnew(Label);
		l->set_text(TTR("Launch Arguments"));
		l->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		main_vb->add_child(l);
	}

	{
		ScrollContainer *arguments_scroll = memnew(ScrollContainer);
		arguments_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		arguments_scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		arguments_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		main_vb->add_child(arguments_scroll);

		argument_container = memnew(VBoxContainer);
		argument_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		arguments_scroll->add_child(argument_container);
	}

	_refresh_argument_count();

	set_title(TTR("Run Multiple Instances"));
	set_min_size(Size2i(400 * EDSCALE, 0));
}
