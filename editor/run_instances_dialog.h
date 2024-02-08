/**************************************************************************/
/*  run_instances_dialog.h                                                */
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

#ifndef RUN_INSTANCES_DIALOG_H
#define RUN_INSTANCES_DIALOG_H

#include "scene/gui/dialogs.h"

class CheckBox;
class LineEdit;
class SpinBox;
class Timer;
class VBoxContainer;

class RunInstancesDialog : public AcceptDialog {
	GDCLASS(RunInstancesDialog, AcceptDialog);

private:
	static RunInstancesDialog *singleton;

	Timer *main_apply_timer = nullptr;
	Timer *instance_apply_timer = nullptr;

	LineEdit *main_args_edit = nullptr;
	SpinBox *instance_count = nullptr;
	CheckBox *enable_multiple_instances_checkbox = nullptr;
	VBoxContainer *argument_container = nullptr;

	Array override_list;
	PackedStringArray argument_list;

	void _fetch_main_args();
	// These 2 methods are necessary due to callable_mp() not supporting default arguments.
	void _start_main_timer();
	void _start_instance_timer();

	void _refresh_argument_count();
	void _save_main_args();
	void _save_arguments();
	// Separates command line arguments without splitting up quoted strings.
	Vector<String> _split_cmdline_args(const String &p_arg_string) const;

public:
	int get_instance_count() const;
	void get_argument_list_for_instance(int p_idx, List<String> &r_list) const;

	static RunInstancesDialog *get_singleton() { return singleton; }
	RunInstancesDialog();
};

#endif // RUN_INSTANCES_DIALOG_H
