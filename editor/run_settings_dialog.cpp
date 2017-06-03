/*************************************************************************/
/*  run_settings_dialog.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "run_settings_dialog.h"

void RunSettingsDialog::popup_run_settings() {

	popup_centered(Size2(300, 150));
}

void RunSettingsDialog::set_custom_arguments(const String &p_arguments) {

	arguments->set_text(p_arguments);
}
String RunSettingsDialog::get_custom_arguments() const {

	return arguments->get_text();
}

void RunSettingsDialog::_bind_methods() {

	ClassDB::bind_method("_run_mode_changed", &RunSettingsDialog::_run_mode_changed);
	//ClassDB::bind_method("_browse_selected_file",&RunSettingsDialog::_browse_selected_file);
}

void RunSettingsDialog::_run_mode_changed(int idx) {

	if (idx == 0)
		arguments->set_editable(false);
	else
		arguments->set_editable(true);
}

int RunSettingsDialog::get_run_mode() const {

	return run_mode->get_selected();
}

void RunSettingsDialog::set_run_mode(int p_run_mode) {

	run_mode->select(p_run_mode);
	arguments->set_editable(p_run_mode);
}

RunSettingsDialog::RunSettingsDialog() {

	/* SNAP DIALOG */

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	//set_child_rect(vbc);

	run_mode = memnew(OptionButton);
	vbc->add_margin_child(TTR("Run Mode:"), run_mode);
	run_mode->add_item(TTR("Current Scene"));
	run_mode->add_item(TTR("Main Scene"));
	run_mode->connect("item_selected", this, "_run_mode_changed");
	arguments = memnew(LineEdit);
	vbc->add_margin_child(TTR("Main Scene Arguments:"), arguments);
	arguments->set_editable(false);

	get_ok()->set_text(TTR("Close"));
	//get_cancel()->set_text("Close");

	arguments->set_text("-l $scene");

	set_title(TTR("Scene Run Settings"));
}
