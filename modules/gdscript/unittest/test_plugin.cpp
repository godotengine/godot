/*************************************************************************/
/*  test_plugin.cpp                                                      */
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

#include "test_plugin.h"

#include "editor/editor_node.h"

DebugButton::DebugButton() : m_status(STATUS_STOP) {
	set_text(TTR("Test"));
}

 void DebugButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect("pressed", this, "_run");
			break;
		}
	}
}

Error DebugButton::_run() {
	List<String> args;

	String resource_path = ProjectSettings::get_singleton()->get_resource_path();
	String remote_host = EditorSettings::get_singleton()->get("network/debug/remote_host");
	int remote_port = (int)EditorSettings::get_singleton()->get("network/debug/remote_port");

	if (resource_path != "") {
		args.push_back("--path");
		args.push_back(resource_path.replace(" ", "%20"));
	}

	args.push_back("--remote-debug");
	args.push_back(remote_host + ":" + String::num(remote_port));

	args.push_back("--allow_focus_steal_pid");
	args.push_back(itos(OS::get_singleton()->get_process_id()));

	if (OS::get_singleton()->is_disable_crash_handler()) {
		args.push_back("--disable-crash-handler");
	}

	args.push_back("--main-loop-type");
	args.push_back("TestRunner");

	String exec = OS::get_singleton()->get_executable_path();

	pid = 0;
	Error err = OS::get_singleton()->execute(exec, args, false, &pid);
	ERR_FAIL_COND_V(err, err);

	m_status = STATUS_PLAY;

	return OK;
}

void DebugButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_run"), &DebugButton::_run);
}

TestToolbar::TestToolbar() {
}

void TestToolbar::_bind_methods() {
}

CoverageToolbar::CoverageToolbar() {
}

void CoverageToolbar::_bind_methods() {
}

DocumentationToolbar::DocumentationToolbar() {
}

void DocumentationToolbar::_bind_methods() {
}

TestPlugin::TestPlugin() {
	m_debug_button = memnew(DebugButton);
	add_control_to_container(CONTAINER_TOOLBAR, m_debug_button);
	m_test_toolbar = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Test"), memnew(TestToolbar));
	m_coverage_toolbar = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Coverage"), memnew(TestToolbar));
	m_documentation_toolbar = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Documentation"), memnew(TestToolbar));
}

TestPlugin::~TestPlugin() {
}

void TestPlugin::_bind_methods() {
}
