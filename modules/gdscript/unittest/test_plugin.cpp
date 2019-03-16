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

#include "editor/editor_data.h"
#include "editor/editor_node.h"

DebugButton::DebugButton() {
	set_text(TTR("Test"));
	m_debugger = memnew(ScriptEditorDebugger(EditorNode::get_singleton()));
}

DebugButton ::~DebugButton() {
	stop();
}

EditorRun::Status DebugButton::get_status() const {
	return m_editor_run.get_status();
}

Error DebugButton::run() {
	m_debugger->start();
	List<String> breakpoints;
	EditorNode::get_singleton()->get_editor_data().get_editor_breakpoints(&breakpoints);
	return m_editor_run.run("", "--main-loop-type TestRunner", breakpoints);
}

void DebugButton::stop() {
	m_editor_run.stop();
	m_debugger->stop();
}

void DebugButton::set_debug_collisions(bool p_debug) {
	m_editor_run.set_debug_collisions(p_debug);
}

bool DebugButton::get_debug_collisions() const {
	return m_editor_run.get_debug_collisions();
}

void DebugButton::set_debug_navigation(bool p_debug) {
	m_editor_run.set_debug_navigation(p_debug);
}

bool DebugButton::get_debug_navigation() const {
	return m_editor_run.get_debug_navigation();
}

void DebugButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect("pressed", this, "run");
			break;
		}
	}
}

void DebugButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run"), &DebugButton::run);
}

TestPanel::TestPanel() {
}

void TestPanel::_bind_methods() {
}

CoveragePanel::CoveragePanel() {
}

void CoveragePanel::_bind_methods() {
}

DocumentationPanel::DocumentationPanel() {
}

void DocumentationPanel::_bind_methods() {
}

TestPlugin::TestPlugin() {
	m_debug_button = memnew(DebugButton);
	add_control_to_container(CONTAINER_TOOLBAR, m_debug_button);
	m_test_panel = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Test"), memnew(TestPanel));
	m_coverage_panel = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Coverage"), memnew(CoveragePanel));
	m_documentation_panel = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Documentation"), memnew(DocumentationPanel));
}

TestPlugin::~TestPlugin() {
}

void TestPlugin::_bind_methods() {
}
