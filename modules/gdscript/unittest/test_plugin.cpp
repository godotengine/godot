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

DebugButton::DebugButton() {
	set_text(TTR("Test"));
}

void DebugButton::_bind_methods() {
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
