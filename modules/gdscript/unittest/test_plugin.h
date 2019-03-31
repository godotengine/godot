/*************************************************************************/
/*  test_plugin.h                                                        */
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

#ifndef TEST_PLUGIN_H
#define TEST_PLUGIN_H

#include "test_log.h"

#include "editor/editor_plugin.h"
#include "editor/editor_run.h"
#include "editor/script_editor_debugger.h"
#include "scene/gui/button.h"

class DebugButton : public Button {
	GDCLASS(DebugButton, Button);

public:
	DebugButton();
	void run();
	void enable();
	void disable();

protected:
	static void _bind_methods();
	virtual void _notification(int p_what);

private:
	EditorRun m_editor_run;
	bool m_is_testing;

	void _display_results();
};

class TestPanel : public HBoxContainer {
	GDCLASS(TestPanel, HBoxContainer);

public:
	typedef Ref<TestLog> RefTestLog;
	typedef HashMap<String, RefTestLog> TestFuncLogMap;
	typedef HashMap<String, TestFuncLogMap> TestFileFuncLogMap;

	TestPanel();

protected:
	static void _bind_methods();

private:
	Tree *m_tree;

	TestFileFuncLogMap m_results;
	void _item_selected();

	void _display_results();
};

class TestPlugin : public EditorPlugin {
	GDCLASS(TestPlugin, EditorPlugin);

public:
	TestPlugin();
	virtual ~TestPlugin();

protected:
	static void _bind_methods();
	void _display_test_panel();

private:
	DebugButton *m_debug_button;
	TestPanel *m_test_panel;
	ToolButton *m_test_panel_button;
};

#endif // TEST_PLUGIN_H
