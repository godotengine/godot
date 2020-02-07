/*************************************************************************/
/*  debugger_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "debugger_editor_plugin.h"

#include "core/os/keyboard.h"
#include "editor/debugger/editor_debugger_node.h"

DebuggerEditorPlugin::DebuggerEditorPlugin(EditorNode *p_editor) {
	ED_SHORTCUT("debugger/step_into", TTR("Step Into"), KEY_F11);
	ED_SHORTCUT("debugger/step_over", TTR("Step Over"), KEY_F10);
	ED_SHORTCUT("debugger/break", TTR("Break"));
	ED_SHORTCUT("debugger/continue", TTR("Continue"), KEY_F12);
	ED_SHORTCUT("debugger/keep_debugger_open", TTR("Keep Debugger Open"));
	ED_SHORTCUT("debugger/debug_with_external_editor", TTR("Debug with External Editor"));

	EditorDebuggerNode *debugger = memnew(EditorDebuggerNode);
	Button *db = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Debugger"), debugger);
	debugger->set_tool_button(db);
}

DebuggerEditorPlugin::~DebuggerEditorPlugin() {
	// Should delete debugger?
}
