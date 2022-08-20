/*************************************************************************/
/*  editor_debugger_plugin.h                                             */
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

#ifndef EDITOR_DEBUGGER_PLUGIN_H
#define EDITOR_DEBUGGER_PLUGIN_H

#include "scene/gui/control.h"

class ScriptEditorDebugger;

class EditorDebuggerPlugin : public Control {
	GDCLASS(EditorDebuggerPlugin, Control);

private:
	ScriptEditorDebugger *debugger = nullptr;

	void _breaked(bool p_really_did, bool p_can_debug, String p_message, bool p_has_stackdump);
	void _started();
	void _stopped();

protected:
	static void _bind_methods();

public:
	void attach_debugger(ScriptEditorDebugger *p_debugger);
	void detach_debugger(bool p_call_debugger);
	void send_message(const String &p_message, const Array &p_args);
	void register_message_capture(const StringName &p_name, const Callable &p_callable);
	void unregister_message_capture(const StringName &p_name);
	bool has_capture(const StringName &p_name);
	bool is_breaked();
	bool is_debuggable();
	bool is_session_active();
	~EditorDebuggerPlugin();
};

#endif // EDITOR_DEBUGGER_PLUGIN_H
