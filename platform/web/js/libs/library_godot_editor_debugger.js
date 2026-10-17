/**************************************************************************/
/*  library_godot_editor_debugger.js                                      */
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

const GodotEditorDebugger = {
	$GodotEditorDebugger__postset: [
		'Module["add_debugger_session"] = GodotEditorDebugger.addSession;',
	].join(''),
	$GodotEditorDebugger__deps: ['$GodotDebugger', '$GodotRuntime', '$IDHandler'],
	$GodotEditorDebugger: {
		session_callback: null,
		addSession: function (port) {
			if (!GodotEditorDebugger.session_callback) {
				GodotRuntime.error('Editor debugger not bound.');
				return;
			}
			const id = IDHandler.add({
				port: port,
				active: true,
			});
			GodotEditorDebugger.session_callback(id);
		},
	},

	godot_js_editor_debugger_cb__proxy: 'sync',
	godot_js_editor_debugger_cb__sig: 'vp',
	godot_js_editor_debugger_cb: function (p_cb) {
		GodotEditorDebugger.session_callback = p_cb ? GodotRuntime.get_func(p_cb) : null;
	},

	godot_js_editor_debugger_active__proxy: 'sync',
	godot_js_editor_debugger_active__sig: 'i',
	godot_js_editor_debugger_active: function () {
		return GodotEditorDebugger.session_callback ? 1 : 0;
	},
};

autoAddDeps(GodotEditorDebugger, '$GodotEditorDebugger');
mergeInto(LibraryManager.library, GodotEditorDebugger);
