/**************************************************************************/
/*  javascript_tools_editor_plugin.h                                      */
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

#ifndef JAVASCRIPT_TOOLS_EDITOR_PLUGIN_H
#define JAVASCRIPT_TOOLS_EDITOR_PLUGIN_H

#if defined(TOOLS_ENABLED) && defined(JAVASCRIPT_ENABLED)
#include "core/io/zip_io.h"
#include "editor/editor_plugin.h"

class JavaScriptToolsEditorPlugin : public EditorPlugin {
	GDCLASS(JavaScriptToolsEditorPlugin, EditorPlugin);

private:
	void _zip_file(String p_path, String p_base_path, zipFile p_zip);
	void _zip_recursive(String p_path, String p_base_path, zipFile p_zip);

protected:
	static void _bind_methods();

	void _download_zip(Variant p_v);

public:
	static void initialize();

	JavaScriptToolsEditorPlugin(EditorNode *p_editor);
};
#else
class JavaScriptToolsEditorPlugin {
public:
	static void initialize() {}
};
#endif

#endif // JAVASCRIPT_TOOLS_EDITOR_PLUGIN_H
