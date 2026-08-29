/**************************************************************************/
/*  vertex_mobile_editor_plugin.h                                        */
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

#ifndef VERTEX_MOBILE_EDITOR_PLUGIN_H
#define VERTEX_MOBILE_EDITOR_PLUGIN_H

#ifdef TOOLS_ENABLED

#include "editor/plugins/editor_plugin.h"
#include "modules/vertex_mobile_editor/vertex_mobile_settings.h"

class VertexMobileEditorPanel;

// Registers the Vertex mobile editor layout helper: a touch-friendly bottom
// panel exposing the responsive layout settings (large touch targets, compact
// toolbar, collapsible panels, pinch-zoom range, gesture thresholds) so the
// editor can be used comfortably on Android phones and small screens.
class VertexMobileEditorPlugin : public EditorPlugin {
	GDCLASS(VertexMobileEditorPlugin, EditorPlugin);

	VertexMobileEditorPanel *panel = nullptr;

protected:
	static void _bind_methods() {}

public:
	String get_plugin_name() const override;
	void make_visible(bool p_visible) override;
	bool has_main_screen() const override { return false; }

	VertexMobileEditorPlugin();
	~VertexMobileEditorPlugin();
};

#endif // TOOLS_ENABLED
#endif // VERTEX_MOBILE_EDITOR_PLUGIN_H
