/**************************************************************************/
/*  sub_viewport_preview_editor_plugin.h                                  */
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

#ifndef SUB_VIEWPORT_PREVIEW_EDITOR_PLUGIN_H
#define SUB_VIEWPORT_PREVIEW_EDITOR_PLUGIN_H

#include "editor/plugins/editor_plugin.h"
#include "editor/plugins/texture_editor_plugin.h"
#include "scene/main/viewport.h"

class EditorInspectorPluginSubViewportPreview : public EditorInspectorPluginTexture {
	GDCLASS(EditorInspectorPluginSubViewportPreview, EditorInspectorPluginTexture);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class SubViewportPreviewEditorPlugin : public EditorPlugin {
	GDCLASS(SubViewportPreviewEditorPlugin, EditorPlugin);

public:
	virtual String get_name() const override { return "SubViewportPreview"; }

	SubViewportPreviewEditorPlugin();
};

#endif // SUB_VIEWPORT_PREVIEW_EDITOR_PLUGIN_H
