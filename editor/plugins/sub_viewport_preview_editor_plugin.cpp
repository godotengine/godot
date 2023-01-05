/**************************************************************************/
/*  sub_viewport_preview_editor_plugin.cpp                                */
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

#include "sub_viewport_preview_editor_plugin.h"

bool EditorInspectorPluginSubViewportPreview::can_handle(Object *p_object) {
	return Object::cast_to<SubViewport>(p_object) != nullptr;
}

void EditorInspectorPluginSubViewportPreview::parse_begin(Object *p_object) {
	SubViewport *sub_viewport = Object::cast_to<SubViewport>(p_object);

	TexturePreview *sub_viewport_preview = memnew(TexturePreview(sub_viewport->get_texture(), false));
	// Otherwise `sub_viewport_preview`'s `texture_display` doesn't update properly when `sub_viewport`'s size changes.
	sub_viewport->connect("size_changed", callable_mp((CanvasItem *)sub_viewport_preview->get_texture_display(), &CanvasItem::queue_redraw));
	add_custom_control(sub_viewport_preview);
}

SubViewportPreviewEditorPlugin::SubViewportPreviewEditorPlugin() {
	Ref<EditorInspectorPluginSubViewportPreview> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
