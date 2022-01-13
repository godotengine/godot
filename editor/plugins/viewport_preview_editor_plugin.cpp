/*************************************************************************/
/*  viewport_preview_editor_plugin.cpp                                   */
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

#include "viewport_preview_editor_plugin.h"

bool EditorInspectorPluginViewportPreview::can_handle(Object *p_object) {
	return Object::cast_to<Viewport>(p_object) != nullptr;
}

void EditorInspectorPluginViewportPreview::parse_begin(Object *p_object) {
	Viewport *viewport = Object::cast_to<Viewport>(p_object);

	TexturePreview *viewport_preview = memnew(TexturePreview(viewport->get_texture(), false));
	viewport_preview->get_texture_display()->set_flip_v(true); // flip as ViewportTexture in 3.x is upside-down.

	// Otherwise `viewport_preview`'s `texture_display` doesn't update properly when `viewport`'s size changes.
	viewport->connect("size_changed", viewport_preview->get_texture_display(), "update");
	add_custom_control(viewport_preview);
}

ViewportPreviewEditorPlugin::ViewportPreviewEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginViewportPreview> plugin;
	plugin.instance();
	add_inspector_plugin(plugin);
}
