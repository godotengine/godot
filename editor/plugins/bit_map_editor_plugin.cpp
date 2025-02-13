/**************************************************************************/
/*  bit_map_editor_plugin.cpp                                             */
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

#include "bit_map_editor_plugin.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/image_texture.h"

void BitMapEditor::setup(const Ref<BitMap> &p_bitmap) {
	texture_rect->set_texture(ImageTexture::create_from_image(p_bitmap->convert_to_image()));
	size_label->set_text(vformat(U"%s×%s", p_bitmap->get_size().width, p_bitmap->get_size().height));
}

BitMapEditor::BitMapEditor() {
	texture_rect = memnew(TextureRect);
	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	texture_rect->set_texture_filter(TEXTURE_FILTER_NEAREST);
	texture_rect->set_custom_minimum_size(Size2(0, 250) * EDSCALE);
	add_child(texture_rect);

	size_label = memnew(Label);
	size_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
	add_child(size_label);

	// Reduce extra padding on top and bottom of size label.
	Ref<StyleBoxEmpty> stylebox;
	stylebox.instantiate();
	stylebox->set_content_margin(SIDE_RIGHT, 4 * EDSCALE);
	size_label->add_theme_style_override(CoreStringName(normal), stylebox);
}

///////////////////////

bool EditorInspectorPluginBitMap::can_handle(Object *p_object) {
	return Object::cast_to<BitMap>(p_object) != nullptr;
}

void EditorInspectorPluginBitMap::parse_begin(Object *p_object) {
	BitMap *bitmap = Object::cast_to<BitMap>(p_object);
	if (!bitmap) {
		return;
	}
	Ref<BitMap> bm(bitmap);

	BitMapEditor *editor = memnew(BitMapEditor);
	editor->setup(bm);
	add_custom_control(editor);
}

///////////////////////

BitMapEditorPlugin::BitMapEditorPlugin() {
	Ref<EditorInspectorPluginBitMap> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
