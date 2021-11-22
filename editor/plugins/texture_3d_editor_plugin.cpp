/*************************************************************************/
/*  texture_3d_editor_plugin.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "texture_3d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "editor/editor_settings.h"

void Texture3DEditor::_texture_rect_draw() {
	texture_rect->draw_rect(Rect2(Point2(), texture_rect->get_size()), Color(1, 1, 1, 1));
}

void Texture3DEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		//get_scene()->connect("node_removed",this,"_node_removed");
	}
	if (p_what == NOTIFICATION_RESIZED) {
		_texture_rect_update_area();
	}

	if (p_what == NOTIFICATION_DRAW) {
		Ref<Texture2D> checkerboard = get_theme_icon(SNAME("Checkerboard"), SNAME("EditorIcons"));
		Size2 size = get_size();

		draw_texture_rect(checkerboard, Rect2(Point2(), size), true);
	}
}

void Texture3DEditor::_texture_changed() {
	if (!is_visible()) {
		return;
	}
	update();
}

void Texture3DEditor::_update_material() {
	material->set_shader_param("layer", (layer->get_value() + 0.5) / texture->get_depth());
	material->set_shader_param("tex", texture->get_rid());

	String format = Image::get_format_name(texture->get_format());

	String text;
	text = itos(texture->get_width()) + "x" + itos(texture->get_height()) + "x" + itos(texture->get_depth()) + " " + format;

	info->set_text(text);
}

void Texture3DEditor::_make_shaders() {
	shader.instantiate();
	shader->set_code(R"(
// Texture3DEditor preview shader.

shader_type canvas_item;

uniform sampler3D tex;
uniform float layer;

void fragment() {
	COLOR = textureLod(tex, vec3(UV, layer), 0.0);
}
)");
	material.instantiate();
	material->set_shader(shader);
}

void Texture3DEditor::_texture_rect_update_area() {
	Size2 size = get_size();
	int tex_width = texture->get_width() * size.height / texture->get_height();
	int tex_height = size.height;

	if (tex_width > size.width) {
		tex_width = size.width;
		tex_height = texture->get_height() * tex_width / texture->get_width();
	}

	// Prevent the texture from being unpreviewable after the rescale, so that we can still see something
	if (tex_height <= 0) {
		tex_height = 1;
	}
	if (tex_width <= 0) {
		tex_width = 1;
	}

	int ofs_x = (size.width - tex_width) / 2;
	int ofs_y = (size.height - tex_height) / 2;

	texture_rect->set_position(Vector2(ofs_x, ofs_y));
	texture_rect->set_size(Vector2(tex_width, tex_height));
}

void Texture3DEditor::edit(Ref<Texture3D> p_texture) {
	if (!texture.is_null()) {
		texture->disconnect("changed", callable_mp(this, &Texture3DEditor::_texture_changed));
	}

	texture = p_texture;

	if (!texture.is_null()) {
		if (shader.is_null()) {
			_make_shaders();
		}

		texture->connect("changed", callable_mp(this, &Texture3DEditor::_texture_changed));
		update();
		texture_rect->set_material(material);
		setting = true;
		layer->set_max(texture->get_depth() - 1);
		layer->set_value(0);
		layer->show();
		_update_material();
		setting = false;
		_texture_rect_update_area();
	} else {
		hide();
	}
}

void Texture3DEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_layer_changed"), &Texture3DEditor::_layer_changed);
}

Texture3DEditor::Texture3DEditor() {
	set_texture_repeat(TextureRepeat::TEXTURE_REPEAT_ENABLED);
	set_custom_minimum_size(Size2(1, 150));
	texture_rect = memnew(Control);
	texture_rect->connect("draw", callable_mp(this, &Texture3DEditor::_texture_rect_draw));
	texture_rect->set_mouse_filter(MOUSE_FILTER_IGNORE);
	add_child(texture_rect);

	layer = memnew(SpinBox);
	layer->set_step(1);
	layer->set_max(100);
	add_child(layer);
	layer->set_anchor(SIDE_RIGHT, 1);
	layer->set_anchor(SIDE_LEFT, 1);
	layer->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	layer->set_modulate(Color(1, 1, 1, 0.8));
	info = memnew(Label);
	add_child(info);
	info->set_anchor(SIDE_RIGHT, 1);
	info->set_anchor(SIDE_LEFT, 1);
	info->set_anchor(SIDE_BOTTOM, 1);
	info->set_anchor(SIDE_TOP, 1);
	info->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	info->set_v_grow_direction(GROW_DIRECTION_BEGIN);
	info->add_theme_color_override("font_color", Color(1, 1, 1, 1));
	info->add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.5));
	info->add_theme_constant_override("shadow_outline_size", 1);
	info->add_theme_constant_override("shadow_offset_x", 2);
	info->add_theme_constant_override("shadow_offset_y", 2);

	setting = false;
	layer->connect("value_changed", Callable(this, "_layer_changed"));
}

Texture3DEditor::~Texture3DEditor() {
	if (!texture.is_null()) {
		texture->disconnect("changed", callable_mp(this, &Texture3DEditor::_texture_changed));
	}
}

//
bool EditorInspectorPlugin3DTexture::can_handle(Object *p_object) {
	return Object::cast_to<Texture3D>(p_object) != nullptr;
}

void EditorInspectorPlugin3DTexture::parse_begin(Object *p_object) {
	Texture3D *texture = Object::cast_to<Texture3D>(p_object);
	if (!texture) {
		return;
	}
	Ref<Texture3D> m(texture);

	Texture3DEditor *editor = memnew(Texture3DEditor);
	editor->edit(m);
	add_custom_control(editor);
}

Texture3DEditorPlugin::Texture3DEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPlugin3DTexture> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
