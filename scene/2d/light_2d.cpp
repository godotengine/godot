/*************************************************************************/
/*  light_2d.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "light_2d.h"
#include "servers/visual_server.h"

void Light2D::edit_set_pivot(const Point2 &p_pivot) {

	set_texture_offset(p_pivot);
}

Point2 Light2D::edit_get_pivot() const {

	return get_texture_offset();
}
bool Light2D::edit_has_pivot() const {

	return true;
}

Rect2 Light2D::get_item_rect() const {

	if (texture.is_null())
		return Rect2(0, 0, 1, 1);

	Size2i s;

	s = texture->get_size() * _scale;
	Point2i ofs = texture_offset;
	ofs -= s / 2;

	if (s == Size2(0, 0))
		s = Size2(1, 1);

	return Rect2(ofs, s);
}

void Light2D::_update_light_visibility() {

	if (!is_inside_tree())
		return;

	bool editor_ok = true;

#ifdef TOOLS_ENABLED
	if (editor_only) {
		if (!get_tree()->is_editor_hint()) {
			editor_ok = false;
		} else {
			editor_ok = (get_tree()->get_edited_scene_root() && (this == get_tree()->get_edited_scene_root() || get_owner() == get_tree()->get_edited_scene_root()));
		}
	}
#else
	if (editor_only) {
		editor_ok = false;
	}
#endif

	VS::get_singleton()->canvas_light_set_enabled(canvas_light, enabled && is_visible() && editor_ok);
}

void Light2D::set_enabled(bool p_enabled) {

	enabled = p_enabled;
	_update_light_visibility();
}

bool Light2D::is_enabled() const {

	return enabled;
}

void Light2D::set_editor_only(bool p_editor_only) {

	editor_only = p_editor_only;
	_update_light_visibility();
}

bool Light2D::is_editor_only() const {

	return editor_only;
}

void Light2D::set_texture(const Ref<Texture> &p_texture) {

	texture = p_texture;
	if (texture.is_valid())
		VS::get_singleton()->canvas_light_set_texture(canvas_light, texture->get_rid());
	else
		VS::get_singleton()->canvas_light_set_texture(canvas_light, RID());

	update_configuration_warning();
}

Ref<Texture> Light2D::get_texture() const {

	return texture;
}

void Light2D::set_texture_offset(const Vector2 &p_offset) {

	texture_offset = p_offset;
	VS::get_singleton()->canvas_light_set_texture_offset(canvas_light, texture_offset);
	item_rect_changed();
}

Vector2 Light2D::get_texture_offset() const {

	return texture_offset;
}

void Light2D::set_color(const Color &p_color) {

	color = p_color;
	VS::get_singleton()->canvas_light_set_color(canvas_light, color);
}
Color Light2D::get_color() const {

	return color;
}

void Light2D::set_height(float p_height) {

	height = p_height;
	VS::get_singleton()->canvas_light_set_height(canvas_light, height);
}

float Light2D::get_height() const {

	return height;
}

void Light2D::set_energy(float p_energy) {

	energy = p_energy;
	VS::get_singleton()->canvas_light_set_energy(canvas_light, energy);
}

float Light2D::get_energy() const {

	return energy;
}

void Light2D::set_texture_scale(float p_scale) {

	_scale = p_scale;
	VS::get_singleton()->canvas_light_set_scale(canvas_light, _scale);
	item_rect_changed();
}

float Light2D::get_texture_scale() const {

	return _scale;
}

void Light2D::set_z_range_min(int p_min_z) {

	z_min = p_min_z;
	VS::get_singleton()->canvas_light_set_z_range(canvas_light, z_min, z_max);
}
int Light2D::get_z_range_min() const {

	return z_min;
}

void Light2D::set_z_range_max(int p_max_z) {

	z_max = p_max_z;
	VS::get_singleton()->canvas_light_set_z_range(canvas_light, z_min, z_max);
}
int Light2D::get_z_range_max() const {

	return z_max;
}

void Light2D::set_layer_range_min(int p_min_layer) {

	layer_min = p_min_layer;
	VS::get_singleton()->canvas_light_set_layer_range(canvas_light, layer_min, layer_max);
}
int Light2D::get_layer_range_min() const {

	return layer_min;
}

void Light2D::set_layer_range_max(int p_max_layer) {

	layer_max = p_max_layer;
	VS::get_singleton()->canvas_light_set_layer_range(canvas_light, layer_min, layer_max);
}
int Light2D::get_layer_range_max() const {

	return layer_max;
}

void Light2D::set_item_mask(int p_mask) {

	item_mask = p_mask;
	VS::get_singleton()->canvas_light_set_item_mask(canvas_light, item_mask);
}

int Light2D::get_item_mask() const {

	return item_mask;
}

void Light2D::set_item_shadow_mask(int p_mask) {

	item_shadow_mask = p_mask;
	VS::get_singleton()->canvas_light_set_item_shadow_mask(canvas_light, item_shadow_mask);
}

int Light2D::get_item_shadow_mask() const {

	return item_shadow_mask;
}

void Light2D::set_mode(Mode p_mode) {

	mode = p_mode;
	VS::get_singleton()->canvas_light_set_mode(canvas_light, VS::CanvasLightMode(p_mode));
}

Light2D::Mode Light2D::get_mode() const {

	return mode;
}

void Light2D::set_shadow_enabled(bool p_enabled) {

	shadow = p_enabled;
	VS::get_singleton()->canvas_light_set_shadow_enabled(canvas_light, shadow);
}
bool Light2D::is_shadow_enabled() const {

	return shadow;
}

void Light2D::set_shadow_buffer_size(int p_size) {

	shadow_buffer_size = p_size;
	VS::get_singleton()->canvas_light_set_shadow_buffer_size(canvas_light, shadow_buffer_size);
}

int Light2D::get_shadow_buffer_size() const {

	return shadow_buffer_size;
}

void Light2D::set_shadow_esm_multiplier(float p_multiplier) {

	shadow_esm_multiplier = p_multiplier;
	VS::get_singleton()->canvas_light_set_shadow_esm_multiplier(canvas_light, p_multiplier);
}

float Light2D::get_shadow_esm_multiplier() const {

	return shadow_esm_multiplier;
}

void Light2D::set_shadow_color(const Color &p_shadow_color) {
	shadow_color = p_shadow_color;
	VS::get_singleton()->canvas_light_set_shadow_color(canvas_light, shadow_color);
}

Color Light2D::get_shadow_color() const {
	return shadow_color;
}

void Light2D::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		VS::get_singleton()->canvas_light_attach_to_canvas(canvas_light, get_canvas());
		_update_light_visibility();
	}

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {

		VS::get_singleton()->canvas_light_set_transform(canvas_light, get_global_transform());
	}
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		_update_light_visibility();
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		VS::get_singleton()->canvas_light_attach_to_canvas(canvas_light, RID());
		_update_light_visibility();
	}
}

String Light2D::get_configuration_warning() const {

	if (!texture.is_valid()) {
		return TTR("A texture with the shape of the light must be supplied to the 'texture' property.");
	}

	return String();
}

void Light2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_enabled", "enabled"), &Light2D::set_enabled);
	ObjectTypeDB::bind_method(_MD("is_enabled"), &Light2D::is_enabled);

	ObjectTypeDB::bind_method(_MD("set_editor_only", "editor_only"), &Light2D::set_editor_only);
	ObjectTypeDB::bind_method(_MD("is_editor_only"), &Light2D::is_editor_only);

	ObjectTypeDB::bind_method(_MD("set_texture", "texture"), &Light2D::set_texture);
	ObjectTypeDB::bind_method(_MD("get_texture"), &Light2D::get_texture);

	ObjectTypeDB::bind_method(_MD("set_texture_offset", "texture_offset"), &Light2D::set_texture_offset);
	ObjectTypeDB::bind_method(_MD("get_texture_offset"), &Light2D::get_texture_offset);

	ObjectTypeDB::bind_method(_MD("set_color", "color"), &Light2D::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"), &Light2D::get_color);

	ObjectTypeDB::bind_method(_MD("set_height", "height"), &Light2D::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"), &Light2D::get_height);

	ObjectTypeDB::bind_method(_MD("set_energy", "energy"), &Light2D::set_energy);
	ObjectTypeDB::bind_method(_MD("get_energy"), &Light2D::get_energy);

	ObjectTypeDB::bind_method(_MD("set_texture_scale", "texture_scale"), &Light2D::set_texture_scale);
	ObjectTypeDB::bind_method(_MD("get_texture_scale"), &Light2D::get_texture_scale);

	ObjectTypeDB::bind_method(_MD("set_z_range_min", "z"), &Light2D::set_z_range_min);
	ObjectTypeDB::bind_method(_MD("get_z_range_min"), &Light2D::get_z_range_min);

	ObjectTypeDB::bind_method(_MD("set_z_range_max", "z"), &Light2D::set_z_range_max);
	ObjectTypeDB::bind_method(_MD("get_z_range_max"), &Light2D::get_z_range_max);

	ObjectTypeDB::bind_method(_MD("set_layer_range_min", "layer"), &Light2D::set_layer_range_min);
	ObjectTypeDB::bind_method(_MD("get_layer_range_min"), &Light2D::get_layer_range_min);

	ObjectTypeDB::bind_method(_MD("set_layer_range_max", "layer"), &Light2D::set_layer_range_max);
	ObjectTypeDB::bind_method(_MD("get_layer_range_max"), &Light2D::get_layer_range_max);

	ObjectTypeDB::bind_method(_MD("set_item_mask", "item_mask"), &Light2D::set_item_mask);
	ObjectTypeDB::bind_method(_MD("get_item_mask"), &Light2D::get_item_mask);

	ObjectTypeDB::bind_method(_MD("set_item_shadow_mask", "item_shadow_mask"), &Light2D::set_item_shadow_mask);
	ObjectTypeDB::bind_method(_MD("get_item_shadow_mask"), &Light2D::get_item_shadow_mask);

	ObjectTypeDB::bind_method(_MD("set_mode", "mode"), &Light2D::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"), &Light2D::get_mode);

	ObjectTypeDB::bind_method(_MD("set_shadow_enabled", "enabled"), &Light2D::set_shadow_enabled);
	ObjectTypeDB::bind_method(_MD("is_shadow_enabled"), &Light2D::is_shadow_enabled);

	ObjectTypeDB::bind_method(_MD("set_shadow_buffer_size", "size"), &Light2D::set_shadow_buffer_size);
	ObjectTypeDB::bind_method(_MD("get_shadow_buffer_size"), &Light2D::get_shadow_buffer_size);

	ObjectTypeDB::bind_method(_MD("set_shadow_esm_multiplier", "multiplier"), &Light2D::set_shadow_esm_multiplier);
	ObjectTypeDB::bind_method(_MD("get_shadow_esm_multiplier"), &Light2D::get_shadow_esm_multiplier);

	ObjectTypeDB::bind_method(_MD("set_shadow_color", "shadow_color"), &Light2D::set_shadow_color);
	ObjectTypeDB::bind_method(_MD("get_shadow_color"), &Light2D::get_shadow_color);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), _SCS("set_enabled"), _SCS("is_enabled"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_only"), _SCS("set_editor_only"), _SCS("is_editor_only"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_texture"), _SCS("get_texture"));
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), _SCS("set_texture_offset"), _SCS("get_texture_offset"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "scale", PROPERTY_HINT_RANGE, "0.01,50,0.01"), _SCS("set_texture_scale"), _SCS("get_texture_scale"));
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), _SCS("set_color"), _SCS("get_color"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "energy", PROPERTY_HINT_RANGE, "0.01,100,0.01"), _SCS("set_energy"), _SCS("get_energy"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Add,Sub,Mix,Mask"), _SCS("set_mode"), _SCS("get_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "range/height", PROPERTY_HINT_RANGE, "-100,100,0.1"), _SCS("set_height"), _SCS("get_height"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range/z_min", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"), _SCS("set_z_range_min"), _SCS("get_z_range_min"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range/z_max", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"), _SCS("set_z_range_max"), _SCS("get_z_range_max"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range/layer_min", PROPERTY_HINT_RANGE, "-512,512,1"), _SCS("set_layer_range_min"), _SCS("get_layer_range_min"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range/layer_max", PROPERTY_HINT_RANGE, "-512,512,1"), _SCS("set_layer_range_max"), _SCS("get_layer_range_max"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range/item_mask", PROPERTY_HINT_ALL_FLAGS), _SCS("set_item_mask"), _SCS("get_item_mask"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow/enabled"), _SCS("set_shadow_enabled"), _SCS("is_shadow_enabled"));
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow/color"), _SCS("set_shadow_color"), _SCS("get_shadow_color"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow/buffer_size", PROPERTY_HINT_RANGE, "32,16384,1"), _SCS("set_shadow_buffer_size"), _SCS("get_shadow_buffer_size"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "shadow/esm_multiplier", PROPERTY_HINT_RANGE, "1,4096,0.1"), _SCS("set_shadow_esm_multiplier"), _SCS("get_shadow_esm_multiplier"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow/item_mask", PROPERTY_HINT_ALL_FLAGS), _SCS("set_item_shadow_mask"), _SCS("get_item_shadow_mask"));

	BIND_CONSTANT(MODE_ADD);
	BIND_CONSTANT(MODE_SUB);
	BIND_CONSTANT(MODE_MIX);
	BIND_CONSTANT(MODE_MASK);
}

Light2D::Light2D() {

	canvas_light = VisualServer::get_singleton()->canvas_light_create();
	enabled = true;
	editor_only = false;
	shadow = false;
	color = Color(1, 1, 1);
	height = 0;
	_scale = 1.0;
	z_min = -1024;
	z_max = 1024;
	layer_min = 0;
	layer_max = 0;
	item_mask = 1;
	item_shadow_mask = 1;
	mode = MODE_ADD;
	shadow_buffer_size = 2048;
	shadow_esm_multiplier = 80;
	energy = 1.0;
	shadow_color = Color(0, 0, 0, 0);
}

Light2D::~Light2D() {

	VisualServer::get_singleton()->free(canvas_light);
}
