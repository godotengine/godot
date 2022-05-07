/*************************************************************************/
/*  light_2d.cpp                                                         */
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

#include "light_2d.h"

#include "core/engine.h"
#include "servers/visual_server.h"

#ifdef TOOLS_ENABLED
Dictionary Light2D::_edit_get_state() const {
	Dictionary state = Node2D::_edit_get_state();
	state["offset"] = get_texture_offset();
	return state;
}

void Light2D::_edit_set_state(const Dictionary &p_state) {
	Node2D::_edit_set_state(p_state);
	set_texture_offset(p_state["offset"]);
}

void Light2D::_edit_set_pivot(const Point2 &p_pivot) {
	set_position(get_transform().xform(p_pivot));
	set_texture_offset(get_texture_offset() - p_pivot);
}

Point2 Light2D::_edit_get_pivot() const {
	return Vector2();
}

bool Light2D::_edit_use_pivot() const {
	return true;
}

Rect2 Light2D::_edit_get_rect() const {
	if (texture.is_null()) {
		return Rect2();
	}

	Size2 s = texture->get_size() * _scale;
	return Rect2(texture_offset - s / 2.0, s);
}

bool Light2D::_edit_use_rect() const {
	return !texture.is_null();
}
#endif

Rect2 Light2D::get_anchorable_rect() const {
	if (texture.is_null()) {
		return Rect2();
	}

	Size2 s = texture->get_size() * _scale;
	return Rect2(texture_offset - s / 2.0, s);
}

void Light2D::_update_light_visibility() {
	if (!is_inside_tree()) {
		return;
	}

	bool editor_ok = true;

#ifdef TOOLS_ENABLED
	if (editor_only) {
		if (!Engine::get_singleton()->is_editor_hint()) {
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

	VS::get_singleton()->canvas_light_set_enabled(canvas_light, enabled && is_visible_in_tree() && editor_ok);
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
	if (texture.is_valid()) {
		VS::get_singleton()->canvas_light_set_texture(canvas_light, texture->get_rid());
	} else {
		VS::get_singleton()->canvas_light_set_texture(canvas_light, RID());
	}

	update_configuration_warning();
}

Ref<Texture> Light2D::get_texture() const {
	return texture;
}

void Light2D::set_texture_offset(const Vector2 &p_offset) {
	texture_offset = p_offset;
	VS::get_singleton()->canvas_light_set_texture_offset(canvas_light, texture_offset);
	item_rect_changed();
	_change_notify("offset");
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
	// Avoid having 0 scale values, can lead to errors in physics and rendering.
	if (_scale == 0) {
		_scale = CMP_EPSILON;
	}
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

void Light2D::set_item_cull_mask(int p_mask) {
	item_mask = p_mask;
	VS::get_singleton()->canvas_light_set_item_cull_mask(canvas_light, item_mask);
}

int Light2D::get_item_cull_mask() const {
	return item_mask;
}

void Light2D::set_item_shadow_cull_mask(int p_mask) {
	item_shadow_mask = p_mask;
	VS::get_singleton()->canvas_light_set_item_shadow_cull_mask(canvas_light, item_shadow_mask);
}

int Light2D::get_item_shadow_cull_mask() const {
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

void Light2D::set_shadow_gradient_length(float p_multiplier) {
	shadow_gradient_length = p_multiplier;
	VS::get_singleton()->canvas_light_set_shadow_gradient_length(canvas_light, p_multiplier);
}

float Light2D::get_shadow_gradient_length() const {
	return shadow_gradient_length;
}

void Light2D::set_shadow_filter(ShadowFilter p_filter) {
	shadow_filter = p_filter;
	VS::get_singleton()->canvas_light_set_shadow_filter(canvas_light, VS::CanvasLightShadowFilter(p_filter));
}

Light2D::ShadowFilter Light2D::get_shadow_filter() const {
	return shadow_filter;
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
	String warning = Node2D::get_configuration_warning();
	if (!texture.is_valid()) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("A texture with the shape of the light must be supplied to the \"Texture\" property.");
	}

	return warning;
}

void Light2D::set_shadow_smooth(float p_amount) {
	shadow_smooth = p_amount;
	VS::get_singleton()->canvas_light_set_shadow_smooth(canvas_light, shadow_smooth);
}

float Light2D::get_shadow_smooth() const {
	return shadow_smooth;
}

void Light2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &Light2D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &Light2D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_editor_only", "editor_only"), &Light2D::set_editor_only);
	ClassDB::bind_method(D_METHOD("is_editor_only"), &Light2D::is_editor_only);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &Light2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &Light2D::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_offset", "texture_offset"), &Light2D::set_texture_offset);
	ClassDB::bind_method(D_METHOD("get_texture_offset"), &Light2D::get_texture_offset);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Light2D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Light2D::get_color);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &Light2D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &Light2D::get_height);

	ClassDB::bind_method(D_METHOD("set_energy", "energy"), &Light2D::set_energy);
	ClassDB::bind_method(D_METHOD("get_energy"), &Light2D::get_energy);

	ClassDB::bind_method(D_METHOD("set_texture_scale", "texture_scale"), &Light2D::set_texture_scale);
	ClassDB::bind_method(D_METHOD("get_texture_scale"), &Light2D::get_texture_scale);

	ClassDB::bind_method(D_METHOD("set_z_range_min", "z"), &Light2D::set_z_range_min);
	ClassDB::bind_method(D_METHOD("get_z_range_min"), &Light2D::get_z_range_min);

	ClassDB::bind_method(D_METHOD("set_z_range_max", "z"), &Light2D::set_z_range_max);
	ClassDB::bind_method(D_METHOD("get_z_range_max"), &Light2D::get_z_range_max);

	ClassDB::bind_method(D_METHOD("set_layer_range_min", "layer"), &Light2D::set_layer_range_min);
	ClassDB::bind_method(D_METHOD("get_layer_range_min"), &Light2D::get_layer_range_min);

	ClassDB::bind_method(D_METHOD("set_layer_range_max", "layer"), &Light2D::set_layer_range_max);
	ClassDB::bind_method(D_METHOD("get_layer_range_max"), &Light2D::get_layer_range_max);

	ClassDB::bind_method(D_METHOD("set_item_cull_mask", "item_cull_mask"), &Light2D::set_item_cull_mask);
	ClassDB::bind_method(D_METHOD("get_item_cull_mask"), &Light2D::get_item_cull_mask);

	ClassDB::bind_method(D_METHOD("set_item_shadow_cull_mask", "item_shadow_cull_mask"), &Light2D::set_item_shadow_cull_mask);
	ClassDB::bind_method(D_METHOD("get_item_shadow_cull_mask"), &Light2D::get_item_shadow_cull_mask);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &Light2D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &Light2D::get_mode);

	ClassDB::bind_method(D_METHOD("set_shadow_enabled", "enabled"), &Light2D::set_shadow_enabled);
	ClassDB::bind_method(D_METHOD("is_shadow_enabled"), &Light2D::is_shadow_enabled);

	ClassDB::bind_method(D_METHOD("set_shadow_buffer_size", "size"), &Light2D::set_shadow_buffer_size);
	ClassDB::bind_method(D_METHOD("get_shadow_buffer_size"), &Light2D::get_shadow_buffer_size);

	ClassDB::bind_method(D_METHOD("set_shadow_smooth", "smooth"), &Light2D::set_shadow_smooth);
	ClassDB::bind_method(D_METHOD("get_shadow_smooth"), &Light2D::get_shadow_smooth);

	ClassDB::bind_method(D_METHOD("set_shadow_gradient_length", "multiplier"), &Light2D::set_shadow_gradient_length);
	ClassDB::bind_method(D_METHOD("get_shadow_gradient_length"), &Light2D::get_shadow_gradient_length);

	ClassDB::bind_method(D_METHOD("set_shadow_filter", "filter"), &Light2D::set_shadow_filter);
	ClassDB::bind_method(D_METHOD("get_shadow_filter"), &Light2D::get_shadow_filter);

	ClassDB::bind_method(D_METHOD("set_shadow_color", "shadow_color"), &Light2D::set_shadow_color);
	ClassDB::bind_method(D_METHOD("get_shadow_color"), &Light2D::get_shadow_color);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_only"), "set_editor_only", "is_editor_only");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_texture_offset", "get_texture_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "texture_scale", PROPERTY_HINT_RANGE, "0.01,50,0.01"), "set_texture_scale", "get_texture_scale");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "energy", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_energy", "get_energy");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Add,Sub,Mix,Mask"), "set_mode", "get_mode");
	ADD_GROUP("Range", "range_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "range_height", PROPERTY_HINT_RANGE, "-2048,2048,0.1,or_lesser,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range_z_min", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"), "set_z_range_min", "get_z_range_min");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range_z_max", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"), "set_z_range_max", "get_z_range_max");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range_layer_min", PROPERTY_HINT_RANGE, "-512,512,1"), "set_layer_range_min", "get_layer_range_min");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range_layer_max", PROPERTY_HINT_RANGE, "-512,512,1"), "set_layer_range_max", "get_layer_range_max");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "range_item_cull_mask", PROPERTY_HINT_LAYERS_2D_RENDER), "set_item_cull_mask", "get_item_cull_mask");

	ADD_GROUP("Shadow", "shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow_enabled"), "set_shadow_enabled", "is_shadow_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color"), "set_shadow_color", "get_shadow_color");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_buffer_size", PROPERTY_HINT_RANGE, "32,16384,1"), "set_shadow_buffer_size", "get_shadow_buffer_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "shadow_gradient_length", PROPERTY_HINT_RANGE, "0,4096,0.1"), "set_shadow_gradient_length", "get_shadow_gradient_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_filter", PROPERTY_HINT_ENUM, "None,PCF3,PCF5,PCF7,PCF9,PCF13"), "set_shadow_filter", "get_shadow_filter");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "shadow_filter_smooth", PROPERTY_HINT_RANGE, "0,64,0.1"), "set_shadow_smooth", "get_shadow_smooth");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_item_cull_mask", PROPERTY_HINT_LAYERS_2D_RENDER), "set_item_shadow_cull_mask", "get_item_shadow_cull_mask");

	BIND_ENUM_CONSTANT(MODE_ADD);
	BIND_ENUM_CONSTANT(MODE_SUB);
	BIND_ENUM_CONSTANT(MODE_MIX);
	BIND_ENUM_CONSTANT(MODE_MASK);

	BIND_ENUM_CONSTANT(SHADOW_FILTER_NONE);
	BIND_ENUM_CONSTANT(SHADOW_FILTER_PCF3);
	BIND_ENUM_CONSTANT(SHADOW_FILTER_PCF5);
	BIND_ENUM_CONSTANT(SHADOW_FILTER_PCF7);
	BIND_ENUM_CONSTANT(SHADOW_FILTER_PCF9);
	BIND_ENUM_CONSTANT(SHADOW_FILTER_PCF13);
}

Light2D::Light2D() {
	canvas_light = RID_PRIME(VisualServer::get_singleton()->canvas_light_create());
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
	shadow_gradient_length = 0;
	energy = 1.0;
	shadow_color = Color(0, 0, 0, 0);
	shadow_filter = SHADOW_FILTER_NONE;
	shadow_smooth = 0;

	set_notify_transform(true);
}

Light2D::~Light2D() {
	VisualServer::get_singleton()->free(canvas_light);
}
