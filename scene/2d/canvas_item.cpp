/**************************************************************************/
/*  canvas_item.cpp                                                       */
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

#include "canvas_item.h"
#include "core/message_queue.h"
#include "core/method_bind_ext.gen.inc"
#include "core/os/input.h"
#include "core/version.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/viewport.h"
#include "scene/resources/font.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"
#include "scene/scene_string_names.h"
#include "servers/visual/visual_server_constants.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual_server.h"

Mutex CanvasItemMaterial::material_mutex;
SelfList<CanvasItemMaterial>::List *CanvasItemMaterial::dirty_materials = nullptr;
Map<CanvasItemMaterial::MaterialKey, CanvasItemMaterial::ShaderData> CanvasItemMaterial::shader_map;
CanvasItemMaterial::ShaderNames *CanvasItemMaterial::shader_names = nullptr;

void CanvasItemMaterial::init_shaders() {
	dirty_materials = memnew(SelfList<CanvasItemMaterial>::List);

	shader_names = memnew(ShaderNames);

	shader_names->particles_anim_h_frames = "particles_anim_h_frames";
	shader_names->particles_anim_v_frames = "particles_anim_v_frames";
	shader_names->particles_anim_loop = "particles_anim_loop";
}

void CanvasItemMaterial::finish_shaders() {
	memdelete(dirty_materials);
	memdelete(shader_names);
	dirty_materials = nullptr;
}

void CanvasItemMaterial::_update_shader() {
	dirty_materials->remove(&element);

	MaterialKey mk = _compute_key();
	if (mk.key == current_key.key) {
		return; //no update required in the end
	}

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {
		VS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}

	//must create a shader!

	// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
	String code = "// NOTE: Shader automatically converted from " VERSION_NAME " " VERSION_FULL_CONFIG "'s CanvasItemMaterial.\n\n";

	code += "shader_type canvas_item;\nrender_mode ";
	switch (blend_mode) {
		case BLEND_MODE_MIX:
			code += "blend_mix";
			break;
		case BLEND_MODE_ADD:
			code += "blend_add";
			break;
		case BLEND_MODE_SUB:
			code += "blend_sub";
			break;
		case BLEND_MODE_MUL:
			code += "blend_mul";
			break;
		case BLEND_MODE_PREMULT_ALPHA:
			code += "blend_premul_alpha";
			break;
		case BLEND_MODE_DISABLED:
			code += "blend_disabled";
			break;
	}

	switch (light_mode) {
		case LIGHT_MODE_NORMAL:
			break;
		case LIGHT_MODE_UNSHADED:
			code += ",unshaded";
			break;
		case LIGHT_MODE_LIGHT_ONLY:
			code += ",light_only";
			break;
	}

	code += ";\n";

	if (particles_animation) {
		code += "uniform int particles_anim_h_frames;\n";
		code += "uniform int particles_anim_v_frames;\n";
		code += "uniform bool particles_anim_loop;\n";

		code += "void vertex() {\n";

		code += "\tfloat h_frames = float(particles_anim_h_frames);\n";
		code += "\tfloat v_frames = float(particles_anim_v_frames);\n";

		code += "\tVERTEX.xy /= vec2(h_frames, v_frames);\n";

		code += "\tfloat particle_total_frames = float(particles_anim_h_frames * particles_anim_v_frames);\n";
		code += "\tfloat particle_frame = floor(INSTANCE_CUSTOM.z * float(particle_total_frames));\n";
		code += "\tif (!particles_anim_loop) {\n";
		code += "\t\tparticle_frame = clamp(particle_frame, 0.0, particle_total_frames - 1.0);\n";
		code += "\t} else {\n";
		code += "\t\tparticle_frame = mod(particle_frame, particle_total_frames);\n";
		code += "\t}";
		code += "\tUV /= vec2(h_frames, v_frames);\n";
		code += "\tUV += vec2(mod(particle_frame, h_frames) / h_frames, floor((particle_frame + 0.5) / h_frames) / v_frames);\n";
		code += "}\n";
	}

	ShaderData shader_data;
	shader_data.shader = VS::get_singleton()->shader_create();
	shader_data.users = 1;

	VS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	VS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}

void CanvasItemMaterial::flush_changes() {
	material_mutex.lock();

	while (dirty_materials->first()) {
		dirty_materials->first()->self()->_update_shader();
	}

	material_mutex.unlock();
}

void CanvasItemMaterial::_queue_shader_change() {
	material_mutex.lock();

	if (is_initialized && !element.in_list()) {
		dirty_materials->add(&element);
	}

	material_mutex.unlock();
}

bool CanvasItemMaterial::_is_shader_dirty() const {
	bool dirty = false;

	material_mutex.lock();

	dirty = element.in_list();

	material_mutex.unlock();

	return dirty;
}
void CanvasItemMaterial::set_blend_mode(BlendMode p_blend_mode) {
	blend_mode = p_blend_mode;
	_queue_shader_change();
}

CanvasItemMaterial::BlendMode CanvasItemMaterial::get_blend_mode() const {
	return blend_mode;
}

void CanvasItemMaterial::set_light_mode(LightMode p_light_mode) {
	light_mode = p_light_mode;
	_queue_shader_change();
}

CanvasItemMaterial::LightMode CanvasItemMaterial::get_light_mode() const {
	return light_mode;
}

void CanvasItemMaterial::set_particles_animation(bool p_particles_anim) {
	particles_animation = p_particles_anim;
	_queue_shader_change();
	_change_notify();
}

bool CanvasItemMaterial::get_particles_animation() const {
	return particles_animation;
}

void CanvasItemMaterial::set_particles_anim_h_frames(int p_frames) {
	particles_anim_h_frames = p_frames;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_h_frames, p_frames);
}

int CanvasItemMaterial::get_particles_anim_h_frames() const {
	return particles_anim_h_frames;
}
void CanvasItemMaterial::set_particles_anim_v_frames(int p_frames) {
	particles_anim_v_frames = p_frames;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_v_frames, p_frames);
}

int CanvasItemMaterial::get_particles_anim_v_frames() const {
	return particles_anim_v_frames;
}

void CanvasItemMaterial::set_particles_anim_loop(bool p_loop) {
	particles_anim_loop = p_loop;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_loop, particles_anim_loop);
}

bool CanvasItemMaterial::get_particles_anim_loop() const {
	return particles_anim_loop;
}

void CanvasItemMaterial::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("particles_anim_") && !particles_animation) {
		property.usage = 0;
	}
}

RID CanvasItemMaterial::get_shader_rid() const {
	ERR_FAIL_COND_V(!shader_map.has(current_key), RID());
	return shader_map[current_key].shader;
}

Shader::Mode CanvasItemMaterial::get_shader_mode() const {
	return Shader::MODE_CANVAS_ITEM;
}

void CanvasItemMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_blend_mode", "blend_mode"), &CanvasItemMaterial::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &CanvasItemMaterial::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_light_mode", "light_mode"), &CanvasItemMaterial::set_light_mode);
	ClassDB::bind_method(D_METHOD("get_light_mode"), &CanvasItemMaterial::get_light_mode);

	ClassDB::bind_method(D_METHOD("set_particles_animation", "particles_anim"), &CanvasItemMaterial::set_particles_animation);
	ClassDB::bind_method(D_METHOD("get_particles_animation"), &CanvasItemMaterial::get_particles_animation);

	ClassDB::bind_method(D_METHOD("set_particles_anim_h_frames", "frames"), &CanvasItemMaterial::set_particles_anim_h_frames);
	ClassDB::bind_method(D_METHOD("get_particles_anim_h_frames"), &CanvasItemMaterial::get_particles_anim_h_frames);

	ClassDB::bind_method(D_METHOD("set_particles_anim_v_frames", "frames"), &CanvasItemMaterial::set_particles_anim_v_frames);
	ClassDB::bind_method(D_METHOD("get_particles_anim_v_frames"), &CanvasItemMaterial::get_particles_anim_v_frames);

	ClassDB::bind_method(D_METHOD("set_particles_anim_loop", "loop"), &CanvasItemMaterial::set_particles_anim_loop);
	ClassDB::bind_method(D_METHOD("get_particles_anim_loop"), &CanvasItemMaterial::get_particles_anim_loop);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Sub,Mul,Premult Alpha"), "set_blend_mode", "get_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_mode", PROPERTY_HINT_ENUM, "Normal,Unshaded,Light Only"), "set_light_mode", "get_light_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "particles_animation"), "set_particles_animation", "get_particles_animation");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "particles_anim_h_frames", PROPERTY_HINT_RANGE, "1,128,1"), "set_particles_anim_h_frames", "get_particles_anim_h_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "particles_anim_v_frames", PROPERTY_HINT_RANGE, "1,128,1"), "set_particles_anim_v_frames", "get_particles_anim_v_frames");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "particles_anim_loop"), "set_particles_anim_loop", "get_particles_anim_loop");

	BIND_ENUM_CONSTANT(BLEND_MODE_MIX);
	BIND_ENUM_CONSTANT(BLEND_MODE_ADD);
	BIND_ENUM_CONSTANT(BLEND_MODE_SUB);
	BIND_ENUM_CONSTANT(BLEND_MODE_MUL);
	BIND_ENUM_CONSTANT(BLEND_MODE_PREMULT_ALPHA);

	BIND_ENUM_CONSTANT(LIGHT_MODE_NORMAL);
	BIND_ENUM_CONSTANT(LIGHT_MODE_UNSHADED);
	BIND_ENUM_CONSTANT(LIGHT_MODE_LIGHT_ONLY);
}

CanvasItemMaterial::CanvasItemMaterial() :
		element(this) {
	blend_mode = BLEND_MODE_MIX;
	light_mode = LIGHT_MODE_NORMAL;
	particles_animation = false;

	set_particles_anim_h_frames(1);
	set_particles_anim_v_frames(1);
	set_particles_anim_loop(false);

	current_key.key = 0;
	current_key.invalid_key = 1;
	is_initialized = true;
	_queue_shader_change();
}

CanvasItemMaterial::~CanvasItemMaterial() {
	material_mutex.lock();

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}

		VS::get_singleton()->material_set_shader(_get_material(), RID());
	}

	material_mutex.unlock();
}

///////////////////////////////////////////////////////////////////
#ifdef TOOLS_ENABLED
bool CanvasItem::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	if (_edit_use_rect()) {
		return _edit_get_rect().has_point(p_point);
	} else {
		return p_point.length() < p_tolerance;
	}
}

Transform2D CanvasItem::_edit_get_transform() const {
	return Transform2D(_edit_get_rotation(), _edit_get_position() + _edit_get_pivot());
}
#endif

bool CanvasItem::is_visible_in_tree() const {
	if (!is_inside_tree()) {
		return false;
	}

	const CanvasItem *p = this;

	while (p) {
		if (!p->visible) {
			return false;
		}
		p = p->get_parent_item();
	}

	if (canvas_layer) {
		return canvas_layer->is_visible();
	}

	return true;
}

void CanvasItem::_toplevel_visibility_changed(bool p_visible) {
	VisualServer::get_singleton()->canvas_item_set_visible(canvas_item, visible && p_visible);

	if (visible) {
		_propagate_visibility_changed(p_visible);
	} else {
		notification(NOTIFICATION_VISIBILITY_CHANGED);
	}
}

void CanvasItem::_propagate_visibility_changed(bool p_visible) {
	if (p_visible && first_draw) { //avoid propagating it twice
		first_draw = false;
	}
	notification(NOTIFICATION_VISIBILITY_CHANGED);

	if (p_visible) {
		update(); // Todo optimize.
	} else {
		emit_signal(SceneStringNames::get_singleton()->hide);
	}
	_block();

	for (int i = 0; i < get_child_count(); i++) {
		CanvasItem *c = Object::cast_to<CanvasItem>(get_child(i));

		if (c && c->visible && !c->toplevel) {
			c->_propagate_visibility_changed(p_visible);
		}
	}

	_unblock();
}

void CanvasItem::set_visible(bool p_visible) {
	if (visible == p_visible) {
		return;
	}

	visible = p_visible;
	VisualServer::get_singleton()->canvas_item_set_visible(canvas_item, p_visible);

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed(p_visible);
	_change_notify("visible");
}

void CanvasItem::show() {
	set_visible(true);
}

void CanvasItem::hide() {
	set_visible(false);
}

bool CanvasItem::is_visible() const {
	return visible;
}

CanvasItem *CanvasItem::current_item_drawn = nullptr;
CanvasItem *CanvasItem::get_current_item_drawn() {
	return current_item_drawn;
}

void CanvasItem::_update_callback() {
	if (!is_inside_tree()) {
		pending_update = false;
		return;
	}

	VisualServer::get_singleton()->canvas_item_clear(get_canvas_item());
	//todo updating = true - only allow drawing here
	if (is_visible_in_tree()) { // Todo optimize this!!
		if (first_draw) {
			notification(NOTIFICATION_VISIBILITY_CHANGED);
			first_draw = false;
		}
		drawing = true;
		current_item_drawn = this;
		notification(NOTIFICATION_DRAW);
		emit_signal(SceneStringNames::get_singleton()->draw);
		if (get_script_instance()) {
			get_script_instance()->call_multilevel_reversed(SceneStringNames::get_singleton()->_draw, nullptr, 0);
		}
		current_item_drawn = nullptr;
		drawing = false;
	}
	//todo updating = false
	pending_update = false; // don't change to false until finished drawing (avoid recursive update)
}

Transform2D CanvasItem::get_global_transform_with_canvas() const {
	if (canvas_layer) {
		return canvas_layer->get_final_transform() * get_global_transform();
	} else if (is_inside_tree()) {
		return get_viewport()->get_canvas_transform() * get_global_transform();
	} else {
		return get_global_transform();
	}
}

Transform2D CanvasItem::get_global_transform() const {
	if (global_invalid) {
		const CanvasItem *pi = get_parent_item();
		if (pi) {
			global_transform = pi->get_global_transform() * get_transform();
		} else {
			global_transform = get_transform();
		}

		global_invalid = false;
	}

	return global_transform;
}

// Same as get_global_transform() but no reset for `global_invalid`.
Transform2D CanvasItem::get_global_transform_const() const {
	if (global_invalid) {
		const CanvasItem *pi = get_parent_item();
		if (pi) {
			global_transform = pi->get_global_transform_const() * get_transform();
		} else {
			global_transform = get_transform();
		}
	}

	return global_transform;
}

void CanvasItem::_toplevel_raise_self() {
	if (!is_inside_tree()) {
		return;
	}

	if (canvas_layer) {
		VisualServer::get_singleton()->canvas_item_set_draw_index(canvas_item, canvas_layer->get_sort_index());
	} else {
		VisualServer::get_singleton()->canvas_item_set_draw_index(canvas_item, get_viewport()->gui_get_canvas_sort_index());
	}
}

void CanvasItem::_enter_canvas() {
	if ((!Object::cast_to<CanvasItem>(get_parent())) || toplevel) {
		Node *n = this;

		canvas_layer = nullptr;

		while (n) {
			canvas_layer = Object::cast_to<CanvasLayer>(n);
			if (canvas_layer) {
				break;
			}
			if (Object::cast_to<Viewport>(n)) {
				break;
			}
			n = n->get_parent();
		}

		RID canvas;
		if (canvas_layer) {
			canvas = canvas_layer->get_canvas();
		} else {
			canvas = get_viewport()->find_world_2d()->get_canvas();
		}

		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item, canvas);

		canvas_group = "root_canvas" + itos(canvas.get_id());

		add_to_group(canvas_group);
		if (canvas_layer) {
			canvas_layer->reset_sort_index();
		} else {
			get_viewport()->gui_reset_canvas_sort_index();
		}

		get_tree()->call_group_flags(SceneTree::GROUP_CALL_UNIQUE, canvas_group, "_toplevel_raise_self");

	} else {
		CanvasItem *parent = get_parent_item();
		canvas_layer = parent->canvas_layer;
		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item, parent->get_canvas_item());
		VisualServer::get_singleton()->canvas_item_set_draw_index(canvas_item, get_index());
	}

	pending_update = false;
	update();

	notification(NOTIFICATION_ENTER_CANVAS);
}

void CanvasItem::_exit_canvas() {
	notification(NOTIFICATION_EXIT_CANVAS, true); //reverse the notification
	VisualServer::get_singleton()->canvas_item_set_parent(canvas_item, RID());
	canvas_layer = nullptr;
	if (canvas_group != "") {
		remove_from_group(canvas_group);
		canvas_group = "";
	}
}

void CanvasItem::_physics_interpolated_changed() {
	VisualServer::get_singleton()->canvas_item_set_interpolated(canvas_item, is_physics_interpolated());
}

void CanvasItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			ERR_FAIL_COND(!is_inside_tree());
			first_draw = true;

			Node *parent = get_parent();
			if (parent) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(parent);
				if (ci) {
					C = ci->children_items.push_back(this);
				}
			}
			_enter_canvas();
			if (!block_transform_notify && !xform_change.in_list()) {
				get_tree()->xform_change_list.add(&xform_change);
			}

			// If using physics interpolation, reset for this node only,
			// as a helper, as in most cases, users will want items reset when
			// adding to the tree.
			// In cases where they move immediately after adding,
			// there will be little cost in having two resets as these are cheap,
			// and it is worth it for convenience.
			// Do not propagate to children, as each child of an added branch
			// receives its own NOTIFICATION_ENTER_TREE, and this would
			// cause unnecessary duplicate resets.
			if (is_physics_interpolated_and_enabled()) {
				notification(NOTIFICATION_RESET_PHYSICS_INTERPOLATION);
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (xform_change.in_list()) {
				get_tree()->xform_change_list.remove(&xform_change);
			}
			_exit_canvas();
			if (C) {
				Object::cast_to<CanvasItem>(get_parent())->children_items.erase(C);
				C = nullptr;
			}
			global_invalid = true;
		} break;
		case NOTIFICATION_RESET_PHYSICS_INTERPOLATION: {
			if (is_visible_in_tree() && is_physics_interpolated()) {
				VisualServer::get_singleton()->canvas_item_reset_physics_interpolation(canvas_item);
			}
		} break;

		case NOTIFICATION_DRAW:
		case NOTIFICATION_TRANSFORM_CHANGED: {
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			emit_signal(SceneStringNames::get_singleton()->visibility_changed);
		} break;
	}
}

#ifdef DEV_ENABLED
void CanvasItem::_name_changed_notify() {
	// Even in DEV builds, there is no point in calling this unless we are debugging
	// canvas item names. Even calling the stub function will be expensive, as there
	// are a lot of canvas items.
#ifdef VISUAL_SERVER_CANVAS_DEBUG_ITEM_NAMES
	VisualServer::get_singleton()->canvas_item_set_name(canvas_item, get_name());
#endif
}
#endif

void CanvasItem::update_draw_order() {
	if (!is_inside_tree()) {
		return;
	}

	if (canvas_group != "") {
		get_tree()->call_group_flags(SceneTree::GROUP_CALL_UNIQUE, canvas_group, "_toplevel_raise_self");
	} else {
		CanvasItem *p = get_parent_item();
		ERR_FAIL_NULL(p);
		VisualServer::get_singleton()->canvas_item_set_draw_index(canvas_item, get_index());
	}
}

void CanvasItem::update() {
	if (!is_inside_tree()) {
		return;
	}
	if (pending_update) {
		return;
	}

	pending_update = true;

	MessageQueue::get_singleton()->push_call(this, "_update_callback");
}

void CanvasItem::set_modulate(const Color &p_modulate) {
	if (modulate == p_modulate) {
		return;
	}

	modulate = p_modulate;
	VisualServer::get_singleton()->canvas_item_set_modulate(canvas_item, modulate);
}
Color CanvasItem::get_modulate() const {
	return modulate;
}

void CanvasItem::set_as_toplevel(bool p_toplevel) {
	if (toplevel == p_toplevel) {
		return;
	}

	if (!is_inside_tree()) {
		toplevel = p_toplevel;
		return;
	}

	_exit_canvas();
	toplevel = p_toplevel;
	_enter_canvas();

	_notify_transform();
}

bool CanvasItem::is_set_as_toplevel() const {
	return toplevel;
}

CanvasItem *CanvasItem::get_parent_item() const {
	if (toplevel) {
		return nullptr;
	}

	return Object::cast_to<CanvasItem>(get_parent());
}

void CanvasItem::set_self_modulate(const Color &p_self_modulate) {
	if (self_modulate == p_self_modulate) {
		return;
	}

	self_modulate = p_self_modulate;
	VisualServer::get_singleton()->canvas_item_set_self_modulate(canvas_item, self_modulate);
}
Color CanvasItem::get_self_modulate() const {
	return self_modulate;
}

void CanvasItem::set_light_mask(int p_light_mask) {
	if (light_mask == p_light_mask) {
		return;
	}

	light_mask = p_light_mask;
	VS::get_singleton()->canvas_item_set_light_mask(canvas_item, p_light_mask);
}

int CanvasItem::get_light_mask() const {
	return light_mask;
}

void CanvasItem::set_canvas_item_use_identity_transform(bool p_enable) {
	// Prevent sending item transforms to VisualServer when using global coords.
	_set_use_identity_transform(p_enable);

	// Let VisualServer know not to concatenate the parent transform during the render.
	VisualServer::get_singleton()->canvas_item_set_use_identity_transform(get_canvas_item(), p_enable);

	if (is_inside_tree()) {
		if (p_enable) {
			// Make sure item is using identity transform in server.
			VisualServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), Transform2D());
		} else {
			// Make sure item transform is up to date in server if switching identity transform off.
			VisualServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), get_transform());
		}
	}
}

void CanvasItem::item_rect_changed(bool p_size_changed) {
	if (p_size_changed) {
		update();
	}
	emit_signal(SceneStringNames::get_singleton()->item_rect_changed);
}

void CanvasItem::draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	VisualServer::get_singleton()->canvas_item_add_line(canvas_item, p_from, p_to, p_color, p_width, p_antialiased);
}

void CanvasItem::draw_polyline(const Vector<Point2> &p_points, const Color &p_color, float p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Vector<Color> colors;
	colors.push_back(p_color);
	VisualServer::get_singleton()->canvas_item_add_polyline(canvas_item, p_points, colors, p_width, p_antialiased);
}

void CanvasItem::draw_polyline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, float p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	VisualServer::get_singleton()->canvas_item_add_polyline(canvas_item, p_points, p_colors, p_width, p_antialiased);
}

void CanvasItem::draw_arc(const Vector2 &p_center, float p_radius, float p_start_angle, float p_end_angle, int p_point_count, const Color &p_color, float p_width, bool p_antialiased) {
	Vector<Point2> points;
	points.resize(p_point_count);
	const float delta_angle = p_end_angle - p_start_angle;
	for (int i = 0; i < p_point_count; i++) {
		float theta = (i / (p_point_count - 1.0f)) * delta_angle + p_start_angle;
		points.set(i, p_center + Vector2(Math::cos(theta), Math::sin(theta)) * p_radius);
	}

	draw_polyline(points, p_color, p_width, p_antialiased);
}

void CanvasItem::draw_multiline(const Vector<Point2> &p_points, const Color &p_color, float p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Vector<Color> colors;
	colors.push_back(p_color);
	VisualServer::get_singleton()->canvas_item_add_multiline(canvas_item, p_points, colors, p_width, p_antialiased);
}

void CanvasItem::draw_multiline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, float p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	VisualServer::get_singleton()->canvas_item_add_multiline(canvas_item, p_points, p_colors, p_width, p_antialiased);
}

void CanvasItem::draw_rect(const Rect2 &p_rect, const Color &p_color, bool p_filled, float p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	if (p_filled) {
		if (p_width != 1.0) {
			WARN_PRINT("The draw_rect() \"width\" argument has no effect when \"filled\" is \"true\".");
		}

		if (p_antialiased) {
			WARN_PRINT("The draw_rect() \"antialiased\" argument has no effect when \"filled\" is \"true\".");
		}

		VisualServer::get_singleton()->canvas_item_add_rect(canvas_item, p_rect, p_color);
	} else {
		// Thick lines are offset depending on their width to avoid partial overlapping.
		// Thin lines don't require an offset, so don't apply one in this case
		float offset;
		if (p_width >= 2) {
			offset = p_width / 2.0;
		} else {
			offset = 0.0;
		}

		VisualServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(-offset, 0),
				p_rect.position + Size2(p_rect.size.width + offset, 0),
				p_color,
				p_width,
				p_antialiased);
		VisualServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(p_rect.size.width, offset),
				p_rect.position + Size2(p_rect.size.width, p_rect.size.height - offset),
				p_color,
				p_width,
				p_antialiased);
		VisualServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(p_rect.size.width + offset, p_rect.size.height),
				p_rect.position + Size2(-offset, p_rect.size.height),
				p_color,
				p_width,
				p_antialiased);
		VisualServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(0, p_rect.size.height - offset),
				p_rect.position + Size2(0, offset),
				p_color,
				p_width,
				p_antialiased);
	}
}

void CanvasItem::draw_circle(const Point2 &p_pos, float p_radius, const Color &p_color) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	VisualServer::get_singleton()->canvas_item_add_circle(canvas_item, p_pos, p_radius, p_color);
}

void CanvasItem::draw_texture(const Ref<Texture> &p_texture, const Point2 &p_pos, const Color &p_modulate, const Ref<Texture> &p_normal_map) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_texture.is_null());

	p_texture->draw(canvas_item, p_pos, p_modulate, false, p_normal_map);
}

void CanvasItem::draw_texture_rect(const Ref<Texture> &p_texture, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_texture.is_null());
	p_texture->draw_rect(canvas_item, p_rect, p_tile, p_modulate, p_transpose, p_normal_map);
}
void CanvasItem::draw_texture_rect_region(const Ref<Texture> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map, bool p_clip_uv) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
	ERR_FAIL_COND(p_texture.is_null());
	p_texture->draw_rect_region(canvas_item, p_rect, p_src_rect, p_modulate, p_transpose, p_normal_map, p_clip_uv);
}

void CanvasItem::draw_style_box(const Ref<StyleBox> &p_style_box, const Rect2 &p_rect) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_style_box.is_null());

	p_style_box->draw(canvas_item, p_rect);
}
void CanvasItem::draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, Ref<Texture> p_texture, float p_width, const Ref<Texture> &p_normal_map) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RID rid_normal = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_primitive(canvas_item, p_points, p_colors, p_uvs, rid, p_width, rid_normal);
}
void CanvasItem::draw_set_transform(const Point2 &p_offset, float p_rot, const Size2 &p_scale) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Transform2D xform(p_rot, p_offset);
	xform.scale_basis(p_scale);
	VisualServer::get_singleton()->canvas_item_add_set_transform(canvas_item, xform);
}

void CanvasItem::draw_set_transform_matrix(const Transform2D &p_matrix) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	VisualServer::get_singleton()->canvas_item_add_set_transform(canvas_item, p_matrix);
}

void CanvasItem::draw_polygon(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, Ref<Texture> p_texture, const Ref<Texture> &p_normal_map, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RID rid_normal = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_polygon(canvas_item, p_points, p_colors, p_uvs, rid, rid_normal, p_antialiased);
}

void CanvasItem::draw_colored_polygon(const Vector<Point2> &p_points, const Color &p_color, const Vector<Point2> &p_uvs, Ref<Texture> p_texture, const Ref<Texture> &p_normal_map, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Vector<Color> colors;
	colors.push_back(p_color);
	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RID rid_normal = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_polygon(canvas_item, p_points, colors, p_uvs, rid, rid_normal, p_antialiased);
}

void CanvasItem::draw_mesh(const Ref<Mesh> &p_mesh, const Ref<Texture> &p_texture, const Ref<Texture> &p_normal_map, const Transform2D &p_transform, const Color &p_modulate) {
	ERR_FAIL_COND(p_mesh.is_null());
	RID texture_rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RID normal_map_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();

	VisualServer::get_singleton()->canvas_item_add_mesh(canvas_item, p_mesh->get_rid(), p_transform, p_modulate, texture_rid, normal_map_rid);
}
void CanvasItem::draw_multimesh(const Ref<MultiMesh> &p_multimesh, const Ref<Texture> &p_texture, const Ref<Texture> &p_normal_map) {
	ERR_FAIL_COND(p_multimesh.is_null());
	RID texture_rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RID normal_map_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_multimesh(canvas_item, p_multimesh->get_rid(), texture_rid, normal_map_rid);
}

void CanvasItem::select_font(const Ref<Font> &p_font) {
	// Purely to keep canvas item SDF state up to date for now.
	bool new_font_sdf_selected = p_font.is_valid() && p_font->is_distance_field_hint();

	if (font_sdf_selected != new_font_sdf_selected) {
		ERR_FAIL_COND(!get_canvas_item().is_valid());
		font_sdf_selected = new_font_sdf_selected;
		VisualServer::get_singleton()->canvas_item_set_distance_field_mode(get_canvas_item(), font_sdf_selected);
	}
}

void CanvasItem::draw_string(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_text, const Color &p_modulate, int p_clip_w) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_font.is_null());
	p_font->draw(canvas_item, p_pos, p_text, p_modulate, p_clip_w);
}

float CanvasItem::draw_char(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_char, const String &p_next, const Color &p_modulate) {
	ERR_FAIL_COND_V_MSG(!drawing, 0, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND_V(p_char.length() != 1, 0);
	ERR_FAIL_COND_V(p_font.is_null(), 0);

	if (p_font->has_outline()) {
		p_font->draw_char(canvas_item, p_pos, p_char[0], p_next.c_str()[0], Color(1, 1, 1), true);
	}
	return p_font->draw_char(canvas_item, p_pos, p_char[0], p_next.c_str()[0], p_modulate);
}

void CanvasItem::_notify_transform(CanvasItem *p_node) {
	/* This check exists to avoid re-propagating the transform
	 * notification down the tree on dirty nodes. It provides
	 * optimization by avoiding redundancy (nodes are dirty, will get the
	 * notification anyway).
	 */

	if (/*p_node->xform_change.in_list() &&*/ p_node->global_invalid) {
		return; //nothing to do
	}

	p_node->global_invalid = true;

	if (p_node->notify_transform && !p_node->xform_change.in_list()) {
		if (!p_node->block_transform_notify) {
			if (p_node->is_inside_tree()) {
				get_tree()->xform_change_list.add(&p_node->xform_change);
			}
		}
	}

	for (List<CanvasItem *>::Element *E = p_node->children_items.front(); E; E = E->next()) {
		CanvasItem *ci = E->get();
		if (ci->toplevel) {
			continue;
		}
		_notify_transform(ci);
	}
}

Rect2 CanvasItem::get_viewport_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	return get_viewport()->get_visible_rect();
}

RID CanvasItem::get_canvas() const {
	ERR_FAIL_COND_V(!is_inside_tree(), RID());

	if (canvas_layer) {
		return canvas_layer->get_canvas();
	} else {
		return get_viewport()->find_world_2d()->get_canvas();
	}
}

ObjectID CanvasItem::get_canvas_layer_instance_id() const {
	if (canvas_layer) {
		return canvas_layer->get_instance_id();
	} else {
		return 0;
	}
}

CanvasItem *CanvasItem::get_toplevel() const {
	CanvasItem *ci = const_cast<CanvasItem *>(this);
	while (!ci->toplevel && Object::cast_to<CanvasItem>(ci->get_parent())) {
		ci = Object::cast_to<CanvasItem>(ci->get_parent());
	}

	return ci;
}

Ref<World2D> CanvasItem::get_world_2d() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Ref<World2D>());

	CanvasItem *tl = get_toplevel();

	if (tl->get_viewport()) {
		return tl->get_viewport()->find_world_2d();
	} else {
		return Ref<World2D>();
	}
}

RID CanvasItem::get_viewport_rid() const {
	ERR_FAIL_COND_V(!is_inside_tree(), RID());
	return get_viewport()->get_viewport_rid();
}

void CanvasItem::set_block_transform_notify(bool p_enable) {
	block_transform_notify = p_enable;
}

bool CanvasItem::is_block_transform_notify_enabled() const {
	return block_transform_notify;
}

void CanvasItem::set_draw_behind_parent(bool p_enable) {
	if (behind == p_enable) {
		return;
	}
	behind = p_enable;
	VisualServer::get_singleton()->canvas_item_set_draw_behind_parent(canvas_item, behind);
}

bool CanvasItem::is_draw_behind_parent_enabled() const {
	return behind;
}

void CanvasItem::set_material(const Ref<Material> &p_material) {
	material = p_material;
	RID rid;
	if (material.is_valid()) {
		rid = material->get_rid();
	}
	VS::get_singleton()->canvas_item_set_material(canvas_item, rid);
	_change_notify(); //properties for material exposed
}

void CanvasItem::set_use_parent_material(bool p_use_parent_material) {
	use_parent_material = p_use_parent_material;
	VS::get_singleton()->canvas_item_set_use_parent_material(canvas_item, p_use_parent_material);
}

bool CanvasItem::get_use_parent_material() const {
	return use_parent_material;
}

Ref<Material> CanvasItem::get_material() const {
	return material;
}

Vector2 CanvasItem::make_canvas_position_local(const Vector2 &screen_point) const {
	ERR_FAIL_COND_V(!is_inside_tree(), screen_point);

	Transform2D local_matrix = (get_canvas_transform() * get_global_transform()).affine_inverse();

	return local_matrix.xform(screen_point);
}

Ref<InputEvent> CanvasItem::make_input_local(const Ref<InputEvent> &p_event) const {
	ERR_FAIL_COND_V(p_event.is_null(), p_event);
	ERR_FAIL_COND_V(!is_inside_tree(), p_event);

	return p_event->xformed_by((get_canvas_transform() * get_global_transform()).affine_inverse());
}

Vector2 CanvasItem::get_global_mouse_position() const {
	ERR_FAIL_COND_V(!get_viewport(), Vector2());
	return get_canvas_transform().affine_inverse().xform(get_viewport()->get_mouse_position());
}

Vector2 CanvasItem::get_local_mouse_position() const {
	ERR_FAIL_COND_V(!get_viewport(), Vector2());

	return get_global_transform().affine_inverse().xform(get_global_mouse_position());
}

void CanvasItem::force_update_transform() {
	ERR_FAIL_COND(!is_inside_tree());
	if (!xform_change.in_list()) {
		return;
	}

	get_tree()->xform_change_list.remove(&xform_change);

	notification(NOTIFICATION_TRANSFORM_CHANGED);
}

void CanvasItem::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_toplevel_visibility_changed", "visible"), &CanvasItem::_toplevel_visibility_changed);
	ClassDB::bind_method(D_METHOD("_toplevel_raise_self"), &CanvasItem::_toplevel_raise_self);
	ClassDB::bind_method(D_METHOD("_update_callback"), &CanvasItem::_update_callback);

#ifdef TOOLS_ENABLED
	ClassDB::bind_method(D_METHOD("_edit_set_state", "state"), &CanvasItem::_edit_set_state);
	ClassDB::bind_method(D_METHOD("_edit_get_state"), &CanvasItem::_edit_get_state);
	ClassDB::bind_method(D_METHOD("_edit_set_position", "position"), &CanvasItem::_edit_set_position);
	ClassDB::bind_method(D_METHOD("_edit_get_position"), &CanvasItem::_edit_get_position);
	ClassDB::bind_method(D_METHOD("_edit_set_scale", "scale"), &CanvasItem::_edit_set_scale);
	ClassDB::bind_method(D_METHOD("_edit_get_scale"), &CanvasItem::_edit_get_scale);
	ClassDB::bind_method(D_METHOD("_edit_set_rect", "rect"), &CanvasItem::_edit_set_rect);
	ClassDB::bind_method(D_METHOD("_edit_get_rect"), &CanvasItem::_edit_get_rect);
	ClassDB::bind_method(D_METHOD("_edit_use_rect"), &CanvasItem::_edit_use_rect);
	ClassDB::bind_method(D_METHOD("_edit_set_rotation", "degrees"), &CanvasItem::_edit_set_rotation);
	ClassDB::bind_method(D_METHOD("_edit_get_rotation"), &CanvasItem::_edit_get_rotation);
	ClassDB::bind_method(D_METHOD("_edit_use_rotation"), &CanvasItem::_edit_use_rotation);
	ClassDB::bind_method(D_METHOD("_edit_set_pivot", "pivot"), &CanvasItem::_edit_set_pivot);
	ClassDB::bind_method(D_METHOD("_edit_get_pivot"), &CanvasItem::_edit_get_pivot);
	ClassDB::bind_method(D_METHOD("_edit_use_pivot"), &CanvasItem::_edit_use_pivot);
	ClassDB::bind_method(D_METHOD("_edit_get_transform"), &CanvasItem::_edit_get_transform);
#endif

	ClassDB::bind_method(D_METHOD("get_canvas_item"), &CanvasItem::get_canvas_item);

	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &CanvasItem::set_visible);
	ClassDB::bind_method(D_METHOD("is_visible"), &CanvasItem::is_visible);
	ClassDB::bind_method(D_METHOD("is_visible_in_tree"), &CanvasItem::is_visible_in_tree);
	ClassDB::bind_method(D_METHOD("show"), &CanvasItem::show);
	ClassDB::bind_method(D_METHOD("hide"), &CanvasItem::hide);

	ClassDB::bind_method(D_METHOD("update"), &CanvasItem::update);

	ClassDB::bind_method(D_METHOD("set_as_toplevel", "enable"), &CanvasItem::set_as_toplevel);
	ClassDB::bind_method(D_METHOD("is_set_as_toplevel"), &CanvasItem::is_set_as_toplevel);

	ClassDB::bind_method(D_METHOD("set_light_mask", "light_mask"), &CanvasItem::set_light_mask);
	ClassDB::bind_method(D_METHOD("get_light_mask"), &CanvasItem::get_light_mask);

	ClassDB::bind_method(D_METHOD("set_modulate", "modulate"), &CanvasItem::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &CanvasItem::get_modulate);
	ClassDB::bind_method(D_METHOD("set_self_modulate", "self_modulate"), &CanvasItem::set_self_modulate);
	ClassDB::bind_method(D_METHOD("get_self_modulate"), &CanvasItem::get_self_modulate);

	ClassDB::bind_method(D_METHOD("set_draw_behind_parent", "enable"), &CanvasItem::set_draw_behind_parent);
	ClassDB::bind_method(D_METHOD("is_draw_behind_parent_enabled"), &CanvasItem::is_draw_behind_parent_enabled);

	ClassDB::bind_method(D_METHOD("_set_on_top", "on_top"), &CanvasItem::_set_on_top);
	ClassDB::bind_method(D_METHOD("_is_on_top"), &CanvasItem::_is_on_top);
	//ClassDB::bind_method(D_METHOD("get_transform"),&CanvasItem::get_transform);

	ClassDB::bind_method(D_METHOD("draw_line", "from", "to", "color", "width", "antialiased"), &CanvasItem::draw_line, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_polyline", "points", "color", "width", "antialiased"), &CanvasItem::draw_polyline, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_polyline_colors", "points", "colors", "width", "antialiased"), &CanvasItem::draw_polyline_colors, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_arc", "center", "radius", "start_angle", "end_angle", "point_count", "color", "width", "antialiased"), &CanvasItem::draw_arc, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_multiline", "points", "color", "width", "antialiased"), &CanvasItem::draw_multiline, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_multiline_colors", "points", "colors", "width", "antialiased"), &CanvasItem::draw_multiline_colors, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_rect", "rect", "color", "filled", "width", "antialiased"), &CanvasItem::draw_rect, DEFVAL(true), DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_circle", "position", "radius", "color"), &CanvasItem::draw_circle);
	ClassDB::bind_method(D_METHOD("draw_texture", "texture", "position", "modulate", "normal_map"), &CanvasItem::draw_texture, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("draw_texture_rect", "texture", "rect", "tile", "modulate", "transpose", "normal_map"), &CanvasItem::draw_texture_rect, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("draw_texture_rect_region", "texture", "rect", "src_rect", "modulate", "transpose", "normal_map", "clip_uv"), &CanvasItem::draw_texture_rect_region, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(Variant()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("draw_style_box", "style_box", "rect"), &CanvasItem::draw_style_box);
	ClassDB::bind_method(D_METHOD("draw_primitive", "points", "colors", "uvs", "texture", "width", "normal_map"), &CanvasItem::draw_primitive, DEFVAL(Variant()), DEFVAL(1.0), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("draw_polygon", "points", "colors", "uvs", "texture", "normal_map", "antialiased"), &CanvasItem::draw_polygon, DEFVAL(PoolVector2Array()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_colored_polygon", "points", "color", "uvs", "texture", "normal_map", "antialiased"), &CanvasItem::draw_colored_polygon, DEFVAL(PoolVector2Array()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_string", "font", "position", "text", "modulate", "clip_w"), &CanvasItem::draw_string, DEFVAL(Color(1, 1, 1)), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("draw_char", "font", "position", "char", "next", "modulate"), &CanvasItem::draw_char, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_mesh", "mesh", "texture", "normal_map", "transform", "modulate"), &CanvasItem::draw_mesh, DEFVAL(Ref<Texture>()), DEFVAL(Transform2D()), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_multimesh", "multimesh", "texture", "normal_map"), &CanvasItem::draw_multimesh, DEFVAL(Ref<Texture>()));

	ClassDB::bind_method(D_METHOD("draw_set_transform", "position", "rotation", "scale"), &CanvasItem::draw_set_transform);
	ClassDB::bind_method(D_METHOD("draw_set_transform_matrix", "xform"), &CanvasItem::draw_set_transform_matrix);
	ClassDB::bind_method(D_METHOD("get_transform"), &CanvasItem::get_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform"), &CanvasItem::get_global_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform_with_canvas"), &CanvasItem::get_global_transform_with_canvas);
	ClassDB::bind_method(D_METHOD("get_viewport_transform"), &CanvasItem::get_viewport_transform);
	ClassDB::bind_method(D_METHOD("get_viewport_rect"), &CanvasItem::get_viewport_rect);
	ClassDB::bind_method(D_METHOD("get_canvas_transform"), &CanvasItem::get_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_local_mouse_position"), &CanvasItem::get_local_mouse_position);
	ClassDB::bind_method(D_METHOD("get_global_mouse_position"), &CanvasItem::get_global_mouse_position);
	ClassDB::bind_method(D_METHOD("get_canvas"), &CanvasItem::get_canvas);
	ClassDB::bind_method(D_METHOD("get_world_2d"), &CanvasItem::get_world_2d);
	//ClassDB::bind_method(D_METHOD("get_viewport"),&CanvasItem::get_viewport);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CanvasItem::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CanvasItem::get_material);

	ClassDB::bind_method(D_METHOD("set_use_parent_material", "enable"), &CanvasItem::set_use_parent_material);
	ClassDB::bind_method(D_METHOD("get_use_parent_material"), &CanvasItem::get_use_parent_material);

	ClassDB::bind_method(D_METHOD("set_notify_local_transform", "enable"), &CanvasItem::set_notify_local_transform);
	ClassDB::bind_method(D_METHOD("is_local_transform_notification_enabled"), &CanvasItem::is_local_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("set_notify_transform", "enable"), &CanvasItem::set_notify_transform);
	ClassDB::bind_method(D_METHOD("is_transform_notification_enabled"), &CanvasItem::is_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("force_update_transform"), &CanvasItem::force_update_transform);

	ClassDB::bind_method(D_METHOD("make_canvas_position_local", "screen_point"), &CanvasItem::make_canvas_position_local);
	ClassDB::bind_method(D_METHOD("make_input_local", "event"), &CanvasItem::make_input_local);

	BIND_VMETHOD(MethodInfo("_draw"));

	ADD_GROUP("Visibility", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "self_modulate"), "set_self_modulate", "get_self_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_behind_parent"), "set_draw_behind_parent", "is_draw_behind_parent_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_on_top", PROPERTY_HINT_NONE, "", 0), "_set_on_top", "_is_on_top"); //compatibility
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_mask", PROPERTY_HINT_LAYERS_2D_RENDER), "set_light_mask", "get_light_mask");

	ADD_GROUP("Material", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,CanvasItemMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_parent_material"), "set_use_parent_material", "get_use_parent_material");
	//exporting these things doesn't really make much sense i think
	// ADD_PROPERTY(PropertyInfo(Variant::BOOL, "toplevel", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_as_toplevel", "is_set_as_toplevel");
	// ADD_PROPERTY(PropertyInfo(Variant::BOOL,"transform/notify"),"set_transform_notify","is_transform_notify_enabled");

	ADD_SIGNAL(MethodInfo("draw"));
	ADD_SIGNAL(MethodInfo("visibility_changed"));
	ADD_SIGNAL(MethodInfo("hide"));
	ADD_SIGNAL(MethodInfo("item_rect_changed"));

	BIND_ENUM_CONSTANT(BLEND_MODE_MIX);
	BIND_ENUM_CONSTANT(BLEND_MODE_ADD);
	BIND_ENUM_CONSTANT(BLEND_MODE_SUB);
	BIND_ENUM_CONSTANT(BLEND_MODE_MUL);
	BIND_ENUM_CONSTANT(BLEND_MODE_PREMULT_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_MODE_DISABLED);

	BIND_CONSTANT(NOTIFICATION_TRANSFORM_CHANGED);
	BIND_CONSTANT(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	BIND_CONSTANT(NOTIFICATION_DRAW);
	BIND_CONSTANT(NOTIFICATION_VISIBILITY_CHANGED);
	BIND_CONSTANT(NOTIFICATION_ENTER_CANVAS);
	BIND_CONSTANT(NOTIFICATION_EXIT_CANVAS);
}

Transform2D CanvasItem::get_canvas_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform2D());

	if (canvas_layer) {
		return canvas_layer->get_final_transform();
	} else if (Object::cast_to<CanvasItem>(get_parent())) {
		return Object::cast_to<CanvasItem>(get_parent())->get_canvas_transform();
	} else {
		return get_viewport()->get_canvas_transform();
	}
}

Transform2D CanvasItem::get_viewport_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform2D());

	if (canvas_layer) {
		if (get_viewport()) {
			return get_viewport()->get_final_transform() * canvas_layer->get_final_transform();
		} else {
			return canvas_layer->get_final_transform();
		}

	} else {
		return get_viewport()->get_final_transform() * get_viewport()->get_canvas_transform();
	}
}

void CanvasItem::set_notify_local_transform(bool p_enable) {
	notify_local_transform = p_enable;
}

bool CanvasItem::is_local_transform_notification_enabled() const {
	return notify_local_transform;
}

void CanvasItem::set_notify_transform(bool p_enable) {
	if (notify_transform == p_enable) {
		return;
	}

	notify_transform = p_enable;

	if (notify_transform && is_inside_tree()) {
		//this ensures that invalid globals get resolved, so notifications can be received
		_ALLOW_DISCARD_ get_global_transform();
	}
}

bool CanvasItem::is_transform_notification_enabled() const {
	return notify_transform;
}

int CanvasItem::get_canvas_layer() const {
	if (canvas_layer) {
		return canvas_layer->get_layer();
	} else {
		return 0;
	}
}

CanvasItem::CanvasItem() :
		xform_change(this) {
	canvas_item = RID_PRIME(VisualServer::get_singleton()->canvas_item_create());
	visible = true;
	pending_update = false;
	modulate = Color(1, 1, 1, 1);
	self_modulate = Color(1, 1, 1, 1);
	toplevel = false;
	first_draw = false;
	drawing = false;
	behind = false;
	block_transform_notify = false;
	//viewport=NULL;
	canvas_layer = nullptr;
	use_parent_material = false;
	global_invalid = true;
	notify_local_transform = false;
	notify_transform = false;
	font_sdf_selected = false;
	light_mask = 1;

	C = nullptr;
}

CanvasItem::~CanvasItem() {
	VisualServer::get_singleton()->free(canvas_item);
}
