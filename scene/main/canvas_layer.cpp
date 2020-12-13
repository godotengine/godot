/*************************************************************************/
/*  canvas_layer.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "canvas_layer.h"
#include "viewport.h"

void CanvasLayer::set_layer(int p_xform) {
	layer = p_xform;
	if (viewport.is_valid()) {
		RenderingServer::get_singleton()->viewport_set_canvas_stacking(viewport, canvas, layer, get_index());
	}
}

int CanvasLayer::get_layer() const {
	return layer;
}

void CanvasLayer::set_transform(const Transform2D &p_xform) {
	transform = p_xform;
	locrotscale_dirty = true;
	if (viewport.is_valid()) {
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, canvas, transform);
	}
}

Transform2D CanvasLayer::get_transform() const {
	return transform;
}

void CanvasLayer::_update_xform() {
	transform.set_rotation_and_scale(rot, scale);
	transform.set_origin(ofs);
	if (viewport.is_valid()) {
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, canvas, transform);
	}
}

void CanvasLayer::_update_locrotscale() {
	ofs = transform.elements[2];
	rot = transform.get_rotation();
	scale = transform.get_scale();
	locrotscale_dirty = false;
}

void CanvasLayer::set_offset(const Vector2 &p_offset) {
	if (locrotscale_dirty) {
		_update_locrotscale();
	}

	ofs = p_offset;
	_update_xform();
}

Vector2 CanvasLayer::get_offset() const {
	if (locrotscale_dirty) {
		const_cast<CanvasLayer *>(this)->_update_locrotscale();
	}

	return ofs;
}

void CanvasLayer::set_rotation(real_t p_radians) {
	if (locrotscale_dirty) {
		_update_locrotscale();
	}

	rot = p_radians;
	_update_xform();
}

real_t CanvasLayer::get_rotation() const {
	if (locrotscale_dirty) {
		const_cast<CanvasLayer *>(this)->_update_locrotscale();
	}

	return rot;
}

void CanvasLayer::set_rotation_degrees(real_t p_degrees) {
	set_rotation(Math::deg2rad(p_degrees));
}

real_t CanvasLayer::get_rotation_degrees() const {
	return Math::rad2deg(get_rotation());
}

void CanvasLayer::set_scale(const Vector2 &p_scale) {
	if (locrotscale_dirty) {
		_update_locrotscale();
	}

	scale = p_scale;
	_update_xform();
}

Vector2 CanvasLayer::get_scale() const {
	if (locrotscale_dirty) {
		const_cast<CanvasLayer *>(this)->_update_locrotscale();
	}

	return scale;
}

void CanvasLayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (custom_viewport && ObjectDB::get_instance(custom_viewport_id)) {
				vp = custom_viewport;
			} else {
				vp = Node::get_viewport();
			}
			ERR_FAIL_COND(!vp);

			vp->_canvas_layer_add(this);
			viewport = vp->get_viewport_rid();

			RenderingServer::get_singleton()->viewport_attach_canvas(viewport, canvas);
			RenderingServer::get_singleton()->viewport_set_canvas_stacking(viewport, canvas, layer, get_index());
			RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, canvas, transform);
			_update_follow_viewport();

		} break;
		case NOTIFICATION_EXIT_TREE: {
			vp->_canvas_layer_remove(this);
			RenderingServer::get_singleton()->viewport_remove_canvas(viewport, canvas);
			viewport = RID();
			_update_follow_viewport(false);

		} break;
		case NOTIFICATION_MOVED_IN_PARENT: {
			if (is_inside_tree()) {
				RenderingServer::get_singleton()->viewport_set_canvas_stacking(viewport, canvas, layer, get_index());
			}

		} break;
	}
}

Size2 CanvasLayer::get_viewport_size() const {
	if (!is_inside_tree()) {
		return Size2(1, 1);
	}

	Rect2 r = vp->get_visible_rect();
	return r.size;
}

RID CanvasLayer::get_viewport() const {
	return viewport;
}

void CanvasLayer::set_custom_viewport(Node *p_viewport) {
	ERR_FAIL_NULL(p_viewport);
	if (is_inside_tree()) {
		vp->_canvas_layer_remove(this);
		RenderingServer::get_singleton()->viewport_remove_canvas(viewport, canvas);
		viewport = RID();
	}

	custom_viewport = Object::cast_to<Viewport>(p_viewport);

	if (custom_viewport) {
		custom_viewport_id = custom_viewport->get_instance_id();
	} else {
		custom_viewport_id = ObjectID();
	}

	if (is_inside_tree()) {
		if (custom_viewport) {
			vp = custom_viewport;
		} else {
			vp = Node::get_viewport();
		}

		vp->_canvas_layer_add(this);
		viewport = vp->get_viewport_rid();

		RenderingServer::get_singleton()->viewport_attach_canvas(viewport, canvas);
		RenderingServer::get_singleton()->viewport_set_canvas_stacking(viewport, canvas, layer, get_index());
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, canvas, transform);
	}
}

Node *CanvasLayer::get_custom_viewport() const {
	return custom_viewport;
}

void CanvasLayer::reset_sort_index() {
	sort_index = 0;
}

int CanvasLayer::get_sort_index() {
	return sort_index++;
}

RID CanvasLayer::get_canvas() const {
	return canvas;
}

void CanvasLayer::set_follow_viewport(bool p_enable) {
	if (follow_viewport == p_enable) {
		return;
	}

	follow_viewport = p_enable;
	_update_follow_viewport();
}

bool CanvasLayer::is_following_viewport() const {
	return follow_viewport;
}

void CanvasLayer::set_follow_viewport_scale(float p_ratio) {
	follow_viewport_scale = p_ratio;
	_update_follow_viewport();
}

float CanvasLayer::get_follow_viewport_scale() const {
	return follow_viewport_scale;
}

void CanvasLayer::_update_follow_viewport(bool p_force_exit) {
	if (!is_inside_tree()) {
		return;
	}
	if (p_force_exit || !follow_viewport) {
		RS::get_singleton()->canvas_set_parent(canvas, RID(), 1.0);
	} else {
		RS::get_singleton()->canvas_set_parent(canvas, vp->get_world_2d()->get_canvas(), follow_viewport_scale);
	}
}

void CanvasLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_layer", "layer"), &CanvasLayer::set_layer);
	ClassDB::bind_method(D_METHOD("get_layer"), &CanvasLayer::get_layer);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &CanvasLayer::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &CanvasLayer::get_transform);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &CanvasLayer::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &CanvasLayer::get_offset);

	ClassDB::bind_method(D_METHOD("set_rotation", "radians"), &CanvasLayer::set_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation"), &CanvasLayer::get_rotation);

	ClassDB::bind_method(D_METHOD("set_rotation_degrees", "degrees"), &CanvasLayer::set_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_rotation_degrees"), &CanvasLayer::get_rotation_degrees);

	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &CanvasLayer::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &CanvasLayer::get_scale);

	ClassDB::bind_method(D_METHOD("set_follow_viewport", "enable"), &CanvasLayer::set_follow_viewport);
	ClassDB::bind_method(D_METHOD("is_following_viewport"), &CanvasLayer::is_following_viewport);

	ClassDB::bind_method(D_METHOD("set_follow_viewport_scale", "scale"), &CanvasLayer::set_follow_viewport_scale);
	ClassDB::bind_method(D_METHOD("get_follow_viewport_scale"), &CanvasLayer::get_follow_viewport_scale);

	ClassDB::bind_method(D_METHOD("set_custom_viewport", "viewport"), &CanvasLayer::set_custom_viewport);
	ClassDB::bind_method(D_METHOD("get_custom_viewport"), &CanvasLayer::get_custom_viewport);

	ClassDB::bind_method(D_METHOD("get_canvas"), &CanvasLayer::get_canvas);

	ADD_GROUP("Layer", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layer", PROPERTY_HINT_RANGE, "-128,128,1"), "set_layer", "get_layer");
	ADD_GROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation_degrees", PROPERTY_HINT_RANGE, "-1080,1080,0.1,or_lesser,or_greater", PROPERTY_USAGE_EDITOR), "set_rotation_degrees", "get_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scale"), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform"), "set_transform", "get_transform");
	ADD_GROUP("", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "custom_viewport", PROPERTY_HINT_RESOURCE_TYPE, "Viewport", 0), "set_custom_viewport", "get_custom_viewport");
	ADD_GROUP("Follow Viewport", "follow_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_viewport_enable"), "set_follow_viewport", "is_following_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "follow_viewport_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001,or_greater,or_lesser"), "set_follow_viewport_scale", "get_follow_viewport_scale");
}

CanvasLayer::CanvasLayer() {
	vp = nullptr;
	scale = Vector2(1, 1);
	rot = 0;
	locrotscale_dirty = false;
	layer = 1;
	canvas = RS::get_singleton()->canvas_create();
	custom_viewport = nullptr;

	sort_index = 0;
	follow_viewport = false;
	follow_viewport_scale = 1.0;
}

CanvasLayer::~CanvasLayer() {
	RS::get_singleton()->free(canvas);
}
