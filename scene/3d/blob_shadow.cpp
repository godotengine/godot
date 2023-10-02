/**************************************************************************/
/*  blob_shadow.cpp                                                       */
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

#include "blob_shadow.h"

#include "servers/visual/visual_server_blob_shadows.h"

void BlobShadow::_physics_interpolated_changed() {
	set_notify_transform(!is_physics_interpolated_and_enabled());
	Spatial::_physics_interpolated_changed();
}

void BlobShadow::_update_server(bool p_force_update) {
	if (data.blob.is_valid()) {
		Vector3 pos = get_global_transform_interpolated().origin;
		if (!pos.is_equal_approx(data.prev_pos) || p_force_update) {
			data.prev_pos = pos;
			VisualServer::get_singleton()->blob_shadow_update(data.blob, pos, data.radius[0]);
		}
	} else if (data.capsule.is_valid()) {
		Transform tr = get_global_transform_interpolated();
		if (!tr.is_equal_approx(data.prev_xform) || p_force_update) {
			data.prev_xform = tr;
			Vector3 pos = tr.origin;
			Vector3 pos_b = tr.xform(data.offset);
			VisualServer::get_singleton()->capsule_shadow_update(data.capsule, pos, data.radius[0], pos_b, data.radius[1]);
		}
	}
}

void BlobShadow::fti_update_servers_xform() {
	if (is_visible_in_tree()) {
		_update_server(false);
	}

	Spatial::fti_update_servers_xform();
}

void BlobShadow::_validate_property(PropertyInfo &property) const {
	if (property.name == "offset_radius" || property.name == "offset") {
		if (get_shadow_type() == BLOB_SHADOW_SPHERE) {
			property.usage = 0;
		}
	}
}

void BlobShadow::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_refresh_visibility(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_refresh_visibility(false);
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (!is_physics_interpolated_and_enabled() && is_visible_in_tree()) {
				_update_server(false);
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_refresh_visibility(is_inside_tree());
		} break;
	}
}

void BlobShadow::_refresh_visibility(bool p_in_tree) {
	bool sphere_present = false;
	bool capsule_present = false;

	bool visible = p_in_tree && is_visible_in_tree();

	if (visible) {
		sphere_present = data.type == BLOB_SHADOW_SPHERE;
		capsule_present = data.type == BLOB_SHADOW_CAPSULE;
	}

	if (sphere_present) {
		if (!data.blob.is_valid()) {
			data.blob = RID_PRIME(VisualServer::get_singleton()->blob_shadow_create());
		}
	} else {
		if (data.blob.is_valid()) {
			VisualServer::get_singleton()->free(data.blob);
			data.blob = RID();
		}
	}

	if (capsule_present) {
		if (!data.capsule.is_valid()) {
			data.capsule = RID_PRIME(VisualServer::get_singleton()->capsule_shadow_create());
		}
	} else {
		if (data.capsule.is_valid()) {
			VisualServer::get_singleton()->free(data.capsule);
			data.capsule = RID();
		}
	}

	_update_server(true);
}

void BlobShadow::set_radius(int p_index, real_t p_radius) {
	ERR_FAIL_INDEX(p_index, 2);

	if (p_radius == data.radius[p_index]) {
		return;
	}
	data.radius[p_index] = p_radius;
	_update_server(true);
	update_gizmo();
	_change_notify("radius");
	_change_notify("offset_radius");
}

real_t BlobShadow::get_radius(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, 2, 0);
	return data.radius[p_index];
}

void BlobShadow::set_offset(const Vector3 &p_offset) {
	if (p_offset == data.offset) {
		return;
	}
	data.offset = p_offset;
	_update_server(true);
	update_gizmo();
}

void BlobShadow::set_shadow_type(BlobShadowType p_type) {
	if (data.type == p_type) {
		return;
	}

	data.type = p_type;
	_refresh_visibility(is_inside_tree());
	update_gizmo();
	_change_notify();
}

void BlobShadow::_bind_methods() {
	BIND_ENUM_CONSTANT(BLOB_SHADOW_SPHERE);
	BIND_ENUM_CONSTANT(BLOB_SHADOW_CAPSULE);

	ClassDB::bind_method(D_METHOD("set_radius", "index", "radius"), &BlobShadow::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius", "index"), &BlobShadow::get_radius);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &BlobShadow::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &BlobShadow::get_offset);

	ClassDB::bind_method(D_METHOD("set_shadow_type", "type"), &BlobShadow::set_shadow_type);
	ClassDB::bind_method(D_METHOD("get_shadow_type"), &BlobShadow::get_shadow_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, "Sphere,Capsule"), "set_shadow_type", "get_shadow_type");

	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0,16,0.001,or_greater"), "set_radius", "get_radius", 0);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "offset_radius", PROPERTY_HINT_RANGE, "0,16,0.001,or_greater"), "set_radius", "get_radius", 1);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "offset"), "set_offset", "get_offset");
}

BlobShadow::BlobShadow() {
	set_notify_transform(true);
}

BlobShadow::~BlobShadow() {
	_refresh_visibility(false);
}
