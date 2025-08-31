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

void BlobShadow::fti_update_servers_xform() {
	if (data.blob.is_valid() && VisualServerBlobShadows::is_allowed() && is_visible_in_tree()) {
		Vector3 new_pos = _get_cached_global_transform_interpolated().origin;

		// This is to prevent resending on rotations.
		if (!new_pos.is_equal_approx(data.prev_pos)) {
			VisualServer::get_singleton()->blob_shadow_update(data.blob, new_pos, data.radius);
			data.prev_pos = new_pos;
		}
	}

	Spatial::fti_update_servers_xform();
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
			if (data.blob.is_valid() && VisualServerBlobShadows::is_allowed() && !is_physics_interpolated_and_enabled() && is_visible_in_tree()) {
				Vector3 new_pos = get_global_transform().origin;

				// This is to prevent resending on rotations.
				if (!new_pos.is_equal_approx(data.prev_pos)) {
					VisualServer::get_singleton()->blob_shadow_update(data.blob, new_pos, data.radius);
					data.prev_pos = new_pos;
				}
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_refresh_visibility(is_inside_tree());
		} break;
	}
}

void BlobShadow::_refresh_visibility(bool p_in_tree) {
	bool visible = p_in_tree && is_visible_in_tree();
	if (visible) {
		if (!data.blob.is_valid()) {
			data.blob = RID_PRIME(VisualServer::get_singleton()->blob_shadow_create());
			VisualServer::get_singleton()->blob_shadow_update(data.blob, get_global_transform().origin, data.radius);
		}
	} else {
		if (data.blob.is_valid()) {
			VisualServer::get_singleton()->free(data.blob);
			data.blob = RID();
		}
	}
}

void BlobShadow::set_radius(real_t p_radius) {
	if (p_radius == data.radius) {
		return;
	}
	data.radius = p_radius;
	if (data.blob.is_valid()) {
		VisualServer::get_singleton()->blob_shadow_update(data.blob, get_global_transform().origin, data.radius);
	}
}

void BlobShadow::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &BlobShadow::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &BlobShadow::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0,16,0.001,or_greater"), "set_radius", "get_radius");
}

BlobShadow::BlobShadow() {
	set_notify_transform(true);
}

BlobShadow::~BlobShadow() {
	_refresh_visibility(false);
}
