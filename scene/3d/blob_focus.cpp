/**************************************************************************/
/*  blob_focus.cpp                                                        */
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

#include "blob_focus.h"

#include "scene/3d/camera.h"
#include "servers/visual/visual_server_blob_shadows.h"

void BlobFocus::_bind_methods() {
}

void BlobFocus::_physics_interpolated_changed() {
	set_notify_transform(!is_physics_interpolated_and_enabled());
	Spatial::_physics_interpolated_changed();
}

void BlobFocus::fti_update_servers_xform() {
	if (is_visible_in_tree()) {
		Viewport *viewport = get_viewport();
		if (viewport) {
			Camera *camera = viewport->get_camera();
			if (camera) {
				Vector3 new_pos = _get_cached_global_transform_interpolated().origin;

				if (!new_pos.is_equal_approx(_prev_pos)) {
					RID rid = camera->get_camera();
					VisualServer::get_singleton()->camera_set_blob_focus_position(rid, new_pos);
					_prev_pos = new_pos;
				}
			}
		}
	}

	Spatial::fti_update_servers_xform();
}

void BlobFocus::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible_in_tree()) {
				Viewport *viewport = get_viewport();
				if (viewport) {
					Camera *camera = viewport->get_camera();
					if (camera) {
						RID rid = camera->get_camera();
						VisualServer::get_singleton()->camera_set_blob_focus_position(rid, get_global_transform_interpolated().origin);
					}
				}
			}
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (is_visible_in_tree() && !is_physics_interpolated_and_enabled()) {
				Viewport *viewport = get_viewport();
				if (viewport) {
					Camera *camera = viewport->get_camera();
					if (camera) {
						RID rid = camera->get_camera();
						VisualServer::get_singleton()->camera_set_blob_focus_position(rid, get_global_transform().origin);
					}
				}
			}
		} break;
	}
}

BlobFocus::BlobFocus() {
	set_notify_transform(true);
}
