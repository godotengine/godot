/**************************************************************************/
/*  scene_tree_fti.h                                                      */
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

#ifndef SCENE_TREE_FTI_H
#define SCENE_TREE_FTI_H

#include <core/local_vector.h>
#include <core/os/mutex.h>

class Spatial;
class Node;
class Transform;

#ifdef _3D_DISABLED
// Stubs
class SceneTreeFTI {
public:
	void frame_update(Node *p_root, bool p_frame_start) {}
	void tick_update() {}
	void set_enabled(Node *p_root, bool p_enabled) {}
	bool is_enabled() const { return false; }

	void spatial_notify_changed(Spatial &r_spatial, bool p_transform_changed) {}
	void spatial_request_reset(Spatial *p_spatial) {}
	void spatial_notify_delete(Spatial *p_spatial) {}
};
#else

// Important.
// This class uses raw pointers, so it is essential that on deletion, this class is notified
// so that any references can be cleared up to prevent dangling pointer access.

// This class can be used from a custom SceneTree.

// Note we could potentially make SceneTreeFTI static / global to avoid the lookup through scene tree,
// but this covers the custom case of multiple scene trees.

// This class is not thread safe, but can be made thread safe easily with a mutex as in the 4.x version.

class SceneTreeFTI {
	struct Data {
		// Prev / Curr lists of spatials having local xforms pumped.
		LocalVector<Spatial *> tick_xform_list[2];

		// Prev / Curr lists of spatials having actively interpolated properties.
		LocalVector<Spatial *> tick_property_list[2];

		LocalVector<Spatial *> frame_property_list;

		LocalVector<Spatial *> request_reset_list;

		uint32_t mirror = 0;

		bool enabled = false;

		// Whether we are in physics ticks, or in a frame.
		bool in_frame = false;

		// Updating at the start of the frame, or the end on second pass.
		bool frame_start = true;

		bool debug = false;
	} data;

	void _update_dirty_spatials(Node *p_node, uint32_t p_current_frame, float p_interpolation_fraction, bool p_active, const Transform *p_parent_global_xform = nullptr, int p_depth = 0);
	void _update_request_resets();

	void _reset_flags(Node *p_node);
	void _spatial_notify_set_xform(Spatial &r_spatial);
	void _spatial_notify_set_property(Spatial &r_spatial);

public:
	// Hottest function, allow inlining the data.enabled check.
	void spatial_notify_changed(Spatial &r_spatial, bool p_transform_changed) {
		if (!data.enabled) {
			return;
		}
		if (p_transform_changed) {
			_spatial_notify_set_xform(r_spatial);
		} else {
			_spatial_notify_set_property(r_spatial);
		}
	}

	void spatial_request_reset(Spatial *p_spatial);
	void spatial_notify_delete(Spatial *p_spatial);

	// Calculate interpolated xforms, send to visual server.
	void frame_update(Node *p_root, bool p_frame_start);

	// Update local xform pumps.
	void tick_update();

	void set_enabled(Node *p_root, bool p_enabled);
	bool is_enabled() const { return data.enabled; }

	void set_debug_next_frame() { data.debug = true; }
};

#endif // ndef _3D_DISABLED

#endif // SCENE_TREE_FTI_H
