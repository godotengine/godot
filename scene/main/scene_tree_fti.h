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

#pragma once

#include "core/os/mutex.h"
#include "core/templates/local_vector.h"

class Node3D;
class Node;
struct Transform3D;

#ifdef _3D_DISABLED
// Stubs
class SceneTreeFTI {
public:
	void frame_update(Node *p_root, bool p_frame_start) {}
	void tick_update() {}
	void set_enabled(Node *p_root, bool p_enabled) {}
	bool is_enabled() const { return false; }

	void node_3d_notify_changed(Node3D &r_node, bool p_transform_changed) {}
	void node_3d_notify_delete(Node3D *p_node) {}
	void node_3d_request_reset(Node3D *p_node) {}
};
#else

// Important.
// This class uses raw pointers, so it is essential that on deletion, this class is notified
// so that any references can be cleared up to prevent dangling pointer access.

// This class can be used from a custom SceneTree.

// Note we could potentially make SceneTreeFTI static / global to avoid the lookup through scene tree,
// but this covers the custom case of multiple scene trees.

class SceneTreeFTI {
	struct Data {
		// Prev / Curr lists of Node3Ds having local xforms pumped.
		LocalVector<Node3D *> tick_xform_list[2];

		// Prev / Curr lists of Node3Ds having actively interpolated properties.
		LocalVector<Node3D *> tick_property_list[2];

		LocalVector<Node3D *> frame_property_list;

		LocalVector<Node3D *> request_reset_list;

		uint32_t mirror = 0;

		bool enabled = false;

		// Whether we are in physics ticks, or in a frame.
		bool in_frame = false;

		// Updating at the start of the frame, or the end on second pass.
		bool frame_start = true;

		Mutex mutex;

		bool debug = false;
	} data;

	void _update_dirty_nodes(Node *p_node, uint32_t p_current_frame, float p_interpolation_fraction, bool p_active, const Transform3D *p_parent_global_xform = nullptr, int p_depth = 0);
	void _update_request_resets();

	void _reset_flags(Node *p_node);
	void _node_3d_notify_set_xform(Node3D &r_node);
	void _node_3d_notify_set_property(Node3D &r_node);

public:
	// Hottest function, allow inlining the data.enabled check.
	void node_3d_notify_changed(Node3D &r_node, bool p_transform_changed) {
		if (!data.enabled) {
			return;
		}
		MutexLock(data.mutex);

		if (p_transform_changed) {
			_node_3d_notify_set_xform(r_node);
		} else {
			_node_3d_notify_set_property(r_node);
		}
	}

	void node_3d_request_reset(Node3D *p_node);
	void node_3d_notify_delete(Node3D *p_node);

	// Calculate interpolated xforms, send to visual server.
	void frame_update(Node *p_root, bool p_frame_start);

	// Update local xform pumps.
	void tick_update();

	void set_enabled(Node *p_root, bool p_enabled);
	bool is_enabled() const { return data.enabled; }

	void set_debug_next_frame() { data.debug = true; }
};

#endif // ndef _3D_DISABLED
