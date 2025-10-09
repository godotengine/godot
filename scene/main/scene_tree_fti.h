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
class SceneTreeFTITests;

#ifdef DEV_ENABLED
// Uncomment this to verify traversal method results.
// #define GODOT_SCENE_TREE_FTI_VERIFY
#endif

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
	friend class SceneTreeFTITests;

	enum TraversalMode : unsigned {
		TM_DEFAULT,
		TM_LEGACY,
		TM_DEBUG,
	};

	struct Data {
		static const uint32_t scene_tree_depth_limit = 48;

		// Prev / Curr lists of spatials having local xforms pumped.
		LocalVector<Spatial *> tick_xform_list[2];

		// The frame lists are changed nodes that need to start traversal,
		// either longterm (on the tick list) or single frame forced.
		LocalVector<Spatial *> frame_xform_list;
		LocalVector<Spatial *> frame_xform_list_forced;

		// Prev / Curr lists of spatials having actively interpolated properties.
		LocalVector<Spatial *> tick_property_list[2];

		LocalVector<Spatial *> frame_property_list;
		LocalVector<Spatial *> request_reset_list;
		LocalVector<Spatial *> dirty_spatial_depth_lists[scene_tree_depth_limit];

		// When we are using two alternating lists,
		// which one is current.
		uint32_t mirror = 0;

		// Global on / off switch for SceneTreeFTI.
		bool enabled = false;

		// Whether we are in physics ticks, or in a frame.
		bool in_frame = false;

		// Updating at the start of the frame, or the end on second pass.
		bool frame_start = true;

		TraversalMode traversal_mode = TM_DEFAULT;
		bool use_optimized_traversal_method = true;

		// DEBUGGING
		bool periodic_debug_log = false;
		uint32_t debug_node_count = 0;
		uint32_t debug_nodes_processed = 0;

	} data;

#ifdef GODOT_SCENE_TREE_FTI_VERIFY
	SceneTreeFTITests *_tests = nullptr;
#endif

	void _update_dirty_spatials(Node *p_node, uint32_t p_current_half_frame, float p_interpolation_fraction, bool p_active, const Transform *p_parent_global_xform = nullptr, int p_depth = 0);
	void _update_request_resets();

	void _reset_flags(Node *p_node);
	void _reset_spatial_flags(Spatial &r_spatial);
	void _spatial_notify_set_xform(Spatial &r_spatial);
	void _spatial_notify_set_property(Spatial &r_spatial);

	void _spatial_add_to_frame_list(Spatial &r_spatial, bool p_forced);
	void _spatial_remove_from_frame_list(Spatial &r_spatial, bool p_forced);

	void _create_depth_lists();
	void _clear_depth_lists();

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

	void set_debug_next_frame() { data.periodic_debug_log = true; }

	SceneTreeFTI();
	~SceneTreeFTI();
};

#endif // ndef _3D_DISABLED

#endif // SCENE_TREE_FTI_H
