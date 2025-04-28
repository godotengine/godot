/**************************************************************************/
/*  scene_tree_fti.cpp                                                    */
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

#ifndef _3D_DISABLED

#include "scene_tree_fti.h"

#include "core/config/engine.h"
#include "core/math/transform_interpolator.h"
#include "core/os/os.h"
#include "scene/3d/visual_instance_3d.h"

// Uncomment this to enable some slow extra DEV_ENABLED
// checks to ensure there aren't more than one object added to the lists.
// #define GODOT_SCENE_TREE_FTI_EXTRA_CHECKS

void SceneTreeFTI::_reset_flags(Node *p_node) {
	Node3D *s = Object::cast_to<Node3D>(p_node);

	if (s) {
		s->data.fti_on_tick_xform_list = false;
		s->data.fti_on_tick_property_list = false;
		s->data.fti_on_frame_xform_list = false;
		s->data.fti_on_frame_property_list = false;
		s->data.fti_global_xform_interp_set = false;
		s->data.fti_frame_xform_force_update = false;

		// In most cases the later  NOTIFICATION_RESET_PHYSICS_INTERPOLATION
		// will reset this, but this should help cover hidden nodes.
		s->data.local_transform_prev = s->get_transform();
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		_reset_flags(p_node->get_child(n));
	}
}

void SceneTreeFTI::set_enabled(Node *p_root, bool p_enabled) {
	if (data.enabled == p_enabled) {
		return;
	}
	MutexLock(data.mutex);

	data.tick_xform_list[0].clear();
	data.tick_xform_list[1].clear();

	// Node3D flags must be reset.
	if (p_root) {
		_reset_flags(p_root);
	}

	data.enabled = p_enabled;
}

void SceneTreeFTI::tick_update() {
	if (!data.enabled) {
		return;
	}
	MutexLock(data.mutex);

	_update_request_resets();

	uint32_t curr_mirror = data.mirror;
	uint32_t prev_mirror = curr_mirror ? 0 : 1;

	LocalVector<Node3D *> &curr = data.tick_xform_list[curr_mirror];
	LocalVector<Node3D *> &prev = data.tick_xform_list[prev_mirror];

	// First detect on the previous list but not on this tick list.
	for (uint32_t n = 0; n < prev.size(); n++) {
		Node3D *s = prev[n];
		if (!s->data.fti_on_tick_xform_list) {
			// Needs a reset so jittering will stop.
			s->fti_pump_xform();

			// This may not get updated so set it to the same as global xform.
			// TODO: double check this is the best value.
			s->data.global_transform_interpolated = s->get_global_transform();

			// Remove from interpolation list.
			if (s->data.fti_on_frame_xform_list) {
				s->data.fti_on_frame_xform_list = false;
			}

			// Ensure that the spatial gets at least ONE further
			// update in the resting position in the next frame update.
			s->data.fti_frame_xform_force_update = true;
		}
	}

	LocalVector<Node3D *> &curr_prop = data.tick_property_list[curr_mirror];
	LocalVector<Node3D *> &prev_prop = data.tick_property_list[prev_mirror];

	// Detect on the previous property list but not on this tick list.
	for (uint32_t n = 0; n < prev_prop.size(); n++) {
		Node3D *s = prev_prop[n];

		if (!s->data.fti_on_tick_property_list) {
			// Needs a reset so jittering will stop.
			s->fti_pump_xform();

			// Ensure the servers are up to date with the final resting value.
			s->fti_update_servers_property();

			// Remove from interpolation list.
			if (s->data.fti_on_frame_property_list) {
				s->data.fti_on_frame_property_list = false;
				data.frame_property_list.erase_unordered(s);

#ifdef GODOT_SCENE_TREE_FTI_EXTRA_CHECKS
				DEV_CHECK_ONCE(data.frame_property_list.find(s) == -1);
#endif
			}
		}
	}

	// Pump all on the property list that are NOT on the tick list.
	for (uint32_t n = 0; n < curr_prop.size(); n++) {
		Node3D *s = curr_prop[n];

		// Reset, needs to be marked each tick.
		s->data.fti_on_tick_property_list = false;
		s->fti_pump_property();
	}

	// Now pump all on the current list.
	for (uint32_t n = 0; n < curr.size(); n++) {
		Node3D *s = curr[n];

		// Reset, needs to be marked each tick.
		s->data.fti_on_tick_xform_list = false;

		// Pump.
		s->fti_pump_xform();
	}

	// Clear previous list and flip.
	prev.clear();
	prev_prop.clear();
	data.mirror = prev_mirror;
}

void SceneTreeFTI::_update_request_resets() {
	// For instance when first adding to the tree, when the previous transform is
	// unset, to prevent streaking from the origin.
	for (uint32_t n = 0; n < data.request_reset_list.size(); n++) {
		Node3D *s = data.request_reset_list[n];
		if (s->_is_physics_interpolation_reset_requested()) {
			if (s->_is_vi_visible() && !s->_is_using_identity_transform()) {
				s->notification(Node3D::NOTIFICATION_RESET_PHYSICS_INTERPOLATION);
			}

			s->_set_physics_interpolation_reset_requested(false);
		}
	}

	data.request_reset_list.clear();
}

void SceneTreeFTI::node_3d_request_reset(Node3D *p_node) {
	DEV_CHECK_ONCE(data.enabled);
	DEV_ASSERT(p_node);

	MutexLock(data.mutex);

	if (!p_node->_is_physics_interpolation_reset_requested()) {
		p_node->_set_physics_interpolation_reset_requested(true);
#ifdef GODOT_SCENE_TREE_FTI_EXTRA_CHECKS
		DEV_CHECK_ONCE(data.request_reset_list.find(p_node) == -1);
#endif
		data.request_reset_list.push_back(p_node);
	}
}

void SceneTreeFTI::_node_3d_notify_set_property(Node3D &r_node) {
	if (!r_node.is_physics_interpolated()) {
		return;
	}

	DEV_CHECK_ONCE(data.enabled);

	// Note that a Node3D can be on BOTH the transform list and the property list.
	if (!r_node.data.fti_on_tick_property_list) {
		r_node.data.fti_on_tick_property_list = true;

		// Should only appear once in the property list.
#ifdef GODOT_SCENE_TREE_FTI_EXTRA_CHECKS
		DEV_CHECK_ONCE(data.tick_property_list[data.mirror].find(&r_node) == -1);
#endif
		data.tick_property_list[data.mirror].push_back(&r_node);
	}

	if (!r_node.data.fti_on_frame_property_list) {
		r_node.data.fti_on_frame_property_list = true;

		// Should only appear once in the property frame list.
#ifdef GODOT_SCENE_TREE_FTI_EXTRA_CHECKS
		DEV_CHECK_ONCE(data.frame_property_list.find(&r_node) == -1);
#endif
		data.frame_property_list.push_back(&r_node);
	}
}

void SceneTreeFTI::_node_3d_notify_set_xform(Node3D &r_node) {
	DEV_CHECK_ONCE(data.enabled);

	if (!r_node.is_physics_interpolated()) {
		// Force an update of non-interpolated to servers
		// on the next traversal.
		r_node.data.fti_frame_xform_force_update = true;
		return;
	}

	if (!r_node.data.fti_on_tick_xform_list) {
		r_node.data.fti_on_tick_xform_list = true;

		// Should only appear once in the xform list.
#ifdef GODOT_SCENE_TREE_FTI_EXTRA_CHECKS
		DEV_CHECK_ONCE(data.tick_xform_list[data.mirror].find(&r_node) == -1);
#endif
		data.tick_xform_list[data.mirror].push_back(&r_node);

		// The following flag could have been previously set
		// (for removal from the tick list).
		// We no longer need this guarantee,
		// however there is probably no downside to leaving it set
		// as it will be cleared on the next frame anyway.
		// This line is left for reference.
		// r_spatial.data.fti_frame_xform_force_update = false;
	}

	if (!r_node.data.fti_on_frame_xform_list) {
		r_node.data.fti_on_frame_xform_list = true;
	}
}

void SceneTreeFTI::node_3d_notify_delete(Node3D *p_node) {
	if (!data.enabled) {
		return;
	}

	ERR_FAIL_NULL(p_node);

	MutexLock(data.mutex);

	p_node->data.fti_on_frame_xform_list = false;

	// Ensure this is kept in sync with the lists, in case a node
	// is removed and re-added to the scene tree multiple times
	// on the same frame / tick.
	p_node->_set_physics_interpolation_reset_requested(false);

	// This can potentially be optimized for large scenes with large churn,
	// as it will be doing a linear search through the lists.
	data.tick_xform_list[0].erase_unordered(p_node);
	data.tick_xform_list[1].erase_unordered(p_node);

	data.tick_property_list[0].erase_unordered(p_node);
	data.tick_property_list[1].erase_unordered(p_node);

	data.frame_property_list.erase_unordered(p_node);
	data.request_reset_list.erase_unordered(p_node);

#ifdef GODOT_SCENE_TREE_FTI_EXTRA_CHECKS
	// There should only be one occurrence on the lists.
	// Check this in DEV_ENABLED builds.
	DEV_CHECK_ONCE(data.tick_xform_list[0].find(p_node) == -1);
	DEV_CHECK_ONCE(data.tick_xform_list[1].find(p_node) == -1);

	DEV_CHECK_ONCE(data.tick_property_list[0].find(p_node) == -1);
	DEV_CHECK_ONCE(data.tick_property_list[1].find(p_node) == -1);

	DEV_CHECK_ONCE(data.frame_property_list.find(p_node) == -1);
	DEV_CHECK_ONCE(data.request_reset_list.find(p_node) == -1);
#endif
}

void SceneTreeFTI::_update_dirty_nodes(Node *p_node, uint32_t p_current_frame, float p_interpolation_fraction, bool p_active, const Transform3D *p_parent_global_xform, int p_depth) {
	Node3D *s = Object::cast_to<Node3D>(p_node);

	// Don't recurse into hidden branches.
	if (s && !s->is_visible()) {
		// NOTE : If we change from recursing entire tree, we should do an is_visible_in_tree()
		// check for the first of the branch.
		return;
	}

	// Not a Node3D.
	// Could be e.g. a viewport or something
	// so we should still recurse to children.
	if (!s) {
		for (int n = 0; n < p_node->get_child_count(); n++) {
			_update_dirty_nodes(p_node->get_child(n), p_current_frame, p_interpolation_fraction, p_active, nullptr, p_depth + 1);
		}
		return;
	}

	// We are going to be using data.global_transform, so
	// we need to ensure data.global_transform is not dirty!
	if (s->_test_dirty_bits(Node3D::DIRTY_GLOBAL_TRANSFORM)) {
		_ALLOW_DISCARD_ s->get_global_transform();
	}

	// Start the active interpolation chain from here onwards
	// as we recurse further into the SceneTree.
	// Once we hit an active (interpolated) node, we have to fully
	// process all ancestors because their xform will also change.
	// Anything not moving (inactive) higher in the tree need not be processed.
	if (!p_active) {
		if (data.frame_start) {
			// On the frame start, activate whenever we hit something that requests interpolation.
			if (s->data.fti_on_frame_xform_list || s->data.fti_frame_xform_force_update) {
				p_active = true;
			}
		} else {
			// On the frame end, we want to re-interpolate *anything* that has moved
			// since the frame start.

			if (s->_test_dirty_bits(Node3D::DIRTY_GLOBAL_INTERPOLATED_TRANSFORM)) {
				p_active = true;
			}
		}
	}

	if (data.frame_start) {
		// Mark on the Node3D whether we have set global_transform_interp.
		// This can later be used when calling `get_global_transform_interpolated()`
		// to know which xform to return.
		s->data.fti_global_xform_interp_set = p_active;
	}

	if (p_active) {
#if 0
		bool dirty = s->data.dirty & Node3D::DIRTY_GLOBAL_INTERP;

		if (data.debug) {
			String sz;
			for (int n = 0; n < p_depth; n++) {
				sz += "\t";
			}
			print_line(sz + p_node->get_name() + (dirty ? " DIRTY" : ""));
		}
#endif

		// First calculate our local xform.
		// This will either use interpolation, or just use the current local if not interpolated.
		Transform3D local_interp;
		if (s->is_physics_interpolated()) {
			// Make sure to call `get_transform()` rather than using local_transform directly, because
			// local_transform may be dirty and need updating from rotation / scale.
			TransformInterpolator::interpolate_transform_3d(s->data.local_transform_prev, s->get_transform(), local_interp, p_interpolation_fraction);
		} else {
			local_interp = s->get_transform();
		}

		// Concatenate parent xform.
		if (!s->is_set_as_top_level()) {
			if (p_parent_global_xform) {
				s->data.global_transform_interpolated = (*p_parent_global_xform) * local_interp;
			} else {
				const Node3D *parent = s->get_parent_node_3d();

				if (parent) {
					const Transform3D &parent_glob = parent->data.fti_global_xform_interp_set ? parent->data.global_transform_interpolated : parent->data.global_transform;
					s->data.global_transform_interpolated = parent_glob * local_interp;
				} else {
					s->data.global_transform_interpolated = local_interp;
				}
			}
		} else {
			s->data.global_transform_interpolated = local_interp;
		}

		// Watch for this, disable_scale can cause incredibly confusing bugs
		// and must be checked for when calculating global xforms.
		if (s->data.disable_scale) {
			s->data.global_transform_interpolated.basis.orthonormalize();
		}

		// Upload to RenderingServer the interpolated global xform.
		s->fti_update_servers_xform();

		// Only do this at most for one frame,
		// it is used to catch objects being removed from the tick lists
		// that have a deferred frame update.
		s->data.fti_frame_xform_force_update = false;

	} // if active.

	// Remove the dirty interp flag from EVERYTHING as we go.
	s->_clear_dirty_bits(Node3D::DIRTY_GLOBAL_INTERPOLATED_TRANSFORM);

	// Recurse to children.
	for (int n = 0; n < p_node->get_child_count(); n++) {
		_update_dirty_nodes(p_node->get_child(n), p_current_frame, p_interpolation_fraction, p_active, s->data.fti_global_xform_interp_set ? &s->data.global_transform_interpolated : &s->data.global_transform, p_depth + 1);
	}
}

void SceneTreeFTI::frame_update(Node *p_root, bool p_frame_start) {
	if (!data.enabled || !p_root) {
		return;
	}
	MutexLock(data.mutex);

	_update_request_resets();

	data.frame_start = p_frame_start;

	float f = Engine::get_singleton()->get_physics_interpolation_fraction();
	uint32_t frame = Engine::get_singleton()->get_frames_drawn();

// #define SCENE_TREE_FTI_TAKE_TIMINGS
#ifdef SCENE_TREE_FTI_TAKE_TIMINGS
	uint64_t before = OS::get_singleton()->get_ticks_usec();
#endif

	if (data.debug) {
		print_line(String("\nScene: ") + (data.frame_start ? "start" : "end") + "\n");
	}

	// Probably not the most optimal approach as we traverse the entire SceneTree
	// but simple and foolproof.
	// Can be optimized later.
	_update_dirty_nodes(p_root, frame, f, false);

	if (!p_frame_start && data.debug) {
		data.debug = false;
	}

#ifdef SCENE_TREE_FTI_TAKE_TIMINGS
	uint64_t after = OS::get_singleton()->get_ticks_usec();
	if ((Engine::get_singleton()->get_frames_drawn() % 60) == 0) {
		print_line("Took " + itos(after - before) + " usec " + (data.frame_start ? "start" : "end"));
	}
#endif

	// Update the properties once off at the end of the frame.
	// No need for two passes for properties.
	if (!p_frame_start) {
		for (uint32_t n = 0; n < data.frame_property_list.size(); n++) {
			Node3D *s = data.frame_property_list[n];
			s->fti_update_servers_property();
		}
	}
}

#endif // ndef _3D_DISABLED
