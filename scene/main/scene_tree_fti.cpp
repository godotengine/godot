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
#include "core/engine.h"
#include "core/error_macros.h"
#include "core/math/transform_interpolator.h"
#include "core/os/os.h"
#include "scene/3d/spatial.h"
#include "scene/3d/visual_instance.h"

void SceneTreeFTI::_reset_flags(Node *p_node) {
	Spatial *s = Object::cast_to<Spatial>(p_node);

	if (s) {
		s->data.fti_on_frame_list = false;
		s->data.fti_on_tick_list = false;

		// In most cases the later  NOTIFICATION_RESET_PHYSICS_INTERPOLATION
		// will reset this, but this should help cover hidden nodes.
		s->data.local_transform_prev = s->data.local_transform;
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		_reset_flags(p_node->get_child(n));
	}
}

void SceneTreeFTI::set_enabled(Node *p_root, bool p_enabled) {
	if (data.enabled == p_enabled) {
		return;
	}

	data.spatial_tick_list[0].clear();
	data.spatial_tick_list[1].clear();

	// Spatial flags must be reset.
	if (p_root) {
		_reset_flags(p_root);
	}

	data.enabled = p_enabled;
}

void SceneTreeFTI::tick_update() {
	if (!data.enabled) {
		return;
	}

	uint32_t curr_mirror = data.mirror;
	uint32_t prev_mirror = curr_mirror ? 0 : 1;

	LocalVector<Spatial *> &curr = data.spatial_tick_list[curr_mirror];
	LocalVector<Spatial *> &prev = data.spatial_tick_list[prev_mirror];

	// First detect on the previous list but not on this tick list.
	for (uint32_t n = 0; n < prev.size(); n++) {
		Spatial *s = prev[n];
		if (!s->data.fti_on_tick_list) {
			// Needs a reset so jittering will stop.
			s->fti_pump();

			// This may not get updated so set it to the same as global xform.
			// TODO: double check this is the best value.
			s->data.global_transform_interpolated = s->get_global_transform();

			// Remove from interpolation list.
			if (s->data.fti_on_frame_list) {
				s->data.fti_on_frame_list = false;
			}
		}
	}

	// Now pump all on the current list.
	for (uint32_t n = 0; n < curr.size(); n++) {
		Spatial *s = curr[n];

		// Reset, needs to be marked each tick.
		s->data.fti_on_tick_list = false;

		// Pump.
		s->fti_pump();
	}

	// Clear previous list and flip.
	prev.clear();
	data.mirror = prev_mirror;
}

void SceneTreeFTI::_spatial_notify_set_transform(Spatial &r_spatial) {
	// This may be checked by the calling routine already,
	// but needs to be double checked for custom SceneTrees.
	if (!data.enabled || !r_spatial.is_physics_interpolated()) {
		return;
	}

	if (!r_spatial.data.fti_on_tick_list) {
		r_spatial.data.fti_on_tick_list = true;
		data.spatial_tick_list[data.mirror].push_back(&r_spatial);
	}

	if (!r_spatial.data.fti_on_frame_list) {
		r_spatial.data.fti_on_frame_list = true;
	}
}

void SceneTreeFTI::spatial_notify_delete(Spatial *p_spatial) {
	if (!data.enabled) {
		return;
	}

	if (p_spatial->data.fti_on_frame_list) {
		p_spatial->data.fti_on_frame_list = false;
	}

	// This can potentially be optimized for large scenes with large churn,
	// as it will be doing a linear search through the lists.
	data.spatial_tick_list[0].erase_unordered(p_spatial);
	data.spatial_tick_list[1].erase_unordered(p_spatial);
}

void SceneTreeFTI::_update_dirty_spatials(Node *p_node, uint32_t p_current_frame, float p_interpolation_fraction, bool p_active, const Transform *p_parent_global_xform, int p_depth) {
	Spatial *s = Object::cast_to<Spatial>(p_node);

	// Don't recurse into hidden branches.
	if (s && !s->is_visible()) {
		// NOTE : If we change from recursing entire tree, we should do an is_visible_in_tree()
		// check for the first of the branch.
		return;
	}

	// Not a Spatial.
	// Could be e.g. a viewport or something
	// so we should still recurse to children.
	if (!s) {
		for (int n = 0; n < p_node->get_child_count(); n++) {
			_update_dirty_spatials(p_node->get_child(n), p_current_frame, p_interpolation_fraction, p_active, nullptr, p_depth + 1);
		}
		return;
	}

	// Start the active interpolation chain from here onwards
	// as we recurse further into the SceneTree.
	// Once we hit an active (interpolated) node, we have to fully
	// process all ancestors because their xform will also change.
	// Anything not moving (inactive) higher in the tree need not be processed.
	if (!p_active) {
		if (data.frame_start) {
			// On the frame start, activate whenever we hit something that requests interpolation.
			if (s->data.fti_on_frame_list) {
				p_active = true;
			}
		} else {
			// On the frame end, we want to re-interpolate *anything* that has moved
			// since the frame start.
			if (s->data.dirty & Spatial::DIRTY_GLOBAL_INTERPOLATED) {
				p_active = true;
			}
		}
	}

	if (data.frame_start) {
		// Mark on the Spatial whether we have set global_transform_interp.
		// This can later be used when calling `get_global_transform_interpolated()`
		// to know which xform to return.
		s->data.fti_global_xform_interp_set = p_active;
	}

	if (p_active) {
#if 0
		bool dirty = s->data.dirty & Spatial::DIRTY_GLOBAL_INTERP;

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
		Transform local_interp;
		if (s->is_physics_interpolated()) {
			TransformInterpolator::interpolate_transform(s->data.local_transform_prev, s->data.local_transform, local_interp, p_interpolation_fraction);
		} else {
			local_interp = s->data.local_transform;
		}

		// Concatenate parent xform.
		if (!s->is_set_as_toplevel()) {
			if (p_parent_global_xform) {
				s->data.global_transform_interpolated = (*p_parent_global_xform) * local_interp;
			} else {
				const Spatial *parent = s->get_parent_spatial();

				if (parent) {
					const Transform &parent_glob = parent->data.fti_global_xform_interp_set ? parent->data.global_transform_interpolated : parent->data.global_transform;
					s->data.global_transform_interpolated = parent_glob * local_interp;
				} else {
					s->data.global_transform_interpolated = local_interp;
				}
			}
		} else {
			s->data.global_transform_interpolated = local_interp;
		}

		// Upload to VisualServer the interpolated global xform.
		s->fti_update_servers();

	} // if active.

	// Remove the dirty interp flag from EVERYTHING as we go.
	s->data.dirty &= ~Spatial::DIRTY_GLOBAL_INTERPOLATED;

	// Recurse to children.
	for (int n = 0; n < p_node->get_child_count(); n++) {
		_update_dirty_spatials(p_node->get_child(n), p_current_frame, p_interpolation_fraction, p_active, s->data.fti_global_xform_interp_set ? &s->data.global_transform_interpolated : &s->data.global_transform, p_depth + 1);
	}
}

void SceneTreeFTI::frame_update(Node *p_root, bool p_frame_start) {
	if (!data.enabled || !p_root) {
		return;
	}

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
	_update_dirty_spatials(p_root, frame, f, false);

	if (!p_frame_start && data.debug) {
		data.debug = false;
	}

#ifdef SCENE_TREE_FTI_TAKE_TIMINGS
	uint64_t after = OS::get_singleton()->get_ticks_usec();
	if ((Engine::get_singleton()->get_frames_drawn() % 60) == 0) {
		print_line("Took " + itos(after - before) + " usec " + (data.frame_start ? "start" : "end"));
	}
#endif
}

#endif // ndef _3D_DISABLED
