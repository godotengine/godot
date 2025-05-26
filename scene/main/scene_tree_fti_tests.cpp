/**************************************************************************/
/*  scene_tree_fti_tests.cpp                                              */
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

#ifdef GODOT_SCENE_TREE_FTI_VERIFY
#include "scene_tree_fti_tests.h"

#include "scene/3d/spatial.h"
#include "scene/3d/visual_instance.h"
#include "scene/main/scene_tree_fti.h"

void SceneTreeFTITests::debug_verify_failed(const Spatial *p_spatial, const Transform &p_test) {
	print_line("VERIFY FAILED\n");
	print_line("test xform : " + String(Variant(p_test)));

	bool first = true;

	while (p_spatial) {
		int32_t depth = MAX(p_spatial->_get_scene_tree_depth(), 0);
		String tabs;
		for (int32_t n = 0; n < depth; n++) {
			tabs += "\t";
		}

		bool interp_equal = p_spatial->_get_cached_global_transform_interpolated() == p_test;
		bool glob_equal = p_spatial->get_global_transform() == p_test;

		String sz = tabs + p_spatial->get_name() + " [ " + p_spatial->get_class_name() + " ]\n";

		if (first) {
			sz += tabs + "... " + String(Variant(p_test)) + "\n";
		}

		sz += tabs + (p_spatial->data.fti_global_xform_interp_set ? "[I] " : "[i] ") + String(Variant(p_spatial->_get_cached_global_transform_interpolated())) + (interp_equal ? " ***" : "") + "\n";
		sz += tabs + "[g] " + String(Variant(p_spatial->get_global_transform())) + (glob_equal ? " ***" : "");

		print_line(sz);

		p_spatial = p_spatial->get_parent_spatial();
		first = false;
	}
}

void SceneTreeFTITests::update_dirty_spatials(Node *p_node, uint32_t p_current_half_frame, float p_interpolation_fraction, bool p_active, const Transform *p_parent_global_xform, int p_depth) {
	SceneTreeFTI::Data &data = _fti.data;

	// There are two runs going on here.
	// FIRST the naive entire scene tree (reference), where we are
	// setting state (i.e. writing out xforms, and other state)
	// SECOND the optimized run, where we are NOT
	// writing state, but only verifying that the xforms calculated
	// match those from the reference approach.
	bool should_verify = (data.traversal_mode == SceneTreeFTI::TM_DEBUG) && data.use_optimized_traversal_method;
	bool set_state = !should_verify;

	Spatial *s = Object::cast_to<Spatial>(p_node);

	if (s && !s->is_visible()) {
		return;
	}

	if (!s) {
		for (int n = 0; n < p_node->get_child_count(); n++) {
			update_dirty_spatials(p_node->get_child(n), p_current_half_frame, p_interpolation_fraction, p_active, nullptr, p_depth + 1);
		}
		return;
	}

	if (s->data.dirty & Spatial::DIRTY_GLOBAL) {
		_ALLOW_DISCARD_ s->get_global_transform();
	}

	if (!p_active) {
		if (data.frame_start) {
			if (s->data.fti_on_frame_xform_list || s->data.fti_frame_xform_force_update) {
				p_active = true;
			}
		} else {
			if (s->data.dirty & Spatial::DIRTY_GLOBAL_INTERPOLATED) {
				p_active = true;
			}
		}
	}

	if (data.frame_start) {
		s->data.fti_global_xform_interp_set = p_active;
	}

	if (p_active) {
		Transform local_interp;
		if (s->is_physics_interpolated()) {
			if (s->data.fti_on_tick_xform_list) {
				TransformInterpolator::interpolate_transform(s->data.local_transform_prev, s->get_transform(), local_interp, p_interpolation_fraction);
			} else {
				local_interp = s->get_transform();
			}
		} else {
			local_interp = s->get_transform();
		}

		if (!s->is_set_as_toplevel()) {
			if (p_parent_global_xform) {
				if (should_verify) {
					Transform test = (*p_parent_global_xform) * local_interp;
					if (s->data.disable_scale) {
						test.basis.orthonormalize();
					}
					if (s->data.global_transform_interpolated != test) {
						debug_verify_failed(s, test);
						DEV_ASSERT(s->data.global_transform_interpolated == test);
					}
				} else {
					s->data.global_transform_interpolated = s->data.fti_is_identity_xform ? (*p_parent_global_xform) : (*p_parent_global_xform) * local_interp;
				}
			} else {
				const Spatial *parent = s->get_parent_spatial();

				if (parent) {
					const Transform &parent_glob = parent->data.fti_global_xform_interp_set ? parent->data.global_transform_interpolated : parent->get_global_transform();

					if (should_verify) {
						Transform test = parent_glob * local_interp;
						if (s->data.disable_scale) {
							test.basis.orthonormalize();
						}
						if (s->data.global_transform_interpolated != test) {
							debug_verify_failed(s, test);
							DEV_ASSERT(s->data.global_transform_interpolated == test);
						}

					} else {
						s->data.global_transform_interpolated = s->data.fti_is_identity_xform ? parent_glob : parent_glob * local_interp;
					}
				} else {
					if (set_state) {
						s->data.global_transform_interpolated = local_interp;
					}
				}
			}
		} else {
			if (set_state) {
				s->data.global_transform_interpolated = local_interp;
			}
		}

		if (set_state) {
			if (s->data.disable_scale) {
				s->data.global_transform_interpolated.basis.orthonormalize();
			}

			s->fti_update_servers_xform();

			s->data.fti_frame_xform_force_update = false;
		}

		s->data.fti_processed = true;
	} // if active.

	if (set_state) {
		s->data.dirty &= ~Spatial::DIRTY_GLOBAL_INTERPOLATED;
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		update_dirty_spatials(p_node->get_child(n), p_current_half_frame, p_interpolation_fraction, p_active, s->data.fti_global_xform_interp_set ? &s->data.global_transform_interpolated : &s->data.global_transform, p_depth + 1);
	}
}

void SceneTreeFTITests::frame_update(Node *p_root, uint32_t p_half_frame, float p_interpolation_fraction) {
	SceneTreeFTI::Data &data = _fti.data;

	// For testing, use both methods.
	// FIRST the entire tree, writing out state.
	{
		data.use_optimized_traversal_method = false;
		update_dirty_spatials(p_root, p_half_frame, p_interpolation_fraction, false);
	}

	// SECOND the optimized depth lists only,
	// no writing of state, and verifying results.
	{
		data.use_optimized_traversal_method = true;

		_fti._create_depth_lists();

		for (uint32_t d = 0; d < data.scene_tree_depth_limit; d++) {
			const LocalVector<Spatial *> &list = data.dirty_spatial_depth_lists[d];

			for (uint32_t n = 0; n < list.size(); n++) {
				Spatial *s = list[n];

				if (s->data.fti_processed) {
					continue;
				}

				if (Object::cast_to<VisualInstance>(s)) {
					if (!s->_is_vi_visible()) {
						continue;
					}
				} else if (!s->is_visible_in_tree()) {
					continue;
				}

				update_dirty_spatials(s, p_half_frame, p_interpolation_fraction, true);
			}
		}

		_fti._clear_depth_lists();
	}
}

SceneTreeFTITests::SceneTreeFTITests(SceneTreeFTI &p_fti) :
		_fti(p_fti) {
	print_line("SceneTreeFTI : GODOT_SCENE_TREE_FTI_VERIFY defined");
}

#endif // def GODOT_SCENE_TREE_FTI_VERIFY

#endif // ndef _3D_DISABLED
