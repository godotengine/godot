/*************************************************************************/
/*  animation_blend_space_cyclic_1d.cpp                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "animation_blend_space_cyclic_1d.h"
#include "scene/animation/animation_blend_tree.h"

/**
	@author Marios Staikopoulos <marios@staik.net>
*/

const Ref<Animation> AnimationNodeBlendSpaceCyclic1D::get_animation_for_point(const int p_point) {
	AnimationPlayer *ap = state->player;
	ERR_FAIL_COND_V(!ap, nullptr);

	AnimationNodeAnimation *anim_node = Object::cast_to<AnimationNodeAnimation>(blend_points[p_point].node.ptr());

	const StringName anim_name = anim_node->get_animation();
	ERR_FAIL_COND_V(!ap->has_animation(anim_name), nullptr);

	return ap->get_animation(anim_name);
}

void AnimationNodeBlendSpaceCyclic1D::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, cycle, PROPERTY_HINT_NONE, "", 0));

	AnimationNodeBlendSpace1D::get_parameter_list(r_list);
}

Variant AnimationNodeBlendSpaceCyclic1D::get_parameter_default_value(const StringName &p_parameter) const {
	if (p_parameter == cycle) {
		return 0.0;
	}

	return AnimationNodeBlendSpace1D::get_parameter_default_value(p_parameter);
}

void AnimationNodeBlendSpaceCyclic1D::add_blend_point(const Ref<AnimationRootNode> &p_node, float p_position, int p_at_index) {
	ERR_FAIL_COND(Object::cast_to<AnimationNodeAnimation>(p_node.ptr()) == nullptr);
	AnimationNodeBlendSpace1D::add_blend_point(p_node, p_position, p_at_index);
}

void AnimationNodeBlendSpaceCyclic1D::set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node) {
	ERR_FAIL_COND(Object::cast_to<AnimationNodeAnimation>(p_node.ptr()) == nullptr);
	AnimationNodeBlendSpace1D::set_blend_point_node(p_point, p_node);
}

float AnimationNodeBlendSpaceCyclic1D::process(float p_time, bool p_seek) {
	if (blend_points_used == 0) {
		return 0.0;
	}

	float cycle = get_parameter(this->cycle);

	if (blend_points_used == 1) {
		Ref<Animation> anim = get_animation_for_point(0);
		const float length = anim->get_length();

		if (p_seek) {
			cycle = Math::fposmod(p_time, length) / length;
		} else {
			cycle = Math::fposmod(cycle * length + p_time, length) / length;
		}

		// Only one point available, just play that animation
		blend_node(blend_points[0].name, blend_points[0].node, cycle * length, true, 1.0, FILTER_IGNORE, false);
	}

	// We have points to blend!
	else {
		const float blend_pos = get_parameter(blend_position);

		const BlendWeights b_values = get_blend_values(blend_pos);

		float total_runtime = 0;
		float runtimes[2] = { 0 };

		for (int i = 0; i < 2; ++i) {
			// Having a weight means the animation definitely exists...
			if (b_values.weights[i] > 0.0) {
				Ref<Animation> anim = get_animation_for_point(b_values.points[i]);
				runtimes[i] = anim->get_length();
				total_runtime += b_values.weights[i] * runtimes[i];
			}
		}

		if (p_seek) {
			cycle = Math::fposmod(p_time, total_runtime) / total_runtime;
		} else {
			cycle = Math::fposmod(cycle * total_runtime + p_time, total_runtime) / total_runtime;
		}

		for (int i = 0; i < 2; i++) {
			if (b_values.weights[i] > 0.0) {
				const BlendPoint &blend_point = blend_points[b_values.points[i]];
				blend_node(blend_point.name, blend_point.node, cycle * runtimes[i], true, b_values.weights[i], FILTER_IGNORE, false);
			}
		}
	}

	set_parameter(this->cycle, cycle);
	return 1 - cycle;
}

String AnimationNodeBlendSpaceCyclic1D::get_caption() const {
	return "BlendSpaceCyclic1D";
}
