/*************************************************************************/
/*  animation_blend_space_cyclic_2d.cpp                                  */
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

#include "animation_blend_space_cyclic_2d.h"
#include "scene/animation/animation_blend_tree.h"

/**
	@author Marios Staikopoulos <marios@staik.net>
*/

const Ref<Animation> AnimationNodeBlendSpaceCyclic2D::get_animation_for_point(const int p_point) {
	AnimationPlayer *ap = state->player;
	ERR_FAIL_COND_V(!ap, nullptr);

	AnimationNodeAnimation *anim_node = Object::cast_to<AnimationNodeAnimation>(blend_points[p_point].node.ptr());

	const StringName anim_name = anim_node->get_animation();
	ERR_FAIL_COND_V(!ap->has_animation(anim_name), nullptr);

	return ap->get_animation(anim_name);
}

void AnimationNodeBlendSpaceCyclic2D::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, cycle, PROPERTY_HINT_NONE, "", 0));

	AnimationNodeBlendSpace2D::get_parameter_list(r_list);
}

Variant AnimationNodeBlendSpaceCyclic2D::get_parameter_default_value(const StringName &p_parameter) const {
	if (p_parameter == cycle) {
		return 0.0;
	}

	return AnimationNodeBlendSpace2D::get_parameter_default_value(p_parameter);
}

void AnimationNodeBlendSpaceCyclic2D::add_blend_point(const Ref<AnimationRootNode> &p_node, const Vector2 &p_position, int p_at_index) {
	ERR_FAIL_COND(Object::cast_to<AnimationNodeAnimation>(p_node.ptr()) == nullptr);
	AnimationNodeBlendSpace2D::add_blend_point(p_node, p_position, p_at_index);
}

void AnimationNodeBlendSpaceCyclic2D::set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node) {
	ERR_FAIL_COND(Object::cast_to<AnimationNodeAnimation>(p_node.ptr()) == nullptr);
	AnimationNodeBlendSpace2D::set_blend_point_node(p_point, p_node);
}

float AnimationNodeBlendSpaceCyclic2D::process(float p_time, bool p_seek) {
	_update_triangles();

	Vector2 blend_pos = get_parameter(blend_position);

	float cycle = get_parameter(this->cycle);
	float length_internal = get_parameter(this->length_internal);
	int closest = get_parameter(this->closest);

	if (blend_mode == BLEND_MODE_INTERPOLATED) {
		if (triangles.size() == 0)
			return 0;

		TriangleWeights t_weights = get_triangle_blends(blend_pos);

		length_internal = 0;
		float runtimes[3] = { 0 };

		for (int i = 0; i < 3; ++i) {
			Ref<Animation> anim = get_animation_for_point(t_weights.points[i]);
			runtimes[i] = anim->get_length();
			length_internal += t_weights.weights[i] * runtimes[i];
		}

		if (p_seek) {
			cycle = Math::fposmod(p_time, length_internal) / length_internal;
		} else {
			cycle = Math::fposmod(cycle * length_internal + p_time, length_internal) / length_internal;
		}

		float largest_weight = -1;
		for (int i = 0; i < 3; ++i) {
			const BlendPoint &blend_point = blend_points[t_weights.points[i]];
			blend_node(blend_point.name, blend_point.node, cycle * runtimes[i], true, t_weights.weights[i], FILTER_IGNORE, false);

			if (t_weights.weights[i] > largest_weight) {
				largest_weight = t_weights.weights[i];
				closest = t_weights.points[i];
			}
		}
	}

	// In a cyclic BlendSpace, Discrete and Discrete-Carry are the same thing
	else {
		float closest_dist = 1e20;

		for (int i = 0; i < blend_points_used; i++) {
			float d = blend_points[i].position.distance_squared_to(blend_pos);
			if (d < closest_dist) {
				closest = i;
				closest_dist = d;
			}
		}

		Ref<Animation> anim = get_animation_for_point(closest);
		length_internal = anim->get_length();

		if (p_seek) {
			cycle = Math::fposmod(p_time, length_internal) / length_internal;
		} else {
			cycle = Math::fposmod(cycle * length_internal + p_time, length_internal) / length_internal;
		}

		blend_node(blend_points[closest].name, blend_points[closest].node, cycle * length_internal, true, 1.0, FILTER_IGNORE, false);
	}

	set_parameter(this->cycle, cycle);
	set_parameter(this->closest, closest);
	set_parameter(this->length_internal, length_internal);

	return 1 - cycle;
}

String AnimationNodeBlendSpaceCyclic2D::get_caption() const {
	return "BlendSpaceCyclic2D";
}
