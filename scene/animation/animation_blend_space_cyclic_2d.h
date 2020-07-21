/*************************************************************************/
/*  animation_blend_space_cyclic_2d.h                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef ANIMATION_BLEND_SPACE_CYCLIC_2D_H
#define ANIMATION_BLEND_SPACE_CYCLIC_2D_H

#include "scene/animation/animation_blend_space_2d.h"

/**
	@author Marios Staikopoulos <marios@staik.net>
*/

class AnimationNodeBlendSpaceCyclic2D : public AnimationNodeBlendSpace2D {
	GDCLASS(AnimationNodeBlendSpaceCyclic2D, AnimationNodeBlendSpace2D);

private:
	const Ref<Animation> get_animation_for_point(const int p_point);

protected:
	StringName cycle;

public:
	virtual void get_parameter_list(List<PropertyInfo> *r_list) const override;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const override;

	virtual void add_blend_point(const Ref<AnimationRootNode> &p_node, const Vector2 &p_position, int p_at_index = -1) override;
	virtual void set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node) override;

	virtual float process(float p_time, bool p_seek) override;
	virtual String get_caption() const override;

	AnimationNodeBlendSpaceCyclic2D() {}
	~AnimationNodeBlendSpaceCyclic2D() {}
};

#endif // ANIMATION_BLEND_SPACE_CYCLIC_2D_H
