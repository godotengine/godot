/**************************************************************************/
/*  animation_blend_space_1d.h                                            */
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

#include "scene/animation/animation_tree.h"

class AnimationNodeBlendSpace1D : public AnimationRootNode {
	GDCLASS(AnimationNodeBlendSpace1D, AnimationRootNode);

public:
	enum BlendMode {
		BLEND_MODE_INTERPOLATED,
		BLEND_MODE_DISCRETE,
		BLEND_MODE_DISCRETE_CARRY,
	};

protected:
	enum {
		MAX_BLEND_POINTS = 64
	};

	struct BlendPoint {
		StringName name;
		Ref<AnimationRootNode> node;
		float position = 0.0;
		float weight = -1.0;
		float velocity_limit = 0.1;
		bool override_velocity_limit = false;
	};
	bool use_velocity_limit = false;
	float default_velocity_limit = 0.0;
	float velocity_limit_ease = 1.0;

	BlendPoint blend_points[MAX_BLEND_POINTS];
	int blend_points_used = 0;

	float max_space = 1.0;
	float min_space = -1.0;

	float snap = 0.1;

	String value_label = "value";

	void _add_blend_point(int p_index, const Ref<AnimationRootNode> &p_node);

	StringName blend_position = "blend_position";
	StringName closest = "closest";

	BlendMode blend_mode = BLEND_MODE_INTERPOLATED;

	bool sync = false;

	void _validate_property(PropertyInfo &p_property) const;
	static void _bind_methods();

	virtual void _tree_changed() override;
	virtual void _animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name) override;
	virtual void _animation_node_removed(const ObjectID &p_oid, const StringName &p_node) override;

public:
	virtual void get_parameter_list(List<PropertyInfo> *r_list) const override;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const override;

	virtual void get_child_nodes(List<ChildNode> *r_child_nodes) override;

	void add_blend_point(const Ref<AnimationRootNode> &p_node, float p_position, int p_at_index = -1);
	void set_blend_point_position(int p_point, float p_position);
	void set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node);

	float get_blend_point_position(int p_point) const;
	Ref<AnimationRootNode> get_blend_point_node(int p_point) const;
	void remove_blend_point(int p_point);
	int get_blend_point_count() const;

	void set_min_space(float p_min);
	float get_min_space() const;

	void set_max_space(float p_max);
	float get_max_space() const;

	void set_snap(float p_snap);
	float get_snap() const;

	void set_value_label(const String &p_label);
	String get_value_label() const;

	void set_blend_mode(BlendMode p_blend_mode);
	BlendMode get_blend_mode() const;

	void set_use_sync(bool p_sync);
	bool is_using_sync() const;

	float velocity_limit(float p_start, float p_target, float p_velocity_limit, float p_delta);

	void set_blend_point_vl(int p_point, const float &p_velocity_limit);
	float get_blend_point_vl(int p_point) const;

	void set_blend_point_ovl(int p_point, const bool &p_override_velocity_limit);
	bool get_blend_point_ovl(int p_point) const;

	void set_use_velocity_limit(bool p_use_velocity_limit);
	bool get_use_velocity_limit() const;

	void set_velocity_limit(const double &p_default_blend_time);
	double get_velocity_limit() const;

	void set_velocity_limit_ease(const float p_ease);
	float get_velocity_limit_ease() const;

	virtual NodeTimeInfo _process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only = false) override;
	String get_caption() const override;

	Ref<AnimationNode> get_child_by_name(const StringName &p_name) const override;

	AnimationNodeBlendSpace1D();
	~AnimationNodeBlendSpace1D();
};

VARIANT_ENUM_CAST(AnimationNodeBlendSpace1D::BlendMode)
