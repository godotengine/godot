/*************************************************************************/
/*  area_2d.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef AREA_2D_H
#define AREA_2D_H

#include "scene/2d/collision_object_2d.h"
#include "vset.h"

class Area2D : public CollisionObject2D {

	GDCLASS(Area2D, CollisionObject2D);

public:
	enum SpaceOverride {
		SPACE_OVERRIDE_DISABLED,
		SPACE_OVERRIDE_COMBINE,
		SPACE_OVERRIDE_COMBINE_REPLACE,
		SPACE_OVERRIDE_REPLACE,
		SPACE_OVERRIDE_REPLACE_COMBINE
	};

private:
	SpaceOverride space_override;
	Vector2 gravity_vec;
	real_t gravity;
	bool gravity_is_point;
	real_t gravity_distance_scale;
	real_t linear_damp;
	real_t angular_damp;
	uint32_t collision_mask;
	uint32_t layer_mask;
	int priority;
	bool monitoring;
	bool monitorable;
	bool locked;

	void _body_inout(int p_status, const RID &p_body, int p_instance, int p_body_shape, int p_area_shape);

	void _body_enter_tree(ObjectID p_id);
	void _body_exit_tree(ObjectID p_id);

	struct ShapePair {

		int body_shape;
		int area_shape;
		bool operator<(const ShapePair &p_sp) const {
			if (body_shape == p_sp.body_shape)
				return area_shape < p_sp.area_shape;
			else
				return body_shape < p_sp.body_shape;
		}

		ShapePair() {}
		ShapePair(int p_bs, int p_as) {
			body_shape = p_bs;
			area_shape = p_as;
		}
	};

	struct BodyState {

		int rc;
		bool in_tree;
		VSet<ShapePair> shapes;
	};

	Map<ObjectID, BodyState> body_map;

	void _area_inout(int p_status, const RID &p_area, int p_instance, int p_area_shape, int p_self_shape);

	void _area_enter_tree(ObjectID p_id);
	void _area_exit_tree(ObjectID p_id);

	struct AreaShapePair {

		int area_shape;
		int self_shape;
		bool operator<(const AreaShapePair &p_sp) const {
			if (area_shape == p_sp.area_shape)
				return self_shape < p_sp.self_shape;
			else
				return area_shape < p_sp.area_shape;
		}

		AreaShapePair() {}
		AreaShapePair(int p_bs, int p_as) {
			area_shape = p_bs;
			self_shape = p_as;
		}
	};

	struct AreaState {

		int rc;
		bool in_tree;
		VSet<AreaShapePair> shapes;
	};

	Map<ObjectID, AreaState> area_map;
	void _clear_monitoring();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_space_override_mode(SpaceOverride p_mode);
	SpaceOverride get_space_override_mode() const;

	void set_gravity_is_point(bool p_enabled);
	bool is_gravity_a_point() const;

	void set_gravity_distance_scale(real_t p_scale);
	real_t get_gravity_distance_scale() const;

	void set_gravity_vector(const Vector2 &p_vec);
	Vector2 get_gravity_vector() const;

	void set_gravity(real_t p_gravity);
	real_t get_gravity() const;

	void set_linear_damp(real_t p_linear_damp);
	real_t get_linear_damp() const;

	void set_angular_damp(real_t p_angular_damp);
	real_t get_angular_damp() const;

	void set_priority(real_t p_priority);
	real_t get_priority() const;

	void set_monitoring(bool p_enable);
	bool is_monitoring() const;

	void set_monitorable(bool p_enable);
	bool is_monitorable() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_layer_mask(uint32_t p_mask);
	uint32_t get_layer_mask() const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	void set_layer_mask_bit(int p_bit, bool p_value);
	bool get_layer_mask_bit(int p_bit) const;

	Array get_overlapping_bodies() const; //function for script
	Array get_overlapping_areas() const; //function for script

	bool overlaps_area(Node *p_area) const;
	bool overlaps_body(Node *p_body) const;

	Area2D();
	~Area2D();
};

VARIANT_ENUM_CAST(Area2D::SpaceOverride);

#endif // AREA_2D_H
