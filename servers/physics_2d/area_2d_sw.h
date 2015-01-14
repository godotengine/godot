/*************************************************************************/
/*  area_2d_sw.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef AREA_2D_SW_H
#define AREA_2D_SW_H

#include "servers/physics_2d_server.h"
#include "collision_object_2d_sw.h"
#include "self_list.h"
//#include "servers/physics/query_sw.h"

class Space2DSW;
class Body2DSW;
class Constraint2DSW;


class Area2DSW : public CollisionObject2DSW{


	Physics2DServer::AreaSpaceOverrideMode space_override_mode;
	float gravity;
	Vector2 gravity_vector;
	bool gravity_is_point;
	float point_attenuation;
	float linear_damp;
	float angular_damp;
	int priority;

	ObjectID monitor_callback_id;
	StringName monitor_callback_method;

	SelfList<Area2DSW> monitor_query_list;
	SelfList<Area2DSW> moved_list;

	struct BodyKey {

		RID rid;
		ObjectID instance_id;
		uint32_t body_shape;
		uint32_t area_shape;

		_FORCE_INLINE_ bool operator<( const BodyKey& p_key) const {

			if (rid==p_key.rid) {

				if (body_shape==p_key.body_shape) {

					return area_shape < p_key.area_shape;
				} else
					return body_shape < p_key.body_shape;
			} else
				return rid < p_key.rid;

		}

		_FORCE_INLINE_ BodyKey() {}
		BodyKey(Body2DSW *p_body, uint32_t p_body_shape,uint32_t p_area_shape);
	};

	struct BodyState {

		int state;
		_FORCE_INLINE_ void inc() { state++; }
		_FORCE_INLINE_ void dec() { state--; }
		_FORCE_INLINE_ BodyState() { state=0; }
	};

	Map<BodyKey,BodyState> monitored_bodies;

	//virtual void shape_changed_notify(Shape2DSW *p_shape);
	//virtual void shape_deleted_notify(Shape2DSW *p_shape);
	Set<Constraint2DSW*> constraints;


	virtual void _shapes_changed();
	void _queue_monitor_update();

public:

	//_FORCE_INLINE_ const Matrix32& get_inverse_transform() const { return inverse_transform; }
	//_FORCE_INLINE_ SpaceSW* get_owner() { return owner; }

	void set_monitor_callback(ObjectID p_id, const StringName& p_method);
	_FORCE_INLINE_ bool has_monitor_callback() const { return monitor_callback_id; }

	_FORCE_INLINE_ void add_body_to_query(Body2DSW *p_body, uint32_t p_body_shape,uint32_t p_area_shape);
	_FORCE_INLINE_ void remove_body_from_query(Body2DSW *p_body, uint32_t p_body_shape,uint32_t p_area_shape);

	void set_param(Physics2DServer::AreaParameter p_param, const Variant& p_value);
	Variant get_param(Physics2DServer::AreaParameter p_param) const;

	void set_space_override_mode(Physics2DServer::AreaSpaceOverrideMode p_mode);
	Physics2DServer::AreaSpaceOverrideMode get_space_override_mode() const { return space_override_mode; }

	_FORCE_INLINE_ void set_gravity(float p_gravity) { gravity=p_gravity; }
	_FORCE_INLINE_ float get_gravity() const { return gravity; }

	_FORCE_INLINE_ void set_gravity_vector(const Vector2& p_gravity) { gravity_vector=p_gravity; }
	_FORCE_INLINE_ Vector2 get_gravity_vector() const { return gravity_vector; }

	_FORCE_INLINE_ void set_gravity_as_point(bool p_enable) { gravity_is_point=p_enable; }
	_FORCE_INLINE_ bool is_gravity_point() const { return gravity_is_point; }

	_FORCE_INLINE_ void set_point_attenuation(float p_point_attenuation) { point_attenuation=p_point_attenuation; }
	_FORCE_INLINE_ float get_point_attenuation() const { return point_attenuation; }

	_FORCE_INLINE_ void set_linear_damp(float p_linear_damp) { linear_damp=p_linear_damp; }
	_FORCE_INLINE_ float get_linear_damp() const { return linear_damp; }

	_FORCE_INLINE_ void set_angular_damp(float p_angular_damp) { angular_damp=p_angular_damp; }
	_FORCE_INLINE_ float get_angular_damp() const { return angular_damp; }

	_FORCE_INLINE_ void set_priority(int p_priority) { priority=p_priority; }
	_FORCE_INLINE_ int get_priority() const { return priority; }

	_FORCE_INLINE_ void add_constraint( Constraint2DSW* p_constraint) { constraints.insert(p_constraint); }
	_FORCE_INLINE_ void remove_constraint( Constraint2DSW* p_constraint) { constraints.erase(p_constraint); }
	_FORCE_INLINE_ const Set<Constraint2DSW*>& get_constraints() const { return constraints; }

	void set_transform(const Matrix32& p_transform);

	void set_space(Space2DSW *p_space);


	void call_queries();

	Area2DSW();
	~Area2DSW();
};

void Area2DSW::add_body_to_query(Body2DSW *p_body, uint32_t p_body_shape,uint32_t p_area_shape) {

	BodyKey bk(p_body,p_body_shape,p_area_shape);
	monitored_bodies[bk].inc();
	if (!monitor_query_list.in_list())
		_queue_monitor_update();
}
void Area2DSW::remove_body_from_query(Body2DSW *p_body, uint32_t p_body_shape,uint32_t p_area_shape) {

	BodyKey bk(p_body,p_body_shape,p_area_shape);
	monitored_bodies[bk].dec();
	if (!monitor_query_list.in_list())
		_queue_monitor_update();
}






#endif // AREA_2D_SW_H
