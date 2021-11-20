/*************************************************************************/
/*  body_2d_sw.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "body_2d_sw.h"
#include "area_2d_sw.h"
#include "physics_2d_server_sw.h"
#include "space_2d_sw.h"

void Body2DSW::_update_inertia() {
	if (!user_inertia && get_space() && !inertia_update_list.in_list()) {
		get_space()->body_add_to_inertia_update_list(&inertia_update_list);
	}
}

void Body2DSW::update_inertias() {
	//update shapes and motions

	switch (mode) {
		case Physics2DServer::BODY_MODE_RIGID: {
			if (user_inertia) {
				_inv_inertia = inertia > 0 ? (1.0 / inertia) : 0;
				break;
			}
			//update tensor for allshapes, not the best way but should be somehow OK. (inspired from bullet)
			real_t total_area = 0;

			for (int i = 0; i < get_shape_count(); i++) {
				if (is_shape_disabled(i)) {
					continue;
				}

				total_area += get_shape_aabb(i).get_area();
			}

			inertia = 0;

			for (int i = 0; i < get_shape_count(); i++) {
				if (is_shape_disabled(i)) {
					continue;
				}

				const Shape2DSW *shape = get_shape(i);

				real_t area = get_shape_aabb(i).get_area();

				real_t mass = area * this->mass / total_area;

				Transform2D mtx = get_shape_transform(i);
				Vector2 scale = mtx.get_scale();
				inertia += shape->get_moment_of_inertia(mass, scale) + mass * mtx.get_origin().length_squared();
			}

			_inv_inertia = inertia > 0 ? (1.0 / inertia) : 0;

			if (mass) {
				_inv_mass = 1.0 / mass;
			} else {
				_inv_mass = 0;
			}

		} break;
		case Physics2DServer::BODY_MODE_KINEMATIC:
		case Physics2DServer::BODY_MODE_STATIC: {
			_inv_inertia = 0;
			_inv_mass = 0;
		} break;
		case Physics2DServer::BODY_MODE_CHARACTER: {
			_inv_inertia = 0;
			_inv_mass = 1.0 / mass;

		} break;
	}
	//_update_inertia_tensor();

	//_update_shapes();
}

void Body2DSW::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}

	active = p_active;
	if (!p_active) {
		if (get_space()) {
			get_space()->body_remove_from_active_list(&active_list);
		}
	} else {
		if (mode == Physics2DServer::BODY_MODE_STATIC) {
			return; //static bodies can't become active
		}
		if (get_space()) {
			get_space()->body_add_to_active_list(&active_list);
		}

		//still_time=0;
	}
	/*
	if (!space)
		return;

	for(int i=0;i<get_shape_count();i++) {
		Shape &s=shapes[i];
		if (s.bpid>0) {
			get_space()->get_broadphase()->set_active(s.bpid,active);
		}
	}
*/
}

void Body2DSW::set_param(Physics2DServer::BodyParameter p_param, real_t p_value) {
	switch (p_param) {
		case Physics2DServer::BODY_PARAM_BOUNCE: {
			bounce = p_value;
		} break;
		case Physics2DServer::BODY_PARAM_FRICTION: {
			friction = p_value;
		} break;
		case Physics2DServer::BODY_PARAM_MASS: {
			ERR_FAIL_COND(p_value <= 0);
			mass = p_value;
			_update_inertia();

		} break;
		case Physics2DServer::BODY_PARAM_INERTIA: {
			if (p_value <= 0) {
				user_inertia = false;
				_update_inertia();
			} else {
				user_inertia = true;
				inertia = p_value;
				_inv_inertia = 1.0 / p_value;
			}
		} break;
		case Physics2DServer::BODY_PARAM_GRAVITY_SCALE: {
			gravity_scale = p_value;
		} break;
		case Physics2DServer::BODY_PARAM_LINEAR_DAMP: {
			linear_damp = p_value;
		} break;
		case Physics2DServer::BODY_PARAM_ANGULAR_DAMP: {
			angular_damp = p_value;
		} break;
		default: {
		}
	}
}

real_t Body2DSW::get_param(Physics2DServer::BodyParameter p_param) const {
	switch (p_param) {
		case Physics2DServer::BODY_PARAM_BOUNCE: {
			return bounce;
		}
		case Physics2DServer::BODY_PARAM_FRICTION: {
			return friction;
		}
		case Physics2DServer::BODY_PARAM_MASS: {
			return mass;
		}
		case Physics2DServer::BODY_PARAM_INERTIA: {
			return inertia;
		}
		case Physics2DServer::BODY_PARAM_GRAVITY_SCALE: {
			return gravity_scale;
		}
		case Physics2DServer::BODY_PARAM_LINEAR_DAMP: {
			return linear_damp;
		}
		case Physics2DServer::BODY_PARAM_ANGULAR_DAMP: {
			return angular_damp;
		}
		default: {
		}
	}

	return 0;
}

void Body2DSW::set_mode(Physics2DServer::BodyMode p_mode) {
	Physics2DServer::BodyMode prev = mode;
	mode = p_mode;

	switch (p_mode) {
		//CLEAR UP EVERYTHING IN CASE IT NOT WORKS!
		case Physics2DServer::BODY_MODE_STATIC:
		case Physics2DServer::BODY_MODE_KINEMATIC: {
			_set_inv_transform(get_transform().affine_inverse());
			_inv_mass = 0;
			_inv_inertia = 0;
			_set_static(p_mode == Physics2DServer::BODY_MODE_STATIC);
			set_active(p_mode == Physics2DServer::BODY_MODE_KINEMATIC && contacts.size());
			linear_velocity = Vector2();
			angular_velocity = 0;
			if (mode == Physics2DServer::BODY_MODE_KINEMATIC && prev != mode) {
				first_time_kinematic = true;
			}
		} break;
		case Physics2DServer::BODY_MODE_RIGID: {
			_inv_mass = mass > 0 ? (1.0 / mass) : 0;
			_inv_inertia = inertia > 0 ? (1.0 / inertia) : 0;
			_set_static(false);
			set_active(true);

		} break;
		case Physics2DServer::BODY_MODE_CHARACTER: {
			_inv_mass = mass > 0 ? (1.0 / mass) : 0;
			_inv_inertia = 0;
			_set_static(false);
			set_active(true);
			angular_velocity = 0;
		} break;
	}
	if (p_mode == Physics2DServer::BODY_MODE_RIGID && _inv_inertia == 0) {
		_update_inertia();
	}
	/*
	if (get_space())
		_update_queries();
	*/
}
Physics2DServer::BodyMode Body2DSW::get_mode() const {
	return mode;
}

void Body2DSW::_shapes_changed() {
	_update_inertia();
	wakeup();
	wakeup_neighbours();
}

void Body2DSW::set_state(Physics2DServer::BodyState p_state, const Variant &p_variant) {
	switch (p_state) {
		case Physics2DServer::BODY_STATE_TRANSFORM: {
			if (mode == Physics2DServer::BODY_MODE_KINEMATIC) {
				new_transform = p_variant;
				//wakeup_neighbours();
				set_active(true);
				if (first_time_kinematic) {
					_set_transform(p_variant);
					_set_inv_transform(get_transform().affine_inverse());
					first_time_kinematic = false;
				}
			} else if (mode == Physics2DServer::BODY_MODE_STATIC) {
				_set_transform(p_variant);
				_set_inv_transform(get_transform().affine_inverse());
				wakeup_neighbours();
			} else {
				Transform2D t = p_variant;
				t.orthonormalize();
				new_transform = get_transform(); //used as old to compute motion
				if (t == new_transform) {
					break;
				}
				_set_transform(t);
				_set_inv_transform(get_transform().inverse());
			}
			wakeup();

		} break;
		case Physics2DServer::BODY_STATE_LINEAR_VELOCITY: {
			/*
			if (mode==Physics2DServer::BODY_MODE_STATIC)
				break;
			*/
			linear_velocity = p_variant;
			wakeup();

		} break;
		case Physics2DServer::BODY_STATE_ANGULAR_VELOCITY: {
			/*
			if (mode!=Physics2DServer::BODY_MODE_RIGID)
				break;
			*/
			angular_velocity = p_variant;
			wakeup();

		} break;
		case Physics2DServer::BODY_STATE_SLEEPING: {
			//?
			if (mode == Physics2DServer::BODY_MODE_STATIC || mode == Physics2DServer::BODY_MODE_KINEMATIC) {
				break;
			}
			bool do_sleep = p_variant;
			if (do_sleep) {
				linear_velocity = Vector2();
				//biased_linear_velocity=Vector3();
				angular_velocity = 0;
				//biased_angular_velocity=Vector3();
				set_active(false);
			} else {
				if (mode != Physics2DServer::BODY_MODE_STATIC) {
					set_active(true);
				}
			}
		} break;
		case Physics2DServer::BODY_STATE_CAN_SLEEP: {
			can_sleep = p_variant;
			if (mode == Physics2DServer::BODY_MODE_RIGID && !active && !can_sleep) {
				set_active(true);
			}

		} break;
	}
}
Variant Body2DSW::get_state(Physics2DServer::BodyState p_state) const {
	switch (p_state) {
		case Physics2DServer::BODY_STATE_TRANSFORM: {
			return get_transform();
		}
		case Physics2DServer::BODY_STATE_LINEAR_VELOCITY: {
			return linear_velocity;
		}
		case Physics2DServer::BODY_STATE_ANGULAR_VELOCITY: {
			return angular_velocity;
		}
		case Physics2DServer::BODY_STATE_SLEEPING: {
			return !is_active();
		}
		case Physics2DServer::BODY_STATE_CAN_SLEEP: {
			return can_sleep;
		}
	}

	return Variant();
}

void Body2DSW::set_space(Space2DSW *p_space) {
	if (get_space()) {
		wakeup_neighbours();

		if (inertia_update_list.in_list()) {
			get_space()->body_remove_from_inertia_update_list(&inertia_update_list);
		}
		if (active_list.in_list()) {
			get_space()->body_remove_from_active_list(&active_list);
		}
		if (direct_state_query_list.in_list()) {
			get_space()->body_remove_from_state_query_list(&direct_state_query_list);
		}
	}

	_set_space(p_space);

	if (get_space()) {
		_update_inertia();
		if (active) {
			get_space()->body_add_to_active_list(&active_list);
		}
		/*
		_update_queries();
		if (is_active()) {
			active=false;
			set_active(true);
		}
		*/
	}

	first_integration = false;
}

void Body2DSW::_compute_area_gravity_and_dampenings(const Area2DSW *p_area) {
	if (p_area->is_gravity_point()) {
		if (p_area->get_gravity_distance_scale() > 0) {
			Vector2 v = p_area->get_transform().xform(p_area->get_gravity_vector()) - get_transform().get_origin();
			gravity += v.normalized() * (p_area->get_gravity() / Math::pow(v.length() * p_area->get_gravity_distance_scale() + 1, 2));
		} else {
			gravity += (p_area->get_transform().xform(p_area->get_gravity_vector()) - get_transform().get_origin()).normalized() * p_area->get_gravity();
		}
	} else {
		gravity += p_area->get_gravity_vector() * p_area->get_gravity();
	}

	area_linear_damp += p_area->get_linear_damp();
	area_angular_damp += p_area->get_angular_damp();
}

void Body2DSW::integrate_forces(real_t p_step) {
	if (mode == Physics2DServer::BODY_MODE_STATIC) {
		return;
	}

	Area2DSW *def_area = get_space()->get_default_area();
	// Area2DSW *damp_area = def_area;
	ERR_FAIL_COND(!def_area);

	int ac = areas.size();
	bool stopped = false;
	gravity = Vector2(0, 0);
	area_angular_damp = 0;
	area_linear_damp = 0;
	if (ac) {
		areas.sort();
		const AreaCMP *aa = &areas[0];
		// damp_area = aa[ac-1].area;
		for (int i = ac - 1; i >= 0 && !stopped; i--) {
			Physics2DServer::AreaSpaceOverrideMode mode = aa[i].area->get_space_override_mode();
			switch (mode) {
				case Physics2DServer::AREA_SPACE_OVERRIDE_COMBINE:
				case Physics2DServer::AREA_SPACE_OVERRIDE_COMBINE_REPLACE: {
					_compute_area_gravity_and_dampenings(aa[i].area);
					stopped = mode == Physics2DServer::AREA_SPACE_OVERRIDE_COMBINE_REPLACE;
				} break;
				case Physics2DServer::AREA_SPACE_OVERRIDE_REPLACE:
				case Physics2DServer::AREA_SPACE_OVERRIDE_REPLACE_COMBINE: {
					gravity = Vector2(0, 0);
					area_angular_damp = 0;
					area_linear_damp = 0;
					_compute_area_gravity_and_dampenings(aa[i].area);
					stopped = mode == Physics2DServer::AREA_SPACE_OVERRIDE_REPLACE;
				} break;
				default: {
				}
			}
		}
	}
	if (!stopped) {
		_compute_area_gravity_and_dampenings(def_area);
	}
	gravity *= gravity_scale;

	// If less than 0, override dampenings with that of the Body2D
	if (angular_damp >= 0) {
		area_angular_damp = angular_damp;
	}
	/*
	else
		area_angular_damp=damp_area->get_angular_damp();
	*/

	if (linear_damp >= 0) {
		area_linear_damp = linear_damp;
	}
	/*
	else
		area_linear_damp=damp_area->get_linear_damp();
	*/

	Vector2 motion;
	bool do_motion = false;

	if (mode == Physics2DServer::BODY_MODE_KINEMATIC) {
		//compute motion, angular and etc. velocities from prev transform
		motion = new_transform.get_origin() - get_transform().get_origin();
		linear_velocity = motion / p_step;

		real_t rot = new_transform.get_rotation() - get_transform().get_rotation();
		angular_velocity = remainder(rot, 2.0 * Math_PI) / p_step;

		do_motion = true;

		/*
		for(int i=0;i<get_shape_count();i++) {
			set_shape_kinematic_advance(i,Vector2());
			set_shape_kinematic_retreat(i,0);
		}
		*/

	} else {
		if (!omit_force_integration && !first_integration) {
			//overridden by direct state query

			Vector2 force = gravity * mass;
			force += applied_force;
			real_t torque = applied_torque;

			real_t damp = 1.0 - p_step * area_linear_damp;

			if (damp < 0) { // reached zero in the given time
				damp = 0;
			}

			real_t angular_damp = 1.0 - p_step * area_angular_damp;

			if (angular_damp < 0) { // reached zero in the given time
				angular_damp = 0;
			}

			linear_velocity *= damp;
			angular_velocity *= angular_damp;

			linear_velocity += _inv_mass * force * p_step;
			angular_velocity += _inv_inertia * torque * p_step;
		}

		if (continuous_cd_mode != Physics2DServer::CCD_MODE_DISABLED) {
			motion = linear_velocity * p_step;
			do_motion = true;
		}
	}

	//motion=linear_velocity*p_step;

	first_integration = false;
	biased_angular_velocity = 0;
	biased_linear_velocity = Vector2();

	if (do_motion) { //shapes temporarily extend for raycast
		_update_shapes_with_motion(motion);
	}

	// damp_area=NULL; // clear the area, so it is set in the next frame
	def_area = nullptr; // clear the area, so it is set in the next frame
	contact_count = 0;
}

void Body2DSW::integrate_velocities(real_t p_step) {
	if (mode == Physics2DServer::BODY_MODE_STATIC) {
		return;
	}

	if (fi_callback) {
		get_space()->body_add_to_state_query_list(&direct_state_query_list);
	}

	if (mode == Physics2DServer::BODY_MODE_KINEMATIC) {
		_set_transform(new_transform, false);
		_set_inv_transform(new_transform.affine_inverse());
		if (contacts.size() == 0 && linear_velocity == Vector2() && angular_velocity == 0) {
			set_active(false); //stopped moving, deactivate
		}
		return;
	}

	real_t total_angular_velocity = angular_velocity + biased_angular_velocity;
	Vector2 total_linear_velocity = linear_velocity + biased_linear_velocity;

	real_t angle = get_transform().get_rotation() + total_angular_velocity * p_step;
	Vector2 pos = get_transform().get_origin() + total_linear_velocity * p_step;

	_set_transform(Transform2D(angle, pos), continuous_cd_mode == Physics2DServer::CCD_MODE_DISABLED);
	_set_inv_transform(get_transform().inverse());

	if (continuous_cd_mode != Physics2DServer::CCD_MODE_DISABLED) {
		new_transform = get_transform();
	}

	//_update_inertia_tensor();
}

void Body2DSW::wakeup_neighbours() {
	for (Map<Constraint2DSW *, int>::Element *E = constraint_map.front(); E; E = E->next()) {
		const Constraint2DSW *c = E->key();
		Body2DSW **n = c->get_body_ptr();
		int bc = c->get_body_count();

		for (int i = 0; i < bc; i++) {
			if (i == E->get()) {
				continue;
			}
			Body2DSW *b = n[i];
			if (b->mode != Physics2DServer::BODY_MODE_RIGID) {
				continue;
			}

			if (!b->is_active()) {
				b->set_active(true);
			}
		}
	}
}

void Body2DSW::call_queries() {
	if (fi_callback) {
		Variant v = direct_access;
		const Variant *vp[2] = { &v, &fi_callback->callback_udata };

		Object *obj = ObjectDB::get_instance(fi_callback->id);
		if (!obj) {
			set_force_integration_callback(0, StringName());
		} else {
			Variant::CallError ce;
			if (fi_callback->callback_udata.get_type() != Variant::NIL) {
				obj->call(fi_callback->method, vp, 2, ce);

			} else {
				obj->call(fi_callback->method, vp, 1, ce);
			}
		}
	}
}

bool Body2DSW::sleep_test(real_t p_step) {
	if (mode == Physics2DServer::BODY_MODE_STATIC || mode == Physics2DServer::BODY_MODE_KINEMATIC) {
		return true; //
	} else if (mode == Physics2DServer::BODY_MODE_CHARACTER) {
		return !active; // characters and kinematic bodies don't sleep unless asked to sleep
	} else if (!can_sleep) {
		return false;
	}

	if (Math::abs(angular_velocity) < get_space()->get_body_angular_velocity_sleep_threshold() && Math::abs(linear_velocity.length_squared()) < get_space()->get_body_linear_velocity_sleep_threshold() * get_space()->get_body_linear_velocity_sleep_threshold()) {
		still_time += p_step;

		return still_time > get_space()->get_body_time_to_sleep();
	} else {
		still_time = 0; //maybe this should be set to 0 on set_active?
		return false;
	}
}

void Body2DSW::set_force_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata) {
	if (fi_callback) {
		memdelete(fi_callback);
		fi_callback = nullptr;
	}

	if (p_id != 0) {
		fi_callback = memnew(ForceIntegrationCallback);
		fi_callback->id = p_id;
		fi_callback->method = p_method;
		fi_callback->callback_udata = p_udata;
	}
}

Body2DSW::Body2DSW() :
		CollisionObject2DSW(TYPE_BODY),
		active_list(this),
		inertia_update_list(this),
		direct_state_query_list(this) {
	mode = Physics2DServer::BODY_MODE_RIGID;
	active = true;
	angular_velocity = 0;
	biased_angular_velocity = 0;
	mass = 1;
	inertia = 0;
	user_inertia = false;
	_inv_inertia = 0;
	_inv_mass = 1;
	bounce = 0;
	friction = 1;
	omit_force_integration = false;
	applied_torque = 0;
	island_step = 0;
	island_next = nullptr;
	island_list_next = nullptr;
	_set_static(false);
	first_time_kinematic = false;
	linear_damp = -1;
	angular_damp = -1;
	area_angular_damp = 0;
	area_linear_damp = 0;
	contact_count = 0;
	gravity_scale = 1.0;
	first_integration = false;

	still_time = 0;
	continuous_cd_mode = Physics2DServer::CCD_MODE_DISABLED;
	can_sleep = true;
	fi_callback = nullptr;

	direct_access = memnew(Physics2DDirectBodyStateSW);
	direct_access->body = this;
}

Body2DSW::~Body2DSW() {
	memdelete(direct_access);
	if (fi_callback) {
		memdelete(fi_callback);
	}
}

Physics2DDirectSpaceState *Physics2DDirectBodyStateSW::get_space_state() {
	return body->get_space()->get_direct_state();
}

real_t Physics2DDirectBodyStateSW::get_step() const {
	return body->get_space()->get_step();
}

Variant Physics2DDirectBodyStateSW::get_contact_collider_shape_metadata(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, Variant());

	if (!Physics2DServerSW::singletonsw->body_owner.owns(body->contacts[p_contact_idx].collider)) {
		return Variant();
	}
	Body2DSW *other = Physics2DServerSW::singletonsw->body_owner.get(body->contacts[p_contact_idx].collider);

	int sidx = body->contacts[p_contact_idx].collider_shape;
	if (sidx < 0 || sidx >= other->get_shape_count()) {
		return Variant();
	}

	return other->get_shape_metadata(sidx);
}
