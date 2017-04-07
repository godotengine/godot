/*************************************************************************/
/*  character_camera.cpp                                                 */
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
#include "character_camera.h"

#include "physics_body.h"
#if 0
void CharacterCamera::_set(const String& p_name, const Variant& p_value) {

	if (p_name=="type")
		set_camera_type((CameraType)((int)(p_value)));
	else if (p_name=="orbit")
		set_orbit(p_value);
	else if (p_name=="height")
		set_height(p_value);
	else if (p_name=="inclination")
		set_inclination(p_value);
	else if (p_name=="max_orbit_x")
		set_max_orbit_x(p_value);
	else if (p_name=="min_orbit_x")
		set_min_orbit_x(p_value);
	else if (p_name=="max_distance")
		set_max_distance(p_value);
	else if (p_name=="min_distance")
		set_min_distance(p_value);
	else if (p_name=="distance")
		set_distance(p_value);
	else if (p_name=="clip")
		set_clip(p_value);
	else if (p_name=="autoturn")
		set_autoturn(p_value);
	else if (p_name=="autoturn_tolerance")
		set_autoturn_tolerance(p_value);
	else if (p_name=="autoturn_speed")
		set_autoturn_speed(p_value);

}
Variant CharacterCamera::_get(const String& p_name) const {

	if (p_name=="type")
		return get_camera_type();
	else if (p_name=="orbit")
		return get_orbit();
	else if (p_name=="height")
		return get_height();
	else if (p_name=="inclination")
		return get_inclination();
	else if (p_name=="max_orbit_x")
		return get_max_orbit_x();
	else if (p_name=="min_orbit_x")
		return get_min_orbit_x();
	else if (p_name=="max_distance")
		return get_max_distance();
	else if (p_name=="min_distance")
		return get_min_distance();
	else if (p_name=="distance")
		return get_distance();
	else if (p_name=="clip")
		return has_clip();
	else if (p_name=="autoturn")
		return has_autoturn();
	else if (p_name=="autoturn_tolerance")
		return get_autoturn_tolerance();
	else if (p_name=="autoturn_speed")
		return get_autoturn_speed();

	return Variant();
}

void CharacterCamera::_get_property_list( List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo( Variant::INT, "type", PROPERTY_HINT_ENUM, "Fixed,Follow") );
	p_list->push_back( PropertyInfo( Variant::VECTOR2, "orbit" ) );
	p_list->push_back( PropertyInfo( Variant::REAL, "height", PROPERTY_HINT_RANGE,"-1024,1024,0.01" ) );
	p_list->push_back( PropertyInfo( Variant::REAL, "inclination", PROPERTY_HINT_RANGE,"-90,90,0.01" )  );
	p_list->push_back( PropertyInfo( Variant::REAL, "max_orbit_x", PROPERTY_HINT_RANGE,"-90,90,0.01" )  );
	p_list->push_back( PropertyInfo( Variant::REAL, "min_orbit_x", PROPERTY_HINT_RANGE,"-90,90,0.01" ) );
	p_list->push_back( PropertyInfo( Variant::REAL, "min_distance", PROPERTY_HINT_RANGE,"0,100,0.01" ) );
	p_list->push_back( PropertyInfo( Variant::REAL, "max_distance", PROPERTY_HINT_RANGE,"0,100,0.01" ) );
	p_list->push_back( PropertyInfo( Variant::REAL, "distance", PROPERTY_HINT_RANGE,"0.01,1024,0,01")  );
	p_list->push_back( PropertyInfo( Variant::BOOL, "clip")  );
	p_list->push_back( PropertyInfo( Variant::BOOL, "autoturn")  );
	p_list->push_back( PropertyInfo( Variant::REAL, "autoturn_tolerance", PROPERTY_HINT_RANGE,"1,90,0.01")  );
	p_list->push_back( PropertyInfo( Variant::REAL, "autoturn_speed", PROPERTY_HINT_RANGE,"1,90,0.01")  );


}

void CharacterCamera::_compute_camera() {


	// update the transform with the next proposed transform (camera is 1 logic frame delayed)

	/*
	float time = get_root_node()->get_frame_time();
	Vector3 oldp = accepted.get_origin();
	Vector3 newp = proposed.get_origin();

	float frame_dist = time *
	if (oldp.distance_to(newp) >
	*/

	float time = get_root_node()->get_frame_time();

	if (true) {

		if (clip_ray[0].clipped && clip_ray[1].clipped && clip_ray[2].clipped) {
			//all have been clipped
			proposed.origin=clip_ray[1].clip_pos;


		} else {

			Vector3 rel=proposed.origin-target_pos;

			if (clip_ray[0].clipped && !clip_ray[2].clipped) {

				float distance = target_pos.distance_to(clip_ray[0].clip_pos);
				real_t amount = 1.0-(distance/clip_len);
				amount = CLAMP(amount,0,1);


				rel=Matrix3(Vector3(0,1,0)),
				rotate_orbit(Vector2(0,autoturn_speed*time*amount));
			}
			if (clip_ray[2].clipped && !clip_ray[0].clipped) {

				float distance = target_pos.distance_to(clip_ray[2].clip_pos);
				real_t amount = 1.0-(distance/clip_len);
				amount = CLAMP(amount,0,1);

				rotate_orbit(Vector2(0,-autoturn_speed*time*amount));
			}

		}
	}


	Transform final;

	static float pos_ratio = 0.9;
	static float rot_ratio = 10;

	Vector3 vec1 = accepted.origin;
	Vector3 vec2 = proposed.origin;
	final.origin = vec2.linear_interpolate(vec1, pos_ratio * time);

	Quat q1 = accepted.basis;
	Quat q2 = proposed.basis;
	final.basis = q1.slerp(q2, rot_ratio * time);

	accepted=final;

	_update_camera();

	// calculate the next proposed transform


	Vector3 new_pos;
	Vector3 character_pos = get_global_transform().origin;
	character_pos.y+=height; // height compensate

	if(type==CAMERA_FOLLOW) {



		/* calculate some variables */

		Vector3 rel = follow_pos - character_pos;

		float l = rel.length();
		Vector3 rel_n = (l > 0) ? (rel/l) : Vector3();
#if 1
		float ang = Math::acos(rel_n.dot( Vector3(0,1,0) ));

		Vector3 tangent = rel_n;
		tangent.y=0; // get rid of y
		if (tangent.length_squared() < CMP_EPSILON2)
			tangent=Vector3(0,0,1); // use Z as tangent if rel is parallel to y
		else
			tangent.normalize();

		/* now start applying the rules */

		//clip distance
		if (l > max_distance)
			l=max_distance;
		if (l < min_distance)
			l=min_distance;

		//fix angle

		float ang_min = Math_PI * 0.5 + Math::deg2rad(min_orbit_x);
		float ang_max = Math_PI * 0.5 + Math::deg2rad(max_orbit_x);

		if (ang<ang_min)
			ang=ang_min;
		if (ang>ang_max)
			ang=ang_max;

		/* finally, rebuild the validated camera position */

		new_pos=Vector3(0,Math::cos(ang),0);
		new_pos+=tangent*Math::sin(ang);
		new_pos*=l;
		new_pos+=character_pos;
#else
		if (l > max_distance)
			l=max_distance;
		if (l < min_distance)
			l=min_distance;

		new_pos = character_pos + rel_n * l;

#endif
		follow_pos=new_pos;

	} else if (type==CAMERA_FIXED) {


		if (distance<min_distance)
			distance=min_distance;
		if (distance>max_distance)
			distance=max_distance;

		if (orbit.x<min_orbit_x)
			orbit.x=min_orbit_x;
		if (orbit.x>max_orbit_x)
			orbit.x=max_orbit_x;

		Matrix3 m;
		m.rotate(Vector3(0,1,0),-Math::deg2rad(orbit.y));
		m.rotate(Vector3(1,0,0),-Math::deg2rad(orbit.x));

		new_pos = (m.get_axis(2) * distance) + character_pos;

		if (use_lookat_target) {

			Transform t = get_global_transform();
			Vector3 y = t.basis.get_axis(1).normalized();
			Vector3 z = lookat_target - character_pos;
			z= (z - y * y.dot(z)).normalized();
			orbit.y = -Math::rad2deg(Math::atan2(z.x,z.z)) + 180;

			/*
			Transform t = get_global_transform();
			Vector3 y = t.basis.get_axis(1).normalized();
			Vector3 z = lookat_target  - t.origin;
			z= (z - y * y.dot(z)).normalized();
			Vector3 x = z.cross(y).normalized();
			Transform t2;
			t2.basis.set_axis(0,x);
			t2.basis.set_axis(1,y);
			t2.basis.set_axis(2,z);
			t2.origin=t.origin;

			Vector3 local = t2.xform_inv(camera_pos);

			float ang = Math::atan2(local.x,local.y);
			*/

			/*

			Vector3 vec1 = lookat_target - new_pos;
			vec1.normalize();
			Vector3 vec2 = character_pos - new_pos;
			vec2.normalize();

			float dot = vec1.dot(vec2);
			printf("dot %f\n", dot);
			if ( dot < 0.5) {

				rotate_orbit(Vector2(0, 90));
			};
			*/


		};
	}

	Vector3 target;
	if (use_lookat_target) {

		target = lookat_target;
	} else {
		target = character_pos;
	};

	proposed.set_look_at(new_pos,target,Vector3(0,1,0));
	proposed = proposed * Transform(Matrix3(Vector3(1,0,0),Math::deg2rad(inclination)),Vector3()); //inclination


	Vector<RID> exclude;
	exclude.push_back(target_body);



	Vector3 rel = new_pos-target;

	for(int i=0;i<3;i++) {

		PhysicsServer::get_singleton()->query_intersection(clip_ray[i].query,get_world().get_space(),exclude);
		PhysicsServer::get_singleton()->query_intersection_segment(clip_ray[i].query,target,target+Matrix3(Vector3(0,1,0),Math::deg2rad(autoturn_tolerance*(i-1.0))).xform(rel));
		clip_ray[i].clipped=false;
		clip_ray[i].clip_pos=Vector3();
	}
	target_pos=target;
	clip_len=rel.length();



}

void CharacterCamera::set_use_lookat_target(bool p_use, const Vector3 &p_lookat) {

	use_lookat_target = p_use;
	lookat_target = p_lookat;
};


void CharacterCamera::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_PROCESS: {


			_compute_camera();
		} break;

		case NOTIFICATION_ENTER_SCENE: {

			if (type==CAMERA_FOLLOW) {

				set_orbit(orbit);
				set_distance(distance);
			}

			accepted=get_global_transform();
			proposed=accepted;

			target_body = RID();

			Node* parent = get_parent();
			while (parent) {
				PhysicsBody* p = parent->cast_to<PhysicsBody>();
				if (p) {
					target_body = p->get_body();
					break;
				};
				parent = parent->get_parent();
			};

		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {

		} break;
		case NOTIFICATION_EXIT_SCENE: {

			if (type==CAMERA_FOLLOW) {
				distance=get_distance();
				orbit=get_orbit();

			}
		} break;

		case NOTIFICATION_BECAME_CURRENT: {

			set_process(true);
		} break;
		case NOTIFICATION_LOST_CURRENT: {

			set_process(false);
		} break;
	}

}


void CharacterCamera::set_camera_type(CameraType p_camera_type) {


	if (p_camera_type==type)
		return;

	type=p_camera_type;

	// do conversions
}

CharacterCamera::CameraType CharacterCamera::get_camera_type() const {

	return type;

}

void CharacterCamera::set_orbit(const Vector2& p_orbit) {

	orbit=p_orbit;

	if(type == CAMERA_FOLLOW && is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		float d = char_pos.distance_to(follow_pos);

		Matrix3 m;
		m.rotate(Vector3(0,1,0),-orbit.y);
		m.rotate(Vector3(1,0,0),-orbit.x);

		follow_pos=char_pos + m.get_axis(2) * d;

	}

}
void CharacterCamera::set_orbit_x(float p_x) {

	orbit.x=p_x;
	if(type == CAMERA_FOLLOW && is_inside_scene())
		set_orbit(Vector2( p_x, get_orbit().y ));
}
void CharacterCamera::set_orbit_y(float p_y) {


	orbit.y=p_y;
	if(type == CAMERA_FOLLOW && is_inside_scene())
		set_orbit(Vector2( get_orbit().x, p_y ));

}
Vector2 CharacterCamera::get_orbit() const {


	if (type == CAMERA_FOLLOW && is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		Vector3 rel = (follow_pos - char_pos).normalized();
		Vector2 ret_orbit;
		ret_orbit.x = Math::acos( Vector3(0,1,0).dot( rel ) ) - Math_PI * 0.5;
		ret_orbit.y = Math::atan2(rel.x,rel.z);
		return ret_orbit;
	}
	return orbit;
}

void CharacterCamera::rotate_orbit(const Vector2& p_relative) {

	if (type == CAMERA_FOLLOW && is_inside_scene()) {

		Matrix3 m;
		m.rotate(Vector3(0,1,0),-Math::deg2rad(p_relative.y));
		m.rotate(Vector3(1,0,0),-Math::deg2rad(p_relative.x));

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		Vector3 rel = (follow_pos - char_pos);
		rel = m.xform(rel);
		follow_pos=char_pos+rel;

	}

	orbit+=p_relative;
}

void CharacterCamera::set_height(float p_height) {


	height=p_height;
}

float CharacterCamera::get_height() const {

	return height;

}

void CharacterCamera::set_max_orbit_x(float p_max) {

	max_orbit_x=p_max;
}

float CharacterCamera::get_max_orbit_x() const {

	return max_orbit_x;
}

void CharacterCamera::set_min_orbit_x(float p_min) {

	min_orbit_x=p_min;
}

float CharacterCamera::get_min_orbit_x() const {

	return min_orbit_x;
}

float CharacterCamera::get_min_distance() const {

	return min_distance;
}
float CharacterCamera::get_max_distance() const {

	return max_distance;
}

void CharacterCamera::set_min_distance(float p_min) {

	min_distance=p_min;
}

void CharacterCamera::set_max_distance(float p_max) {

	max_distance = p_max;
}


void CharacterCamera::set_distance(float p_distance) {

	if (type == CAMERA_FOLLOW && is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		Vector3 rel = (follow_pos - char_pos).normalized();
		rel*=p_distance;
		follow_pos=char_pos+rel;

	}

	distance=p_distance;
}

float CharacterCamera::get_distance() const {

	if (type == CAMERA_FOLLOW && is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		return (follow_pos - char_pos).length();

	}

	return distance;
}

void CharacterCamera::set_clip(bool p_enabled) {


	clip=p_enabled;
}

bool CharacterCamera::has_clip() const {

	return clip;

}


void CharacterCamera::set_autoturn(bool p_enabled) {


	autoturn=p_enabled;
}

bool CharacterCamera::has_autoturn() const {

	return autoturn;

}

void CharacterCamera::set_autoturn_tolerance(float p_degrees) {


	autoturn_tolerance=p_degrees;
}
float CharacterCamera::get_autoturn_tolerance() const {


	return autoturn_tolerance;
}

void CharacterCamera::set_inclination(float p_degrees) {


	inclination=p_degrees;
}
float CharacterCamera::get_inclination() const {


	return inclination;
}


void CharacterCamera::set_autoturn_speed(float p_speed) {


	autoturn_speed=p_speed;
}
float CharacterCamera::get_autoturn_speed() const {

	return autoturn_speed;

}





void CharacterCamera::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_camera_type","type"),&CharacterCamera::set_camera_type);
	ClassDB::bind_method(D_METHOD("get_camera_type"),&CharacterCamera::get_camera_type);
	ClassDB::bind_method(D_METHOD("set_orbit","orbit"),&CharacterCamera::set_orbit);
	ClassDB::bind_method(D_METHOD("get_orbit"),&CharacterCamera::get_orbit);
	ClassDB::bind_method(D_METHOD("set_orbit_x","x"),&CharacterCamera::set_orbit_x);
	ClassDB::bind_method(D_METHOD("set_orbit_y","y"),&CharacterCamera::set_orbit_y);
	ClassDB::bind_method(D_METHOD("set_min_orbit_x","x"),&CharacterCamera::set_min_orbit_x);
	ClassDB::bind_method(D_METHOD("get_min_orbit_x"),&CharacterCamera::get_min_orbit_x);
	ClassDB::bind_method(D_METHOD("set_max_orbit_x","x"),&CharacterCamera::set_max_orbit_x);
	ClassDB::bind_method(D_METHOD("get_max_orbit_x"),&CharacterCamera::get_max_orbit_x);
	ClassDB::bind_method(D_METHOD("rotate_orbit"),&CharacterCamera::rotate_orbit);
	ClassDB::bind_method(D_METHOD("set_distance","distance"),&CharacterCamera::set_distance);
	ClassDB::bind_method(D_METHOD("get_distance"),&CharacterCamera::get_distance);
	ClassDB::bind_method(D_METHOD("set_clip","enable"),&CharacterCamera::set_clip);
	ClassDB::bind_method(D_METHOD("has_clip"),&CharacterCamera::has_clip);
	ClassDB::bind_method(D_METHOD("set_autoturn","enable"),&CharacterCamera::set_autoturn);
	ClassDB::bind_method(D_METHOD("has_autoturn"),&CharacterCamera::has_autoturn);
	ClassDB::bind_method(D_METHOD("set_autoturn_tolerance","degrees"),&CharacterCamera::set_autoturn_tolerance);
	ClassDB::bind_method(D_METHOD("get_autoturn_tolerance"),&CharacterCamera::get_autoturn_tolerance);
	ClassDB::bind_method(D_METHOD("set_autoturn_speed","speed"),&CharacterCamera::set_autoturn_speed);
	ClassDB::bind_method(D_METHOD("get_autoturn_speed"),&CharacterCamera::get_autoturn_speed);
	ClassDB::bind_method(D_METHOD("set_use_lookat_target","use","lookat"),&CharacterCamera::set_use_lookat_target, DEFVAL(Vector3()));

	ClassDB::bind_method(D_METHOD("_ray_collision"),&CharacterCamera::_ray_collision);

	BIND_CONSTANT( CAMERA_FIXED );
	BIND_CONSTANT( CAMERA_FOLLOW );
}

void CharacterCamera::_ray_collision(Vector3 p_point, Vector3 p_normal, int p_subindex, ObjectID p_against,int p_idx) {


	clip_ray[p_idx].clip_pos=p_point;
	clip_ray[p_idx].clipped=true;
};

Transform CharacterCamera::get_camera_transform() const {

	return accepted;
}


CharacterCamera::CharacterCamera() {


	type=CAMERA_FOLLOW;
	height=1;

	orbit=Vector2(0,0);

	distance=3;
	min_distance=2;
	max_distance=5;

	autoturn=false;
	autoturn_tolerance=15;
	autoturn_speed=20;

	min_orbit_x=-50;
	max_orbit_x=70;
	inclination=0;

	clip=false;
	use_lookat_target = false;

	for(int i=0;i<3;i++) {
		clip_ray[i].query=PhysicsServer::get_singleton()->query_create(this, "_ray_collision", i, true);
		clip_ray[i].clipped=false;
	}


}

CharacterCamera::~CharacterCamera() {

	for(int i=0;i<3;i++) {
		PhysicsServer::get_singleton()->free(clip_ray[i].query);
	}


}
#endif
