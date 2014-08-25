/*************************************************************************/
/*  follow_camera.cpp                                                    */
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
#include "follow_camera.h"

 
#include "physics_body.h"
#include "scene/resources/surface_tool.h"

void FollowCamera::_set_initial_orbit(const Vector2& p_orbit) {

	initial_orbit=p_orbit;
	set_orbit(p_orbit);
}



void FollowCamera::_clear_queries() {

	if (!queries_active)
		return;
#if 0
	for(int i=0;i<3;i++)
		PhysicsServer::get_singleton()->query_clear(clip_ray[i].query);
#endif
	queries_active=false;

}

void FollowCamera::_compute_camera() {

	// update the transform with the next proposed transform (camera is 1 logic frame delayed)

	/*
	float time = get_root_node()->get_frame_time();
	Vector3 oldp = accepted.get_origin();
	Vector3 newp = proposed.get_origin();

	float frame_dist = time *
	if (oldp.distance_to(newp) >
	*/

	float time = get_process_delta_time();
	bool noblend=false;

	if (clip) {

		if ((clip_ray[0].clipped==clip_ray[2].clipped || fullclip) && clip_ray[1].clipped) {
			//all have been clipped
			proposed_pos=clip_ray[1].clip_pos-extraclip*(proposed_pos-target_pos).normalized();
			if (clip_ray[0].clipped)
				fullclip=true;
			noblend=true;


		} else {


			//Vector3 rel=follow_pos-target_pos;

			if (clip_ray[0].clipped && !clip_ray[2].clipped) {

				float distance = target_pos.distance_to(clip_ray[0].clip_pos);
				real_t amount = 1.0-(distance/clip_len);
				amount = CLAMP(amount,0,1)*autoturn_speed*time;
				if (clip_ray[1].clipped)
					amount*=2.0;
				//rotate_rel=Matrix3(Vector3(0,1,0),amount).xform(rel);
				rotate_orbit(Vector2(0,amount));

			}  else if (clip_ray[2].clipped && !clip_ray[0].clipped) {

				float distance = target_pos.distance_to(clip_ray[2].clip_pos);
				real_t amount = 1.0-(distance/clip_len);
				amount = CLAMP(amount,0,1)*autoturn_speed*time;
				if (clip_ray[1].clipped)
					amount*=2.0;
				rotate_orbit(Vector2(0,-amount));
			}

			fullclip=false;

		}
	}


	Vector3 base_pos = get_global_transform().origin;
	Vector3 pull_from = base_pos;
	pull_from.y+=height; // height compensate



	Vector3 camera_target;
	if (use_lookat_target) {

		camera_target = lookat_target;
	} else {
		camera_target = base_pos;
	};

	Transform proposed;
	proposed.set_look_at(proposed_pos,camera_target,up_vector);
	proposed = proposed * Transform(Matrix3(Vector3(1,0,0),Math::deg2rad(inclination)),Vector3()); //inclination


	accepted=proposed;
	if (smooth && !noblend) {


		Vector3 vec1 = accepted.origin;
		Vector3 vec2 = final.origin;
		final.origin = vec2.linear_interpolate(vec1, MIN(1,smooth_pos_ratio * time));;

		Quat q1 = accepted.basis;
		Quat q2 = final.basis;
		final.basis = q2.slerp(q1, MIN(1,smooth_rot_ratio * time));

	} else {
		final=accepted;
	}

	_update_camera();

	// calculate the next proposed transform


	Vector3 new_pos;

	{ /*follow code*/



		/* calculate some variables */

		Vector3 rel = follow_pos - pull_from;

		float l = rel.length();
		Vector3 rel_n = (l > 0) ? (rel/l) : Vector3();
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
		new_pos+=pull_from;
		follow_pos=new_pos;

	}

	proposed_pos=new_pos;

	Vector3 rel = new_pos-camera_target;


	if (clip) {

		Vector<RID> exclude;
		exclude.push_back(target_body);

		for(int i=0;i<3;i++) {

			clip_ray[i].clipped=false;
			clip_ray[i].clip_pos=Vector3();
			clip_ray[i].cast_pos=camera_target;

			Vector3 cast_to = camera_target+Matrix3(Vector3(0,1,0),Math::deg2rad(autoturn_tolerance*(i-1.0))).xform(rel);


			if (i!=1) {

				Vector3 side = rel.cross(Vector3(0,1,0)).normalized()*(i-1.0);
				clip_ray[i].cast_pos+side*target_width+rel.normalized()*target_width;

				Vector3 d = -rel;
				d.rotate(Vector3(0,1,0),Math::deg2rad(get_fov())*(i-1.0));
				Plane p(new_pos,new_pos+d,new_pos+Vector3(0,1,0)); //fov clipping plane, build a face and use it as plane, facing doesn't matter
				Vector3 intersect;
				if (p.intersects_segment(clip_ray[i].cast_pos,cast_to,&intersect))
					cast_to=intersect;

			} else {

				cast_to+=rel.normalized()*extraclip;
			}

	//		PhysicsServer::get_singleton()->query_intersection(clip_ray[i].query,get_world()->get_space(),exclude);
	//		PhysicsServer::get_singleton()->query_intersection_segment(clip_ray[i].query,clip_ray[i].cast_pos,cast_to);




		}

		queries_active=true;
	} else {

		_clear_queries();
	}
	target_pos=camera_target;
	clip_len=rel.length();

}

void FollowCamera::set_use_lookat_target(bool p_use, const Vector3 &p_lookat) {

	use_lookat_target = p_use;
	lookat_target = p_lookat;
};


void FollowCamera::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_PROCESS: {


			_compute_camera();
		} break;

		case NOTIFICATION_ENTER_WORLD: {

			set_orbit(orbit);
			set_distance(distance);

			accepted=final=get_global_transform();
			proposed_pos=accepted.origin;

			target_body = RID();
/*
			Node* parent = get_parent();
			while (parent) {
				PhysicsBody* p = parent->cast_to<PhysicsBody>();
				if (p) {
					target_body = p->get_body();
					break;
				};
				parent = parent->get_parent();
			};
*/
			set_process(true);

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			distance=get_distance();
			orbit=get_orbit();
			_clear_queries();

		} break;
		case NOTIFICATION_BECAME_CURRENT: {

			set_process(true);
		} break;
		case NOTIFICATION_LOST_CURRENT: {

			set_process(false);
			_clear_queries();

		} break;
	}

}



void FollowCamera::set_orbit(const Vector2& p_orbit) {

	orbit=p_orbit;

	if(is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		float d = char_pos.distance_to(follow_pos);

		Matrix3 m;
		m.rotate(Vector3(0,1,0),orbit.y);
		m.rotate(Vector3(1,0,0),orbit.x);

		follow_pos=char_pos + m.get_axis(2) * d;

	}

	update_gizmo();

}
void FollowCamera::set_orbit_x(float p_x) {

	orbit.x=p_x;
	if(is_inside_scene())
		set_orbit(Vector2( p_x, get_orbit().y ));
}
void FollowCamera::set_orbit_y(float p_y) {


	orbit.y=p_y;
	if(is_inside_scene())
		set_orbit(Vector2( get_orbit().x, p_y ));

}
Vector2 FollowCamera::get_orbit() const {


	if (is_inside_scene()) {

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

void FollowCamera::rotate_orbit(const Vector2& p_relative) {

	if (is_inside_scene()) {

		Matrix3 m;
		m.rotate(Vector3(0,1,0),Math::deg2rad(p_relative.y));
		m.rotate(Vector3(1,0,0),Math::deg2rad(p_relative.x));

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		Vector3 rel = (follow_pos - char_pos);
		rel = m.xform(rel);
		follow_pos=char_pos+rel;

	}

	orbit+=p_relative;
	update_gizmo();
}

void FollowCamera::set_height(float p_height) {


	height=p_height;
	update_gizmo();
}

float FollowCamera::get_height() const {

	return height;

}

void FollowCamera::set_max_orbit_x(float p_max) {

	max_orbit_x=p_max;
	update_gizmo();
}

float FollowCamera::get_max_orbit_x() const {

	return max_orbit_x;
}

void FollowCamera::set_min_orbit_x(float p_min) {

	min_orbit_x=p_min;
	update_gizmo();
}

float FollowCamera::get_min_orbit_x() const {

	return min_orbit_x;
}

float FollowCamera::get_min_distance() const {

	return min_distance;
}
float FollowCamera::get_max_distance() const {

	return max_distance;
}

void FollowCamera::set_min_distance(float p_min) {

	min_distance=p_min;
	update_gizmo();
}

void FollowCamera::set_max_distance(float p_max) {

	max_distance = p_max;
	update_gizmo();
}


void FollowCamera::set_distance(float p_distance) {

	if (is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		Vector3 rel = (follow_pos - char_pos).normalized();
		rel*=p_distance;
		follow_pos=char_pos+rel;

	}

	distance=p_distance;
}

float FollowCamera::get_distance() const {

	if (is_inside_scene()) {

		Vector3 char_pos = get_global_transform().origin;
		char_pos.y+=height;
		return (follow_pos - char_pos).length();

	}

	return distance;
}

void FollowCamera::set_clip(bool p_enabled) {


	clip=p_enabled;

	if (!p_enabled)
		_clear_queries();
}

bool FollowCamera::has_clip() const {

	return clip;

}


void FollowCamera::set_autoturn(bool p_enabled) {


	autoturn=p_enabled;
}

bool FollowCamera::has_autoturn() const {

	return autoturn;

}

void FollowCamera::set_autoturn_tolerance(float p_degrees) {


	autoturn_tolerance=p_degrees;
}
float FollowCamera::get_autoturn_tolerance() const {


	return autoturn_tolerance;
}

void FollowCamera::set_inclination(float p_degrees) {


	inclination=p_degrees;
}
float FollowCamera::get_inclination() const {


	return inclination;
}


void FollowCamera::set_autoturn_speed(float p_speed) {


	autoturn_speed=p_speed;
}

float FollowCamera::get_autoturn_speed() const {

	return autoturn_speed;

}


RES FollowCamera::_get_gizmo_geometry() const {

	Ref<SurfaceTool> surface_tool( memnew( SurfaceTool ));

	Ref<FixedMaterial> mat( memnew( FixedMaterial ));

	mat->set_parameter( FixedMaterial::PARAM_DIFFUSE,Color(1.0,0.5,1.0,0.3) );
	mat->set_line_width(4);
	mat->set_flag(Material::FLAG_DOUBLE_SIDED,true);
	mat->set_flag(Material::FLAG_UNSHADED,true);
//	mat->set_hint(Material::HINT_NO_DEPTH_DRAW,true);

	surface_tool->begin(Mesh::PRIMITIVE_LINES);
	surface_tool->set_material(mat);


	int steps=16;

	Vector3 base_up = Matrix3(Vector3(1,0,0),Math::deg2rad(max_orbit_x)).get_axis(2);
	Vector3 base_down = Matrix3(Vector3(1,0,0),Math::deg2rad(min_orbit_x)).get_axis(2);

	Vector3 ofs(0,height,0);

	for(int i=0;i<steps;i++) {


		Matrix3 rot(Vector3(0,1,0),Math_PI*2*float(i)/steps);
		Matrix3 rot2(Vector3(0,1,0),Math_PI*2*float(i+1)/steps);

		Vector3 up = rot.xform(base_up);
		Vector3 up2 = rot2.xform(base_up);

		Vector3 down = rot.xform(base_down);
		Vector3 down2 = rot2.xform(base_down);

		surface_tool->add_vertex(ofs+up*min_distance);
		surface_tool->add_vertex(ofs+up*max_distance);
		surface_tool->add_vertex(ofs+up*min_distance);
		surface_tool->add_vertex(ofs+up2*min_distance);
		surface_tool->add_vertex(ofs+up*max_distance);
		surface_tool->add_vertex(ofs+up2*max_distance);

		surface_tool->add_vertex(ofs+down*min_distance);
		surface_tool->add_vertex(ofs+down*max_distance);
		surface_tool->add_vertex(ofs+down*min_distance);
		surface_tool->add_vertex(ofs+down2*min_distance);
		surface_tool->add_vertex(ofs+down*max_distance);
		surface_tool->add_vertex(ofs+down2*max_distance);

		int substeps = 8;

		for(int j=0;j<substeps;j++) {

			Vector3 a = up.linear_interpolate(down,float(j)/substeps).normalized()*max_distance;
			Vector3 b = up.linear_interpolate(down,float(j+1)/substeps).normalized()*max_distance;
			Vector3 am = up.linear_interpolate(down,float(j)/substeps).normalized()*min_distance;
			Vector3 bm = up.linear_interpolate(down,float(j+1)/substeps).normalized()*min_distance;

			surface_tool->add_vertex(ofs+a);
			surface_tool->add_vertex(ofs+b);
			surface_tool->add_vertex(ofs+am);
			surface_tool->add_vertex(ofs+bm);

		}
	}


	return surface_tool->commit();


}


void FollowCamera::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_set_initial_orbit","orbit"),&FollowCamera::_set_initial_orbit);
	ObjectTypeDB::bind_method(_MD("set_orbit","orbit"),&FollowCamera::set_orbit);
	ObjectTypeDB::bind_method(_MD("get_orbit"),&FollowCamera::get_orbit);
	ObjectTypeDB::bind_method(_MD("set_orbit_x","x"),&FollowCamera::set_orbit_x);
	ObjectTypeDB::bind_method(_MD("set_orbit_y","y"),&FollowCamera::set_orbit_y);
	ObjectTypeDB::bind_method(_MD("set_min_orbit_x","x"),&FollowCamera::set_min_orbit_x);
	ObjectTypeDB::bind_method(_MD("get_min_orbit_x"),&FollowCamera::get_min_orbit_x);
	ObjectTypeDB::bind_method(_MD("set_max_orbit_x","x"),&FollowCamera::set_max_orbit_x);
	ObjectTypeDB::bind_method(_MD("get_max_orbit_x"),&FollowCamera::get_max_orbit_x);
	ObjectTypeDB::bind_method(_MD("set_height","height"),&FollowCamera::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"),&FollowCamera::get_height);
	ObjectTypeDB::bind_method(_MD("set_inclination","inclination"),&FollowCamera::set_inclination);
	ObjectTypeDB::bind_method(_MD("get_inclination"),&FollowCamera::get_inclination);

	ObjectTypeDB::bind_method(_MD("rotate_orbit"),&FollowCamera::rotate_orbit);
	ObjectTypeDB::bind_method(_MD("set_distance","distance"),&FollowCamera::set_distance);
	ObjectTypeDB::bind_method(_MD("get_distance"),&FollowCamera::get_distance);
	ObjectTypeDB::bind_method(_MD("set_max_distance","max_distance"),&FollowCamera::set_max_distance);
	ObjectTypeDB::bind_method(_MD("get_max_distance"),&FollowCamera::get_max_distance);
	ObjectTypeDB::bind_method(_MD("set_min_distance","min_distance"),&FollowCamera::set_min_distance);
	ObjectTypeDB::bind_method(_MD("get_min_distance"),&FollowCamera::get_min_distance);
	ObjectTypeDB::bind_method(_MD("set_clip","enable"),&FollowCamera::set_clip);
	ObjectTypeDB::bind_method(_MD("has_clip"),&FollowCamera::has_clip);
	ObjectTypeDB::bind_method(_MD("set_autoturn","enable"),&FollowCamera::set_autoturn);
	ObjectTypeDB::bind_method(_MD("has_autoturn"),&FollowCamera::has_autoturn);
	ObjectTypeDB::bind_method(_MD("set_autoturn_tolerance","degrees"),&FollowCamera::set_autoturn_tolerance);
	ObjectTypeDB::bind_method(_MD("get_autoturn_tolerance"),&FollowCamera::get_autoturn_tolerance);
	ObjectTypeDB::bind_method(_MD("set_autoturn_speed","speed"),&FollowCamera::set_autoturn_speed);
	ObjectTypeDB::bind_method(_MD("get_autoturn_speed"),&FollowCamera::get_autoturn_speed);
	ObjectTypeDB::bind_method(_MD("set_smoothing","enable"),&FollowCamera::set_smoothing);
	ObjectTypeDB::bind_method(_MD("has_smoothing"),&FollowCamera::has_smoothing);
	ObjectTypeDB::bind_method(_MD("set_rotation_smoothing","amount"),&FollowCamera::set_rotation_smoothing);
	ObjectTypeDB::bind_method(_MD("get_rotation_smoothing"),&FollowCamera::get_rotation_smoothing);
	ObjectTypeDB::bind_method(_MD("set_translation_smoothing","amount"),&FollowCamera::set_translation_smoothing);
	ObjectTypeDB::bind_method(_MD("get_translation_smoothing"),&FollowCamera::get_translation_smoothing);
	ObjectTypeDB::bind_method(_MD("set_use_lookat_target","use","lookat"),&FollowCamera::set_use_lookat_target, DEFVAL(Vector3()));
	ObjectTypeDB::bind_method(_MD("set_up_vector","vector"),&FollowCamera::set_up_vector);
	ObjectTypeDB::bind_method(_MD("get_up_vector"),&FollowCamera::get_up_vector);

	ObjectTypeDB::bind_method(_MD("_ray_collision"),&FollowCamera::_ray_collision);

	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2, "orbit" ), _SCS("_set_initial_orbit"),_SCS("get_orbit") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "height", PROPERTY_HINT_RANGE,"-1024,1024,0.01" ), _SCS("set_height"), _SCS("get_height") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "inclination", PROPERTY_HINT_RANGE,"-90,90,0.01" ), _SCS("set_inclination"), _SCS("get_inclination")  );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "max_orbit_x", PROPERTY_HINT_RANGE,"-90,90,0.01" ), _SCS("set_max_orbit_x"), _SCS("get_max_orbit_x")  );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "min_orbit_x", PROPERTY_HINT_RANGE,"-90,90,0.01" ), _SCS("set_min_orbit_x"), _SCS("get_min_orbit_x") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "min_distance", PROPERTY_HINT_RANGE,"0,100,0.01" ), _SCS("set_min_distance"), _SCS("get_min_distance") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "max_distance", PROPERTY_HINT_RANGE,"0,100,0.01" ), _SCS("set_max_distance"), _SCS("get_max_distance") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "distance", PROPERTY_HINT_RANGE,"0.01,1024,0,01"), _SCS("set_distance"), _SCS("get_distance")  );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "clip"), _SCS("set_clip"), _SCS("has_clip")  );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "autoturn"), _SCS("set_autoturn"), _SCS("has_autoturn")  );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "autoturn_tolerance", PROPERTY_HINT_RANGE,"1,90,0.01") , _SCS("set_autoturn_tolerance"), _SCS("get_autoturn_tolerance") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "autoturn_speed", PROPERTY_HINT_RANGE,"1,90,0.01"), _SCS("set_autoturn_speed"), _SCS("get_autoturn_speed")  );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "smoothing"), _SCS("set_smoothing"), _SCS("has_smoothing")  );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "translation_smooth", PROPERTY_HINT_RANGE,"0.01,128,0.01"), _SCS("set_translation_smoothing"), _SCS("get_translation_smoothing")  );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "rotation_smooth", PROPERTY_HINT_RANGE,"0.01,128,0.01"), _SCS("set_rotation_smoothing"), _SCS("get_rotation_smoothing")  );


}

void FollowCamera::_ray_collision(Vector3 p_point, Vector3 p_normal, int p_subindex, ObjectID p_against,int p_idx) {

	clip_ray[p_idx].clip_pos=p_point;
	clip_ray[p_idx].clipped=true;

};

Transform FollowCamera::get_camera_transform() const {

	return final;
}

void FollowCamera::set_smoothing(bool p_enable) {

	smooth=p_enable;
}

bool FollowCamera::has_smoothing() const {

	return smooth;
}

void FollowCamera::set_translation_smoothing(float p_amount) {

	smooth_pos_ratio=p_amount;
}
float FollowCamera::get_translation_smoothing() const {

	return smooth_pos_ratio;
}

void FollowCamera::set_rotation_smoothing(float p_amount) {

	smooth_rot_ratio=p_amount;

}

void FollowCamera::set_up_vector(const Vector3& p_up) {

	up_vector=p_up;
}

Vector3 FollowCamera::get_up_vector() const{

	return up_vector;
}

float FollowCamera::get_rotation_smoothing() const {

	return smooth_pos_ratio;

}


FollowCamera::FollowCamera() {


	height=1;

	orbit=Vector2(0,0);
	up_vector=Vector3(0,1,0);

	distance=3;
	min_distance=2;
	max_distance=5;

	autoturn=true;
	autoturn_tolerance=10;
	autoturn_speed=80;

	min_orbit_x=-50;
	max_orbit_x=70;
	inclination=0;
	target_width=0.3;

	clip=true;
	use_lookat_target = false;
	extraclip=0.3;
	fullclip=false;

	smooth=true;
	smooth_rot_ratio=10;
	smooth_pos_ratio=10;


	for(int i=0;i<3;i++) {
//		clip_ray[i].query=PhysicsServer::get_singleton()->query_create(this, "_ray_collision", i, true);
		clip_ray[i].clipped=false;
	}

	queries_active=false;


}

FollowCamera::~FollowCamera() {

	for(int i=0;i<3;i++) {
		PhysicsServer::get_singleton()->free(clip_ray[i].query);
	}


}
