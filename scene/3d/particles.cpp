/*************************************************************************/
/*  particles.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "particles.h"
#include "servers/visual_server.h"
#include "scene/resources/surface_tool.h"

#if 0
/*
static const char* _var_names[Particles::VAR_MAX]={
	"vars/lifetime",
	"vars/spread",
	"vars/gravity",
	"vars/linear_vel",
	"vars/angular_vel",
	"vars/linear_accel",
	"vars/radial_accel",
	"vars/tan_accel",
	"vars/initial_size",
	"vars/final_size",
	"vars/initial_angle",
	"vars/height",
	"vars/height_speed_scale",
};
*/
static const char* _rand_names[Particles::VAR_MAX]={
	"rand/lifetime",
	"rand/spread",
	"rand/gravity",
	"rand/linear_vel",
	"rand/angular_vel",
	"rand/linear_accel",
	"rand/radial_accel",
	"rand/tan_accel",
	"rand/damping",
	"rand/initial_size",
	"rand/final_size",
	"rand/initial_angle",
	"rand/height",
	"rand/height_speed_scale",
};

static const Particles::Variable _var_indices[Particles::VAR_MAX]={
	Particles::VAR_LIFETIME,
	Particles::VAR_SPREAD,
	Particles::VAR_GRAVITY,
	Particles::VAR_LINEAR_VELOCITY,
	Particles::VAR_ANGULAR_VELOCITY,
	Particles::VAR_LINEAR_ACCELERATION,
	Particles::VAR_DRAG,
	Particles::VAR_TANGENTIAL_ACCELERATION,
	Particles::VAR_DAMPING,
	Particles::VAR_INITIAL_SIZE,
	Particles::VAR_FINAL_SIZE,
	Particles::VAR_INITIAL_ANGLE,
	Particles::VAR_HEIGHT,
	Particles::VAR_HEIGHT_SPEED_SCALE,
};



AABB Particles::get_aabb() const {

	return AABB( Vector3(-1,-1,-1), Vector3(2, 2, 2 ) );
}
PoolVector<Face3> Particles::get_faces(uint32_t p_usage_flags) const {

	return PoolVector<Face3>();
}


void Particles::set_amount(int p_amount) {

	ERR_FAIL_INDEX(p_amount,1024);
	amount=p_amount;
	VisualServer::get_singleton()->particles_set_amount(particles,p_amount);
}
int Particles::get_amount() const {

	return amount;
}

void Particles::set_emitting(bool p_emitting) {

	emitting=p_emitting;
	VisualServer::get_singleton()->particles_set_emitting(particles,p_emitting);

	setup_timer();
}
bool Particles::is_emitting() const {

	return emitting;
}

void Particles::set_visibility_aabb(const AABB& p_aabb) {

	visibility_aabb=p_aabb;
	VisualServer::get_singleton()->particles_set_visibility_aabb(particles,p_aabb);
	update_gizmo();

}
AABB Particles::get_visibility_aabb() const {

	return visibility_aabb;
}


void Particles::set_emission_points(const PoolVector<Vector3>& p_points) {

	using_points = p_points.size();
	VisualServer::get_singleton()->particles_set_emission_points(particles,p_points);
}

PoolVector<Vector3> Particles::get_emission_points() const {

	if (!using_points)
		return PoolVector<Vector3>();

	return VisualServer::get_singleton()->particles_get_emission_points(particles);

}

void Particles::set_emission_half_extents(const Vector3& p_half_extents) {

	emission_half_extents=p_half_extents;
	VisualServer::get_singleton()->particles_set_emission_half_extents(particles,p_half_extents);

}

Vector3 Particles::get_emission_half_extents() const {

	return emission_half_extents;
}

void Particles::set_emission_base_velocity(const Vector3& p_base_velocity) {

	emission_base_velocity=p_base_velocity;
	VisualServer::get_singleton()->particles_set_emission_base_velocity(particles,p_base_velocity);

}

Vector3 Particles::get_emission_base_velocity() const {

	return emission_base_velocity;
}

void Particles::set_gravity_normal(const Vector3& p_normal)  {

	gravity_normal=p_normal;
	VisualServer::get_singleton()->particles_set_gravity_normal(particles,p_normal);
}

Vector3 Particles::get_gravity_normal() const {

	return gravity_normal;

}

void Particles::set_variable(Variable p_variable,float p_value) {

	ERR_FAIL_INDEX(p_variable,VAR_MAX);
	var[p_variable]=p_value;
	VisualServer::get_singleton()->particles_set_variable(particles,(VS::ParticleVariable)p_variable,p_value);
	if (p_variable==VAR_SPREAD)
		update_gizmo();
}

float Particles::get_variable(Variable p_variable) const {

	ERR_FAIL_INDEX_V(p_variable,VAR_MAX,-1);
	return var[p_variable];

}

void Particles::set_randomness(Variable p_variable,float p_randomness) {

	ERR_FAIL_INDEX(p_variable,VAR_MAX);
	var_random[p_variable]=p_randomness;
	VisualServer::get_singleton()->particles_set_randomness(particles,(VS::ParticleVariable)p_variable,p_randomness);

}
float Particles::get_randomness(Variable p_variable) const {

	ERR_FAIL_INDEX_V(p_variable,VAR_MAX,-1);
	return var_random[p_variable];

}

void Particles::set_color_phase_pos(int p_phase, float p_pos) {

	ERR_FAIL_INDEX(p_phase,VS::MAX_PARTICLE_COLOR_PHASES);
	color_phase[p_phase].pos=p_pos;
	VisualServer::get_singleton()->particles_set_color_phase_pos(particles,p_phase,p_pos);

}
float Particles::get_color_phase_pos(int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase,VS::MAX_PARTICLE_COLOR_PHASES,-1);
	return color_phase[p_phase].pos;
}

void Particles::set_color_phase_color(int p_phase, const Color& p_color) {

	ERR_FAIL_INDEX(p_phase,VS::MAX_PARTICLE_COLOR_PHASES);
	color_phase[p_phase].color=p_color;
	VisualServer::get_singleton()->particles_set_color_phase_color(particles,p_phase,p_color);

}
Color Particles::get_color_phase_color(int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase,VS::MAX_PARTICLE_COLOR_PHASES,Color());
	return color_phase[p_phase].color;

}

void Particles::set_material(const Ref<Material>& p_material) {

	material=p_material;
	if(material.is_null()) {
		VisualServer::get_singleton()->particles_set_material(particles,RID());
	} else {
		VisualServer::get_singleton()->particles_set_material(particles,material->get_rid());
	}

}

void Particles::setup_timer() {

	if (emitting && emit_timeout > 0) {

		timer->set_wait_time(emit_timeout);
		timer->start();
		timer->set_one_shot(true);
	};
};

void Particles::set_emit_timeout(float p_timeout) {

	emit_timeout = p_timeout;
	setup_timer();
};

float Particles::get_emit_timeout() const {

	return emit_timeout;
};


Ref<Material> Particles::get_material() const {

	return material;
}

void Particles::set_height_from_velocity(bool p_enable) {

	height_from_velocity=p_enable;
	VisualServer::get_singleton()->particles_set_height_from_velocity(particles,height_from_velocity);
}

bool Particles::has_height_from_velocity() const {

	return height_from_velocity;
}

void Particles::set_color_phases(int p_phases) {

	color_phase_count=p_phases;
	VisualServer::get_singleton()->particles_set_color_phases(particles,p_phases);
}

int Particles::get_color_phases() const{

	return color_phase_count;
}

bool Particles::_can_gizmo_scale() const {

	return false;
}

void Particles::set_use_local_coordinates(bool p_use) {

	local_coordinates=p_use;
	VisualServer::get_singleton()->particles_set_use_local_coordinates(particles,local_coordinates);
}

bool Particles::is_using_local_coordinates() const{

	return local_coordinates;
}


RES Particles::_get_gizmo_geometry() const {

	Ref<SurfaceTool> surface_tool( memnew( SurfaceTool ));

	Ref<FixedSpatialMaterial> mat( memnew( FixedSpatialMaterial ));

	mat->set_parameter( FixedSpatialMaterial::PARAM_DIFFUSE,Color(0.0,0.6,0.7,0.2) );
	mat->set_parameter( FixedSpatialMaterial::PARAM_EMISSION,Color(0.5,0.7,0.8) );
	mat->set_blend_mode( Material::BLEND_MODE_ADD );
	mat->set_flag(Material::FLAG_DOUBLE_SIDED,true);
	//mat->set_hint(Material::HINT_NO_DEPTH_DRAW,true);


	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	surface_tool->set_material(mat);

	int sides=16;
	int sections=24;

	//float len=1;
	float deg=Math::deg2rad(var[VAR_SPREAD]*180);
	if (deg==180)
		deg=179.5;

	Vector3 to=Vector3(0,0,-1);

	for(int j=0;j<sections;j++) {

		Vector3 p1=Matrix3(Vector3(1,0,0),deg*j/sections).xform(to);
		Vector3 p2=Matrix3(Vector3(1,0,0),deg*(j+1)/sections).xform(to);

		for(int i=0;i<sides;i++) {

			Vector3 p1r = Matrix3(Vector3(0,0,1),Math_PI*2*float(i)/sides).xform(p1);
			Vector3 p1s = Matrix3(Vector3(0,0,1),Math_PI*2*float(i+1)/sides).xform(p1);
			Vector3 p2s = Matrix3(Vector3(0,0,1),Math_PI*2*float(i+1)/sides).xform(p2);
			Vector3 p2r = Matrix3(Vector3(0,0,1),Math_PI*2*float(i)/sides).xform(p2);

			surface_tool->add_normal(p1r.normalized());
			surface_tool->add_vertex(p1r);
			surface_tool->add_normal(p1s.normalized());
			surface_tool->add_vertex(p1s);
			surface_tool->add_normal(p2s.normalized());
			surface_tool->add_vertex(p2s);

			surface_tool->add_normal(p1r.normalized());
			surface_tool->add_vertex(p1r);
			surface_tool->add_normal(p2s.normalized());
			surface_tool->add_vertex(p2s);
			surface_tool->add_normal(p2r.normalized());
			surface_tool->add_vertex(p2r);

			if (j==sections-1) {

				surface_tool->add_normal(p2r.normalized());
				surface_tool->add_vertex(p2r);
				surface_tool->add_normal(p2s.normalized());
				surface_tool->add_vertex(p2s);
				surface_tool->add_normal(Vector3(0,0,1));
				surface_tool->add_vertex(Vector3());
			}
		}
	}


	Ref<Mesh> mesh = surface_tool->commit();

	Ref<FixedSpatialMaterial> mat_aabb( memnew( FixedSpatialMaterial ));

	mat_aabb->set_parameter( FixedSpatialMaterial::PARAM_DIFFUSE,Color(0.8,0.8,0.9,0.7) );
	mat_aabb->set_line_width(3);
	mat_aabb->set_flag( Material::FLAG_UNSHADED, true );

	surface_tool->begin(Mesh::PRIMITIVE_LINES);
	surface_tool->set_material(mat_aabb);

	for(int i=0;i<12;i++) {

		Vector3 f,t;
		visibility_aabb.get_edge(i,f,t);
		surface_tool->add_vertex(f);
		surface_tool->add_vertex(t);
	}

	return surface_tool->commit(mesh);

}


void Particles::_bind_methods() {

	ClassDB::bind_method(_MD("set_amount","amount"),&Particles::set_amount);
	ClassDB::bind_method(_MD("get_amount"),&Particles::get_amount);
	ClassDB::bind_method(_MD("set_emitting","enabled"),&Particles::set_emitting);
	ClassDB::bind_method(_MD("is_emitting"),&Particles::is_emitting);
	ClassDB::bind_method(_MD("set_visibility_aabb","aabb"),&Particles::set_visibility_aabb);
	ClassDB::bind_method(_MD("get_visibility_aabb"),&Particles::get_visibility_aabb);
	ClassDB::bind_method(_MD("set_emission_half_extents","half_extents"),&Particles::set_emission_half_extents);
	ClassDB::bind_method(_MD("get_emission_half_extents"),&Particles::get_emission_half_extents);
	ClassDB::bind_method(_MD("set_emission_base_velocity","base_velocity"),&Particles::set_emission_base_velocity);
	ClassDB::bind_method(_MD("get_emission_base_velocity"),&Particles::get_emission_base_velocity);
	ClassDB::bind_method(_MD("set_emission_points","points"),&Particles::set_emission_points);
	ClassDB::bind_method(_MD("get_emission_points"),&Particles::get_emission_points);
	ClassDB::bind_method(_MD("set_gravity_normal","normal"),&Particles::set_gravity_normal);
	ClassDB::bind_method(_MD("get_gravity_normal"),&Particles::get_gravity_normal);
	ClassDB::bind_method(_MD("set_variable","variable","value"),&Particles::set_variable);
	ClassDB::bind_method(_MD("get_variable","variable"),&Particles::get_variable);
	ClassDB::bind_method(_MD("set_randomness","variable","randomness"),&Particles::set_randomness);
	ClassDB::bind_method(_MD("get_randomness","variable"),&Particles::get_randomness);
	ClassDB::bind_method(_MD("set_color_phase_pos","phase","pos"),&Particles::set_color_phase_pos);
	ClassDB::bind_method(_MD("get_color_phase_pos","phase"),&Particles::get_color_phase_pos);
	ClassDB::bind_method(_MD("set_color_phase_color","phase","color"),&Particles::set_color_phase_color);
	ClassDB::bind_method(_MD("get_color_phase_color","phase"),&Particles::get_color_phase_color);
	ClassDB::bind_method(_MD("set_material","material:Material"),&Particles::set_material);
	ClassDB::bind_method(_MD("get_material:Material"),&Particles::get_material);
	ClassDB::bind_method(_MD("set_emit_timeout","timeout"),&Particles::set_emit_timeout);
	ClassDB::bind_method(_MD("get_emit_timeout"),&Particles::get_emit_timeout);
	ClassDB::bind_method(_MD("set_height_from_velocity","enable"),&Particles::set_height_from_velocity);
	ClassDB::bind_method(_MD("has_height_from_velocity"),&Particles::has_height_from_velocity);
	ClassDB::bind_method(_MD("set_use_local_coordinates","enable"),&Particles::set_use_local_coordinates);
	ClassDB::bind_method(_MD("is_using_local_coordinates"),&Particles::is_using_local_coordinates);

	ClassDB::bind_method(_MD("set_color_phases","count"),&Particles::set_color_phases);
	ClassDB::bind_method(_MD("get_color_phases"),&Particles::get_color_phases);

	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Material" ), _SCS("set_material"), _SCS("get_material") );

	ADD_PROPERTY( PropertyInfo( Variant::INT, "amount", PROPERTY_HINT_RANGE, "1,1024,1" ), _SCS("set_amount"), _SCS("get_amount") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "emitting" ), _SCS("set_emitting"), _SCS("is_emitting") );
	ADD_PROPERTY( PropertyInfo( Variant::_AABB, "visibility" ), _SCS("set_visibility_aabb"), _SCS("get_visibility_aabb") );
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR3, "emission_extents" ), _SCS("set_emission_half_extents"), _SCS("get_emission_half_extents") );
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR3, "emission_base_velocity" ), _SCS("set_emission_base_velocity"), _SCS("get_emission_base_velocity") );
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR3_ARRAY, "emission_points" ), _SCS("set_emission_points"), _SCS("get_emission_points") );
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR3, "gravity_normal" ), _SCS("set_gravity_normal"), _SCS("get_gravity_normal") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "local_coords" ), _SCS("set_use_local_coordinates"), _SCS("is_using_local_coordinates") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "emit_timeout",PROPERTY_HINT_RANGE,"0,256,0.01"), _SCS("set_emit_timeout"), _SCS("get_emit_timeout") );


	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/lifetime", PROPERTY_HINT_RANGE,"0.1,60,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_LIFETIME );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/spread", PROPERTY_HINT_RANGE,"0,1,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_SPREAD );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/gravity", PROPERTY_HINT_RANGE,"-48,48,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_GRAVITY );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/linear_vel", PROPERTY_HINT_RANGE,"-100,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_LINEAR_VELOCITY );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/angular_vel", PROPERTY_HINT_RANGE,"-100,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_ANGULAR_VELOCITY );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/linear_accel", PROPERTY_HINT_RANGE,"-100,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_LINEAR_ACCELERATION );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/radial_accel", PROPERTY_HINT_RANGE,"-100,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_DRAG );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/tan_accel", PROPERTY_HINT_RANGE,"-100,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_TANGENTIAL_ACCELERATION );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/damping", PROPERTY_HINT_RANGE,"0,128,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_DAMPING );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/initial_size", PROPERTY_HINT_RANGE,"0,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_INITIAL_SIZE );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/final_size", PROPERTY_HINT_RANGE,"0,100,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_FINAL_SIZE );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/initial_angle",PROPERTY_HINT_RANGE,"0,1,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_INITIAL_ANGLE );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "vars/height_from_velocity"), _SCS("set_height_from_velocity"), _SCS("has_height_from_velocity") );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/height",PROPERTY_HINT_RANGE,"0,4096,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_HEIGHT);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "vars/height_speed_scale",PROPERTY_HINT_RANGE,"0,4096,0.01"), _SCS("set_variable"), _SCS("get_variable"), VAR_HEIGHT_SPEED_SCALE );

	for(int i=0;i<VAR_MAX;i++)
		ADD_PROPERTYI( PropertyInfo( Variant::REAL, _rand_names[i], PROPERTY_HINT_RANGE,"-16.0,16.0,0.01"),_SCS("set_randomness"), _SCS("get_randomness"),_var_indices[i] );


	ADD_PROPERTY( PropertyInfo( Variant::INT, "color_phases/count",PROPERTY_HINT_RANGE,"0,4,1"), _SCS("set_color_phases"), _SCS("get_color_phases"));

	for(int i=0;i<VS::MAX_PARTICLE_COLOR_PHASES;i++) {
		String phase="phase_"+itos(i)+"/";
		ADD_PROPERTYI( PropertyInfo( Variant::REAL, phase+"pos", PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_color_phase_pos"),_SCS("get_color_phase_pos"),i );
		ADD_PROPERTYI( PropertyInfo( Variant::COLOR, phase+"color"),_SCS("set_color_phase_color"),_SCS("get_color_phase_color"),i );
	}

	BIND_CONSTANT( VAR_LIFETIME );
	BIND_CONSTANT( VAR_SPREAD );
	BIND_CONSTANT( VAR_GRAVITY );
	BIND_CONSTANT( VAR_LINEAR_VELOCITY );
	BIND_CONSTANT( VAR_ANGULAR_VELOCITY );
	BIND_CONSTANT( VAR_LINEAR_ACCELERATION );
	BIND_CONSTANT( VAR_DRAG );
	BIND_CONSTANT( VAR_TANGENTIAL_ACCELERATION );
	BIND_CONSTANT( VAR_INITIAL_SIZE );
	BIND_CONSTANT( VAR_FINAL_SIZE );
	BIND_CONSTANT( VAR_INITIAL_ANGLE );
	BIND_CONSTANT( VAR_HEIGHT );
	BIND_CONSTANT( VAR_HEIGHT_SPEED_SCALE );
	BIND_CONSTANT( VAR_MAX );

}

Particles::Particles() {

	particles = VisualServer::get_singleton()->particles_create();
	timer = memnew(Timer);
	add_child(timer);
	emit_timeout = 0;

	set_amount(64);
	set_emitting(true);
	set_visibility_aabb(AABB( Vector3(-4,-4,-4), Vector3(8,8,8) ) );

	for (int i=0;i<VAR_MAX;i++) {
		set_randomness((Variable)i,0.0);
	}

	set_variable( VAR_LIFETIME, 5.0);
	set_variable( VAR_SPREAD, 0.2);
	set_variable( VAR_GRAVITY, 9.8);
	set_variable( VAR_LINEAR_VELOCITY, 0.2);
	set_variable( VAR_ANGULAR_VELOCITY, 0.0);
	set_variable( VAR_LINEAR_ACCELERATION, 0.0);
	set_variable( VAR_DRAG, 0.0);
	set_variable( VAR_TANGENTIAL_ACCELERATION, 0.0);
	set_variable( VAR_DAMPING, 0.0);
	set_variable( VAR_INITIAL_SIZE, 1.0);
	set_variable( VAR_FINAL_SIZE, 1.0);
	set_variable( VAR_INITIAL_ANGLE, 0.0);
	set_variable( VAR_HEIGHT, 1.0);
	set_variable( VAR_HEIGHT_SPEED_SCALE, 0.0);

	color_phase_count=0;

	set_color_phase_pos(0,0.0);
	set_color_phase_pos(1,1.0);
	set_color_phase_pos(2,1.0);
	set_color_phase_pos(3,1.0);

	set_color_phase_color(0,Color(1,1,1));
	set_color_phase_color(1,Color(0,0,0));
	set_color_phase_color(2,Color(0,0,0));
	set_color_phase_color(3,Color(0,0,0));

	set_gravity_normal(Vector3(0,-1.0,0));
	set_emission_half_extents(Vector3(0.1,0.1,0.1));

	height_from_velocity=false;

	Vector<Variant> pars;
	pars.push_back(false);
	timer->connect("timeout", this, "set_emitting", pars);
	set_base(particles);
	local_coordinates=false;
}


Particles::~Particles() {

	VisualServer::get_singleton()->free(particles);
}

#endif
