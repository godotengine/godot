/*************************************************************************/
/*  light.cpp                                                            */
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
#include "light.h"
 
#include "globals.h"
#include "scene/resources/surface_tool.h"


static const char* _light_param_names[VS::LIGHT_PARAM_MAX]={
	"params/spot_attenuation",
	"params/spot_angle",
	"params/radius",
	"params/energy",
	"params/attenuation",
	"shadow/darkening",
	"shadow/z_offset",
	"shadow/z_slope_scale",
	"shadow/esm_multiplier",
	"shadow/blur_passes"
};

void Light::set_parameter(Parameter p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	vars[p_param]=p_value;
	VisualServer::get_singleton()->light_set_param(light,(VisualServer::LightParam)p_param,p_value);
	if (p_param==PARAM_RADIUS || p_param==PARAM_SPOT_ANGLE)
		update_gizmo();
	_change_notify(_light_param_names[p_param]);
//	_change_notify(_param_names[p_param]);
}

float Light::get_parameter(Parameter p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return vars[p_param];

}

void Light::set_color(LightColor p_color, const Color& p_value) {

	ERR_FAIL_INDEX(p_color, 3);
	colors[p_color]=p_value;
	VisualServer::get_singleton()->light_set_color(light,(VisualServer::LightColor)p_color,p_value);
	//_change_notify(_color_names[p_color]);

}
Color Light::get_color(LightColor p_color) const {

	ERR_FAIL_INDEX_V(p_color, 3, Color());
	return colors[p_color];

}


void Light::set_project_shadows(bool p_enabled) {

	shadows=p_enabled;
	VisualServer::get_singleton()->light_set_shadow(light, p_enabled);
	_change_notify("shadow");
}
bool Light::has_project_shadows() const {
	
	return shadows;
}

void Light::set_projector(const Ref<Texture>& p_projector) {

	projector=p_projector;
	VisualServer::get_singleton()->light_set_projector(light, projector.is_null()?RID():projector->get_rid());
}

Ref<Texture> Light::get_projector() const {

	return projector;
}


bool Light::_can_gizmo_scale() const {

	return false;
}


static void _make_sphere(int p_lats, int p_lons, float p_radius,  Ref<SurfaceTool> p_tool) {


	p_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

	for(int i = 1; i <= p_lats; i++) {
		double lat0 = Math_PI * (-0.5 + (double) (i - 1) / p_lats);
		double z0  = Math::sin(lat0);
		double zr0 =  Math::cos(lat0);

		double lat1 = Math_PI * (-0.5 + (double) i / p_lats);
		double z1 = Math::sin(lat1);
		double zr1 = Math::cos(lat1);

		for(int j = p_lons; j >= 1; j--) {

			double lng0 = 2 * Math_PI * (double) (j - 1) / p_lons;
			double x0 = Math::cos(lng0);
			double y0 = Math::sin(lng0);

			double lng1 = 2 * Math_PI * (double) (j) / p_lons;
			double x1 = Math::cos(lng1);
			double y1 = Math::sin(lng1);


			Vector3 v[4]={
				Vector3(x1 * zr0, z0, y1 *zr0),
				Vector3(x1 * zr1, z1, y1 *zr1),
				Vector3(x0 * zr1, z1, y0 *zr1),
				Vector3(x0 * zr0, z0, y0 *zr0)
			};

#define ADD_POINT(m_idx) \
	p_tool->add_normal(v[m_idx]);\
	p_tool->add_vertex(v[m_idx]*p_radius);

			ADD_POINT(0);
			ADD_POINT(1);
			ADD_POINT(2);

			ADD_POINT(2);
			ADD_POINT(3);
			ADD_POINT(0);
		}
	}

}

RES Light::_get_gizmo_geometry() const {


	Ref<FixedMaterial> mat_area( memnew( FixedMaterial ));

	mat_area->set_parameter( FixedMaterial::PARAM_DIFFUSE,Color(0.7,0.6,0.0,0.05) );
	mat_area->set_parameter( FixedMaterial::PARAM_EMISSION,Color(0.7,0.7,0.7) );
	mat_area->set_blend_mode( Material::BLEND_MODE_ADD );
	mat_area->set_flag(Material::FLAG_DOUBLE_SIDED,true);
//	mat_area->set_hint(Material::HINT_NO_DEPTH_DRAW,true);

	Ref<FixedMaterial> mat_light( memnew( FixedMaterial ));

	mat_light->set_parameter( FixedMaterial::PARAM_DIFFUSE, Color(1.0,1.0,0.8,0.9) );
	mat_light->set_flag(Material::FLAG_UNSHADED,true);

	Ref< Mesh > mesh;

	Ref<SurfaceTool> surftool( memnew( SurfaceTool ));
	
	switch(type) {
	
		case VisualServer::LIGHT_DIRECTIONAL: {


			mat_area->set_parameter( FixedMaterial::PARAM_DIFFUSE,Color(0.9,0.8,0.1,0.8) );
			mat_area->set_blend_mode( Material::BLEND_MODE_MIX);
			mat_area->set_flag(Material::FLAG_DOUBLE_SIDED,false);
			mat_area->set_flag(Material::FLAG_UNSHADED,true);

			_make_sphere( 5,5,0.6, surftool );
			surftool->set_material(mat_light);
			mesh=surftool->commit(mesh);

	//		float radius=1;

			surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

			const int arrow_points=5;
			Vector3 arrow[arrow_points]={
				Vector3(0,0,2),
				Vector3(1,1,2),
				Vector3(1,1,-1),
				Vector3(2,2,-1),
				Vector3(0,0,-3)
			};

			int arrow_sides=4;


			for(int i = 0; i < arrow_sides ; i++) {


				Matrix3 ma(Vector3(0,0,1),Math_PI*2*float(i)/arrow_sides);
				Matrix3 mb(Vector3(0,0,1),Math_PI*2*float(i+1)/arrow_sides);


				for(int j=0;j<arrow_points-1;j++) {

					Vector3 points[4]={
						ma.xform(arrow[j]),
						mb.xform(arrow[j]),
						mb.xform(arrow[j+1]),
						ma.xform(arrow[j+1]),
					};

					Vector3 n = Plane(points[0],points[1],points[2]).normal;

					surftool->add_normal(n);
					surftool->add_vertex(points[0]);
					surftool->add_normal(n);
					surftool->add_vertex(points[1]);
					surftool->add_normal(n);
					surftool->add_vertex(points[2]);

					surftool->add_normal(n);
					surftool->add_vertex(points[0]);
					surftool->add_normal(n);
					surftool->add_vertex(points[2]);
					surftool->add_normal(n);
					surftool->add_vertex(points[3]);


				}


			}

			surftool->set_material(mat_area);
			mesh=surftool->commit(mesh);



		} break;
		case VisualServer::LIGHT_OMNI: {


			_make_sphere( 20,20,vars[PARAM_RADIUS],  surftool );
			surftool->set_material(mat_area);
			mesh=surftool->commit(mesh);
			_make_sphere(5,5, 0.1, surftool );
			surftool->set_material(mat_light);
			mesh=surftool->commit(mesh);
		} break;
				
		case VisualServer::LIGHT_SPOT: {
	
			_make_sphere( 5,5,0.1, surftool );
			surftool->set_material(mat_light);
			mesh=surftool->commit(mesh);

			// make cone
			int points=24;
			float len=vars[PARAM_RADIUS];
			float size=Math::tan(Math::deg2rad(vars[PARAM_SPOT_ANGLE]))*len;

			surftool->begin(Mesh::PRIMITIVE_TRIANGLES);
			
			for(int i = 0; i < points; i++) {
			
				float x0=Math::sin(i * Math_PI * 2 / points);
				float y0=Math::cos(i * Math_PI * 2 / points);
				float x1=Math::sin((i+1) * Math_PI * 2 / points);
				float y1=Math::cos((i+1) * Math_PI * 2 / points);
				
				Vector3 v1=Vector3(x0*size,y0*size,-len).normalized()*len;
				Vector3 v2=Vector3(x1*size,y1*size,-len).normalized()*len;

				Vector3 v3=Vector3(0,0,0);
				Vector3 v4=Vector3(0,0,v1.z);

				Vector3 n = Plane(v1,v2,v3).normal;
			

				surftool->add_normal(n);
				surftool->add_vertex(v1);
				surftool->add_normal(n);
				surftool->add_vertex(v2);
				surftool->add_normal(n);
				surftool->add_vertex(v3);

				n=Vector3(0,0,-1);

				surftool->add_normal(n);
				surftool->add_vertex(v1);
				surftool->add_normal(n);
				surftool->add_vertex(v2);
				surftool->add_normal(n);
				surftool->add_vertex(v4);

			
			}

			surftool->set_material(mat_area);
			mesh=surftool->commit(mesh);


		} break;
	}

	return mesh;
}


AABB Light::get_aabb() const {

	if (type==VisualServer::LIGHT_DIRECTIONAL) {
	
		return AABB( Vector3(-1,-1,-1), Vector3(2, 2, 2 ) );	
		
	} else if (type==VisualServer::LIGHT_OMNI) {
	
		return AABB( Vector3(-1,-1,-1) * vars[PARAM_RADIUS], Vector3(2, 2, 2 ) * vars[PARAM_RADIUS]);
		
	} else if (type==VisualServer::LIGHT_SPOT) {
	
		float len=vars[PARAM_RADIUS];
		float size=Math::tan(Math::deg2rad(vars[PARAM_SPOT_ANGLE]))*len;
		return AABB( Vector3( -size,-size,-len ), Vector3( size*2, size*2, len ) );
	}

	return AABB();
}

DVector<Face3> Light::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
}


void Light::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(p_op,2);
	op=p_op;
	VisualServer::get_singleton()->light_set_operator(light,VS::LightOp(op));

}

void Light::set_bake_mode(BakeMode p_bake_mode) {

	bake_mode=p_bake_mode;
}

Light::BakeMode Light::get_bake_mode() const {

	return bake_mode;
}


Light::Operator Light::get_operator() const {

	return op;
}

void Light::approximate_opengl_attenuation(float p_constant, float p_linear, float p_quadratic,float p_radius_treshold) {

	//this is horrible and must never be used

	float a  = p_quadratic * p_radius_treshold;
	float b  = p_linear    * p_radius_treshold;
	float c  = p_constant  * p_radius_treshold -1;

	float radius=10000;

	if(a == 0) { // solve linear
		float d = Math::abs(-c/b);
		if(d<radius)
			radius=d;


	} else {  // solve quadratic
		// now ad^2 + bd + c = 0, solve quadratic equation:

		float denominator = 2*a;

		if(denominator != 0) {


			float root = b*b - 4*a*c;

			if(root >=0) {

				root = sqrt(root);

				float solution1 = fabs( (-b + root) / denominator);
				float solution2 = fabs( (-b - root) / denominator);

				if(solution1 > radius)
					solution1 = radius;

				if(solution2 > radius)
					solution2 = radius;

				radius = (solution1 > solution2 ? solution1 : solution2);
			}
		}
	}

	float energy=1.0;

	/*if (p_constant>0)
		energy=1.0/p_constant; //energy is this
	else
		energy=8.0; // some high number..
*/

	if (radius==10000)
		radius=100; //bug?

	set_parameter(PARAM_RADIUS,radius);
	set_parameter(PARAM_ENERGY,energy);

}


void Light::_update_visibility() {

	if (!is_inside_tree())
		return;


bool editor_ok=true;

#ifdef TOOLS_ENABLED
	if (editor_only) {
		if (!get_tree()->is_editor_hint()) {
			editor_ok=false;
		} else {
			editor_ok = (get_tree()->get_edited_scene_root() && (this==get_tree()->get_edited_scene_root() || get_owner()==get_tree()->get_edited_scene_root()));
		}
	}
#endif

	VS::get_singleton()->instance_light_set_enabled(get_instance(),is_visible() && enabled && editor_ok);
	_change_notify("geometry/visible");

}


void Light::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE || p_what==NOTIFICATION_VISIBILITY_CHANGED) {
		_update_visibility();
	}
}

void Light::set_enabled(bool p_enabled) {

	enabled=p_enabled;
	_update_visibility();
}

bool Light::is_enabled() const{

	return enabled;
}

void Light::set_editor_only(bool p_editor_only) {

	editor_only=p_editor_only;
	_update_visibility();
}

bool Light::is_editor_only() const{

	return editor_only;
}


void Light::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_parameter","variable","value"), &Light::set_parameter );
	ObjectTypeDB::bind_method(_MD("get_parameter"), &Light::get_parameter );
	ObjectTypeDB::bind_method(_MD("set_color","color","value"), &Light::set_color );
	ObjectTypeDB::bind_method(_MD("get_color"), &Light::get_color );
	ObjectTypeDB::bind_method(_MD("set_project_shadows","enable"), &Light::set_project_shadows );
	ObjectTypeDB::bind_method(_MD("has_project_shadows"), &Light::has_project_shadows );
	ObjectTypeDB::bind_method(_MD("set_projector","projector:Texture"), &Light::set_projector );
	ObjectTypeDB::bind_method(_MD("get_projector:Texture"), &Light::get_projector );
	ObjectTypeDB::bind_method(_MD("set_operator","operator"), &Light::set_operator );
	ObjectTypeDB::bind_method(_MD("get_operator"), &Light::get_operator );
	ObjectTypeDB::bind_method(_MD("set_bake_mode","bake_mode"), &Light::set_bake_mode );
	ObjectTypeDB::bind_method(_MD("get_bake_mode"), &Light::get_bake_mode );
	ObjectTypeDB::bind_method(_MD("set_enabled","enabled"), &Light::set_enabled );
	ObjectTypeDB::bind_method(_MD("is_enabled"), &Light::is_enabled );
	ObjectTypeDB::bind_method(_MD("set_editor_only","editor_only"), &Light::set_editor_only );
	ObjectTypeDB::bind_method(_MD("is_editor_only"), &Light::is_editor_only );


	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "params/enabled"), _SCS("set_enabled"), _SCS("is_enabled"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "params/editor_only"), _SCS("set_editor_only"), _SCS("is_editor_only"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "params/bake_mode",PROPERTY_HINT_ENUM,"Disabled,Indirect,Indirect+Shadows,Full"), _SCS("set_bake_mode"), _SCS("get_bake_mode"));
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/energy", PROPERTY_HINT_EXP_RANGE, "0,64,0.01"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_ENERGY );
	/*
	if (type == VisualServer::LIGHT_OMNI || type == VisualServer::LIGHT_SPOT) {
		ADD_PROPERTY( PropertyInfo( Variant::REAL, "params/radius", PROPERTY_HINT_RANGE, "0.01,4096,0.01"));
		ADD_PROPERTY( PropertyInfo( Variant::REAL, "params/attenuation", PROPERTY_HINT_RANGE, "0,8,0.01"));
	}

	if (type == VisualServer::LIGHT_SPOT) {
		ADD_PROPERTY( PropertyInfo( Variant::REAL, "params/spot_angle", PROPERTY_HINT_RANGE, "0.01,90.0,0.01"));
		ADD_PROPERTY( PropertyInfo( Variant::REAL, "params/spot_attenuation", PROPERTY_HINT_RANGE, "0,8,0.01"));

	}*/

	ADD_PROPERTYI( PropertyInfo( Variant::COLOR, "colors/diffuse"), _SCS("set_color"), _SCS("get_color"),COLOR_DIFFUSE);
	ADD_PROPERTYI( PropertyInfo( Variant::COLOR, "colors/specular"), _SCS("set_color"), _SCS("get_color"),COLOR_SPECULAR);
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "shadow/shadow"), _SCS("set_project_shadows"), _SCS("has_project_shadows"));
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/darkening", PROPERTY_HINT_RANGE, "0,1,0.01"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SHADOW_DARKENING );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/z_offset", PROPERTY_HINT_RANGE, "0,128,0.001"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SHADOW_Z_OFFSET);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/z_slope_scale", PROPERTY_HINT_RANGE, "0,128,0.001"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SHADOW_Z_SLOPE_SCALE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/esm_multiplier", PROPERTY_HINT_RANGE, "1.0,512.0,0.1"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SHADOW_ESM_MULTIPLIER);
	ADD_PROPERTYI( PropertyInfo( Variant::INT, "shadow/blur_passes", PROPERTY_HINT_RANGE, "0,4,1"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SHADOW_BLUR_PASSES);
	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "projector",PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_projector"), _SCS("get_projector"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "operator",PROPERTY_HINT_ENUM,"Add,Sub"), _SCS("set_operator"), _SCS("get_operator"));


	BIND_CONSTANT( PARAM_RADIUS );
	BIND_CONSTANT( PARAM_ENERGY );
	BIND_CONSTANT( PARAM_ATTENUATION );
	BIND_CONSTANT( PARAM_SPOT_ANGLE );
	BIND_CONSTANT( PARAM_SPOT_ATTENUATION );
	BIND_CONSTANT( PARAM_SHADOW_DARKENING );
	BIND_CONSTANT( PARAM_SHADOW_Z_OFFSET );


	BIND_CONSTANT( COLOR_DIFFUSE );
	BIND_CONSTANT( COLOR_SPECULAR );	

	BIND_CONSTANT( BAKE_MODE_DISABLED );
	BIND_CONSTANT( BAKE_MODE_INDIRECT );
	BIND_CONSTANT( BAKE_MODE_INDIRECT_AND_SHADOWS );
	BIND_CONSTANT( BAKE_MODE_FULL );


}


Light::Light(VisualServer::LightType p_type) {

	type=p_type;
	light=VisualServer::get_singleton()->light_create(p_type);

	set_parameter(PARAM_SPOT_ATTENUATION,1.0);
	set_parameter(PARAM_SPOT_ANGLE,30.0);
	set_parameter(PARAM_RADIUS,2.0);
	set_parameter(PARAM_ENERGY,1.0);
	set_parameter(PARAM_ATTENUATION,1.0);
	set_parameter(PARAM_SHADOW_DARKENING,0.0);
	set_parameter(PARAM_SHADOW_Z_OFFSET,0.05);
	set_parameter(PARAM_SHADOW_Z_SLOPE_SCALE,0);
	set_parameter(PARAM_SHADOW_ESM_MULTIPLIER,60);
	set_parameter(PARAM_SHADOW_BLUR_PASSES,1);


	set_color( COLOR_DIFFUSE, Color(1,1,1));
	set_color( COLOR_SPECULAR, Color(1,1,1));

	op=OPERATOR_ADD;
	set_project_shadows( false );
	set_base(light);
	enabled=true;
	editor_only=false;
	bake_mode=BAKE_MODE_DISABLED;

}


Light::Light() {

	type=VisualServer::LIGHT_DIRECTIONAL;
	ERR_PRINT("Light shouldn't be instanced dircetly, use the subtypes.");
}


Light::~Light() {

	if (light.is_valid())
		VisualServer::get_singleton()->free(light);
}
/////////////////////////////////////////


void DirectionalLight::set_shadow_mode(ShadowMode p_mode) {

	shadow_mode=p_mode;
	VS::get_singleton()->light_directional_set_shadow_mode(light,(VS::LightDirectionalShadowMode)p_mode);

}

DirectionalLight::ShadowMode DirectionalLight::get_shadow_mode() const{

	return shadow_mode;
}

void DirectionalLight::set_shadow_param(ShadowParam p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,3);
	shadow_param[p_param]=p_value;
	VS::get_singleton()->light_directional_set_shadow_param(light,VS::LightDirectionalShadowParam(p_param),p_value);
}

float DirectionalLight::get_shadow_param(ShadowParam p_param) const {
	ERR_FAIL_INDEX_V(p_param,3,0);
	return shadow_param[p_param];
}

void DirectionalLight::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_shadow_mode","mode"),&DirectionalLight::set_shadow_mode);
	ObjectTypeDB::bind_method(_MD("get_shadow_mode"),&DirectionalLight::get_shadow_mode);
	ObjectTypeDB::bind_method(_MD("set_shadow_param","param","value"),&DirectionalLight::set_shadow_param);
	ObjectTypeDB::bind_method(_MD("get_shadow_param","param"),&DirectionalLight::get_shadow_param);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"shadow/mode",PROPERTY_HINT_ENUM,"Orthogonal,Perspective,PSSM 2 Splits,PSSM 4 Splits"),_SCS("set_shadow_mode"),_SCS("get_shadow_mode"));
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"shadow/max_distance",PROPERTY_HINT_EXP_RANGE,"0.00,99999,0.01"),_SCS("set_shadow_param"),_SCS("get_shadow_param"), SHADOW_PARAM_MAX_DISTANCE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"shadow/split_weight",PROPERTY_HINT_RANGE,"0.01,1.0,0.01"),_SCS("set_shadow_param"),_SCS("get_shadow_param"), SHADOW_PARAM_PSSM_SPLIT_WEIGHT);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"shadow/zoffset_scale",PROPERTY_HINT_RANGE,"0.01,1024.0,0.01"),_SCS("set_shadow_param"),_SCS("get_shadow_param"), SHADOW_PARAM_PSSM_ZOFFSET_SCALE);

	BIND_CONSTANT( SHADOW_ORTHOGONAL );
	BIND_CONSTANT( SHADOW_PERSPECTIVE );
	BIND_CONSTANT( SHADOW_PARALLEL_2_SPLITS );
	BIND_CONSTANT( SHADOW_PARALLEL_4_SPLITS );
	BIND_CONSTANT( SHADOW_PARAM_MAX_DISTANCE );
	BIND_CONSTANT( SHADOW_PARAM_PSSM_SPLIT_WEIGHT );
	BIND_CONSTANT( SHADOW_PARAM_PSSM_ZOFFSET_SCALE );

}


DirectionalLight::DirectionalLight() : Light( VisualServer::LIGHT_DIRECTIONAL ) {

	shadow_mode=SHADOW_ORTHOGONAL;
	shadow_param[SHADOW_PARAM_MAX_DISTANCE]=0;
	shadow_param[SHADOW_PARAM_PSSM_SPLIT_WEIGHT]=0.5;
	shadow_param[SHADOW_PARAM_PSSM_ZOFFSET_SCALE]=2.0;


}


void OmniLight::_bind_methods() {

	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/radius", PROPERTY_HINT_EXP_RANGE, "0.2,4096,0.01"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_RADIUS );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_ATTENUATION );

}

void SpotLight::_bind_methods() {

	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/radius", PROPERTY_HINT_EXP_RANGE, "0.2,4096,0.01"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_RADIUS );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_ATTENUATION );

	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/spot_angle", PROPERTY_HINT_RANGE, "0.01,89.9,0.01"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SPOT_ANGLE );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/spot_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), _SCS("set_parameter"), _SCS("get_parameter"), PARAM_SPOT_ATTENUATION );

}


