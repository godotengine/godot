/*************************************************************************/
/*  light.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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



bool Light::_can_gizmo_scale() const {

	return false;
}


void Light::set_param(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	param[p_param]=p_value;

	VS::get_singleton()->light_set_param(light,VS::LightParam(p_param),p_value);

	if (p_param==PARAM_SPOT_ANGLE || p_param==PARAM_RANGE) {
		update_gizmo();;
	}


}

float Light::get_param(Param p_param) const{

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return param[p_param];

}

void Light::set_shadow(bool p_enable){

	shadow=p_enable;
	VS::get_singleton()->light_set_shadow(light,p_enable);

}
bool Light::has_shadow() const{

	return shadow;
}

void Light::set_negative(bool p_enable){

	negative=p_enable;
	VS::get_singleton()->light_set_negative(light,p_enable);
}
bool Light::is_negative() const{

	return negative;
}

void Light::set_cull_mask(uint32_t p_cull_mask){

	cull_mask=p_cull_mask;
	VS::get_singleton()->light_set_cull_mask(light,p_cull_mask);

}
uint32_t Light::get_cull_mask() const{

	return cull_mask;
}

void Light::set_color(const Color& p_color){

	color=p_color;
	VS::get_singleton()->light_set_color(light,p_color);
}
Color Light::get_color() const{

	return color;
}


AABB Light::get_aabb() const {


	if (type==VisualServer::LIGHT_DIRECTIONAL) {

		return AABB( Vector3(-1,-1,-1), Vector3(2, 2, 2 ) );

	} else if (type==VisualServer::LIGHT_OMNI) {

		return AABB( Vector3(-1,-1,-1) * param[PARAM_RANGE], Vector3(2, 2, 2 ) * param[PARAM_RANGE]);

	} else if (type==VisualServer::LIGHT_SPOT) {

		float len=param[PARAM_RANGE];
		float size=Math::tan(Math::deg2rad(param[PARAM_SPOT_ANGLE]))*len;
		return AABB( Vector3( -size,-size,-len ), Vector3( size*2, size*2, len ) );
	}

	return AABB();
}

DVector<Face3> Light::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
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

	//VS::get_singleton()->instance_light_set_enabled(get_instance(),is_visible() && editor_ok);
	_change_notify("geometry/visible");

}


void Light::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE || p_what==NOTIFICATION_VISIBILITY_CHANGED) {
		_update_visibility();
	}
}


void Light::set_editor_only(bool p_editor_only) {

	editor_only=p_editor_only;
	_update_visibility();
}

bool Light::is_editor_only() const{

	return editor_only;
}


void Light::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_editor_only","editor_only"), &Light::set_editor_only );
	ObjectTypeDB::bind_method(_MD("is_editor_only"), &Light::is_editor_only );


	ObjectTypeDB::bind_method(_MD("set_param","param","value"), &Light::set_param );
	ObjectTypeDB::bind_method(_MD("get_param","param"), &Light::get_param );

	ObjectTypeDB::bind_method(_MD("set_shadow","enabled"), &Light::set_shadow );
	ObjectTypeDB::bind_method(_MD("has_shadow"), &Light::has_shadow );

	ObjectTypeDB::bind_method(_MD("set_negative","enabled"), &Light::set_negative );
	ObjectTypeDB::bind_method(_MD("is_negative"), &Light::is_negative );

	ObjectTypeDB::bind_method(_MD("set_cull_mask","cull_mask"), &Light::set_cull_mask );
	ObjectTypeDB::bind_method(_MD("get_cull_mask"), &Light::get_cull_mask );

	ObjectTypeDB::bind_method(_MD("set_color","color"), &Light::set_color );
	ObjectTypeDB::bind_method(_MD("get_color"), &Light::get_color );

	ADD_PROPERTY( PropertyInfo( Variant::COLOR, "light/color"), _SCS("set_color"), _SCS("get_color"));
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "light/energy"), _SCS("set_param"), _SCS("get_param"), PARAM_ENERGY);
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "light/negative"), _SCS("set_negative"), _SCS("is_negative"));
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "light/specular"), _SCS("set_param"), _SCS("get_param"), PARAM_SPECULAR);
	ADD_PROPERTY( PropertyInfo( Variant::INT, "light/cull_mask"), _SCS("set_cull_mask"), _SCS("get_cull_mask"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "shadow/enabled"), _SCS("set_shadow"), _SCS("has_shadow"));
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/darkness"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_DARKNESS);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/normal_bias"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_NORMAL_BIAS);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/bias"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_BIAS);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/bias_split_scale"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_BIAS_SPLIT_SCALE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "shadow/max_distance"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_MAX_DISTANCE);

	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "editor/editor_only"), _SCS("set_editor_only"), _SCS("is_editor_only"));

	BIND_CONSTANT( PARAM_ENERGY );
	BIND_CONSTANT( PARAM_SPECULAR );
	BIND_CONSTANT( PARAM_RANGE );
	BIND_CONSTANT( PARAM_ATTENUATION );
	BIND_CONSTANT( PARAM_SPOT_ANGLE );
	BIND_CONSTANT( PARAM_SPOT_ATTENUATION );
	BIND_CONSTANT( PARAM_SHADOW_MAX_DISTANCE );
	BIND_CONSTANT( PARAM_SHADOW_DARKNESS );
	BIND_CONSTANT( PARAM_SHADOW_SPLIT_1_OFFSET );
	BIND_CONSTANT( PARAM_SHADOW_SPLIT_2_OFFSET );
	BIND_CONSTANT( PARAM_SHADOW_SPLIT_3_OFFSET );
	BIND_CONSTANT( PARAM_SHADOW_NORMAL_BIAS );
	BIND_CONSTANT( PARAM_SHADOW_BIAS );
	BIND_CONSTANT( PARAM_SHADOW_BIAS_SPLIT_SCALE );
	BIND_CONSTANT( PARAM_MAX );


}


Light::Light(VisualServer::LightType p_type) {

	type=p_type;
	light=VisualServer::get_singleton()->light_create(p_type);
	VS::get_singleton()->instance_set_base(get_instance(),light);

	editor_only=false;
	set_color(Color(1,1,1,1));
	set_shadow(false);
	set_negative(false);
	set_cull_mask(0xFFFFFFFF);

	set_param(PARAM_ENERGY,1);
	set_param(PARAM_SPECULAR,1);
	set_param(PARAM_RANGE,5);
	set_param(PARAM_ATTENUATION,1);
	set_param(PARAM_SPOT_ANGLE,45);
	set_param(PARAM_SPOT_ATTENUATION,1);
	set_param(PARAM_SHADOW_MAX_DISTANCE,0);
	set_param(PARAM_SHADOW_DARKNESS,0);
	set_param(PARAM_SHADOW_SPLIT_1_OFFSET,0.1);
	set_param(PARAM_SHADOW_SPLIT_2_OFFSET,0.2);
	set_param(PARAM_SHADOW_SPLIT_3_OFFSET,0.5);
	set_param(PARAM_SHADOW_NORMAL_BIAS,0.1);
	set_param(PARAM_SHADOW_BIAS,0.1);
	set_param(PARAM_SHADOW_BIAS_SPLIT_SCALE,0.1);

}


Light::Light() {

	type=VisualServer::LIGHT_DIRECTIONAL;
	ERR_PRINT("Light shouldn't be instanced dircetly, use the subtypes.");
}


Light::~Light() {

	VS::get_singleton()->instance_set_base(get_instance(),RID());

	if (light.is_valid())
		VisualServer::get_singleton()->free(light);
}
/////////////////////////////////////////

void DirectionalLight::set_shadow_mode(ShadowMode p_mode) {

	shadow_mode=p_mode;
	VS::get_singleton()->light_directional_set_shadow_mode(light,VS::LightDirectionalShadowMode(p_mode));
}

DirectionalLight::ShadowMode DirectionalLight::get_shadow_mode() const {

	return shadow_mode;
}

void DirectionalLight::set_blend_splits(bool p_enable) {

	blend_splits=p_enable;
}

bool DirectionalLight::is_blend_splits_enabled() const {

	return blend_splits;
}


void DirectionalLight::_bind_methods() {

	ObjectTypeDB::bind_method( _MD("set_shadow_mode","mode"),&DirectionalLight::set_shadow_mode);
	ObjectTypeDB::bind_method( _MD("get_shadow_mode"),&DirectionalLight::get_shadow_mode);

	ObjectTypeDB::bind_method( _MD("set_blend_splits","enabled"),&DirectionalLight::set_blend_splits);
	ObjectTypeDB::bind_method( _MD("is_blend_splits_enabled"),&DirectionalLight::is_blend_splits_enabled);

	ADD_PROPERTY( PropertyInfo( Variant::INT, "directional/shadow_mode",PROPERTY_HINT_ENUM,"Orthogonal,PSSM 2 Splits,PSSM 4 Splits"), _SCS("set_shadow_mode"), _SCS("get_shadow_mode"));
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "directional/split_1"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_SPLIT_1_OFFSET);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "directional/split_2"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_SPLIT_2_OFFSET);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "directional/split_3"), _SCS("set_param"), _SCS("get_param"), PARAM_SHADOW_SPLIT_3_OFFSET);
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "directional/blend_splits"), _SCS("set_blend_splits"), _SCS("is_blend_splits_enabled"));

	BIND_CONSTANT( SHADOW_ORTHOGONAL );
	BIND_CONSTANT( SHADOW_PARALLEL_2_SPLITS );
	BIND_CONSTANT( SHADOW_PARALLEL_4_SPLITS );

}


DirectionalLight::DirectionalLight() : Light( VisualServer::LIGHT_DIRECTIONAL ) {

	set_shadow_mode(SHADOW_PARALLEL_4_SPLITS);
	blend_splits=false;
}

void OmniLight::set_shadow_mode(ShadowMode p_mode) {

	shadow_mode=p_mode;
	VS::get_singleton()->light_omni_set_shadow_mode(light,VS::LightOmniShadowMode(p_mode));
}

OmniLight::ShadowMode OmniLight::get_shadow_mode() const{

	return shadow_mode;
}

void OmniLight::set_shadow_detail(ShadowDetail p_detail){

	shadow_detail=p_detail;
	VS::get_singleton()->light_omni_set_shadow_detail(light,VS::LightOmniShadowDetail(p_detail));
}
OmniLight::ShadowDetail OmniLight::get_shadow_detail() const{

	return shadow_detail;
}




void OmniLight::_bind_methods() {

	ObjectTypeDB::bind_method( _MD("set_shadow_mode","mode"),&OmniLight::set_shadow_mode);
	ObjectTypeDB::bind_method( _MD("get_shadow_mode"),&OmniLight::get_shadow_mode);

	ObjectTypeDB::bind_method( _MD("set_shadow_detail","detail"),&OmniLight::set_shadow_detail);
	ObjectTypeDB::bind_method( _MD("get_shadow_detail"),&OmniLight::get_shadow_detail);

	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "omni/range"), _SCS("set_param"), _SCS("get_param"), PARAM_RANGE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "omni/attenuation"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION);
	ADD_PROPERTY( PropertyInfo( Variant::INT, "omni/shadow_mode",PROPERTY_HINT_ENUM,"Dual Paraboloid,Cube"), _SCS("set_shadow_mode"), _SCS("get_shadow_mode"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "omni/shadow_detail",PROPERTY_HINT_ENUM,"Vertical,Horizontal"), _SCS("set_shadow_detail"), _SCS("get_shadow_detail"));

}

OmniLight::OmniLight() : Light( VisualServer::LIGHT_OMNI ) {

	set_shadow_mode(SHADOW_DUAL_PARABOLOID);
	set_shadow_detail(SHADOW_DETAIL_HORIZONTAL);

}

void SpotLight::_bind_methods() {

	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "spot/range"), _SCS("set_param"), _SCS("get_param"), PARAM_RANGE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "spot/attenuation"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "spot/spot_angle"), _SCS("set_param"), _SCS("get_param"), PARAM_SPOT_ANGLE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "spot/spot_attenuation"), _SCS("set_param"), _SCS("get_param"), PARAM_SPOT_ATTENUATION);

}


