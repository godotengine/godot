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


AABB Light::get_aabb() const {

#if 0
	if (type==VisualServer::LIGHT_DIRECTIONAL) {

		return AABB( Vector3(-1,-1,-1), Vector3(2, 2, 2 ) );

	} else if (type==VisualServer::LIGHT_OMNI) {

		return AABB( Vector3(-1,-1,-1) * vars[PARAM_RADIUS], Vector3(2, 2, 2 ) * vars[PARAM_RADIUS]);

	} else if (type==VisualServer::LIGHT_SPOT) {

		float len=vars[PARAM_RADIUS];
		float size=Math::tan(Math::deg2rad(vars[PARAM_SPOT_ANGLE]))*len;
		return AABB( Vector3( -size,-size,-len ), Vector3( size*2, size*2, len ) );
	}
#endif
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


	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "params/editor_only"), _SCS("set_editor_only"), _SCS("is_editor_only"));





}


Light::Light(VisualServer::LightType p_type) {

	type=p_type;
	light=VisualServer::get_singleton()->light_create(p_type);


	editor_only=false;

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


void DirectionalLight::_bind_methods() {



}


DirectionalLight::DirectionalLight() : Light( VisualServer::LIGHT_DIRECTIONAL ) {



}


void OmniLight::_bind_methods() {


}

void SpotLight::_bind_methods() {

}


