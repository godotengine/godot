/*************************************************************************/
/*  environment.cpp                                                      */
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
#include "environment.h"
#include "texture.h"
#include "globals.h"
#include "servers/visual_server.h"

RID Environment::get_rid() const {

	return environment;
}


void Environment::set_background(BGMode p_bg) {

	bg_mode=p_bg;
	VS::get_singleton()->environment_set_background(environment,VS::EnvironmentBG(p_bg));
	_change_notify();
}

void Environment::set_skybox(const Ref<CubeMap>& p_skybox){

	bg_skybox=p_skybox;

	RID sb_rid;
	if (bg_skybox.is_valid())
		sb_rid=bg_skybox->get_rid();
	print_line("skybox valid: "+itos(sb_rid.is_valid()));

	VS::get_singleton()->environment_set_skybox(environment,sb_rid,Globals::get_singleton()->get("rendering/skybox/radiance_cube_resolution"),Globals::get_singleton()->get("rendering/skybox/iradiance_cube_resolution"));
}

void Environment::set_skybox_scale(float p_scale) {

	bg_skybox_scale=p_scale;
	VS::get_singleton()->environment_set_skybox_scale(environment,p_scale);
}

void Environment::set_bg_color(const Color& p_color){

	bg_color=p_color;
	VS::get_singleton()->environment_set_bg_color(environment,p_color);
}
void Environment::set_bg_energy(float p_energy){

	bg_energy=p_energy;
	VS::get_singleton()->environment_set_bg_energy(environment,p_energy);
}
void Environment::set_canvas_max_layer(int p_max_layer){

	bg_canvas_max_layer=p_max_layer;
	VS::get_singleton()->environment_set_canvas_max_layer(environment,p_max_layer);
}
void Environment::set_ambient_light_color(const Color& p_color){

	ambient_color=p_color;
	VS::get_singleton()->environment_set_ambient_light(environment,ambient_color,ambient_energy,ambient_skybox_energy);
}
void Environment::set_ambient_light_energy(float p_energy){

	ambient_energy=p_energy;
	VS::get_singleton()->environment_set_ambient_light(environment,ambient_color,ambient_energy,ambient_skybox_energy);
}
void Environment::set_ambient_light_skybox_energy(float p_energy){

	ambient_skybox_energy=p_energy;
	VS::get_singleton()->environment_set_ambient_light(environment,ambient_color,ambient_energy,ambient_skybox_energy);
}

Environment::BGMode Environment::get_background() const{

	return bg_mode;
}
Ref<CubeMap> Environment::get_skybox() const{

	return bg_skybox;
}

float Environment::get_skybox_scale() const {

	return bg_skybox_scale;
}

Color Environment::get_bg_color() const{

	return bg_color;
}
float Environment::get_bg_energy() const{

	return bg_energy;
}
int Environment::get_canvas_max_layer() const{

	return bg_canvas_max_layer;
}
Color Environment::get_ambient_light_color() const{

	return ambient_color;
}
float Environment::get_ambient_light_energy() const{

	return ambient_energy;
}
float Environment::get_ambient_light_skybox_energy() const{

	return ambient_skybox_energy;
}



void Environment::_validate_property(PropertyInfo& property) const {

	if (property.name=="background/skybox" || property.name=="ambient_light/skybox_energy") {
		if (bg_mode!=BG_SKYBOX) {
			property.usage=PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name=="background/color") {
		if (bg_mode!=BG_COLOR) {
			property.usage=PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name=="background/canvas_max_layer") {
		if (bg_mode!=BG_CANVAS) {
			property.usage=PROPERTY_USAGE_NOEDITOR;
		}
	}

}

void Environment::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_background","mode"),&Environment::set_background);
	ObjectTypeDB::bind_method(_MD("set_skybox","skybox:CubeMap"),&Environment::set_skybox);
	ObjectTypeDB::bind_method(_MD("set_skybox_scale","scale"),&Environment::set_skybox_scale);
	ObjectTypeDB::bind_method(_MD("set_bg_color","color"),&Environment::set_bg_color);
	ObjectTypeDB::bind_method(_MD("set_bg_energy","energy"),&Environment::set_bg_energy);
	ObjectTypeDB::bind_method(_MD("set_canvas_max_layer","layer"),&Environment::set_canvas_max_layer);
	ObjectTypeDB::bind_method(_MD("set_ambient_light_color","color"),&Environment::set_ambient_light_color);
	ObjectTypeDB::bind_method(_MD("set_ambient_light_energy","energy"),&Environment::set_ambient_light_energy);
	ObjectTypeDB::bind_method(_MD("set_ambient_light_skybox_energy","energy"),&Environment::set_ambient_light_skybox_energy);


	ObjectTypeDB::bind_method(_MD("get_background"),&Environment::get_background);
	ObjectTypeDB::bind_method(_MD("get_skybox:CubeMap"),&Environment::get_skybox);
	ObjectTypeDB::bind_method(_MD("get_skybox_scale"),&Environment::get_skybox_scale);
	ObjectTypeDB::bind_method(_MD("get_bg_color"),&Environment::get_bg_color);
	ObjectTypeDB::bind_method(_MD("get_bg_energy"),&Environment::get_bg_energy);
	ObjectTypeDB::bind_method(_MD("get_canvas_max_layer"),&Environment::get_canvas_max_layer);
	ObjectTypeDB::bind_method(_MD("get_ambient_light_color"),&Environment::get_ambient_light_color);
	ObjectTypeDB::bind_method(_MD("get_ambient_light_energy"),&Environment::get_ambient_light_energy);
	ObjectTypeDB::bind_method(_MD("get_ambient_light_skybox_energy"),&Environment::get_ambient_light_skybox_energy);


	ADD_PROPERTY(PropertyInfo(Variant::INT,"background/mode",PROPERTY_HINT_ENUM,"Clear Color,Custom Color,Skybox,Canvas,Keep"),_SCS("set_background"),_SCS("get_background") );
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"background/skybox",PROPERTY_HINT_RESOURCE_TYPE,"CubeMap"),_SCS("set_skybox"),_SCS("get_skybox") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"background/skybox_scale",PROPERTY_HINT_RANGE,"0,32,0.01"),_SCS("set_skybox_scale"),_SCS("get_skybox_scale") );
	ADD_PROPERTY(PropertyInfo(Variant::COLOR,"background/color"),_SCS("set_bg_color"),_SCS("get_bg_color") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"background/energy",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_bg_energy"),_SCS("get_bg_energy") );
	ADD_PROPERTY(PropertyInfo(Variant::INT,"background/canvas_max_layer",PROPERTY_HINT_RANGE,"-1000,1000,1"),_SCS("set_canvas_max_layer"),_SCS("get_canvas_max_layer") );
	ADD_PROPERTY(PropertyInfo(Variant::COLOR,"ambient_light/color"),_SCS("set_ambient_light_color"),_SCS("get_ambient_light_color") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ambient_light/energy",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_ambient_light_energy"),_SCS("get_ambient_light_energy") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ambient_light/skybox_energy",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_ambient_light_skybox_energy"),_SCS("get_ambient_light_skybox_energy") );

	GLOBAL_DEF("rendering/skybox/irradiance_cube_resolution",256);
	GLOBAL_DEF("rendering/skybox/radiance_cube_resolution",64);

	BIND_CONSTANT(BG_KEEP);
	BIND_CONSTANT(BG_CLEAR_COLOR);
	BIND_CONSTANT(BG_COLOR);
	BIND_CONSTANT(BG_SKYBOX);
	BIND_CONSTANT(BG_CANVAS);
	BIND_CONSTANT(BG_MAX);
	BIND_CONSTANT(GLOW_BLEND_MODE_ADDITIVE);
	BIND_CONSTANT(GLOW_BLEND_MODE_SCREEN);
	BIND_CONSTANT(GLOW_BLEND_MODE_SOFTLIGHT);
	BIND_CONSTANT(GLOW_BLEND_MODE_DISABLED);

}

Environment::Environment() {

	bg_mode=BG_CLEAR_COLOR;
	bg_energy=1.0;
	bg_canvas_max_layer=0;
	ambient_energy=1.0;
	ambient_skybox_energy=0;


	environment = VS::get_singleton()->environment_create();

}

Environment::~Environment() {

	VS::get_singleton()->free(environment);
}
