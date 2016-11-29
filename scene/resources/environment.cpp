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

void Environment::set_skybox(const Ref<SkyBox> &p_skybox){

	bg_skybox=p_skybox;

	RID sb_rid;
	if (bg_skybox.is_valid())
		sb_rid=bg_skybox->get_rid();

	VS::get_singleton()->environment_set_skybox(environment,sb_rid);
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
	VS::get_singleton()->environment_set_ambient_light(environment,ambient_color,ambient_energy,ambient_skybox_contribution);
}
void Environment::set_ambient_light_energy(float p_energy){

	ambient_energy=p_energy;
	VS::get_singleton()->environment_set_ambient_light(environment,ambient_color,ambient_energy,ambient_skybox_contribution);
}
void Environment::set_ambient_light_skybox_contribution(float p_energy){

	ambient_skybox_contribution=p_energy;
	VS::get_singleton()->environment_set_ambient_light(environment,ambient_color,ambient_energy,ambient_skybox_contribution);
}

Environment::BGMode Environment::get_background() const{

	return bg_mode;
}
Ref<SkyBox> Environment::get_skybox() const{

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
float Environment::get_ambient_light_skybox_contribution() const{

	return ambient_skybox_contribution;
}



void Environment::set_tonemapper(ToneMapper p_tone_mapper) {

	tone_mapper=p_tone_mapper;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));
}

Environment::ToneMapper Environment::get_tonemapper() const{

	return tone_mapper;
}

void Environment::set_tonemap_exposure(float p_exposure){

	tonemap_exposure=p_exposure;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));
}

float Environment::get_tonemap_exposure() const{

	return get_tonemap_auto_exposure();
}

void Environment::set_tonemap_white(float p_white){

	tonemap_white=p_white;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));

}
float Environment::get_tonemap_white() const {

	return tonemap_white;
}

void Environment::set_tonemap_auto_exposure(bool p_enabled) {

	tonemap_auto_exposure=p_enabled;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));

}
bool Environment::get_tonemap_auto_exposure() const {

	return tonemap_auto_exposure;
}

void Environment::set_tonemap_auto_exposure_max(float p_auto_exposure_max) {

	tonemap_auto_exposure_max=p_auto_exposure_max;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));

}
float Environment::get_tonemap_auto_exposure_max() const {

	return tonemap_auto_exposure_max;
}

void Environment::set_tonemap_auto_exposure_min(float p_auto_exposure_min) {

	tonemap_auto_exposure_min=p_auto_exposure_min;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));

}
float Environment::get_tonemap_auto_exposure_min() const {

	return tonemap_auto_exposure_min;
}

void Environment::set_tonemap_auto_exposure_speed(float p_auto_exposure_speed) {

	tonemap_auto_exposure_speed=p_auto_exposure_speed;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));

}
float Environment::get_tonemap_auto_exposure_speed() const {

	return tonemap_auto_exposure_speed;
}

void Environment::set_tonemap_auto_exposure_scale(float p_auto_exposure_scale) {

	tonemap_auto_exposure_scale=p_auto_exposure_scale;
	VS::get_singleton()->environment_set_tonemap(environment,tonemap_auto_exposure,tonemap_exposure,tonemap_white,tonemap_auto_exposure_min,tonemap_auto_exposure_max,tonemap_auto_exposure_speed,tonemap_auto_exposure_scale,VS::EnvironmentToneMapper(tone_mapper));

}
float Environment::get_tonemap_auto_exposure_scale() const {

	return tonemap_auto_exposure_scale;
}

void Environment::set_adjustment_enable(bool p_enable) {

	adjustment_enabled=p_enable;
	VS::get_singleton()->environment_set_adjustment(environment,adjustment_enabled,adjustment_brightness,adjustment_contrast,adjustment_saturation,adjustment_color_correction.is_valid()?adjustment_color_correction->get_rid():RID());
}

bool Environment::is_adjustment_enabled() const {

	return adjustment_enabled;
}


void Environment::set_adjustment_brightness(float p_brightness) {

	adjustment_brightness=p_brightness;
	VS::get_singleton()->environment_set_adjustment(environment,adjustment_enabled,adjustment_brightness,adjustment_contrast,adjustment_saturation,adjustment_color_correction.is_valid()?adjustment_color_correction->get_rid():RID());

}
float Environment::get_adjustment_brightness() const {

	return adjustment_brightness;
}

void Environment::set_adjustment_contrast(float p_contrast) {

	adjustment_contrast=p_contrast;
	VS::get_singleton()->environment_set_adjustment(environment,adjustment_enabled,adjustment_brightness,adjustment_contrast,adjustment_saturation,adjustment_color_correction.is_valid()?adjustment_color_correction->get_rid():RID());

}
float Environment::get_adjustment_contrast() const {

	return adjustment_contrast;
}

void Environment::set_adjustment_saturation(float p_saturation) {

	adjustment_saturation=p_saturation;
	VS::get_singleton()->environment_set_adjustment(environment,adjustment_enabled,adjustment_brightness,adjustment_contrast,adjustment_saturation,adjustment_color_correction.is_valid()?adjustment_color_correction->get_rid():RID());

}
float Environment::get_adjustment_saturation() const {

	return adjustment_saturation;
}

void Environment::set_adjustment_color_correction(const Ref<Texture>& p_ramp) {

	adjustment_color_correction=p_ramp;
	VS::get_singleton()->environment_set_adjustment(environment,adjustment_enabled,adjustment_brightness,adjustment_contrast,adjustment_saturation,adjustment_color_correction.is_valid()?adjustment_color_correction->get_rid():RID());

}
Ref<Texture> Environment::get_adjustment_color_correction() const {

	return adjustment_color_correction;
}


void Environment::_validate_property(PropertyInfo& property) const {

	if (property.name=="background/skybox" || property.name=="background/skybox_scale" || property.name=="ambient_light/skybox_contribution") {
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

void Environment::set_ssr_enabled(bool p_enable) {

	ssr_enabled=p_enable;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);
}

bool Environment::is_ssr_enabled() const{

	return ssr_enabled;
}

void Environment::set_ssr_max_steps(int p_steps){

	ssr_max_steps=p_steps;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);

}
int Environment::get_ssr_max_steps() const {

	return ssr_max_steps;
}

void Environment::set_ssr_accel(float p_accel) {

	ssr_accel=p_accel;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);

}
float Environment::get_ssr_accel() const {

	return ssr_accel;
}

void Environment::set_ssr_fade(float p_fade) {

	ssr_fade=p_fade;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);

}
float Environment::get_ssr_fade() const {

	return ssr_fade;
}

void Environment::set_ssr_depth_tolerance(float p_depth_tolerance) {

	ssr_depth_tolerance=p_depth_tolerance;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);

}
float Environment::get_ssr_depth_tolerance() const {

	return ssr_depth_tolerance;
}

void Environment::set_ssr_smooth(bool p_enable) {

	ssr_smooth=p_enable;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);

}
bool Environment::is_ssr_smooth() const {

	return ssr_smooth;
}

void Environment::set_ssr_rough(bool p_enable) {

	ssr_roughness=p_enable;
	VS::get_singleton()->environment_set_ssr(environment,ssr_enabled,ssr_max_steps,ssr_accel,ssr_fade,ssr_depth_tolerance,ssr_smooth,ssr_roughness);

}
bool Environment::is_ssr_rough() const {

	return ssr_roughness;
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
	ObjectTypeDB::bind_method(_MD("set_ambient_light_skybox_contribution","energy"),&Environment::set_ambient_light_skybox_contribution);


	ObjectTypeDB::bind_method(_MD("get_background"),&Environment::get_background);
	ObjectTypeDB::bind_method(_MD("get_skybox:CubeMap"),&Environment::get_skybox);
	ObjectTypeDB::bind_method(_MD("get_skybox_scale"),&Environment::get_skybox_scale);
	ObjectTypeDB::bind_method(_MD("get_bg_color"),&Environment::get_bg_color);
	ObjectTypeDB::bind_method(_MD("get_bg_energy"),&Environment::get_bg_energy);
	ObjectTypeDB::bind_method(_MD("get_canvas_max_layer"),&Environment::get_canvas_max_layer);
	ObjectTypeDB::bind_method(_MD("get_ambient_light_color"),&Environment::get_ambient_light_color);
	ObjectTypeDB::bind_method(_MD("get_ambient_light_energy"),&Environment::get_ambient_light_energy);
	ObjectTypeDB::bind_method(_MD("get_ambient_light_skybox_contribution"),&Environment::get_ambient_light_skybox_contribution);


	ADD_PROPERTY(PropertyInfo(Variant::INT,"background/mode",PROPERTY_HINT_ENUM,"Clear Color,Custom Color,Skybox,Canvas,Keep"),_SCS("set_background"),_SCS("get_background") );
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"background/skybox",PROPERTY_HINT_RESOURCE_TYPE,"SkyBox"),_SCS("set_skybox"),_SCS("get_skybox") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"background/skybox_scale",PROPERTY_HINT_RANGE,"0,32,0.01"),_SCS("set_skybox_scale"),_SCS("get_skybox_scale") );
	ADD_PROPERTY(PropertyInfo(Variant::COLOR,"background/color"),_SCS("set_bg_color"),_SCS("get_bg_color") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"background/energy",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_bg_energy"),_SCS("get_bg_energy") );
	ADD_PROPERTY(PropertyInfo(Variant::INT,"background/canvas_max_layer",PROPERTY_HINT_RANGE,"-1000,1000,1"),_SCS("set_canvas_max_layer"),_SCS("get_canvas_max_layer") );
	ADD_PROPERTY(PropertyInfo(Variant::COLOR,"ambient_light/color"),_SCS("set_ambient_light_color"),_SCS("get_ambient_light_color") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ambient_light/energy",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_ambient_light_energy"),_SCS("get_ambient_light_energy") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ambient_light/skybox_contribution",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_ambient_light_skybox_contribution"),_SCS("get_ambient_light_skybox_contribution") );


	ObjectTypeDB::bind_method(_MD("set_ssr_enabled","enabled"),&Environment::set_ssr_enabled);
	ObjectTypeDB::bind_method(_MD("is_ssr_enabled"),&Environment::is_ssr_enabled);

	ObjectTypeDB::bind_method(_MD("set_ssr_max_steps","max_steps"),&Environment::set_ssr_max_steps);
	ObjectTypeDB::bind_method(_MD("get_ssr_max_steps"),&Environment::get_ssr_max_steps);

	ObjectTypeDB::bind_method(_MD("set_ssr_accel","accel"),&Environment::set_ssr_accel);
	ObjectTypeDB::bind_method(_MD("get_ssr_accel"),&Environment::get_ssr_accel);

	ObjectTypeDB::bind_method(_MD("set_ssr_fade","fade"),&Environment::set_ssr_fade);
	ObjectTypeDB::bind_method(_MD("get_ssr_fade"),&Environment::get_ssr_fade);

	ObjectTypeDB::bind_method(_MD("set_ssr_depth_tolerance","depth_tolerance"),&Environment::set_ssr_depth_tolerance);
	ObjectTypeDB::bind_method(_MD("get_ssr_depth_tolerance"),&Environment::get_ssr_depth_tolerance);

	ObjectTypeDB::bind_method(_MD("set_ssr_smooth","smooth"),&Environment::set_ssr_smooth);
	ObjectTypeDB::bind_method(_MD("is_ssr_smooth"),&Environment::is_ssr_smooth);

	ObjectTypeDB::bind_method(_MD("set_ssr_rough","rough"),&Environment::set_ssr_rough);
	ObjectTypeDB::bind_method(_MD("is_ssr_rough"),&Environment::is_ssr_rough);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"ss_reflections/enable"),_SCS("set_ssr_enabled"),_SCS("is_ssr_enabled") );
	ADD_PROPERTY(PropertyInfo(Variant::INT,"ss_reflections/max_steps",PROPERTY_HINT_RANGE,"1,512,1"),_SCS("set_ssr_max_steps"),_SCS("get_ssr_max_steps") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ss_reflections/accel",PROPERTY_HINT_RANGE,"0,4,0.01"),_SCS("set_ssr_accel"),_SCS("get_ssr_accel") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ss_reflections/fade",PROPERTY_HINT_EXP_EASING),_SCS("set_ssr_fade"),_SCS("get_ssr_fade") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"ss_reflections/depth_tolerance",PROPERTY_HINT_RANGE,"0.1,128,0.1"),_SCS("set_ssr_depth_tolerance"),_SCS("get_ssr_depth_tolerance") );
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"ss_reflections/accel_smooth"),_SCS("set_ssr_smooth"),_SCS("is_ssr_smooth") );
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"ss_reflections/roughness"),_SCS("set_ssr_rough"),_SCS("is_ssr_rough") );


	ObjectTypeDB::bind_method(_MD("set_tonemapper","mode"),&Environment::set_tonemapper);
	ObjectTypeDB::bind_method(_MD("get_tonemapper"),&Environment::get_tonemapper);

	ObjectTypeDB::bind_method(_MD("set_tonemap_exposure","exposure"),&Environment::set_tonemap_exposure);
	ObjectTypeDB::bind_method(_MD("get_tonemap_exposure"),&Environment::get_tonemap_exposure);

	ObjectTypeDB::bind_method(_MD("set_tonemap_white","white"),&Environment::set_tonemap_white);
	ObjectTypeDB::bind_method(_MD("get_tonemap_white"),&Environment::get_tonemap_white);

	ObjectTypeDB::bind_method(_MD("set_tonemap_auto_exposure","auto_exposure"),&Environment::set_tonemap_auto_exposure);
	ObjectTypeDB::bind_method(_MD("get_tonemap_auto_exposure"),&Environment::get_tonemap_auto_exposure);

	ObjectTypeDB::bind_method(_MD("set_tonemap_auto_exposure_max","exposure_max"),&Environment::set_tonemap_auto_exposure_max);
	ObjectTypeDB::bind_method(_MD("get_tonemap_auto_exposure_max"),&Environment::get_tonemap_auto_exposure_max);

	ObjectTypeDB::bind_method(_MD("set_tonemap_auto_exposure_min","exposure_min"),&Environment::set_tonemap_auto_exposure_min);
	ObjectTypeDB::bind_method(_MD("get_tonemap_auto_exposure_min"),&Environment::get_tonemap_auto_exposure_min);

	ObjectTypeDB::bind_method(_MD("set_tonemap_auto_exposure_speed","exposure_speed"),&Environment::set_tonemap_auto_exposure_speed);
	ObjectTypeDB::bind_method(_MD("get_tonemap_auto_exposure_speed"),&Environment::get_tonemap_auto_exposure_speed);

	ObjectTypeDB::bind_method(_MD("set_tonemap_auto_exposure_scale","exposure_scale"),&Environment::set_tonemap_auto_exposure_scale);
	ObjectTypeDB::bind_method(_MD("get_tonemap_auto_exposure_scale"),&Environment::get_tonemap_auto_exposure_scale);




	ADD_PROPERTY(PropertyInfo(Variant::INT,"tonemap/mode",PROPERTY_HINT_ENUM,"Linear,Log,Reindhart,Filmic,Aces"),_SCS("set_tonemapper"),_SCS("get_tonemapper") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"tonemap/exposure",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_tonemap_exposure"),_SCS("get_tonemap_exposure") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"tonemap/white",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_tonemap_white"),_SCS("get_tonemap_white") );
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"auto_exposure/enable"),_SCS("set_tonemap_auto_exposure"),_SCS("get_tonemap_auto_exposure") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"auto_exposure/scale",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_tonemap_auto_exposure_scale"),_SCS("get_tonemap_auto_exposure_scale") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"auto_exposure/min_luma",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_tonemap_auto_exposure_min"),_SCS("get_tonemap_auto_exposure_min") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"auto_exposure/max_luma",PROPERTY_HINT_RANGE,"0,16,0.01"),_SCS("set_tonemap_auto_exposure_max"),_SCS("get_tonemap_auto_exposure_max") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"auto_exposure/speed",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_tonemap_auto_exposure_speed"),_SCS("get_tonemap_auto_exposure_speed") );

	ObjectTypeDB::bind_method(_MD("set_adjustment_enable","enabled"),&Environment::set_adjustment_enable);
	ObjectTypeDB::bind_method(_MD("is_adjustment_enabled"),&Environment::is_adjustment_enabled);

	ObjectTypeDB::bind_method(_MD("set_adjustment_brightness","brightness"),&Environment::set_adjustment_brightness);
	ObjectTypeDB::bind_method(_MD("get_adjustment_brightness"),&Environment::get_adjustment_brightness);

	ObjectTypeDB::bind_method(_MD("set_adjustment_contrast","contrast"),&Environment::set_adjustment_contrast);
	ObjectTypeDB::bind_method(_MD("get_adjustment_contrast"),&Environment::get_adjustment_contrast);

	ObjectTypeDB::bind_method(_MD("set_adjustment_saturation","saturation"),&Environment::set_adjustment_saturation);
	ObjectTypeDB::bind_method(_MD("get_adjustment_saturation"),&Environment::get_adjustment_saturation);

	ObjectTypeDB::bind_method(_MD("set_adjustment_color_correction","color_correction"),&Environment::set_adjustment_color_correction);
	ObjectTypeDB::bind_method(_MD("get_adjustment_color_correction"),&Environment::get_adjustment_color_correction);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"adjustment/enabled"),_SCS("set_adjustment_enable"),_SCS("is_adjustment_enabled") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"adjustment/brightness",PROPERTY_HINT_RANGE,"0.01,8,0.01"),_SCS("set_adjustment_brightness"),_SCS("get_adjustment_brightness") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"adjustment/contrast",PROPERTY_HINT_RANGE,"0.01,8,0.01"),_SCS("set_adjustment_contrast"),_SCS("get_adjustment_contrast") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"adjustment/saturation",PROPERTY_HINT_RANGE,"0.01,8,0.01"),_SCS("set_adjustment_saturation"),_SCS("get_adjustment_saturation") );
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"adjustment/color_correction",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_adjustment_color_correction"),_SCS("get_adjustment_color_correction") );


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
	BIND_CONSTANT(TONE_MAPPER_LINEAR);
	BIND_CONSTANT(TONE_MAPPER_LOG);
	BIND_CONSTANT(TONE_MAPPER_REINHARDT);
	BIND_CONSTANT(TONE_MAPPER_FILMIC);
	BIND_CONSTANT(TONE_MAPPER_ACES_FILMIC);


}

Environment::Environment() {

	bg_mode=BG_CLEAR_COLOR;
	bg_skybox_scale=1.0;
	bg_energy=1.0;
	bg_canvas_max_layer=0;
	ambient_energy=1.0;
	ambient_skybox_contribution=0;


	tone_mapper=TONE_MAPPER_LINEAR;
	tonemap_exposure=1.0;
	tonemap_white=1.0;
	tonemap_auto_exposure=false;
	tonemap_auto_exposure_max=8;
	tonemap_auto_exposure_min=0.4;
	tonemap_auto_exposure_speed=0.5;
	tonemap_auto_exposure_scale=0.4;

	set_tonemapper(tone_mapper); //update

	adjustment_enabled=false;
	adjustment_contrast=1.0;
	adjustment_saturation=1.0;
	adjustment_brightness=1.0;

	set_adjustment_enable(adjustment_enabled); //update

	environment = VS::get_singleton()->environment_create();

	ssr_enabled=false;
	ssr_max_steps=64;
	ssr_accel=0.04;
	ssr_fade=2.0;
	ssr_depth_tolerance=0.2;
	ssr_smooth=true;
	ssr_roughness=true;

}

Environment::~Environment() {

	VS::get_singleton()->free(environment);
}
