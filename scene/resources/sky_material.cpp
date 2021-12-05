/*************************************************************************/
/*  sky_material.cpp                                                     */
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

#include "sky_material.h"

#include "core/version.h"

Mutex ProceduralSkyMaterial::shader_mutex;
RID ProceduralSkyMaterial::shader;

void ProceduralSkyMaterial::set_sky_top_color(const Color &p_sky_top) {
	sky_top_color = p_sky_top;
	RS::get_singleton()->material_set_param(_get_material(), "sky_top_color", sky_top_color);
}

Color ProceduralSkyMaterial::get_sky_top_color() const {
	return sky_top_color;
}

void ProceduralSkyMaterial::set_sky_horizon_color(const Color &p_sky_horizon) {
	sky_horizon_color = p_sky_horizon;
	RS::get_singleton()->material_set_param(_get_material(), "sky_horizon_color", sky_horizon_color);
}

Color ProceduralSkyMaterial::get_sky_horizon_color() const {
	return sky_horizon_color;
}

void ProceduralSkyMaterial::set_sky_curve(float p_curve) {
	sky_curve = p_curve;
	RS::get_singleton()->material_set_param(_get_material(), "sky_curve", sky_curve);
}

float ProceduralSkyMaterial::get_sky_curve() const {
	return sky_curve;
}

void ProceduralSkyMaterial::set_sky_energy(float p_energy) {
	sky_energy = p_energy;
	RS::get_singleton()->material_set_param(_get_material(), "sky_energy", sky_energy);
}

float ProceduralSkyMaterial::get_sky_energy() const {
	return sky_energy;
}

void ProceduralSkyMaterial::set_ground_bottom_color(const Color &p_ground_bottom) {
	ground_bottom_color = p_ground_bottom;
	RS::get_singleton()->material_set_param(_get_material(), "ground_bottom_color", ground_bottom_color);
}

Color ProceduralSkyMaterial::get_ground_bottom_color() const {
	return ground_bottom_color;
}

void ProceduralSkyMaterial::set_ground_horizon_color(const Color &p_ground_horizon) {
	ground_horizon_color = p_ground_horizon;
	RS::get_singleton()->material_set_param(_get_material(), "ground_horizon_color", ground_horizon_color);
}

Color ProceduralSkyMaterial::get_ground_horizon_color() const {
	return ground_horizon_color;
}

void ProceduralSkyMaterial::set_ground_curve(float p_curve) {
	ground_curve = p_curve;
	RS::get_singleton()->material_set_param(_get_material(), "ground_curve", ground_curve);
}

float ProceduralSkyMaterial::get_ground_curve() const {
	return ground_curve;
}

void ProceduralSkyMaterial::set_ground_energy(float p_energy) {
	ground_energy = p_energy;
	RS::get_singleton()->material_set_param(_get_material(), "ground_energy", ground_energy);
}

float ProceduralSkyMaterial::get_ground_energy() const {
	return ground_energy;
}

void ProceduralSkyMaterial::set_sun_angle_max(float p_angle) {
	sun_angle_max = p_angle;
	RS::get_singleton()->material_set_param(_get_material(), "sun_angle_max", Math::deg2rad(sun_angle_max));
}

float ProceduralSkyMaterial::get_sun_angle_max() const {
	return sun_angle_max;
}

void ProceduralSkyMaterial::set_sun_curve(float p_curve) {
	sun_curve = p_curve;
	RS::get_singleton()->material_set_param(_get_material(), "sun_curve", sun_curve);
}

float ProceduralSkyMaterial::get_sun_curve() const {
	return sun_curve;
}

Shader::Mode ProceduralSkyMaterial::get_shader_mode() const {
	return Shader::MODE_SKY;
}

RID ProceduralSkyMaterial::get_rid() const {
	_update_shader();
	if (!shader_set) {
		RS::get_singleton()->material_set_shader(_get_material(), shader);
		shader_set = true;
	}
	return _get_material();
}

RID ProceduralSkyMaterial::get_shader_rid() const {
	_update_shader();
	return shader;
}

void ProceduralSkyMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sky_top_color", "color"), &ProceduralSkyMaterial::set_sky_top_color);
	ClassDB::bind_method(D_METHOD("get_sky_top_color"), &ProceduralSkyMaterial::get_sky_top_color);

	ClassDB::bind_method(D_METHOD("set_sky_horizon_color", "color"), &ProceduralSkyMaterial::set_sky_horizon_color);
	ClassDB::bind_method(D_METHOD("get_sky_horizon_color"), &ProceduralSkyMaterial::get_sky_horizon_color);

	ClassDB::bind_method(D_METHOD("set_sky_curve", "curve"), &ProceduralSkyMaterial::set_sky_curve);
	ClassDB::bind_method(D_METHOD("get_sky_curve"), &ProceduralSkyMaterial::get_sky_curve);

	ClassDB::bind_method(D_METHOD("set_sky_energy", "energy"), &ProceduralSkyMaterial::set_sky_energy);
	ClassDB::bind_method(D_METHOD("get_sky_energy"), &ProceduralSkyMaterial::get_sky_energy);

	ClassDB::bind_method(D_METHOD("set_ground_bottom_color", "color"), &ProceduralSkyMaterial::set_ground_bottom_color);
	ClassDB::bind_method(D_METHOD("get_ground_bottom_color"), &ProceduralSkyMaterial::get_ground_bottom_color);

	ClassDB::bind_method(D_METHOD("set_ground_horizon_color", "color"), &ProceduralSkyMaterial::set_ground_horizon_color);
	ClassDB::bind_method(D_METHOD("get_ground_horizon_color"), &ProceduralSkyMaterial::get_ground_horizon_color);

	ClassDB::bind_method(D_METHOD("set_ground_curve", "curve"), &ProceduralSkyMaterial::set_ground_curve);
	ClassDB::bind_method(D_METHOD("get_ground_curve"), &ProceduralSkyMaterial::get_ground_curve);

	ClassDB::bind_method(D_METHOD("set_ground_energy", "energy"), &ProceduralSkyMaterial::set_ground_energy);
	ClassDB::bind_method(D_METHOD("get_ground_energy"), &ProceduralSkyMaterial::get_ground_energy);

	ClassDB::bind_method(D_METHOD("set_sun_angle_max", "degrees"), &ProceduralSkyMaterial::set_sun_angle_max);
	ClassDB::bind_method(D_METHOD("get_sun_angle_max"), &ProceduralSkyMaterial::get_sun_angle_max);

	ClassDB::bind_method(D_METHOD("set_sun_curve", "curve"), &ProceduralSkyMaterial::set_sun_curve);
	ClassDB::bind_method(D_METHOD("get_sun_curve"), &ProceduralSkyMaterial::get_sun_curve);

	ADD_GROUP("Sky", "sky_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "sky_top_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_sky_top_color", "get_sky_top_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "sky_horizon_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_sky_horizon_color", "get_sky_horizon_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sky_curve", PROPERTY_HINT_EXP_EASING), "set_sky_curve", "get_sky_curve");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sky_energy", PROPERTY_HINT_RANGE, "0,64,0.01"), "set_sky_energy", "get_sky_energy");

	ADD_GROUP("Ground", "ground_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ground_bottom_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ground_bottom_color", "get_ground_bottom_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ground_horizon_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ground_horizon_color", "get_ground_horizon_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ground_curve", PROPERTY_HINT_EXP_EASING), "set_ground_curve", "get_ground_curve");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ground_energy", PROPERTY_HINT_RANGE, "0,64,0.01"), "set_ground_energy", "get_ground_energy");

	ADD_GROUP("Sun", "sun_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sun_angle_max", PROPERTY_HINT_RANGE, "0,360,0.01"), "set_sun_angle_max", "get_sun_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sun_curve", PROPERTY_HINT_EXP_EASING), "set_sun_curve", "get_sun_curve");
}

void ProceduralSkyMaterial::cleanup_shader() {
	if (shader.is_valid()) {
		RS::get_singleton()->free(shader);
	}
}

void ProceduralSkyMaterial::_update_shader() {
	shader_mutex.lock();
	if (shader.is_null()) {
		shader = RS::get_singleton()->shader_create();

		// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
		RS::get_singleton()->shader_set_code(shader, R"(
// NOTE: Shader automatically converted from )" VERSION_NAME " " VERSION_FULL_CONFIG R"('s ProceduralSkyMaterial.

shader_type sky;

uniform vec4 sky_top_color : hint_color = vec4(0.35, 0.46, 0.71, 1.0);
uniform vec4 sky_horizon_color : hint_color = vec4(0.55, 0.69, 0.81, 1.0);
uniform float sky_curve : hint_range(0, 1) = 0.09;
uniform float sky_energy = 1.0;
uniform vec4 ground_bottom_color : hint_color = vec4(0.12, 0.12, 0.13, 1.0);
uniform vec4 ground_horizon_color : hint_color = vec4(0.37, 0.33, 0.31, 1.0);
uniform float ground_curve : hint_range(0, 1) = 0.02;
uniform float ground_energy = 1.0;
uniform float sun_angle_max = 1.74;
uniform float sun_curve : hint_range(0, 1) = 0.05;

void sky() {
	float v_angle = acos(clamp(EYEDIR.y, -1.0, 1.0));
	float c = (1.0 - v_angle / (PI * 0.5));
	vec3 sky = mix(sky_horizon_color.rgb, sky_top_color.rgb, clamp(1.0 - pow(1.0 - c, 1.0 / sky_curve), 0.0, 1.0));
	sky *= sky_energy;

	if (LIGHT0_ENABLED) {
		float sun_angle = acos(dot(LIGHT0_DIRECTION, EYEDIR));
		if (sun_angle < LIGHT0_SIZE) {
			sky = LIGHT0_COLOR * LIGHT0_ENERGY;
		} else if (sun_angle < sun_angle_max) {
			float c2 = (sun_angle - LIGHT0_SIZE) / (sun_angle_max - LIGHT0_SIZE);
			sky = mix(LIGHT0_COLOR * LIGHT0_ENERGY, sky, clamp(1.0 - pow(1.0 - c2, 1.0 / sun_curve), 0.0, 1.0));
		}
	}

	if (LIGHT1_ENABLED) {
		float sun_angle = acos(dot(LIGHT1_DIRECTION, EYEDIR));
		if (sun_angle < LIGHT1_SIZE) {
			sky = LIGHT1_COLOR * LIGHT1_ENERGY;
		} else if (sun_angle < sun_angle_max) {
			float c2 = (sun_angle - LIGHT1_SIZE) / (sun_angle_max - LIGHT1_SIZE);
			sky = mix(LIGHT1_COLOR * LIGHT1_ENERGY, sky, clamp(1.0 - pow(1.0 - c2, 1.0 / sun_curve), 0.0, 1.0));
		}
	}

	if (LIGHT2_ENABLED) {
		float sun_angle = acos(dot(LIGHT2_DIRECTION, EYEDIR));
		if (sun_angle < LIGHT2_SIZE) {
			sky = LIGHT2_COLOR * LIGHT2_ENERGY;
		} else if (sun_angle < sun_angle_max) {
			float c2 = (sun_angle - LIGHT2_SIZE) / (sun_angle_max - LIGHT2_SIZE);
			sky = mix(LIGHT2_COLOR * LIGHT2_ENERGY, sky, clamp(1.0 - pow(1.0 - c2, 1.0 / sun_curve), 0.0, 1.0));
		}
	}

	if (LIGHT3_ENABLED) {
		float sun_angle = acos(dot(LIGHT3_DIRECTION, EYEDIR));
		if (sun_angle < LIGHT3_SIZE) {
			sky = LIGHT3_COLOR * LIGHT3_ENERGY;
		} else if (sun_angle < sun_angle_max) {
			float c2 = (sun_angle - LIGHT3_SIZE) / (sun_angle_max - LIGHT3_SIZE);
			sky = mix(LIGHT3_COLOR * LIGHT3_ENERGY, sky, clamp(1.0 - pow(1.0 - c2, 1.0 / sun_curve), 0.0, 1.0));
		}
	}

	c = (v_angle - (PI * 0.5)) / (PI * 0.5);
	vec3 ground = mix(ground_horizon_color.rgb, ground_bottom_color.rgb, clamp(1.0 - pow(1.0 - c, 1.0 / ground_curve), 0.0, 1.0));
	ground *= ground_energy;

	COLOR = mix(ground, sky, step(0.0, EYEDIR.y));
}
)");
	}
	shader_mutex.unlock();
}

ProceduralSkyMaterial::ProceduralSkyMaterial() {
	set_sky_top_color(Color(0.35, 0.46, 0.71));
	set_sky_horizon_color(Color(0.55, 0.69, 0.81));
	set_sky_curve(0.09);
	set_sky_energy(1.0);

	set_ground_bottom_color(Color(0.12, 0.12, 0.13));
	set_ground_horizon_color(Color(0.37, 0.33, 0.31));
	set_ground_curve(0.02);
	set_ground_energy(1.0);

	set_sun_angle_max(100.0);
	set_sun_curve(0.05);
}

ProceduralSkyMaterial::~ProceduralSkyMaterial() {
	RS::get_singleton()->material_set_shader(_get_material(), RID());
}

/////////////////////////////////////////
/* PanoramaSkyMaterial */

void PanoramaSkyMaterial::set_panorama(const Ref<Texture2D> &p_panorama) {
	panorama = p_panorama;
	RID tex_rid = p_panorama.is_valid() ? p_panorama->get_rid() : RID();
	RS::get_singleton()->material_set_param(_get_material(), "source_panorama", tex_rid);
}

Ref<Texture2D> PanoramaSkyMaterial::get_panorama() const {
	return panorama;
}

Shader::Mode PanoramaSkyMaterial::get_shader_mode() const {
	return Shader::MODE_SKY;
}

RID PanoramaSkyMaterial::get_rid() const {
	_update_shader();
	if (!shader_set) {
		RS::get_singleton()->material_set_shader(_get_material(), shader);
		shader_set = true;
	}
	return _get_material();
}

RID PanoramaSkyMaterial::get_shader_rid() const {
	_update_shader();
	return shader;
}

void PanoramaSkyMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_panorama", "texture"), &PanoramaSkyMaterial::set_panorama);
	ClassDB::bind_method(D_METHOD("get_panorama"), &PanoramaSkyMaterial::get_panorama);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "panorama", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_panorama", "get_panorama");
}

Mutex PanoramaSkyMaterial::shader_mutex;
RID PanoramaSkyMaterial::shader;

void PanoramaSkyMaterial::cleanup_shader() {
	if (shader.is_valid()) {
		RS::get_singleton()->free(shader);
	}
}

void PanoramaSkyMaterial::_update_shader() {
	shader_mutex.lock();
	if (shader.is_null()) {
		shader = RS::get_singleton()->shader_create();

		// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
		RS::get_singleton()->shader_set_code(shader, R"(
// NOTE: Shader automatically converted from )" VERSION_NAME " " VERSION_FULL_CONFIG R"('s PanoramaSkyMaterial.

shader_type sky;

uniform sampler2D source_panorama : filter_linear, hint_albedo;

void sky() {
	COLOR = texture(source_panorama, SKY_COORDS).rgb;
}
)");
	}

	shader_mutex.unlock();
}

PanoramaSkyMaterial::PanoramaSkyMaterial() {
}

PanoramaSkyMaterial::~PanoramaSkyMaterial() {
	RS::get_singleton()->material_set_shader(_get_material(), RID());
}

//////////////////////////////////
/* PhysicalSkyMaterial */

void PhysicalSkyMaterial::set_rayleigh_coefficient(float p_rayleigh) {
	rayleigh = p_rayleigh;
	RS::get_singleton()->material_set_param(_get_material(), "rayleigh", rayleigh);
}

float PhysicalSkyMaterial::get_rayleigh_coefficient() const {
	return rayleigh;
}

void PhysicalSkyMaterial::set_rayleigh_color(Color p_rayleigh_color) {
	rayleigh_color = p_rayleigh_color;
	RS::get_singleton()->material_set_param(_get_material(), "rayleigh_color", rayleigh_color);
}

Color PhysicalSkyMaterial::get_rayleigh_color() const {
	return rayleigh_color;
}

void PhysicalSkyMaterial::set_mie_coefficient(float p_mie) {
	mie = p_mie;
	RS::get_singleton()->material_set_param(_get_material(), "mie", mie);
}

float PhysicalSkyMaterial::get_mie_coefficient() const {
	return mie;
}

void PhysicalSkyMaterial::set_mie_eccentricity(float p_eccentricity) {
	mie_eccentricity = p_eccentricity;
	RS::get_singleton()->material_set_param(_get_material(), "mie_eccentricity", mie_eccentricity);
}

float PhysicalSkyMaterial::get_mie_eccentricity() const {
	return mie_eccentricity;
}

void PhysicalSkyMaterial::set_mie_color(Color p_mie_color) {
	mie_color = p_mie_color;
	RS::get_singleton()->material_set_param(_get_material(), "mie_color", mie_color);
}

Color PhysicalSkyMaterial::get_mie_color() const {
	return mie_color;
}

void PhysicalSkyMaterial::set_turbidity(float p_turbidity) {
	turbidity = p_turbidity;
	RS::get_singleton()->material_set_param(_get_material(), "turbidity", turbidity);
}

float PhysicalSkyMaterial::get_turbidity() const {
	return turbidity;
}

void PhysicalSkyMaterial::set_sun_disk_scale(float p_sun_disk_scale) {
	sun_disk_scale = p_sun_disk_scale;
	RS::get_singleton()->material_set_param(_get_material(), "sun_disk_scale", sun_disk_scale);
}

float PhysicalSkyMaterial::get_sun_disk_scale() const {
	return sun_disk_scale;
}

void PhysicalSkyMaterial::set_ground_color(Color p_ground_color) {
	ground_color = p_ground_color;
	RS::get_singleton()->material_set_param(_get_material(), "ground_color", ground_color);
}

Color PhysicalSkyMaterial::get_ground_color() const {
	return ground_color;
}

void PhysicalSkyMaterial::set_exposure(float p_exposure) {
	exposure = p_exposure;
	RS::get_singleton()->material_set_param(_get_material(), "exposure", exposure);
}

float PhysicalSkyMaterial::get_exposure() const {
	return exposure;
}

void PhysicalSkyMaterial::set_dither_strength(float p_dither_strength) {
	dither_strength = p_dither_strength;
	RS::get_singleton()->material_set_param(_get_material(), "dither_strength", dither_strength);
}

float PhysicalSkyMaterial::get_dither_strength() const {
	return dither_strength;
}

void PhysicalSkyMaterial::set_night_sky(const Ref<Texture2D> &p_night_sky) {
	night_sky = p_night_sky;
	RID tex_rid = p_night_sky.is_valid() ? p_night_sky->get_rid() : RID();
	RS::get_singleton()->material_set_param(_get_material(), "night_sky", tex_rid);
}

Ref<Texture2D> PhysicalSkyMaterial::get_night_sky() const {
	return night_sky;
}

Shader::Mode PhysicalSkyMaterial::get_shader_mode() const {
	return Shader::MODE_SKY;
}

RID PhysicalSkyMaterial::get_rid() const {
	_update_shader();
	if (!shader_set) {
		RS::get_singleton()->material_set_shader(_get_material(), shader);
		shader_set = true;
	}
	return _get_material();
}

RID PhysicalSkyMaterial::get_shader_rid() const {
	_update_shader();
	return shader;
}

Mutex PhysicalSkyMaterial::shader_mutex;
RID PhysicalSkyMaterial::shader;

void PhysicalSkyMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rayleigh_coefficient", "rayleigh"), &PhysicalSkyMaterial::set_rayleigh_coefficient);
	ClassDB::bind_method(D_METHOD("get_rayleigh_coefficient"), &PhysicalSkyMaterial::get_rayleigh_coefficient);

	ClassDB::bind_method(D_METHOD("set_rayleigh_color", "color"), &PhysicalSkyMaterial::set_rayleigh_color);
	ClassDB::bind_method(D_METHOD("get_rayleigh_color"), &PhysicalSkyMaterial::get_rayleigh_color);

	ClassDB::bind_method(D_METHOD("set_mie_coefficient", "mie"), &PhysicalSkyMaterial::set_mie_coefficient);
	ClassDB::bind_method(D_METHOD("get_mie_coefficient"), &PhysicalSkyMaterial::get_mie_coefficient);

	ClassDB::bind_method(D_METHOD("set_mie_eccentricity", "eccentricity"), &PhysicalSkyMaterial::set_mie_eccentricity);
	ClassDB::bind_method(D_METHOD("get_mie_eccentricity"), &PhysicalSkyMaterial::get_mie_eccentricity);

	ClassDB::bind_method(D_METHOD("set_mie_color", "color"), &PhysicalSkyMaterial::set_mie_color);
	ClassDB::bind_method(D_METHOD("get_mie_color"), &PhysicalSkyMaterial::get_mie_color);

	ClassDB::bind_method(D_METHOD("set_turbidity", "turbidity"), &PhysicalSkyMaterial::set_turbidity);
	ClassDB::bind_method(D_METHOD("get_turbidity"), &PhysicalSkyMaterial::get_turbidity);

	ClassDB::bind_method(D_METHOD("set_sun_disk_scale", "scale"), &PhysicalSkyMaterial::set_sun_disk_scale);
	ClassDB::bind_method(D_METHOD("get_sun_disk_scale"), &PhysicalSkyMaterial::get_sun_disk_scale);

	ClassDB::bind_method(D_METHOD("set_ground_color", "color"), &PhysicalSkyMaterial::set_ground_color);
	ClassDB::bind_method(D_METHOD("get_ground_color"), &PhysicalSkyMaterial::get_ground_color);

	ClassDB::bind_method(D_METHOD("set_exposure", "exposure"), &PhysicalSkyMaterial::set_exposure);
	ClassDB::bind_method(D_METHOD("get_exposure"), &PhysicalSkyMaterial::get_exposure);

	ClassDB::bind_method(D_METHOD("set_dither_strength", "strength"), &PhysicalSkyMaterial::set_dither_strength);
	ClassDB::bind_method(D_METHOD("get_dither_strength"), &PhysicalSkyMaterial::get_dither_strength);

	ClassDB::bind_method(D_METHOD("set_night_sky", "night_sky"), &PhysicalSkyMaterial::set_night_sky);
	ClassDB::bind_method(D_METHOD("get_night_sky"), &PhysicalSkyMaterial::get_night_sky);

	ADD_GROUP("Rayleigh", "rayleigh_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rayleigh_coefficient", PROPERTY_HINT_RANGE, "0,64,0.01"), "set_rayleigh_coefficient", "get_rayleigh_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "rayleigh_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_rayleigh_color", "get_rayleigh_color");

	ADD_GROUP("Mie", "mie_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mie_coefficient", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_mie_coefficient", "get_mie_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mie_eccentricity", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_mie_eccentricity", "get_mie_eccentricity");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "mie_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_mie_color", "get_mie_color");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "turbidity", PROPERTY_HINT_RANGE, "0,1000,0.01"), "set_turbidity", "get_turbidity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sun_disk_scale", PROPERTY_HINT_RANGE, "0,360,0.01"), "set_sun_disk_scale", "get_sun_disk_scale");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ground_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ground_color", "get_ground_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "exposure", PROPERTY_HINT_RANGE, "0,128,0.01"), "set_exposure", "get_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dither_strength", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_dither_strength", "get_dither_strength");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "night_sky", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_night_sky", "get_night_sky");
}

void PhysicalSkyMaterial::cleanup_shader() {
	if (shader.is_valid()) {
		RS::get_singleton()->free(shader);
	}
}

void PhysicalSkyMaterial::_update_shader() {
	shader_mutex.lock();
	if (shader.is_null()) {
		shader = RS::get_singleton()->shader_create();

		// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
		RS::get_singleton()->shader_set_code(shader, R"(
// NOTE: Shader automatically converted from )" VERSION_NAME " " VERSION_FULL_CONFIG R"('s PhysicalSkyMaterial.

shader_type sky;

uniform float rayleigh : hint_range(0, 64) = 2.0;
uniform vec4 rayleigh_color : hint_color = vec4(0.26, 0.41, 0.58, 1.0);
uniform float mie : hint_range(0, 1) = 0.005;
uniform float mie_eccentricity : hint_range(-1, 1) = 0.8;
uniform vec4 mie_color : hint_color = vec4(0.63, 0.77, 0.92, 1.0);

uniform float turbidity : hint_range(0, 1000) = 10.0;
uniform float sun_disk_scale : hint_range(0, 360) = 1.0;
uniform vec4 ground_color : hint_color = vec4(1.0);
uniform float exposure : hint_range(0, 128) = 0.1;
uniform float dither_strength : hint_range(0, 10) = 1.0;

uniform sampler2D night_sky : hint_black_albedo;

const vec3 UP = vec3( 0.0, 1.0, 0.0 );

// Sun constants
const float SUN_ENERGY = 1000.0;

// Optical length at zenith for molecules.
const float rayleigh_zenith_size = 8.4e3;
const float mie_zenith_size = 1.25e3;

float henyey_greenstein(float cos_theta, float g) {
	const float k = 0.0795774715459;
	return k * (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

// From: https://www.shadertoy.com/view/4sfGzS credit to iq
float hash(vec3 p) {
	p  = fract( p * 0.3183099 + 0.1 );
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

void sky() {
	if (LIGHT0_ENABLED) {
		float zenith_angle = clamp( dot(UP, normalize(LIGHT0_DIRECTION)), -1.0, 1.0 );
		float sun_energy = max(0.0, 1.0 - exp(-((PI * 0.5) - acos(zenith_angle)))) * SUN_ENERGY * LIGHT0_ENERGY;
		float sun_fade = 1.0 - clamp(1.0 - exp(LIGHT0_DIRECTION.y), 0.0, 1.0);

		// Rayleigh coefficients.
		float rayleigh_coefficient = rayleigh - ( 1.0 * ( 1.0 - sun_fade ) );
		vec3 rayleigh_beta = rayleigh_coefficient * rayleigh_color.rgb * 0.0001;
		// mie coefficients from Preetham
		vec3 mie_beta = turbidity * mie * mie_color.rgb * 0.000434;

		// Optical length.
		float zenith = acos(max(0.0, dot(UP, EYEDIR)));
		float optical_mass = 1.0 / (cos(zenith) + 0.15 * pow(93.885 - degrees(zenith), -1.253));
		float rayleigh_scatter = rayleigh_zenith_size * optical_mass;
		float mie_scatter = mie_zenith_size * optical_mass;

		// Light extinction based on thickness of atmosphere.
		vec3 extinction = exp(-(rayleigh_beta * rayleigh_scatter + mie_beta * mie_scatter));

		// In scattering.
		float cos_theta = dot(EYEDIR, normalize(LIGHT0_DIRECTION));

		float rayleigh_phase = (3.0 / (16.0 * PI)) * (1.0 + pow(cos_theta * 0.5 + 0.5, 2.0));
		vec3 betaRTheta = rayleigh_beta * rayleigh_phase;

		float mie_phase = henyey_greenstein(cos_theta, mie_eccentricity);
		vec3 betaMTheta = mie_beta * mie_phase;

		vec3 Lin = pow(sun_energy * ((betaRTheta + betaMTheta) / (rayleigh_beta + mie_beta)) * (1.0 - extinction), vec3(1.5));
		// Hack from https://github.com/mrdoob/three.js/blob/master/examples/jsm/objects/Sky.js
		Lin *= mix(vec3(1.0), pow(sun_energy * ((betaRTheta + betaMTheta) / (rayleigh_beta + mie_beta)) * extinction, vec3(0.5)), clamp(pow(1.0 - zenith_angle, 5.0), 0.0, 1.0));

		// Hack in the ground color.
		Lin  *= mix(ground_color.rgb, vec3(1.0), smoothstep(-0.1, 0.1, dot(UP, EYEDIR)));

		// Solar disk and out-scattering.
		float sunAngularDiameterCos = cos(LIGHT0_SIZE * sun_disk_scale);
		float sunAngularDiameterCos2 = cos(LIGHT0_SIZE * sun_disk_scale*0.5);
		float sundisk = smoothstep(sunAngularDiameterCos, sunAngularDiameterCos2, cos_theta);
		vec3 L0 = (sun_energy * 1900.0 * extinction) * sundisk * LIGHT0_COLOR;
		L0 += texture(night_sky, SKY_COORDS).xyz * extinction;

		vec3 color = (Lin + L0) * 0.04;
		COLOR = pow(color, vec3(1.0 / (1.2 + (1.2 * sun_fade))));
		COLOR *= exposure;
		// Make optional, eliminates banding.
		COLOR += (hash(EYEDIR * 1741.9782) * 0.08 - 0.04) * 0.016 * dither_strength;
	} else {
		// There is no sun, so display night_sky and nothing else.
		COLOR = texture(night_sky, SKY_COORDS).xyz * 0.04;
		COLOR *= exposure;
	}
}
)");
	}

	shader_mutex.unlock();
}

PhysicalSkyMaterial::PhysicalSkyMaterial() {
	set_rayleigh_coefficient(2.0);
	set_rayleigh_color(Color(0.26, 0.41, 0.58));
	set_mie_coefficient(0.005);
	set_mie_eccentricity(0.8);
	set_mie_color(Color(0.63, 0.77, 0.92));
	set_turbidity(10.0);
	set_sun_disk_scale(1.0);
	set_ground_color(Color(1.0, 1.0, 1.0));
	set_exposure(0.1);
	set_dither_strength(1.0);
}

PhysicalSkyMaterial::~PhysicalSkyMaterial() {
}
