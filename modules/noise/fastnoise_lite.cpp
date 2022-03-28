/*************************************************************************/
/*  fastnoise_lite.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "fastnoise_lite.h"

FastNoiseLite::FastNoiseLite() {
	// Most defaults copied from the library.
	set_noise_type(TYPE_SIMPLEX_SMOOTH);
	set_seed(0);
	set_frequency(0.01);
	set_in_3d_space(false);

	set_fractal_type(FRACTAL_FBM);
	set_fractal_octaves(5);
	set_fractal_lacunarity(2.0);
	set_fractal_gain(0.5);
	set_fractal_weighted_strength(0.0);
	set_fractal_ping_pong_strength(2.0);

	set_cellular_distance_function(DISTANCE_EUCLIDEAN);
	set_cellular_return_type(RETURN_CELL_VALUE);
	set_cellular_jitter(0.45);

	set_domain_warp_enabled(false);
	set_domain_warp_type(DOMAIN_WARP_SIMPLEX);
	set_domain_warp_amplitude(30.0);
	set_domain_warp_frequency(0.05);
	set_domain_warp_fractal_type(DOMAIN_WARP_FRACTAL_PROGRESSIVE);
	set_domain_warp_fractal_octaves(5);
	set_domain_warp_fractal_lacunarity(6);
	set_domain_warp_fractal_gain(0.5);
}

FastNoiseLite::~FastNoiseLite() {
}

// General settings.

void FastNoiseLite::set_noise_type(NoiseType p_noise_type) {
	noise_type = p_noise_type;
	_noise.SetNoiseType((_FastNoiseLite::NoiseType)p_noise_type);
	emit_changed();
	notify_property_list_changed();
}

FastNoiseLite::NoiseType FastNoiseLite::get_noise_type() const {
	return noise_type;
}

void FastNoiseLite::set_seed(int p_seed) {
	seed = p_seed;
	_noise.SetSeed(p_seed);
	emit_changed();
}

int FastNoiseLite::get_seed() const {
	return seed;
}

void FastNoiseLite::set_frequency(real_t p_freq) {
	frequency = p_freq;
	_noise.SetFrequency(p_freq);
	emit_changed();
}

real_t FastNoiseLite::get_frequency() const {
	return frequency;
}

void FastNoiseLite::set_in_3d_space(bool p_enable) {
	in_3d_space = p_enable;
	emit_changed();
}
bool FastNoiseLite::is_in_3d_space() const {
	return in_3d_space;
}

void FastNoiseLite::set_offset(Vector3 p_offset) {
	offset = p_offset;
	emit_changed();
}

Vector3 FastNoiseLite::get_offset() const {
	return offset;
}

void FastNoiseLite::set_color_ramp(const Ref<Gradient> &p_gradient) {
	color_ramp = p_gradient;
	if (color_ramp.is_valid()) {
		color_ramp->connect(SNAME("changed"), callable_mp(this, &FastNoiseLite::_changed));
		emit_changed();
	}
}

Ref<Gradient> FastNoiseLite::get_color_ramp() const {
	return color_ramp;
}

// Noise functions.

real_t FastNoiseLite::get_noise_1d(real_t p_x) {
	return get_noise_2d(p_x, 0.0);
}

real_t FastNoiseLite::get_noise_2dv(Vector2 p_v) {
	return get_noise_2d(p_v.x, p_v.y);
}

real_t FastNoiseLite::get_noise_2d(real_t p_x, real_t p_y) {
	if (domain_warp_enabled) {
		_domain_warp_noise.DomainWarp(p_x, p_y);
	}
	return _noise.GetNoise(p_x + offset.x, p_y + offset.y);
}

real_t FastNoiseLite::get_noise_3dv(Vector3 p_v) {
	return get_noise_3d(p_v.x, p_v.y, p_v.z);
}

real_t FastNoiseLite::get_noise_3d(real_t p_x, real_t p_y, real_t p_z) {
	if (domain_warp_enabled) {
		_domain_warp_noise.DomainWarp(p_x, p_y, p_z);
	}
	return _noise.GetNoise(p_x + offset.x, p_y + offset.y, p_z + offset.z);
}

// Fractal.

void FastNoiseLite::set_fractal_type(FractalType p_type) {
	fractal_type = p_type;
	_noise.SetFractalType((_FastNoiseLite::FractalType)p_type);
	emit_changed();
	notify_property_list_changed();
}

FastNoiseLite::FractalType FastNoiseLite::get_fractal_type() const {
	return fractal_type;
}

void FastNoiseLite::set_fractal_octaves(int p_octaves) {
	fractal_octaves = p_octaves;
	_noise.SetFractalOctaves(p_octaves);
	emit_changed();
}

int FastNoiseLite::get_fractal_octaves() const {
	return fractal_octaves;
}

void FastNoiseLite::set_fractal_lacunarity(real_t p_lacunarity) {
	fractal_lacunarity = p_lacunarity;
	_noise.SetFractalLacunarity(p_lacunarity);
	emit_changed();
}

real_t FastNoiseLite::get_fractal_lacunarity() const {
	return fractal_lacunarity;
}

void FastNoiseLite::set_fractal_gain(real_t p_gain) {
	fractal_gain = p_gain;
	_noise.SetFractalGain(p_gain);
	emit_changed();
}

real_t FastNoiseLite::get_fractal_gain() const {
	return fractal_gain;
}

void FastNoiseLite::set_fractal_weighted_strength(real_t p_weighted_strength) {
	fractal_weighted_strength = p_weighted_strength;
	_noise.SetFractalWeightedStrength(p_weighted_strength);
	emit_changed();
}
real_t FastNoiseLite::get_fractal_weighted_strength() const {
	return fractal_weighted_strength;
}

void FastNoiseLite::set_fractal_ping_pong_strength(real_t p_ping_pong_strength) {
	fractal_pinp_pong_strength = p_ping_pong_strength;
	_noise.SetFractalPingPongStrength(p_ping_pong_strength);
	emit_changed();
}
real_t FastNoiseLite::get_fractal_ping_pong_strength() const {
	return fractal_pinp_pong_strength;
}

// Cellular.

void FastNoiseLite::set_cellular_distance_function(CellularDistanceFunction p_func) {
	cellular_distance_function = p_func;
	_noise.SetCellularDistanceFunction((_FastNoiseLite::CellularDistanceFunction)p_func);
	emit_changed();
}

FastNoiseLite::CellularDistanceFunction FastNoiseLite::get_cellular_distance_function() const {
	return cellular_distance_function;
}

void FastNoiseLite::set_cellular_jitter(real_t p_jitter) {
	cellular_jitter = p_jitter;
	_noise.SetCellularJitter(p_jitter);
	emit_changed();
}

real_t FastNoiseLite::get_cellular_jitter() const {
	return cellular_jitter;
}

void FastNoiseLite::set_cellular_return_type(CellularReturnType p_ret) {
	cellular_return_type = p_ret;
	_noise.SetCellularReturnType((_FastNoiseLite::CellularReturnType)p_ret);

	emit_changed();
}

FastNoiseLite::CellularReturnType FastNoiseLite::get_cellular_return_type() const {
	return cellular_return_type;
}

// Domain warp specific.

void FastNoiseLite::set_domain_warp_enabled(bool p_enabled) {
	if (domain_warp_enabled != p_enabled) {
		domain_warp_enabled = p_enabled;
		emit_changed();
		notify_property_list_changed();
	}
}

bool FastNoiseLite::is_domain_warp_enabled() const {
	return domain_warp_enabled;
}

void FastNoiseLite::set_domain_warp_type(DomainWarpType p_domain_warp_type) {
	domain_warp_type = p_domain_warp_type;
	_domain_warp_noise.SetDomainWarpType((_FastNoiseLite::DomainWarpType)p_domain_warp_type);
	emit_changed();
}

FastNoiseLite::DomainWarpType FastNoiseLite::get_domain_warp_type() const {
	return domain_warp_type;
}

void FastNoiseLite::set_domain_warp_amplitude(real_t p_amplitude) {
	domain_warp_amplitude = p_amplitude;
	_domain_warp_noise.SetDomainWarpAmp(p_amplitude);
	emit_changed();
}
real_t FastNoiseLite::get_domain_warp_amplitude() const {
	return domain_warp_amplitude;
}

void FastNoiseLite::set_domain_warp_frequency(real_t p_frequency) {
	domain_warp_frequency = p_frequency;
	_domain_warp_noise.SetFrequency(p_frequency);
	emit_changed();
}

real_t FastNoiseLite::get_domain_warp_frequency() const {
	return domain_warp_frequency;
}

void FastNoiseLite::set_domain_warp_fractal_type(DomainWarpFractalType p_domain_warp_fractal_type) {
	domain_warp_fractal_type = p_domain_warp_fractal_type;

	// This needs manual conversion because Godots Inspector property API does not support discontiguous enum indices.
	_FastNoiseLite::FractalType type;
	switch (p_domain_warp_fractal_type) {
		case DOMAIN_WARP_FRACTAL_NONE:
			type = _FastNoiseLite::FractalType_None;
			break;
		case DOMAIN_WARP_FRACTAL_PROGRESSIVE:
			type = _FastNoiseLite::FractalType_DomainWarpProgressive;
			break;
		case DOMAIN_WARP_FRACTAL_INDEPENDENT:
			type = _FastNoiseLite::FractalType_DomainWarpIndependent;
			break;
		default:
			type = _FastNoiseLite::FractalType_None;
	}

	_domain_warp_noise.SetFractalType(type);
	emit_changed();
}

FastNoiseLite::DomainWarpFractalType FastNoiseLite::get_domain_warp_fractal_type() const {
	return domain_warp_fractal_type;
}

void FastNoiseLite::set_domain_warp_fractal_octaves(int p_octaves) {
	domain_warp_fractal_octaves = p_octaves;
	_domain_warp_noise.SetFractalOctaves(p_octaves);
	emit_changed();
}

int FastNoiseLite::get_domain_warp_fractal_octaves() const {
	return domain_warp_fractal_octaves;
}

void FastNoiseLite::set_domain_warp_fractal_lacunarity(real_t p_lacunarity) {
	domain_warp_fractal_lacunarity = p_lacunarity;
	_domain_warp_noise.SetFractalLacunarity(p_lacunarity);
	emit_changed();
}

real_t FastNoiseLite::get_domain_warp_fractal_lacunarity() const {
	return domain_warp_fractal_lacunarity;
}

void FastNoiseLite::set_domain_warp_fractal_gain(real_t p_gain) {
	domain_warp_fractal_gain = p_gain;
	_domain_warp_noise.SetFractalGain(p_gain);
	emit_changed();
}

real_t FastNoiseLite::get_domain_warp_fractal_gain() const {
	return domain_warp_fractal_gain;
}

// Textures.

Ref<Image> FastNoiseLite::get_image(int p_width, int p_height, bool p_invert) {
	bool grayscale = color_ramp.is_null();

	Vector<uint8_t> data;
	data.resize(p_width * p_height * (grayscale ? 1 : 4));

	uint8_t *wd8 = data.ptrw();

	// Get all values and identify min/max values.
	Vector<real_t> values;
	values.resize(p_width * p_height);
	real_t min_val = 100;
	real_t max_val = -100;

	for (int y = 0, i = 0; y < p_height; y++) {
		for (int x = 0; x < p_width; x++, i++) {
			values.set(i, is_in_3d_space() ? get_noise_3d(x, y, 0.0) : get_noise_2d(x, y));
			if (values[i] > max_val) {
				max_val = values[i];
			}
			if (values[i] < min_val) {
				min_val = values[i];
			}
		}
	}

	// Normalize values and write to texture.
	uint8_t value;
	for (int i = 0, x = 0; i < p_height; i++) {
		for (int j = 0; j < p_width; j++, x++) {
			if (max_val == min_val) {
				value = 0;
			} else {
				value = uint8_t(CLAMP((values[x] - min_val) / (max_val - min_val) * 255.f, 0, 255));
			}
			if (p_invert) {
				value = 255 - value;
			}
			if (grayscale) {
				wd8[x] = value;
			} else {
				float luminance = value / 255.0;
				Color ramp_color = color_ramp->get_color_at_offset(luminance);
				wd8[x * 4 + 0] = uint8_t(CLAMP(ramp_color.r * 255, 0, 255));
				wd8[x * 4 + 1] = uint8_t(CLAMP(ramp_color.g * 255, 0, 255));
				wd8[x * 4 + 2] = uint8_t(CLAMP(ramp_color.b * 255, 0, 255));
				wd8[x * 4 + 3] = uint8_t(CLAMP(ramp_color.a * 255, 0, 255));
			}
		}
	}
	if (grayscale) {
		return memnew(Image(p_width, p_height, false, Image::FORMAT_L8, data));
	} else {
		return memnew(Image(p_width, p_height, false, Image::FORMAT_RGBA8, data));
	}
}

Ref<Image> FastNoiseLite::get_seamless_image(int p_width, int p_height, bool p_invert, real_t p_blend_skirt) {
	// Just return parent function. This is here only so Godot will properly document this function.
	return Noise::get_seamless_image(p_width, p_height, p_invert, p_blend_skirt);
}

void FastNoiseLite::_changed() {
	emit_changed();
}

void FastNoiseLite::_bind_methods() {
	// General settings.

	ClassDB::bind_method(D_METHOD("set_noise_type", "type"), &FastNoiseLite::set_noise_type);
	ClassDB::bind_method(D_METHOD("get_noise_type"), &FastNoiseLite::get_noise_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "noise_type", PROPERTY_HINT_ENUM, "Simplex,Simplex Smooth,Cellular,Perlin,Value Cubic,Value"), "set_noise_type", "get_noise_type");

	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &FastNoiseLite::set_seed);
	ClassDB::bind_method(D_METHOD("get_seed"), &FastNoiseLite::get_seed);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

	ClassDB::bind_method(D_METHOD("set_frequency", "freq"), &FastNoiseLite::set_frequency);
	ClassDB::bind_method(D_METHOD("get_frequency"), &FastNoiseLite::get_frequency);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "frequency", PROPERTY_HINT_RANGE, ".001,1"), "set_frequency", "get_frequency");

	ClassDB::bind_method(D_METHOD("set_in_3d_space", "enable"), &FastNoiseLite::set_in_3d_space);
	ClassDB::bind_method(D_METHOD("is_in_3d_space"), &FastNoiseLite::is_in_3d_space);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "in_3d_space"), "set_in_3d_space", "is_in_3d_space");

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &FastNoiseLite::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &FastNoiseLite::get_offset);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "offset", PROPERTY_HINT_RANGE, "-999999999,999999999,1"), "set_offset", "get_offset");

	ClassDB::bind_method(D_METHOD("set_color_ramp", "gradient"), &FastNoiseLite::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &FastNoiseLite::get_color_ramp);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_color_ramp", "get_color_ramp");

	// Fractal.

	ADD_GROUP("Fractal", "fractal_");
	ClassDB::bind_method(D_METHOD("set_fractal_type", "type"), &FastNoiseLite::set_fractal_type);
	ClassDB::bind_method(D_METHOD("get_fractal_type"), &FastNoiseLite::get_fractal_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fractal_type", PROPERTY_HINT_ENUM, "None,FBM,Ridged,PingPong"), "set_fractal_type", "get_fractal_type");

	ClassDB::bind_method(D_METHOD("set_fractal_octaves", "octave_count"), &FastNoiseLite::set_fractal_octaves);
	ClassDB::bind_method(D_METHOD("get_fractal_octaves"), &FastNoiseLite::get_fractal_octaves);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fractal_octaves", PROPERTY_HINT_RANGE, "1,10,1"), "set_fractal_octaves", "get_fractal_octaves");

	ClassDB::bind_method(D_METHOD("set_fractal_lacunarity", "lacunarity"), &FastNoiseLite::set_fractal_lacunarity);
	ClassDB::bind_method(D_METHOD("get_fractal_lacunarity"), &FastNoiseLite::get_fractal_lacunarity);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fractal_lacunarity"), "set_fractal_lacunarity", "get_fractal_lacunarity");

	ClassDB::bind_method(D_METHOD("set_fractal_gain", "gain"), &FastNoiseLite::set_fractal_gain);
	ClassDB::bind_method(D_METHOD("get_fractal_gain"), &FastNoiseLite::get_fractal_gain);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fractal_gain"), "set_fractal_gain", "get_fractal_gain");

	ClassDB::bind_method(D_METHOD("set_fractal_weighted_strength", "weighted_strength"), &FastNoiseLite::set_fractal_weighted_strength);
	ClassDB::bind_method(D_METHOD("get_fractal_weighted_strength"), &FastNoiseLite::get_fractal_weighted_strength);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fractal_weighted_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_fractal_weighted_strength", "get_fractal_weighted_strength");

	ClassDB::bind_method(D_METHOD("set_fractal_ping_pong_strength", "ping_pong_strength"), &FastNoiseLite::set_fractal_ping_pong_strength);
	ClassDB::bind_method(D_METHOD("get_fractal_ping_pong_strength"), &FastNoiseLite::get_fractal_ping_pong_strength);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fractal_ping_pong_strength"), "set_fractal_ping_pong_strength", "get_fractal_ping_pong_strength");

	// Cellular.

	ADD_GROUP("Cellular", "cellular_");
	ClassDB::bind_method(D_METHOD("set_cellular_distance_function", "func"), &FastNoiseLite::set_cellular_distance_function);
	ClassDB::bind_method(D_METHOD("get_cellular_distance_function"), &FastNoiseLite::get_cellular_distance_function);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cellular_distance_function", PROPERTY_HINT_ENUM, "Euclidean,EuclideanSquared,Manhattan,Hybrid"), "set_cellular_distance_function", "get_cellular_distance_function");

	ClassDB::bind_method(D_METHOD("set_cellular_jitter", "jitter"), &FastNoiseLite::set_cellular_jitter);
	ClassDB::bind_method(D_METHOD("get_cellular_jitter"), &FastNoiseLite::get_cellular_jitter);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cellular_jitter"), "set_cellular_jitter", "get_cellular_jitter");

	ClassDB::bind_method(D_METHOD("set_cellular_return_type", "ret"), &FastNoiseLite::set_cellular_return_type);
	ClassDB::bind_method(D_METHOD("get_cellular_return_type"), &FastNoiseLite::get_cellular_return_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cellular_return_type", PROPERTY_HINT_ENUM, "CellValue,Distance,Distance2,Distance2Add,Distance2Sub,Distance2Mul,Distance2Div"), "set_cellular_return_type", "get_cellular_return_type");

	// Domain warp.

	ADD_GROUP("Domain warp", "domain_warp_");

	ClassDB::bind_method(D_METHOD("set_domain_warp_enabled", "domain_warp_enabled"), &FastNoiseLite::set_domain_warp_enabled);
	ClassDB::bind_method(D_METHOD("is_domain_warp_enabled"), &FastNoiseLite::is_domain_warp_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "domain_warp_enabled"), "set_domain_warp_enabled", "is_domain_warp_enabled");

	ClassDB::bind_method(D_METHOD("set_domain_warp_type", "domain_warp_type"), &FastNoiseLite::set_domain_warp_type);
	ClassDB::bind_method(D_METHOD("get_domain_warp_type"), &FastNoiseLite::get_domain_warp_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "domain_warp_type", PROPERTY_HINT_ENUM, "Simplex,SimplexReduced,BasicGrid"), "set_domain_warp_type", "get_domain_warp_type");

	ClassDB::bind_method(D_METHOD("set_domain_warp_amplitude", "domain_warp_amplitude"), &FastNoiseLite::set_domain_warp_amplitude);
	ClassDB::bind_method(D_METHOD("get_domain_warp_amplitude"), &FastNoiseLite::get_domain_warp_amplitude);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "domain_warp_amplitude"), "set_domain_warp_amplitude", "get_domain_warp_amplitude");

	ClassDB::bind_method(D_METHOD("set_domain_warp_frequency", "domain_warp_frequency"), &FastNoiseLite::set_domain_warp_frequency);
	ClassDB::bind_method(D_METHOD("get_domain_warp_frequency"), &FastNoiseLite::get_domain_warp_frequency);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "domain_warp_frequency"), "set_domain_warp_frequency", "get_domain_warp_frequency");

	ClassDB::bind_method(D_METHOD("set_domain_warp_fractal_type", "domain_warp_fractal_type"), &FastNoiseLite::set_domain_warp_fractal_type);
	ClassDB::bind_method(D_METHOD("get_domain_warp_fractal_type"), &FastNoiseLite::get_domain_warp_fractal_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "domain_warp_fractal_type", PROPERTY_HINT_ENUM, "None,Progressive,Independent"), "set_domain_warp_fractal_type", "get_domain_warp_fractal_type");

	ClassDB::bind_method(D_METHOD("set_domain_warp_fractal_octaves", "domain_warp_octave_count"), &FastNoiseLite::set_domain_warp_fractal_octaves);
	ClassDB::bind_method(D_METHOD("get_domain_warp_fractal_octaves"), &FastNoiseLite::get_domain_warp_fractal_octaves);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "domain_warp_fractal_octaves", PROPERTY_HINT_RANGE, "1,10,1"), "set_domain_warp_fractal_octaves", "get_domain_warp_fractal_octaves");

	ClassDB::bind_method(D_METHOD("set_domain_warp_fractal_lacunarity", "domain_warp_lacunarity"), &FastNoiseLite::set_domain_warp_fractal_lacunarity);
	ClassDB::bind_method(D_METHOD("get_domain_warp_fractal_lacunarity"), &FastNoiseLite::get_domain_warp_fractal_lacunarity);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "domain_warp_fractal_lacunarity"), "set_domain_warp_fractal_lacunarity", "get_domain_warp_fractal_lacunarity");

	ClassDB::bind_method(D_METHOD("set_domain_warp_fractal_gain", "domain_warp_gain"), &FastNoiseLite::set_domain_warp_fractal_gain);
	ClassDB::bind_method(D_METHOD("get_domain_warp_fractal_gain"), &FastNoiseLite::get_domain_warp_fractal_gain);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "domain_warp_fractal_gain"), "set_domain_warp_fractal_gain", "get_domain_warp_fractal_gain");

	ClassDB::bind_method(D_METHOD("_changed"), &FastNoiseLite::_changed);

	BIND_ENUM_CONSTANT(TYPE_VALUE);
	BIND_ENUM_CONSTANT(TYPE_VALUE_CUBIC);
	BIND_ENUM_CONSTANT(TYPE_PERLIN);
	BIND_ENUM_CONSTANT(TYPE_CELLULAR);
	BIND_ENUM_CONSTANT(TYPE_SIMPLEX);
	BIND_ENUM_CONSTANT(TYPE_SIMPLEX_SMOOTH);

	BIND_ENUM_CONSTANT(FRACTAL_NONE);
	BIND_ENUM_CONSTANT(FRACTAL_FBM);
	BIND_ENUM_CONSTANT(FRACTAL_RIDGED);
	BIND_ENUM_CONSTANT(FRACTAL_PING_PONG);

	BIND_ENUM_CONSTANT(DISTANCE_EUCLIDEAN);
	BIND_ENUM_CONSTANT(DISTANCE_EUCLIDEAN_SQUARED);
	BIND_ENUM_CONSTANT(DISTANCE_MANHATTAN);
	BIND_ENUM_CONSTANT(DISTANCE_HYBRID);

	BIND_ENUM_CONSTANT(RETURN_CELL_VALUE);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_ADD);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_SUB);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_MUL);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_DIV);

	BIND_ENUM_CONSTANT(DOMAIN_WARP_SIMPLEX);
	BIND_ENUM_CONSTANT(DOMAIN_WARP_SIMPLEX_REDUCED);
	BIND_ENUM_CONSTANT(DOMAIN_WARP_BASIC_GRID);

	BIND_ENUM_CONSTANT(DOMAIN_WARP_FRACTAL_NONE);
	BIND_ENUM_CONSTANT(DOMAIN_WARP_FRACTAL_PROGRESSIVE);
	BIND_ENUM_CONSTANT(DOMAIN_WARP_FRACTAL_INDEPENDENT);
}

void FastNoiseLite::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("cellular") && get_noise_type() != TYPE_CELLULAR) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
		return;
	}

	if (property.name != "fractal_type" && property.name.begins_with("fractal") && get_fractal_type() == FRACTAL_NONE) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
		return;
	}

	if (property.name == "fractal_ping_pong_strength" && get_fractal_type() != FRACTAL_PING_PONG) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
		return;
	}

	if (property.name != "domain_warp_enabled" && property.name.begins_with("domain_warp") && !domain_warp_enabled) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
		return;
	}
}
