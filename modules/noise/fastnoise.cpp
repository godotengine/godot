/*************************************************************************/
/*  fastnoise.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "fastnoise.h"

FastNoise::FastNoise() {
	// Most defaults copied from the library
	set_noise_type(TYPE_VALUE);
	set_seed(0);
	set_frequency(0.01);
	set_interpolation(INTERP_QUINTIC);

	set_fractal_type(FRACTAL_FBM);
	set_fractal_octaves(3);
	set_fractal_lacunarity(2.0);
	set_fractal_gain(0.5);

	set_cellular_distance_function(DISTANCE_EUCLIDEAN);
	set_cellular_return_type(RETURN_CELL_VALUE);
	set_cellular_distance2_indices(0, 1);
	set_cellular_jitter(0.45);

	set_perturb_amplitude(30.0);
	set_perturb_frequency(0.05);
	set_perturb_type(PERTURB_NONE);
}

FastNoise::~FastNoise() {
}

// General settings

void FastNoise::set_noise_type(NoiseType p_noise_type) {
	_noise.SetNoiseType((_FastNoise::NoiseType)p_noise_type);
	_change_notify();
	emit_changed();
}

FastNoise::NoiseType FastNoise::get_noise_type() const {
	return (NoiseType)_noise.GetNoiseType();
}

void FastNoise::set_seed(int p_seed) {
	_noise.SetSeed(p_seed);
	emit_changed();
}

int FastNoise::get_seed() const {
	return _noise.GetSeed();
}

void FastNoise::set_frequency(real_t p_freq) {
	_noise.SetFrequency(p_freq);
	emit_changed();
}

real_t FastNoise::get_frequency() const {
	return _noise.GetFrequency();
}

void FastNoise::set_interpolation(Interpolation p_interp) {
	_noise.SetInterp((_FastNoise::Interp)p_interp);
	emit_changed();
}

FastNoise::Interpolation FastNoise::get_interpolation() const {
	return (Interpolation)_noise.GetInterp();
}

void FastNoise::set_offset(Vector3 p_offset) {
	_offset = p_offset;
	emit_changed();
}

Vector3 FastNoise::get_offset() const {
	return _offset;
}

// Noise functions

real_t FastNoise::get_noise_1d(real_t p_x) {
	return get_noise_2d(p_x, 0.0);
}

real_t FastNoise::get_noise_2dv(Vector2 p_v) {
	return get_noise_2d(p_v.x, p_v.y);
}

real_t FastNoise::get_noise_2d(real_t p_x, real_t p_y) {
	real_t nx = p_x + _offset.x;
	real_t ny = p_y + _offset.y;

	// Twist (perturb) coordinates before noise function
	if (_perturb == PERTURB_GRADIENT) {
		_noise.GradientPerturb(nx, ny);
	} else if (_perturb == PERTURB_GRADIENT_FRACTAL) {
		_noise.GradientPerturbFractal(nx, ny);
	}

	return _noise.GetNoise(nx, ny);
}

real_t FastNoise::get_noise_3dv(Vector3 p_v) {
	return get_noise_3d(p_v.x, p_v.y, p_v.z);
}

real_t FastNoise::get_noise_3d(real_t p_x, real_t p_y, real_t p_z) {
	real_t nx = p_x + _offset.x;
	real_t ny = p_y + _offset.y;
	real_t nz = p_z + _offset.z;

	// Twist (perturb) coordinates before noise function
	if (_perturb == PERTURB_GRADIENT) {
		_noise.GradientPerturb(nx, ny, nz);
	} else if (_perturb == PERTURB_GRADIENT_FRACTAL) {
		_noise.GradientPerturbFractal(nx, ny, nz);
	}

	return _noise.GetNoise(nx, ny, nz);
}

real_t FastNoise::get_white_noise_4d(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
	return _noise.GetWhiteNoise(p_x, p_y, p_z, p_w);
}

#ifdef SIMPLEX_ENABLED
real_t FastNoise::get_simplex_4d(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
	return _noise.GetSimplex(p_x, p_y, p_z, p_w);
}
#endif

// Perturb

void FastNoise::set_perturb_type(PerturbType p_type) {
	_perturb = p_type;
	_change_notify();
	emit_changed();
}

FastNoise::PerturbType FastNoise::get_perturb_type() const {
	return _perturb;
}

void FastNoise::set_perturb_amplitude(real_t p_amp) {
	_noise.SetGradientPerturbAmp(p_amp);
	emit_changed();
}

real_t FastNoise::get_perturb_amplitude() const {
	return _noise.GetGradientPerturbAmp();
}

void FastNoise::set_perturb_frequency(real_t p_freq) {
	_noise.SetGradientPerturbFreq(p_freq);
	emit_changed();
}

real_t FastNoise::get_perturb_frequency() const {
	return _noise.GetGradientPerturbFreq();
}

void FastNoise::perturb_2d(real_t &p_x, real_t &p_y) {
	_noise.GradientPerturb(p_x, p_y);
}

Vector2 FastNoise::perturb_2dv(Vector2 p_pos) {
	real_t x = p_pos.x;
	real_t y = p_pos.y;
	_noise.GradientPerturb(x, y);
	Vector2 v = { x, y };
	return v;
}

void FastNoise::perturb_fractal_2d(real_t &p_x, real_t &p_y) {
	_noise.GradientPerturbFractal(p_x, p_y);
}

Vector2 FastNoise::perturb_fractal_2dv(Vector2 p_pos) {
	real_t x = p_pos.x;
	real_t y = p_pos.y;
	_noise.GradientPerturbFractal(x, y);
	Vector2 v = { x, y };
	return v;
}

void FastNoise::perturb_3d(real_t &p_x, real_t &p_y, real_t &p_z) {
	_noise.GradientPerturb(p_x, p_y, p_z);
}

Vector3 FastNoise::perturb_3dv(Vector3 p_pos) {
	real_t x = p_pos.x;
	real_t y = p_pos.y;
	real_t z = p_pos.z;
	_noise.GradientPerturb(x, y, z);
	Vector3 v = { x, y, z };
	return v;
}

void FastNoise::perturb_fractal_3d(real_t &p_x, real_t &p_y, real_t &p_z) {
	_noise.GradientPerturbFractal(p_x, p_y, p_z);
}

Vector3 FastNoise::perturb_fractal_3dv(Vector3 p_pos) {
	real_t x = p_pos.x;
	real_t y = p_pos.y;
	real_t z = p_pos.z;
	_noise.GradientPerturbFractal(x, y, z);
	Vector3 v = { x, y, z };
	return v;
}

// Fractal

void FastNoise::set_fractal_type(FractalType p_type) {
	_noise.SetFractalType((_FastNoise::FractalType)p_type);
	emit_changed();
}

FastNoise::FractalType FastNoise::get_fractal_type() const {
	return (FractalType)_noise.GetFractalType();
}

void FastNoise::set_fractal_octaves(int p_octaves) {
	_noise.SetFractalOctaves(p_octaves);
	emit_changed();
}

int FastNoise::get_fractal_octaves() const {
	return _noise.GetFractalOctaves();
}

void FastNoise::set_fractal_lacunarity(real_t p_lacunarity) {
	_noise.SetFractalLacunarity(p_lacunarity);
	emit_changed();
}

real_t FastNoise::get_fractal_lacunarity() const {
	return _noise.GetFractalLacunarity();
}

void FastNoise::set_fractal_gain(real_t p_gain) {
	_noise.SetFractalGain(p_gain);
	emit_changed();
}

real_t FastNoise::get_fractal_gain() const {
	return _noise.GetFractalGain();
}

// Cellular

void FastNoise::set_cellular_distance_function(CellularDistanceFunction p_func) {
	_noise.SetCellularDistanceFunction((_FastNoise::CellularDistanceFunction)p_func);
	emit_changed();
}

FastNoise::CellularDistanceFunction FastNoise::get_cellular_distance_function() const {
	return (CellularDistanceFunction)_noise.GetCellularDistanceFunction();
}

void FastNoise::set_cellular_jitter(real_t p_jitter) {
	_noise.SetCellularJitter(p_jitter);
	emit_changed();
}

real_t FastNoise::get_cellular_jitter() const {
	return _noise.GetCellularJitter();
}

void FastNoise::set_cellular_return_type(CellularReturnType p_ret) {
	_noise.SetCellularReturnType((_FastNoise::CellularReturnType)p_ret);

	if (p_ret == RETURN_NOISE_LOOKUP && _cellular_lookup_ref.is_null()) {
		Ref<FastNoise> noise;
		noise.instance();
		set_cellular_noise_lookup(noise);
	}

	emit_changed();
}

FastNoise::CellularReturnType FastNoise::get_cellular_return_type() const {
	return (CellularReturnType)_noise.GetCellularReturnType();
}

void FastNoise::set_cellular_distance2_indices(int p_index0, int p_index1) {
	// Valid range for index1: 1-3
	if (p_index1 > 3) {
		_cell_dist_index1 = 3;
	} else if (p_index1 < 1) {
		_cell_dist_index1 = 1;
	} else {
		_cell_dist_index1 = p_index1;
	}

	// Valid range for index0: 0-2 and < index1
	_cell_dist_index0 = p_index0;
	if (_cell_dist_index0 >= _cell_dist_index1) {
		_cell_dist_index0 = _cell_dist_index1 - 1;
	}

	if (_cell_dist_index0 < 0) {
		_cell_dist_index0 = 0;
	}

	_noise.SetCellularDistance2Indices(_cell_dist_index0, _cell_dist_index1);
	emit_changed();
}

PackedInt32Array FastNoise::get_cellular_distance2_indices() const {
	int i0, i1;
	_noise.GetCellularDistance2Indices(i0, i1);
	PackedInt32Array a;
	a.append(i0);
	a.append(i1);
	return a;
}

void FastNoise::set_cellular_distance2_index0(int p_index0) {
	set_cellular_distance2_indices(p_index0, _cell_dist_index1);
}

int FastNoise::get_cellular_distance2_index0() const {
	return _cell_dist_index0;
}

void FastNoise::set_cellular_distance2_index1(int p_index1) {
	set_cellular_distance2_indices(_cell_dist_index0, p_index1);
}

int FastNoise::get_cellular_distance2_index1() const {
	return _cell_dist_index1;
}

void FastNoise::set_cellular_noise_lookup(Ref<FastNoise> p_noise_obj) {
	_cellular_lookup_ref = p_noise_obj;

	if (_cellular_lookup_ref.is_null()) {
		if (get_cellular_return_type() == RETURN_NOISE_LOOKUP) {
			set_cellular_return_type(RETURN_CELL_VALUE);
		}
		_noise.SetCellularNoiseLookup(NULL);
	} else {
		_noise.SetCellularNoiseLookup(&_cellular_lookup_ref->_noise);
		_cellular_lookup_ref->connect("changed", callable_mp(this, &FastNoise::_changed));
	}

	_change_notify();
	emit_changed();
}

Ref<FastNoise> FastNoise::get_cellular_noise_lookup() const {
	return _cellular_lookup_ref;
}

// Textures

Ref<Image> FastNoise::get_image(int p_width, int p_height, bool p_invert) {
	Vector<uint8_t> data;
	data.resize(p_width * p_height * 4);

	uint8_t *wd8 = data.ptrw();

	// Get all values and identify min/max values
	Vector<real_t> values;
	values.resize(p_width * p_height);
	real_t min_val = 100;
	real_t max_val = -100;

	for (int y = 0, i = 0; y < p_height; y++) {
		for (int x = 0; x < p_width; x++, i++) {
			values.set(i, get_noise_2d(x, y));
			if (values[i] > max_val) {
				max_val = values[i];
			}
			if (values[i] < min_val) {
				min_val = values[i];
			}
		}
	}

	// Normalize values and write to texture
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
			wd8[x * 4 + 0] = value;
			wd8[x * 4 + 1] = value;
			wd8[x * 4 + 2] = value;
			wd8[x * 4 + 3] = 255;
		}
	}

	Ref<Image> image = memnew(Image(p_width, p_height, false, Image::FORMAT_RGBA8, data));
	return image;
}

Ref<Image> FastNoise::get_seamless_image(int p_width, int p_height, bool p_invert) {
	// Just return parent function. This is here only so Godot will properly document this function.
	return Noise::get_seamless_image(p_width, p_height, p_invert);
}

void FastNoise::_changed() {
	emit_changed();
}

void FastNoise::_bind_methods() {
	// General settings

	ClassDB::bind_method(D_METHOD("set_noise_type", "type"), &FastNoise::set_noise_type);
	ClassDB::bind_method(D_METHOD("get_noise_type"), &FastNoise::get_noise_type);
#ifdef SIMPLEX_ENABLED
	ADD_PROPERTY(PropertyInfo(Variant::INT, "noise_type", PROPERTY_HINT_ENUM,
						 "Value,ValueFractal,Perlin,PerlinFractal,Simplex,SimplexFractal,Cellular,WhiteNoise,Cubic,CubicFractal"),
			"set_noise_type", "get_noise_type");
#else
	ADD_PROPERTY(PropertyInfo(Variant::INT, "noise_type", PROPERTY_HINT_ENUM,
						 "Value,ValueFractal,Perlin,PerlinFractal,Cellular,WhiteNoise,Cubic,CubicFractal"),
			"set_noise_type", "get_noise_type");
#endif

	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &FastNoise::set_seed);
	ClassDB::bind_method(D_METHOD("get_seed"), &FastNoise::get_seed);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

	ClassDB::bind_method(D_METHOD("set_frequency", "freq"), &FastNoise::set_frequency);
	ClassDB::bind_method(D_METHOD("get_frequency"), &FastNoise::get_frequency);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "frequency", PROPERTY_HINT_RANGE, ".001,1"), "set_frequency", "get_frequency");

	ClassDB::bind_method(D_METHOD("set_interpolation", "interp"), &FastNoise::set_interpolation);
	ClassDB::bind_method(D_METHOD("get_interpolation"), &FastNoise::get_interpolation);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "interpolation", PROPERTY_HINT_ENUM, "Linear (Fastest),Hermite,Quintic (Slowest)"), "set_interpolation", "get_interpolation");

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &FastNoise::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &FastNoise::get_offset);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "offset", PROPERTY_HINT_RANGE, "-999999999,999999999,1"), "set_offset", "get_offset");

	// Noise functions

	ClassDB::bind_method(D_METHOD("get_noise_1d", "x"), &FastNoise::get_noise_1d);
	ClassDB::bind_method(D_METHOD("get_noise_2d", "x", "y"), &FastNoise::get_noise_2d);
	ClassDB::bind_method(D_METHOD("get_noise_2dv", "v"), &FastNoise::get_noise_2dv);
	ClassDB::bind_method(D_METHOD("get_noise_3d", "x", "y", "z"), &FastNoise::get_noise_3d);
	ClassDB::bind_method(D_METHOD("get_noise_3dv", "v"), &FastNoise::get_noise_3dv);
	ClassDB::bind_method(D_METHOD("get_white_noise_4d", "x", "y", "z", "w"), &FastNoise::get_white_noise_4d);
#ifdef SIMPLEX_ENABLED
	ClassDB::bind_method(D_METHOD("get_simplex_4d", "x", "y", "z", "w"), &FastNoise::get_simplex_4d);
#endif

	// Perturb

	ADD_GROUP("Perturb", "perturb_");
	ClassDB::bind_method(D_METHOD("set_perturb_type", "type"), &FastNoise::set_perturb_type);
	ClassDB::bind_method(D_METHOD("get_perturb_type"), &FastNoise::get_perturb_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "perturb_type", PROPERTY_HINT_ENUM, "None,Gradient,Gradient_Fractal"), "set_perturb_type", "get_perturb_type");

	ClassDB::bind_method(D_METHOD("set_perturb_amplitude", "amp"), &FastNoise::set_perturb_amplitude);
	ClassDB::bind_method(D_METHOD("get_perturb_amplitude"), &FastNoise::get_perturb_amplitude);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "perturb_amplitude"), "set_perturb_amplitude", "get_perturb_amplitude");

	ClassDB::bind_method(D_METHOD("set_perturb_frequency", "freq"), &FastNoise::set_perturb_frequency);
	ClassDB::bind_method(D_METHOD("get_perturb_frequency"), &FastNoise::get_perturb_frequency);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "perturb_frequency"), "set_perturb_frequency", "get_perturb_frequency");

	ClassDB::bind_method(D_METHOD("perturb_2dv", "pos"), &FastNoise::perturb_2dv);
	ClassDB::bind_method(D_METHOD("perturb_fractal_2dv", "pos"), &FastNoise::perturb_fractal_2dv);
	ClassDB::bind_method(D_METHOD("perturb_3dv", "pos"), &FastNoise::perturb_3dv);
	ClassDB::bind_method(D_METHOD("perturb_fractal_3dv", "pos"), &FastNoise::perturb_fractal_3dv);

	// Fractal

	ADD_GROUP("Fractal", "fractal_");
	ClassDB::bind_method(D_METHOD("set_fractal_type", "type"), &FastNoise::set_fractal_type);
	ClassDB::bind_method(D_METHOD("get_fractal_type"), &FastNoise::get_fractal_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fractal_type", PROPERTY_HINT_ENUM, "FBM,Billow,RidgedMulti"), "set_fractal_type", "get_fractal_type");

	ClassDB::bind_method(D_METHOD("set_fractal_octaves", "octave_count"), &FastNoise::set_fractal_octaves);
	ClassDB::bind_method(D_METHOD("get_fractal_octaves"), &FastNoise::get_fractal_octaves);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fractal_octaves", PROPERTY_HINT_RANGE, "1,10,1"), "set_fractal_octaves", "get_fractal_octaves");

	ClassDB::bind_method(D_METHOD("set_fractal_lacunarity", "lacunarity"), &FastNoise::set_fractal_lacunarity);
	ClassDB::bind_method(D_METHOD("get_fractal_lacunarity"), &FastNoise::get_fractal_lacunarity);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fractal_lacunarity"), "set_fractal_lacunarity", "get_fractal_lacunarity");

	ClassDB::bind_method(D_METHOD("set_fractal_gain", "gain"), &FastNoise::set_fractal_gain);
	ClassDB::bind_method(D_METHOD("get_fractal_gain"), &FastNoise::get_fractal_gain);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fractal_gain"), "set_fractal_gain", "get_fractal_gain");

	// Cellular

	ADD_GROUP("Cellular", "cellular_");
	ClassDB::bind_method(D_METHOD("set_cellular_distance_function", "func"), &FastNoise::set_cellular_distance_function);
	ClassDB::bind_method(D_METHOD("get_cellular_distance_function"), &FastNoise::get_cellular_distance_function);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cellular_dist_func", PROPERTY_HINT_ENUM, "Euclidean,Manhattan,Natural"), "set_cellular_distance_function", "get_cellular_distance_function");

	ClassDB::bind_method(D_METHOD("set_cellular_jitter", "jitter"), &FastNoise::set_cellular_jitter);
	ClassDB::bind_method(D_METHOD("get_cellular_jitter"), &FastNoise::get_cellular_jitter);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cellular_jitter"), "set_cellular_jitter", "get_cellular_jitter");

	ClassDB::bind_method(D_METHOD("set_cellular_return_type", "ret"), &FastNoise::set_cellular_return_type);
	ClassDB::bind_method(D_METHOD("get_cellular_return_type"), &FastNoise::get_cellular_return_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cellular_return_type", PROPERTY_HINT_ENUM, "CellValue,Distance,Distance2,Distance2Add,Distance2Sub,Distance2Mul,Distance2Div,NoiseLookup"), "set_cellular_return_type", "get_cellular_return_type");

	ClassDB::bind_method(D_METHOD("set_cellular_distance2_indices", "index0", "index1"), &FastNoise::set_cellular_distance2_indices);
	ClassDB::bind_method(D_METHOD("get_cellular_distance2_indices"), &FastNoise::get_cellular_distance2_indices);
	ClassDB::bind_method(D_METHOD("set_cellular_distance2_index0", "index0"), &FastNoise::set_cellular_distance2_index0);
	ClassDB::bind_method(D_METHOD("get_cellular_distance2_index0"), &FastNoise::get_cellular_distance2_index0);
	ClassDB::bind_method(D_METHOD("set_cellular_distance2_index1", "index1"), &FastNoise::set_cellular_distance2_index1);
	ClassDB::bind_method(D_METHOD("get_cellular_distance2_index1"), &FastNoise::get_cellular_distance2_index1);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cellular_distance2_index0", PROPERTY_HINT_RANGE, "0,2,1"), "set_cellular_distance2_index0", "get_cellular_distance2_index0");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cellular_distance2_index1", PROPERTY_HINT_RANGE, "1,3,1"), "set_cellular_distance2_index1", "get_cellular_distance2_index1");

	ClassDB::bind_method(D_METHOD("set_cellular_noise_lookup", "other_noise"), &FastNoise::set_cellular_noise_lookup);
	ClassDB::bind_method(D_METHOD("get_cellular_noise_lookup"), &FastNoise::get_cellular_noise_lookup);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "cellular_noise_lookup", PROPERTY_HINT_RESOURCE_TYPE, "FastNoise"), "set_cellular_noise_lookup", "get_cellular_noise_lookup");

	// Textures

	ClassDB::bind_method(D_METHOD("get_image", "width", "height", "invert"), &FastNoise::get_image, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_seamless_image", "width", "height", "invert"), &FastNoise::get_seamless_image, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("_changed"), &FastNoise::_changed);

	BIND_ENUM_CONSTANT(TYPE_VALUE);
	BIND_ENUM_CONSTANT(TYPE_VALUE_FRACTAL);
	BIND_ENUM_CONSTANT(TYPE_PERLIN);
	BIND_ENUM_CONSTANT(TYPE_PERLIN_FRACTAL);
#ifdef SIMPLEX_ENABLED
	BIND_ENUM_CONSTANT(TYPE_SIMPLEX);
	BIND_ENUM_CONSTANT(TYPE_SIMPLEX_FRACTAL);
#endif

	BIND_ENUM_CONSTANT(TYPE_CELLULAR);
	BIND_ENUM_CONSTANT(TYPE_WHITE_NOISE);
	BIND_ENUM_CONSTANT(TYPE_CUBIC);
	BIND_ENUM_CONSTANT(TYPE_CUBIC_FRACTAL);

	BIND_ENUM_CONSTANT(INTERP_LINEAR);
	BIND_ENUM_CONSTANT(INTERP_HERMITE);
	BIND_ENUM_CONSTANT(INTERP_QUINTIC);

	BIND_ENUM_CONSTANT(FRACTAL_FBM);
	BIND_ENUM_CONSTANT(FRACTAL_BILLOW);
	BIND_ENUM_CONSTANT(FRACTAL_RIDGED_MULTI);

	BIND_ENUM_CONSTANT(DISTANCE_EUCLIDEAN);
	BIND_ENUM_CONSTANT(DISTANCE_MANHATTAN);
	BIND_ENUM_CONSTANT(DISTANCE_NATURAL);

	BIND_ENUM_CONSTANT(RETURN_CELL_VALUE);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_ADD);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_SUB);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_MUL);
	BIND_ENUM_CONSTANT(RETURN_DISTANCE2_DIV);
	BIND_ENUM_CONSTANT(RETURN_NOISE_LOOKUP);

	BIND_ENUM_CONSTANT(PERTURB_NONE);
	BIND_ENUM_CONSTANT(PERTURB_GRADIENT);
	BIND_ENUM_CONSTANT(PERTURB_GRADIENT_FRACTAL);
}

void FastNoise::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("cellular_") && get_noise_type() != TYPE_CELLULAR) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
		return;
	}

	if (property.name.begins_with("fractal_")) {
		switch (get_noise_type()) {
			case TYPE_VALUE:
			case TYPE_PERLIN:
#ifdef SIMPLEX_ENABLED
			case TYPE_SIMPLEX:
#endif
			case TYPE_WHITE_NOISE:
			case TYPE_CELLULAR:
			case TYPE_CUBIC:
				if (get_perturb_type() != PERTURB_GRADIENT_FRACTAL) {
					property.usage = PROPERTY_USAGE_NOEDITOR;
					return;
				}
				break;
			default:
				break;
		}
	}
}
