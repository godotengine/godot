/*************************************************************************/
/*  fastnoise.h                                                          */
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

#ifndef FASTNOISE_H
#define FASTNOISE_H

#include "core/image.h"
#include "core/reference.h"
#include "noise.h"
#include "scene/resources/texture.h"

#include <thirdparty/noise/FastNoise.h>

typedef fastnoise::FastNoise _FastNoise;

class FastNoise : public Noise {
	GDCLASS(FastNoise, Noise);
	OBJ_SAVE_TYPE(FastNoise);

public:
	enum NoiseType {
		TYPE_VALUE = _FastNoise::Value,
		TYPE_VALUE_FRACTAL = _FastNoise::ValueFractal,
		TYPE_PERLIN = _FastNoise::Perlin,
		TYPE_PERLIN_FRACTAL = _FastNoise::PerlinFractal,
#ifdef SIMPLEX_ENABLED
		TYPE_SIMPLEX = _FastNoise::Simplex,
		TYPE_SIMPLEX_FRACTAL = _FastNoise::SimplexFractal,
#endif
		TYPE_CELLULAR = _FastNoise::Cellular,
		TYPE_WHITE_NOISE = _FastNoise::WhiteNoise,
		TYPE_CUBIC = _FastNoise::Cubic,
		TYPE_CUBIC_FRACTAL = _FastNoise::CubicFractal
	};

	enum Interpolation {
		INTERP_LINEAR = _FastNoise::Linear,
		INTERP_HERMITE = _FastNoise::Hermite,
		INTERP_QUINTIC = _FastNoise::Quintic
	};

	enum FractalType {
		FRACTAL_FBM = _FastNoise::FBM,
		FRACTAL_BILLOW = _FastNoise::Billow,
		FRACTAL_RIDGED_MULTI = _FastNoise::RigidMulti
	};

	enum CellularDistanceFunction {
		DISTANCE_EUCLIDEAN = _FastNoise::Euclidean,
		DISTANCE_MANHATTAN = _FastNoise::Manhattan,
		DISTANCE_NATURAL = _FastNoise::Natural
	};

	enum CellularReturnType {
		RETURN_CELL_VALUE = _FastNoise::CellValue,
		RETURN_DISTANCE = _FastNoise::Distance,
		RETURN_DISTANCE2 = _FastNoise::Distance2,
		RETURN_DISTANCE2_ADD = _FastNoise::Distance2Add,
		RETURN_DISTANCE2_SUB = _FastNoise::Distance2Sub,
		RETURN_DISTANCE2_MUL = _FastNoise::Distance2Mul,
		RETURN_DISTANCE2_DIV = _FastNoise::Distance2Div,
		RETURN_NOISE_LOOKUP = _FastNoise::NoiseLookup
	};

	enum PerturbType {
		PERTURB_NONE = 0,
		PERTURB_GRADIENT,
		PERTURB_GRADIENT_FRACTAL
	};

	FastNoise();
	~FastNoise();

	// General noise settings

	void set_noise_type(NoiseType p_noise_type);
	NoiseType get_noise_type() const;

	void set_seed(int p_seed);
	int get_seed() const;

	void set_frequency(real_t p_freq);
	real_t get_frequency() const;

	void set_interpolation(Interpolation p_interp);
	Interpolation get_interpolation() const;

	void set_offset(Vector3 p_offset);
	Vector3 get_offset() const;

	// Noise functions

	real_t get_noise_1d(real_t p_x);

	real_t get_noise_2dv(Vector2 p_v);
	real_t get_noise_2d(real_t p_x, real_t p_y);

	real_t get_noise_3dv(Vector3 p_v);
	real_t get_noise_3d(real_t p_x, real_t p_y, real_t p_z);

	real_t get_white_noise_4d(real_t p_x, real_t p_y, real_t p_z, real_t p_w);
#ifdef SIMPLEX_ENABLED
	real_t get_simplex_4d(real_t p_x, real_t p_y, real_t p_z, real_t p_w);
#endif

	// Perturb your coordinates prior to using the noise functions

	void set_perturb_type(PerturbType p_type);
	PerturbType get_perturb_type() const;

	void set_perturb_amplitude(real_t p_amp);
	real_t get_perturb_amplitude() const;

	void set_perturb_frequency(real_t p_freq);
	real_t get_perturb_frequency() const;

	void perturb_2d(real_t &p_x, real_t &p_y);
	Vector2 perturb_2dv(Vector2 p_pos);
	void perturb_fractal_2d(real_t &p_x, real_t &p_y);
	Vector2 perturb_fractal_2dv(Vector2 p_pos);

	void perturb_3d(real_t &p_x, real_t &p_y, real_t &p_z);
	Vector3 perturb_3dv(Vector3 p_pos);
	void perturb_fractal_3d(real_t &p_x, real_t &p_y, real_t &p_z);
	Vector3 perturb_fractal_3dv(Vector3 p_pos);

	// Fractal specific

	void set_fractal_type(FractalType p_type);
	FractalType get_fractal_type() const;

	void set_fractal_octaves(int p_octaves);
	int get_fractal_octaves() const;

	void set_fractal_lacunarity(real_t p_lacunarity);
	real_t get_fractal_lacunarity() const;

	void set_fractal_gain(real_t p_gain);
	real_t get_fractal_gain() const;

	// Cellular specific

	void set_cellular_distance_function(CellularDistanceFunction p_func);
	CellularDistanceFunction get_cellular_distance_function() const;

	void set_cellular_jitter(real_t p_jitter);
	real_t get_cellular_jitter() const;

	void set_cellular_return_type(CellularReturnType p_ret);
	CellularReturnType get_cellular_return_type() const;

	void set_cellular_distance2_indices(int p_index0, int p_index1);
	PackedInt32Array get_cellular_distance2_indices() const;

	void set_cellular_distance2_index0(int p_index0); // Editor helpers
	int get_cellular_distance2_index0() const;
	void set_cellular_distance2_index1(int p_index1);
	int get_cellular_distance2_index1() const;

	void set_cellular_noise_lookup(Ref<FastNoise> p_noise_obj);
	Ref<FastNoise> get_cellular_noise_lookup() const;

	// Generate Textures

	Ref<Image> get_image(int p_width, int p_height, bool p_invert = false) override;
	Ref<Image> get_seamless_image(int p_width, int p_height, bool p_invert = false) override;

	void _changed();

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;

private:
	_FastNoise _noise;
	Vector3 _offset;
	PerturbType _perturb;
	Ref<FastNoise> _cellular_lookup_ref;
	int _cell_dist_index0;
	int _cell_dist_index1;
};

VARIANT_ENUM_CAST(FastNoise::NoiseType);
VARIANT_ENUM_CAST(FastNoise::FractalType);
VARIANT_ENUM_CAST(FastNoise::Interpolation);
VARIANT_ENUM_CAST(FastNoise::CellularDistanceFunction);
VARIANT_ENUM_CAST(FastNoise::CellularReturnType);
VARIANT_ENUM_CAST(FastNoise::PerturbType);

#endif // FASTNOISE_H
