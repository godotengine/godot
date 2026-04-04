/**************************************************************************/
/*  fast_noise_lite.hpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/noise.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class FastNoiseLite : public Noise {
	GDEXTENSION_CLASS(FastNoiseLite, Noise)

public:
	enum NoiseType {
		TYPE_VALUE = 5,
		TYPE_VALUE_CUBIC = 4,
		TYPE_PERLIN = 3,
		TYPE_CELLULAR = 2,
		TYPE_SIMPLEX = 0,
		TYPE_SIMPLEX_SMOOTH = 1,
	};

	enum FractalType {
		FRACTAL_NONE = 0,
		FRACTAL_FBM = 1,
		FRACTAL_RIDGED = 2,
		FRACTAL_PING_PONG = 3,
	};

	enum CellularDistanceFunction {
		DISTANCE_EUCLIDEAN = 0,
		DISTANCE_EUCLIDEAN_SQUARED = 1,
		DISTANCE_MANHATTAN = 2,
		DISTANCE_HYBRID = 3,
	};

	enum CellularReturnType {
		RETURN_CELL_VALUE = 0,
		RETURN_DISTANCE = 1,
		RETURN_DISTANCE2 = 2,
		RETURN_DISTANCE2_ADD = 3,
		RETURN_DISTANCE2_SUB = 4,
		RETURN_DISTANCE2_MUL = 5,
		RETURN_DISTANCE2_DIV = 6,
	};

	enum DomainWarpType {
		DOMAIN_WARP_SIMPLEX = 0,
		DOMAIN_WARP_SIMPLEX_REDUCED = 1,
		DOMAIN_WARP_BASIC_GRID = 2,
	};

	enum DomainWarpFractalType {
		DOMAIN_WARP_FRACTAL_NONE = 0,
		DOMAIN_WARP_FRACTAL_PROGRESSIVE = 1,
		DOMAIN_WARP_FRACTAL_INDEPENDENT = 2,
	};

	void set_noise_type(FastNoiseLite::NoiseType p_type);
	FastNoiseLite::NoiseType get_noise_type() const;
	void set_seed(int32_t p_seed);
	int32_t get_seed() const;
	void set_frequency(float p_freq);
	float get_frequency() const;
	void set_offset(const Vector3 &p_offset);
	Vector3 get_offset() const;
	void set_fractal_type(FastNoiseLite::FractalType p_type);
	FastNoiseLite::FractalType get_fractal_type() const;
	void set_fractal_octaves(int32_t p_octave_count);
	int32_t get_fractal_octaves() const;
	void set_fractal_lacunarity(float p_lacunarity);
	float get_fractal_lacunarity() const;
	void set_fractal_gain(float p_gain);
	float get_fractal_gain() const;
	void set_fractal_weighted_strength(float p_weighted_strength);
	float get_fractal_weighted_strength() const;
	void set_fractal_ping_pong_strength(float p_ping_pong_strength);
	float get_fractal_ping_pong_strength() const;
	void set_cellular_distance_function(FastNoiseLite::CellularDistanceFunction p_func);
	FastNoiseLite::CellularDistanceFunction get_cellular_distance_function() const;
	void set_cellular_jitter(float p_jitter);
	float get_cellular_jitter() const;
	void set_cellular_return_type(FastNoiseLite::CellularReturnType p_ret);
	FastNoiseLite::CellularReturnType get_cellular_return_type() const;
	void set_domain_warp_enabled(bool p_domain_warp_enabled);
	bool is_domain_warp_enabled() const;
	void set_domain_warp_type(FastNoiseLite::DomainWarpType p_domain_warp_type);
	FastNoiseLite::DomainWarpType get_domain_warp_type() const;
	void set_domain_warp_amplitude(float p_domain_warp_amplitude);
	float get_domain_warp_amplitude() const;
	void set_domain_warp_frequency(float p_domain_warp_frequency);
	float get_domain_warp_frequency() const;
	void set_domain_warp_fractal_type(FastNoiseLite::DomainWarpFractalType p_domain_warp_fractal_type);
	FastNoiseLite::DomainWarpFractalType get_domain_warp_fractal_type() const;
	void set_domain_warp_fractal_octaves(int32_t p_domain_warp_octave_count);
	int32_t get_domain_warp_fractal_octaves() const;
	void set_domain_warp_fractal_lacunarity(float p_domain_warp_lacunarity);
	float get_domain_warp_fractal_lacunarity() const;
	void set_domain_warp_fractal_gain(float p_domain_warp_gain);
	float get_domain_warp_fractal_gain() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Noise::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(FastNoiseLite::NoiseType);
VARIANT_ENUM_CAST(FastNoiseLite::FractalType);
VARIANT_ENUM_CAST(FastNoiseLite::CellularDistanceFunction);
VARIANT_ENUM_CAST(FastNoiseLite::CellularReturnType);
VARIANT_ENUM_CAST(FastNoiseLite::DomainWarpType);
VARIANT_ENUM_CAST(FastNoiseLite::DomainWarpFractalType);

