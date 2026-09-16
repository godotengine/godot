/**************************************************************************/
/*  foliage_spawner_3d.h                                                  */
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

#pragma once

#include "scene/3d/multimesh_instance_3d.h"

class Image;
class Mesh;
class RandomPCG;
class Texture2D;

// Scatters instances of a mesh inside a configurable box volume, similar in spirit
// to Unreal Engine's Procedural Foliage Spawner: it fills the volume according to
// density/spacing rules, an optional grayscale distribution mask, and can project
// the instances onto physics colliders below them (e.g. terrain).
class FoliageSpawner3D : public MultiMeshInstance3D {
	GDCLASS(FoliageSpawner3D, MultiMeshInstance3D);

	Ref<Mesh> mesh;

	// Volume.
	Vector3 volume_size = Vector3(10, 4, 10);

	// Distribution.
	float density = 1.0;
	int seed = 0;
	int max_instances = 2048;
	int max_attempts_factor = 30;
	float min_distance = 0.5;

	// Mask.
	Ref<Texture2D> distribution_mask;
	bool mask_invert = false;

	// Ground projection.
	bool project_on_collision = true;
	uint32_t collision_mask = 1;
	float max_slope_degrees = 45.0;
	bool align_to_normal = true;
	float align_to_normal_amount = 1.0;

	// Randomization.
	bool random_rotation = true;
	float random_tilt_degrees = 0.0;
	float min_scale = 0.9;
	float max_scale = 1.1;

	Ref<Image> _get_mask_image() const;
	bool _sample_mask(const Ref<Image> &p_image, const Vector2 &p_uv, RandomPCG &p_rng) const;
	bool _project_point(const Vector3 &p_local_xz_top, Vector3 &r_local_position, Vector3 &r_world_normal) const;
	Callable _get_regenerate_button() const;

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_volume_size(const Vector3 &p_size);
	Vector3 get_volume_size() const;

	void set_density(float p_density);
	float get_density() const;

	void set_seed(int p_seed);
	int get_seed() const;

	void set_max_instances(int p_max_instances);
	int get_max_instances() const;

	void set_max_attempts_factor(int p_factor);
	int get_max_attempts_factor() const;

	void set_min_distance(float p_min_distance);
	float get_min_distance() const;

	void set_distribution_mask(const Ref<Texture2D> &p_mask);
	Ref<Texture2D> get_distribution_mask() const;

	void set_mask_invert(bool p_invert);
	bool is_mask_inverted() const;

	void set_project_on_collision(bool p_project);
	bool is_projecting_on_collision() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_max_slope_degrees(float p_degrees);
	float get_max_slope_degrees() const;

	void set_align_to_normal(bool p_align);
	bool is_aligned_to_normal() const;

	void set_align_to_normal_amount(float p_amount);
	float get_align_to_normal_amount() const;

	void set_random_rotation(bool p_random);
	bool is_random_rotation_enabled() const;

	void set_random_tilt_degrees(float p_degrees);
	float get_random_tilt_degrees() const;

	void set_min_scale(float p_scale);
	float get_min_scale() const;

	void set_max_scale(float p_scale);
	float get_max_scale() const;

	void regenerate();

	virtual AABB get_aabb() const override;
	PackedStringArray get_configuration_warnings() const override;

	FoliageSpawner3D();
};
