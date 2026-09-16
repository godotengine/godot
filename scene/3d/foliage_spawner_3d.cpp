/**************************************************************************/
/*  foliage_spawner_3d.cpp                                                */
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

#include "foliage_spawner_3d.h"

#include "core/io/image.h"
#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"
#include "core/object/class_db.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/mesh.h"
#include "scene/resources/multimesh.h"
#include "scene/resources/texture.h"
#include "servers/physics_3d/physics_server_3d.h"

void FoliageSpawner3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &FoliageSpawner3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &FoliageSpawner3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_volume_size", "size"), &FoliageSpawner3D::set_volume_size);
	ClassDB::bind_method(D_METHOD("get_volume_size"), &FoliageSpawner3D::get_volume_size);

	ClassDB::bind_method(D_METHOD("set_density", "density"), &FoliageSpawner3D::set_density);
	ClassDB::bind_method(D_METHOD("get_density"), &FoliageSpawner3D::get_density);

	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &FoliageSpawner3D::set_seed);
	ClassDB::bind_method(D_METHOD("get_seed"), &FoliageSpawner3D::get_seed);

	ClassDB::bind_method(D_METHOD("set_max_instances", "max_instances"), &FoliageSpawner3D::set_max_instances);
	ClassDB::bind_method(D_METHOD("get_max_instances"), &FoliageSpawner3D::get_max_instances);

	ClassDB::bind_method(D_METHOD("set_max_attempts_factor", "factor"), &FoliageSpawner3D::set_max_attempts_factor);
	ClassDB::bind_method(D_METHOD("get_max_attempts_factor"), &FoliageSpawner3D::get_max_attempts_factor);

	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &FoliageSpawner3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &FoliageSpawner3D::get_min_distance);

	ClassDB::bind_method(D_METHOD("set_distribution_mask", "mask"), &FoliageSpawner3D::set_distribution_mask);
	ClassDB::bind_method(D_METHOD("get_distribution_mask"), &FoliageSpawner3D::get_distribution_mask);

	ClassDB::bind_method(D_METHOD("set_mask_invert", "invert"), &FoliageSpawner3D::set_mask_invert);
	ClassDB::bind_method(D_METHOD("is_mask_inverted"), &FoliageSpawner3D::is_mask_inverted);

	ClassDB::bind_method(D_METHOD("set_project_on_collision", "project"), &FoliageSpawner3D::set_project_on_collision);
	ClassDB::bind_method(D_METHOD("is_projecting_on_collision"), &FoliageSpawner3D::is_projecting_on_collision);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &FoliageSpawner3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &FoliageSpawner3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_max_slope_degrees", "degrees"), &FoliageSpawner3D::set_max_slope_degrees);
	ClassDB::bind_method(D_METHOD("get_max_slope_degrees"), &FoliageSpawner3D::get_max_slope_degrees);

	ClassDB::bind_method(D_METHOD("set_align_to_normal", "align"), &FoliageSpawner3D::set_align_to_normal);
	ClassDB::bind_method(D_METHOD("is_aligned_to_normal"), &FoliageSpawner3D::is_aligned_to_normal);

	ClassDB::bind_method(D_METHOD("set_align_to_normal_amount", "amount"), &FoliageSpawner3D::set_align_to_normal_amount);
	ClassDB::bind_method(D_METHOD("get_align_to_normal_amount"), &FoliageSpawner3D::get_align_to_normal_amount);

	ClassDB::bind_method(D_METHOD("set_random_rotation", "random"), &FoliageSpawner3D::set_random_rotation);
	ClassDB::bind_method(D_METHOD("is_random_rotation_enabled"), &FoliageSpawner3D::is_random_rotation_enabled);

	ClassDB::bind_method(D_METHOD("set_random_tilt_degrees", "degrees"), &FoliageSpawner3D::set_random_tilt_degrees);
	ClassDB::bind_method(D_METHOD("get_random_tilt_degrees"), &FoliageSpawner3D::get_random_tilt_degrees);

	ClassDB::bind_method(D_METHOD("set_min_scale", "scale"), &FoliageSpawner3D::set_min_scale);
	ClassDB::bind_method(D_METHOD("get_min_scale"), &FoliageSpawner3D::get_min_scale);

	ClassDB::bind_method(D_METHOD("set_max_scale", "scale"), &FoliageSpawner3D::set_max_scale);
	ClassDB::bind_method(D_METHOD("get_max_scale"), &FoliageSpawner3D::get_max_scale);

	ClassDB::bind_method(D_METHOD("regenerate"), &FoliageSpawner3D::regenerate);
	ClassDB::bind_method(D_METHOD("get_regenerate_button"), &FoliageSpawner3D::_get_regenerate_button);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "BoxMesh,SphereMesh,CapsuleMesh,CylinderMesh,PrismMesh,TorusMesh,PlaneMesh,QuadMesh,TextMesh,RibbonTrailMesh,TubeTrailMesh,PointMesh,ArrayMesh,ImmediateMesh,PlaceholderMesh,Mesh", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_mesh", "get_mesh");

	ADD_GROUP("Volume", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "volume_size", PROPERTY_HINT_RANGE, "0.01,4096,0.01,or_greater,suffix:m"), "set_volume_size", "get_volume_size");

	ADD_GROUP("Distribution", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "density", PROPERTY_HINT_RANGE, "0.0,50.0,0.001,or_greater"), "set_density", "get_density");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_instances", PROPERTY_HINT_RANGE, "0,65536,1,or_greater"), "set_max_instances", "get_max_instances");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_attempts_factor", PROPERTY_HINT_RANGE, "1,200,1,or_greater"), "set_max_attempts_factor", "get_max_attempts_factor");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_RANGE, "0.0,100.0,0.001,or_greater,suffix:m"), "set_min_distance", "get_min_distance");

	ADD_GROUP("Mask", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "distribution_mask", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_distribution_mask", "get_distribution_mask");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mask_invert"), "set_mask_invert", "is_mask_inverted");

	ADD_GROUP("Ground Projection", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "project_on_collision"), "set_project_on_collision", "is_projecting_on_collision");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_slope_degrees", PROPERTY_HINT_RANGE, "0,90,0.1,suffix:°"), "set_max_slope_degrees", "get_max_slope_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "align_to_normal"), "set_align_to_normal", "is_aligned_to_normal");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "align_to_normal_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_align_to_normal_amount", "get_align_to_normal_amount");

	ADD_GROUP("Randomization", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "random_rotation"), "set_random_rotation", "is_random_rotation_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "random_tilt_degrees", PROPERTY_HINT_RANGE, "0,90,0.1,suffix:°"), "set_random_tilt_degrees", "get_random_tilt_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_scale", PROPERTY_HINT_RANGE, "0.01,10.0,0.001,or_greater"), "set_min_scale", "get_min_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_scale", PROPERTY_HINT_RANGE, "0.01,10.0,0.001,or_greater"), "set_max_scale", "get_max_scale");

	ADD_GROUP("", "");
	ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "regenerate_button", PROPERTY_HINT_TOOL_BUTTON, "Regenerate", PROPERTY_USAGE_EDITOR), "", "get_regenerate_button");
}

void FoliageSpawner3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "multimesh") {
		// The multimesh is generated by this node; keep it out of the inspector
		// but still save it so baked instances persist with the scene.
		p_property.usage = PROPERTY_USAGE_STORAGE;
	}
}

Callable FoliageSpawner3D::_get_regenerate_button() const {
	return Callable(const_cast<FoliageSpawner3D *>(this), "regenerate");
}

void FoliageSpawner3D::set_mesh(const Ref<Mesh> &p_mesh) {
	mesh = p_mesh;
	Ref<MultiMesh> mm = get_multimesh();
	if (mm.is_valid()) {
		mm->set_mesh(mesh);
	}
	update_configuration_warnings();
}

Ref<Mesh> FoliageSpawner3D::get_mesh() const {
	return mesh;
}

void FoliageSpawner3D::set_volume_size(const Vector3 &p_size) {
	volume_size = p_size.maxf(0);
	update_gizmos();
	update_configuration_warnings();
}

Vector3 FoliageSpawner3D::get_volume_size() const {
	return volume_size;
}

void FoliageSpawner3D::set_density(float p_density) {
	density = MAX(p_density, 0.0f);
}

float FoliageSpawner3D::get_density() const {
	return density;
}

void FoliageSpawner3D::set_seed(int p_seed) {
	seed = p_seed;
}

int FoliageSpawner3D::get_seed() const {
	return seed;
}

void FoliageSpawner3D::set_max_instances(int p_max_instances) {
	max_instances = MAX(p_max_instances, 0);
}

int FoliageSpawner3D::get_max_instances() const {
	return max_instances;
}

void FoliageSpawner3D::set_max_attempts_factor(int p_factor) {
	max_attempts_factor = MAX(p_factor, 1);
}

int FoliageSpawner3D::get_max_attempts_factor() const {
	return max_attempts_factor;
}

void FoliageSpawner3D::set_min_distance(float p_min_distance) {
	min_distance = MAX(p_min_distance, 0.0f);
}

float FoliageSpawner3D::get_min_distance() const {
	return min_distance;
}

void FoliageSpawner3D::set_distribution_mask(const Ref<Texture2D> &p_mask) {
	distribution_mask = p_mask;
}

Ref<Texture2D> FoliageSpawner3D::get_distribution_mask() const {
	return distribution_mask;
}

void FoliageSpawner3D::set_mask_invert(bool p_invert) {
	mask_invert = p_invert;
}

bool FoliageSpawner3D::is_mask_inverted() const {
	return mask_invert;
}

void FoliageSpawner3D::set_project_on_collision(bool p_project) {
	project_on_collision = p_project;
	notify_property_list_changed();
}

bool FoliageSpawner3D::is_projecting_on_collision() const {
	return project_on_collision;
}

void FoliageSpawner3D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t FoliageSpawner3D::get_collision_mask() const {
	return collision_mask;
}

void FoliageSpawner3D::set_max_slope_degrees(float p_degrees) {
	max_slope_degrees = CLAMP(p_degrees, 0.0f, 90.0f);
}

float FoliageSpawner3D::get_max_slope_degrees() const {
	return max_slope_degrees;
}

void FoliageSpawner3D::set_align_to_normal(bool p_align) {
	align_to_normal = p_align;
	notify_property_list_changed();
}

bool FoliageSpawner3D::is_aligned_to_normal() const {
	return align_to_normal;
}

void FoliageSpawner3D::set_align_to_normal_amount(float p_amount) {
	align_to_normal_amount = CLAMP(p_amount, 0.0f, 1.0f);
}

float FoliageSpawner3D::get_align_to_normal_amount() const {
	return align_to_normal_amount;
}

void FoliageSpawner3D::set_random_rotation(bool p_random) {
	random_rotation = p_random;
}

bool FoliageSpawner3D::is_random_rotation_enabled() const {
	return random_rotation;
}

void FoliageSpawner3D::set_random_tilt_degrees(float p_degrees) {
	random_tilt_degrees = CLAMP(p_degrees, 0.0f, 90.0f);
}

float FoliageSpawner3D::get_random_tilt_degrees() const {
	return random_tilt_degrees;
}

void FoliageSpawner3D::set_min_scale(float p_scale) {
	min_scale = MAX(p_scale, 0.001f);
}

float FoliageSpawner3D::get_min_scale() const {
	return min_scale;
}

void FoliageSpawner3D::set_max_scale(float p_scale) {
	max_scale = MAX(p_scale, 0.001f);
}

float FoliageSpawner3D::get_max_scale() const {
	return max_scale;
}

Ref<Image> FoliageSpawner3D::_get_mask_image() const {
	if (distribution_mask.is_null()) {
		return Ref<Image>();
	}
	Ref<Image> img = distribution_mask->get_image();
	if (img.is_null()) {
		return Ref<Image>();
	}
	if (img->is_compressed()) {
		img = img->duplicate();
		img->decompress();
	}
	if (img->get_width() <= 0 || img->get_height() <= 0) {
		return Ref<Image>();
	}
	return img;
}

bool FoliageSpawner3D::_sample_mask(const Ref<Image> &p_image, const Vector2 &p_uv, RandomPCG &p_rng) const {
	const int w = p_image->get_width();
	const int h = p_image->get_height();
	const int px = CLAMP(int(p_uv.x * w), 0, w - 1);
	const int py = CLAMP(int((1.0f - p_uv.y) * h), 0, h - 1);

	float value = p_image->get_pixel(px, py).get_luminance();
	if (mask_invert) {
		value = 1.0f - value;
	}
	return p_rng.randf() <= value;
}

bool FoliageSpawner3D::_project_point(const Vector3 &p_local_xz_top, Vector3 &r_local_position, Vector3 &r_world_normal) const {
	Ref<World3D> world = get_world_3d();
	if (world.is_null()) {
		return false;
	}

	PhysicsServer3D *physics_server = PhysicsServer3D::get_singleton();
	if (physics_server == nullptr) {
		return false;
	}

	PhysicsDirectSpaceState3D *dss = physics_server->space_get_direct_state(world->get_space());
	if (dss == nullptr) {
		return false;
	}

	const Transform3D gt = get_global_transform();
	const Vector3 local_bottom = p_local_xz_top - Vector3(0, volume_size.y, 0);

	PS3DT::RayParameters ray_params;
	ray_params.from = gt.xform(p_local_xz_top);
	ray_params.to = gt.xform(local_bottom);
	ray_params.collision_mask = collision_mask;
	ray_params.collide_with_bodies = true;
	ray_params.collide_with_areas = false;

	PS3DT::RayResult result;
	if (!dss->intersect_ray(ray_params, result)) {
		return false;
	}

	if (max_slope_degrees < 90.0f) {
		const float angle = Math::rad_to_deg(Math::acos(CLAMP(result.normal.dot(Vector3(0, 1, 0)), -1.0f, 1.0f)));
		if (angle > max_slope_degrees) {
			return false;
		}
	}

	r_local_position = gt.affine_inverse().xform(result.position);
	r_world_normal = result.normal;
	return true;
}

void FoliageSpawner3D::regenerate() {
	if (mesh.is_null()) {
		set_multimesh(Ref<MultiMesh>());
		update_configuration_warnings();
		return;
	}

	const Vector3 half = volume_size * 0.5f;
	if (half.x <= 0.0f || half.z <= 0.0f) {
		set_multimesh(Ref<MultiMesh>());
		update_configuration_warnings();
		return;
	}

	const double area = double(volume_size.x) * double(volume_size.z);
	int target_count = int(Math::round(area * double(density)));
	target_count = CLAMP(target_count, 0, max_instances);

	RandomPCG rng;
	rng.seed(uint64_t(uint32_t(seed)));

	Ref<Image> mask_image = _get_mask_image();

	const Transform3D gt = get_global_transform();
	const Transform3D gt_inv = gt.affine_inverse();

	const float cell_size = MAX(min_distance, 0.001f);
	const float min_distance_sq = min_distance * min_distance;
	HashMap<Vector2i, LocalVector<Vector2>> grid;

	LocalVector<Transform3D> transforms;

	const int max_attempts = target_count > 0 ? target_count * MAX(max_attempts_factor, 1) : 0;
	int attempts = 0;

	while ((int)transforms.size() < target_count && attempts < max_attempts) {
		attempts++;

		const float lx = rng.random(-half.x, half.x);
		const float lz = rng.random(-half.z, half.z);

		if (mask_image.is_valid()) {
			const Vector2 uv((lx / volume_size.x) + 0.5f, (lz / volume_size.z) + 0.5f);
			if (!_sample_mask(mask_image, uv, rng)) {
				continue;
			}
		}

		const Vector2i cell(int(Math::floor(lx / cell_size)), int(Math::floor(lz / cell_size)));

		if (min_distance > 0.0f) {
			bool too_close = false;
			for (int cx = -1; cx <= 1 && !too_close; cx++) {
				for (int cz = -1; cz <= 1 && !too_close; cz++) {
					const LocalVector<Vector2> *bucket = grid.getptr(cell + Vector2i(cx, cz));
					if (bucket == nullptr) {
						continue;
					}
					for (const Vector2 &p : *bucket) {
						if (p.distance_squared_to(Vector2(lx, lz)) < min_distance_sq) {
							too_close = true;
							break;
						}
					}
				}
			}
			if (too_close) {
				continue;
			}
		}

		Vector3 local_pos;
		Vector3 world_normal(0, 1, 0);

		if (project_on_collision) {
			if (!_project_point(Vector3(lx, half.y, lz), local_pos, world_normal)) {
				continue;
			}
		} else {
			const float ly = rng.random(-half.y, half.y);
			local_pos = Vector3(lx, ly, lz);
		}

		Vector3 up_target(0, 1, 0);
		if (align_to_normal) {
			up_target = Vector3(0, 1, 0).lerp(world_normal, align_to_normal_amount);
			if (up_target.length_squared() < 0.0001f) {
				up_target = Vector3(0, 1, 0);
			} else {
				up_target.normalize();
			}
		}

		if (random_tilt_degrees > 0.0f) {
			Vector3 jitter_axis(rng.random(-1.0f, 1.0f), 0.0f, rng.random(-1.0f, 1.0f));
			if (jitter_axis.length_squared() > 0.0001f) {
				jitter_axis.normalize();
				const float jitter_angle = Math::deg_to_rad(rng.random(0.0f, random_tilt_degrees));
				up_target = up_target.rotated(jitter_axis, jitter_angle).normalized();
			}
		}

		const float yaw = random_rotation ? rng.random(0.0f, (float)Math::TAU) : 0.0f;
		Basis basis(Vector3(0, 1, 0), yaw);
		basis.rotate_to_align(Vector3(0, 1, 0), up_target);

		const float lo_scale = MIN(min_scale, max_scale);
		const float hi_scale = MAX(min_scale, max_scale);
		const float s = rng.random(lo_scale, hi_scale);
		basis = basis.scaled_local(Vector3(s, s, s));

		transforms.push_back(Transform3D(gt_inv.basis * basis, local_pos));

		if (min_distance > 0.0f) {
			grid[cell].push_back(Vector2(lx, lz));
		}
	}

	Ref<MultiMesh> mm;
	mm.instantiate();
	mm->set_transform_format(MultiMesh::TRANSFORM_3D);
	mm->set_mesh(mesh);
	mm->set_instance_count(transforms.size());
	for (uint32_t i = 0; i < transforms.size(); i++) {
		mm->set_instance_transform(i, transforms[i]);
	}

	set_multimesh(mm);
	update_gizmos();
	update_configuration_warnings();
}

AABB FoliageSpawner3D::get_aabb() const {
	const Vector3 sz = volume_size.abs();
	AABB box(sz * -0.5f, sz);
	box.merge_with(MultiMeshInstance3D::get_aabb());
	return box;
}

PackedStringArray FoliageSpawner3D::get_configuration_warnings() const {
	PackedStringArray warnings = MultiMeshInstance3D::get_configuration_warnings();

	if (mesh.is_null()) {
		warnings.push_back(RTR("No Mesh assigned. Set a Mesh and press Regenerate to scatter instances."));
	} else {
		Ref<MultiMesh> mm = get_multimesh();
		if (mm.is_null() || mm->get_instance_count() == 0) {
			warnings.push_back(RTR("No instances have been generated yet (or none matched the current settings). Press Regenerate after adjusting Density, Min Distance, the Distribution Mask, or the Collision Mask."));
		}
	}

	if (volume_size.x <= 0.0f || volume_size.y <= 0.0f || volume_size.z <= 0.0f) {
		warnings.push_back(RTR("Volume Size must be greater than zero on every axis."));
	}

	return warnings;
}

FoliageSpawner3D::FoliageSpawner3D() {
}
