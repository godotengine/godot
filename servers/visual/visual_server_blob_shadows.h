/**************************************************************************/
/*  visual_server_blob_shadows.h                                          */
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

#ifndef VISUAL_SERVER_BLOB_SHADOWS_H
#define VISUAL_SERVER_BLOB_SHADOWS_H

#include "core/fixed_array.h"
#include "core/math/bvh.h"
#include "core/math/camera_matrix.h"
#include "core/math/vector3.h"
#include "core/pooled_list.h"

class VisualServerBlobShadows {
	enum {
		MAX_CASTERS = 128, // blobs and capsules
	};
	enum InstanceType {
		IT_BLOB,
		IT_CAPSULE,
		IT_LIGHT,
	};

	struct Instance {
		InstanceType type;
		uint32_t handle;
		void clear() {
			type = IT_BLOB;
			handle = 0;
		}
	};

	// Blobs can be switched off either in project setting,
	// or if there are no blob occuders or blob lights.
	static bool _active_project_setting;
	static bool _active_blobs_present;
	static bool _active;

	void _refresh_active();

public:
	static bool is_active() { return _active; }
	static bool is_allowed() { return _active_project_setting; }

	real_t get_range() const { return data.range; }
	real_t get_gamma() const { return data.gamma; }
	real_t get_intensity() const { return data.intensity; }

	void set_range(real_t p_value) { data.range = p_value; }
	void set_gamma(real_t p_value) { data.gamma = p_value; }
	void set_intensity(real_t p_value) { data.intensity = p_value; }

	enum LightType {
		SPOT,
		OMNI,
		DIRECTIONAL,
	};

	struct Light {
		uint32_t instance_handle;
		BVHHandle bvh_handle;
		Vector3 pos;
		LightType type;
		bool visible;
		real_t energy;
		AABB aabb;

		Vector3 direction;
		real_t range_max;
		real_t range_mid;
		real_t range_mid_max;
		real_t intensity;
		real_t energy_intensity;

		// Precalculated spotlight params.
		real_t spot_degrees;
		real_t spot_dot_threshold;
		real_t spot_dot_multiplier;
		void set_spot_degrees(real_t p_degrees) {
			spot_degrees = MAX(p_degrees, (real_t)1); // Prevent divide by zero.
			spot_dot_threshold = Math::cos(Math::deg2rad(spot_degrees));
			spot_dot_multiplier = (real_t)1 / (1 - spot_dot_threshold);
		}
		void calculate_energy_intensity() {
			real_t e = Math::pow(energy, (real_t)0.5);
			energy_intensity = e * intensity;
		}

		const char *get_type_string() const {
			switch (type) {
				case SPOT: {
					return "SPOT";
				} break;

				case DIRECTIONAL: {
					return "DIR";
				} break;
				default:
					break;
			}

			return "OMNI";
		}
		void clear() {
			instance_handle = 0;
			bvh_handle.set(0);
			pos = Vector3();
			set_spot_degrees(45);
			direction = Vector3();
			range_max = 10;
			range_mid = 9;
			range_mid_max = 1;
			type = OMNI;
			visible = true;
			energy = 1;
			energy_intensity = 1;
			intensity = 1;
			aabb = AABB();
		}
		Light() {
			clear();
		}
		~Light() {
		}
	};

	struct Caster {
		// Casters are ref counted so that they can have
		// delayed release as they slowly fade out from focus,
		// instead of instantaneous.
		// (Looks better!)
		uint32_t ref_count;
		uint32_t instance_handle;
		BVHHandle bvh_handle;

		Vector3 pos;
		real_t size;

		// "pos" is the center for spheres,
		// but for capsules it is between pos and pos_b.
		// We should use the center for light distance calculations.
		Vector3 pos_center;

		AABB aabb;
		AABB aabb_boosted;
		LocalVector<uint32_t> *linked_lights;

		void clear() {
			pos = Vector3();
			size = 1;
			ref_count = 0;
			instance_handle = 0;
			bvh_handle.set(0);
			aabb = AABB();
			aabb_boosted = AABB();
			linked_lights->clear();
		}
		Caster() {
			linked_lights = memnew(LocalVector<uint32_t>);
			clear();
		}
		~Caster() {
			if (linked_lights) {
				memdelete(linked_lights);
				linked_lights = nullptr;
			}
		}
	};

	struct Blob : public Caster {
	};

	struct Capsule : public Caster {
		Vector3 pos_b;
		real_t size_b;
		void clear() {
			pos_b = Vector3();
			size_b = 1;
			Caster::clear();
		}

		Capsule() {
			size_b = 1;
		}
	};

	struct SortFocusCaster {
		uint32_t caster_id = 0;
		real_t distance = 0;
	};

	// A caster that is in the current focus.
	// Keeping track of how faded it it is, etc.
	struct FocusCaster {
		enum State {
			FCS_ON,
			FCS_ENTERING,
			FCS_EXITING,
		} state = FCS_ENTERING;

		uint32_t caster_id = 0;

		uint32_t in_count = 0; // The fraction in is in the in_count / transition_ticks
		uint32_t last_update_frame = UINT32_MAX;

		real_t fraction = 0;

		// We only need to store lighting data for casters that are IN FOCUS.
		// Casters that aren't in focus, storing lighting data would be a waste.
		real_t cone_degrees = 45;
		Vector3 light_pos;
		Vector3 light_dir;
		real_t light_modulate = 1;

		// If no lights are within range, a focus caster is not active.
		bool active = false;
	};

	struct FocusInfo {
		FixedArray<FocusCaster, MAX_CASTERS> blobs;
		FixedArray<FocusCaster, MAX_CASTERS> blobs_pending;
		uint8_t blobs_in_camera[MAX_CASTERS] = {};

		void clear() {
			blobs.resize(0);
			blobs_pending.resize(0);
		}
	};

	struct Focus {
		Vector3 pos;
		uint32_t last_updated_frame = UINT32_MAX;

		FocusInfo blobs;
		FocusInfo capsules;

		void clear() {
			pos = Vector3();
			blobs.clear();
			capsules.clear();
		}
	};

	// Lights
	Light *request_light(uint32_t &r_handle);
	void delete_light(uint32_t p_handle);
	Light &get_light(uint32_t p_handle) { return data.lights[--p_handle]; }
	const Light &get_light(uint32_t p_handle) const { return data.lights[--p_handle]; }
	void set_light_visible(uint32_t p_handle, bool p_visible);
	void make_light_dirty(Light &p_light);

	// Blobs
	Blob *request_blob(uint32_t &r_handle);
	void delete_blob(uint32_t p_handle);
	Blob &get_blob(uint32_t p_handle) { return data.blobs[--p_handle]; }
	const Blob &get_blob(uint32_t p_handle) const { return data.blobs[--p_handle]; }
	void make_blob_dirty(Blob &p_caster);

	// Capsules
	Capsule *request_capsule(uint32_t &r_handle);
	void delete_capsule(uint32_t p_handle);
	Capsule &get_capsule(uint32_t p_handle) { return data.capsules[--p_handle]; }
	const Capsule &get_capsule(uint32_t p_handle) const { return data.capsules[--p_handle]; }
	void make_capsule_dirty(Capsule &p_caster);

	// Focus
	Focus *request_focus(uint32_t &r_handle);
	void delete_focus(uint32_t p_handle);
	Focus &get_focus(uint32_t p_handle) { return data.foci[--p_handle]; }

	// Note that data for the shader is 32 bit, even if real_t calculations done as 64 bit.
	uint32_t fill_background_uniforms_blobs(const AABB &p_aabb, float *r_casters, float *r_lights, uint32_t p_max_casters);
	uint32_t fill_background_uniforms_capsules(const AABB &p_aabb, float *r_casters, float *r_lights, uint32_t p_max_casters);

	void render_set_focus_handle(uint32_t p_focus_handle, const Vector3 &p_pos, const Transform &p_cam_transform, const CameraMatrix &p_cam_matrix);
	void update();

private:
	static int qsort_cmp_func(const void *a, const void *b);

	void ref_blob(uint32_t p_handle) {
		Blob &caster = get_blob(p_handle);
		caster.ref_count++;
	}

	void unref_blob(uint32_t p_handle) {
		Blob &caster = get_blob(p_handle);
		DEV_ASSERT(caster.ref_count);
		caster.ref_count--;

		if (!caster.ref_count) {
			data.blobs.free(--p_handle);
			// print_line("releasing blob " + itos(p_handle));
		}
	}

	void ref_capsule(uint32_t p_handle) {
		Capsule &caster = get_capsule(p_handle);
		caster.ref_count++;
	}

	void unref_capsule(uint32_t p_handle) {
		Capsule &caster = get_capsule(p_handle);
		DEV_ASSERT(caster.ref_count);
		caster.ref_count--;

		if (!caster.ref_count) {
			data.capsules.free(--p_handle);
			// print_line("releasing capsule " + itos(p_handle));
		}
	}

	template <class T, bool BLOBS_OR_CAPSULES>
	void update_focus_blobs_or_capsules(const Focus &p_focus, FocusInfo &r_focus_info, const TrackedPooledList<T> &p_blobs, uint32_t p_max_casters);

	void update_focus(Focus &r_focus);
	void update_focus_caster(bool p_blobs_or_capsules, FocusInfo &r_focus_info, const SortFocusCaster &p_sort_focus_caster);
	void find_best_light(bool p_blobs_or_capsules, FocusCaster &r_focus_caster);

	// note this is actually the BVH id +1, so that visual server can test against zero
	// for validity to maintain compatibility with octree (where 0 indicates invalid)
	typedef uint32_t SpatialPartitionID;

	static void *_instance_pair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int);
	static void _instance_unpair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int, void *);

	struct Data {
		TrackedPooledList<Light> lights;
		TrackedPooledList<Focus> foci;
		TrackedPooledList<Blob> blobs;
		TrackedPooledList<Capsule> capsules;
		PooledList<Instance> instances;
		BVH_Manager<void, 2, true, 32> bvh;
		uint32_t blob_max_casters = MAX_CASTERS;
		uint32_t capsule_max_casters = MAX_CASTERS;

		real_t range = 6.0f;
		real_t gamma = 1.0f;
		real_t intensity = 1.0f;

		uint32_t update_frame = 0;
		uint32_t render_focus_handle = 0;

		// Camera frustum planes (world space).
		LocalVector<Plane> frustum_planes;
		LocalVector<Vector3> frustum_points;

		bool debug_output = false;
	} data;

public:
	VisualServerBlobShadows();
};

#endif // VISUAL_SERVER_BLOB_SHADOWS_H
