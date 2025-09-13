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

#pragma once

#include "core/fixed_array.h"
#include "core/math/bvh.h"
#include "core/math/camera_matrix.h"
#include "core/math/vector3.h"
#include "core/pooled_list.h"

class VisualServerBlobShadows {
	enum {
		BLOB_MAX_CASTERS = 128
	};
	enum InstanceType {
		IT_CASTER,
		IT_LIGHT,
	};

	struct Instance {
		InstanceType type;
		uint32_t handle;
		void clear() {
			type = IT_CASTER;
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
	float get_cutoff_boost() const { return data.cutoff_boost; }

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
		AABB aabb;

		Vector3 direction;
		real_t range_max;
		real_t range_mid;
		real_t range_mid_max;
		real_t intensity;

		// Precalculated spotlight params.
		real_t spot_degrees;
		real_t spot_dot_threshold;
		real_t spot_dot_multiplier;
		void set_spot_degrees(float p_degrees) {
			spot_degrees = MAX(p_degrees, 1.0f); // Prevent divide by zero.
			spot_dot_threshold = Math::cos(Math::deg2rad(spot_degrees));
			spot_dot_multiplier = 1.0f / (1 - spot_dot_threshold);
		}

		LocalVector<uint32_t> *linked_casters;
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
			set_spot_degrees(45.0f);
			direction = Vector3();
			range_max = 10.0f;
			range_mid = 9.0f;
			range_mid_max = 1.0f;
			intensity = 1.0f;
			type = OMNI;
			visible = true;
			aabb = AABB();
			linked_casters->clear();
		}
		Light() {
			linked_casters = memnew(LocalVector<uint32_t>);
			clear();
		}
		~Light() {
			if (linked_casters) {
				memdelete(linked_casters);
				linked_casters = nullptr;
			}
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

		AABB aabb;
		AABB aabb_boosted;
		Vector3 pos;
		real_t size;
		LocalVector<uint32_t> *linked_lights;

		void clear() {
			ref_count = 0;
			instance_handle = 0;
			bvh_handle.set(0);
			aabb = AABB();
			aabb_boosted = AABB();
			pos = Vector3();
			size = 1.0f;
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

	struct FocusCaster {
		enum State {
			FCS_ON,
			FCS_ENTERING,
			FCS_EXITING,
		} state = FCS_ENTERING;

		uint32_t caster_id = 0;

		uint32_t in_count = 0; // The fraction in is in the in_count / transition_ticks
		uint32_t last_update_frame = UINT32_MAX;

		//uint32_t light_handle = 0;
		float fraction = 0.0f;
		float cone_degrees = 45.0f;
		Vector3 light_pos;
		Vector3 light_dir;
		float light_modulate = 1.0f;
	};

	struct SortFocusCaster {
		uint32_t caster_id = 0;
		float distance = 0.0f;
	};

	struct Focus {
		Vector3 pos;
		uint32_t last_updated_frame = UINT32_MAX;
		const static uint32_t MAX_CASTERS_LIMIT = BLOB_MAX_CASTERS;
		FixedArray<FocusCaster, MAX_CASTERS_LIMIT> casters;
		FixedArray<FocusCaster, MAX_CASTERS_LIMIT> pending;
		uint8_t casters_in_camera[MAX_CASTERS_LIMIT] = {};

		void clear() {
			pos = Vector3();
			casters.resize(0);
			pending.resize(0);
		}
	};

	// Lights
	Light *request_light(uint32_t &r_handle);
	void delete_light(uint32_t p_handle);
	Light &get_light(uint32_t p_handle) { return data.lights[--p_handle]; }
	const Light &get_light(uint32_t p_handle) const { return data.lights[--p_handle]; }
	void set_light_visible(uint32_t p_handle, bool p_visible);
	void make_light_dirty(Light &p_light);

	// Casters
	Caster *request_caster(uint32_t &r_handle);
	void delete_caster(uint32_t p_handle);
	Caster &get_caster(uint32_t p_handle) { return data.casters[--p_handle]; }
	const Caster &get_caster(uint32_t p_handle) const { return data.casters[--p_handle]; }
	void make_caster_dirty(Caster &p_caster);

	// Focus
	Focus *request_focus(uint32_t &r_handle);
	void delete_focus(uint32_t p_handle);
	Focus &get_focus(uint32_t p_handle) { return data.foci[--p_handle]; }

	uint32_t fill_background_uniforms(const AABB &p_aabb, float *r_casters, float *r_lights, uint32_t p_max_casters);

	void render_set_focus_handle(uint32_t p_focus_handle, const Vector3 &p_pos, const Transform &p_cam_transform, const CameraMatrix &p_cam_matrix);
	void update();

private:
	static int qsort_cmp_func(const void *a, const void *b);

	void ref_caster(uint32_t p_handle) {
		Caster &caster = get_caster(p_handle);
		caster.ref_count++;
	}

	void release_caster(uint32_t p_handle) {
		Caster &caster = get_caster(p_handle);
		DEV_ASSERT(caster.ref_count);
		caster.ref_count--;

		if (!caster.ref_count) {
			data.casters.free(--p_handle);
			print_line("releasing caster " + itos(p_handle));
		}
	}

	void update_focus(Focus &r_focus);
	void update_focus_caster(Focus &r_focus, const SortFocusCaster &p_sort_focus_caster);
	void find_best_light(FocusCaster &r_focus_caster);

	// note this is actually the BVH id +1, so that visual server can test against zero
	// for validity to maintain compatibility with octree (where 0 indicates invalid)
	typedef uint32_t SpatialPartitionID;

	static void *_instance_pair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int);
	static void _instance_unpair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int, void *);

	struct Data {
		TrackedPooledList<Light> lights;
		TrackedPooledList<Focus> foci;
		TrackedPooledList<Caster> casters;
		PooledList<Instance> instances;
		BVH_Manager<void, 2, true, 32> bvh;
		uint32_t max_casters = BLOB_MAX_CASTERS;
		float cutoff_boost = 6.0f;
		float global_intensity = 1.0f;
		uint32_t update_frame = 0;
		uint32_t render_focus_handle = 0;

		// Camera frustum planes (world space).
		Vector<Plane> frustum_planes;
		bool debug_output = false;
	} data;

public:
	VisualServerBlobShadows();
	~VisualServerBlobShadows();
};
