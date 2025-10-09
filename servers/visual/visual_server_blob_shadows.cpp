/**************************************************************************/
/*  visual_server_blob_shadows.cpp                                        */
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

#include "visual_server_blob_shadows.h"

#include "core/engine.h"
#include "core/math/plane.h"
#include "core/project_settings.h"

#include <string.h>

// #define GODOT_BLOB_SHADOWS_TEST_DIRECTION_INTERPOLATION

bool VisualServerBlobShadows::_active_project_setting = true;
bool VisualServerBlobShadows::_active_blobs_present = false;
bool VisualServerBlobShadows::_active = true;

void VisualServerBlobShadows::update() {
	if (ProjectSettings::get_singleton()->has_changes()) {
		// Only change these via project setting in the editor.
		// In game use VisualServer API as it is more efficient and won't cause stalls.
		if (Engine::get_singleton()->is_editor_hint()) {
			data.range = GLOBAL_GET("rendering/quality/blob_shadows/range");
			data.gamma = GLOBAL_GET("rendering/quality/blob_shadows/gamma");
			data.intensity = GLOBAL_GET("rendering/quality/blob_shadows/intensity");
		}
		_active_project_setting = GLOBAL_GET("rendering/quality/blob_shadows/enable");
		_refresh_active();
	}

	if (is_active()) {
		data.bvh.update();
	}
}

VisualServerBlobShadows::Light *VisualServerBlobShadows::request_light(uint32_t &r_handle) {
	Light *light = data.lights.request(r_handle);
	light->clear();

	// Associate an instance with this light.
	Instance *instance = data.instances.request(light->instance_handle);
	instance->clear();
	instance->handle = r_handle;
	instance->type = IT_LIGHT;

	// Lights only pair with casters.
	light->bvh_handle = data.bvh.create((void *)(uintptr_t)light->instance_handle, true, 0, 2);

	r_handle++;
	_refresh_active();
	return light;
}

void VisualServerBlobShadows::delete_light(uint32_t p_handle) {
	const Light &light = get_light(p_handle);

	data.bvh.erase(light.bvh_handle);

	// Free instance associated with this light.
	data.instances.free(light.instance_handle);

	data.lights.free(--p_handle);
	_refresh_active();
}

void VisualServerBlobShadows::set_light_visible(uint32_t p_handle, bool p_visible) {
	Light &light = get_light(p_handle);
	light.visible = p_visible;
	make_light_dirty(light);
	_refresh_active();
}

void VisualServerBlobShadows::make_light_dirty(Light &p_light) {
	// Just immediate update for now.
	AABB &aabb = p_light.aabb;
	aabb.size = Vector3();

	switch (p_light.type) {
		case VisualServerBlobShadows::OMNI: {
			aabb.position = p_light.pos;
			aabb.grow_by(p_light.range_max);
		} break;
		case VisualServerBlobShadows::SPOT: {
			if (p_light.direction.length_squared() < 0.001f) {
				return;
			}
#define SPOT_SIMPLE_BOUND
#ifdef SPOT_SIMPLE_BOUND
			aabb.position = p_light.pos;
			aabb.grow_by(p_light.range_max);

#else
			// Pointing in -Z direction (look_at convention)
			Vector3 corn_a = Vector3(p_light.range_max, p_light.range_max, 0);
			Vector3 corn_b = Vector3(-p_light.range_max, -p_light.range_max, -p_light.range_max);
			aabb.position = corn_a;
			aabb.expand_to(corn_b);

			// Test
			//aabb = AABB(Vector3(), Vector3(1, 1, -10));

			// Rotate.
			Transform tr;
			tr.set_look_at(Vector3(), p_light.direction, Vector3(0, 1, 0));

			aabb = tr.xform(aabb);

			// Shift
			aabb.position += p_light.pos;
#endif
		} break;
		case VisualServerBlobShadows::DIRECTIONAL: {
			const real_t cfmax = FLT_MAX / 4.0;
			Vector3 corn_a = Vector3(-cfmax, p_light.pos.y, -cfmax);
			Vector3 corn_b = Vector3(cfmax, p_light.pos.y - p_light.range_max, cfmax);
			aabb.position = corn_a;
			aabb.expand_to(corn_b);
		} break;
	}

	data.bvh.move(p_light.bvh_handle, aabb);
}

VisualServerBlobShadows::Blob *VisualServerBlobShadows::request_blob(uint32_t &r_handle) {
	Blob *caster = data.blobs.request(r_handle);
	caster->clear();
	caster->ref_count = 1;

	// Associate an instance with this caster.
	Instance *instance = data.instances.request(caster->instance_handle);
	instance->clear();
	instance->handle = r_handle;
	instance->type = IT_BLOB;

	// Casters only pair with lights.
	caster->bvh_handle = data.bvh.create((void *)(uintptr_t)caster->instance_handle, true, 1, 1);

	r_handle++;

	_refresh_active();
	return caster;
}

void VisualServerBlobShadows::delete_blob(uint32_t p_handle) {
	const Caster &caster = get_blob(p_handle);

	data.bvh.erase(caster.bvh_handle);

	// Free instance associated with this caster.
	data.instances.free(caster.instance_handle);

	// Deferred free of the caster so it can fade,
	// if it has a ref count.
	unref_blob(p_handle);
	_refresh_active();
}

void VisualServerBlobShadows::make_blob_dirty(Blob &p_caster) {
	// Just immediate update for now.
	AABB &aabb = p_caster.aabb;
	aabb.position = p_caster.pos;
	aabb.size = Vector3();
	aabb.grow_by(p_caster.size);

	data.bvh.move(p_caster.bvh_handle, aabb);
}

VisualServerBlobShadows::Capsule *VisualServerBlobShadows::request_capsule(uint32_t &r_handle) {
	Capsule *caster = data.capsules.request(r_handle);
	caster->clear();
	caster->ref_count = 1;

	// Associate an instance with this caster.
	Instance *instance = data.instances.request(caster->instance_handle);
	instance->clear();
	instance->handle = r_handle;
	instance->type = IT_CAPSULE;

	// Casters only pair with lights.
	caster->bvh_handle = data.bvh.create((void *)(uintptr_t)caster->instance_handle, true, 1, 1);

	r_handle++;

	_refresh_active();
	return caster;
}

void VisualServerBlobShadows::delete_capsule(uint32_t p_handle) {
	const Capsule &caster = get_capsule(p_handle);

	data.bvh.erase(caster.bvh_handle);

	// Free instance associated with this caster.
	data.instances.free(caster.instance_handle);

	// Deferred free of the caster so it can fade,
	// if it has a ref count.
	unref_capsule(p_handle);
	_refresh_active();
}

void VisualServerBlobShadows::make_capsule_dirty(Capsule &p_caster) {
	// Just immediate update for now.
	AABB &aabb = p_caster.aabb;
	aabb.position = p_caster.pos;
	aabb.size = Vector3();
	aabb.grow_by(p_caster.size);

	AABB aabb2;
	aabb2.position = p_caster.pos_b;
	aabb2.grow_by(p_caster.size_b);
	aabb.merge_with(aabb2);

	data.bvh.move(p_caster.bvh_handle, aabb);
}

int VisualServerBlobShadows::qsort_cmp_func(const void *a, const void *b) {
	const SortFocusCaster *pa = (const SortFocusCaster *)a;
	const SortFocusCaster *pb = (const SortFocusCaster *)b;

	return (pa->distance > pb->distance);
}

VisualServerBlobShadows::Focus *VisualServerBlobShadows::request_focus(uint32_t &r_handle) {
	Focus *focus = data.foci.request(r_handle);
	focus->clear();
	r_handle++;
	return focus;
}

void VisualServerBlobShadows::delete_focus(uint32_t p_handle) {
	data.foci.free(--p_handle);
}

template <class T, bool BLOBS_OR_CAPSULES>
void VisualServerBlobShadows::update_focus_blobs_or_capsules(const Focus &p_focus, FocusInfo &r_focus_info, const TrackedPooledList<T> &p_blobs, uint32_t p_max_casters) {
	// This is incredibly naive going through each caster, calculating offsets and sorting.
	// It should work fine up to 1000 or so casters, but can probably be optimized to use BVH
	// or quick reject.
	uint32_t sortcount = p_blobs.active_size();
	SortFocusCaster *sortlist = (SortFocusCaster *)alloca(sortcount * sizeof(SortFocusCaster));
	memset((void *)sortlist, 0, sortcount * sizeof(SortFocusCaster));

	constexpr bool blobs_or_capsules = BLOBS_OR_CAPSULES;

	uint32_t validcount = 0;
	for (uint32_t n = 0; n < p_blobs.active_size(); n++) {
		const T &caster = p_blobs.get_active(n);
		if (!caster.linked_lights->size()) {
			sortlist[n].distance = FLT_MAX;
			sortlist[n].caster_id = UINT32_MAX;
			continue;
		}

		sortlist[n].distance = (caster.pos_center - p_focus.pos).length_squared();
		sortlist[n].caster_id = blobs_or_capsules ? data.blobs.get_active_id(n) : data.capsules.get_active_id(n);
		validcount++;
	}

	qsort((void *)sortlist, sortcount, sizeof(SortFocusCaster), qsort_cmp_func);
	uint32_t num_closest = MIN(p_max_casters, validcount);

	for (uint32_t n = 0; n < num_closest; n++) {
		update_focus_caster(blobs_or_capsules, r_focus_info, sortlist[n]);
	}

	// Update existing focus casters and pending
	FocusInfo &fi = r_focus_info;

	for (uint32_t n = 0; n < fi.blobs.size(); n++) {
		FocusCaster &fc = fi.blobs[n];
		if (fc.last_update_frame != data.update_frame) {
			fc.state = FocusCaster::FCS_EXITING;
		}

		switch (fc.state) {
			case FocusCaster::FCS_ON: {
				fc.fraction = 1.0f;
			} break;
			case FocusCaster::FCS_ENTERING: {
				fc.in_count++;
				fc.fraction = (real_t)fc.in_count / 60;

				// Fully faded in, change to on
				if (fc.in_count >= 60) {
					fc.state = FocusCaster::FCS_ON;
					fc.in_count = 60;
				}

			} break;
			case FocusCaster::FCS_EXITING: {
				// unexit?
				if (fc.last_update_frame == data.update_frame) {
					fc.state = FocusCaster::FCS_ENTERING;
				} else {
					fc.in_count--;
					fc.fraction = (real_t)fc.in_count / 60;
					if (!fc.in_count) {
						// Decrement ref count so we can free
						// when no more fades.
						if (blobs_or_capsules) {
							unref_blob(fc.caster_id + 1);
						} else {
							unref_capsule(fc.caster_id + 1);
						}

						fi.blobs.remove_unordered(n);

						// repeat this element
						n--;
					}
				}

			} break;
		}
	}

	// If the pending is still not being actively requested, remove it
	for (uint32_t n = 0; n < fi.blobs_pending.size(); n++) {
		FocusCaster &fc = fi.blobs_pending[n];
		if (fc.last_update_frame != data.update_frame) {
			fi.blobs_pending.remove_unordered(n);
			n--;
		}
	}

	// Finally add any pending if there is room
	uint32_t max_casters = blobs_or_capsules ? data.blob_max_casters : data.capsule_max_casters;

	while ((fi.blobs.size() < max_casters) && fi.blobs_pending.size()) {
		// When on the focus, we add a reference to
		// prevent deletion.
		FocusCaster &fc = fi.blobs_pending.last();

		if (blobs_or_capsules) {
			ref_blob(fc.caster_id + 1);
		} else {
			ref_capsule(fc.caster_id + 1);
		}

		fi.blobs.push_back(fc);
		fi.blobs_pending.pop();

		fc.state = FocusCaster::FCS_ENTERING;
		fc.in_count = 1;
		fc.fraction = (real_t)fc.in_count / 60;
	}
}

void VisualServerBlobShadows::update_focus(Focus &r_focus) {
	update_focus_blobs_or_capsules<Blob, true>(r_focus, r_focus.blobs, data.blobs, data.blob_max_casters);
	update_focus_blobs_or_capsules<Capsule, false>(r_focus, r_focus.capsules, data.capsules, data.capsule_max_casters);
}

void VisualServerBlobShadows::update_focus_caster(bool p_blobs_or_capsules, FocusInfo &r_focus_info, const SortFocusCaster &p_sort_focus_caster) {
	FocusInfo &fi = r_focus_info;

	// Does the focus caster exist already?
	for (uint32_t n = 0; n < fi.blobs.size(); n++) {
		FocusCaster &fc = fi.blobs[n];
		if (fc.caster_id == p_sort_focus_caster.caster_id) {
			fc.last_update_frame = data.update_frame;
			find_best_light(p_blobs_or_capsules, fc);
			return;
		}
	}

	// if we got to here, not existing, add to pending list
	for (uint32_t n = 0; n < fi.blobs_pending.size(); n++) {
		FocusCaster &fc = fi.blobs_pending[n];
		if (fc.caster_id == p_sort_focus_caster.caster_id) {
			fc.last_update_frame = data.update_frame;
		}
	}

	if (fi.blobs_pending.is_full()) {
		return;
	}

	// Add to pending
	fi.blobs_pending.resize(fi.blobs_pending.size() + 1);
	FocusCaster &fc = fi.blobs_pending.last();
	fc.caster_id = p_sort_focus_caster.caster_id;
	fc.last_update_frame = data.update_frame;
}

#ifdef GODOT_BLOB_SHADOWS_TEST_DIRECTION_INTERPOLATION
Vector3 choose_random_dir() {
	Vector3 d = Vector3(Math::randf(), Math::randf(), Math::randf());
	d *= 2.0f;
	d -= Vector3(1, 1, 1);
	d.normalize();
	return d;
}

void test_direction_interpolation() {
	const int MAX_DIRS = 3;
	Vector3 dirs[MAX_DIRS];
	float weights[MAX_DIRS];
	Vector3 orig_dirs[MAX_DIRS];
	float orig_weights[MAX_DIRS];
	for (int run = 0; run < 10; run++) {
		float total_weight = 0.0f;
		for (int i = 0; i < MAX_DIRS; i++) {
			orig_dirs[i] = choose_random_dir();
			orig_weights[i] = Math::randf();
			total_weight += orig_weights[i];
		}
		for (int i = 0; i < MAX_DIRS; i++) {
			orig_weights[i] /= total_weight;
		}

		for (int i = 0; i < MAX_DIRS; i++) {
			dirs[i] = orig_dirs[i];
			weights[i] = orig_weights[i];
		}

		for (int n = MAX_DIRS - 1; n >= 1; n--) {
			float w0 = weights[n - 1];
			float w1 = weights[n];
			float fraction = w0 / (w0 + w1);
			Vector3 new_dir = dirs[n - 1].slerp(dirs[n], fraction);
			new_dir.normalize();
			dirs[n - 1] = new_dir;
			weights[n - 1] = w0 + w1;
		}

		print_line("final result : " + String(Variant(dirs[0])));

		for (int i = 0; i < MAX_DIRS; i++) {
			dirs[i] = orig_dirs[MAX_DIRS - i - 1];
			weights[i] = orig_weights[MAX_DIRS - i - 1];
		}

		for (int n = MAX_DIRS - 1; n >= 1; n--) {
			float w0 = weights[n - 1];
			float w1 = weights[n];
			float fraction = w0 / (w0 + w1);
			Vector3 new_dir = dirs[n - 1].slerp(dirs[n], fraction);
			new_dir.normalize();
			dirs[n - 1] = new_dir;
			weights[n - 1] = w0 + w1;
		}

		print_line("final result2 : " + String(Variant(dirs[0])));
		print_line("\n");
	}
}
#endif

void VisualServerBlobShadows::find_best_light(bool p_blobs_or_capsules, FocusCaster &r_focus_caster) {
#ifdef GODOT_BLOB_SHADOWS_TEST_DIRECTION_INTERPOLATION
	test_direction_interpolation();
#endif
	Caster *caster = nullptr;
	if (p_blobs_or_capsules) {
		Blob &blob = data.blobs[r_focus_caster.caster_id];
		caster = &blob;
	} else {
		Capsule &capsule = data.capsules[r_focus_caster.caster_id];
		caster = &capsule;
	}

	// first find relative weights of lights
	real_t total_weight = 0;

	struct LightCalc {
		Vector3 pos;
		real_t dist;
		real_t intensity;
		real_t weight;
	};

	LightCalc *lights = (LightCalc *)alloca(sizeof(LightCalc) * caster->linked_lights->size());
	memset((void *)lights, 0, sizeof(LightCalc) * caster->linked_lights->size());
	uint32_t num_lights = 0;

	r_focus_caster.light_modulate = 0;

	if (data.debug_output) {
		print_line("num linked lights for caster " + itos(caster->instance_handle) + " : " + itos(caster->linked_lights->size()) + ", caster pos " + String(Variant(caster->pos)));
	}

	r_focus_caster.active = false;

	for (uint32_t n = 0; n < caster->linked_lights->size(); n++) {
		uint32_t light_handle = (*caster->linked_lights)[n];
		const Light &light = data.lights[light_handle];

		if (!light.visible) {
			continue;
		}

		LightCalc &lc = lights[num_lights++];

		Vector3 offset_light_caster;
		if (light.type != DIRECTIONAL) {
			offset_light_caster = light.pos - caster->pos_center;
			lc.dist = offset_light_caster.length();
		} else {
			lc.dist = Math::abs(light.pos.y - caster->pos_center.y);
		}
		lc.intensity = light.energy_intensity;

		if (lc.dist >= light.range_max) // out of range
		{
			lc.dist = 0;
			num_lights--;
			continue;
		}

		if (Math::is_zero_approx(light.energy)) {
			num_lights--;
			continue;
		}

		// Total weight is scaled impact of each light, from 0 to 1.
		lc.weight = (light.range_max - lc.dist) / light.range_max;
		lc.weight *= light.energy;

		// Apply cone on spot
		if (light.type == SPOT) {
			real_t dot = -offset_light_caster.normalized().dot(light.direction);

			dot -= light.spot_dot_threshold;
			dot *= light.spot_dot_multiplier;

			if (dot <= 0) {
				lc.dist = 0;
				num_lights--;
				continue;
			}

			lc.weight *= dot;
			lc.intensity *= dot;
		}

		total_weight += lc.weight;

		if (data.debug_output) {
			print_line("(" + itos(n) + ") light handle " + itos(light_handle) + " weight " + String(Variant(lc.weight)) + ", dist " + String(Variant(lc.dist)) + " over max " + String(Variant(light.range_max)));
			print_line("\tLight AABB " + String(Variant(light.aabb)) + " ... type " + light.get_type_string());
		}

		// Modify distance to take into account fade.
		real_t dist_left = lc.dist;
		if (dist_left > light.range_mid) {
			dist_left -= light.range_mid;

			// Scale 0 to 1
			dist_left /= light.range_mid_max;
			real_t fade = 1.0f - dist_left;

			r_focus_caster.light_modulate = MAX(r_focus_caster.light_modulate, fade);
		} else {
			r_focus_caster.light_modulate = 1;
		}

		if (light.type != DIRECTIONAL) {
			lc.pos = light.pos;
		} else {
			lc.pos = caster->pos_center - (light.direction * 10000);
		}

		r_focus_caster.active = true;
	}

	// No lights affect this caster, do no more work.
	if (!r_focus_caster.active) {
		return;
	}

	r_focus_caster.light_pos = Vector3();
	real_t intensity = 0;

	// prevent divide by zero
	total_weight = MAX(total_weight, (real_t)0.0000001);

	// second pass
	for (uint32_t n = 0; n < num_lights; n++) {
		LightCalc &lc = lights[n];

		// scale weight by total weight
		lc.weight /= total_weight;

		r_focus_caster.light_pos += lc.pos * lc.weight;
		intensity += lc.intensity * lc.weight;
	}

	r_focus_caster.light_modulate *= intensity;
	r_focus_caster.light_modulate *= data.intensity;

	// Precalculate light direction so we don't have to do per pixel in the shader.
	Vector3 light_dir = (r_focus_caster.light_pos - caster->pos_center).normalized();

	if (p_blobs_or_capsules) {
		r_focus_caster.light_dir = light_dir;
	} else {
		r_focus_caster.light_dir = r_focus_caster.light_pos;
	}

	r_focus_caster.cone_degrees = 45.0f;

	// Calculate boosted AABB based on the light direction.
	AABB &boosted = caster->aabb_boosted;
	boosted = caster->aabb;
	Vector3 bvec = light_dir * -data.range;
	boosted.size += bvec.abs();

	// Boosted AABB may not be totally accurate in the case of capsule,
	// especially when light close to capsule.
	// ToDo: Maybe this can be improved.
	if (bvec.x < 0) {
		boosted.position.x += bvec.x;
	}
	if (bvec.y < 0) {
		boosted.position.y += bvec.y;
	}
	if (bvec.z < 0) {
		boosted.position.z += bvec.z;
	}
}

void VisualServerBlobShadows::render_set_focus_handle(uint32_t p_focus_handle, const Vector3 &p_pos, const Transform &p_cam_transform, const CameraMatrix &p_cam_matrix) {
	data.render_focus_handle = p_focus_handle;

	// Get the camera frustum planes in world space.
	if (!p_cam_matrix.get_projection_planes_and_endpoints(p_cam_transform, data.frustum_planes.ptr(), data.frustum_points.ptr())) {
		// Invalid frustum / xform.
		WARN_PRINT_ONCE("Invalid camera transform detected.");
		return;
	}

	if (p_focus_handle) {
		Focus &focus = get_focus(p_focus_handle);
		focus.pos = p_pos;

		data.update_frame = Engine::get_singleton()->get_frames_drawn();

		// Update the focus only the first time it is demanded on a frame.
		if (focus.last_updated_frame != data.update_frame) {
			focus.last_updated_frame = data.update_frame;
			update_focus(focus);
		}

		FocusInfo &blobs = focus.blobs;

		// Cull spheres to camera.
		for (uint32_t i = 0; i < blobs.blobs.size(); i++) {
			const FocusCaster &fc = blobs.blobs[i];
			const Blob &caster = get_blob(fc.caster_id + 1);

			blobs.blobs_in_camera[i] = caster.aabb_boosted.intersects_convex_shape(data.frustum_planes.ptr(), 6, data.frustum_points.ptr(), 8) ? 255 : 0;
		}

		FocusInfo &capsules = focus.capsules;

		// Cull capsules to camera.
		for (uint32_t i = 0; i < capsules.blobs.size(); i++) {
			const FocusCaster &fc = capsules.blobs[i];
			const Capsule &caster = get_capsule(fc.caster_id + 1);

			capsules.blobs_in_camera[i] = caster.aabb_boosted.intersects_convex_shape(data.frustum_planes.ptr(), 6, data.frustum_points.ptr(), 8) ? 255 : 0;
		}
	}
}

uint32_t VisualServerBlobShadows::fill_background_uniforms_blobs(const AABB &p_aabb, float *r_casters, float *r_lights, uint32_t p_max_casters) {
	DEV_ASSERT(data.render_focus_handle);

	uint32_t count = 0;

	struct LightData {
		Vector3 dir;
		real_t modulate = 1.0f;
	} ldata;

	struct CasterData {
		Vector3 pos;
		float size;
	} cdata;

	if (sizeof(real_t) == 4) {
		const Focus &focus = data.foci[data.render_focus_handle - 1];

		const FocusInfo &fi = focus.blobs;

		for (uint32_t n = 0; n < fi.blobs.size(); n++) {
			const FocusCaster &fc = fi.blobs[n];

			// Out of view.
			if (!fi.blobs_in_camera[n]) {
				continue;
			}

			ldata.modulate = fc.fraction * fc.light_modulate;

			// If the light modulate is zero, there is no light affecting this caster,
			// the direction will be unset, and we would get NaN in the shader,
			// so we avoid all this work.
			if (Math::is_zero_approx(ldata.modulate)) {
				continue;
			}

			const Blob &caster = data.blobs[fc.caster_id];

			cdata.pos = caster.pos;
			cdata.size = caster.size;

			// Does caster + shadow intersect the geometry?
			if (!p_aabb.intersects(caster.aabb_boosted)) {
				continue;
			}

			memcpy(r_casters, &cdata, sizeof(CasterData));
			r_casters += 4;

			ldata.dir = fc.light_dir;

			memcpy(r_lights, &ldata, sizeof(LightData));
			r_lights += 4;

			count++;
			if (count >= p_max_casters) {
				break;
			}
		}

	} else {
		WARN_PRINT_ONCE("blob shadows with double NYI");
	}

	return count;
}

uint32_t VisualServerBlobShadows::fill_background_uniforms_capsules(const AABB &p_aabb, float *r_casters, float *r_lights, uint32_t p_max_casters) {
	DEV_ASSERT(data.render_focus_handle);

	uint32_t count = 0;

	struct LightData {
		Vector3 dir;
		real_t modulate = 1.0f;
	} ldata;

	struct CasterData {
		Vector3 pos;
		float size;
		Vector3 pos_b;
		float size_b;
	} cdata;

	if (sizeof(real_t) == 4) {
		const Focus &focus = data.foci[data.render_focus_handle - 1];

		const FocusInfo &fi = focus.capsules;

		for (uint32_t n = 0; n < fi.blobs.size(); n++) {
			const FocusCaster &fc = fi.blobs[n];

			// Out of view.
			if (!fi.blobs_in_camera[n]) {
				continue;
			}

			ldata.modulate = fc.fraction * fc.light_modulate;

			// If the light modulate is zero, there is no light affecting this caster,
			// the direction will be unset, and we would get NaN in the shader,
			// so we avoid all this work.
			if (!fc.active) {
				continue;
			}

			const Capsule &caster = data.capsules[fc.caster_id];

			cdata.pos = caster.pos;
			cdata.size = caster.size;
			cdata.pos_b = caster.pos_b;
			cdata.size_b = caster.size_b;

			// Does caster + shadow intersect the geometry?
			if (!p_aabb.intersects(caster.aabb_boosted)) {
				continue;
			}

			memcpy(r_casters, &cdata, sizeof(CasterData));
			r_casters += 8;

			ldata.dir = fc.light_dir;

			memcpy(r_lights, &ldata, sizeof(LightData));
			r_lights += 4;

			count++;
			if (count >= p_max_casters) {
				break;
			}
		}

	} else {
		WARN_PRINT_ONCE("capsule shadows with double NYI");
	}

	return count;
}

void *VisualServerBlobShadows::_instance_pair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int) {
	uint32_t handle_a = (uint32_t)(uint64_t)p_A;
	uint32_t handle_b = (uint32_t)(uint64_t)p_B;

	VisualServerBlobShadows *self = static_cast<VisualServerBlobShadows *>(p_self);

	Instance &a = self->data.instances[handle_a];
	Instance &b = self->data.instances[handle_b];

	uint32_t caster_handle = 0;
	uint32_t light_handle = 0;

	Caster *caster = nullptr;

	if (a.type == VisualServerBlobShadows::IT_LIGHT) {
		DEV_ASSERT((b.type == VisualServerBlobShadows::IT_BLOB) || (b.type == VisualServerBlobShadows::IT_CAPSULE));
		light_handle = a.handle;
		caster_handle = b.handle;
		caster = b.type == VisualServerBlobShadows::IT_BLOB ? (Caster *)&self->data.blobs[caster_handle] : (Caster *)&self->data.capsules[caster_handle];
	} else {
		DEV_ASSERT((a.type == VisualServerBlobShadows::IT_BLOB) || (a.type == VisualServerBlobShadows::IT_CAPSULE));
		DEV_ASSERT(b.type == VisualServerBlobShadows::IT_LIGHT);
		light_handle = b.handle;
		caster_handle = a.handle;
		caster = a.type == VisualServerBlobShadows::IT_BLOB ? (Caster *)&self->data.blobs[caster_handle] : (Caster *)&self->data.capsules[caster_handle];
	}

	DEV_ASSERT(caster->linked_lights);
	DEV_ASSERT(caster->linked_lights->find(light_handle) == -1);
	caster->linked_lights->push_back(light_handle);

	return nullptr;
}

void VisualServerBlobShadows::_instance_unpair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int, void *udata) {
	uint32_t handle_a = (uint32_t)(uint64_t)p_A;
	uint32_t handle_b = (uint32_t)(uint64_t)p_B;

	VisualServerBlobShadows *self = static_cast<VisualServerBlobShadows *>(p_self);

	Instance &a = self->data.instances[handle_a];
	Instance &b = self->data.instances[handle_b];

	uint32_t caster_handle = 0;
	uint32_t light_handle = 0;

	Caster *caster = nullptr;

	if (a.type == VisualServerBlobShadows::IT_LIGHT) {
		DEV_ASSERT((b.type == VisualServerBlobShadows::IT_BLOB) || (b.type == VisualServerBlobShadows::IT_CAPSULE));
		light_handle = a.handle;
		caster_handle = b.handle;
		caster = b.type == VisualServerBlobShadows::IT_BLOB ? (Caster *)&self->data.blobs[caster_handle] : (Caster *)&self->data.capsules[caster_handle];
	} else {
		DEV_ASSERT((a.type == VisualServerBlobShadows::IT_BLOB) || (a.type == VisualServerBlobShadows::IT_CAPSULE));
		DEV_ASSERT(b.type == VisualServerBlobShadows::IT_LIGHT);
		light_handle = b.handle;
		caster_handle = a.handle;
		caster = a.type == VisualServerBlobShadows::IT_BLOB ? (Caster *)&self->data.blobs[caster_handle] : (Caster *)&self->data.capsules[caster_handle];
	}

	DEV_ASSERT(caster->linked_lights);
	int64_t found = caster->linked_lights->find(light_handle);
	DEV_ASSERT(found != -1);
	caster->linked_lights->remove_unordered(found);
}

void VisualServerBlobShadows::_refresh_active() {
	_active_blobs_present = (data.blobs.active_size() || data.capsules.active_size()) && data.lights.active_size();
	_active = _active_project_setting && _active_blobs_present;
}

VisualServerBlobShadows::VisualServerBlobShadows() {
	data.bvh.set_pair_callback(_instance_pair, this);
	data.bvh.set_unpair_callback(_instance_unpair, this);

	_active_project_setting = GLOBAL_DEF("rendering/quality/blob_shadows/enable", true);
	_refresh_active();

	data.blob_max_casters = GLOBAL_DEF_RST("rendering/quality/blob_shadows/max_spheres", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/max_spheres", PropertyInfo(Variant::INT, "rendering/quality/blob_shadows/max_spheres", PROPERTY_HINT_RANGE, "0,128,1"));
	data.capsule_max_casters = GLOBAL_DEF_RST("rendering/quality/blob_shadows/max_capsules", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/max_capsules", PropertyInfo(Variant::INT, "rendering/quality/blob_shadows/max_capsules", PROPERTY_HINT_RANGE, "0,128,1"));
	data.range = GLOBAL_DEF("rendering/quality/blob_shadows/range", 2.0f);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/range", PropertyInfo(Variant::REAL, "rendering/quality/blob_shadows/range", PROPERTY_HINT_RANGE, "0.0,256.0"));
	data.gamma = GLOBAL_DEF("rendering/quality/blob_shadows/gamma", 1.0f);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/gamma", PropertyInfo(Variant::REAL, "rendering/quality/blob_shadows/gamma", PROPERTY_HINT_RANGE, "0.01,10.0,0.01"));
	data.intensity = GLOBAL_DEF("rendering/quality/blob_shadows/intensity", 0.7f);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/intensity", PropertyInfo(Variant::REAL, "rendering/quality/blob_shadows/intensity", PROPERTY_HINT_RANGE, "0.0,6.0,0.01"));

	data.frustum_planes.resize(6);
	data.frustum_points.resize(8);
}
