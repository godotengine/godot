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

bool VisualServerBlobShadows::_active = true;

void VisualServerBlobShadows::update() {
	if (ProjectSettings::get_singleton()->has_changes()) {
		data.cutoff_boost = GLOBAL_GET("rendering/quality/blob_shadows/cutoff_boost");
	}

	data.bvh.update();

	// Update foci.
	//	for (uint32_t n = 0; n < data.foci.active_size(); n++) {
	//		uint32_t active_id = data.foci.get_active_id(n);
	//		//print_line("Updating focus : " + itos(active_id));
	//		update_focus(data.foci[active_id]);
	//	}

	//data.update_tick++;
}

void VisualServerBlobShadows::set_light_visible(uint32_t p_handle, bool p_visible) {
	Light &light = get_light(p_handle);
	light.visible = p_visible;
	make_light_dirty(light);
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
			const float cfmax = FLT_MAX / 4.0;
			Vector3 corn_a = Vector3(-cfmax, p_light.pos.y, -cfmax);
			Vector3 corn_b = Vector3(cfmax, p_light.pos.y - p_light.range_max, cfmax);
			aabb.position = corn_a;
			aabb.expand_to(corn_b);
		} break;
	}

	data.bvh.move(p_light.bvh_handle, aabb);
}

void VisualServerBlobShadows::make_caster_dirty(Caster &p_caster) {
	// Just immediate update for now.
	AABB &aabb = p_caster.aabb;
	aabb.position = p_caster.pos;
	aabb.size = Vector3();
	aabb.grow_by(p_caster.size);

	AABB &aabb_boosted = p_caster.aabb_boosted;
	aabb_boosted = aabb;
	aabb_boosted.position.y -= data.cutoff_boost;
	aabb_boosted.size.y += data.cutoff_boost;

	data.bvh.move(p_caster.bvh_handle, aabb);
}

int VisualServerBlobShadows::qsort_cmp_func(const void *a, const void *b) {
	const SortFocusCaster *pa = (const SortFocusCaster *)a;
	const SortFocusCaster *pb = (const SortFocusCaster *)b;

	return (pa->distance > pb->distance);
}

void VisualServerBlobShadows::update_focus(Focus &r_focus) {
	uint32_t sortcount = data.casters.active_size();
	SortFocusCaster *sortlist = (SortFocusCaster *)alloca(sortcount * sizeof(SortFocusCaster));
	memset((void *)sortlist, 0, sortcount * sizeof(SortFocusCaster));

	uint32_t validcount = 0;
	for (uint32_t n = 0; n < data.casters.active_size(); n++) {
		const Caster &caster = data.casters.get_active(n);
		if (!caster.linked_lights->size()) {
			sortlist[n].distance = FLT_MAX;
			sortlist[n].caster_id = UINT32_MAX;
			continue;
		}

		Vector3 offset = caster.pos - r_focus.pos;
		real_t dist = offset.length_squared();

		sortlist[n].distance = dist;
		sortlist[n].caster_id = data.casters.get_active_id(n);
		validcount++;
	}

	qsort((void *)sortlist, sortcount, sizeof(SortFocusCaster), qsort_cmp_func);

	uint32_t num_closest = MIN(data.max_casters, validcount);

	for (uint32_t n = 0; n < num_closest; n++) {
		update_focus_caster(r_focus, sortlist[n]);
	}

	// Update existing focus casters and pending
	for (uint32_t n = 0; n < r_focus.casters.size(); n++) {
		FocusCaster &fc = r_focus.casters[n];
		if (fc.last_update_frame != data.update_frame) {
			fc.state = FocusCaster::FCS_EXITING;
		}

		switch (fc.state) {
			case FocusCaster::FCS_ON: {
				fc.fraction = 1.0f;
			} break;
			case FocusCaster::FCS_ENTERING: {
				fc.in_count++;
				fc.fraction = (float)fc.in_count / 60;

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
					fc.fraction = (float)fc.in_count / 60;
					if (!fc.in_count) {
						// Decrement ref count so we can free
						// when no more fades.
						release_caster(fc.caster_id + 1);

						r_focus.casters.remove_unordered(n);
						// repeat this element
						n--;
					}
				}

			} break;
		}
	}

	// If the pending is still not being actively requested, remove it
	for (uint32_t n = 0; n < r_focus.pending.size(); n++) {
		FocusCaster &fc = r_focus.pending[n];
		if (fc.last_update_frame != data.update_frame) {
			r_focus.pending.remove_unordered(n);
			n--;
		}
	}

	// Finally add any pending if there is room
	while ((r_focus.casters.size() < data.max_casters) && r_focus.pending.size()) {
		// When on the focus, we add a reference to
		// prevent deletion.
		FocusCaster &fc = r_focus.pending.last();
		ref_caster(fc.caster_id + 1);

		r_focus.casters.push_back(fc);
		r_focus.pending.pop();

		fc.state = FocusCaster::FCS_ENTERING;
		fc.in_count = 1;
		fc.fraction = (float)fc.in_count / 60;
	}
}

/*
void VisualServerBlobShadows::update_focus_OLD(Focus &r_focus) {
	FixedArray<SortFocusCaster, Focus::MAX_CASTERS_LIMIT> closest;

	real_t furthest_dist = FLT_MAX;

	// Find the closest e.g. 4.
	SortFocusCaster sc;

	for (uint32_t n = 0; n < data.casters.active_size(); n++) {
		const Caster &caster = data.casters.get_active(n);
		if (!caster.linked_lights->size()) {
			continue;
		}

		Vector3 offset = caster.pos - r_focus.pos;
		real_t dist = offset.length_squared();

		// We are only concerned with casters that could enter the closest list.
		if (dist <= furthest_dist) {
			// Each caster is expensive, so potentially do a more accurate cull test
			// here to check at least one light is casting shadows on this caster.
			// i.e. direction / range, rather than AABB.

			// Find the insertion position.
			uint32_t insert_pos = closest.size();
			bool found_insert_pos = false;

			for (uint32_t i = 0; i < closest.size(); i++) {
				if (dist < closest[i].distance) {
					insert_pos = i;
					found_insert_pos = true;
					break;
				}
			}

			// If we have a tie with the end of the closest list,
			// don't bother inserting.
			if (!(!found_insert_pos && (closest.size() == data.max_casters))) {
				sc.caster_handle = data.casters.get_active_id(n);
				sc.distance = dist;

				// Keep to max size which may be less than MAX_CASTERS_LIMIT.
				closest.insert(sc, insert_pos, data.max_casters);

				// New furthest.. Only set if we have filled up.
				if (closest.size() == data.max_casters) {
					furthest_dist = closest[closest.size() - 1].distance;
				}
			}
		}
	}

	// Assign light to each caster
	//	for (uint32_t n = 0; n < closest.size(); n++) {
	//		closest[n].light_handle = find_best_light(closest[n].caster_handle);
	//	}

	for (uint32_t n = 0; n < closest.size(); n++) {
		update_focus_caster(r_focus, closest[n]);
	}

	//	r_focus.num_casters = closest.size();
	//	for (uint32_t n = 0; n < r_focus.num_casters; n++) {
	//		FocusCaster &fc = r_focus.casters[n];
	//		fc.caster_handle = closest[n].caster_handle;
	//		//fc.light_handle = closest[n].light_handle;
	//		fc.fraction = 1.0f;
	//		find_best_light(fc);
	//	}

	// Update existing focus casters and pending
	for (uint32_t n = 0; n < r_focus.casters.size(); n++) {
		FocusCaster &fc = r_focus.casters[n];
		if (fc.last_update_tick != data.update_tick) {
			fc.state = FocusCaster::FCS_EXITING;

//			// When reaches zero, we remove
//			if (!fc.in_count)
//			{
//				r_focus.casters.remove_unordered(n);
//				// repeat this element
//				n--;
//				continue;
//			}
//			else
//			{
//				fc.in_count--;
//			}
		}

		switch (fc.state) {
			case FocusCaster::FCS_ON: {
				fc.fraction = 1.0f;
			} break;
			case FocusCaster::FCS_ENTERING: {
				fc.in_count++;
				fc.fraction = (float)fc.in_count / 60;

				// Fully faded in, change to on
				if (fc.in_count >= 60) {
					fc.state = FocusCaster::FCS_ON;
					fc.in_count = 60;
				}

			} break;
			case FocusCaster::FCS_EXITING: {
				// unexit?
				if (fc.last_update_tick == data.update_tick) {
					fc.state = FocusCaster::FCS_ENTERING;
				} else {
					fc.in_count--;
					fc.fraction = (float)fc.in_count / 60;
					if (!fc.in_count) {
						// Decrement ref count so we can free
						// when no more fades.
						release_caster(fc.caster_handle);

						r_focus.casters.remove_unordered(n);
						// repeat this element
						n--;
					}
				}

			} break;
		}
	}

	// If the pending is still not being actively requested, remove it
	for (uint32_t n = 0; n < r_focus.pending.size(); n++) {
		FocusCaster &fc = r_focus.pending[n];
		if (fc.last_update_tick != data.update_tick) {
			r_focus.pending.remove_unordered(n);
			n--;
		}
	}

	// Finally add any pending if there is room
	//	while ((r_focus.casters.size() < r_focus.max_casters) && r_focus.pending.size()) {
	while ((r_focus.casters.size() < data.max_casters) && r_focus.pending.size()) {
		// If the pending is still not being actively requested, remove it
		//		if (r_focus.pending.last().last_update_tick != data.update_tick) {
		//			r_focus.pending.pop();
		//		} else {
		FocusCaster &fc = r_focus.pending.last();

		// When on the focus, we add a reference to
		// prevent deletion.
		ref_caster(fc.caster_handle);

		r_focus.casters.push_back(fc);
		r_focus.pending.pop();

		//FocusCaster &fc = r_focus.casters.last();
		fc.state = FocusCaster::FCS_ENTERING;
		fc.in_count = 1;
		fc.fraction = (float)fc.in_count / 60;
		//		}
	}
}
*/

void VisualServerBlobShadows::update_focus_caster(Focus &r_focus, const SortFocusCaster &p_sort_focus_caster) {
	//	Caster &caster = data.casters[p_sort_focus_caster.caster_handle];
	//	if (caster.state == Caster::CS_PENDING)
	//	{
	//		// already pending
	//		return;
	//	}

	// Does the focus caster exist already?
	for (uint32_t n = 0; n < r_focus.casters.size(); n++) {
		FocusCaster &fc = r_focus.casters[n];
		if (fc.caster_id == p_sort_focus_caster.caster_id) {
			fc.last_update_frame = data.update_frame;
			find_best_light(fc);
			return;
		}
	}

	// if we got to here, not existing, add to pending list
	for (uint32_t n = 0; n < r_focus.pending.size(); n++) {
		FocusCaster &fc = r_focus.pending[n];
		if (fc.caster_id == p_sort_focus_caster.caster_id) {
			fc.last_update_frame = data.update_frame;
		}
	}

	if (r_focus.pending.is_full()) {
		return;
	}

	// Add to pending
	r_focus.pending.resize(r_focus.pending.size() + 1);
	FocusCaster &fc = r_focus.pending.last();
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

void VisualServerBlobShadows::find_best_light(FocusCaster &r_focus_caster) {
#ifdef GODOT_BLOB_SHADOWS_TEST_DIRECTION_INTERPOLATION
	test_direction_interpolation();
#endif
	const Caster &caster = data.casters[r_focus_caster.caster_id];

	// first find relative weights of lights
	float total_weight = 0.0f;

	struct LightCalc {
		Vector3 pos;
		float dist;
		float intensity;
		float weight;
	};

	LightCalc *lights = (LightCalc *)alloca(sizeof(LightCalc) * caster.linked_lights->size());
	memset((void *)lights, 0, sizeof(LightCalc) * caster.linked_lights->size());
	uint32_t num_lights = 0;

	r_focus_caster.light_modulate = 0.0f;

	if (data.debug_output) {
		print_line("num linked lights for caster " + itos(caster.instance_handle) + " : " + itos(caster.linked_lights->size()) + ", caster pos " + String(Variant(caster.pos)));
	}

	for (uint32_t n = 0; n < caster.linked_lights->size(); n++) {
		uint32_t light_handle = (*caster.linked_lights)[n];
		const Light &light = data.lights[light_handle];

		if (!light.visible) {
			continue;
		}

		LightCalc &lc = lights[num_lights++];

		Vector3 offset_light_caster;
		if (light.type != DIRECTIONAL) {
			offset_light_caster = light.pos - caster.pos;
			lc.dist = offset_light_caster.length();
		} else {
			lc.dist = Math::abs(light.pos.y - caster.pos.y);
		}
		lc.intensity = light.intensity;

		if (lc.dist >= light.range_max) // out of range
		{
			lc.dist = 0.0f;
			num_lights--;
			continue;
		}

		// Total weight is scaled impact of each light, from 0 to 1.
		lc.weight = (light.range_max - lc.dist) / light.range_max;

		// Apply cone on spot
		if (light.type == SPOT) {
			float dot = -offset_light_caster.normalized().dot(light.direction);

			//float dot_before = dot;
			dot -= light.spot_dot_threshold;
			dot *= light.spot_dot_multiplier;
			//print_line("dot before " + String(Variant(dot_before)) + ", after " + String(Variant(dot)));

			if (dot <= 0.0f) {
				lc.dist = 0.0f;
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
		float dist_left = lc.dist;
		if (dist_left > light.range_mid) {
			dist_left -= light.range_mid;

			// Scale 0 to 1
			dist_left /= light.range_mid_max;
			float fade = 1.0f - dist_left;

			r_focus_caster.light_modulate = MAX(r_focus_caster.light_modulate, fade);
		} else {
			r_focus_caster.light_modulate = 1.0f;
		}

		if (light.type != DIRECTIONAL) {
			lc.pos = light.pos;
		} else {
			lc.pos = caster.pos - light.direction;
		}
	}

	r_focus_caster.light_pos = Vector3();
	float intensity = 0.0f;

	// prevent divide by zero
	total_weight = MAX(total_weight, 0.0000001f);

	// second pass
	for (uint32_t n = 0; n < num_lights; n++) {
		LightCalc &lc = lights[n];

		// scale weight by total weight
		lc.weight /= total_weight;

		//		if (data.debug_output) {
		//			print_line("(" + itos(n) + ") light pos " + String(Variant(lc.pos)) + " weight " + String(Variant(lc.weight)));
		//		}

		r_focus_caster.light_pos += lc.pos * lc.weight;
		intensity += lc.intensity * lc.weight;
	}
	r_focus_caster.light_modulate *= intensity;
	r_focus_caster.light_modulate *= data.global_intensity;

	//print_line("focus light pos " + String(Variant(r_focus_caster.light_pos)));

	// Precalculate light direction so we don't have to do per pixel in the shader.
	//	r_focus_caster.light_dir = r_focus_caster.light_pos - caster.pos;
	//if (!Engine::get_singleton()->is_editor_hint())
	//	print_line("light_pos " + String(Variant(r_focus_caster.light_pos)));
	r_focus_caster.light_dir = caster.pos - r_focus_caster.light_pos;
	r_focus_caster.light_dir.normalize();

	r_focus_caster.cone_degrees = 45.0f;
}

void VisualServerBlobShadows::render_set_focus_handle(uint32_t p_focus_handle, const Vector3 &p_pos, const Transform &p_cam_transform, const CameraMatrix &p_cam_matrix) {
	data.render_focus_handle = p_focus_handle;
	// Get the camera frustum planes in world space.
	data.frustum_planes = p_cam_matrix.get_projection_planes(p_cam_transform);

	if (p_focus_handle) {
		Focus &focus = get_focus(p_focus_handle);
		focus.pos = p_pos;

		data.update_frame = Engine::get_singleton()->get_frames_drawn();

		// Update the focus only the first time it is demanded on a frame.
		if (focus.last_updated_frame != data.update_frame) {
			focus.last_updated_frame = data.update_frame;
			update_focus(focus);
		}

		// Cull focus to camera.
		for (uint32_t i = 0; i < focus.casters.size(); i++) {
			const FocusCaster &fc = focus.casters[i];
			const Caster &caster = get_caster(fc.caster_id + 1);

			focus.casters_in_camera[i] = 255;
			for (int n = 0; n < data.frustum_planes.size(); n++) {
				if (data.frustum_planes[n].distance_to(caster.pos) >= caster.size) {
					focus.casters_in_camera[i] = 0;
					break;
				}
			}
		}
	}
}

uint32_t VisualServerBlobShadows::fill_background_uniforms(const AABB &p_aabb, float *r_casters, float *r_lights, uint32_t p_max_casters) {
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

		for (uint32_t n = 0; n < focus.casters.size(); n++) {
			const FocusCaster &fc = focus.casters[n];

			// Out of view.
			if (!focus.casters_in_camera[n]) {
				continue;
			}

			const Caster &caster = data.casters[fc.caster_id];

			cdata.pos = caster.pos;
			cdata.size = caster.size;

			// Does caster + shadow intersect the geometry?
			if (!p_aabb.intersects(caster.aabb_boosted)) {
				continue;
			}

			memcpy(r_casters, &cdata, sizeof(CasterData));
			r_casters += 4;

			ldata.dir = -fc.light_dir;

			//if (!Engine::get_singleton()->is_editor_hint())
			//	print_line("light dir " + String(Variant(fc.light_dir)));

			//ldata.cone_degrees = fc.cone_degrees;
			ldata.modulate = fc.fraction * fc.light_modulate;
			//			ldata.pos = light.pos;
			//			ldata.cone_degrees = light.cone_degrees;

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

	//print_line("count " + itos(count));

	return count;
}

void *VisualServerBlobShadows::_instance_pair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int) {
	//print_line("pairing");

	uint32_t handle_a = (uint32_t)(uint64_t)p_A;
	uint32_t handle_b = (uint32_t)(uint64_t)p_B;

	VisualServerBlobShadows *self = static_cast<VisualServerBlobShadows *>(p_self);

	Instance &a = self->data.instances[handle_a];
	Instance &b = self->data.instances[handle_b];

	uint32_t caster_handle = 0;
	uint32_t light_handle = 0;

	if (a.type == VisualServerBlobShadows::IT_LIGHT) {
		DEV_ASSERT(b.type == VisualServerBlobShadows::IT_CASTER);
		light_handle = a.handle;
		caster_handle = b.handle;
	} else {
		DEV_ASSERT(a.type == VisualServerBlobShadows::IT_CASTER);
		DEV_ASSERT(b.type == VisualServerBlobShadows::IT_LIGHT);
		light_handle = b.handle;
		caster_handle = a.handle;
	}

	Light *light = &self->data.lights[light_handle];
	Caster *caster = &self->data.casters[caster_handle];

	DEV_ASSERT(light->linked_casters);
	DEV_ASSERT(caster->linked_lights);

	DEV_ASSERT(light->linked_casters->find(caster_handle) == -1);
	light->linked_casters->push_back(caster_handle);

	DEV_ASSERT(caster->linked_lights->find(light_handle) == -1);
	caster->linked_lights->push_back(light_handle);

	return nullptr;
}

void VisualServerBlobShadows::_instance_unpair(void *p_self, SpatialPartitionID, void *p_A, int, SpatialPartitionID, void *p_B, int, void *udata) {
	//print_line("unpairing");

	uint32_t handle_a = (uint32_t)(uint64_t)p_A;
	uint32_t handle_b = (uint32_t)(uint64_t)p_B;

	VisualServerBlobShadows *self = static_cast<VisualServerBlobShadows *>(p_self);

	Instance &a = self->data.instances[handle_a];
	Instance &b = self->data.instances[handle_b];

	uint32_t caster_handle = 0;
	uint32_t light_handle = 0;

	if (a.type == VisualServerBlobShadows::IT_LIGHT) {
		DEV_ASSERT(b.type == VisualServerBlobShadows::IT_CASTER);
		light_handle = a.handle;
		caster_handle = b.handle;
	} else {
		DEV_ASSERT(a.type == VisualServerBlobShadows::IT_CASTER);
		DEV_ASSERT(b.type == VisualServerBlobShadows::IT_LIGHT);
		light_handle = b.handle;
		caster_handle = a.handle;
	}

	Light *light = &self->data.lights[light_handle];
	Caster *caster = &self->data.casters[caster_handle];

	DEV_ASSERT(light->linked_casters);
	DEV_ASSERT(caster->linked_lights);

	int64_t found = light->linked_casters->find(caster_handle);
	DEV_ASSERT(found != -1);
	light->linked_casters->remove_unordered(found);

	found = caster->linked_lights->find(light_handle);
	DEV_ASSERT(found != -1);
	caster->linked_lights->remove_unordered(found);
}

VisualServerBlobShadows::VisualServerBlobShadows() {
	data.bvh.set_pair_callback(_instance_pair, this);
	data.bvh.set_unpair_callback(_instance_unpair, this);

	_active = GLOBAL_DEF("rendering/quality/blob_shadows/enable", false);
	data.max_casters = GLOBAL_DEF_RST("rendering/quality/blob_shadows/max_blobs", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/max_blobs", PropertyInfo(Variant::INT, "rendering/quality/blob_shadows/max_blobs", PROPERTY_HINT_RANGE, "1,128,1"));
	data.cutoff_boost = GLOBAL_DEF("rendering/quality/blob_shadows/cutoff_boost", 2.0f);
	//ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/cutoff_boost", PropertyInfo(Variant::REAL, "rendering/quality/blob_shadows/cutoff_boost", PROPERTY_HINT_RANGE, "0,128,1"));
	data.global_intensity = GLOBAL_DEF("rendering/quality/blob_shadows/global_intensity", 0.7f);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/blob_shadows/global_intensity", PropertyInfo(Variant::REAL, "rendering/quality/blob_shadows/global_intensity", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"));
}

VisualServerBlobShadows::~VisualServerBlobShadows() {
}
