/*************************************************************************/
/*  lightmapper_cpu.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "lightmapper_cpu.h"

#include "core/math/geometry.h"
#include "core/os/os.h"
#include "core/os/threaded_array_processor.h"
#include "core/project_settings.h"
#include "modules/raycast/lightmap_raycaster.h"

Error LightmapperCPU::_layout_atlas(int p_max_size, Vector2i *r_atlas_size, int *r_atlas_slices) {

	Vector2i atlas_size;
	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		if (mesh_instances[i].generate_lightmap) {
			Vector2i size = mesh_instances[i].size;
			atlas_size.width = MAX(atlas_size.width, size.width + 2);
			atlas_size.height = MAX(atlas_size.height, size.height + 2);
		}
	}

	int max = nearest_power_of_2_templated(atlas_size.width);
	max = MAX(max, nearest_power_of_2_templated(atlas_size.height));

	if (max > p_max_size) {
		return ERR_INVALID_DATA;
	}

	Vector2i best_atlas_size;
	int best_atlas_slices = 0;
	int best_atlas_memory = 0x7FFFFFFF;
	float best_atlas_mem_utilization = 0;
	Vector<AtlasOffset> best_atlas_offsets;
	Vector<Vector2i> best_scaled_sizes;

	int first_try_mem_occupied = 0;
	int first_try_mem_used = 0;
	for (int recovery_percent = 0; recovery_percent <= 100; recovery_percent += 10) {
		// These only make sense from the second round of the loop
		float recovery_scale = 1;
		int target_mem_occupied = 0;
		if (recovery_percent != 0) {
			target_mem_occupied = first_try_mem_occupied + (first_try_mem_used - first_try_mem_occupied) * recovery_percent * 0.01f;
			float new_squared_recovery_scale = static_cast<float>(target_mem_occupied) / first_try_mem_occupied;
			if (new_squared_recovery_scale > 1.0f) {
				recovery_scale = Math::sqrt(new_squared_recovery_scale);
			}
		}

		atlas_size = Vector2i(max, max);
		while (atlas_size.x <= p_max_size && atlas_size.y <= p_max_size) {

			if (recovery_percent != 0) {
				// Find out how much memory is not recoverable (because of lightmaps that can't grow),
				// to compute a greater recovery scale for those that can.
				int mem_unrecoverable = 0;

				for (unsigned int i = 0; i < mesh_instances.size(); i++) {
					if (mesh_instances[i].generate_lightmap) {
						Vector2i scaled_size = Vector2i(
								static_cast<int>(recovery_scale * mesh_instances[i].size.x),
								static_cast<int>(recovery_scale * mesh_instances[i].size.y));
						if (scaled_size.x + 2 > atlas_size.x || scaled_size.y + 2 > atlas_size.y) {
							mem_unrecoverable += scaled_size.x * scaled_size.y - mesh_instances[i].size.x * mesh_instances[i].size.y;
						}
					}
				}
				float new_squared_recovery_scale = static_cast<float>(target_mem_occupied - mem_unrecoverable) / (first_try_mem_occupied - mem_unrecoverable);
				if (new_squared_recovery_scale > 1.0f) {
					recovery_scale = Math::sqrt(new_squared_recovery_scale);
				}
			}

			Vector<Vector2i> scaled_sizes;
			scaled_sizes.resize(mesh_instances.size());
			{
				for (unsigned int i = 0; i < mesh_instances.size(); i++) {
					if (mesh_instances[i].generate_lightmap) {
						if (recovery_percent == 0) {
							scaled_sizes.write[i] = mesh_instances[i].size;
						} else {
							Vector2i scaled_size = Vector2i(
									static_cast<int>(recovery_scale * mesh_instances[i].size.x),
									static_cast<int>(recovery_scale * mesh_instances[i].size.y));
							if (scaled_size.x + 2 <= atlas_size.x && scaled_size.y + 2 <= atlas_size.y) {
								scaled_sizes.write[i] = scaled_size;
							} else {
								scaled_sizes.write[i] = mesh_instances[i].size;
							}
						}
					} else {
						// Don't consider meshes with no generated lightmap here; will compensate later
						scaled_sizes.write[i] = Vector2i();
					}
				}
			}

			Vector<Vector2i> source_sizes;
			source_sizes.resize(scaled_sizes.size());
			Vector<int> source_indices;
			source_indices.resize(scaled_sizes.size());
			for (int i = 0; i < source_sizes.size(); i++) {
				source_sizes.write[i] = scaled_sizes[i] + Vector2i(2, 2); // Add padding between lightmaps
				source_indices.write[i] = i;
			}

			Vector<AtlasOffset> curr_atlas_offsets;
			curr_atlas_offsets.resize(source_sizes.size());

			int slices = 0;

			while (source_sizes.size() > 0) {

				Vector<Geometry::PackRectsResult> offsets = Geometry::partial_pack_rects(source_sizes, atlas_size);
				Vector<int> new_indices;
				Vector<Vector2i> new_sources;
				for (int i = 0; i < offsets.size(); i++) {
					Geometry::PackRectsResult ofs = offsets[i];
					int sidx = source_indices[i];
					if (ofs.packed) {
						curr_atlas_offsets.write[sidx] = { slices, ofs.x + 1, ofs.y + 1 };
					} else {
						new_indices.push_back(sidx);
						new_sources.push_back(source_sizes[i]);
					}
				}

				source_sizes = new_sources;
				source_indices = new_indices;
				slices++;
			}

			int mem_used = atlas_size.x * atlas_size.y * slices;
			int mem_occupied = 0;
			for (int i = 0; i < curr_atlas_offsets.size(); i++) {
				mem_occupied += scaled_sizes[i].x * scaled_sizes[i].y;
			}

			float mem_utilization = static_cast<float>(mem_occupied) / mem_used;
			if (slices * atlas_size.y < 16384) { // Maximum Image size
				if (mem_used < best_atlas_memory || (mem_used == best_atlas_memory && mem_utilization > best_atlas_mem_utilization)) {
					best_atlas_size = atlas_size;
					best_atlas_offsets = curr_atlas_offsets;
					best_atlas_slices = slices;
					best_atlas_memory = mem_used;
					best_atlas_mem_utilization = mem_utilization;
					best_scaled_sizes = scaled_sizes;
				}
			}

			if (recovery_percent == 0) {
				first_try_mem_occupied = mem_occupied;
				first_try_mem_used = mem_used;
			}

			if (atlas_size.width == atlas_size.height) {
				atlas_size.width *= 2;
			} else {
				atlas_size.height *= 2;
			}
		}
	}

	if (best_atlas_size == Vector2i()) {
		return ERR_INVALID_DATA;
	}

	*r_atlas_size = best_atlas_size;
	*r_atlas_slices = best_atlas_slices;

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		if (best_scaled_sizes[i] != Vector2i()) {
			mesh_instances[i].size = best_scaled_sizes[i];
			mesh_instances[i].offset = Vector2i(best_atlas_offsets[i].x, best_atlas_offsets[i].y);
			mesh_instances[i].slice = best_atlas_offsets[i].slice;
		}
	}
	return OK;
}

void LightmapperCPU::_thread_func_callback(void *p_thread_data) {
	ThreadData *thread_data = reinterpret_cast<ThreadData *>(p_thread_data);
	thread_process_array(thread_data->count, thread_data->instance, &LightmapperCPU::_thread_func_wrapper, thread_data);
}

void LightmapperCPU::_thread_func_wrapper(uint32_t p_idx, ThreadData *p_thread_data) {

	if (thread_cancelled) {
		return;
	}

	(p_thread_data->instance->*p_thread_data->thread_func)(p_idx, p_thread_data->userdata);

	thread_progress++;
}

bool LightmapperCPU::_parallel_run(int p_count, const String &p_description, BakeThreadFunc p_thread_func, void *p_userdata, BakeStepFunc p_substep_func) {

	bool cancelled = false;
	if (p_substep_func) {
		cancelled = p_substep_func(0.0f, vformat("%s (%d/%d)", p_description, 0, p_count), nullptr, false);
	}

	thread_progress = 0;
	thread_cancelled = false;

#ifdef NO_THREAD
	for (int i = 0; !cancelled && i < p_count; i++) {
		(this->*p_thread_func)(i, p_userdata);
		float p = float(i) / p_count;
		if (p_substep_func) {
			cancelled = p_substep_func(p, vformat("%s (%d/%d)", p_description, i + 1, p_count), nullptr, false);
		}
	}
#else

	if (p_count == 0) {
		return cancelled;
	}

	ThreadData td;
	td.instance = this;
	td.count = p_count;
	td.thread_func = p_thread_func;
	td.userdata = p_userdata;
	Thread runner_thread;
	runner_thread.start(_thread_func_callback, &td);

	int progress = thread_progress;

	while (!cancelled && progress < p_count) {
		float p = float(progress) / p_count;
		if (p_substep_func) {
			cancelled = p_substep_func(p, vformat("%s (%d/%d)", p_description, progress + 1, p_count), nullptr, false);
		}
		progress = thread_progress;
	}
	thread_cancelled = cancelled;
	runner_thread.wait_to_finish();
#endif

	thread_cancelled = false;

	return cancelled;
}

void LightmapperCPU::_generate_buffer(uint32_t p_idx, void *p_unused) {

	const Size2i &size = mesh_instances[p_idx].size;

	int buffer_size = size.x * size.y;

	LocalVector<LightmapTexel> &lightmap = scene_lightmaps[p_idx];
	LocalVector<int> &lightmap_indices = scene_lightmap_indices[p_idx];

	lightmap_indices.resize(buffer_size);

	for (unsigned int i = 0; i < lightmap_indices.size(); i++) {
		lightmap_indices[i] = -1;
	}

	MeshData &md = mesh_instances[p_idx].data;

	LocalVector<Ref<Image> > albedo_images;
	LocalVector<Ref<Image> > emission_images;

	for (int surface_id = 0; surface_id < md.albedo.size(); surface_id++) {
		albedo_images.push_back(_init_bake_texture(md.albedo[surface_id], albedo_textures, Image::FORMAT_RGBA8));
		emission_images.push_back(_init_bake_texture(md.emission[surface_id], emission_textures, Image::FORMAT_RGBH));
	}

	int surface_id = 0;
	int surface_facecount = 0;
	const Vector3 *points_ptr = md.points.ptr();
	const Vector3 *normals_ptr = md.normal.ptr();
	const Vector2 *uvs_ptr = md.uv.empty() ? nullptr : md.uv.ptr();
	const Vector2 *uv2s_ptr = md.uv2.ptr();

	for (int i = 0; i < md.points.size() / 3; i++) {

		Ref<Image> albedo = albedo_images[surface_id];
		Ref<Image> emission = emission_images[surface_id];

		albedo->lock();
		emission->lock();
		_plot_triangle(&(uv2s_ptr[i * 3]), &(points_ptr[i * 3]), &(normals_ptr[i * 3]), uvs_ptr ? &(uvs_ptr[i * 3]) : nullptr, albedo, emission, size, lightmap, lightmap_indices);
		albedo->unlock();
		emission->unlock();

		surface_facecount++;
		if (surface_facecount == md.surface_facecounts[surface_id]) {
			surface_id++;
			surface_facecount = 0;
		}
	}
}

Ref<Image> LightmapperCPU::_init_bake_texture(const MeshData::TextureDef &p_texture_def, const Map<RID, Ref<Image> > &p_tex_cache, Image::Format p_default_format) {
	Ref<Image> ret;
	if (p_texture_def.tex_rid.is_valid()) {
		ret = p_tex_cache[p_texture_def.tex_rid]->duplicate();
		ret->lock();
		for (int j = 0; j < ret->get_height(); j++) {
			for (int i = 0; i < ret->get_width(); i++) {
				ret->set_pixel(i, j, ret->get_pixel(i, j) * p_texture_def.mul + p_texture_def.add);
			}
		}
		ret->unlock();
	} else {
		ret.instance();
		ret->create(8, 8, false, p_default_format);
		ret->fill(p_texture_def.add * p_texture_def.mul);
	}
	return ret;
}

Color LightmapperCPU::_bilinear_sample(const Ref<Image> &p_img, const Vector2 &p_uv, bool p_clamp_x, bool p_clamp_y) {

	int width = p_img->get_width();
	int height = p_img->get_height();

	Vector2 uv;
	uv.x = p_clamp_x ? p_uv.x : Math::fposmod(p_uv.x, 1.0f);
	uv.y = p_clamp_y ? p_uv.y : Math::fposmod(p_uv.y, 1.0f);

	float xf = uv.x * width;
	float yf = uv.y * height;

	int xi = (int)xf;
	int yi = (int)yf;

	Color texels[4];
	for (int i = 0; i < 4; i++) {
		int sample_x = xi + i % 2;
		int sample_y = yi + i / 2;

		sample_x = CLAMP(sample_x, 0, width - 1);
		sample_y = CLAMP(sample_y, 0, height - 1);

		texels[i] = p_img->get_pixel(sample_x, sample_y);
	}

	float tx = xf - xi;
	float ty = yf - yi;

	Color c = Color(0, 0, 0, 0);
	for (int i = 0; i < 4; i++) {
		c[i] = Math::lerp(Math::lerp(texels[0][i], texels[1][i], tx), Math::lerp(texels[2][i], texels[3][i], tx), ty);
	}
	return c;
}

Vector3 LightmapperCPU::_fix_sample_position(const Vector3 &p_position, const Vector3 &p_texel_center, const Vector3 &p_normal, const Vector3 &p_tangent, const Vector3 &p_bitangent, const Vector2 &p_texel_size) {

	Basis tangent_basis(p_tangent, p_bitangent, p_normal);
	tangent_basis.orthonormalize();
	Vector2 half_size = p_texel_size / 2.0f;
	Vector3 corrected = p_position;

	for (int i = -1; i <= 1; i += 1) {
		for (int j = -1; j <= 1; j += 1) {
			if (i == 0 && j == 0) continue;
			Vector3 offset = Vector3(half_size.x * i, half_size.y * j, 0.0);
			Vector3 rotated_offset = tangent_basis.xform_inv(offset);
			Vector3 target = p_texel_center + rotated_offset;
			Vector3 ray_vector = target - corrected;

			Vector3 ray_back_offset = -ray_vector.normalized() * parameters.bias / 2.0;
			Vector3 ray_origin = corrected + ray_back_offset;
			ray_vector = target - ray_origin;
			float ray_length = ray_vector.length();
			LightmapRaycaster::Ray ray(ray_origin + p_normal * parameters.bias, ray_vector.normalized(), 0.0f, ray_length + parameters.bias / 2.0);

			bool hit = raycaster->intersect(ray);
			if (hit) {
				ray.normal.normalize();
				if (ray.normal.dot(ray_vector.normalized()) > 0.0f) {
					corrected = ray_origin + ray.dir * ray.tfar + ray.normal * (parameters.bias * 2.0f);
				}
			}
		}
	}

	return corrected;
}

void LightmapperCPU::_plot_triangle(const Vector2 *p_vertices, const Vector3 *p_positions, const Vector3 *p_normals, const Vector2 *p_uvs, const Ref<Image> &p_albedo, const Ref<Image> &p_emission, Vector2i p_size, LocalVector<LightmapTexel> &r_lightmap, LocalVector<int> &r_lightmap_indices) {
	Vector2 pv0 = p_vertices[0];
	Vector2 pv1 = p_vertices[1];
	Vector2 pv2 = p_vertices[2];

	Vector2 v0 = pv0 * p_size;
	Vector2 v1 = pv1 * p_size;
	Vector2 v2 = pv2 * p_size;

	Vector3 p0 = p_positions[0];
	Vector3 p1 = p_positions[1];
	Vector3 p2 = p_positions[2];

	Vector3 n0 = p_normals[0];
	Vector3 n1 = p_normals[1];
	Vector3 n2 = p_normals[2];

	Vector2 uv0 = p_uvs == nullptr ? Vector2(0.5f, 0.5f) : p_uvs[0];
	Vector2 uv1 = p_uvs == nullptr ? Vector2(0.5f, 0.5f) : p_uvs[1];
	Vector2 uv2 = p_uvs == nullptr ? Vector2(0.5f, 0.5f) : p_uvs[2];

#define edgeFunction(a, b, c) ((c)[0] - (a)[0]) * ((b)[1] - (a)[1]) - ((c)[1] - (a)[1]) * ((b)[0] - (a)[0])

	if (edgeFunction(v0, v1, v2) < 0.0) {
		SWAP(pv1, pv2);
		SWAP(v1, v2);
		SWAP(p1, p2);
		SWAP(n1, n2);
		SWAP(uv1, uv2);
	}

	Vector3 edge1 = p1 - p0;
	Vector3 edge2 = p2 - p0;

	Vector2 uv_edge1 = pv1 - pv0;
	Vector2 uv_edge2 = pv2 - pv0;

	float r = 1.0f / (uv_edge1.x * uv_edge2.y - uv_edge1.y * uv_edge2.x);

	Vector3 tangent = (edge1 * uv_edge2.y - edge2 * uv_edge1.y) * r;
	Vector3 bitangent = (edge2 * uv_edge1.x - edge1 * uv_edge2.x) * r;

	tangent.normalize();
	bitangent.normalize();

	// Compute triangle bounding box
	Vector2 bbox_min = Vector2(MIN(v0.x, MIN(v1.x, v2.x)), MIN(v0.y, MIN(v1.y, v2.y)));
	Vector2 bbox_max = Vector2(MAX(v0.x, MAX(v1.x, v2.x)), MAX(v0.y, MAX(v1.y, v2.y)));

	bbox_min = bbox_min.floor();
	bbox_max = bbox_max.ceil();

	uint32_t min_x = MAX(bbox_min.x - 2, 0);
	uint32_t min_y = MAX(bbox_min.y - 2, 0);
	uint32_t max_x = MIN(bbox_max.x, p_size.x - 1);
	uint32_t max_y = MIN(bbox_max.y, p_size.y - 1);

	Vector2 texel_size;
	Vector2 centroid = (v0 + v1 + v2) / 3.0f;
	Vector3 centroid_pos = (p0 + p1 + p2) / 3.0f;
	for (int i = 0; i < 2; i++) {
		Vector2 p = centroid;
		p[i] += 1;
		Vector3 bary = Geometry::barycentric_coordinates_2d(p, v0, v1, v2);
		if (bary.length() <= 1.0) {
			Vector3 pos = p0 * bary[0] + p1 * bary[1] + p2 * bary[2];
			texel_size[i] = centroid_pos.distance_to(pos);
		}
	}

	Vector<Vector2> pixel_polygon;
	pixel_polygon.resize(4);
	static const Vector2 corners[4] = { Vector2(0, 0), Vector2(0, 1), Vector2(1, 1), Vector2(1, 0) };

	Vector<Vector2> triangle_polygon;
	triangle_polygon.push_back(v0);
	triangle_polygon.push_back(v1);
	triangle_polygon.push_back(v2);

	for (uint32_t j = min_y; j <= max_y; ++j) {
		for (uint32_t i = min_x; i <= max_x; i++) {

			int ofs = j * p_size.x + i;
			int texel_idx = r_lightmap_indices[ofs];

			if (texel_idx >= 0 && r_lightmap[texel_idx].area_coverage >= 0.5f) {
				continue;
			}

			Vector3 barycentric_coords;
			float area_coverage = 0.0f;
			bool intersected = false;

			for (int k = 0; k < 4; k++) {
				pixel_polygon.write[k] = Vector2(i, j) + corners[k];
			}

			const float max_dist = 0.05;
			bool v0eqv1 = v0.distance_squared_to(v1) < max_dist;
			bool v1eqv2 = v1.distance_squared_to(v2) < max_dist;
			bool v2eqv0 = v2.distance_squared_to(v0) < max_dist;
			if (v0eqv1 && v1eqv2 && v2eqv0) {
				intersected = true;
				barycentric_coords = Vector3(1, 0, 0);
			} else if (v0eqv1 || v1eqv2 || v2eqv0) {

				Vector<Vector2> segment;
				segment.resize(2);
				if (v0eqv1) {
					segment.write[0] = v0;
					segment.write[1] = v2;
				} else if (v1eqv2) {
					segment.write[0] = v1;
					segment.write[1] = v0;
				} else {
					segment.write[0] = v0;
					segment.write[1] = v1;
				}

				Vector<Vector<Vector2> > intersected_segments = Geometry::intersect_polyline_with_polygon_2d(segment, pixel_polygon);
				ERR_FAIL_COND_MSG(intersected_segments.size() > 1, "[Lightmapper] Itersecting a segment and a convex polygon should give at most one segment.");
				if (!intersected_segments.empty()) {
					const Vector<Vector2> &intersected_segment = intersected_segments[0];
					ERR_FAIL_COND_MSG(intersected_segment.size() != 2, "[Lightmapper] Itersecting a segment and a convex polygon should give at most one segment.");
					Vector2 sample_pos = (intersected_segment[0] + intersected_segment[1]) / 2.0f;

					float u = (segment[0].distance_to(sample_pos)) / (segment[0].distance_to(segment[1]));
					float v = (1.0f - u) / 2.0f;
					intersected = true;
					if (v0eqv1) {
						barycentric_coords = Vector3(v, v, u);
					} else if (v1eqv2) {
						barycentric_coords = Vector3(u, v, v);
					} else {
						barycentric_coords = Vector3(v, u, v);
					}
				}

			} else if (edgeFunction(v0, v1, v2) < 0.005) {
				Vector2 direction = v0 - v1;
				Vector2 perpendicular = Vector2(direction.y, -direction.x);

				Vector<Vector2> line;
				int middle_vertex;

				if (SGN(edgeFunction(v0, v0 + perpendicular, v1)) != SGN(edgeFunction(v0, v0 + perpendicular, v2))) {
					line.push_back(v1);
					line.push_back(v2);
					middle_vertex = 0;
				} else if (SGN(edgeFunction(v1, v1 + perpendicular, v0)) != SGN(edgeFunction(v1, v1 + perpendicular, v2))) {
					line.push_back(v0);
					line.push_back(v2);
					middle_vertex = 1;
				} else {
					line.push_back(v0);
					line.push_back(v1);
					middle_vertex = 2;
				}

				Vector<Vector<Vector2> > intersected_lines = Geometry::intersect_polyline_with_polygon_2d(line, pixel_polygon);

				ERR_FAIL_COND_MSG(intersected_lines.size() > 1, "[Lightmapper] Itersecting a line and a convex polygon should give at most one line.");

				if (!intersected_lines.empty()) {
					intersected = true;
					const Vector<Vector2> &intersected_line = intersected_lines[0];
					Vector2 sample_pos = (intersected_line[0] + intersected_line[1]) / 2.0f;

					float line_length = line[0].distance_to(line[1]);
					float norm = line[0].distance_to(sample_pos) / line_length;

					if (middle_vertex == 0) {
						barycentric_coords = Vector3(0.0f, 1.0f - norm, norm);
					} else if (middle_vertex == 1) {
						barycentric_coords = Vector3(1.0f - norm, 0.0f, norm);
					} else {
						barycentric_coords = Vector3(1.0f - norm, norm, 0.0f);
					}
				}
			} else {

				Vector<Vector<Vector2> > intersected_polygons = Geometry::intersect_polygons_2d(pixel_polygon, triangle_polygon);

				ERR_FAIL_COND_MSG(intersected_polygons.size() > 1, "[Lightmapper] Itersecting two convex polygons should give at most one polygon.");

				if (!intersected_polygons.empty()) {
					const Vector<Vector2> &intersected_polygon = intersected_polygons[0];

					// do centroid sampling
					Vector2 sample_pos = intersected_polygon[0];
					Vector2 area_center = Vector2(i, j) + Vector2(0.5f, 0.5f);
					float intersected_area = (intersected_polygon[0] - area_center).cross(intersected_polygon[intersected_polygon.size() - 1] - area_center);
					for (int k = 1; k < intersected_polygon.size(); k++) {
						sample_pos += intersected_polygon[k];
						intersected_area += (intersected_polygon[k] - area_center).cross(intersected_polygon[k - 1] - area_center);
					}

					if (intersected_area != 0.0f) {
						sample_pos /= intersected_polygon.size();
						barycentric_coords = Geometry::barycentric_coordinates_2d(sample_pos, v0, v1, v2);
						intersected = true;
						area_coverage = ABS(intersected_area) / 2.0f;
					}
				}

				if (!intersected) {
					for (int k = 0; k < 4; ++k) {
						for (int l = 0; l < 3; ++l) {
							Vector2 intersection_point;
							if (Geometry::segment_intersects_segment_2d(pixel_polygon[k], pixel_polygon[(k + 1) % 4], triangle_polygon[l], triangle_polygon[(l + 1) % 3], &intersection_point)) {
								intersected = true;
								barycentric_coords = Geometry::barycentric_coordinates_2d(intersection_point, v0, v1, v2);
								break;
							}
						}
						if (intersected) {
							break;
						}
					}
				}
			}

			if (texel_idx >= 0 && area_coverage < r_lightmap[texel_idx].area_coverage) {
				continue; // A previous triangle gives better pixel coverage
			}

			Vector2 pixel = Vector2(i, j);
			if (!intersected && v0.floor() == pixel) {
				intersected = true;
				barycentric_coords = Vector3(1, 0, 0);
			}

			if (!intersected && v1.floor() == pixel) {
				intersected = true;
				barycentric_coords = Vector3(0, 1, 0);
			}

			if (!intersected && v2.floor() == pixel) {
				intersected = true;
				barycentric_coords = Vector3(0, 0, 1);
			}

			if (!intersected) {
				continue;
			}

			if (Math::is_nan(barycentric_coords.x) || Math::is_nan(barycentric_coords.y) || Math::is_nan(barycentric_coords.z)) {
				continue;
			}

			if (Math::is_inf(barycentric_coords.x) || Math::is_inf(barycentric_coords.y) || Math::is_inf(barycentric_coords.z)) {
				continue;
			}

			r_lightmap_indices[ofs] = r_lightmap.size();

			Vector3 pos = p0 * barycentric_coords[0] + p1 * barycentric_coords[1] + p2 * barycentric_coords[2];
			Vector3 normal = n0 * barycentric_coords[0] + n1 * barycentric_coords[1] + n2 * barycentric_coords[2];

			Vector2 uv = uv0 * barycentric_coords[0] + uv1 * barycentric_coords[1] + uv2 * barycentric_coords[2];
			Color c = _bilinear_sample(p_albedo, uv);
			Color e = _bilinear_sample(p_emission, uv);

			Vector2 texel_center = Vector2(i, j) + Vector2(0.5f, 0.5f);
			Vector3 texel_center_bary = Geometry::barycentric_coordinates_2d(texel_center, v0, v1, v2);

			if (texel_center_bary.length_squared() <= 1.3 && !Math::is_nan(texel_center_bary.x) && !Math::is_nan(texel_center_bary.y) && !Math::is_nan(texel_center_bary.z) && !Math::is_inf(texel_center_bary.x) && !Math::is_inf(texel_center_bary.y) && !Math::is_inf(texel_center_bary.z)) {
				Vector3 texel_center_pos = p0 * texel_center_bary[0] + p1 * texel_center_bary[1] + p2 * texel_center_bary[2];
				pos = _fix_sample_position(pos, texel_center_pos, normal, tangent, bitangent, texel_size);
			}

			LightmapTexel texel;
			texel.normal = normal.normalized();
			texel.pos = pos;
			texel.albedo = Vector3(c.r, c.g, c.b);
			texel.alpha = c.a;
			texel.emission = Vector3(e.r, e.g, e.b);
			texel.area_coverage = area_coverage;
			r_lightmap.push_back(texel);
		}
	}
}

void LightmapperCPU::_compute_direct_light(uint32_t p_idx, void *r_lightmap) {

	LightmapTexel *lightmap = (LightmapTexel *)r_lightmap;
	for (unsigned int i = 0; i < lights.size(); ++i) {

		const Light &light = lights[i];
		Vector3 normal = lightmap[p_idx].normal;
		Vector3 position = lightmap[p_idx].pos;
		Vector3 final_energy;
		Color c = light.color;
		Vector3 light_energy = Vector3(c.r, c.g, c.b) * light.energy;

		if (light.type == LIGHT_TYPE_OMNI) {
			Vector3 light_direction = (position - light.position).normalized();
			if (normal.dot(light_direction) >= 0.0) {
				continue;
			}
			float dist = position.distance_to(light.position);

			if (dist <= light.range) {
				LightmapRaycaster::Ray ray = LightmapRaycaster::Ray(position, -light_direction, parameters.bias, dist - parameters.bias);
				if (raycaster->intersect(ray)) {
					continue;
				}
				float att = powf(1.0 - dist / light.range, light.attenuation);
				final_energy = light_energy * att * MAX(0, normal.dot(-light_direction));
			}
		}

		if (light.type == LIGHT_TYPE_SPOT) {

			Vector3 light_direction = (position - light.position).normalized();
			if (normal.dot(light_direction) >= 0.0) {
				continue;
			}

			float angle = Math::acos(light.direction.dot(light_direction));

			if (angle > light.spot_angle) {
				continue;
			}

			float dist = position.distance_to(light.position);
			if (dist > light.range) {
				continue;
			}

			LightmapRaycaster::Ray ray = LightmapRaycaster::Ray(position, -light_direction, parameters.bias, dist);
			if (raycaster->intersect(ray)) {
				continue;
			}

			float normalized_dist = dist * (1.0f / MAX(0.001f, light.range));
			float norm_light_attenuation = Math::pow(MAX(1.0f - normalized_dist, 0.001f), light.attenuation);

			float spot_cutoff = Math::cos(light.spot_angle);
			float scos = MAX(light_direction.dot(light.direction), spot_cutoff);
			float spot_rim = (1.0f - scos) / (1.0f - spot_cutoff);
			norm_light_attenuation *= 1.0f - pow(MAX(spot_rim, 0.001f), light.spot_attenuation);
			final_energy = light_energy * norm_light_attenuation * MAX(0, normal.dot(-light_direction));
		}

		if (light.type == LIGHT_TYPE_DIRECTIONAL) {
			if (normal.dot(light.direction) >= 0.0) {
				continue;
			}

			LightmapRaycaster::Ray ray = LightmapRaycaster::Ray(position + normal * parameters.bias, -light.direction, parameters.bias);
			if (raycaster->intersect(ray)) {
				continue;
			}

			final_energy = light_energy * MAX(0, normal.dot(-light.direction));
		}

		lightmap[p_idx].direct_light += final_energy * light.indirect_multiplier;
		if (light.bake_direct) {
			lightmap[p_idx].output_light += final_energy;
		}
	}
}

_ALWAYS_INLINE_ float uniform_rand() {
	/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	static thread_local uint32_t state = Math::rand();
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	/* implicit conversion from 'unsigned int' to 'float' changes value from 4294967295 to 4294967296 */
	return float(state) / float(UINT32_MAX);
}

void LightmapperCPU::_compute_indirect_light(uint32_t p_idx, void *r_lightmap) {

	LightmapTexel *lightmap = (LightmapTexel *)r_lightmap;
	LightmapTexel &texel = lightmap[p_idx];

	Vector3 accum;

	const Vector3 const_forward = Vector3(0, 0, 1);
	const Vector3 const_up = Vector3(0, 1, 0);

	for (int i = 0; i < parameters.samples; i++) {

		Vector3 color;
		Vector3 throughput = Vector3(1.0f, 1.0f, 1.0f);

		Vector3 position = texel.pos;
		Vector3 normal = texel.normal;
		Vector3 direction;

		for (int depth = 0; depth < parameters.bounces; depth++) {

			Vector3 tangent = const_forward.cross(normal);
			if (unlikely(tangent.length_squared() < 0.005f)) {
				tangent = const_up.cross(normal);
			}
			tangent.normalize();
			Vector3 bitangent = tangent.cross(normal);
			bitangent.normalize();

			Basis normal_xform = Basis(tangent, bitangent, normal);
			normal_xform.transpose();

			float u1 = uniform_rand();
			float u2 = uniform_rand();

			float radius = Math::sqrt(u1);
			float theta = Math_TAU * u2;

			Vector3 axis = Vector3(radius * Math::cos(theta), radius * Math::sin(theta), Math::sqrt(MAX(0.0f, 1.0f - u1)));

			direction = normal_xform.xform(axis);

			// We can skip multiplying throughput by cos(theta) because de sampling PDF is also cos(theta) and they cancel each other
			//float pdf = normal.dot(direction);
			//throughput *= normal.dot(direction)/pdf;

			LightmapRaycaster::Ray ray(position, direction, parameters.bias);
			bool hit = raycaster->intersect(ray);

			if (!hit) {
				if (parameters.environment_panorama.is_valid()) {
					direction = parameters.environment_transform.xform_inv(direction);
					Vector2 st = Vector2(Math::atan2(direction.z, direction.x), Math::acos(direction.y));

					if (Math::is_nan(st.y)) {
						st.y = direction.y > 0.0 ? 0.0 : Math_PI;
					}

					st.x += Math_PI;
					st /= Vector2(Math_TAU, Math_PI);
					st.x = Math::fmod(st.x + 0.75, 1.0);
					Color c = _bilinear_sample(parameters.environment_panorama, st, false, true);
					color += throughput * Vector3(c.r, c.g, c.b) * c.a;
				}
				break;
			}

			unsigned int hit_mesh_id = ray.geomID;
			const Vector2i &size = mesh_instances[hit_mesh_id].size;

			int x = CLAMP(ray.u * size.x, 0, size.x - 1);
			int y = CLAMP(ray.v * size.y, 0, size.y - 1);

			const int idx = scene_lightmap_indices[hit_mesh_id][y * size.x + x];

			if (idx < 0) {
				break;
			}

			const LightmapTexel &sample = scene_lightmaps[hit_mesh_id][idx];

			if (sample.normal.dot(ray.dir) > 0.0 && !no_shadow_meshes.has(hit_mesh_id)) {
				// We hit a back-face
				break;
			}

			color += throughput * sample.emission;
			throughput *= sample.albedo;
			color += throughput * sample.direct_light;

			// Russian Roulette
			// https://computergraphics.stackexchange.com/questions/2316/is-russian-roulette-really-the-answer
			const float p = throughput[throughput.max_axis()];
			if (uniform_rand() > p) {
				break;
			}
			throughput *= 1.0f / p;

			position = sample.pos;
			normal = sample.normal;
		}
		accum += color;
	}

	texel.output_light += accum / parameters.samples;
}

void LightmapperCPU::_post_process(uint32_t p_idx, void *r_output) {

	const MeshInstance &mesh = mesh_instances[p_idx];

	if (!mesh.generate_lightmap) {
		return;
	}

	LocalVector<int> &indices = scene_lightmap_indices[p_idx];
	LocalVector<LightmapTexel> &lightmap = scene_lightmaps[p_idx];
	Vector3 *output = ((LocalVector<Vector3> *)r_output)[p_idx].ptr();
	Vector2i size = mesh.size;

	// Blit texels to buffer
	const int margin = 4;
	for (int i = 0; i < size.y; i++) {
		for (int j = 0; j < size.x; j++) {
			int idx = indices[i * size.x + j];
			if (idx >= 0) {
				output[i * size.x + j] = lightmap[idx].output_light;
				continue; // filled, skip
			}

			int closest_idx = -1;
			float closest_dist = 1e20;

			for (int y = i - margin; y <= i + margin; y++) {
				for (int x = j - margin; x <= j + margin; x++) {

					if (x == j && y == i)
						continue;
					if (x < 0 || x >= size.x)
						continue;
					if (y < 0 || y >= size.y)
						continue;
					int cell_idx = indices[y * size.x + x];
					if (cell_idx < 0) {
						continue; //also ensures that blitted stuff is not reused
					}

					float dist = Vector2(i - y, j - x).length_squared();
					if (dist < closest_dist) {
						closest_dist = dist;
						closest_idx = cell_idx;
					}
				}
			}

			if (closest_idx != -1) {
				output[i * size.x + j] = lightmap[closest_idx].output_light;
			}
		}
	}

	lightmap.clear();

	LocalVector<UVSeam> seams;
	_compute_seams(mesh, seams);

	_fix_seams(seams, output, size);
	_dilate_lightmap(output, indices, size, margin);

	if (parameters.use_denoiser) {
		Ref<LightmapDenoiser> denoiser = LightmapDenoiser::create();

		if (denoiser.is_valid()) {
			int data_size = size.x * size.y * sizeof(Vector3);
			Ref<Image> current_image;
			current_image.instance();
			{
				PoolByteArray data;
				data.resize(data_size);
				PoolByteArray::Write w = data.write();
				copymem(w.ptr(), output, data_size);
				current_image->create(size.x, size.y, false, Image::FORMAT_RGBF, data);
			}

			Ref<Image> denoised_image = denoiser->denoise_image(current_image);

			PoolByteArray denoised_data = denoised_image->get_data();
			denoised_image.unref();
			PoolByteArray::Read r = denoised_data.read();
			copymem(output, r.ptr(), data_size);
		}
	}

	_dilate_lightmap(output, indices, size, margin);
	_fix_seams(seams, output, size);
	_dilate_lightmap(output, indices, size, margin);

	indices.clear();
}

void LightmapperCPU::_compute_seams(const MeshInstance &p_mesh, LocalVector<UVSeam> &r_seams) {
	float max_uv_distance = 1.0f / MAX(p_mesh.size.x, p_mesh.size.y);
	max_uv_distance *= max_uv_distance; // We use distance_to_squared(), so wee need to square the max distance as well
	float max_pos_distance = 0.00025f;
	float max_normal_distance = 0.05f;

	const Vector<Vector3> &points = p_mesh.data.points;
	const Vector<Vector2> &uv2s = p_mesh.data.uv2;
	const Vector<Vector3> &normals = p_mesh.data.normal;

	LocalVector<SeamEdge> edges;
	edges.resize(points.size()); // One edge per vertex

	for (int i = 0; i < points.size(); i += 3) {
		Vector3 triangle_vtxs[3] = { points[i + 0], points[i + 1], points[i + 2] };
		Vector2 triangle_uvs[3] = { uv2s[i + 0], uv2s[i + 1], uv2s[i + 2] };
		Vector3 triangle_normals[3] = { normals[i + 0], normals[i + 1], normals[i + 2] };

		for (int k = 0; k < 3; k++) {
			int idx[2];
			idx[0] = k;
			idx[1] = (k + 1) % 3;

			if (triangle_vtxs[idx[1]] < triangle_vtxs[idx[0]]) {
				SWAP(idx[0], idx[1]);
			}

			SeamEdge e;
			for (int l = 0; l < 2; ++l) {
				e.pos[l] = triangle_vtxs[idx[l]];
				e.uv[l] = triangle_uvs[idx[l]];
				e.normal[l] = triangle_normals[idx[l]];
			}
			edges[i + k] = e;
		}
	}

	edges.sort();

	for (unsigned int j = 0; j < edges.size(); j++) {
		const SeamEdge &edge0 = edges[j];

		if (edge0.uv[0].distance_squared_to(edge0.uv[1]) < 0.001) {
			continue;
		}

		if (edge0.pos[0].distance_squared_to(edge0.pos[1]) < 0.001) {
			continue;
		}

		for (unsigned int k = j + 1; k < edges.size() && edges[k].pos[0].x < (edge0.pos[0].x + max_pos_distance * 1.1f); k++) {
			const SeamEdge &edge1 = edges[k];

			if (edge1.uv[0].distance_squared_to(edge1.uv[1]) < 0.001) {
				continue;
			}

			if (edge1.pos[0].distance_squared_to(edge1.pos[1]) < 0.001) {
				continue;
			}

			if (edge0.uv[0].distance_squared_to(edge1.uv[0]) < max_uv_distance && edge0.uv[1].distance_squared_to(edge1.uv[1]) < max_uv_distance) {
				continue;
			}

			if (edge0.pos[0].distance_squared_to(edge1.pos[0]) > max_pos_distance || edge0.pos[1].distance_squared_to(edge1.pos[1]) > max_pos_distance) {
				continue;
			}

			if (edge0.normal[0].distance_squared_to(edge1.normal[0]) > max_normal_distance || edge0.normal[1].distance_squared_to(edge1.normal[1]) > max_normal_distance) {
				continue;
			}

			UVSeam s;
			s.edge0[0] = edge0.uv[0];
			s.edge0[1] = edge0.uv[1];
			s.edge1[0] = edge1.uv[0];
			s.edge1[1] = edge1.uv[1];
			r_seams.push_back(s);
		}
	}
}

void LightmapperCPU::_fix_seams(const LocalVector<UVSeam> &p_seams, Vector3 *r_lightmap, Vector2i p_size) {
	LocalVector<Vector3> extra_buffer;
	extra_buffer.resize(p_size.x * p_size.y);

	copymem(extra_buffer.ptr(), r_lightmap, p_size.x * p_size.y * sizeof(Vector3));

	Vector3 *read_ptr = extra_buffer.ptr();
	Vector3 *write_ptr = r_lightmap;

	for (int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < p_seams.size(); j++) {
			_fix_seam(p_seams[j].edge0[0], p_seams[j].edge0[1], p_seams[j].edge1[0], p_seams[j].edge1[1], read_ptr, write_ptr, p_size);
			_fix_seam(p_seams[j].edge1[0], p_seams[j].edge1[1], p_seams[j].edge0[0], p_seams[j].edge0[1], read_ptr, write_ptr, p_size);
		}
		copymem(read_ptr, write_ptr, p_size.x * p_size.y * sizeof(Vector3));
	}
}

void LightmapperCPU::_fix_seam(const Vector2 &p_pos0, const Vector2 &p_pos1, const Vector2 &p_uv0, const Vector2 &p_uv1, const Vector3 *p_read_buffer, Vector3 *r_write_buffer, const Vector2i &p_size) {

	Vector2 line[2];
	line[0] = p_pos0 * p_size;
	line[1] = p_pos1 * p_size;

	const Vector2i start_pixel = line[0].floor();
	const Vector2i end_pixel = line[1].floor();

	Vector2 seam_dir = (line[1] - line[0]).normalized();
	Vector2 t_delta = Vector2(1.0f / Math::abs(seam_dir.x), 1.0f / Math::abs(seam_dir.y));
	Vector2i step = Vector2(seam_dir.x > 0 ? 1 : (seam_dir.x < 0 ? -1 : 0), seam_dir.y > 0 ? 1 : (seam_dir.y < 0 ? -1 : 0));

	Vector2 t_next = Vector2(Math::fmod(line[0].x, 1.0f), Math::fmod(line[0].y, 1.0f));

	if (step.x == 1) {
		t_next.x = 1.0f - t_next.x;
	}

	if (step.y == 1) {
		t_next.y = 1.0f - t_next.y;
	}

	t_next.x /= Math::abs(seam_dir.x);
	t_next.y /= Math::abs(seam_dir.y);

	if (Math::is_nan(t_next.x)) {
		t_next.x = 1e20f;
	}

	if (Math::is_nan(t_next.y)) {
		t_next.y = 1e20f;
	}

	Vector2i pixel = start_pixel;
	Vector2 start_p = start_pixel;
	float line_length = line[0].distance_to(line[1]);

	if (line_length == 0.0f) {
		return;
	}

	while (start_p.distance_to(pixel) < line_length + 1.0f) {

		Vector2 current_point = Vector2(pixel) + Vector2(0.5f, 0.5f);
		current_point = Geometry::get_closest_point_to_segment_2d(current_point, line);
		float t = line[0].distance_to(current_point) / line_length;

		Vector2 current_uv = p_uv0 * (1.0 - t) + p_uv1 * t;
		Vector2i sampled_point = (current_uv * p_size).floor();

		Vector3 current_color = r_write_buffer[pixel.y * p_size.x + pixel.x];
		Vector3 sampled_color = p_read_buffer[sampled_point.y * p_size.x + sampled_point.x];

		r_write_buffer[pixel.y * p_size.x + pixel.x] = current_color * 0.6f + sampled_color * 0.4f;

		if (pixel == end_pixel) {
			break;
		}

		if (t_next.x < t_next.y) {
			pixel.x += step.x;
			t_next.x += t_delta.x;
		} else {
			pixel.y += step.y;
			t_next.y += t_delta.y;
		}
	}
}

void LightmapperCPU::_dilate_lightmap(Vector3 *r_lightmap, const LocalVector<int> p_indices, Vector2i p_size, int margin) {
	for (int i = 0; i < p_size.y; i++) {
		for (int j = 0; j < p_size.x; j++) {
			int idx = p_indices[i * p_size.x + j];
			if (idx >= 0) {
				continue; //filled, skip
			}

			Vector2i closest;
			float closest_dist = 1e20;

			for (int y = i - margin; y <= i + margin; y++) {
				for (int x = j - margin; x <= j + margin; x++) {

					if (x == j && y == i)
						continue;
					if (x < 0 || x >= p_size.x)
						continue;
					if (y < 0 || y >= p_size.y)
						continue;
					int cell_idx = p_indices[y * p_size.x + x];
					if (cell_idx < 0) {
						continue; //also ensures that blitted stuff is not reused
					}

					float dist = Vector2(i - y, j - x).length_squared();
					if (dist < closest_dist) {
						closest_dist = dist;
						closest = Vector2(x, y);
					}
				}
			}

			if (closest_dist < 1e20) {
				r_lightmap[i * p_size.x + j] = r_lightmap[closest.y * p_size.x + closest.x];
			}
		}
	}
}

void LightmapperCPU::_blit_lightmap(const Vector<Vector3> &p_src, const Vector2i &p_size, Ref<Image> &p_dst, int p_x, int p_y, bool p_with_padding) {
	int padding = p_with_padding ? 1 : 0;
	ERR_FAIL_COND(p_x < padding || p_y < padding);
	ERR_FAIL_COND(p_x + p_size.x > p_dst->get_width() - padding);
	ERR_FAIL_COND(p_y + p_size.y > p_dst->get_height() - padding);

	p_dst->lock();
	for (int y = 0; y < p_size.y; y++) {
		const Vector3 *__restrict src = p_src.ptr() + y * p_size.x;
		for (int x = 0; x < p_size.x; x++) {
			p_dst->set_pixel(p_x + x, p_y + y, Color(src->x, src->y, src->z));
			src++;
		}
	}

	if (p_with_padding) {
		for (int y = -1; y < p_size.y + 1; y++) {
			int yy = CLAMP(y, 0, p_size.y - 1);
			int idx_left = yy * p_size.x;
			int idx_right = idx_left + p_size.x - 1;
			p_dst->set_pixel(p_x - 1, p_y + y, Color(p_src[idx_left].x, p_src[idx_left].y, p_src[idx_left].z));
			p_dst->set_pixel(p_x + p_size.x, p_y + y, Color(p_src[idx_right].x, p_src[idx_right].y, p_src[idx_right].z));
		}

		for (int x = -1; x < p_size.x + 1; x++) {
			int xx = CLAMP(x, 0, p_size.x - 1);
			int idx_top = xx;
			int idx_bot = idx_top + (p_size.y - 1) * p_size.x;
			p_dst->set_pixel(p_x + x, p_y - 1, Color(p_src[idx_top].x, p_src[idx_top].y, p_src[idx_top].z));
			p_dst->set_pixel(p_x + x, p_y + p_size.y, Color(p_src[idx_bot].x, p_src[idx_bot].y, p_src[idx_bot].z));
		}
	}
	p_dst->unlock();
}

LightmapperCPU::BakeError LightmapperCPU::bake(BakeQuality p_quality, bool p_use_denoiser, int p_bounces, float p_bias, bool p_generate_atlas, int p_max_texture_size, const Ref<Image> &p_environment_panorama, const Basis &p_environment_transform, BakeStepFunc p_step_function, void *p_bake_userdata, BakeStepFunc p_substep_function) {

	if (p_step_function) {
		bool cancelled = p_step_function(0.0, TTR("Begin Bake"), p_bake_userdata, true);
		if (cancelled) {
			return BAKE_ERROR_USER_ABORTED;
		}
	}

	raycaster = LightmapRaycaster::create();

	ERR_FAIL_COND_V(raycaster.is_null(), BAKE_ERROR_NO_RAYCASTER);

	// Collect parameters
	parameters.use_denoiser = p_use_denoiser;
	parameters.bias = p_bias;
	parameters.bounces = p_bounces;
	parameters.environment_transform = p_environment_transform;
	parameters.environment_panorama = p_environment_panorama;

	switch (p_quality) {
		case BAKE_QUALITY_LOW: {
			parameters.samples = GLOBAL_GET("rendering/cpu_lightmapper/quality/low_quality_ray_count");
		} break;
		case BAKE_QUALITY_MEDIUM: {
			parameters.samples = GLOBAL_GET("rendering/cpu_lightmapper/quality/medium_quality_ray_count");
		} break;
		case BAKE_QUALITY_HIGH: {
			parameters.samples = GLOBAL_GET("rendering/cpu_lightmapper/quality/high_quality_ray_count");
		} break;
		case BAKE_QUALITY_ULTRA: {
			parameters.samples = GLOBAL_GET("rendering/cpu_lightmapper/quality/ultra_quality_ray_count");
		} break;
	}

	bake_textures.clear();

	if (p_step_function) {
		bool cancelled = p_step_function(0.1, TTR("Preparing data structures"), p_bake_userdata, true);
		if (cancelled) {
			return BAKE_ERROR_USER_ABORTED;
		}
	}

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		raycaster->add_mesh(mesh_instances[i].data.points, mesh_instances[i].data.normal, mesh_instances[i].data.uv2, i);
	}
	raycaster->commit();

	scene_lightmaps.resize(mesh_instances.size());
	scene_lightmap_indices.resize(mesh_instances.size());

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		if (!mesh_instances[i].cast_shadows) {
			no_shadow_meshes.insert(i);
		}
	}

	raycaster->set_mesh_filter(no_shadow_meshes);

	Vector2i atlas_size = Vector2i(-1, -1);
	int atlas_slices = -1;
	if (p_generate_atlas) {
		Error err = _layout_atlas(p_max_texture_size, &atlas_size, &atlas_slices);
		if (err != OK) {
			return BAKE_ERROR_LIGHTMAP_TOO_SMALL;
		}
	}

	if (p_step_function) {
		bool cancelled = p_step_function(0.2, TTR("Generate buffers"), p_bake_userdata, true);
		if (cancelled) {
			return BAKE_ERROR_USER_ABORTED;
		}
	}

	if (_parallel_run(mesh_instances.size(), "Rasterizing meshes", &LightmapperCPU::_generate_buffer, nullptr, p_substep_function)) {
		return BAKE_ERROR_USER_ABORTED;
	}

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		const Size2i &size = mesh_instances[i].size;
		bool has_alpha = false;
		PoolVector<uint8_t> alpha_data;
		alpha_data.resize(size.x * size.y);
		{
			PoolVector<uint8_t>::Write w = alpha_data.write();
			for (unsigned int j = 0; j < scene_lightmap_indices[i].size(); ++j) {
				int idx = scene_lightmap_indices[i][j];
				uint8_t alpha = 0;
				if (idx >= 0) {
					alpha = CLAMP(scene_lightmaps[i][idx].alpha * 255, 0, 255);
					if (alpha < 255) {
						has_alpha = true;
					}
				}
				w[j] = alpha;
			}
		}

		if (has_alpha) {
			Ref<Image> alpha_texture;
			alpha_texture.instance();
			alpha_texture->create(size.x, size.y, false, Image::FORMAT_L8, alpha_data);
			raycaster->set_mesh_alpha_texture(alpha_texture, i);
		}
	}

	albedo_textures.clear();
	emission_textures.clear();

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		if (p_step_function) {
			float p = float(i) / mesh_instances.size();
			bool cancelled = p_step_function(0.2 + p * 0.2, vformat("%s (%d/%d)", TTR("Direct lighting"), i, mesh_instances.size()), p_bake_userdata, false);
			if (cancelled) {
				return BAKE_ERROR_USER_ABORTED;
			}
		}

		if (_parallel_run(scene_lightmaps[i].size(), "Computing direct light", &LightmapperCPU::_compute_direct_light, scene_lightmaps[i].ptr(), p_substep_function)) {
			return BAKE_ERROR_USER_ABORTED;
		}
	}
	raycaster->clear_mesh_filter();

	int n_lit_meshes = 0;
	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		if (mesh_instances[i].generate_lightmap) {
			n_lit_meshes++;
		}
	}

	if (parameters.environment_panorama.is_valid()) {
		parameters.environment_panorama->lock();
	}

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {

		if (!mesh_instances[i].generate_lightmap) {
			continue;
		}

		if (p_step_function) {
			float p = float(i) / n_lit_meshes;
			bool cancelled = p_step_function(0.4 + p * 0.4, vformat("%s (%d/%d)", TTR("Indirect lighting"), i, mesh_instances.size()), p_bake_userdata, false);
			if (cancelled) {
				return BAKE_ERROR_USER_ABORTED;
			}
		}

		if (!scene_lightmaps[i].empty()) {
			if (_parallel_run(scene_lightmaps[i].size(), "Computing indirect light", &LightmapperCPU::_compute_indirect_light, scene_lightmaps[i].ptr(), p_substep_function)) {
				return BAKE_ERROR_USER_ABORTED;
			}
		}
	}

	if (parameters.environment_panorama.is_valid()) {
		parameters.environment_panorama->unlock();
	}

	raycaster.unref(); // Not needed anymore, free some memory.

	LocalVector<LocalVector<Vector3> > lightmaps_data;
	lightmaps_data.resize(mesh_instances.size());

	for (unsigned int i = 0; i < mesh_instances.size(); i++) {
		if (mesh_instances[i].generate_lightmap) {
			const Vector2i size = mesh_instances[i].size;
			lightmaps_data[i].resize(size.x * size.y);
		}
	}

	if (p_step_function) {
		bool cancelled = p_step_function(0.8, TTR("Post processing"), p_bake_userdata, true);
		if (cancelled) {
			return BAKE_ERROR_USER_ABORTED;
		}
	}

	if (_parallel_run(mesh_instances.size(), "Denoise & fix seams", &LightmapperCPU::_post_process, lightmaps_data.ptr(), p_substep_function)) {
		return BAKE_ERROR_USER_ABORTED;
	}

	if (p_generate_atlas) {
		bake_textures.resize(atlas_slices);

		for (int i = 0; i < atlas_slices; i++) {
			Ref<Image> image;
			image.instance();
			image->create(atlas_size.x, atlas_size.y, false, Image::FORMAT_RGBH);
			bake_textures[i] = image;
		}
	} else {
		bake_textures.resize(mesh_instances.size());

		Set<String> used_mesh_names;

		for (unsigned int i = 0; i < mesh_instances.size(); i++) {
			if (!mesh_instances[i].generate_lightmap) {
				continue;
			}

			String mesh_name = mesh_instances[i].node_name;
			if (mesh_name == "" || mesh_name.find(":") != -1 || mesh_name.find("/") != -1) {
				mesh_name = "LightMap";
			}

			if (used_mesh_names.has(mesh_name)) {
				int idx = 2;
				String base = mesh_name;
				while (true) {
					mesh_name = base + itos(idx);
					if (!used_mesh_names.has(mesh_name))
						break;
					idx++;
				}
			}
			used_mesh_names.insert(mesh_name);

			Ref<Image> image;
			image.instance();
			image->create(mesh_instances[i].size.x, mesh_instances[i].size.y, false, Image::FORMAT_RGBH);
			image->set_name(mesh_name);
			bake_textures[i] = image;
		}
	}

	if (p_step_function) {
		bool cancelled = p_step_function(0.9, TTR("Plotting lightmaps"), p_bake_userdata, true);
		if (cancelled) {
			return BAKE_ERROR_USER_ABORTED;
		}
	}

	{
		int j = 0;
		for (unsigned int i = 0; i < mesh_instances.size(); i++) {
			if (!mesh_instances[i].generate_lightmap) {
				continue;
			}

			if (p_generate_atlas) {
				_blit_lightmap(lightmaps_data[i], mesh_instances[i].size, bake_textures[mesh_instances[i].slice], mesh_instances[i].offset.x, mesh_instances[i].offset.y, true);
			} else {
				_blit_lightmap(lightmaps_data[i], mesh_instances[i].size, bake_textures[j], 0, 0, false);
			}
			j++;
		}
	}

	return BAKE_OK;
}

int LightmapperCPU::get_bake_texture_count() const {
	return bake_textures.size();
}

Ref<Image> LightmapperCPU::get_bake_texture(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)bake_textures.size(), Ref<Image>());
	return bake_textures[p_index];
}

int LightmapperCPU::get_bake_mesh_count() const {
	return mesh_instances.size();
}

Variant LightmapperCPU::get_bake_mesh_userdata(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)mesh_instances.size(), Variant());
	return mesh_instances[p_index].data.userdata;
}

Rect2 LightmapperCPU::get_bake_mesh_uv_scale(int p_index) const {
	ERR_FAIL_COND_V(bake_textures.size() == 0, Rect2());
	Rect2 uv_ofs;
	Vector2 atlas_size = Vector2(bake_textures[0]->get_width(), bake_textures[0]->get_height());
	uv_ofs.position = Vector2(mesh_instances[p_index].offset) / atlas_size;
	uv_ofs.size = Vector2(mesh_instances[p_index].size) / atlas_size;
	return uv_ofs;
}

int LightmapperCPU::get_bake_mesh_texture_slice(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)mesh_instances.size(), Variant());
	return mesh_instances[p_index].slice;
}

void LightmapperCPU::add_albedo_texture(Ref<Texture> p_texture) {
	if (p_texture.is_null()) {
		return;
	}

	RID texture_rid = p_texture->get_rid();
	if (!texture_rid.is_valid() || albedo_textures.has(texture_rid)) {
		return;
	}

	Ref<Image> texture_data = p_texture->get_data();

	if (texture_data.is_null()) {
		return;
	}

	if (texture_data->is_compressed()) {
		texture_data->decompress();
	}

	texture_data->convert(Image::FORMAT_RGBA8);

	albedo_textures.insert(texture_rid, texture_data);
}

void LightmapperCPU::add_emission_texture(Ref<Texture> p_texture) {
	if (p_texture.is_null()) {
		return;
	}

	RID texture_rid = p_texture->get_rid();
	if (!texture_rid.is_valid() || emission_textures.has(texture_rid)) {
		return;
	}

	Ref<Image> texture_data = p_texture->get_data();

	if (texture_data.is_null()) {
		return;
	}

	if (texture_data->is_compressed()) {
		texture_data->decompress();
	}

	texture_data->convert(Image::FORMAT_RGBH);

	emission_textures.insert(texture_rid, texture_data);
}

void LightmapperCPU::add_mesh(const MeshData &p_mesh, Vector2i p_size) {
	ERR_FAIL_COND(p_mesh.points.size() == 0);
	ERR_FAIL_COND(p_mesh.points.size() != p_mesh.uv2.size());
	ERR_FAIL_COND(p_mesh.points.size() != p_mesh.normal.size());
	ERR_FAIL_COND(!p_mesh.uv.empty() && p_mesh.points.size() != p_mesh.uv.size());
	ERR_FAIL_COND(p_mesh.surface_facecounts.size() != p_mesh.albedo.size());
	ERR_FAIL_COND(p_mesh.surface_facecounts.size() != p_mesh.emission.size());

	MeshInstance mi;
	mi.data = p_mesh;
	mi.size = p_size;
	mi.generate_lightmap = true;
	mi.cast_shadows = true;
	mi.node_name = "";

	Dictionary userdata = p_mesh.userdata;
	if (userdata.has("cast_shadows")) {
		mi.cast_shadows = userdata["cast_shadows"];
	}
	if (userdata.has("generate_lightmap")) {
		mi.generate_lightmap = userdata["generate_lightmap"];
	}
	if (userdata.has("node_name")) {
		mi.node_name = userdata["node_name"];
	}

	mesh_instances.push_back(mi);
}

void LightmapperCPU::add_directional_light(bool p_bake_direct, const Vector3 &p_direction, const Color &p_color, float p_energy, float p_indirect_multiplier) {
	Light l;
	l.type = LIGHT_TYPE_DIRECTIONAL;
	l.direction = p_direction;
	l.color = p_color;
	l.energy = p_energy;
	l.indirect_multiplier = p_indirect_multiplier;
	l.bake_direct = p_bake_direct;
	lights.push_back(l);
}

void LightmapperCPU::add_omni_light(bool p_bake_direct, const Vector3 &p_position, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_range, float p_attenuation) {
	Light l;
	l.type = LIGHT_TYPE_OMNI;
	l.position = p_position;
	l.range = p_range;
	l.attenuation = p_attenuation;
	l.color = p_color;
	l.energy = p_energy;
	l.indirect_multiplier = p_indirect_multiplier;
	l.bake_direct = p_bake_direct;
	lights.push_back(l);
}

void LightmapperCPU::add_spot_light(bool p_bake_direct, const Vector3 &p_position, const Vector3 p_direction, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_range, float p_attenuation, float p_spot_angle, float p_spot_attenuation) {
	Light l;
	l.type = LIGHT_TYPE_SPOT;
	l.position = p_position;
	l.direction = p_direction;
	l.range = p_range;
	l.attenuation = p_attenuation;
	l.spot_angle = Math::deg2rad(p_spot_angle);
	l.spot_attenuation = p_spot_attenuation;
	l.color = p_color;
	l.energy = p_energy;
	l.indirect_multiplier = p_indirect_multiplier;
	l.bake_direct = p_bake_direct;
	lights.push_back(l);
}

LightmapperCPU::LightmapperCPU() {
	thread_progress = 0;
	thread_cancelled = false;
}
