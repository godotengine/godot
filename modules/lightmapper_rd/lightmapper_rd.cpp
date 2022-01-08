/*************************************************************************/
/*  lightmapper_rd.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "lightmapper_rd.h"
#include "core/config/project_settings.h"
#include "core/math/geometry_2d.h"
#include "lm_blendseams.glsl.gen.h"
#include "lm_compute.glsl.gen.h"
#include "lm_raster.glsl.gen.h"
#include "servers/rendering/rendering_device_binds.h"

//uncomment this if you want to see textures from all the process saved
//#define DEBUG_TEXTURES

void LightmapperRD::add_mesh(const MeshData &p_mesh) {
	ERR_FAIL_COND(p_mesh.albedo_on_uv2.is_null() || p_mesh.albedo_on_uv2->is_empty());
	ERR_FAIL_COND(p_mesh.emission_on_uv2.is_null() || p_mesh.emission_on_uv2->is_empty());
	ERR_FAIL_COND(p_mesh.albedo_on_uv2->get_width() != p_mesh.emission_on_uv2->get_width());
	ERR_FAIL_COND(p_mesh.albedo_on_uv2->get_height() != p_mesh.emission_on_uv2->get_height());
	ERR_FAIL_COND(p_mesh.points.size() == 0);
	MeshInstance mi;
	mi.data = p_mesh;
	mesh_instances.push_back(mi);
}

void LightmapperRD::add_directional_light(bool p_static, const Vector3 &p_direction, const Color &p_color, float p_energy, float p_angular_distance) {
	Light l;
	l.type = LIGHT_TYPE_DIRECTIONAL;
	l.direction[0] = p_direction.x;
	l.direction[1] = p_direction.y;
	l.direction[2] = p_direction.z;
	l.color[0] = p_color.r;
	l.color[1] = p_color.g;
	l.color[2] = p_color.b;
	l.energy = p_energy;
	l.static_bake = p_static;
	l.size = p_angular_distance;
	lights.push_back(l);
}

void LightmapperRD::add_omni_light(bool p_static, const Vector3 &p_position, const Color &p_color, float p_energy, float p_range, float p_attenuation, float p_size) {
	Light l;
	l.type = LIGHT_TYPE_OMNI;
	l.position[0] = p_position.x;
	l.position[1] = p_position.y;
	l.position[2] = p_position.z;
	l.range = p_range;
	l.attenuation = p_attenuation;
	l.color[0] = p_color.r;
	l.color[1] = p_color.g;
	l.color[2] = p_color.b;
	l.energy = p_energy;
	l.static_bake = p_static;
	l.size = p_size;
	lights.push_back(l);
}

void LightmapperRD::add_spot_light(bool p_static, const Vector3 &p_position, const Vector3 p_direction, const Color &p_color, float p_energy, float p_range, float p_attenuation, float p_spot_angle, float p_spot_attenuation, float p_size) {
	Light l;
	l.type = LIGHT_TYPE_SPOT;
	l.position[0] = p_position.x;
	l.position[1] = p_position.y;
	l.position[2] = p_position.z;
	l.direction[0] = p_direction.x;
	l.direction[1] = p_direction.y;
	l.direction[2] = p_direction.z;
	l.range = p_range;
	l.attenuation = p_attenuation;
	l.cos_spot_angle = Math::cos(Math::deg2rad(p_spot_angle));
	l.inv_spot_attenuation = 1.0f / p_spot_attenuation;
	l.color[0] = p_color.r;
	l.color[1] = p_color.g;
	l.color[2] = p_color.b;
	l.energy = p_energy;
	l.static_bake = p_static;
	l.size = p_size;
	lights.push_back(l);
}

void LightmapperRD::add_probe(const Vector3 &p_position) {
	Probe probe;
	probe.position[0] = p_position.x;
	probe.position[1] = p_position.y;
	probe.position[2] = p_position.z;
	probe.position[3] = 0;
	probe_positions.push_back(probe);
}

void LightmapperRD::_plot_triangle_into_triangle_index_list(int p_size, const Vector3i &p_ofs, const AABB &p_bounds, const Vector3 p_points[3], uint32_t p_triangle_index, LocalVector<TriangleSort> &triangles, uint32_t p_grid_size) {
	int half_size = p_size / 2;

	for (int i = 0; i < 8; i++) {
		AABB aabb = p_bounds;
		aabb.size *= 0.5;
		Vector3i n = p_ofs;

		if (i & 1) {
			aabb.position.x += aabb.size.x;
			n.x += half_size;
		}
		if (i & 2) {
			aabb.position.y += aabb.size.y;
			n.y += half_size;
		}
		if (i & 4) {
			aabb.position.z += aabb.size.z;
			n.z += half_size;
		}

		{
			Vector3 qsize = aabb.size * 0.5; //quarter size, for fast aabb test

			if (!Geometry3D::triangle_box_overlap(aabb.position + qsize, qsize, p_points)) {
				//does not fit in child, go on
				continue;
			}
		}

		if (half_size == 1) {
			//got to the end
			TriangleSort ts;
			ts.cell_index = n.x + (n.y * p_grid_size) + (n.z * p_grid_size * p_grid_size);
			ts.triangle_index = p_triangle_index;
			triangles.push_back(ts);
		} else {
			_plot_triangle_into_triangle_index_list(half_size, n, aabb, p_points, p_triangle_index, triangles, p_grid_size);
		}
	}
}

Lightmapper::BakeError LightmapperRD::_blit_meshes_into_atlas(int p_max_texture_size, Vector<Ref<Image>> &albedo_images, Vector<Ref<Image>> &emission_images, AABB &bounds, Size2i &atlas_size, int &atlas_slices, BakeStepFunc p_step_function, void *p_bake_userdata) {
	Vector<Size2i> sizes;

	for (int m_i = 0; m_i < mesh_instances.size(); m_i++) {
		MeshInstance &mi = mesh_instances.write[m_i];
		Size2i s = Size2i(mi.data.albedo_on_uv2->get_width(), mi.data.albedo_on_uv2->get_height());
		sizes.push_back(s);
		atlas_size.width = MAX(atlas_size.width, s.width + 2);
		atlas_size.height = MAX(atlas_size.height, s.height + 2);
	}

	int max = nearest_power_of_2_templated(atlas_size.width);
	max = MAX(max, nearest_power_of_2_templated(atlas_size.height));

	if (max > p_max_texture_size) {
		return BAKE_ERROR_LIGHTMAP_TOO_SMALL;
	}

	if (p_step_function) {
		p_step_function(0.1, TTR("Determining optimal atlas size"), p_bake_userdata, true);
	}

	atlas_size = Size2i(max, max);

	Size2i best_atlas_size;
	int best_atlas_slices = 0;
	int best_atlas_memory = 0x7FFFFFFF;
	Vector<Vector3i> best_atlas_offsets;

	//determine best texture array atlas size by bruteforce fitting
	while (atlas_size.x <= p_max_texture_size && atlas_size.y <= p_max_texture_size) {
		Vector<Vector2i> source_sizes;
		Vector<int> source_indices;
		source_sizes.resize(sizes.size());
		source_indices.resize(sizes.size());
		for (int i = 0; i < source_indices.size(); i++) {
			source_sizes.write[i] = sizes[i] + Vector2i(2, 2); // Add padding between lightmaps
			source_indices.write[i] = i;
		}
		Vector<Vector3i> atlas_offsets;
		atlas_offsets.resize(source_sizes.size());

		int slices = 0;

		while (source_sizes.size() > 0) {
			Vector<Vector3i> offsets = Geometry2D::partial_pack_rects(source_sizes, atlas_size);
			Vector<int> new_indices;
			Vector<Vector2i> new_sources;
			for (int i = 0; i < offsets.size(); i++) {
				Vector3i ofs = offsets[i];
				int sidx = source_indices[i];
				if (ofs.z > 0) {
					//valid
					ofs.z = slices;
					atlas_offsets.write[sidx] = ofs + Vector3i(1, 1, 0); // Center lightmap in the reserved oversized region
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
		if (mem_used < best_atlas_memory) {
			best_atlas_size = atlas_size;
			best_atlas_offsets = atlas_offsets;
			best_atlas_slices = slices;
			best_atlas_memory = mem_used;
		}

		if (atlas_size.width == atlas_size.height) {
			atlas_size.width *= 2;
		} else {
			atlas_size.height *= 2;
		}
	}
	atlas_size = best_atlas_size;
	atlas_slices = best_atlas_slices;

	// apply the offsets and slice to all images, and also blit albedo and emission
	albedo_images.resize(atlas_slices);
	emission_images.resize(atlas_slices);

	if (p_step_function) {
		p_step_function(0.2, TTR("Blitting albedo and emission"), p_bake_userdata, true);
	}

	for (int i = 0; i < atlas_slices; i++) {
		Ref<Image> albedo;
		albedo.instantiate();
		albedo->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBA8);
		albedo->set_as_black();
		albedo_images.write[i] = albedo;

		Ref<Image> emission;
		emission.instantiate();
		emission->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH);
		emission->set_as_black();
		emission_images.write[i] = emission;
	}

	//assign uv positions

	for (int m_i = 0; m_i < mesh_instances.size(); m_i++) {
		MeshInstance &mi = mesh_instances.write[m_i];
		mi.offset.x = best_atlas_offsets[m_i].x;
		mi.offset.y = best_atlas_offsets[m_i].y;
		mi.slice = best_atlas_offsets[m_i].z;
		albedo_images.write[mi.slice]->blit_rect(mi.data.albedo_on_uv2, Rect2(Vector2(), Size2i(mi.data.albedo_on_uv2->get_width(), mi.data.albedo_on_uv2->get_height())), mi.offset);
		emission_images.write[mi.slice]->blit_rect(mi.data.emission_on_uv2, Rect2(Vector2(), Size2i(mi.data.emission_on_uv2->get_width(), mi.data.emission_on_uv2->get_height())), mi.offset);
	}

	return BAKE_OK;
}

void LightmapperRD::_create_acceleration_structures(RenderingDevice *rd, Size2i atlas_size, int atlas_slices, AABB &bounds, int grid_size, Vector<Probe> &probe_positions, GenerateProbes p_generate_probes, Vector<int> &slice_triangle_count, Vector<int> &slice_seam_count, RID &vertex_buffer, RID &triangle_buffer, RID &lights_buffer, RID &triangle_cell_indices_buffer, RID &probe_positions_buffer, RID &grid_texture, RID &seams_buffer, BakeStepFunc p_step_function, void *p_bake_userdata) {
	HashMap<Vertex, uint32_t, VertexHash> vertex_map;

	//fill triangles array and vertex array
	LocalVector<Triangle> triangles;
	LocalVector<Vertex> vertex_array;
	LocalVector<Seam> seams;

	slice_triangle_count.resize(atlas_slices);
	slice_seam_count.resize(atlas_slices);

	for (int i = 0; i < atlas_slices; i++) {
		slice_triangle_count.write[i] = 0;
		slice_seam_count.write[i] = 0;
	}

	bounds = AABB();

	for (int m_i = 0; m_i < mesh_instances.size(); m_i++) {
		if (p_step_function) {
			float p = float(m_i + 1) / mesh_instances.size() * 0.1;
			p_step_function(0.3 + p, vformat(TTR("Plotting mesh into acceleration structure %d/%d"), m_i + 1, mesh_instances.size()), p_bake_userdata, false);
		}

		HashMap<Edge, EdgeUV2, EdgeHash> edges;

		MeshInstance &mi = mesh_instances.write[m_i];

		Vector2 uv_scale = Vector2(mi.data.albedo_on_uv2->get_width(), mi.data.albedo_on_uv2->get_height()) / Vector2(atlas_size);
		Vector2 uv_offset = Vector2(mi.offset) / Vector2(atlas_size);
		if (m_i == 0) {
			bounds.position = mi.data.points[0];
		}

		for (int i = 0; i < mi.data.points.size(); i += 3) {
			Vector3 vtxs[3] = { mi.data.points[i + 0], mi.data.points[i + 1], mi.data.points[i + 2] };
			Vector2 uvs[3] = { mi.data.uv2[i + 0] * uv_scale + uv_offset, mi.data.uv2[i + 1] * uv_scale + uv_offset, mi.data.uv2[i + 2] * uv_scale + uv_offset };
			Vector3 normal[3] = { mi.data.normal[i + 0], mi.data.normal[i + 1], mi.data.normal[i + 2] };

			AABB taabb;
			Triangle t;
			t.slice = mi.slice;
			for (int k = 0; k < 3; k++) {
				bounds.expand_to(vtxs[k]);

				Vertex v;
				v.position[0] = vtxs[k].x;
				v.position[1] = vtxs[k].y;
				v.position[2] = vtxs[k].z;
				v.uv[0] = uvs[k].x;
				v.uv[1] = uvs[k].y;
				v.normal_xy[0] = normal[k].x;
				v.normal_xy[1] = normal[k].y;
				v.normal_z = normal[k].z;

				uint32_t *indexptr = vertex_map.getptr(v);

				if (indexptr) {
					t.indices[k] = *indexptr;
				} else {
					uint32_t new_index = vertex_map.size();
					t.indices[k] = new_index;
					vertex_map[v] = new_index;
					vertex_array.push_back(v);
				}

				if (k == 0) {
					taabb.position = vtxs[k];
				} else {
					taabb.expand_to(vtxs[k]);
				}
			}

			//compute seams that will need to be blended later
			for (int k = 0; k < 3; k++) {
				int n = (k + 1) % 3;

				Edge edge(vtxs[k], vtxs[n], normal[k], normal[n]);
				Vector2i edge_indices(t.indices[k], t.indices[n]);
				EdgeUV2 uv2(uvs[k], uvs[n], edge_indices);

				if (edge.b == edge.a) {
					continue; //degenerate, somehow
				}
				if (edge.b < edge.a) {
					SWAP(edge.a, edge.b);
					SWAP(edge.na, edge.nb);
					SWAP(uv2.a, uv2.b);
					SWAP(edge_indices.x, edge_indices.y);
				}

				EdgeUV2 *euv2 = edges.getptr(edge);
				if (!euv2) {
					edges[edge] = uv2;
				} else {
					if (*euv2 == uv2) {
						continue; // seam shared UV space, no need to blend
					}
					if (euv2->seam_found) {
						continue; //bad geometry
					}

					Seam seam;
					seam.a = edge_indices;
					seam.b = euv2->indices;
					seam.slice = mi.slice;
					seams.push_back(seam);
					slice_seam_count.write[mi.slice]++;
					euv2->seam_found = true;
				}
			}

			t.min_bounds[0] = taabb.position.x;
			t.min_bounds[1] = taabb.position.y;
			t.min_bounds[2] = taabb.position.z;
			t.max_bounds[0] = taabb.position.x + MAX(taabb.size.x, 0.0001);
			t.max_bounds[1] = taabb.position.y + MAX(taabb.size.y, 0.0001);
			t.max_bounds[2] = taabb.position.z + MAX(taabb.size.z, 0.0001);
			t.pad0 = t.pad1 = 0; //make valgrind not complain
			triangles.push_back(t);
			slice_triangle_count.write[t.slice]++;
		}
	}

	//also consider probe positions for bounds
	for (int i = 0; i < probe_positions.size(); i++) {
		Vector3 pp(probe_positions[i].position[0], probe_positions[i].position[1], probe_positions[i].position[2]);
		bounds.expand_to(pp);
	}
	bounds.grow_by(0.1); //grow a bit to avoid numerical error

	triangles.sort(); //sort by slice
	seams.sort();

	if (p_step_function) {
		p_step_function(0.4, TTR("Optimizing acceleration structure"), p_bake_userdata, true);
	}

	//fill list of triangles in grid
	LocalVector<TriangleSort> triangle_sort;
	for (uint32_t i = 0; i < triangles.size(); i++) {
		const Triangle &t = triangles[i];
		Vector3 face[3] = {
			Vector3(vertex_array[t.indices[0]].position[0], vertex_array[t.indices[0]].position[1], vertex_array[t.indices[0]].position[2]),
			Vector3(vertex_array[t.indices[1]].position[0], vertex_array[t.indices[1]].position[1], vertex_array[t.indices[1]].position[2]),
			Vector3(vertex_array[t.indices[2]].position[0], vertex_array[t.indices[2]].position[1], vertex_array[t.indices[2]].position[2])
		};
		_plot_triangle_into_triangle_index_list(grid_size, Vector3i(), bounds, face, i, triangle_sort, grid_size);
	}
	//sort it
	triangle_sort.sort();

	Vector<uint32_t> triangle_indices;
	triangle_indices.resize(triangle_sort.size());
	Vector<uint32_t> grid_indices;
	grid_indices.resize(grid_size * grid_size * grid_size * 2);
	memset(grid_indices.ptrw(), 0, grid_indices.size() * sizeof(uint32_t));
	Vector<bool> solid;
	solid.resize(grid_size * grid_size * grid_size);
	memset(solid.ptrw(), 0, solid.size() * sizeof(bool));

	{
		uint32_t *tiw = triangle_indices.ptrw();
		uint32_t last_cell = 0xFFFFFFFF;
		uint32_t *giw = grid_indices.ptrw();
		bool *solidw = solid.ptrw();
		for (uint32_t i = 0; i < triangle_sort.size(); i++) {
			uint32_t cell = triangle_sort[i].cell_index;
			if (cell != last_cell) {
				//cell changed, update pointer to indices
				giw[cell * 2 + 1] = i;
				solidw[cell] = true;
			}
			tiw[i] = triangle_sort[i].triangle_index;
			giw[cell * 2]++; //update counter
			last_cell = cell;
		}
	}
#if 0
	for (int i = 0; i < grid_size; i++) {
		for (int j = 0; j < grid_size; j++) {
			for (int k = 0; k < grid_size; k++) {
				uint32_t index = i * (grid_size * grid_size) + j * grid_size + k;
				grid_indices.write[index * 2] = float(i) / grid_size * 255;
				grid_indices.write[index * 2 + 1] = float(j) / grid_size * 255;
			}
		}
	}
#endif

#if 0
	for (int i = 0; i < grid_size; i++) {
		Vector<uint8_t> grid_usage;
		grid_usage.resize(grid_size * grid_size);
		for (int j = 0; j < grid_usage.size(); j++) {
			uint32_t ofs = i * grid_size * grid_size + j;
			uint32_t count = grid_indices[ofs * 2];
			grid_usage.write[j] = count > 0 ? 255 : 0;
		}

		Ref<Image> img;
		img.instantiate();
		img->create(grid_size, grid_size, false, Image::FORMAT_L8, grid_usage);
		img->save_png("res://grid_layer_" + itos(1000 + i).substr(1, 3) + ".png");
	}
#endif

	/*****************************/
	/*** CREATE GPU STRUCTURES ***/
	/*****************************/

	lights.sort();

	Vector<Vector2i> seam_buffer_vec;
	seam_buffer_vec.resize(seams.size() * 2);
	for (uint32_t i = 0; i < seams.size(); i++) {
		seam_buffer_vec.write[i * 2 + 0] = seams[i].a;
		seam_buffer_vec.write[i * 2 + 1] = seams[i].b;
	}

	{ //buffers
		Vector<uint8_t> vb = vertex_array.to_byte_array();
		vertex_buffer = rd->storage_buffer_create(vb.size(), vb);

		Vector<uint8_t> tb = triangles.to_byte_array();
		triangle_buffer = rd->storage_buffer_create(tb.size(), tb);

		Vector<uint8_t> tib = triangle_indices.to_byte_array();
		triangle_cell_indices_buffer = rd->storage_buffer_create(tib.size(), tib);

		Vector<uint8_t> lb = lights.to_byte_array();
		if (lb.size() == 0) {
			lb.resize(sizeof(Light)); //even if no lights, the buffer must exist
		}
		lights_buffer = rd->storage_buffer_create(lb.size(), lb);

		Vector<uint8_t> sb = seam_buffer_vec.to_byte_array();
		if (sb.size() == 0) {
			sb.resize(sizeof(Vector2i) * 2); //even if no seams, the buffer must exist
		}
		seams_buffer = rd->storage_buffer_create(sb.size(), sb);

		Vector<uint8_t> pb = probe_positions.to_byte_array();
		if (pb.size() == 0) {
			pb.resize(sizeof(Probe));
		}
		probe_positions_buffer = rd->storage_buffer_create(pb.size(), pb);
	}

	{ //grid

		RD::TextureFormat tf;
		tf.width = grid_size;
		tf.height = grid_size;
		tf.depth = grid_size;
		tf.texture_type = RD::TEXTURE_TYPE_3D;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;

		Vector<Vector<uint8_t>> texdata;
		texdata.resize(1);
		//grid and indices
		tf.format = RD::DATA_FORMAT_R32G32_UINT;
		texdata.write[0] = grid_indices.to_byte_array();
		grid_texture = rd->texture_create(tf, RD::TextureView(), texdata);
	}
}

void LightmapperRD::_raster_geometry(RenderingDevice *rd, Size2i atlas_size, int atlas_slices, int grid_size, AABB bounds, float p_bias, Vector<int> slice_triangle_count, RID position_tex, RID unocclude_tex, RID normal_tex, RID raster_depth_buffer, RID rasterize_shader, RID raster_base_uniform) {
	Vector<RID> framebuffers;

	for (int i = 0; i < atlas_slices; i++) {
		RID slice_pos_tex = rd->texture_create_shared_from_slice(RD::TextureView(), position_tex, i, 0);
		RID slice_unoc_tex = rd->texture_create_shared_from_slice(RD::TextureView(), unocclude_tex, i, 0);
		RID slice_norm_tex = rd->texture_create_shared_from_slice(RD::TextureView(), normal_tex, i, 0);
		Vector<RID> fb;
		fb.push_back(slice_pos_tex);
		fb.push_back(slice_norm_tex);
		fb.push_back(slice_unoc_tex);
		fb.push_back(raster_depth_buffer);
		framebuffers.push_back(rd->framebuffer_create(fb));
	}

	RD::PipelineDepthStencilState ds;
	ds.enable_depth_test = true;
	ds.enable_depth_write = true;
	ds.depth_compare_operator = RD::COMPARE_OP_LESS; //so it does render same pixel twice

	RID raster_pipeline = rd->render_pipeline_create(rasterize_shader, rd->framebuffer_get_format(framebuffers[0]), RD::INVALID_FORMAT_ID, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(3), 0);
	RID raster_pipeline_wire;
	{
		RD::PipelineRasterizationState rw;
		rw.wireframe = true;
		raster_pipeline_wire = rd->render_pipeline_create(rasterize_shader, rd->framebuffer_get_format(framebuffers[0]), RD::INVALID_FORMAT_ID, RD::RENDER_PRIMITIVE_TRIANGLES, rw, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(3), 0);
	}

	uint32_t triangle_offset = 0;
	Vector<Color> clear_colors;
	clear_colors.push_back(Color(0, 0, 0, 0));
	clear_colors.push_back(Color(0, 0, 0, 0));
	clear_colors.push_back(Color(0, 0, 0, 0));

	for (int i = 0; i < atlas_slices; i++) {
		RasterPushConstant raster_push_constant;
		raster_push_constant.atlas_size[0] = atlas_size.x;
		raster_push_constant.atlas_size[1] = atlas_size.y;
		raster_push_constant.base_triangle = triangle_offset;
		raster_push_constant.to_cell_offset[0] = bounds.position.x;
		raster_push_constant.to_cell_offset[1] = bounds.position.y;
		raster_push_constant.to_cell_offset[2] = bounds.position.z;
		raster_push_constant.bias = p_bias;
		raster_push_constant.to_cell_size[0] = (1.0 / bounds.size.x) * float(grid_size);
		raster_push_constant.to_cell_size[1] = (1.0 / bounds.size.y) * float(grid_size);
		raster_push_constant.to_cell_size[2] = (1.0 / bounds.size.z) * float(grid_size);
		raster_push_constant.grid_size[0] = grid_size;
		raster_push_constant.grid_size[1] = grid_size;
		raster_push_constant.grid_size[2] = grid_size;
		raster_push_constant.uv_offset[0] = 0;
		raster_push_constant.uv_offset[1] = 0;

		RD::DrawListID draw_list = rd->draw_list_begin(framebuffers[i], RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		//draw opaque
		rd->draw_list_bind_render_pipeline(draw_list, raster_pipeline);
		rd->draw_list_bind_uniform_set(draw_list, raster_base_uniform, 0);
		rd->draw_list_set_push_constant(draw_list, &raster_push_constant, sizeof(RasterPushConstant));
		rd->draw_list_draw(draw_list, false, 1, slice_triangle_count[i] * 3);
		//draw wire
		rd->draw_list_bind_render_pipeline(draw_list, raster_pipeline_wire);
		rd->draw_list_bind_uniform_set(draw_list, raster_base_uniform, 0);
		rd->draw_list_set_push_constant(draw_list, &raster_push_constant, sizeof(RasterPushConstant));
		rd->draw_list_draw(draw_list, false, 1, slice_triangle_count[i] * 3);

		rd->draw_list_end();

		triangle_offset += slice_triangle_count[i];
	}
}

LightmapperRD::BakeError LightmapperRD::_dilate(RenderingDevice *rd, Ref<RDShaderFile> &compute_shader, RID &compute_base_uniform_set, PushConstant &push_constant, RID &source_light_tex, RID &dest_light_tex, const Size2i &atlas_size, int atlas_slices) {
	Vector<RD::Uniform> uniforms;
	{
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 0;
			u.ids.push_back(dest_light_tex);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;
			u.ids.push_back(source_light_tex);
			uniforms.push_back(u);
		}
	}

	RID compute_shader_dilate = rd->shader_create_from_spirv(compute_shader->get_spirv_stages("dilate"));
	ERR_FAIL_COND_V(compute_shader_dilate.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES); //internal check, should not happen
	RID compute_shader_dilate_pipeline = rd->compute_pipeline_create(compute_shader_dilate);

	RID dilate_uniform_set = rd->uniform_set_create(uniforms, compute_shader_dilate, 1);

	RD::ComputeListID compute_list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(compute_list, compute_shader_dilate_pipeline);
	rd->compute_list_bind_uniform_set(compute_list, compute_base_uniform_set, 0);
	rd->compute_list_bind_uniform_set(compute_list, dilate_uniform_set, 1);
	push_constant.region_ofs[0] = 0;
	push_constant.region_ofs[1] = 0;
	Vector3i group_size((atlas_size.x - 1) / 8 + 1, (atlas_size.y - 1) / 8 + 1, 1); //restore group size

	for (int i = 0; i < atlas_slices; i++) {
		push_constant.atlas_slice = i;
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
		rd->compute_list_dispatch(compute_list, group_size.x, group_size.y, group_size.z);
		//no barrier, let them run all together
	}
	rd->compute_list_end();
	rd->free(compute_shader_dilate);

#ifdef DEBUG_TEXTURES
	for (int i = 0; i < atlas_slices; i++) {
		Vector<uint8_t> s = rd->texture_get_data(light_accum_tex, i);
		Ref<Image> img;
		img.instantiate();
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);
		img->convert(Image::FORMAT_RGBA8);
		img->save_png("res://5_dilated_" + itos(i) + ".png");
	}
#endif
	return BAKE_OK;
}

LightmapperRD::BakeError LightmapperRD::bake(BakeQuality p_quality, bool p_use_denoiser, int p_bounces, float p_bias, int p_max_texture_size, bool p_bake_sh, GenerateProbes p_generate_probes, const Ref<Image> &p_environment_panorama, const Basis &p_environment_transform, BakeStepFunc p_step_function, void *p_bake_userdata) {
	if (p_step_function) {
		p_step_function(0.0, TTR("Begin Bake"), p_bake_userdata, true);
	}
	bake_textures.clear();
	int grid_size = 128;

	/* STEP 1: Fetch material textures and compute the bounds */

	AABB bounds;
	Size2i atlas_size;
	int atlas_slices;
	Vector<Ref<Image>> albedo_images;
	Vector<Ref<Image>> emission_images;

	BakeError bake_error = _blit_meshes_into_atlas(p_max_texture_size, albedo_images, emission_images, bounds, atlas_size, atlas_slices, p_step_function, p_bake_userdata);
	if (bake_error != BAKE_OK) {
		return bake_error;
	}

#ifdef DEBUG_TEXTURES
	for (int i = 0; i < atlas_slices; i++) {
		albedo_images[i]->save_png("res://0_albedo_" + itos(i) + ".png");
		emission_images[i]->save_png("res://0_emission_" + itos(i) + ".png");
	}
#endif

	RenderingDevice *rd = RenderingDevice::get_singleton()->create_local_device();

	RID albedo_array_tex;
	RID emission_array_tex;
	RID normal_tex;
	RID position_tex;
	RID unocclude_tex;
	RID light_source_tex;
	RID light_dest_tex;
	RID light_accum_tex;
	RID light_accum_tex2;
	RID light_primary_dynamic_tex;
	RID light_environment_tex;

#define FREE_TEXTURES                    \
	rd->free(albedo_array_tex);          \
	rd->free(emission_array_tex);        \
	rd->free(normal_tex);                \
	rd->free(position_tex);              \
	rd->free(unocclude_tex);             \
	rd->free(light_source_tex);          \
	rd->free(light_accum_tex2);          \
	rd->free(light_accum_tex);           \
	rd->free(light_primary_dynamic_tex); \
	rd->free(light_environment_tex);

	{ // create all textures

		Vector<Vector<uint8_t>> albedo_data;
		Vector<Vector<uint8_t>> emission_data;
		for (int i = 0; i < atlas_slices; i++) {
			albedo_data.push_back(albedo_images[i]->get_data());
			emission_data.push_back(emission_images[i]->get_data());
		}

		RD::TextureFormat tf;
		tf.width = atlas_size.width;
		tf.height = atlas_size.height;
		tf.array_layers = atlas_slices;
		tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;

		albedo_array_tex = rd->texture_create(tf, RD::TextureView(), albedo_data);

		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;

		emission_array_tex = rd->texture_create(tf, RD::TextureView(), emission_data);

		//this will be rastered to
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		normal_tex = rd->texture_create(tf, RD::TextureView());
		tf.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
		position_tex = rd->texture_create(tf, RD::TextureView());
		unocclude_tex = rd->texture_create(tf, RD::TextureView());

		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;

		light_source_tex = rd->texture_create(tf, RD::TextureView());
		rd->texture_clear(light_source_tex, Color(0, 0, 0, 0), 0, 1, 0, atlas_slices);
		light_primary_dynamic_tex = rd->texture_create(tf, RD::TextureView());
		rd->texture_clear(light_primary_dynamic_tex, Color(0, 0, 0, 0), 0, 1, 0, atlas_slices);

		if (p_bake_sh) {
			tf.array_layers *= 4;
		}
		light_accum_tex = rd->texture_create(tf, RD::TextureView());
		rd->texture_clear(light_accum_tex, Color(0, 0, 0, 0), 0, 1, 0, tf.array_layers);
		light_dest_tex = rd->texture_create(tf, RD::TextureView());
		rd->texture_clear(light_dest_tex, Color(0, 0, 0, 0), 0, 1, 0, tf.array_layers);
		light_accum_tex2 = light_dest_tex;

		//env
		{
			Ref<Image> panorama_tex;
			if (p_environment_panorama.is_valid()) {
				panorama_tex = p_environment_panorama;
				panorama_tex->convert(Image::FORMAT_RGBAF);
			} else {
				panorama_tex.instantiate();
				panorama_tex->create(8, 8, false, Image::FORMAT_RGBAF);
				panorama_tex->fill(Color(0, 0, 0, 1));
			}

			RD::TextureFormat tfp;
			tfp.width = panorama_tex->get_width();
			tfp.height = panorama_tex->get_height();
			tfp.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
			tfp.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;

			Vector<Vector<uint8_t>> tdata;
			tdata.push_back(panorama_tex->get_data());
			light_environment_tex = rd->texture_create(tfp, RD::TextureView(), tdata);

#ifdef DEBUG_TEXTURES
			panorama_tex->save_exr("res://0_panorama.exr", false);
#endif
		}
	}

	/* STEP 2: create the acceleration structure for the GPU*/

	Vector<int> slice_triangle_count;
	RID vertex_buffer;
	RID triangle_buffer;
	RID lights_buffer;
	RID triangle_cell_indices_buffer;
	RID grid_texture;
	RID seams_buffer;
	RID probe_positions_buffer;

	Vector<int> slice_seam_count;

#define FREE_BUFFERS                        \
	rd->free(vertex_buffer);                \
	rd->free(triangle_buffer);              \
	rd->free(lights_buffer);                \
	rd->free(triangle_cell_indices_buffer); \
	rd->free(grid_texture);                 \
	rd->free(seams_buffer);                 \
	rd->free(probe_positions_buffer);

	_create_acceleration_structures(rd, atlas_size, atlas_slices, bounds, grid_size, probe_positions, p_generate_probes, slice_triangle_count, slice_seam_count, vertex_buffer, triangle_buffer, lights_buffer, triangle_cell_indices_buffer, probe_positions_buffer, grid_texture, seams_buffer, p_step_function, p_bake_userdata);

	if (p_step_function) {
		p_step_function(0.47, TTR("Preparing shaders"), p_bake_userdata, true);
	}

	//shaders
	Ref<RDShaderFile> raster_shader;
	raster_shader.instantiate();
	Error err = raster_shader->parse_versions_from_text(lm_raster_shader_glsl);
	if (err != OK) {
		raster_shader->print_errors("raster_shader");

		FREE_TEXTURES
		FREE_BUFFERS

		memdelete(rd);
	}
	ERR_FAIL_COND_V(err != OK, BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES);

	RID rasterize_shader = rd->shader_create_from_spirv(raster_shader->get_spirv_stages());

	ERR_FAIL_COND_V(rasterize_shader.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES); //this is a bug check, though, should not happen

	RID sampler;
	{
		RD::SamplerState s;
		s.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		s.min_filter = RD::SAMPLER_FILTER_LINEAR;
		s.max_lod = 0;

		sampler = rd->sampler_create(s);
	}

	Vector<RD::Uniform> base_uniforms;
	{
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 1;
			u.ids.push_back(vertex_buffer);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 2;
			u.ids.push_back(triangle_buffer);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 3;
			u.ids.push_back(triangle_cell_indices_buffer);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 4;
			u.ids.push_back(lights_buffer);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 5;
			u.ids.push_back(seams_buffer);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 6;
			u.ids.push_back(probe_positions_buffer);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 7;
			u.ids.push_back(grid_texture);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 8;
			u.ids.push_back(albedo_array_tex);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 9;
			u.ids.push_back(emission_array_tex);
			base_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 10;
			u.ids.push_back(sampler);
			base_uniforms.push_back(u);
		}
	}

	RID raster_base_uniform = rd->uniform_set_create(base_uniforms, rasterize_shader, 0);
	RID raster_depth_buffer;
	{
		RD::TextureFormat tf;
		tf.width = atlas_size.width;
		tf.height = atlas_size.height;
		tf.depth = 1;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		tf.format = RD::DATA_FORMAT_D32_SFLOAT;

		raster_depth_buffer = rd->texture_create(tf, RD::TextureView());
	}

	rd->submit();
	rd->sync();

	/* STEP 3: Raster the geometry to UV2 coords in the atlas textures GPU*/

	_raster_geometry(rd, atlas_size, atlas_slices, grid_size, bounds, p_bias, slice_triangle_count, position_tex, unocclude_tex, normal_tex, raster_depth_buffer, rasterize_shader, raster_base_uniform);

#ifdef DEBUG_TEXTURES

	for (int i = 0; i < atlas_slices; i++) {
		Vector<uint8_t> s = rd->texture_get_data(position_tex, i);
		Ref<Image> img;
		img.instantiate();
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAF, s);
		img->save_exr("res://1_position_" + itos(i) + ".exr", false);

		s = rd->texture_get_data(normal_tex, i);
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);
		img->save_exr("res://1_normal_" + itos(i) + ".exr", false);
	}
#endif

#define FREE_RASTER_RESOURCES   \
	rd->free(rasterize_shader); \
	rd->free(sampler);          \
	rd->free(raster_depth_buffer);

	/* Plot direct light */

	Ref<RDShaderFile> compute_shader;
	compute_shader.instantiate();
	err = compute_shader->parse_versions_from_text(lm_compute_shader_glsl, p_bake_sh ? "\n#define USE_SH_LIGHTMAPS\n" : "");
	if (err != OK) {
		FREE_TEXTURES
		FREE_BUFFERS
		FREE_RASTER_RESOURCES
		memdelete(rd);
		compute_shader->print_errors("compute_shader");
	}
	ERR_FAIL_COND_V(err != OK, BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES);

	// Unoccluder
	RID compute_shader_unocclude = rd->shader_create_from_spirv(compute_shader->get_spirv_stages("unocclude"));
	ERR_FAIL_COND_V(compute_shader_unocclude.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES); // internal check, should not happen
	RID compute_shader_unocclude_pipeline = rd->compute_pipeline_create(compute_shader_unocclude);

	// Direct light
	RID compute_shader_primary = rd->shader_create_from_spirv(compute_shader->get_spirv_stages("primary"));
	ERR_FAIL_COND_V(compute_shader_primary.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES); // internal check, should not happen
	RID compute_shader_primary_pipeline = rd->compute_pipeline_create(compute_shader_primary);

	// Indirect light
	RID compute_shader_secondary = rd->shader_create_from_spirv(compute_shader->get_spirv_stages("secondary"));
	ERR_FAIL_COND_V(compute_shader_secondary.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES); //internal check, should not happen
	RID compute_shader_secondary_pipeline = rd->compute_pipeline_create(compute_shader_secondary);

	// Light probes
	RID compute_shader_light_probes = rd->shader_create_from_spirv(compute_shader->get_spirv_stages("light_probes"));
	ERR_FAIL_COND_V(compute_shader_light_probes.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES); //internal check, should not happen
	RID compute_shader_light_probes_pipeline = rd->compute_pipeline_create(compute_shader_light_probes);

	RID compute_base_uniform_set = rd->uniform_set_create(base_uniforms, compute_shader_primary, 0);

#define FREE_COMPUTE_RESOURCES          \
	rd->free(compute_shader_unocclude); \
	rd->free(compute_shader_primary);   \
	rd->free(compute_shader_secondary); \
	rd->free(compute_shader_light_probes);

	PushConstant push_constant;
	{
		//set defaults
		push_constant.atlas_size[0] = atlas_size.width;
		push_constant.atlas_size[1] = atlas_size.height;
		push_constant.world_size[0] = bounds.size.x;
		push_constant.world_size[1] = bounds.size.y;
		push_constant.world_size[2] = bounds.size.z;
		push_constant.to_cell_offset[0] = bounds.position.x;
		push_constant.to_cell_offset[1] = bounds.position.y;
		push_constant.to_cell_offset[2] = bounds.position.z;
		push_constant.bias = p_bias;
		push_constant.to_cell_size[0] = (1.0 / bounds.size.x) * float(grid_size);
		push_constant.to_cell_size[1] = (1.0 / bounds.size.y) * float(grid_size);
		push_constant.to_cell_size[2] = (1.0 / bounds.size.z) * float(grid_size);
		push_constant.light_count = lights.size();
		push_constant.grid_size = grid_size;
		push_constant.atlas_slice = 0;
		push_constant.region_ofs[0] = 0;
		push_constant.region_ofs[1] = 0;
		push_constant.environment_xform[0] = p_environment_transform.elements[0][0];
		push_constant.environment_xform[1] = p_environment_transform.elements[1][0];
		push_constant.environment_xform[2] = p_environment_transform.elements[2][0];
		push_constant.environment_xform[3] = 0;
		push_constant.environment_xform[4] = p_environment_transform.elements[0][1];
		push_constant.environment_xform[5] = p_environment_transform.elements[1][1];
		push_constant.environment_xform[6] = p_environment_transform.elements[2][1];
		push_constant.environment_xform[7] = 0;
		push_constant.environment_xform[8] = p_environment_transform.elements[0][2];
		push_constant.environment_xform[9] = p_environment_transform.elements[1][2];
		push_constant.environment_xform[10] = p_environment_transform.elements[2][2];
		push_constant.environment_xform[11] = 0;
	}

	Vector3i group_size((atlas_size.x - 1) / 8 + 1, (atlas_size.y - 1) / 8 + 1, 1);
	rd->submit();
	rd->sync();

	if (p_step_function) {
		p_step_function(0.49, TTR("Un-occluding geometry"), p_bake_userdata, true);
	}

	/* UNOCCLUDE */
	{
		Vector<RD::Uniform> uniforms;
		{
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 0;
				u.ids.push_back(position_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(unocclude_tex); //will be unused
				uniforms.push_back(u);
			}
		}

		RID unocclude_uniform_set = rd->uniform_set_create(uniforms, compute_shader_unocclude, 1);

		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, compute_shader_unocclude_pipeline);
		rd->compute_list_bind_uniform_set(compute_list, compute_base_uniform_set, 0);
		rd->compute_list_bind_uniform_set(compute_list, unocclude_uniform_set, 1);

		for (int i = 0; i < atlas_slices; i++) {
			push_constant.atlas_slice = i;
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
			rd->compute_list_dispatch(compute_list, group_size.x, group_size.y, group_size.z);
			//no barrier, let them run all together
		}
		rd->compute_list_end(); //done
	}

	if (p_step_function) {
		p_step_function(0.5, TTR("Plot direct lighting"), p_bake_userdata, true);
	}

	/* PRIMARY (direct) LIGHT PASS */
	{
		Vector<RD::Uniform> uniforms;
		{
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 0;
				u.ids.push_back(light_source_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				u.ids.push_back(light_dest_tex); //will be unused
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 2;
				u.ids.push_back(position_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 3;
				u.ids.push_back(normal_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 4;
				u.ids.push_back(light_accum_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 5;
				u.ids.push_back(light_primary_dynamic_tex);
				uniforms.push_back(u);
			}
		}

		RID light_uniform_set = rd->uniform_set_create(uniforms, compute_shader_primary, 1);

		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, compute_shader_primary_pipeline);
		rd->compute_list_bind_uniform_set(compute_list, compute_base_uniform_set, 0);
		rd->compute_list_bind_uniform_set(compute_list, light_uniform_set, 1);

		for (int i = 0; i < atlas_slices; i++) {
			push_constant.atlas_slice = i;
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
			rd->compute_list_dispatch(compute_list, group_size.x, group_size.y, group_size.z);
			//no barrier, let them run all together
		}
		rd->compute_list_end(); //done
	}

#ifdef DEBUG_TEXTURES

	for (int i = 0; i < atlas_slices; i++) {
		Vector<uint8_t> s = rd->texture_get_data(light_source_tex, i);
		Ref<Image> img;
		img.instantiate();
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);
		img->save_exr("res://2_light_primary_" + itos(i) + ".exr", false);
	}
#endif

	/* SECONDARY (indirect) LIGHT PASS(ES) */
	if (p_step_function) {
		p_step_function(0.6, TTR("Integrate indirect lighting"), p_bake_userdata, true);
	}

	if (p_bounces > 0) {
		Vector<RD::Uniform> uniforms;
		{
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 0;
				u.ids.push_back(light_dest_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				u.ids.push_back(light_source_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 2;
				u.ids.push_back(position_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 3;
				u.ids.push_back(normal_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 4;
				u.ids.push_back(light_accum_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 5;
				u.ids.push_back(unocclude_tex); //reuse unocclude tex
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 6;
				u.ids.push_back(light_environment_tex);
				uniforms.push_back(u);
			}
		}

		RID secondary_uniform_set[2];
		secondary_uniform_set[0] = rd->uniform_set_create(uniforms, compute_shader_secondary, 1);
		uniforms.write[0].ids.write[0] = light_source_tex;
		uniforms.write[1].ids.write[0] = light_dest_tex;
		secondary_uniform_set[1] = rd->uniform_set_create(uniforms, compute_shader_secondary, 1);

		switch (p_quality) {
			case BAKE_QUALITY_LOW: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/low_quality_ray_count");
			} break;
			case BAKE_QUALITY_MEDIUM: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/medium_quality_ray_count");
			} break;
			case BAKE_QUALITY_HIGH: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/high_quality_ray_count");
			} break;
			case BAKE_QUALITY_ULTRA: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/ultra_quality_ray_count");
			} break;
		}

		push_constant.ray_count = CLAMP(push_constant.ray_count, 16, 8192);

		int max_region_size = nearest_power_of_2_templated(int(GLOBAL_GET("rendering/lightmapping/bake_performance/region_size")));
		int max_rays = GLOBAL_GET("rendering/lightmapping/bake_performance/max_rays_per_pass");

		int x_regions = (atlas_size.width - 1) / max_region_size + 1;
		int y_regions = (atlas_size.height - 1) / max_region_size + 1;
		int ray_iterations = (push_constant.ray_count - 1) / max_rays + 1;

		rd->submit();
		rd->sync();

		for (int b = 0; b < p_bounces; b++) {
			int count = 0;
			if (b > 0) {
				SWAP(light_source_tex, light_dest_tex);
				SWAP(secondary_uniform_set[0], secondary_uniform_set[1]);
			}

			for (int s = 0; s < atlas_slices; s++) {
				push_constant.atlas_slice = s;

				for (int i = 0; i < x_regions; i++) {
					for (int j = 0; j < y_regions; j++) {
						int x = i * max_region_size;
						int y = j * max_region_size;
						int w = MIN((i + 1) * max_region_size, atlas_size.width) - x;
						int h = MIN((j + 1) * max_region_size, atlas_size.height) - y;

						push_constant.region_ofs[0] = x;
						push_constant.region_ofs[1] = y;

						group_size = Vector3i((w - 1) / 8 + 1, (h - 1) / 8 + 1, 1);

						for (int k = 0; k < ray_iterations; k++) {
							RD::ComputeListID compute_list = rd->compute_list_begin();
							rd->compute_list_bind_compute_pipeline(compute_list, compute_shader_secondary_pipeline);
							rd->compute_list_bind_uniform_set(compute_list, compute_base_uniform_set, 0);
							rd->compute_list_bind_uniform_set(compute_list, secondary_uniform_set[0], 1);

							push_constant.ray_from = k * max_rays;
							push_constant.ray_to = MIN((k + 1) * max_rays, int32_t(push_constant.ray_count));
							rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
							rd->compute_list_dispatch(compute_list, group_size.x, group_size.y, group_size.z);

							rd->compute_list_end(); //done
							rd->submit();
							rd->sync();

							count++;
							if (p_step_function) {
								int total = (atlas_slices * x_regions * y_regions * ray_iterations);
								int percent = count * 100 / total;
								float p = float(count) / total * 0.1;
								p_step_function(0.6 + p, vformat(TTR("Bounce %d/%d: Integrate indirect lighting %d%%"), b + 1, p_bounces, percent), p_bake_userdata, false);
							}
						}
					}
				}
			}

			if (b == 0) {
				// This disables the environment for subsequent bounces
				push_constant.environment_xform[3] = -99.0f;
			}
		}

		// Restore the correct environment transform
		push_constant.environment_xform[3] = 0.0f;
	}

	/* LIGHTPROBES */

	RID light_probe_buffer;

	if (probe_positions.size()) {
		light_probe_buffer = rd->storage_buffer_create(sizeof(float) * 4 * 9 * probe_positions.size());

		if (p_step_function) {
			p_step_function(0.7, TTR("Baking lightprobes"), p_bake_userdata, true);
		}

		Vector<RD::Uniform> uniforms;
		{
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.ids.push_back(light_probe_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				u.ids.push_back(light_dest_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 2;
				u.ids.push_back(light_primary_dynamic_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 3;
				u.ids.push_back(light_environment_tex);
				uniforms.push_back(u);
			}
		}
		RID light_probe_uniform_set = rd->uniform_set_create(uniforms, compute_shader_light_probes, 1);

		switch (p_quality) {
			case BAKE_QUALITY_LOW: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/low_quality_probe_ray_count");
			} break;
			case BAKE_QUALITY_MEDIUM: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/medium_quality_probe_ray_count");
			} break;
			case BAKE_QUALITY_HIGH: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/high_quality_probe_ray_count");
			} break;
			case BAKE_QUALITY_ULTRA: {
				push_constant.ray_count = GLOBAL_GET("rendering/lightmapping/bake_quality/ultra_quality_probe_ray_count");
			} break;
		}

		push_constant.atlas_size[0] = probe_positions.size();
		push_constant.ray_count = CLAMP(push_constant.ray_count, 16, 8192);

		int max_rays = GLOBAL_GET("rendering/lightmapping/bake_performance/max_rays_per_probe_pass");
		int ray_iterations = (push_constant.ray_count - 1) / max_rays + 1;

		for (int i = 0; i < ray_iterations; i++) {
			RD::ComputeListID compute_list = rd->compute_list_begin();
			rd->compute_list_bind_compute_pipeline(compute_list, compute_shader_light_probes_pipeline);
			rd->compute_list_bind_uniform_set(compute_list, compute_base_uniform_set, 0);
			rd->compute_list_bind_uniform_set(compute_list, light_probe_uniform_set, 1);

			push_constant.ray_from = i * max_rays;
			push_constant.ray_to = MIN((i + 1) * max_rays, int32_t(push_constant.ray_count));
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
			rd->compute_list_dispatch(compute_list, (probe_positions.size() - 1) / 64 + 1, 1, 1);

			rd->compute_list_end(); //done
			rd->submit();
			rd->sync();

			if (p_step_function) {
				int percent = i * 100 / ray_iterations;
				float p = float(i) / ray_iterations * 0.1;
				p_step_function(0.7 + p, vformat(TTR("Integrating light probes %d%%"), percent), p_bake_userdata, false);
			}
		}

		push_constant.atlas_size[0] = atlas_size.x; //restore
	}

#if 0
	for (int i = 0; i < probe_positions.size(); i++) {
		Ref<Image> img;
		img.instantiate();
		img->create(6, 4, false, Image::FORMAT_RGB8);
		for (int j = 0; j < 6; j++) {
			Vector<uint8_t> s = rd->texture_get_data(lightprobe_tex, i * 6 + j);
			Ref<Image> img2;
			img2.instantiate();
			img2->create(2, 2, false, Image::FORMAT_RGBAF, s);
			img2->convert(Image::FORMAT_RGB8);
			img->blit_rect(img2, Rect2(0, 0, 2, 2), Point2((j % 3) * 2, (j / 3) * 2));
		}
		img->save_png("res://3_light_probe_" + itos(i) + ".png");
	}
#endif

	{
		SWAP(light_accum_tex, light_accum_tex2);
		BakeError error = _dilate(rd, compute_shader, compute_base_uniform_set, push_constant, light_accum_tex2, light_accum_tex, atlas_size, atlas_slices * (p_bake_sh ? 4 : 1));
		if (unlikely(error != BAKE_OK)) {
			return error;
		}
	}

	/* DENOISE */

	if (p_use_denoiser) {
		if (p_step_function) {
			p_step_function(0.8, TTR("Denoising"), p_bake_userdata, true);
		}

		Ref<LightmapDenoiser> denoiser = LightmapDenoiser::create();
		if (denoiser.is_valid()) {
			for (int i = 0; i < atlas_slices * (p_bake_sh ? 4 : 1); i++) {
				Vector<uint8_t> s = rd->texture_get_data(light_accum_tex, i);
				Ref<Image> img;
				img.instantiate();
				img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);

				Ref<Image> denoised = denoiser->denoise_image(img);
				if (denoised != img) {
					denoised->convert(Image::FORMAT_RGBAH);
					Vector<uint8_t> ds = denoised->get_data();
					denoised.unref(); //avoid copy on write
					{ //restore alpha
						uint32_t count = s.size() / 2; //uint16s
						const uint16_t *src = (const uint16_t *)s.ptr();
						uint16_t *dst = (uint16_t *)ds.ptrw();
						for (uint32_t j = 0; j < count; j += 4) {
							dst[j + 3] = src[j + 3];
						}
					}
					rd->texture_update(light_accum_tex, i, ds);
				}
			}
		}

		{
			SWAP(light_accum_tex, light_accum_tex2);
			BakeError error = _dilate(rd, compute_shader, compute_base_uniform_set, push_constant, light_accum_tex2, light_accum_tex, atlas_size, atlas_slices * (p_bake_sh ? 4 : 1));
			if (unlikely(error != BAKE_OK)) {
				return error;
			}
		}
	}

#ifdef DEBUG_TEXTURES

	for (int i = 0; i < atlas_slices * (p_bake_sh ? 4 : 1); i++) {
		Vector<uint8_t> s = rd->texture_get_data(light_accum_tex, i);
		Ref<Image> img;
		img.instantiate();
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);
		img->save_exr("res://4_light_secondary_" + itos(i) + ".exr", false);
	}
#endif

	/* BLEND SEAMS */
	//shaders
	Ref<RDShaderFile> blendseams_shader;
	blendseams_shader.instantiate();
	err = blendseams_shader->parse_versions_from_text(lm_blendseams_shader_glsl);
	if (err != OK) {
		FREE_TEXTURES
		FREE_BUFFERS
		FREE_RASTER_RESOURCES
		FREE_COMPUTE_RESOURCES
		memdelete(rd);
		blendseams_shader->print_errors("blendseams_shader");
	}
	ERR_FAIL_COND_V(err != OK, BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES);

	RID blendseams_line_raster_shader = rd->shader_create_from_spirv(blendseams_shader->get_spirv_stages("lines"));

	ERR_FAIL_COND_V(blendseams_line_raster_shader.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES);

	RID blendseams_triangle_raster_shader = rd->shader_create_from_spirv(blendseams_shader->get_spirv_stages("triangles"));

	ERR_FAIL_COND_V(blendseams_triangle_raster_shader.is_null(), BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES);

#define FREE_BLENDSEAMS_RESOURCES            \
	rd->free(blendseams_line_raster_shader); \
	rd->free(blendseams_triangle_raster_shader);

	{
		//pre copy
		for (int i = 0; i < atlas_slices * (p_bake_sh ? 4 : 1); i++) {
			rd->texture_copy(light_accum_tex, light_accum_tex2, Vector3(), Vector3(), Vector3(atlas_size.width, atlas_size.height, 1), 0, 0, i, i);
		}

		Vector<RID> framebuffers;
		for (int i = 0; i < atlas_slices * (p_bake_sh ? 4 : 1); i++) {
			RID slice_tex = rd->texture_create_shared_from_slice(RD::TextureView(), light_accum_tex, i, 0);
			Vector<RID> fb;
			fb.push_back(slice_tex);
			fb.push_back(raster_depth_buffer);
			framebuffers.push_back(rd->framebuffer_create(fb));
		}

		Vector<RD::Uniform> uniforms;
		{
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				u.ids.push_back(light_accum_tex2);
				uniforms.push_back(u);
			}
		}

		RID blendseams_raster_uniform = rd->uniform_set_create(uniforms, blendseams_line_raster_shader, 1);

		bool debug = false;
		RD::PipelineColorBlendState bs = RD::PipelineColorBlendState::create_blend(1);
		bs.attachments.write[0].src_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
		bs.attachments.write[0].dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;

		RD::PipelineDepthStencilState ds;
		ds.enable_depth_test = true;
		ds.enable_depth_write = true;
		ds.depth_compare_operator = RD::COMPARE_OP_LESS; //so it does not render same pixel twice, this avoids wrong blending

		RID blendseams_line_raster_pipeline = rd->render_pipeline_create(blendseams_line_raster_shader, rd->framebuffer_get_format(framebuffers[0]), RD::INVALID_FORMAT_ID, RD::RENDER_PRIMITIVE_LINES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), ds, bs, 0);
		RID blendseams_triangle_raster_pipeline = rd->render_pipeline_create(blendseams_triangle_raster_shader, rd->framebuffer_get_format(framebuffers[0]), RD::INVALID_FORMAT_ID, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), ds, bs, 0);

		uint32_t seam_offset = 0;
		uint32_t triangle_offset = 0;

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0, 0, 0, 1));
		for (int i = 0; i < atlas_slices; i++) {
			int subslices = (p_bake_sh ? 4 : 1);

			if (slice_seam_count[i] == 0) {
				continue;
			}

			for (int k = 0; k < subslices; k++) {
				RasterSeamsPushConstant seams_push_constant;
				seams_push_constant.slice = uint32_t(i * subslices + k);
				seams_push_constant.debug = debug;

				RD::DrawListID draw_list = rd->draw_list_begin(framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);

				rd->draw_list_bind_uniform_set(draw_list, raster_base_uniform, 0);
				rd->draw_list_bind_uniform_set(draw_list, blendseams_raster_uniform, 1);

				const int uv_offset_count = 9;
				static const Vector3 uv_offsets[uv_offset_count] = {
					Vector3(0, 0, 0.5), //using zbuffer, so go inwards-outwards
					Vector3(0, 1, 0.2),
					Vector3(0, -1, 0.2),
					Vector3(1, 0, 0.2),
					Vector3(-1, 0, 0.2),
					Vector3(-1, -1, 0.1),
					Vector3(1, -1, 0.1),
					Vector3(1, 1, 0.1),
					Vector3(-1, 1, 0.1),
				};

				/* step 1 use lines to blend the edges */
				{
					seams_push_constant.base_index = seam_offset;
					rd->draw_list_bind_render_pipeline(draw_list, blendseams_line_raster_pipeline);
					seams_push_constant.uv_offset[0] = uv_offsets[0].x / float(atlas_size.width);
					seams_push_constant.uv_offset[1] = uv_offsets[0].y / float(atlas_size.height);
					seams_push_constant.blend = uv_offsets[0].z;

					rd->draw_list_set_push_constant(draw_list, &seams_push_constant, sizeof(RasterSeamsPushConstant));
					rd->draw_list_draw(draw_list, false, 1, slice_seam_count[i] * 4);
				}

				/* step 2 use triangles to mask the interior */

				{
					seams_push_constant.base_index = triangle_offset;
					rd->draw_list_bind_render_pipeline(draw_list, blendseams_triangle_raster_pipeline);
					seams_push_constant.blend = 0; //do not draw them, just fill the z-buffer so its used as a mask

					rd->draw_list_set_push_constant(draw_list, &seams_push_constant, sizeof(RasterSeamsPushConstant));
					rd->draw_list_draw(draw_list, false, 1, slice_triangle_count[i] * 3);
				}
				/* step 3 blend around the triangle */

				rd->draw_list_bind_render_pipeline(draw_list, blendseams_line_raster_pipeline);

				for (int j = 1; j < uv_offset_count; j++) {
					seams_push_constant.base_index = seam_offset;
					seams_push_constant.uv_offset[0] = uv_offsets[j].x / float(atlas_size.width);
					seams_push_constant.uv_offset[1] = uv_offsets[j].y / float(atlas_size.height);
					seams_push_constant.blend = uv_offsets[0].z;

					rd->draw_list_set_push_constant(draw_list, &seams_push_constant, sizeof(RasterSeamsPushConstant));
					rd->draw_list_draw(draw_list, false, 1, slice_seam_count[i] * 4);
				}
				rd->draw_list_end();
			}
			seam_offset += slice_seam_count[i];
			triangle_offset += slice_triangle_count[i];
		}
	}

#ifdef DEBUG_TEXTURES

	for (int i = 0; i < atlas_slices * (p_bake_sh ? 4 : 1); i++) {
		Vector<uint8_t> s = rd->texture_get_data(light_accum_tex, i);
		Ref<Image> img;
		img.instantiate();
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);
		img->save_exr("res://5_blendseams" + itos(i) + ".exr", false);
	}
#endif
	if (p_step_function) {
		p_step_function(0.9, TTR("Retrieving textures"), p_bake_userdata, true);
	}

	for (int i = 0; i < atlas_slices * (p_bake_sh ? 4 : 1); i++) {
		Vector<uint8_t> s = rd->texture_get_data(light_accum_tex, i);
		Ref<Image> img;
		img.instantiate();
		img->create(atlas_size.width, atlas_size.height, false, Image::FORMAT_RGBAH, s);
		img->convert(Image::FORMAT_RGBH); //remove alpha
		bake_textures.push_back(img);
	}

	if (probe_positions.size() > 0) {
		probe_values.resize(probe_positions.size() * 9);
		Vector<uint8_t> probe_data = rd->buffer_get_data(light_probe_buffer);
		memcpy(probe_values.ptrw(), probe_data.ptr(), probe_data.size());
		rd->free(light_probe_buffer);

#ifdef DEBUG_TEXTURES
		{
			Ref<Image> img2;
			img2.instantiate();
			img2->create(probe_values.size(), 1, false, Image::FORMAT_RGBAF, probe_data);
			img2->save_exr("res://6_lightprobes.exr", false);
		}
#endif
	}

	FREE_TEXTURES
	FREE_BUFFERS
	FREE_RASTER_RESOURCES
	FREE_COMPUTE_RESOURCES
	FREE_BLENDSEAMS_RESOURCES

	memdelete(rd);

	return BAKE_OK;
}

int LightmapperRD::get_bake_texture_count() const {
	return bake_textures.size();
}

Ref<Image> LightmapperRD::get_bake_texture(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, bake_textures.size(), Ref<Image>());
	return bake_textures[p_index];
}

int LightmapperRD::get_bake_mesh_count() const {
	return mesh_instances.size();
}

Variant LightmapperRD::get_bake_mesh_userdata(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh_instances.size(), Variant());
	return mesh_instances[p_index].data.userdata;
}

Rect2 LightmapperRD::get_bake_mesh_uv_scale(int p_index) const {
	ERR_FAIL_COND_V(bake_textures.size() == 0, Rect2());
	Rect2 uv_ofs;
	Vector2 atlas_size = Vector2(bake_textures[0]->get_width(), bake_textures[0]->get_height());
	uv_ofs.position = Vector2(mesh_instances[p_index].offset) / atlas_size;
	uv_ofs.size = Vector2(mesh_instances[p_index].data.albedo_on_uv2->get_width(), mesh_instances[p_index].data.albedo_on_uv2->get_height()) / atlas_size;
	return uv_ofs;
}

int LightmapperRD::get_bake_mesh_texture_slice(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh_instances.size(), Variant());
	return mesh_instances[p_index].slice;
}

int LightmapperRD::get_bake_probe_count() const {
	return probe_positions.size();
}

Vector3 LightmapperRD::get_bake_probe_point(int p_probe) const {
	ERR_FAIL_INDEX_V(p_probe, probe_positions.size(), Variant());
	return Vector3(probe_positions[p_probe].position[0], probe_positions[p_probe].position[1], probe_positions[p_probe].position[2]);
}

Vector<Color> LightmapperRD::get_bake_probe_sh(int p_probe) const {
	ERR_FAIL_INDEX_V(p_probe, probe_positions.size(), Vector<Color>());
	Vector<Color> ret;
	ret.resize(9);
	memcpy(ret.ptrw(), &probe_values[p_probe * 9], sizeof(Color) * 9);
	return ret;
}

LightmapperRD::LightmapperRD() {
}
