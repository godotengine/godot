/*************************************************************************/
/*  visual_server.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "visual_server.h"
#include "global_config.h"
#include "method_bind_ext.inc"

VisualServer *VisualServer::singleton = NULL;
VisualServer *(*VisualServer::create_func)() = NULL;

VisualServer *VisualServer::get_singleton() {

	return singleton;
}

PoolVector<String> VisualServer::_shader_get_param_list(RID p_shader) const {

	//remove at some point

	PoolVector<String> pl;

#if 0
	List<StringName> params;
	shader_get_param_list(p_shader,&params);


	for(List<StringName>::Element *E=params.front();E;E=E->next()) {

		pl.push_back(E->get());
	}
#endif
	return pl;
}

VisualServer *VisualServer::create() {

	ERR_FAIL_COND_V(singleton, NULL);

	if (create_func)
		return create_func();

	return NULL;
}

RID VisualServer::texture_create_from_image(const Image &p_image, uint32_t p_flags) {

	RID texture = texture_create();
	texture_allocate(texture, p_image.get_width(), p_image.get_height(), p_image.get_format(), p_flags); //if it has mipmaps, use, else generate
	ERR_FAIL_COND_V(!texture.is_valid(), texture);

	texture_set_data(texture, p_image);

	return texture;
}

RID VisualServer::get_test_texture() {

	if (test_texture.is_valid()) {
		return test_texture;
	};

#define TEST_TEXTURE_SIZE 256

	PoolVector<uint8_t> test_data;
	test_data.resize(TEST_TEXTURE_SIZE * TEST_TEXTURE_SIZE * 3);

	{
		PoolVector<uint8_t>::Write w = test_data.write();

		for (int x = 0; x < TEST_TEXTURE_SIZE; x++) {

			for (int y = 0; y < TEST_TEXTURE_SIZE; y++) {

				Color c;
				int r = 255 - (x + y) / 2;

				if ((x % (TEST_TEXTURE_SIZE / 8)) < 2 || (y % (TEST_TEXTURE_SIZE / 8)) < 2) {

					c.r = y;
					c.g = r;
					c.b = x;

				} else {

					c.r = r;
					c.g = x;
					c.b = y;
				}

				w[(y * TEST_TEXTURE_SIZE + x) * 3 + 0] = uint8_t(CLAMP(c.r * 255, 0, 255));
				w[(y * TEST_TEXTURE_SIZE + x) * 3 + 1] = uint8_t(CLAMP(c.g * 255, 0, 255));
				w[(y * TEST_TEXTURE_SIZE + x) * 3 + 2] = uint8_t(CLAMP(c.b * 255, 0, 255));
			}
		}
	}

	Image data(TEST_TEXTURE_SIZE, TEST_TEXTURE_SIZE, false, Image::FORMAT_RGB8, test_data);

	test_texture = texture_create_from_image(data);

	return test_texture;
};

void VisualServer::_free_internal_rids() {

	if (test_texture.is_valid())
		free(test_texture);
	if (white_texture.is_valid())
		free(white_texture);
	if (test_material.is_valid())
		free(test_material);

	for (int i = 0; i < 16; i++) {
		if (material_2d[i].is_valid())
			free(material_2d[i]);
	}
}

RID VisualServer::_make_test_cube() {

	PoolVector<Vector3> vertices;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector3> uvs;

	int vtx_idx = 0;
#define ADD_VTX(m_idx)                                                             \
	vertices.push_back(face_points[m_idx]);                                        \
	normals.push_back(normal_points[m_idx]);                                       \
	tangents.push_back(normal_points[m_idx][1]);                                   \
	tangents.push_back(normal_points[m_idx][2]);                                   \
	tangents.push_back(normal_points[m_idx][0]);                                   \
	tangents.push_back(1.0);                                                       \
	uvs.push_back(Vector3(uv_points[m_idx * 2 + 0], uv_points[m_idx * 2 + 1], 0)); \
	vtx_idx++;

	for (int i = 0; i < 6; i++) {

		Vector3 face_points[4];
		Vector3 normal_points[4];
		float uv_points[8] = { 0, 0, 0, 1, 1, 1, 1, 0 };

		for (int j = 0; j < 4; j++) {

			float v[3];
			v[0] = 1.0;
			v[1] = 1 - 2 * ((j >> 1) & 1);
			v[2] = v[1] * (1 - 2 * (j & 1));

			for (int k = 0; k < 3; k++) {

				if (i < 3)
					face_points[j][(i + k) % 3] = v[k] * (i >= 3 ? -1 : 1);
				else
					face_points[3 - j][(i + k) % 3] = v[k] * (i >= 3 ? -1 : 1);
			}
			normal_points[j] = Vector3();
			normal_points[j][i % 3] = (i >= 3 ? -1 : 1);
		}

		//tri 1
		ADD_VTX(0);
		ADD_VTX(1);
		ADD_VTX(2);
		//tri 2
		ADD_VTX(2);
		ADD_VTX(3);
		ADD_VTX(0);
	}

	RID test_cube = mesh_create();

	Array d;
	d.resize(VS::ARRAY_MAX);
	d[VisualServer::ARRAY_NORMAL] = normals;
	d[VisualServer::ARRAY_TANGENT] = tangents;
	d[VisualServer::ARRAY_TEX_UV] = uvs;
	d[VisualServer::ARRAY_VERTEX] = vertices;

	PoolVector<int> indices;
	indices.resize(vertices.size());
	for (int i = 0; i < vertices.size(); i++)
		indices.set(i, i);
	d[VisualServer::ARRAY_INDEX] = indices;

	mesh_add_surface_from_arrays(test_cube, PRIMITIVE_TRIANGLES, d);

	/*
	test_material = fixed_material_create();
	//material_set_flag(material, MATERIAL_FLAG_BILLBOARD_TOGGLE,true);
	fixed_material_set_texture( test_material, FIXED_MATERIAL_PARAM_DIFFUSE, get_test_texture() );
	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_SPECULAR_EXP, 70 );
	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_EMISSION, Color(0.2,0.2,0.2) );

	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_DIFFUSE, Color(1, 1, 1) );
	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_SPECULAR, Color(1,1,1) );
*/
	mesh_surface_set_material(test_cube, 0, test_material);

	return test_cube;
}

RID VisualServer::make_sphere_mesh(int p_lats, int p_lons, float p_radius) {

	PoolVector<Vector3> vertices;
	PoolVector<Vector3> normals;

	for (int i = 1; i <= p_lats; i++) {
		double lat0 = Math_PI * (-0.5 + (double)(i - 1) / p_lats);
		double z0 = Math::sin(lat0);
		double zr0 = Math::cos(lat0);

		double lat1 = Math_PI * (-0.5 + (double)i / p_lats);
		double z1 = Math::sin(lat1);
		double zr1 = Math::cos(lat1);

		for (int j = p_lons; j >= 1; j--) {

			double lng0 = 2 * Math_PI * (double)(j - 1) / p_lons;
			double x0 = Math::cos(lng0);
			double y0 = Math::sin(lng0);

			double lng1 = 2 * Math_PI * (double)(j) / p_lons;
			double x1 = Math::cos(lng1);
			double y1 = Math::sin(lng1);

			Vector3 v[4] = {
				Vector3(x1 * zr0, z0, y1 * zr0),
				Vector3(x1 * zr1, z1, y1 * zr1),
				Vector3(x0 * zr1, z1, y0 * zr1),
				Vector3(x0 * zr0, z0, y0 * zr0)
			};

#define ADD_POINT(m_idx)         \
	normals.push_back(v[m_idx]); \
	vertices.push_back(v[m_idx] * p_radius);

			ADD_POINT(0);
			ADD_POINT(1);
			ADD_POINT(2);

			ADD_POINT(2);
			ADD_POINT(3);
			ADD_POINT(0);
		}
	}

	RID mesh = mesh_create();
	Array d;
	d.resize(VS::ARRAY_MAX);

	d[ARRAY_VERTEX] = vertices;
	d[ARRAY_NORMAL] = normals;

	mesh_add_surface_from_arrays(mesh, PRIMITIVE_TRIANGLES, d);

	return mesh;
}

RID VisualServer::material_2d_get(bool p_shaded, bool p_transparent, bool p_cut_alpha, bool p_opaque_prepass) {

	int version = 0;
	if (p_shaded)
		version = 1;
	if (p_transparent)
		version |= 2;
	if (p_cut_alpha)
		version |= 4;
	if (p_opaque_prepass)
		version |= 8;
	if (material_2d[version].is_valid())
		return material_2d[version];

	//not valid, make

	/*	material_2d[version]=fixed_material_create();
	fixed_material_set_flag(material_2d[version],FIXED_MATERIAL_FLAG_USE_ALPHA,p_transparent);
	fixed_material_set_flag(material_2d[version],FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY,true);
	fixed_material_set_flag(material_2d[version],FIXED_MATERIAL_FLAG_DISCARD_ALPHA,p_cut_alpha);
	material_set_flag(material_2d[version],MATERIAL_FLAG_UNSHADED,!p_shaded);
	material_set_flag(material_2d[version],MATERIAL_FLAG_DOUBLE_SIDED,true);
	material_set_depth_draw_mode(material_2d[version],p_opaque_prepass?MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA:MATERIAL_DEPTH_DRAW_OPAQUE_ONLY);
	fixed_material_set_texture(material_2d[version],FIXED_MATERIAL_PARAM_DIFFUSE,get_white_texture());
	//material cut alpha?*/

	return material_2d[version];
}

RID VisualServer::get_white_texture() {

	if (white_texture.is_valid())
		return white_texture;

	PoolVector<uint8_t> wt;
	wt.resize(16 * 3);
	{
		PoolVector<uint8_t>::Write w = wt.write();
		for (int i = 0; i < 16 * 3; i++)
			w[i] = 255;
	}
	Image white(4, 4, 0, Image::FORMAT_RGB8, wt);
	white_texture = texture_create();
	texture_allocate(white_texture, 4, 4, Image::FORMAT_RGB8);
	texture_set_data(white_texture, white);
	return white_texture;
}

Error VisualServer::_surface_set_data(Array p_arrays, uint32_t p_format, uint32_t *p_offsets, uint32_t p_stride, PoolVector<uint8_t> &r_vertex_array, int p_vertex_array_len, PoolVector<uint8_t> &r_index_array, int p_index_array_len, Rect3 &r_aabb, Vector<Rect3> r_bone_aabb) {

	PoolVector<uint8_t>::Write vw = r_vertex_array.write();

	PoolVector<uint8_t>::Write iw;
	if (r_index_array.size()) {
		print_line("elements: " + itos(r_index_array.size()));

		iw = r_index_array.write();
	}

	int max_bone = 0;

	for (int ai = 0; ai < VS::ARRAY_MAX; ai++) {

		if (!(p_format & (1 << ai))) // no array
			continue;

		switch (ai) {

			case VS::ARRAY_VERTEX: {

				if (p_format & VS::ARRAY_FLAG_USE_2D_VERTICES) {

					PoolVector<Vector2> array = p_arrays[ai];
					ERR_FAIL_COND_V(array.size() != p_vertex_array_len, ERR_INVALID_PARAMETER);

					PoolVector<Vector2>::Read read = array.read();
					const Vector2 *src = read.ptr();

					// setting vertices means regenerating the AABB
					Rect2 aabb;

					if (p_format & ARRAY_COMPRESS_VERTEX) {

						for (int i = 0; i < p_vertex_array_len; i++) {

							uint16_t vector[2] = { Math::make_half_float(src[i].x), Math::make_half_float(src[i].y) };

							copymem(&vw[p_offsets[ai] + i * p_stride], vector, sizeof(uint16_t) * 2);

							if (i == 0) {

								aabb = Rect2(src[i], Vector2());
							} else {

								aabb.expand_to(src[i]);
							}
						}

					} else {
						for (int i = 0; i < p_vertex_array_len; i++) {

							float vector[2] = { src[i].x, src[i].y };

							copymem(&vw[p_offsets[ai] + i * p_stride], vector, sizeof(float) * 2);

							if (i == 0) {

								aabb = Rect2(src[i], Vector2());
							} else {

								aabb.expand_to(src[i]);
							}
						}
					}

					r_aabb = Rect3(Vector3(aabb.pos.x, aabb.pos.y, 0), Vector3(aabb.size.x, aabb.size.y, 0));

				} else {
					PoolVector<Vector3> array = p_arrays[ai];
					ERR_FAIL_COND_V(array.size() != p_vertex_array_len, ERR_INVALID_PARAMETER);

					PoolVector<Vector3>::Read read = array.read();
					const Vector3 *src = read.ptr();

					// setting vertices means regenerating the AABB
					Rect3 aabb;

					if (p_format & ARRAY_COMPRESS_VERTEX) {

						for (int i = 0; i < p_vertex_array_len; i++) {

							uint16_t vector[4] = { Math::make_half_float(src[i].x), Math::make_half_float(src[i].y), Math::make_half_float(src[i].z), Math::make_half_float(1.0) };

							copymem(&vw[p_offsets[ai] + i * p_stride], vector, sizeof(uint16_t) * 4);

							if (i == 0) {

								aabb = Rect3(src[i], Vector3());
							} else {

								aabb.expand_to(src[i]);
							}
						}

					} else {
						for (int i = 0; i < p_vertex_array_len; i++) {

							float vector[3] = { src[i].x, src[i].y, src[i].z };

							copymem(&vw[p_offsets[ai] + i * p_stride], vector, sizeof(float) * 3);

							if (i == 0) {

								aabb = Rect3(src[i], Vector3());
							} else {

								aabb.expand_to(src[i]);
							}
						}
					}

					r_aabb = aabb;
				}

			} break;
			case VS::ARRAY_NORMAL: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_VECTOR3_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Vector3> array = p_arrays[ai];
				ERR_FAIL_COND_V(array.size() != p_vertex_array_len, ERR_INVALID_PARAMETER);

				PoolVector<Vector3>::Read read = array.read();
				const Vector3 *src = read.ptr();

				// setting vertices means regenerating the AABB

				if (p_format & ARRAY_COMPRESS_NORMAL) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint8_t vector[4] = {
							CLAMP(src[i].x * 127, -128, 127),
							CLAMP(src[i].y * 127, -128, 127),
							CLAMP(src[i].z * 127, -128, 127),
							0,
						};

						copymem(&vw[p_offsets[ai] + i * p_stride], vector, 4);
					}

				} else {
					for (int i = 0; i < p_vertex_array_len; i++) {

						float vector[3] = { src[i].x, src[i].y, src[i].z };
						copymem(&vw[p_offsets[ai] + i * p_stride], vector, 3 * 4);
					}
				}

			} break;

			case VS::ARRAY_TANGENT: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_REAL_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<real_t> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_vertex_array_len * 4, ERR_INVALID_PARAMETER);

				PoolVector<real_t>::Read read = array.read();
				const real_t *src = read.ptr();

				if (p_format & ARRAY_COMPRESS_TANGENT) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint8_t xyzw[4] = {
							CLAMP(src[i * 4 + 0] * 127, -128, 127),
							CLAMP(src[i * 4 + 1] * 127, -128, 127),
							CLAMP(src[i * 4 + 2] * 127, -128, 127),
							CLAMP(src[i * 4 + 3] * 127, -128, 127)
						};

						copymem(&vw[p_offsets[ai] + i * p_stride], xyzw, 4);
					}

				} else {
					for (int i = 0; i < p_vertex_array_len; i++) {

						float xyzw[4] = {
							src[i * 4 + 0],
							src[i * 4 + 1],
							src[i * 4 + 2],
							src[i * 4 + 3]
						};

						copymem(&vw[p_offsets[ai] + i * p_stride], xyzw, 4 * 4);
					}
				}

			} break;
			case VS::ARRAY_COLOR: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_COLOR_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Color> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_vertex_array_len, ERR_INVALID_PARAMETER);

				PoolVector<Color>::Read read = array.read();
				const Color *src = read.ptr();

				if (p_format & ARRAY_COMPRESS_COLOR) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint8_t colors[4];

						for (int j = 0; j < 4; j++) {

							colors[j] = CLAMP(int((src[i][j]) * 255.0), 0, 255);
						}

						copymem(&vw[p_offsets[ai] + i * p_stride], colors, 4);
					}
				} else {

					for (int i = 0; i < p_vertex_array_len; i++) {

						copymem(&vw[p_offsets[ai] + i * p_stride], &src[i], 4 * 4);
					}
				}

			} break;
			case VS::ARRAY_TEX_UV: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_VECTOR3_ARRAY && p_arrays[ai].get_type() != Variant::POOL_VECTOR2_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Vector2> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_vertex_array_len, ERR_INVALID_PARAMETER);

				PoolVector<Vector2>::Read read = array.read();

				const Vector2 *src = read.ptr();

				if (p_format & ARRAY_COMPRESS_TEX_UV) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint16_t uv[2] = { Math::make_half_float(src[i].x), Math::make_half_float(src[i].y) };
						copymem(&vw[p_offsets[ai] + i * p_stride], uv, 2 * 2);
					}

				} else {
					for (int i = 0; i < p_vertex_array_len; i++) {

						float uv[2] = { src[i].x, src[i].y };

						copymem(&vw[p_offsets[ai] + i * p_stride], uv, 2 * 4);
					}
				}

			} break;

			case VS::ARRAY_TEX_UV2: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_VECTOR3_ARRAY && p_arrays[ai].get_type() != Variant::POOL_VECTOR2_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Vector2> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_vertex_array_len, ERR_INVALID_PARAMETER);

				PoolVector<Vector2>::Read read = array.read();

				const Vector2 *src = read.ptr();

				if (p_format & ARRAY_COMPRESS_TEX_UV2) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint16_t uv[2] = { Math::make_half_float(src[i].x), Math::make_half_float(src[i].y) };
						copymem(&vw[p_offsets[ai] + i * p_stride], uv, 2 * 2);
					}

				} else {
					for (int i = 0; i < p_vertex_array_len; i++) {

						float uv[2] = { src[i].x, src[i].y };

						copymem(&vw[p_offsets[ai] + i * p_stride], uv, 2 * 4);
					}
				}
			} break;
			case VS::ARRAY_WEIGHTS: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_REAL_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<real_t> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_vertex_array_len * VS::ARRAY_WEIGHTS_SIZE, ERR_INVALID_PARAMETER);

				PoolVector<real_t>::Read read = array.read();

				const real_t *src = read.ptr();

				if (p_format & ARRAY_COMPRESS_WEIGHTS) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint16_t data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = CLAMP(src[i * VS::ARRAY_WEIGHTS_SIZE + j] * 65535, 0, 65535);
						}

						copymem(&vw[p_offsets[ai] + i * p_stride], data, 2 * 4);
					}
				} else {

					for (int i = 0; i < p_vertex_array_len; i++) {

						float data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = src[i * VS::ARRAY_WEIGHTS_SIZE + j];
						}

						copymem(&vw[p_offsets[ai] + i * p_stride], data, 4 * 4);
					}
				}

			} break;
			case VS::ARRAY_BONES: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_INT_ARRAY && p_arrays[ai].get_type() != Variant::POOL_REAL_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<int> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_vertex_array_len * VS::ARRAY_WEIGHTS_SIZE, ERR_INVALID_PARAMETER);

				PoolVector<int>::Read read = array.read();

				const int *src = read.ptr();

				if (!(p_format & ARRAY_FLAG_USE_16_BIT_BONES)) {

					for (int i = 0; i < p_vertex_array_len; i++) {

						uint8_t data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = CLAMP(src[i * VS::ARRAY_WEIGHTS_SIZE + j], 0, 255);
							max_bone = MAX(data[j], max_bone);
						}

						copymem(&vw[p_offsets[ai] + i * p_stride], data, 4);
					}

				} else {
					for (int i = 0; i < p_vertex_array_len; i++) {

						uint16_t data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = src[i * VS::ARRAY_WEIGHTS_SIZE + j];
							max_bone = MAX(data[j], max_bone);
						}

						copymem(&vw[p_offsets[ai] + i * p_stride], data, 2 * 4);
					}
				}

			} break;
			case VS::ARRAY_INDEX: {

				ERR_FAIL_COND_V(p_index_array_len <= 0, ERR_INVALID_DATA);
				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::POOL_INT_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<int> indices = p_arrays[ai];
				ERR_FAIL_COND_V(indices.size() == 0, ERR_INVALID_PARAMETER);
				ERR_FAIL_COND_V(indices.size() != p_index_array_len, ERR_INVALID_PARAMETER);

				/* determine wether using 16 or 32 bits indices */

				PoolVector<int>::Read read = indices.read();
				const int *src = read.ptr();

				for (int i = 0; i < p_index_array_len; i++) {

					if (p_vertex_array_len < (1 << 16)) {
						uint16_t v = src[i];

						copymem(&iw[i * 2], &v, 2);
					} else {
						uint32_t v = src[i];

						copymem(&iw[i * 4], &v, 4);
					}
				}
			} break;
			default: {
				ERR_FAIL_V(ERR_INVALID_DATA);
			}
		}
	}

	if (p_format & VS::ARRAY_FORMAT_BONES) {
		//create AABBs for each detected bone
		int total_bones = max_bone + 1;

		bool first = r_bone_aabb.size() == 0;

		r_bone_aabb.resize(total_bones);

		if (first) {
			for (int i = 0; i < total_bones; i++) {
				r_bone_aabb[i].size == Vector3(-1, -1, -1); //negative means unused
			}
		}

		PoolVector<Vector3> vertices = p_arrays[VS::ARRAY_VERTEX];
		PoolVector<int> bones = p_arrays[VS::ARRAY_BONES];
		PoolVector<float> weights = p_arrays[VS::ARRAY_WEIGHTS];

		bool any_valid = false;

		if (vertices.size() && bones.size() == vertices.size() * 4 && weights.size() == bones.size()) {

			int vs = vertices.size();
			PoolVector<Vector3>::Read rv = vertices.read();
			PoolVector<int>::Read rb = bones.read();
			PoolVector<float>::Read rw = weights.read();

			Rect3 *bptr = r_bone_aabb.ptr();

			for (int i = 0; i < vs; i++) {

				Vector3 v = rv[i];
				for (int j = 0; j < 4; j++) {

					int idx = rb[i * 4 + j];
					float w = rw[i * 4 + j];
					if (w == 0)
						continue; //break;
					ERR_FAIL_INDEX_V(idx, total_bones, ERR_INVALID_DATA);

					if (bptr->size.x < 0) {
						//first
						bptr[idx] = Rect3();
						bptr[idx].pos = v;
						any_valid = true;
					} else {
						bptr[idx].expand_to(v);
					}
				}
			}
		}

		if (!any_valid && first) {

			r_bone_aabb.clear();
		}
	}
	return OK;
}

void VisualServer::mesh_add_surface_from_arrays(RID p_mesh, PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes, uint32_t p_compress_format) {

	ERR_FAIL_INDEX(p_primitive, VS::PRIMITIVE_MAX);
	ERR_FAIL_COND(p_arrays.size() != VS::ARRAY_MAX);

	uint32_t format = 0;

	// validation
	int index_array_len = 0;
	int array_len = 0;

	for (int i = 0; i < p_arrays.size(); i++) {

		if (p_arrays[i].get_type() == Variant::NIL)
			continue;

		format |= (1 << i);

		if (i == VS::ARRAY_VERTEX) {

			Variant var = p_arrays[i];
			switch (var.get_type()) {
				case Variant::POOL_VECTOR2_ARRAY: {
					PoolVector<Vector2> v2 = var;
					array_len = v2.size();
				} break;
				case Variant::POOL_VECTOR3_ARRAY: {
					PoolVector<Vector3> v3 = var;
					array_len = v3.size();
				} break;
				default: {
					Array v = var;
					array_len = v.size();
				} break;
			}

			array_len = PoolVector3Array(p_arrays[i]).size();
			ERR_FAIL_COND(array_len == 0);
		} else if (i == VS::ARRAY_INDEX) {

			index_array_len = PoolIntArray(p_arrays[i]).size();
		}
	}

	ERR_FAIL_COND((format & VS::ARRAY_FORMAT_VERTEX) == 0); // mandatory

	if (p_blend_shapes.size()) {
		//validate format for morphs
		for (int i = 0; i < p_blend_shapes.size(); i++) {

			uint32_t bsformat = 0;
			Array arr = p_blend_shapes[i];
			for (int j = 0; j < arr.size(); j++) {

				if (arr[j].get_type() != Variant::NIL)
					bsformat |= (1 << j);
			}

			ERR_FAIL_COND((bsformat) != (format & (VS::ARRAY_FORMAT_INDEX - 1)));
		}
	}

	uint32_t offsets[VS::ARRAY_MAX];

	int total_elem_size = 0;

	for (int i = 0; i < VS::ARRAY_MAX; i++) {

		offsets[i] = 0; //reset

		if (!(format & (1 << i))) // no array
			continue;

		int elem_size = 0;

		switch (i) {

			case VS::ARRAY_VERTEX: {

				Variant arr = p_arrays[0];
				if (arr.get_type() == Variant::POOL_VECTOR2_ARRAY) {
					elem_size = 2;
					p_compress_format |= ARRAY_FLAG_USE_2D_VERTICES;
				} else if (arr.get_type() == Variant::POOL_VECTOR3_ARRAY) {
					p_compress_format &= ~ARRAY_FLAG_USE_2D_VERTICES;
					elem_size = 3;
				} else {
					elem_size = (p_compress_format & ARRAY_FLAG_USE_2D_VERTICES) ? 2 : 3;
				}

				if (p_compress_format & ARRAY_COMPRESS_VERTEX) {
					elem_size *= sizeof(int16_t);
				} else {
					elem_size *= sizeof(float);
				}

				if (elem_size == 6) {
					//had to pad
					elem_size = 8;
				}

			} break;
			case VS::ARRAY_NORMAL: {

				if (p_compress_format & ARRAY_COMPRESS_NORMAL) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 3;
				}

			} break;

			case VS::ARRAY_TANGENT: {
				if (p_compress_format & ARRAY_COMPRESS_TANGENT) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 4;
				}

			} break;
			case VS::ARRAY_COLOR: {

				if (p_compress_format & ARRAY_COMPRESS_COLOR) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 4;
				}
			} break;
			case VS::ARRAY_TEX_UV: {
				if (p_compress_format & ARRAY_COMPRESS_TEX_UV) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 2;
				}

			} break;

			case VS::ARRAY_TEX_UV2: {
				if (p_compress_format & ARRAY_COMPRESS_TEX_UV2) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 2;
				}

			} break;
			case VS::ARRAY_WEIGHTS: {

				if (p_compress_format & ARRAY_COMPRESS_WEIGHTS) {
					elem_size = sizeof(uint16_t) * 4;
				} else {
					elem_size = sizeof(float) * 4;
				}

			} break;
			case VS::ARRAY_BONES: {

				PoolVector<int> bones = p_arrays[VS::ARRAY_BONES];
				int max_bone = 0;

				{
					int bc = bones.size();
					PoolVector<int>::Read r = bones.read();
					for (int j = 0; j < bc; j++) {
						max_bone = MAX(r[j], max_bone);
					}
				}

				if (max_bone > 255) {
					p_compress_format |= ARRAY_FLAG_USE_16_BIT_BONES;
					elem_size = sizeof(uint16_t) * 4;
				} else {
					p_compress_format &= ~ARRAY_FLAG_USE_16_BIT_BONES;
					elem_size = sizeof(uint32_t);
				}

			} break;
			case VS::ARRAY_INDEX: {

				if (index_array_len <= 0) {
					ERR_PRINT("index_array_len==NO_INDEX_ARRAY");
					break;
				}
				/* determine wether using 16 or 32 bits indices */
				if (array_len >= (1 << 16)) {

					elem_size = 4;

				} else {
					elem_size = 2;
				}
				offsets[i] = elem_size;
				continue;
			} break;
			default: {
				ERR_FAIL();
			}
		}

		offsets[i] = total_elem_size;
		total_elem_size += elem_size;
	}

	uint32_t mask = (1 << ARRAY_MAX) - 1;
	format |= (~mask) & p_compress_format; //make the full format

	int array_size = total_elem_size * array_len;

	PoolVector<uint8_t> vertex_array;
	vertex_array.resize(array_size);

	int index_array_size = offsets[VS::ARRAY_INDEX] * index_array_len;

	PoolVector<uint8_t> index_array;
	index_array.resize(index_array_size);

	Rect3 aabb;
	Vector<Rect3> bone_aabb;

	Error err = _surface_set_data(p_arrays, format, offsets, total_elem_size, vertex_array, array_len, index_array, index_array_len, aabb, bone_aabb);

	if (err) {
		ERR_EXPLAIN("Invalid array format for surface");
		ERR_FAIL_COND(err != OK);
	}

	Vector<PoolVector<uint8_t> > blend_shape_data;

	for (int i = 0; i < p_blend_shapes.size(); i++) {

		PoolVector<uint8_t> vertex_array_shape;
		vertex_array_shape.resize(array_size);
		PoolVector<uint8_t> noindex;

		Rect3 laabb;
		Error err = _surface_set_data(p_blend_shapes[i], format & ~ARRAY_FORMAT_INDEX, offsets, total_elem_size, vertex_array_shape, array_len, noindex, 0, laabb, bone_aabb);
		aabb.merge_with(laabb);
		if (err) {
			ERR_EXPLAIN("Invalid blend shape array format for surface");
			ERR_FAIL_COND(err != OK);
		}

		blend_shape_data.push_back(vertex_array_shape);
	}

	mesh_add_surface(p_mesh, format, p_primitive, vertex_array, array_len, index_array, index_array_len, aabb, blend_shape_data, bone_aabb);
}

Array VisualServer::_get_array_from_surface(uint32_t p_format, PoolVector<uint8_t> p_vertex_data, int p_vertex_len, PoolVector<uint8_t> p_index_data, int p_index_len) const {

	uint32_t offsets[ARRAY_MAX];

	int total_elem_size = 0;

	for (int i = 0; i < VS::ARRAY_MAX; i++) {

		offsets[i] = 0; //reset

		if (!(p_format & (1 << i))) // no array
			continue;

		int elem_size = 0;

		switch (i) {

			case VS::ARRAY_VERTEX: {

				if (p_format & ARRAY_FLAG_USE_2D_VERTICES) {
					elem_size = 2;
				} else {
					elem_size = 3;
				}

				if (p_format & ARRAY_COMPRESS_VERTEX) {
					elem_size *= sizeof(int16_t);
				} else {
					elem_size *= sizeof(float);
				}

				if (elem_size == 6) {
					elem_size = 8;
				}

			} break;
			case VS::ARRAY_NORMAL: {

				if (p_format & ARRAY_COMPRESS_NORMAL) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 3;
				}

			} break;

			case VS::ARRAY_TANGENT: {
				if (p_format & ARRAY_COMPRESS_TANGENT) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 4;
				}

			} break;
			case VS::ARRAY_COLOR: {

				if (p_format & ARRAY_COMPRESS_COLOR) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 4;
				}
			} break;
			case VS::ARRAY_TEX_UV: {
				if (p_format & ARRAY_COMPRESS_TEX_UV) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 2;
				}

			} break;

			case VS::ARRAY_TEX_UV2: {
				if (p_format & ARRAY_COMPRESS_TEX_UV2) {
					elem_size = sizeof(uint32_t);
				} else {
					elem_size = sizeof(float) * 2;
				}

			} break;
			case VS::ARRAY_WEIGHTS: {

				if (p_format & ARRAY_COMPRESS_WEIGHTS) {
					elem_size = sizeof(uint16_t) * 4;
				} else {
					elem_size = sizeof(float) * 4;
				}

			} break;
			case VS::ARRAY_BONES: {

				if (p_format & ARRAY_FLAG_USE_16_BIT_BONES) {
					elem_size = sizeof(uint16_t) * 4;
				} else {
					elem_size = sizeof(uint32_t);
				}

			} break;
			case VS::ARRAY_INDEX: {

				if (p_index_len <= 0) {
					ERR_PRINT("index_array_len==NO_INDEX_ARRAY");
					break;
				}
				/* determine wether using 16 or 32 bits indices */
				if (p_vertex_len >= (1 << 16)) {

					elem_size = 4;

				} else {
					elem_size = 2;
				}
				offsets[i] = elem_size;
				continue;
			} break;
			default: {
				ERR_FAIL_V(Array());
			}
		}

		offsets[i] = total_elem_size;
		total_elem_size += elem_size;
	}

	Array ret;
	ret.resize(VS::ARRAY_MAX);

	PoolVector<uint8_t>::Read r = p_vertex_data.read();

	for (int i = 0; i < VS::ARRAY_MAX; i++) {

		if (!(p_format & (1 << i)))
			continue;

		switch (i) {

			case VS::ARRAY_VERTEX: {

				if (p_format & ARRAY_FLAG_USE_2D_VERTICES) {

					PoolVector<Vector2> arr_2d;
					arr_2d.resize(p_vertex_len);

					if (p_format & ARRAY_COMPRESS_VERTEX) {

						PoolVector<Vector2>::Write w = arr_2d.write();

						for (int j = 0; j < p_vertex_len; j++) {

							const uint16_t *v = (const uint16_t *)&r[j * total_elem_size + offsets[i]];
							w[j] = Vector2(Math::halfptr_to_float(&v[0]), Math::halfptr_to_float(&v[1]));
						}
					} else {

						PoolVector<Vector2>::Write w = arr_2d.write();

						for (int j = 0; j < p_vertex_len; j++) {

							const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
							w[j] = Vector2(v[0], v[1]);
						}
					}

					ret[i] = arr_2d;
				} else {

					PoolVector<Vector3> arr_3d;
					arr_3d.resize(p_vertex_len);

					if (p_format & ARRAY_COMPRESS_VERTEX) {

						PoolVector<Vector3>::Write w = arr_3d.write();

						for (int j = 0; j < p_vertex_len; j++) {

							const uint16_t *v = (const uint16_t *)&r[j * total_elem_size + offsets[i]];
							w[j] = Vector3(Math::halfptr_to_float(&v[0]), Math::halfptr_to_float(&v[1]), Math::halfptr_to_float(&v[2]));
						}
					} else {

						PoolVector<Vector3>::Write w = arr_3d.write();

						for (int j = 0; j < p_vertex_len; j++) {

							const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
							w[j] = Vector3(v[0], v[1], v[2]);
						}
					}

					ret[i] = arr_3d;
				}

			} break;
			case VS::ARRAY_NORMAL: {
				PoolVector<Vector3> arr;
				arr.resize(p_vertex_len);

				if (p_format & ARRAY_COMPRESS_NORMAL) {

					PoolVector<Vector3>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint8_t *v = (const uint8_t *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector3(float(v[0] / 255.0) * 2.0 - 1.0, float(v[1] / 255.0) * 2.0 - 1.0, float(v[2] / 255.0) * 2.0 - 1.0);
					}
				} else {
					PoolVector<Vector3>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector3(v[0], v[1], v[2]);
					}
				}

				ret[i] = arr;

			} break;

			case VS::ARRAY_TANGENT: {
				PoolVector<float> arr;
				arr.resize(p_vertex_len * 4);
				if (p_format & ARRAY_COMPRESS_TANGENT) {
					PoolVector<float>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint8_t *v = (const uint8_t *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = float(v[k] / 255.0) * 2.0 - 1.0;
						}
					}
				} else {

					PoolVector<float>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {
						const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = v[k];
						}
					}
				}

				ret[i] = arr;

			} break;
			case VS::ARRAY_COLOR: {

				PoolVector<Color> arr;
				arr.resize(p_vertex_len);

				if (p_format & ARRAY_COMPRESS_COLOR) {

					PoolVector<Color>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint8_t *v = (const uint8_t *)&r[j * total_elem_size + offsets[i]];
						w[j] = Color(float(v[0] / 255.0) * 2.0 - 1.0, float(v[1] / 255.0) * 2.0 - 1.0, float(v[2] / 255.0) * 2.0 - 1.0, float(v[3] / 255.0) * 2.0 - 1.0);
					}
				} else {
					PoolVector<Color>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
						w[j] = Color(v[0], v[1], v[2], v[3]);
					}
				}

				ret[i] = arr;
			} break;
			case VS::ARRAY_TEX_UV: {

				PoolVector<Vector2> arr;
				arr.resize(p_vertex_len);

				if (p_format & ARRAY_COMPRESS_TEX_UV) {

					PoolVector<Vector2>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint16_t *v = (const uint16_t *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector2(Math::halfptr_to_float(&v[0]), Math::halfptr_to_float(&v[1]));
					}
				} else {

					PoolVector<Vector2>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector2(v[0], v[1]);
					}
				}

				ret[i] = arr;
			} break;

			case VS::ARRAY_TEX_UV2: {
				PoolVector<Vector2> arr;
				arr.resize(p_vertex_len);

				if (p_format & ARRAY_COMPRESS_TEX_UV2) {

					PoolVector<Vector2>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint16_t *v = (const uint16_t *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector2(Math::halfptr_to_float(&v[0]), Math::halfptr_to_float(&v[1]));
					}
				} else {

					PoolVector<Vector2>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector2(v[0], v[1]);
					}
				}

				ret[i] = arr;

			} break;
			case VS::ARRAY_WEIGHTS: {

				PoolVector<float> arr;
				arr.resize(p_vertex_len * 4);
				if (p_format & ARRAY_COMPRESS_WEIGHTS) {
					PoolVector<float>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint16_t *v = (const uint16_t *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = float(v[k] / 65535.0) * 2.0 - 1.0;
						}
					}
				} else {

					PoolVector<float>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {
						const float *v = (const float *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = v[k];
						}
					}
				}

				ret[i] = arr;

			} break;
			case VS::ARRAY_BONES: {

				PoolVector<int> arr;
				arr.resize(p_vertex_len * 4);
				if (p_format & ARRAY_FLAG_USE_16_BIT_BONES) {

					PoolVector<int>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {

						const uint16_t *v = (const uint16_t *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = v[k];
						}
					}
				} else {

					PoolVector<int>::Write w = arr.write();

					for (int j = 0; j < p_vertex_len; j++) {
						const uint8_t *v = (const uint8_t *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = v[k];
						}
					}
				}

				ret[i] = arr;

			} break;
			case VS::ARRAY_INDEX: {
				/* determine wether using 16 or 32 bits indices */

				PoolVector<uint8_t>::Read ir = p_index_data.read();

				PoolVector<int> arr;
				arr.resize(p_index_len);
				if (p_vertex_len < (1 << 16)) {

					PoolVector<int>::Write w = arr.write();

					for (int j = 0; j < p_index_len; j++) {

						const uint16_t *v = (const uint16_t *)&ir[j * 2];
						w[j] = *v;
					}
				} else {

					PoolVector<int>::Write w = arr.write();

					for (int j = 0; j < p_index_len; j++) {
						const int *v = (const int *)&ir[j * 4];
						w[j] = *v;
					}
				}
				ret[i] = arr;
			} break;
			default: {
				ERR_FAIL_V(ret);
			}
		}
	}

	return ret;
}

Array VisualServer::mesh_surface_get_arrays(RID p_mesh, int p_surface) const {

	PoolVector<uint8_t> vertex_data = mesh_surface_get_array(p_mesh, p_surface);
	ERR_FAIL_COND_V(vertex_data.size() == 0, Array());
	int vertex_len = mesh_surface_get_array_len(p_mesh, p_surface);

	PoolVector<uint8_t> index_data = mesh_surface_get_index_array(p_mesh, p_surface);
	int index_len = mesh_surface_get_array_index_len(p_mesh, p_surface);

	uint32_t format = mesh_surface_get_format(p_mesh, p_surface);

	return _get_array_from_surface(format, vertex_data, vertex_len, index_data, index_len);
}

void VisualServer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("texture_create"), &VisualServer::texture_create);
	ClassDB::bind_method(D_METHOD("texture_create_from_image"), &VisualServer::texture_create_from_image, DEFVAL(TEXTURE_FLAGS_DEFAULT));
	//ClassDB::bind_method(D_METHOD("texture_allocate"),&VisualServer::texture_allocate,DEFVAL( TEXTURE_FLAGS_DEFAULT ) );
	//ClassDB::bind_method(D_METHOD("texture_set_data"),&VisualServer::texture_blit_rect,DEFVAL( CUBEMAP_LEFT ) );
	//ClassDB::bind_method(D_METHOD("texture_get_rect"),&VisualServer::texture_get_rect );
	ClassDB::bind_method(D_METHOD("texture_set_flags"), &VisualServer::texture_set_flags);
	ClassDB::bind_method(D_METHOD("texture_get_flags"), &VisualServer::texture_get_flags);
	ClassDB::bind_method(D_METHOD("texture_get_width"), &VisualServer::texture_get_width);
	ClassDB::bind_method(D_METHOD("texture_get_height"), &VisualServer::texture_get_height);

	ClassDB::bind_method(D_METHOD("texture_set_shrink_all_x2_on_set_data", "shrink"), &VisualServer::texture_set_shrink_all_x2_on_set_data);
}

void VisualServer::_canvas_item_add_style_box(RID p_item, const Rect2 &p_rect, const Rect2 &p_source, RID p_texture, const Vector<float> &p_margins, const Color &p_modulate) {

	ERR_FAIL_COND(p_margins.size() != 4);
	//canvas_item_add_style_box(p_item,p_rect,p_source,p_texture,Vector2(p_margins[0],p_margins[1]),Vector2(p_margins[2],p_margins[3]),true,p_modulate);
}

void VisualServer::_camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far) {

	camera_set_orthogonal(p_camera, p_size, p_z_near, p_z_far);
}

void VisualServer::mesh_add_surface_from_mesh_data(RID p_mesh, const Geometry::MeshData &p_mesh_data) {

#if 1
	PoolVector<Vector3> vertices;
	PoolVector<Vector3> normals;

	for (int i = 0; i < p_mesh_data.faces.size(); i++) {

		const Geometry::MeshData::Face &f = p_mesh_data.faces[i];

		for (int j = 2; j < f.indices.size(); j++) {

#define _ADD_VERTEX(m_idx)                                      \
	vertices.push_back(p_mesh_data.vertices[f.indices[m_idx]]); \
	normals.push_back(f.plane.normal);

			_ADD_VERTEX(0);
			_ADD_VERTEX(j - 1);
			_ADD_VERTEX(j);
		}
	}

	Array d;
	d.resize(VS::ARRAY_MAX);
	d[ARRAY_VERTEX] = vertices;
	d[ARRAY_NORMAL] = normals;
	mesh_add_surface_from_arrays(p_mesh, PRIMITIVE_TRIANGLES, d);

#else

	PoolVector<Vector3> vertices;

	for (int i = 0; i < p_mesh_data.edges.size(); i++) {

		const Geometry::MeshData::Edge &f = p_mesh_data.edges[i];
		vertices.push_back(p_mesh_data.vertices[f.a]);
		vertices.push_back(p_mesh_data.vertices[f.b]);
	}

	Array d;
	d.resize(VS::ARRAY_MAX);
	d[ARRAY_VERTEX] = vertices;
	mesh_add_surface(p_mesh, PRIMITIVE_LINES, d);

#endif
}

void VisualServer::mesh_add_surface_from_planes(RID p_mesh, const PoolVector<Plane> &p_planes) {

	Geometry::MeshData mdata = Geometry::build_convex_mesh(p_planes);
	mesh_add_surface_from_mesh_data(p_mesh, mdata);
}

void VisualServer::immediate_vertex_2d(RID p_immediate, const Vector2 &p_vertex) {
	immediate_vertex(p_immediate, Vector3(p_vertex.x, p_vertex.y, 0));
}

RID VisualServer::instance_create2(RID p_base, RID p_scenario) {

	RID instance = instance_create();
	instance_set_base(instance, p_base);
	instance_set_scenario(instance, p_scenario);
	return instance;
}

VisualServer::VisualServer() {

	//ERR_FAIL_COND(singleton);
	singleton = this;
}

VisualServer::~VisualServer() {

	singleton = NULL;
}
