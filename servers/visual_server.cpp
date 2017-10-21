/*************************************************************************/
/*  visual_server.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

#include "method_bind_ext.gen.inc"
#include "project_settings.h"

VisualServer *VisualServer::singleton = NULL;
VisualServer *(*VisualServer::create_func)() = NULL;

VisualServer *VisualServer::get_singleton() {

	return singleton;
}

VisualServer *VisualServer::create() {

	ERR_FAIL_COND_V(singleton, NULL);

	if (create_func)
		return create_func();

	return NULL;
}

RID VisualServer::texture_create_from_image(const Ref<Image> &p_image, uint32_t p_flags) {

	ERR_FAIL_COND_V(!p_image.is_valid(), RID());
	RID texture = texture_create();
	texture_allocate(texture, p_image->get_width(), p_image->get_height(), p_image->get_format(), p_flags); //if it has mipmaps, use, else generate
	ERR_FAIL_COND_V(!texture.is_valid(), texture);

	texture_set_data(texture, p_image);

	return texture;
}

Array VisualServer::_texture_debug_usage_bind() {

	List<TextureInfo> list;
	texture_debug_usage(&list);
	Array arr;
	for (const List<TextureInfo>::Element *E = list.front(); E; E = E->next()) {

		Dictionary dict;
		dict["texture"] = E->get().texture;
		dict["size"] = E->get().size;
		dict["format"] = E->get().format;
		dict["bytes"] = E->get().bytes;
		dict["path"] = E->get().path;
		arr.push_back(dict);
	}
	return arr;
}

Array VisualServer::_shader_get_param_list_bind(RID p_shader) const {

	List<PropertyInfo> l;
	shader_get_param_list(p_shader, &l);
	return convert_property_list(&l);
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

	Ref<Image> data = memnew(Image(TEST_TEXTURE_SIZE, TEST_TEXTURE_SIZE, false, Image::FORMAT_RGB8, test_data));

	test_texture = texture_create_from_image(data);

	return test_texture;
}

void VisualServer::_free_internal_rids() {

	if (test_texture.is_valid())
		free(test_texture);
	if (white_texture.is_valid())
		free(white_texture);
	if (test_material.is_valid())
		free(test_material);
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
	Ref<Image> white = memnew(Image(4, 4, 0, Image::FORMAT_RGB8, wt));
	white_texture = texture_create();
	texture_allocate(white_texture, 4, 4, Image::FORMAT_RGB8);
	texture_set_data(white_texture, white);
	return white_texture;
}

#define SMALL_VEC2 Vector2(0.00001, 0.00001)
#define SMALL_VEC3 Vector3(0.00001, 0.00001, 0.00001)

Error VisualServer::_surface_set_data(Array p_arrays, uint32_t p_format, uint32_t *p_offsets, uint32_t p_stride, PoolVector<uint8_t> &r_vertex_array, int p_vertex_array_len, PoolVector<uint8_t> &r_index_array, int p_index_array_len, Rect3 &r_aabb, Vector<Rect3> r_bone_aabb) {

	PoolVector<uint8_t>::Write vw = r_vertex_array.write();

	PoolVector<uint8_t>::Write iw;
	if (r_index_array.size()) {
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

								aabb = Rect2(src[i], SMALL_VEC2); //must have a bit of size
							} else {

								aabb.expand_to(src[i]);
							}
						}

					} else {
						for (int i = 0; i < p_vertex_array_len; i++) {

							float vector[2] = { src[i].x, src[i].y };

							copymem(&vw[p_offsets[ai] + i * p_stride], vector, sizeof(float) * 2);

							if (i == 0) {

								aabb = Rect2(src[i], SMALL_VEC2); //must have a bit of size
							} else {

								aabb.expand_to(src[i]);
							}
						}
					}

					r_aabb = Rect3(Vector3(aabb.position.x, aabb.position.y, 0), Vector3(aabb.size.x, aabb.size.y, 0));

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

								aabb = Rect3(src[i], SMALL_VEC3);
							} else {

								aabb.expand_to(src[i]);
							}
						}

					} else {
						for (int i = 0; i < p_vertex_array_len; i++) {

							float vector[3] = { src[i].x, src[i].y, src[i].z };

							copymem(&vw[p_offsets[ai] + i * p_stride], vector, sizeof(float) * 3);

							if (i == 0) {

								aabb = Rect3(src[i], SMALL_VEC3);
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

						int8_t vector[4] = {
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
						bptr[idx] = Rect3(v, SMALL_VEC3);
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
					const float multiplier = 1.f / 127.f;

					for (int j = 0; j < p_vertex_len; j++) {

						const int8_t *v = (const int8_t *)&r[j * total_elem_size + offsets[i]];
						w[j] = Vector3(float(v[0]) * multiplier, float(v[1]) * multiplier, float(v[2]) * multiplier);
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

						const int8_t *v = (const int8_t *)&r[j * total_elem_size + offsets[i]];
						for (int k = 0; k < 4; k++) {
							w[j * 4 + k] = float(v[k] / 127.0);
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
						w[j] = Color(float(v[0] / 255.0), float(v[1] / 255.0), float(v[2] / 255.0), float(v[3] / 255.0));
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

Array VisualServer::mesh_surface_get_blend_shape_arrays(RID p_mesh, int p_surface) const {

	Vector<PoolVector<uint8_t> > blend_shape_data = mesh_surface_get_blend_shapes(p_mesh, p_surface);
	if (blend_shape_data.size() > 0) {
		int vertex_len = mesh_surface_get_array_len(p_mesh, p_surface);

		PoolVector<uint8_t> index_data = mesh_surface_get_index_array(p_mesh, p_surface);
		int index_len = mesh_surface_get_array_index_len(p_mesh, p_surface);

		uint32_t format = mesh_surface_get_format(p_mesh, p_surface);

		Array blend_shape_array;
		blend_shape_array.resize(blend_shape_data.size());
		for (int i = 0; i < blend_shape_data.size(); i++) {
			blend_shape_array.set(i, _get_array_from_surface(format, blend_shape_data[i], vertex_len, index_data, index_len));
		}

		return blend_shape_array;
	} else {
		return Array();
	}
}

Array VisualServer::_mesh_surface_get_skeleton_aabb_bind(RID p_mesh, int p_surface) const {

	Vector<Rect3> vec = VS::get_singleton()->mesh_surface_get_skeleton_aabb(p_mesh, p_surface);
	Array arr;
	for (int i = 0; i < vec.size(); i++) {
		arr[i] = vec[i];
	}
	return arr;
}

void VisualServer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("force_sync"), &VisualServer::sync);
	ClassDB::bind_method(D_METHOD("force_draw"), &VisualServer::draw);

	ClassDB::bind_method(D_METHOD("texture_create"), &VisualServer::texture_create);
	ClassDB::bind_method(D_METHOD("texture_create_from_image", "image", "flags"), &VisualServer::texture_create_from_image, DEFVAL(TEXTURE_FLAGS_DEFAULT));
	ClassDB::bind_method(D_METHOD("texture_allocate", "texture", "width", "height", "format", "flags"), &VisualServer::texture_allocate, DEFVAL(TEXTURE_FLAGS_DEFAULT));
	ClassDB::bind_method(D_METHOD("texture_set_data", "texture", "image", "cube_side"), &VisualServer::texture_set_data, DEFVAL(CUBEMAP_LEFT));
	ClassDB::bind_method(D_METHOD("texture_get_data", "texture", "cube_side"), &VisualServer::texture_get_data, DEFVAL(CUBEMAP_LEFT));
	ClassDB::bind_method(D_METHOD("texture_set_flags", "texture", "flags"), &VisualServer::texture_set_flags);
	ClassDB::bind_method(D_METHOD("texture_get_flags", "texture"), &VisualServer::texture_get_flags);
	ClassDB::bind_method(D_METHOD("texture_get_format", "texture"), &VisualServer::texture_get_format);
	ClassDB::bind_method(D_METHOD("texture_get_texid", "texture"), &VisualServer::texture_get_texid);
	ClassDB::bind_method(D_METHOD("texture_get_width", "texture"), &VisualServer::texture_get_width);
	ClassDB::bind_method(D_METHOD("texture_get_height", "texture"), &VisualServer::texture_get_height);
	ClassDB::bind_method(D_METHOD("texture_set_size_override", "texture", "width", "height"), &VisualServer::texture_set_size_override);
	ClassDB::bind_method(D_METHOD("texture_set_path", "texture", "path"), &VisualServer::texture_set_path);
	ClassDB::bind_method(D_METHOD("texture_get_path", "texture"), &VisualServer::texture_get_path);
	ClassDB::bind_method(D_METHOD("texture_set_shrink_all_x2_on_set_data", "shrink"), &VisualServer::texture_set_shrink_all_x2_on_set_data);

	ClassDB::bind_method(D_METHOD("texture_debug_usage"), &VisualServer::_texture_debug_usage_bind);
	ClassDB::bind_method(D_METHOD("textures_keep_original", "enable"), &VisualServer::textures_keep_original);

	ClassDB::bind_method(D_METHOD("sky_create"), &VisualServer::sky_create);
	ClassDB::bind_method(D_METHOD("sky_set_texture", "sky", "cube_map", "radiance_size"), &VisualServer::sky_set_texture);

	ClassDB::bind_method(D_METHOD("shader_create"), &VisualServer::shader_create);
	ClassDB::bind_method(D_METHOD("shader_set_code", "shader", "code"), &VisualServer::shader_set_code);
	ClassDB::bind_method(D_METHOD("shader_get_code", "shader"), &VisualServer::shader_get_code);
	ClassDB::bind_method(D_METHOD("shader_get_param_list", "shader"), &VisualServer::_shader_get_param_list_bind);
	ClassDB::bind_method(D_METHOD("shader_set_default_texture_param", "shader", "name", "texture"), &VisualServer::shader_set_default_texture_param);
	ClassDB::bind_method(D_METHOD("shader_get_default_texture_param", "shader", "name"), &VisualServer::shader_get_default_texture_param);

	ClassDB::bind_method(D_METHOD("material_create"), &VisualServer::material_create);
	ClassDB::bind_method(D_METHOD("material_set_shader", "shader_material", "shader"), &VisualServer::material_set_shader);
	ClassDB::bind_method(D_METHOD("material_get_shader", "shader_material"), &VisualServer::material_get_shader);
	ClassDB::bind_method(D_METHOD("material_set_param", "material", "parameter", "value"), &VisualServer::material_set_param);
	ClassDB::bind_method(D_METHOD("material_get_param", "material", "parameter"), &VisualServer::material_get_param);
	ClassDB::bind_method(D_METHOD("material_set_render_priority", "material", "priority"), &VisualServer::material_set_render_priority);
	ClassDB::bind_method(D_METHOD("material_set_line_width", "material", "width"), &VisualServer::material_set_line_width);
	ClassDB::bind_method(D_METHOD("material_set_next_pass", "material", "next_material"), &VisualServer::material_set_next_pass);

	ClassDB::bind_method(D_METHOD("mesh_create"), &VisualServer::mesh_create);
	ClassDB::bind_method(D_METHOD("mesh_add_surface_from_arrays", "mesh", "primtive", "arrays", "blend_shapes", "compress_format"), &VisualServer::mesh_add_surface_from_arrays, DEFVAL(Array()), DEFVAL(ARRAY_COMPRESS_DEFAULT));
	ClassDB::bind_method(D_METHOD("mesh_set_blend_shape_count", "mesh", "amount"), &VisualServer::mesh_set_blend_shape_count);
	ClassDB::bind_method(D_METHOD("mesh_get_blend_shape_count", "mesh"), &VisualServer::mesh_get_blend_shape_count);
	ClassDB::bind_method(D_METHOD("mesh_set_blend_shape_mode", "mesh", "mode"), &VisualServer::mesh_set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("mesh_get_blend_shape_mode", "mesh"), &VisualServer::mesh_get_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("mesh_surface_set_material", "mesh", "surface", "material"), &VisualServer::mesh_surface_set_material);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_material", "mesh", "surface"), &VisualServer::mesh_surface_get_material);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_array_len", "mesh", "surface"), &VisualServer::mesh_surface_get_array_len);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_array_index_len", "mesh", "surface"), &VisualServer::mesh_surface_get_array_index_len);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_array", "mesh", "surface"), &VisualServer::mesh_surface_get_array);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_index_array", "mesh", "surface"), &VisualServer::mesh_surface_get_index_array);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_arrays", "mesh", "surface"), &VisualServer::mesh_surface_get_arrays);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_blend_shape_arrays", "mesh", "surface"), &VisualServer::mesh_surface_get_blend_shape_arrays);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_format", "mesh", "surface"), &VisualServer::mesh_surface_get_format);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_primitive_type", "mesh", "surface"), &VisualServer::mesh_surface_get_primitive_type);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_aabb", "mesh", "surface"), &VisualServer::mesh_surface_get_aabb);
	ClassDB::bind_method(D_METHOD("mesh_surface_get_skeleton_aabb", "mesh", "surface"), &VisualServer::_mesh_surface_get_skeleton_aabb_bind);
	ClassDB::bind_method(D_METHOD("mesh_remove_surface", "mesh", "index"), &VisualServer::mesh_remove_surface);
	ClassDB::bind_method(D_METHOD("mesh_get_surface_count", "mesh"), &VisualServer::mesh_get_surface_count);
	ClassDB::bind_method(D_METHOD("mesh_set_custom_aabb", "mesh", "aabb"), &VisualServer::mesh_set_custom_aabb);
	ClassDB::bind_method(D_METHOD("mesh_get_custom_aabb", "mesh"), &VisualServer::mesh_get_custom_aabb);
	ClassDB::bind_method(D_METHOD("mesh_clear", "mesh"), &VisualServer::mesh_clear);

	// TODO: multimesh_*, immediate_*, skeleton_*, light_*, reflection_probe_*, gi_probe_*, particles_*, camera_*

	ClassDB::bind_method(D_METHOD("viewport_create"), &VisualServer::viewport_create);
	ClassDB::bind_method(D_METHOD("viewport_set_use_arvr", "viewport", "use_arvr"), &VisualServer::viewport_set_use_arvr);
	ClassDB::bind_method(D_METHOD("viewport_set_size", "viewport", "width", "height"), &VisualServer::viewport_set_size);
	ClassDB::bind_method(D_METHOD("viewport_set_active", "viewport", "active"), &VisualServer::viewport_set_active);
	ClassDB::bind_method(D_METHOD("viewport_set_parent_viewport", "viewport", "parent_viewport"), &VisualServer::viewport_set_parent_viewport);
	ClassDB::bind_method(D_METHOD("viewport_attach_to_screen", "viewport", "rect", "screen"), &VisualServer::viewport_attach_to_screen, DEFVAL(Rect2()), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("viewport_detach", "viewport"), &VisualServer::viewport_detach);
	ClassDB::bind_method(D_METHOD("viewport_set_update_mode", "viewport", "update_mode"), &VisualServer::viewport_set_update_mode);
	ClassDB::bind_method(D_METHOD("viewport_set_vflip", "viewport", "enabled"), &VisualServer::viewport_set_vflip);
	ClassDB::bind_method(D_METHOD("viewport_set_clear_mode", "viewport", "clear_mode"), &VisualServer::viewport_set_clear_mode);
	ClassDB::bind_method(D_METHOD("viewport_get_texture", "viewport"), &VisualServer::viewport_get_texture);
	ClassDB::bind_method(D_METHOD("viewport_set_hide_scenario", "viewport", "hidden"), &VisualServer::viewport_set_hide_scenario);
	ClassDB::bind_method(D_METHOD("viewport_set_hide_canvas", "viewport", "hidden"), &VisualServer::viewport_set_hide_canvas);
	ClassDB::bind_method(D_METHOD("viewport_set_disable_environment", "viewport", "disabled"), &VisualServer::viewport_set_disable_environment);
	ClassDB::bind_method(D_METHOD("viewport_set_disable_3d", "viewport", "disabled"), &VisualServer::viewport_set_disable_3d);
	ClassDB::bind_method(D_METHOD("viewport_attach_camera", "viewport", "camera"), &VisualServer::viewport_attach_camera);
	ClassDB::bind_method(D_METHOD("viewport_set_scenario", "viewport", "scenario"), &VisualServer::viewport_set_scenario);
	ClassDB::bind_method(D_METHOD("viewport_attach_canvas", "viewport", "canvas"), &VisualServer::viewport_attach_canvas);
	ClassDB::bind_method(D_METHOD("viewport_remove_canvas", "viewport", "canvas"), &VisualServer::viewport_remove_canvas);
	ClassDB::bind_method(D_METHOD("viewport_set_canvas_transform", "viewport", "canvas", "offset"), &VisualServer::viewport_set_canvas_transform);
	ClassDB::bind_method(D_METHOD("viewport_set_transparent_background", "viewport", "enabled"), &VisualServer::viewport_set_transparent_background);
	ClassDB::bind_method(D_METHOD("viewport_set_global_canvas_transform", "viewport", "transform"), &VisualServer::viewport_set_global_canvas_transform);
	ClassDB::bind_method(D_METHOD("viewport_set_canvas_layer", "viewport", "canvas", "layer"), &VisualServer::viewport_set_canvas_layer);
	ClassDB::bind_method(D_METHOD("viewport_set_shadow_atlas_size", "viewport", "size"), &VisualServer::viewport_set_shadow_atlas_size);
	ClassDB::bind_method(D_METHOD("viewport_set_shadow_atlas_quadrant_subdivision", "viewport", "quadrant", "subdivision"), &VisualServer::viewport_set_shadow_atlas_quadrant_subdivision);
	ClassDB::bind_method(D_METHOD("viewport_set_msaa", "viewport", "msaa"), &VisualServer::viewport_set_msaa);
	ClassDB::bind_method(D_METHOD("viewport_set_hdr", "viewport", "enabled"), &VisualServer::viewport_set_hdr);
	ClassDB::bind_method(D_METHOD("viewport_set_usage", "viewport", "usage"), &VisualServer::viewport_set_usage);
	ClassDB::bind_method(D_METHOD("viewport_get_render_info", "viewport", "info"), &VisualServer::viewport_get_render_info);
	ClassDB::bind_method(D_METHOD("viewport_set_debug_draw", "viewport", "draw"), &VisualServer::viewport_set_debug_draw);

	// TODO: environment_*, scenario_*, instance_*

	ClassDB::bind_method(D_METHOD("canvas_create"), &VisualServer::canvas_create);
	ClassDB::bind_method(D_METHOD("canvas_set_item_mirroring", "canvas", "item", "mirroring"), &VisualServer::canvas_set_item_mirroring);
	ClassDB::bind_method(D_METHOD("canvas_set_modulate", "canvas", "color"), &VisualServer::canvas_set_modulate);

	ClassDB::bind_method(D_METHOD("canvas_item_create"), &VisualServer::canvas_item_create);
	ClassDB::bind_method(D_METHOD("canvas_item_set_parent", "item", "parent"), &VisualServer::canvas_item_set_parent);
	ClassDB::bind_method(D_METHOD("canvas_item_set_visible", "item", "visible"), &VisualServer::canvas_item_set_visible);
	ClassDB::bind_method(D_METHOD("canvas_item_set_light_mask", "item", "mask"), &VisualServer::canvas_item_set_light_mask);
	ClassDB::bind_method(D_METHOD("canvas_item_set_transform", "item", "transform"), &VisualServer::canvas_item_set_transform);
	ClassDB::bind_method(D_METHOD("canvas_item_set_clip", "item", "clip"), &VisualServer::canvas_item_set_clip);
	ClassDB::bind_method(D_METHOD("canvas_item_set_distance_field_mode", "item", "enabled"), &VisualServer::canvas_item_set_distance_field_mode);
	ClassDB::bind_method(D_METHOD("canvas_item_set_custom_rect", "item", "use_custom_rect", "rect"), &VisualServer::canvas_item_set_custom_rect, DEFVAL(Rect2()));
	ClassDB::bind_method(D_METHOD("canvas_item_set_modulate", "item", "color"), &VisualServer::canvas_item_set_modulate);
	ClassDB::bind_method(D_METHOD("canvas_item_set_self_modulate", "item", "color"), &VisualServer::canvas_item_set_self_modulate);
	ClassDB::bind_method(D_METHOD("canvas_item_set_draw_behind_parent", "item", "enabled"), &VisualServer::canvas_item_set_draw_behind_parent);
	ClassDB::bind_method(D_METHOD("canvas_item_add_line", "item", "from", "to", "color", "width", "antialiased"), &VisualServer::canvas_item_add_line, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("canvas_item_add_polyline", "item", "points", "colors", "width", "antialiased"), &VisualServer::canvas_item_add_polyline, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("canvas_item_add_rect", "item", "rect", "color"), &VisualServer::canvas_item_add_rect);
	ClassDB::bind_method(D_METHOD("canvas_item_add_circle", "item", "pos", "radius", "color"), &VisualServer::canvas_item_add_circle);
	ClassDB::bind_method(D_METHOD("canvas_item_add_texture_rect", "item", "rect", "texture", "tile", "modulate", "transpose", "normal_map"), &VisualServer::canvas_item_add_texture_rect, DEFVAL(false), DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("canvas_item_add_texture_rect_region", "item", "rect", "texture", "src_rect", "modulate", "transpose", "normal_map", "clip_uv"), &VisualServer::canvas_item_add_texture_rect_region, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(RID()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("canvas_item_add_nine_patch", "item", "rect", "source", "texture", "topleft", "bottomright", "x_axis_mode", "y_axis_mode", "draw_center", "modulate", "normal_map"), &VisualServer::canvas_item_add_nine_patch, DEFVAL(NINE_PATCH_STRETCH), DEFVAL(NINE_PATCH_STRETCH), DEFVAL(true), DEFVAL(Color(1, 1, 1)), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("canvas_item_add_primitive", "item", "points", "colors", "uvs", "texture", "width", "normal_map"), &VisualServer::canvas_item_add_primitive, DEFVAL(1.0), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("canvas_item_add_polygon", "item", "points", "colors", "uvs", "texture", "normal_map", "antialiased"), &VisualServer::canvas_item_add_polygon, DEFVAL(Vector<Point2>()), DEFVAL(RID()), DEFVAL(RID()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("canvas_item_add_triangle_array", "item", "indices", "points", "colors", "uvs", "texture", "count", "normal_map"), &VisualServer::canvas_item_add_triangle_array, DEFVAL(Vector<Point2>()), DEFVAL(RID()), DEFVAL(-1), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("canvas_item_add_mesh", "item", "mesh", "skeleton"), &VisualServer::canvas_item_add_mesh, DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("canvas_item_add_multimesh", "item", "mesh", "skeleton"), &VisualServer::canvas_item_add_multimesh, DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("canvas_item_add_particles", "item", "particles", "texture", "normal_map", "h_frames", "v_frames"), &VisualServer::canvas_item_add_particles);
	ClassDB::bind_method(D_METHOD("canvas_item_add_set_transform", "item", "transform"), &VisualServer::canvas_item_add_set_transform);
	ClassDB::bind_method(D_METHOD("canvas_item_add_clip_ignore", "item", "ignore"), &VisualServer::canvas_item_add_clip_ignore);
	ClassDB::bind_method(D_METHOD("canvas_item_set_sort_children_by_y", "item", "enabled"), &VisualServer::canvas_item_set_sort_children_by_y);
	ClassDB::bind_method(D_METHOD("canvas_item_set_z", "item", "z"), &VisualServer::canvas_item_set_z);
	ClassDB::bind_method(D_METHOD("canvas_item_set_z_as_relative_to_parent", "item", "enabled"), &VisualServer::canvas_item_set_z_as_relative_to_parent);
	ClassDB::bind_method(D_METHOD("canvas_item_set_copy_to_backbuffer", "item", "enabled", "rect"), &VisualServer::canvas_item_set_copy_to_backbuffer);
	ClassDB::bind_method(D_METHOD("canvas_item_clear", "item"), &VisualServer::canvas_item_clear);
	ClassDB::bind_method(D_METHOD("canvas_item_set_draw_index", "item", "index"), &VisualServer::canvas_item_set_draw_index);
	ClassDB::bind_method(D_METHOD("canvas_item_set_material", "item", "material"), &VisualServer::canvas_item_set_material);
	ClassDB::bind_method(D_METHOD("canvas_item_set_use_parent_material", "item", "enabled"), &VisualServer::canvas_item_set_use_parent_material);
	ClassDB::bind_method(D_METHOD("canvas_light_create"), &VisualServer::canvas_light_create);
	ClassDB::bind_method(D_METHOD("canvas_light_attach_to_canvas", "light", "canvas"), &VisualServer::canvas_light_attach_to_canvas);
	ClassDB::bind_method(D_METHOD("canvas_light_set_enabled", "light", "enabled"), &VisualServer::canvas_light_set_enabled);
	ClassDB::bind_method(D_METHOD("canvas_light_set_scale", "light", "scale"), &VisualServer::canvas_light_set_scale);
	ClassDB::bind_method(D_METHOD("canvas_light_set_transform", "light", "transform"), &VisualServer::canvas_light_set_transform);
	ClassDB::bind_method(D_METHOD("canvas_light_set_texture", "light", "texture"), &VisualServer::canvas_light_set_texture);
	ClassDB::bind_method(D_METHOD("canvas_light_set_texture_offset", "light", "offset"), &VisualServer::canvas_light_set_texture_offset);
	ClassDB::bind_method(D_METHOD("canvas_light_set_color", "light", "color"), &VisualServer::canvas_light_set_color);
	ClassDB::bind_method(D_METHOD("canvas_light_set_height", "light", "height"), &VisualServer::canvas_light_set_height);
	ClassDB::bind_method(D_METHOD("canvas_light_set_energy", "light", "energy"), &VisualServer::canvas_light_set_energy);
	ClassDB::bind_method(D_METHOD("canvas_light_set_z_range", "light", "min_z", "max_z"), &VisualServer::canvas_light_set_z_range);
	ClassDB::bind_method(D_METHOD("canvas_light_set_layer_range", "light", "min_layer", "max_layer"), &VisualServer::canvas_light_set_layer_range);
	ClassDB::bind_method(D_METHOD("canvas_light_set_item_cull_mask", "light", "mask"), &VisualServer::canvas_light_set_item_cull_mask);
	ClassDB::bind_method(D_METHOD("canvas_light_set_item_shadow_cull_mask", "light", "mask"), &VisualServer::canvas_light_set_item_shadow_cull_mask);
	ClassDB::bind_method(D_METHOD("canvas_light_set_mode", "light", "mode"), &VisualServer::canvas_light_set_mode);
	ClassDB::bind_method(D_METHOD("canvas_light_set_shadow_enabled", "light", "enabled"), &VisualServer::canvas_light_set_shadow_enabled);
	ClassDB::bind_method(D_METHOD("canvas_light_set_shadow_buffer_size", "light", "size"), &VisualServer::canvas_light_set_shadow_buffer_size);
	ClassDB::bind_method(D_METHOD("canvas_light_set_shadow_gradient_length", "light", "length"), &VisualServer::canvas_light_set_shadow_gradient_length);
	ClassDB::bind_method(D_METHOD("canvas_light_set_shadow_filter", "light", "filter"), &VisualServer::canvas_light_set_shadow_filter);
	ClassDB::bind_method(D_METHOD("canvas_light_set_shadow_color", "light", "color"), &VisualServer::canvas_light_set_shadow_color);
	ClassDB::bind_method(D_METHOD("canvas_light_set_shadow_smooth", "light", "smooth"), &VisualServer::canvas_light_set_shadow_smooth);

	ClassDB::bind_method(D_METHOD("canvas_light_occluder_create"), &VisualServer::canvas_light_occluder_create);
	ClassDB::bind_method(D_METHOD("canvas_light_occluder_attach_to_canvas", "occluder", "canvas"), &VisualServer::canvas_light_occluder_attach_to_canvas);
	ClassDB::bind_method(D_METHOD("canvas_light_occluder_set_enabled", "occluder", "enabled"), &VisualServer::canvas_light_occluder_set_enabled);
	ClassDB::bind_method(D_METHOD("canvas_light_occluder_set_polygon", "occluder", "polygon"), &VisualServer::canvas_light_occluder_set_polygon);
	ClassDB::bind_method(D_METHOD("canvas_light_occluder_set_transform", "occluder", "transform"), &VisualServer::canvas_light_occluder_set_transform);
	ClassDB::bind_method(D_METHOD("canvas_light_occluder_set_light_mask", "occluder", "mask"), &VisualServer::canvas_light_occluder_set_light_mask);

	ClassDB::bind_method(D_METHOD("canvas_occluder_polygon_create"), &VisualServer::canvas_occluder_polygon_create);
	ClassDB::bind_method(D_METHOD("canvas_occluder_polygon_set_shape", "occluder_polygon", "shape", "closed"), &VisualServer::canvas_occluder_polygon_set_shape);
	ClassDB::bind_method(D_METHOD("canvas_occluder_polygon_set_shape_as_lines", "occluder_polygon", "shape"), &VisualServer::canvas_occluder_polygon_set_shape_as_lines);
	ClassDB::bind_method(D_METHOD("canvas_occluder_polygon_set_cull_mode", "occluder_polygon", "mode"), &VisualServer::canvas_occluder_polygon_set_cull_mode);

	ClassDB::bind_method(D_METHOD("black_bars_set_margins", "left", "top", "right", "bottom"), &VisualServer::black_bars_set_margins);
	ClassDB::bind_method(D_METHOD("black_bars_set_images", "left", "top", "right", "bottom"), &VisualServer::black_bars_set_images);

	ClassDB::bind_method(D_METHOD("free", "rid"), &VisualServer::free);

	ClassDB::bind_method(D_METHOD("request_frame_drawn_callback", "where", "method", "userdata"), &VisualServer::request_frame_drawn_callback);
	ClassDB::bind_method(D_METHOD("draw"), &VisualServer::draw);
	ClassDB::bind_method(D_METHOD("sync"), &VisualServer::sync);
	ClassDB::bind_method(D_METHOD("has_changed"), &VisualServer::has_changed);
	ClassDB::bind_method(D_METHOD("init"), &VisualServer::init);
	ClassDB::bind_method(D_METHOD("finish"), &VisualServer::finish);
	ClassDB::bind_method(D_METHOD("get_render_info", "info"), &VisualServer::get_render_info);

	ClassDB::bind_method(D_METHOD("get_test_cube"), &VisualServer::get_test_cube);
	ClassDB::bind_method(D_METHOD("get_test_texture"), &VisualServer::get_test_texture);
	ClassDB::bind_method(D_METHOD("get_white_texture"), &VisualServer::get_white_texture);

	ClassDB::bind_method(D_METHOD("make_sphere_mesh", "latitudes", "longitudes", "radius"), &VisualServer::make_sphere_mesh);

	ClassDB::bind_method(D_METHOD("set_boot_image", "image", "color", "scale"), &VisualServer::set_boot_image);
	ClassDB::bind_method(D_METHOD("set_default_clear_color", "color"), &VisualServer::set_default_clear_color);

	ClassDB::bind_method(D_METHOD("has_feature", "feature"), &VisualServer::has_feature);
	ClassDB::bind_method(D_METHOD("has_os_feature", "feature"), &VisualServer::has_os_feature);
	ClassDB::bind_method(D_METHOD("set_debug_generate_wireframes", "generate"), &VisualServer::set_debug_generate_wireframes);

	BIND_CONSTANT(NO_INDEX_ARRAY);
	BIND_CONSTANT(ARRAY_WEIGHTS_SIZE);
	BIND_CONSTANT(CANVAS_ITEM_Z_MIN);
	BIND_CONSTANT(CANVAS_ITEM_Z_MAX);
	BIND_CONSTANT(MAX_GLOW_LEVELS);
	BIND_CONSTANT(MAX_CURSORS);
	BIND_CONSTANT(MATERIAL_RENDER_PRIORITY_MIN);
	BIND_CONSTANT(MATERIAL_RENDER_PRIORITY_MAX);

	BIND_ENUM_CONSTANT(CUBEMAP_LEFT);
	BIND_ENUM_CONSTANT(CUBEMAP_RIGHT);
	BIND_ENUM_CONSTANT(CUBEMAP_BOTTOM);
	BIND_ENUM_CONSTANT(CUBEMAP_TOP);
	BIND_ENUM_CONSTANT(CUBEMAP_FRONT);
	BIND_ENUM_CONSTANT(CUBEMAP_BACK);

	BIND_ENUM_CONSTANT(TEXTURE_FLAG_MIPMAPS);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_REPEAT);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_FILTER);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_ANISOTROPIC_FILTER);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_CONVERT_TO_LINEAR);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_MIRRORED_REPEAT);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_CUBEMAP);
	BIND_ENUM_CONSTANT(TEXTURE_FLAG_USED_FOR_STREAMING);
	BIND_ENUM_CONSTANT(TEXTURE_FLAGS_DEFAULT);

	BIND_ENUM_CONSTANT(SHADER_SPATIAL);
	BIND_ENUM_CONSTANT(SHADER_CANVAS_ITEM);
	BIND_ENUM_CONSTANT(SHADER_PARTICLES);
	BIND_ENUM_CONSTANT(SHADER_MAX);

	BIND_ENUM_CONSTANT(ARRAY_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_BONES);
	BIND_ENUM_CONSTANT(ARRAY_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_INDEX);
	BIND_ENUM_CONSTANT(ARRAY_MAX);

	BIND_ENUM_CONSTANT(ARRAY_FORMAT_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_BONES);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_INDEX);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_BONES);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_INDEX);
	BIND_ENUM_CONSTANT(ARRAY_FLAG_USE_2D_VERTICES);
	BIND_ENUM_CONSTANT(ARRAY_FLAG_USE_16_BIT_BONES);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_DEFAULT);

	BIND_ENUM_CONSTANT(PRIMITIVE_POINTS);
	BIND_ENUM_CONSTANT(PRIMITIVE_LINES);
	BIND_ENUM_CONSTANT(PRIMITIVE_LINE_STRIP);
	BIND_ENUM_CONSTANT(PRIMITIVE_LINE_LOOP);
	BIND_ENUM_CONSTANT(PRIMITIVE_TRIANGLES);
	BIND_ENUM_CONSTANT(PRIMITIVE_TRIANGLE_STRIP);
	BIND_ENUM_CONSTANT(PRIMITIVE_TRIANGLE_FAN);
	BIND_ENUM_CONSTANT(PRIMITIVE_MAX);

	BIND_ENUM_CONSTANT(BLEND_SHAPE_MODE_NORMALIZED);
	BIND_ENUM_CONSTANT(BLEND_SHAPE_MODE_RELATIVE);

	BIND_ENUM_CONSTANT(LIGHT_DIRECTIONAL);
	BIND_ENUM_CONSTANT(LIGHT_OMNI);
	BIND_ENUM_CONSTANT(LIGHT_SPOT);

	BIND_ENUM_CONSTANT(LIGHT_PARAM_ENERGY);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SPECULAR);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_RANGE);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_ATTENUATION);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SPOT_ANGLE);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SPOT_ATTENUATION);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_CONTACT_SHADOW_SIZE);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_MAX_DISTANCE);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_NORMAL_BIAS);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_BIAS);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_SHADOW_BIAS_SPLIT_SCALE);
	BIND_ENUM_CONSTANT(LIGHT_PARAM_MAX);

	BIND_ENUM_CONSTANT(VIEWPORT_UPDATE_DISABLED);
	BIND_ENUM_CONSTANT(VIEWPORT_UPDATE_ONCE);
	BIND_ENUM_CONSTANT(VIEWPORT_UPDATE_WHEN_VISIBLE);
	BIND_ENUM_CONSTANT(VIEWPORT_UPDATE_ALWAYS);

	BIND_ENUM_CONSTANT(VIEWPORT_CLEAR_ALWAYS);
	BIND_ENUM_CONSTANT(VIEWPORT_CLEAR_NEVER);
	BIND_ENUM_CONSTANT(VIEWPORT_CLEAR_ONLY_NEXT_FRAME);

	BIND_ENUM_CONSTANT(VIEWPORT_MSAA_DISABLED);
	BIND_ENUM_CONSTANT(VIEWPORT_MSAA_2X);
	BIND_ENUM_CONSTANT(VIEWPORT_MSAA_4X);
	BIND_ENUM_CONSTANT(VIEWPORT_MSAA_8X);
	BIND_ENUM_CONSTANT(VIEWPORT_MSAA_16X);

	BIND_ENUM_CONSTANT(VIEWPORT_USAGE_2D);
	BIND_ENUM_CONSTANT(VIEWPORT_USAGE_2D_NO_SAMPLING);
	BIND_ENUM_CONSTANT(VIEWPORT_USAGE_3D);
	BIND_ENUM_CONSTANT(VIEWPORT_USAGE_3D_NO_EFFECTS);

	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_VERTICES_IN_FRAME);
	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_MATERIAL_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_SHADER_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_SURFACE_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(VIEWPORT_RENDER_INFO_MAX);

	BIND_ENUM_CONSTANT(VIEWPORT_DEBUG_DRAW_DISABLED);
	BIND_ENUM_CONSTANT(VIEWPORT_DEBUG_DRAW_UNSHADED);
	BIND_ENUM_CONSTANT(VIEWPORT_DEBUG_DRAW_OVERDRAW);
	BIND_ENUM_CONSTANT(VIEWPORT_DEBUG_DRAW_WIREFRAME);

	BIND_ENUM_CONSTANT(SCENARIO_DEBUG_DISABLED);
	BIND_ENUM_CONSTANT(SCENARIO_DEBUG_WIREFRAME);
	BIND_ENUM_CONSTANT(SCENARIO_DEBUG_OVERDRAW);
	BIND_ENUM_CONSTANT(SCENARIO_DEBUG_SHADELESS);

	BIND_ENUM_CONSTANT(INSTANCE_NONE);
	BIND_ENUM_CONSTANT(INSTANCE_MESH);
	BIND_ENUM_CONSTANT(INSTANCE_MULTIMESH);
	BIND_ENUM_CONSTANT(INSTANCE_IMMEDIATE);
	BIND_ENUM_CONSTANT(INSTANCE_PARTICLES);
	BIND_ENUM_CONSTANT(INSTANCE_LIGHT);
	BIND_ENUM_CONSTANT(INSTANCE_REFLECTION_PROBE);
	BIND_ENUM_CONSTANT(INSTANCE_GI_PROBE);
	BIND_ENUM_CONSTANT(INSTANCE_MAX);
	BIND_ENUM_CONSTANT(INSTANCE_GEOMETRY_MASK);

	BIND_ENUM_CONSTANT(NINE_PATCH_STRETCH);
	BIND_ENUM_CONSTANT(NINE_PATCH_TILE);
	BIND_ENUM_CONSTANT(NINE_PATCH_TILE_FIT);

	BIND_ENUM_CONSTANT(CANVAS_LIGHT_MODE_ADD);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_MODE_SUB);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_MODE_MIX);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_MODE_MASK);

	BIND_ENUM_CONSTANT(CANVAS_LIGHT_FILTER_NONE);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_FILTER_PCF3);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_FILTER_PCF5);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_FILTER_PCF7);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_FILTER_PCF9);
	BIND_ENUM_CONSTANT(CANVAS_LIGHT_FILTER_PCF13);

	BIND_ENUM_CONSTANT(CANVAS_OCCLUDER_POLYGON_CULL_DISABLED);
	BIND_ENUM_CONSTANT(CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE);
	BIND_ENUM_CONSTANT(CANVAS_OCCLUDER_POLYGON_CULL_COUNTER_CLOCKWISE);

	BIND_ENUM_CONSTANT(INFO_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(INFO_VERTICES_IN_FRAME);
	BIND_ENUM_CONSTANT(INFO_MATERIAL_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(INFO_SHADER_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(INFO_SURFACE_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(INFO_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(INFO_USAGE_VIDEO_MEM_TOTAL);
	BIND_ENUM_CONSTANT(INFO_VIDEO_MEM_USED);
	BIND_ENUM_CONSTANT(INFO_TEXTURE_MEM_USED);
	BIND_ENUM_CONSTANT(INFO_VERTEX_MEM_USED);

	BIND_ENUM_CONSTANT(FEATURE_SHADERS);
	BIND_ENUM_CONSTANT(FEATURE_MULTITHREADED);
}

void VisualServer::_canvas_item_add_style_box(RID p_item, const Rect2 &p_rect, const Rect2 &p_source, RID p_texture, const Vector<float> &p_margins, const Color &p_modulate) {

	ERR_FAIL_COND(p_margins.size() != 4);
	//canvas_item_add_style_box(p_item,p_rect,p_source,p_texture,Vector2(p_margins[0],p_margins[1]),Vector2(p_margins[2],p_margins[3]),true,p_modulate);
}

void VisualServer::_camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far) {

	camera_set_orthogonal(p_camera, p_size, p_z_near, p_z_far);
}

void VisualServer::mesh_add_surface_from_mesh_data(RID p_mesh, const Geometry::MeshData &p_mesh_data) {

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
	GLOBAL_DEF("rendering/vram_compression/import_s3tc", true);
	GLOBAL_DEF("rendering/vram_compression/import_etc", false);
	GLOBAL_DEF("rendering/vram_compression/import_etc2", true);
	GLOBAL_DEF("rendering/vram_compression/import_pvrtc", false);

	GLOBAL_DEF("rendering/quality/directional_shadow/size", 4096);
	GLOBAL_DEF("rendering/quality/directional_shadow/size.mobile", 2048);
	GLOBAL_DEF("rendering/quality/shadow_atlas/size", 4096);
	GLOBAL_DEF("rendering/quality/shadow_atlas/size.mobile", 2048);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/shadow_atlas/size", PropertyInfo(Variant::INT, "rendering/quality/shadow_atlas/size", PROPERTY_HINT_RANGE, "256,16384"));
	GLOBAL_DEF("rendering/quality/shadow_atlas/quadrant_0_subdiv", 1);
	GLOBAL_DEF("rendering/quality/shadow_atlas/quadrant_1_subdiv", 2);
	GLOBAL_DEF("rendering/quality/shadow_atlas/quadrant_2_subdiv", 3);
	GLOBAL_DEF("rendering/quality/shadow_atlas/quadrant_3_subdiv", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/shadow_atlas/quadrant_0_subdiv", PropertyInfo(Variant::INT, "rendering/quality/shadow_atlas/quadrant_0_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/shadow_atlas/quadrant_1_subdiv", PropertyInfo(Variant::INT, "rendering/quality/shadow_atlas/quadrant_1_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/shadow_atlas/quadrant_2_subdiv", PropertyInfo(Variant::INT, "rendering/quality/shadow_atlas/quadrant_2_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/shadow_atlas/quadrant_3_subdiv", PropertyInfo(Variant::INT, "rendering/quality/shadow_atlas/quadrant_3_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));

	GLOBAL_DEF("rendering/quality/shadows/filter_mode", 1);
	GLOBAL_DEF("rendering/quality/shadows/filter_mode.mobile", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/shadows/filter_mode", PropertyInfo(Variant::INT, "rendering/quality/shadows/filter_mode", PROPERTY_HINT_ENUM, "Disabled,PCF5,PCF13"));

	GLOBAL_DEF("rendering/quality/reflections/texture_array_reflections", true);
	GLOBAL_DEF("rendering/quality/reflections/texture_array_reflections.mobile", false);
	GLOBAL_DEF("rendering/quality/reflections/high_quality_ggx", true);
	GLOBAL_DEF("rendering/quality/reflections/high_quality_ggx.mobile", false);

	GLOBAL_DEF("rendering/quality/shading/force_vertex_shading", false);
	GLOBAL_DEF("rendering/quality/shading/force_vertex_shading.mobile", true);

	GLOBAL_DEF("rendering/quality/depth_prepass/enable", true);
	GLOBAL_DEF("rendering/quality/depth_prepass/disable_for_vendors", "PowerVR,Mali,Adreno");
}

VisualServer::~VisualServer() {

	singleton = NULL;
}
