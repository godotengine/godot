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
#include "globals.h"
#include "method_bind_ext.inc"

VisualServer *VisualServer::singleton = NULL;
VisualServer *(*VisualServer::create_func)() = NULL;

VisualServer *VisualServer::get_singleton() {

	return singleton;
}

void VisualServer::set_mipmap_policy(MipMapPolicy p_policy) {

	mm_policy = p_policy;
}

VisualServer::MipMapPolicy VisualServer::get_mipmap_policy() const {

	return (VisualServer::MipMapPolicy)mm_policy;
}

DVector<String> VisualServer::_shader_get_param_list(RID p_shader) const {

	//remove at some point

	DVector<String> pl;

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

	if (test_texture) {
		return test_texture;
	};

#define TEST_TEXTURE_SIZE 256

	Image data(TEST_TEXTURE_SIZE, TEST_TEXTURE_SIZE, 0, Image::FORMAT_RGB);

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

			data.put_pixel(x, y, c);
		}
	}

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

	for (int i = 0; i < 32; i++) {
		if (material_2d[i].is_valid())
			free(material_2d[i]);
	}
}

RID VisualServer::_make_test_cube() {

	DVector<Vector3> vertices;
	DVector<Vector3> normals;
	DVector<float> tangents;
	DVector<Vector3> uvs;

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

	DVector<int> indices;
	indices.resize(vertices.size());
	for (int i = 0; i < vertices.size(); i++)
		indices.set(i, i);
	d[VisualServer::ARRAY_INDEX] = indices;

	mesh_add_surface(test_cube, PRIMITIVE_TRIANGLES, d);

	test_material = fixed_material_create();
	//material_set_flag(material, MATERIAL_FLAG_BILLBOARD_TOGGLE,true);
	fixed_material_set_texture(test_material, FIXED_MATERIAL_PARAM_DIFFUSE, get_test_texture());
	fixed_material_set_param(test_material, FIXED_MATERIAL_PARAM_SPECULAR_EXP, 70);
	fixed_material_set_param(test_material, FIXED_MATERIAL_PARAM_EMISSION, Color(0.2, 0.2, 0.2));

	fixed_material_set_param(test_material, FIXED_MATERIAL_PARAM_DIFFUSE, Color(1, 1, 1));
	fixed_material_set_param(test_material, FIXED_MATERIAL_PARAM_SPECULAR, Color(1, 1, 1));

	mesh_surface_set_material(test_cube, 0, test_material);

	return test_cube;
}

RID VisualServer::make_sphere_mesh(int p_lats, int p_lons, float p_radius) {

	DVector<Vector3> vertices;
	DVector<Vector3> normals;

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

	mesh_add_surface(mesh, PRIMITIVE_TRIANGLES, d);

	return mesh;
}

RID VisualServer::material_2d_get(bool p_shaded, bool p_transparent, bool p_double_sided, bool p_cut_alpha, bool p_opaque_prepass) {

	int version = 0;
	if (p_shaded)
		version = 1;
	if (p_transparent)
		version |= 2;
	if (p_cut_alpha)
		version |= 4;
	if (p_opaque_prepass)
		version |= 8;
	if (p_double_sided)
		version |= 16;
	if (material_2d[version].is_valid())
		return material_2d[version];

	//not valid, make

	material_2d[version] = fixed_material_create();
	fixed_material_set_flag(material_2d[version], FIXED_MATERIAL_FLAG_USE_ALPHA, p_transparent);
	fixed_material_set_flag(material_2d[version], FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY, true);
	fixed_material_set_flag(material_2d[version], FIXED_MATERIAL_FLAG_DISCARD_ALPHA, p_cut_alpha);
	material_set_flag(material_2d[version], MATERIAL_FLAG_UNSHADED, !p_shaded);
	material_set_flag(material_2d[version], MATERIAL_FLAG_DOUBLE_SIDED, p_double_sided);
	material_set_depth_draw_mode(material_2d[version], p_opaque_prepass ? MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA : MATERIAL_DEPTH_DRAW_OPAQUE_ONLY);
	fixed_material_set_texture(material_2d[version], FIXED_MATERIAL_PARAM_DIFFUSE, get_white_texture());
	//material cut alpha?
	return material_2d[version];
}

RID VisualServer::get_white_texture() {

	if (white_texture.is_valid())
		return white_texture;

	DVector<uint8_t> wt;
	wt.resize(16 * 3);
	{
		DVector<uint8_t>::Write w = wt.write();
		for (int i = 0; i < 16 * 3; i++)
			w[i] = 255;
	}
	Image white(4, 4, 0, Image::FORMAT_RGB, wt);
	white_texture = texture_create();
	texture_allocate(white_texture, 4, 4, Image::FORMAT_RGB);
	texture_set_data(white_texture, white);
	return white_texture;
}

void VisualServer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("texture_create"), &VisualServer::texture_create);
	ObjectTypeDB::bind_method(_MD("texture_create_from_image"), &VisualServer::texture_create_from_image, DEFVAL(TEXTURE_FLAGS_DEFAULT));
	//ObjectTypeDB::bind_method(_MD("texture_allocate"),&VisualServer::texture_allocate,DEFVAL( TEXTURE_FLAGS_DEFAULT ) );
	//ObjectTypeDB::bind_method(_MD("texture_set_data"),&VisualServer::texture_blit_rect,DEFVAL( CUBEMAP_LEFT ) );
	//ObjectTypeDB::bind_method(_MD("texture_get_rect"),&VisualServer::texture_get_rect );
	ObjectTypeDB::bind_method(_MD("texture_set_flags"), &VisualServer::texture_set_flags);
	ObjectTypeDB::bind_method(_MD("texture_get_flags"), &VisualServer::texture_get_flags);
	ObjectTypeDB::bind_method(_MD("texture_get_width"), &VisualServer::texture_get_width);
	ObjectTypeDB::bind_method(_MD("texture_get_height"), &VisualServer::texture_get_height);

	ObjectTypeDB::bind_method(_MD("texture_set_shrink_all_x2_on_set_data", "shrink"), &VisualServer::texture_set_shrink_all_x2_on_set_data);

#ifndef _3D_DISABLED

	ObjectTypeDB::bind_method(_MD("shader_create", "mode"), &VisualServer::shader_create, DEFVAL(SHADER_MATERIAL));
	ObjectTypeDB::bind_method(_MD("shader_set_mode", "shader", "mode"), &VisualServer::shader_set_mode);

	ObjectTypeDB::bind_method(_MD("material_create"), &VisualServer::material_create);

	ObjectTypeDB::bind_method(_MD("material_set_shader", "shader"), &VisualServer::material_set_shader);
	ObjectTypeDB::bind_method(_MD("material_get_shader"), &VisualServer::material_get_shader);

	ObjectTypeDB::bind_method(_MD("material_set_param"), &VisualServer::material_set_param);
	ObjectTypeDB::bind_method(_MD("material_get_param"), &VisualServer::material_get_param);
	ObjectTypeDB::bind_method(_MD("material_set_flag"), &VisualServer::material_set_flag);
	ObjectTypeDB::bind_method(_MD("material_get_flag"), &VisualServer::material_get_flag);
	ObjectTypeDB::bind_method(_MD("material_set_blend_mode"), &VisualServer::material_set_blend_mode);
	ObjectTypeDB::bind_method(_MD("material_get_blend_mode"), &VisualServer::material_get_blend_mode);
	ObjectTypeDB::bind_method(_MD("material_set_line_width"), &VisualServer::material_set_line_width);
	ObjectTypeDB::bind_method(_MD("material_get_line_width"), &VisualServer::material_get_line_width);

	ObjectTypeDB::bind_method(_MD("mesh_create"), &VisualServer::mesh_create);
	ObjectTypeDB::bind_method(_MD("mesh_add_surface"), &VisualServer::mesh_add_surface, DEFVAL(Array()), DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("mesh_surface_set_material"), &VisualServer::mesh_surface_set_material, DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("mesh_surface_get_material"), &VisualServer::mesh_surface_get_material);

	ObjectTypeDB::bind_method(_MD("mesh_surface_get_array_len"), &VisualServer::mesh_surface_get_array_len);
	ObjectTypeDB::bind_method(_MD("mesh_surface_get_array_index_len"), &VisualServer::mesh_surface_get_array_index_len);
	ObjectTypeDB::bind_method(_MD("mesh_surface_get_format"), &VisualServer::mesh_surface_get_format);
	ObjectTypeDB::bind_method(_MD("mesh_surface_get_primitive_type"), &VisualServer::mesh_surface_get_primitive_type);

	ObjectTypeDB::bind_method(_MD("mesh_remove_surface"), &VisualServer::mesh_remove_surface);
	ObjectTypeDB::bind_method(_MD("mesh_get_surface_count"), &VisualServer::mesh_get_surface_count);

	ObjectTypeDB::bind_method(_MD("multimesh_create"), &VisualServer::multimesh_create);
	ObjectTypeDB::bind_method(_MD("multimesh_set_mesh"), &VisualServer::multimesh_set_mesh);
	ObjectTypeDB::bind_method(_MD("multimesh_set_aabb"), &VisualServer::multimesh_set_aabb);
	ObjectTypeDB::bind_method(_MD("multimesh_instance_set_transform"), &VisualServer::multimesh_instance_set_transform);
	ObjectTypeDB::bind_method(_MD("multimesh_instance_set_color"), &VisualServer::multimesh_instance_set_color);
	ObjectTypeDB::bind_method(_MD("multimesh_get_mesh"), &VisualServer::multimesh_get_mesh);
	ObjectTypeDB::bind_method(_MD("multimesh_get_aabb"), &VisualServer::multimesh_get_aabb);
	ObjectTypeDB::bind_method(_MD("multimesh_instance_get_transform"), &VisualServer::multimesh_instance_get_transform);
	ObjectTypeDB::bind_method(_MD("multimesh_instance_get_color"), &VisualServer::multimesh_instance_get_color);

	ObjectTypeDB::bind_method(_MD("particles_create"), &VisualServer::particles_create);
	ObjectTypeDB::bind_method(_MD("particles_set_amount"), &VisualServer::particles_set_amount);
	ObjectTypeDB::bind_method(_MD("particles_get_amount"), &VisualServer::particles_get_amount);
	ObjectTypeDB::bind_method(_MD("particles_set_emitting"), &VisualServer::particles_set_emitting);
	ObjectTypeDB::bind_method(_MD("particles_is_emitting"), &VisualServer::particles_is_emitting);
	ObjectTypeDB::bind_method(_MD("particles_set_visibility_aabb"), &VisualServer::particles_set_visibility_aabb);
	ObjectTypeDB::bind_method(_MD("particles_get_visibility_aabb"), &VisualServer::particles_get_visibility_aabb);
	ObjectTypeDB::bind_method(_MD("particles_set_variable"), &VisualServer::particles_set_variable);
	ObjectTypeDB::bind_method(_MD("particles_get_variable"), &VisualServer::particles_get_variable);
	ObjectTypeDB::bind_method(_MD("particles_set_randomness"), &VisualServer::particles_set_randomness);
	ObjectTypeDB::bind_method(_MD("particles_get_randomness"), &VisualServer::particles_get_randomness);
	ObjectTypeDB::bind_method(_MD("particles_set_color_phases"), &VisualServer::particles_set_color_phases);
	ObjectTypeDB::bind_method(_MD("particles_get_color_phases"), &VisualServer::particles_get_color_phases);
	ObjectTypeDB::bind_method(_MD("particles_set_color_phase_pos"), &VisualServer::particles_set_color_phase_pos);
	ObjectTypeDB::bind_method(_MD("particles_get_color_phase_pos"), &VisualServer::particles_get_color_phase_pos);
	ObjectTypeDB::bind_method(_MD("particles_set_color_phase_color"), &VisualServer::particles_set_color_phase_color);
	ObjectTypeDB::bind_method(_MD("particles_get_color_phase_color"), &VisualServer::particles_get_color_phase_color);
	ObjectTypeDB::bind_method(_MD("particles_set_attractors"), &VisualServer::particles_set_attractors);
	ObjectTypeDB::bind_method(_MD("particles_get_attractors"), &VisualServer::particles_get_attractors);
	ObjectTypeDB::bind_method(_MD("particles_set_attractor_pos"), &VisualServer::particles_set_attractor_pos);
	ObjectTypeDB::bind_method(_MD("particles_get_attractor_pos"), &VisualServer::particles_get_attractor_pos);
	ObjectTypeDB::bind_method(_MD("particles_set_attractor_strength"), &VisualServer::particles_set_attractor_strength);
	ObjectTypeDB::bind_method(_MD("particles_get_attractor_strength"), &VisualServer::particles_get_attractor_strength);
	ObjectTypeDB::bind_method(_MD("particles_set_material"), &VisualServer::particles_set_material, DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("particles_set_height_from_velocity"), &VisualServer::particles_set_height_from_velocity);
	ObjectTypeDB::bind_method(_MD("particles_has_height_from_velocity"), &VisualServer::particles_has_height_from_velocity);

	ObjectTypeDB::bind_method(_MD("light_create"), &VisualServer::light_create);
	ObjectTypeDB::bind_method(_MD("light_get_type"), &VisualServer::light_get_type);
	ObjectTypeDB::bind_method(_MD("light_set_color"), &VisualServer::light_set_color);
	ObjectTypeDB::bind_method(_MD("light_get_color"), &VisualServer::light_get_color);
	ObjectTypeDB::bind_method(_MD("light_set_shadow"), &VisualServer::light_set_shadow);
	ObjectTypeDB::bind_method(_MD("light_has_shadow"), &VisualServer::light_has_shadow);
	ObjectTypeDB::bind_method(_MD("light_set_volumetric"), &VisualServer::light_set_volumetric);
	ObjectTypeDB::bind_method(_MD("light_is_volumetric"), &VisualServer::light_is_volumetric);
	ObjectTypeDB::bind_method(_MD("light_set_projector"), &VisualServer::light_set_projector);
	ObjectTypeDB::bind_method(_MD("light_get_projector"), &VisualServer::light_get_projector);
	ObjectTypeDB::bind_method(_MD("light_set_var"), &VisualServer::light_set_param);
	ObjectTypeDB::bind_method(_MD("light_get_var"), &VisualServer::light_get_param);

	ObjectTypeDB::bind_method(_MD("skeleton_create"), &VisualServer::skeleton_create);
	ObjectTypeDB::bind_method(_MD("skeleton_resize"), &VisualServer::skeleton_resize);
	ObjectTypeDB::bind_method(_MD("skeleton_get_bone_count"), &VisualServer::skeleton_get_bone_count);
	ObjectTypeDB::bind_method(_MD("skeleton_bone_set_transform"), &VisualServer::skeleton_bone_set_transform);
	ObjectTypeDB::bind_method(_MD("skeleton_bone_get_transform"), &VisualServer::skeleton_bone_get_transform);

	ObjectTypeDB::bind_method(_MD("room_create"), &VisualServer::room_create);
	ObjectTypeDB::bind_method(_MD("room_set_bounds"), &VisualServer::room_set_bounds);
	ObjectTypeDB::bind_method(_MD("room_get_bounds"), &VisualServer::room_get_bounds);

	ObjectTypeDB::bind_method(_MD("portal_create"), &VisualServer::portal_create);
	ObjectTypeDB::bind_method(_MD("portal_set_shape"), &VisualServer::portal_set_shape);
	ObjectTypeDB::bind_method(_MD("portal_get_shape"), &VisualServer::portal_get_shape);
	ObjectTypeDB::bind_method(_MD("portal_set_enabled"), &VisualServer::portal_set_enabled);
	ObjectTypeDB::bind_method(_MD("portal_is_enabled"), &VisualServer::portal_is_enabled);
	ObjectTypeDB::bind_method(_MD("portal_set_disable_distance"), &VisualServer::portal_set_disable_distance);
	ObjectTypeDB::bind_method(_MD("portal_get_disable_distance"), &VisualServer::portal_get_disable_distance);
	ObjectTypeDB::bind_method(_MD("portal_set_disabled_color"), &VisualServer::portal_set_disabled_color);
	ObjectTypeDB::bind_method(_MD("portal_get_disabled_color"), &VisualServer::portal_get_disabled_color);

	ObjectTypeDB::bind_method(_MD("camera_create"), &VisualServer::camera_create);
	ObjectTypeDB::bind_method(_MD("camera_set_perspective"), &VisualServer::camera_set_perspective);
	ObjectTypeDB::bind_method(_MD("camera_set_orthogonal"), &VisualServer::_camera_set_orthogonal);
	ObjectTypeDB::bind_method(_MD("camera_set_transform"), &VisualServer::camera_set_transform);

	ObjectTypeDB::bind_method(_MD("viewport_create"), &VisualServer::viewport_create);
	ObjectTypeDB::bind_method(_MD("viewport_set_rect"), &VisualServer::_viewport_set_rect);
	ObjectTypeDB::bind_method(_MD("viewport_get_rect"), &VisualServer::_viewport_get_rect);
	ObjectTypeDB::bind_method(_MD("viewport_attach_camera"), &VisualServer::viewport_attach_camera, DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("viewport_get_attached_camera"), &VisualServer::viewport_get_attached_camera);
	ObjectTypeDB::bind_method(_MD("viewport_get_scenario"), &VisualServer::viewport_get_scenario);
	ObjectTypeDB::bind_method(_MD("viewport_attach_canvas"), &VisualServer::viewport_attach_canvas);
	ObjectTypeDB::bind_method(_MD("viewport_remove_canvas"), &VisualServer::viewport_remove_canvas);
	ObjectTypeDB::bind_method(_MD("viewport_set_global_canvas_transform"), &VisualServer::viewport_set_global_canvas_transform);

	ObjectTypeDB::bind_method(_MD("scenario_create"), &VisualServer::scenario_create);
	ObjectTypeDB::bind_method(_MD("scenario_set_debug"), &VisualServer::scenario_set_debug);

	ObjectTypeDB::bind_method(_MD("instance_create"), &VisualServer::instance_create, DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("instance_get_base"), &VisualServer::instance_get_base);
	ObjectTypeDB::bind_method(_MD("instance_get_base_aabb"), &VisualServer::instance_get_base);
	ObjectTypeDB::bind_method(_MD("instance_set_transform"), &VisualServer::instance_set_transform);
	ObjectTypeDB::bind_method(_MD("instance_get_transform"), &VisualServer::instance_get_transform);
	ObjectTypeDB::bind_method(_MD("instance_attach_object_instance_ID"), &VisualServer::instance_attach_object_instance_ID);
	ObjectTypeDB::bind_method(_MD("instance_get_object_instance_ID"), &VisualServer::instance_get_object_instance_ID);
	ObjectTypeDB::bind_method(_MD("instance_attach_skeleton"), &VisualServer::instance_attach_skeleton);
	ObjectTypeDB::bind_method(_MD("instance_get_skeleton"), &VisualServer::instance_get_skeleton);
	ObjectTypeDB::bind_method(_MD("instance_set_room"), &VisualServer::instance_set_room);
	ObjectTypeDB::bind_method(_MD("instance_get_room"), &VisualServer::instance_get_room);

	ObjectTypeDB::bind_method(_MD("instance_set_exterior"), &VisualServer::instance_set_exterior);
	ObjectTypeDB::bind_method(_MD("instance_is_exterior"), &VisualServer::instance_is_exterior);

	ObjectTypeDB::bind_method(_MD("instances_cull_aabb"), &VisualServer::instances_cull_aabb);
	ObjectTypeDB::bind_method(_MD("instances_cull_ray"), &VisualServer::instances_cull_ray);
	ObjectTypeDB::bind_method(_MD("instances_cull_convex"), &VisualServer::instances_cull_convex);

	ObjectTypeDB::bind_method(_MD("instance_geometry_override_material_param"), &VisualServer::instance_get_room);
	ObjectTypeDB::bind_method(_MD("instance_geometry_get_material_param"), &VisualServer::instance_get_room);

	ObjectTypeDB::bind_method(_MD("get_test_cube"), &VisualServer::get_test_cube);

#endif
	ObjectTypeDB::bind_method(_MD("canvas_create"), &VisualServer::canvas_create);
	ObjectTypeDB::bind_method(_MD("canvas_item_create"), &VisualServer::canvas_item_create);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_parent"), &VisualServer::canvas_item_set_parent);
	ObjectTypeDB::bind_method(_MD("canvas_item_get_parent"), &VisualServer::canvas_item_get_parent);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_transform"), &VisualServer::canvas_item_set_transform);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_custom_rect"), &VisualServer::canvas_item_set_custom_rect);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_clip"), &VisualServer::canvas_item_set_clip);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_opacity"), &VisualServer::canvas_item_set_opacity);
	ObjectTypeDB::bind_method(_MD("canvas_item_get_opacity"), &VisualServer::canvas_item_get_opacity);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_self_opacity"), &VisualServer::canvas_item_set_self_opacity);
	ObjectTypeDB::bind_method(_MD("canvas_item_get_self_opacity"), &VisualServer::canvas_item_get_self_opacity);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_z"), &VisualServer::canvas_item_set_z);
	ObjectTypeDB::bind_method(_MD("canvas_item_set_sort_children_by_y"), &VisualServer::canvas_item_set_sort_children_by_y);

	ObjectTypeDB::bind_method(_MD("canvas_item_add_line"), &VisualServer::canvas_item_add_line, DEFVAL(1.0));
	ObjectTypeDB::bind_method(_MD("canvas_item_add_rect"), &VisualServer::canvas_item_add_rect);
	ObjectTypeDB::bind_method(_MD("canvas_item_add_texture_rect"), &VisualServer::canvas_item_add_texture_rect, DEFVAL(Color(1, 1, 1)), DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("canvas_item_add_texture_rect_region"), &VisualServer::canvas_item_add_texture_rect_region, DEFVAL(Color(1, 1, 1)), DEFVAL(false));

	ObjectTypeDB::bind_method(_MD("canvas_item_add_style_box"), &VisualServer::_canvas_item_add_style_box, DEFVAL(Color(1, 1, 1)));
	//	ObjectTypeDB::bind_method(_MD("canvas_item_add_primitive"),&VisualServer::canvas_item_add_primitive,DEFVAL(Vector<Vector2>()),DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("canvas_item_add_circle"), &VisualServer::canvas_item_add_circle);

	ObjectTypeDB::bind_method(_MD("viewport_set_canvas_transform"), &VisualServer::viewport_set_canvas_transform);

	ObjectTypeDB::bind_method(_MD("canvas_item_clear"), &VisualServer::canvas_item_clear);
	ObjectTypeDB::bind_method(_MD("canvas_item_raise"), &VisualServer::canvas_item_raise);

	ObjectTypeDB::bind_method(_MD("cursor_set_rotation"), &VisualServer::cursor_set_rotation);
	ObjectTypeDB::bind_method(_MD("cursor_set_texture"), &VisualServer::cursor_set_texture);
	ObjectTypeDB::bind_method(_MD("cursor_set_visible"), &VisualServer::cursor_set_visible);
	ObjectTypeDB::bind_method(_MD("cursor_set_pos"), &VisualServer::cursor_set_pos);

	ObjectTypeDB::bind_method(_MD("black_bars_set_margins", "left", "top", "right", "bottom"), &VisualServer::black_bars_set_margins);
	ObjectTypeDB::bind_method(_MD("black_bars_set_images", "left", "top", "right", "bottom"), &VisualServer::black_bars_set_images);

	ObjectTypeDB::bind_method(_MD("make_sphere_mesh"), &VisualServer::make_sphere_mesh);
	ObjectTypeDB::bind_method(_MD("mesh_add_surface_from_planes"), &VisualServer::mesh_add_surface_from_planes);

	ObjectTypeDB::bind_method(_MD("draw"), &VisualServer::draw);
	ObjectTypeDB::bind_method(_MD("sync"), &VisualServer::sync);
	ObjectTypeDB::bind_method(_MD("free_rid"), &VisualServer::free);

	ObjectTypeDB::bind_method(_MD("set_default_clear_color"), &VisualServer::set_default_clear_color);
	ObjectTypeDB::bind_method(_MD("get_default_clear_color"), &VisualServer::get_default_clear_color);

	ObjectTypeDB::bind_method(_MD("get_render_info"), &VisualServer::get_render_info);

	BIND_CONSTANT(NO_INDEX_ARRAY);
	BIND_CONSTANT(CUSTOM_ARRAY_SIZE);
	BIND_CONSTANT(ARRAY_WEIGHTS_SIZE);
	BIND_CONSTANT(MAX_PARTICLE_COLOR_PHASES);
	BIND_CONSTANT(MAX_PARTICLE_ATTRACTORS);
	BIND_CONSTANT(MAX_CURSORS);

	BIND_CONSTANT(TEXTURE_FLAG_MIPMAPS);
	BIND_CONSTANT(TEXTURE_FLAG_REPEAT);
	BIND_CONSTANT(TEXTURE_FLAG_FILTER);
	BIND_CONSTANT(TEXTURE_FLAG_CUBEMAP);
	BIND_CONSTANT(TEXTURE_FLAGS_DEFAULT);

	BIND_CONSTANT(CUBEMAP_LEFT);
	BIND_CONSTANT(CUBEMAP_RIGHT);
	BIND_CONSTANT(CUBEMAP_BOTTOM);
	BIND_CONSTANT(CUBEMAP_TOP);
	BIND_CONSTANT(CUBEMAP_FRONT);
	BIND_CONSTANT(CUBEMAP_BACK);

	BIND_CONSTANT(SHADER_MATERIAL); ///< param 0: name
	BIND_CONSTANT(SHADER_POST_PROCESS); ///< param 0: name

	BIND_CONSTANT(MATERIAL_FLAG_VISIBLE);
	BIND_CONSTANT(MATERIAL_FLAG_DOUBLE_SIDED);
	BIND_CONSTANT(MATERIAL_FLAG_INVERT_FACES);
	BIND_CONSTANT(MATERIAL_FLAG_UNSHADED);
	BIND_CONSTANT(MATERIAL_FLAG_ONTOP);
	BIND_CONSTANT(MATERIAL_FLAG_MAX);

	BIND_CONSTANT(MATERIAL_BLEND_MODE_MIX);
	BIND_CONSTANT(MATERIAL_BLEND_MODE_ADD);
	BIND_CONSTANT(MATERIAL_BLEND_MODE_SUB);
	BIND_CONSTANT(MATERIAL_BLEND_MODE_MUL);

	BIND_CONSTANT(FIXED_MATERIAL_PARAM_DIFFUSE);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_DETAIL);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_SPECULAR);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_EMISSION);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_SPECULAR_EXP);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_GLOW);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_NORMAL);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_SHADE_PARAM);
	BIND_CONSTANT(FIXED_MATERIAL_PARAM_MAX);

	BIND_CONSTANT(FIXED_MATERIAL_TEXCOORD_SPHERE);
	BIND_CONSTANT(FIXED_MATERIAL_TEXCOORD_UV);
	BIND_CONSTANT(FIXED_MATERIAL_TEXCOORD_UV_TRANSFORM);
	BIND_CONSTANT(FIXED_MATERIAL_TEXCOORD_UV2);

	BIND_CONSTANT(ARRAY_VERTEX);
	BIND_CONSTANT(ARRAY_NORMAL);
	BIND_CONSTANT(ARRAY_TANGENT);
	BIND_CONSTANT(ARRAY_COLOR);
	BIND_CONSTANT(ARRAY_TEX_UV);
	BIND_CONSTANT(ARRAY_BONES);
	BIND_CONSTANT(ARRAY_WEIGHTS);
	BIND_CONSTANT(ARRAY_INDEX);
	BIND_CONSTANT(ARRAY_MAX);

	BIND_CONSTANT(ARRAY_FORMAT_VERTEX);
	BIND_CONSTANT(ARRAY_FORMAT_NORMAL);
	BIND_CONSTANT(ARRAY_FORMAT_TANGENT);
	BIND_CONSTANT(ARRAY_FORMAT_COLOR);
	BIND_CONSTANT(ARRAY_FORMAT_TEX_UV);
	BIND_CONSTANT(ARRAY_FORMAT_BONES);
	BIND_CONSTANT(ARRAY_FORMAT_WEIGHTS);
	BIND_CONSTANT(ARRAY_FORMAT_INDEX);

	BIND_CONSTANT(PRIMITIVE_POINTS);
	BIND_CONSTANT(PRIMITIVE_LINES);
	BIND_CONSTANT(PRIMITIVE_LINE_STRIP);
	BIND_CONSTANT(PRIMITIVE_LINE_LOOP);
	BIND_CONSTANT(PRIMITIVE_TRIANGLES);
	BIND_CONSTANT(PRIMITIVE_TRIANGLE_STRIP);
	BIND_CONSTANT(PRIMITIVE_TRIANGLE_FAN);
	BIND_CONSTANT(PRIMITIVE_MAX);

	BIND_CONSTANT(PARTICLE_LIFETIME);
	BIND_CONSTANT(PARTICLE_SPREAD);
	BIND_CONSTANT(PARTICLE_GRAVITY);
	BIND_CONSTANT(PARTICLE_LINEAR_VELOCITY);
	BIND_CONSTANT(PARTICLE_ANGULAR_VELOCITY);
	BIND_CONSTANT(PARTICLE_LINEAR_ACCELERATION);
	BIND_CONSTANT(PARTICLE_RADIAL_ACCELERATION);
	BIND_CONSTANT(PARTICLE_TANGENTIAL_ACCELERATION);
	BIND_CONSTANT(PARTICLE_INITIAL_SIZE);
	BIND_CONSTANT(PARTICLE_FINAL_SIZE);
	BIND_CONSTANT(PARTICLE_INITIAL_ANGLE);
	BIND_CONSTANT(PARTICLE_HEIGHT);
	BIND_CONSTANT(PARTICLE_HEIGHT_SPEED_SCALE);
	BIND_CONSTANT(PARTICLE_VAR_MAX);

	BIND_CONSTANT(LIGHT_DIRECTIONAL);
	BIND_CONSTANT(LIGHT_OMNI);
	BIND_CONSTANT(LIGHT_SPOT);

	BIND_CONSTANT(LIGHT_COLOR_DIFFUSE);
	BIND_CONSTANT(LIGHT_COLOR_SPECULAR);

	BIND_CONSTANT(LIGHT_PARAM_SPOT_ATTENUATION);
	BIND_CONSTANT(LIGHT_PARAM_SPOT_ANGLE);
	BIND_CONSTANT(LIGHT_PARAM_RADIUS);
	BIND_CONSTANT(LIGHT_PARAM_ENERGY);
	BIND_CONSTANT(LIGHT_PARAM_ATTENUATION);
	BIND_CONSTANT(LIGHT_PARAM_MAX);

	BIND_CONSTANT(SCENARIO_DEBUG_DISABLED);
	BIND_CONSTANT(SCENARIO_DEBUG_WIREFRAME);
	BIND_CONSTANT(SCENARIO_DEBUG_OVERDRAW);

	BIND_CONSTANT(INSTANCE_MESH);
	BIND_CONSTANT(INSTANCE_MULTIMESH);

	BIND_CONSTANT(INSTANCE_PARTICLES);
	BIND_CONSTANT(INSTANCE_LIGHT);
	BIND_CONSTANT(INSTANCE_ROOM);
	BIND_CONSTANT(INSTANCE_PORTAL);
	BIND_CONSTANT(INSTANCE_GEOMETRY_MASK);

	BIND_CONSTANT(INFO_OBJECTS_IN_FRAME);
	BIND_CONSTANT(INFO_VERTICES_IN_FRAME);
	BIND_CONSTANT(INFO_MATERIAL_CHANGES_IN_FRAME);
	BIND_CONSTANT(INFO_SHADER_CHANGES_IN_FRAME);
	BIND_CONSTANT(INFO_SURFACE_CHANGES_IN_FRAME);
	BIND_CONSTANT(INFO_DRAW_CALLS_IN_FRAME);
	BIND_CONSTANT(INFO_USAGE_VIDEO_MEM_TOTAL);
	BIND_CONSTANT(INFO_VIDEO_MEM_USED);
	BIND_CONSTANT(INFO_TEXTURE_MEM_USED);
	BIND_CONSTANT(INFO_VERTEX_MEM_USED);
}

void VisualServer::_canvas_item_add_style_box(RID p_item, const Rect2 &p_rect, const Rect2 &p_source, RID p_texture, const Vector<float> &p_margins, const Color &p_modulate) {

	ERR_FAIL_COND(p_margins.size() != 4);
	canvas_item_add_style_box(p_item, p_rect, p_source, p_texture, Vector2(p_margins[0], p_margins[1]), Vector2(p_margins[2], p_margins[3]), true, p_modulate);
}

void VisualServer::_camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far) {

	camera_set_orthogonal(p_camera, p_size, p_z_near, p_z_far);
}

void VisualServer::_viewport_set_rect(RID p_viewport, const Rect2 &p_rect) {

	ViewportRect r;
	r.x = p_rect.pos.x;
	r.y = p_rect.pos.y;
	r.width = p_rect.size.x;
	r.height = p_rect.size.y;
	viewport_set_rect(p_viewport, r);
}
Rect2 VisualServer::_viewport_get_rect(RID p_viewport) const {

	ViewportRect r = viewport_get_rect(p_viewport);
	return Rect2(r.x, r.y, r.width, r.height);
}

void VisualServer::mesh_add_surface_from_mesh_data(RID p_mesh, const Geometry::MeshData &p_mesh_data) {

#if 1
	DVector<Vector3> vertices;
	DVector<Vector3> normals;

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
	mesh_add_surface(p_mesh, PRIMITIVE_TRIANGLES, d);

#else

	DVector<Vector3> vertices;

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

void VisualServer::mesh_add_surface_from_planes(RID p_mesh, const DVector<Plane> &p_planes) {

	Geometry::MeshData mdata = Geometry::build_convex_mesh(p_planes);
	mesh_add_surface_from_mesh_data(p_mesh, mdata);
}

RID VisualServer::instance_create2(RID p_base, RID p_scenario) {

	RID instance = instance_create();
	instance_set_base(instance, p_base);
	instance_set_scenario(instance, p_scenario);
	return instance;
}

VisualServer::VisualServer() {

	//	ERR_FAIL_COND(singleton);
	singleton = this;
	mm_policy = GLOBAL_DEF("render/mipmap_policy", 0);
	if (mm_policy < 0 || mm_policy > 2)
		mm_policy = 0;
}

VisualServer::~VisualServer() {

	singleton = NULL;
}
