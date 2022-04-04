/*************************************************************************/
/*  rasterizer_scene_gles3.cpp                                           */
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

#include "rasterizer_scene_gles3.h"
#include "core/config/project_settings.h"
#include "core/templates/sort_array.h"
#include "servers/rendering/rendering_server_default.h"
#include "storage/config.h"

#ifdef GLES3_ENABLED

uint64_t RasterizerSceneGLES3::auto_exposure_counter = 2;

RasterizerSceneGLES3 *RasterizerSceneGLES3::singleton = nullptr;

RasterizerSceneGLES3 *RasterizerSceneGLES3::get_singleton() {
	return singleton;
}

RendererSceneRender::GeometryInstance *RasterizerSceneGLES3::geometry_instance_create(RID p_base) {
	RS::InstanceType type = storage->get_base_type(p_base);
	ERR_FAIL_COND_V(!((1 << type) & RS::INSTANCE_GEOMETRY_MASK), nullptr);

	GeometryInstanceGLES3 *ginstance = geometry_instance_alloc.alloc();
	ginstance->data = memnew(GeometryInstanceGLES3::Data);

	ginstance->data->base = p_base;
	ginstance->data->base_type = type;

	_geometry_instance_mark_dirty(ginstance);

	return ginstance;
}

void RasterizerSceneGLES3::geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->skeleton = p_skeleton;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RasterizerSceneGLES3::geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->material_override = p_override;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RasterizerSceneGLES3::geometry_instance_set_material_overlay(GeometryInstance *p_geometry_instance, RID p_overlay) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->material_overlay = p_overlay;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RasterizerSceneGLES3::geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_materials) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->surface_materials = p_materials;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RasterizerSceneGLES3::geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ERR_FAIL_COND(!ginstance);
	ginstance->mesh_instance = p_mesh_instance;

	_geometry_instance_mark_dirty(ginstance);
}

void RasterizerSceneGLES3::geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabb) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->transform = p_transform;
	ginstance->mirror = p_transform.basis.determinant() < 0;
	ginstance->data->aabb = p_aabb;
	ginstance->transformed_aabb = p_transformed_aabb;

	Vector3 model_scale_vec = p_transform.basis.get_scale_abs();
	// handle non uniform scale here

	float max_scale = MAX(model_scale_vec.x, MAX(model_scale_vec.y, model_scale_vec.z));
	float min_scale = MIN(model_scale_vec.x, MIN(model_scale_vec.y, model_scale_vec.z));
	ginstance->non_uniform_scale = max_scale >= 0.0 && (min_scale / max_scale) < 0.9;

	ginstance->lod_model_scale = max_scale;
}

void RasterizerSceneGLES3::geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->layer_mask = p_layer_mask;
}

void RasterizerSceneGLES3::geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->lod_bias = p_lod_bias;
}

void RasterizerSceneGLES3::geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->force_alpha = CLAMP(1.0 - p_transparency, 0, 1);
}

void RasterizerSceneGLES3::geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->fade_near = p_enable_near;
	ginstance->fade_near_begin = p_near_begin;
	ginstance->fade_near_end = p_near_end;
	ginstance->fade_far = p_enable_far;
	ginstance->fade_far_begin = p_far_begin;
	ginstance->fade_far_end = p_far_end;
}

void RasterizerSceneGLES3::geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->parent_fade_alpha = p_alpha;
}

void RasterizerSceneGLES3::geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->use_baked_light = p_enable;

	_geometry_instance_mark_dirty(ginstance);
}

void RasterizerSceneGLES3::geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->use_dynamic_gi = p_enable;
	_geometry_instance_mark_dirty(ginstance);
}

void RasterizerSceneGLES3::geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
}

void RasterizerSceneGLES3::geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
}

void RasterizerSceneGLES3::geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->shader_parameters_offset = p_offset;
	_geometry_instance_mark_dirty(ginstance);
}

void RasterizerSceneGLES3::geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->cast_double_sided_shadows = p_enable;
	_geometry_instance_mark_dirty(ginstance);
}

uint32_t RasterizerSceneGLES3::geometry_instance_get_pair_mask() {
	return (1 << RS::INSTANCE_LIGHT);
}

void RasterizerSceneGLES3::geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);

	GLES3::Config *config = GLES3::Config::get_singleton();

	ginstance->omni_light_count = 0;
	ginstance->spot_light_count = 0;
	ginstance->omni_lights.clear();
	ginstance->spot_lights.clear();

	for (uint32_t i = 0; i < p_light_instance_count; i++) {
		RS::LightType type = light_instance_get_type(p_light_instances[i]);
		switch (type) {
			case RS::LIGHT_OMNI: {
				if (ginstance->omni_light_count < (uint32_t)config->max_lights_per_object) {
					ginstance->omni_lights.push_back(p_light_instances[i]);
					ginstance->omni_light_count++;
				}
			} break;
			case RS::LIGHT_SPOT: {
				if (ginstance->spot_light_count < (uint32_t)config->max_lights_per_object) {
					ginstance->spot_lights.push_back(p_light_instances[i]);
					ginstance->spot_light_count++;
				}
			} break;
			default:
				break;
		}
	}
}

void RasterizerSceneGLES3::geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
}

void RasterizerSceneGLES3::geometry_instance_free(GeometryInstance *p_geometry_instance) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	GeometryInstanceSurface *surf = ginstance->surface_caches;
	while (surf) {
		GeometryInstanceSurface *next = surf->next;
		geometry_instance_surface_alloc.free(surf);
		surf = next;
	}
	memdelete(ginstance->data);
	geometry_instance_alloc.free(ginstance);
}

void RasterizerSceneGLES3::_geometry_instance_mark_dirty(GeometryInstance *p_geometry_instance) {
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);
	if (ginstance->dirty_list_element.in_list()) {
		return;
	}

	//clear surface caches
	GeometryInstanceSurface *surf = ginstance->surface_caches;

	while (surf) {
		GeometryInstanceSurface *next = surf->next;
		geometry_instance_surface_alloc.free(surf);
		surf = next;
	}

	ginstance->surface_caches = nullptr;

	geometry_instance_dirty_list.add(&ginstance->dirty_list_element);
}

void RasterizerSceneGLES3::_update_dirty_geometry_instances() {
	while (geometry_instance_dirty_list.first()) {
		_geometry_instance_update(geometry_instance_dirty_list.first()->self());
	}
}

void RasterizerSceneGLES3::_geometry_instance_dependency_changed(RendererStorage::DependencyChangedNotification p_notification, RendererStorage::DependencyTracker *p_tracker) {
	switch (p_notification) {
		case RendererStorage::DEPENDENCY_CHANGED_MATERIAL:
		case RendererStorage::DEPENDENCY_CHANGED_MESH:
		case RendererStorage::DEPENDENCY_CHANGED_PARTICLES:
		case RendererStorage::DEPENDENCY_CHANGED_MULTIMESH:
		case RendererStorage::DEPENDENCY_CHANGED_SKELETON_DATA: {
			static_cast<RasterizerSceneGLES3 *>(singleton)->_geometry_instance_mark_dirty(static_cast<GeometryInstance *>(p_tracker->userdata));
		} break;
		case RendererStorage::DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES: {
			GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_tracker->userdata);
			if (ginstance->data->base_type == RS::INSTANCE_MULTIMESH) {
				ginstance->instance_count = GLES3::MeshStorage::get_singleton()->multimesh_get_instances_to_draw(ginstance->data->base);
			}
		} break;
		default: {
			//rest of notifications of no interest
		} break;
	}
}

void RasterizerSceneGLES3::_geometry_instance_dependency_deleted(const RID &p_dependency, RendererStorage::DependencyTracker *p_tracker) {
	static_cast<RasterizerSceneGLES3 *>(singleton)->_geometry_instance_mark_dirty(static_cast<GeometryInstance *>(p_tracker->userdata));
}

void RasterizerSceneGLES3::_geometry_instance_add_surface_with_material(GeometryInstanceGLES3 *ginstance, uint32_t p_surface, GLES3::SceneMaterialData *p_material, uint32_t p_material_id, uint32_t p_shader_id, RID p_mesh) {
	GLES3::MeshStorage *mesh_storage = GLES3::MeshStorage::get_singleton();

	bool has_read_screen_alpha = p_material->shader_data->uses_screen_texture || p_material->shader_data->uses_depth_texture || p_material->shader_data->uses_normal_texture;
	bool has_base_alpha = ((p_material->shader_data->uses_alpha && !p_material->shader_data->uses_alpha_clip) || has_read_screen_alpha);
	bool has_blend_alpha = p_material->shader_data->uses_blend_alpha;
	bool has_alpha = has_base_alpha || has_blend_alpha;

	uint32_t flags = 0;

	if (p_material->shader_data->uses_screen_texture) {
		flags |= GeometryInstanceSurface::FLAG_USES_SCREEN_TEXTURE;
	}

	if (p_material->shader_data->uses_depth_texture) {
		flags |= GeometryInstanceSurface::FLAG_USES_DEPTH_TEXTURE;
	}

	if (p_material->shader_data->uses_normal_texture) {
		flags |= GeometryInstanceSurface::FLAG_USES_NORMAL_TEXTURE;
	}

	if (ginstance->data->cast_double_sided_shadows) {
		flags |= GeometryInstanceSurface::FLAG_USES_DOUBLE_SIDED_SHADOWS;
	}

	if (has_alpha || has_read_screen_alpha || p_material->shader_data->depth_draw == GLES3::SceneShaderData::DEPTH_DRAW_DISABLED || p_material->shader_data->depth_test == GLES3::SceneShaderData::DEPTH_TEST_DISABLED) {
		//material is only meant for alpha pass
		flags |= GeometryInstanceSurface::FLAG_PASS_ALPHA;
		if (p_material->shader_data->uses_depth_pre_pass && !(p_material->shader_data->depth_draw == GLES3::SceneShaderData::DEPTH_DRAW_DISABLED || p_material->shader_data->depth_test == GLES3::SceneShaderData::DEPTH_TEST_DISABLED)) {
			flags |= GeometryInstanceSurface::FLAG_PASS_DEPTH;
			flags |= GeometryInstanceSurface::FLAG_PASS_SHADOW;
		}
	} else {
		flags |= GeometryInstanceSurface::FLAG_PASS_OPAQUE;
		flags |= GeometryInstanceSurface::FLAG_PASS_DEPTH;
		flags |= GeometryInstanceSurface::FLAG_PASS_SHADOW;
	}

	GLES3::SceneMaterialData *material_shadow = nullptr;
	void *surface_shadow = nullptr;
	if (!p_material->shader_data->uses_particle_trails && !p_material->shader_data->writes_modelview_or_projection && !p_material->shader_data->uses_vertex && !p_material->shader_data->uses_discard && !p_material->shader_data->uses_depth_pre_pass && !p_material->shader_data->uses_alpha_clip) {
		flags |= GeometryInstanceSurface::FLAG_USES_SHARED_SHADOW_MATERIAL;
		material_shadow = static_cast<GLES3::SceneMaterialData *>(GLES3::MaterialStorage::get_singleton()->material_get_data(scene_globals.default_material, RS::SHADER_SPATIAL));

		RID shadow_mesh = mesh_storage->mesh_get_shadow_mesh(p_mesh);

		if (shadow_mesh.is_valid()) {
			surface_shadow = mesh_storage->mesh_get_surface(shadow_mesh, p_surface);
		}

	} else {
		material_shadow = p_material;
	}

	GeometryInstanceSurface *sdcache = geometry_instance_surface_alloc.alloc();

	sdcache->flags = flags;

	sdcache->shader = p_material->shader_data;
	sdcache->material = p_material;
	sdcache->surface = mesh_storage->mesh_get_surface(p_mesh, p_surface);
	sdcache->primitive = mesh_storage->mesh_surface_get_primitive(sdcache->surface);
	sdcache->surface_index = p_surface;

	if (ginstance->data->dirty_dependencies) {
		storage->base_update_dependency(p_mesh, &ginstance->data->dependency_tracker);
	}

	//shadow
	sdcache->shader_shadow = material_shadow->shader_data;
	sdcache->material_shadow = material_shadow;

	sdcache->surface_shadow = surface_shadow ? surface_shadow : sdcache->surface;

	sdcache->owner = ginstance;

	sdcache->next = ginstance->surface_caches;
	ginstance->surface_caches = sdcache;

	//sortkey

	sdcache->sort.sort_key1 = 0;
	sdcache->sort.sort_key2 = 0;

	sdcache->sort.surface_index = p_surface;
	sdcache->sort.material_id_low = p_material_id & 0x0000FFFF;
	sdcache->sort.material_id_hi = p_material_id >> 16;
	sdcache->sort.shader_id = p_shader_id;
	sdcache->sort.geometry_id = p_mesh.get_local_index();
	sdcache->sort.priority = p_material->priority;
}

void RasterizerSceneGLES3::_geometry_instance_add_surface_with_material_chain(GeometryInstanceGLES3 *ginstance, uint32_t p_surface, GLES3::SceneMaterialData *p_material_data, RID p_mat_src, RID p_mesh) {
	GLES3::SceneMaterialData *material_data = p_material_data;
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();

	_geometry_instance_add_surface_with_material(ginstance, p_surface, material_data, p_mat_src.get_local_index(), material_storage->material_get_shader_id(p_mat_src), p_mesh);

	while (material_data->next_pass.is_valid()) {
		RID next_pass = material_data->next_pass;
		material_data = static_cast<GLES3::SceneMaterialData *>(material_storage->material_get_data(next_pass, RS::SHADER_SPATIAL));
		if (!material_data || !material_data->shader_data->valid) {
			break;
		}
		if (ginstance->data->dirty_dependencies) {
			material_storage->material_update_dependency(next_pass, &ginstance->data->dependency_tracker);
		}
		_geometry_instance_add_surface_with_material(ginstance, p_surface, material_data, next_pass.get_local_index(), material_storage->material_get_shader_id(next_pass), p_mesh);
	}
}

void RasterizerSceneGLES3::_geometry_instance_add_surface(GeometryInstanceGLES3 *ginstance, uint32_t p_surface, RID p_material, RID p_mesh) {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	RID m_src;

	m_src = ginstance->data->material_override.is_valid() ? ginstance->data->material_override : p_material;

	GLES3::SceneMaterialData *material_data = nullptr;

	if (m_src.is_valid()) {
		material_data = static_cast<GLES3::SceneMaterialData *>(material_storage->material_get_data(m_src, RS::SHADER_SPATIAL));
		if (!material_data || !material_data->shader_data->valid) {
			material_data = nullptr;
		}
	}

	if (material_data) {
		if (ginstance->data->dirty_dependencies) {
			material_storage->material_update_dependency(m_src, &ginstance->data->dependency_tracker);
		}
	} else {
		material_data = static_cast<GLES3::SceneMaterialData *>(material_storage->material_get_data(scene_globals.default_material, RS::SHADER_SPATIAL));
		m_src = scene_globals.default_material;
	}

	ERR_FAIL_COND(!material_data);

	_geometry_instance_add_surface_with_material_chain(ginstance, p_surface, material_data, m_src, p_mesh);

	if (ginstance->data->material_overlay.is_valid()) {
		m_src = ginstance->data->material_overlay;

		material_data = static_cast<GLES3::SceneMaterialData *>(material_storage->material_get_data(m_src, RS::SHADER_SPATIAL));
		if (material_data && material_data->shader_data->valid) {
			if (ginstance->data->dirty_dependencies) {
				material_storage->material_update_dependency(m_src, &ginstance->data->dependency_tracker);
			}

			_geometry_instance_add_surface_with_material_chain(ginstance, p_surface, material_data, m_src, p_mesh);
		}
	}
}

void RasterizerSceneGLES3::_geometry_instance_update(GeometryInstance *p_geometry_instance) {
	GLES3::MeshStorage *mesh_storage = GLES3::MeshStorage::get_singleton();
	GeometryInstanceGLES3 *ginstance = static_cast<GeometryInstanceGLES3 *>(p_geometry_instance);

	if (ginstance->data->dirty_dependencies) {
		ginstance->data->dependency_tracker.update_begin();
	}

	//add geometry for drawing
	switch (ginstance->data->base_type) {
		case RS::INSTANCE_MESH: {
			const RID *materials = nullptr;
			uint32_t surface_count;
			RID mesh = ginstance->data->base;

			materials = mesh_storage->mesh_get_surface_count_and_materials(mesh, surface_count);
			if (materials) {
				//if no materials, no surfaces.
				const RID *inst_materials = ginstance->data->surface_materials.ptr();
				uint32_t surf_mat_count = ginstance->data->surface_materials.size();

				for (uint32_t j = 0; j < surface_count; j++) {
					RID material = (j < surf_mat_count && inst_materials[j].is_valid()) ? inst_materials[j] : materials[j];
					_geometry_instance_add_surface(ginstance, j, material, mesh);
				}
			}

			ginstance->instance_count = 1;

		} break;

		case RS::INSTANCE_MULTIMESH: {
			RID mesh = mesh_storage->multimesh_get_mesh(ginstance->data->base);
			if (mesh.is_valid()) {
				const RID *materials = nullptr;
				uint32_t surface_count;

				materials = mesh_storage->mesh_get_surface_count_and_materials(mesh, surface_count);
				if (materials) {
					for (uint32_t j = 0; j < surface_count; j++) {
						_geometry_instance_add_surface(ginstance, j, materials[j], mesh);
					}
				}

				ginstance->instance_count = mesh_storage->multimesh_get_instances_to_draw(ginstance->data->base);
			}

		} break;
		case RS::INSTANCE_PARTICLES: {
		} break;

		default: {
		}
	}

	bool store_transform = true;
	ginstance->base_flags = 0;

	if (ginstance->data->base_type == RS::INSTANCE_MULTIMESH) {
		ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH;
		if (mesh_storage->multimesh_get_transform_format(ginstance->data->base) == RS::MULTIMESH_TRANSFORM_2D) {
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D;
		}
		if (mesh_storage->multimesh_uses_colors(ginstance->data->base)) {
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR;
		}
		if (mesh_storage->multimesh_uses_custom_data(ginstance->data->base)) {
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA;
		}

		//ginstance->transforms_uniform_set = mesh_storage->multimesh_get_3d_uniform_set(ginstance->data->base, scene_globals.default_shader_rd, TRANSFORMS_UNIFORM_SET);

	} else if (ginstance->data->base_type == RS::INSTANCE_PARTICLES) {
	} else if (ginstance->data->base_type == RS::INSTANCE_MESH) {
	}

	ginstance->store_transform_cache = store_transform;

	if (ginstance->data->dirty_dependencies) {
		ginstance->data->dependency_tracker.update_end();
		ginstance->data->dirty_dependencies = false;
	}

	ginstance->dirty_list_element.remove_from_list();
}

/* SHADOW ATLAS API */

RID RasterizerSceneGLES3::shadow_atlas_create() {
	return RID();
}

void RasterizerSceneGLES3::shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits) {
}

void RasterizerSceneGLES3::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {
}

bool RasterizerSceneGLES3::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) {
	return false;
}

void RasterizerSceneGLES3::directional_shadow_atlas_set_size(int p_size, bool p_16_bits) {
}

int RasterizerSceneGLES3::get_directional_light_shadow_size(RID p_light_intance) {
	return 0;
}

void RasterizerSceneGLES3::set_directional_shadow_count(int p_count) {
}

/* SKY API */

void RasterizerSceneGLES3::_free_sky_data(Sky *p_sky) {
	if (p_sky->radiance != 0) {
		glDeleteTextures(1, &p_sky->radiance);
		p_sky->radiance = 0;
		glDeleteFramebuffers(1, &p_sky->radiance_framebuffer);
		p_sky->radiance_framebuffer = 0;
	}
}

RID RasterizerSceneGLES3::sky_allocate() {
	return sky_owner.allocate_rid();
}

void RasterizerSceneGLES3::sky_initialize(RID p_rid) {
	sky_owner.initialize_rid(p_rid);
}

void RasterizerSceneGLES3::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
	Sky *sky = sky_owner.get_or_null(p_sky);
	ERR_FAIL_COND(!sky);
	ERR_FAIL_COND_MSG(p_radiance_size < 32 || p_radiance_size > 2048, "Sky radiance size must be between 32 and 2048");

	if (sky->radiance_size == p_radiance_size) {
		return; // No need to update
	}

	sky->radiance_size = p_radiance_size;

	_free_sky_data(sky);
	_invalidate_sky(sky);
}

void RasterizerSceneGLES3::sky_set_mode(RID p_sky, RS::SkyMode p_mode) {
	Sky *sky = sky_owner.get_or_null(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->mode == p_mode) {
		return;
	}

	sky->mode = p_mode;
	_invalidate_sky(sky);
}

void RasterizerSceneGLES3::sky_set_material(RID p_sky, RID p_material) {
	Sky *sky = sky_owner.get_or_null(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->material == p_material) {
		return;
	}

	sky->material = p_material;
	_invalidate_sky(sky);
}

void RasterizerSceneGLES3::_invalidate_sky(Sky *p_sky) {
	if (!p_sky->dirty) {
		p_sky->dirty = true;
		p_sky->dirty_list = dirty_sky_list;
		dirty_sky_list = p_sky;
	}
}

void RasterizerSceneGLES3::_update_dirty_skys() {
	Sky *sky = dirty_sky_list;

	while (sky) {
		if (sky->radiance == 0) {
			sky->mipmap_count = Image::get_image_required_mipmaps(sky->radiance_size, sky->radiance_size, Image::FORMAT_RGBA8) + 1;

			// Left uninitialized, will attach a texture at render time
			glGenFramebuffers(1, &sky->radiance_framebuffer);

			GLenum internal_format = GL_RGB10_A2;

			glGenTextures(1, &sky->radiance);
			glBindTexture(GL_TEXTURE_CUBE_MAP, sky->radiance);

#ifdef GLES_OVER_GL
			GLenum format = GL_RGBA;
			GLenum type = GL_UNSIGNED_INT_2_10_10_10_REV;
			//TODO, on low-end compare this to allocating each face of each mip individually
			// see: https://www.khronos.org/registry/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml
			for (int i = 0; i < 6; i++) {
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internal_format, sky->radiance_size, sky->radiance_size, 0, format, type, nullptr);
			}

			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
#else
			glTexStorage2D(GL_TEXTURE_CUBE_MAP, sky->mipmap_count, internal_format, sky->radiance_size, sky->radiance_size);
#endif
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, sky->mipmap_count);

			glGenTextures(1, &sky->raw_radiance);
			glBindTexture(GL_TEXTURE_CUBE_MAP, sky->raw_radiance);

#ifdef GLES_OVER_GL
			//TODO, on low-end compare this to allocating each face of each mip individually
			// see: https://www.khronos.org/registry/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml
			for (int i = 0; i < 6; i++) {
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internal_format, sky->radiance_size, sky->radiance_size, 0, format, type, nullptr);
			}

			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
#else
			glTexStorage2D(GL_TEXTURE_CUBE_MAP, sky->mipmap_count, internal_format, sky->radiance_size, sky->radiance_size);
#endif
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, sky->mipmap_count);
			glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
		}

		sky->reflection_dirty = true;
		sky->processing_layer = 0;

		Sky *next = sky->dirty_list;
		sky->dirty_list = nullptr;
		sky->dirty = false;
		sky = next;
	}

	dirty_sky_list = nullptr;
}

void RasterizerSceneGLES3::_setup_sky(Environment *p_env, RID p_render_buffers, const PagedArray<RID> &p_lights, const CameraMatrix &p_projection, const Transform3D &p_transform, const Size2i p_screen_size) {
	GLES3::LightStorage *light_storage = GLES3::LightStorage::get_singleton();
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	ERR_FAIL_COND(!p_env);

	GLES3::SkyMaterialData *material = nullptr;
	Sky *sky = sky_owner.get_or_null(p_env->sky);

	RID sky_material;

	GLES3::SkyShaderData *shader_data = nullptr;

	if (sky) {
		sky_material = sky->material;

		if (sky_material.is_valid()) {
			material = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
			if (!material || !material->shader_data->valid) {
				material = nullptr;
			}
		}

		if (!material) {
			sky_material = sky_globals.default_material;
			material = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
		}

		ERR_FAIL_COND(!material);

		shader_data = material->shader_data;

		ERR_FAIL_COND(!shader_data);

		if (shader_data->uses_time && time - sky->prev_time > 0.00001) {
			sky->prev_time = time;
			sky->reflection_dirty = true;
			RenderingServerDefault::redraw_request();
		}

		if (material != sky->prev_material) {
			sky->prev_material = material;
			sky->reflection_dirty = true;
		}

		if (material->uniform_set_updated) {
			material->uniform_set_updated = false;
			sky->reflection_dirty = true;
		}

		if (!p_transform.origin.is_equal_approx(sky->prev_position) && shader_data->uses_position) {
			sky->prev_position = p_transform.origin;
			sky->reflection_dirty = true;
		}

		if (shader_data->uses_light) {
			sky_globals.directional_light_count = 0;
			for (int i = 0; i < (int)p_lights.size(); i++) {
				LightInstance *li = light_instance_owner.get_or_null(p_lights[i]);
				if (!li) {
					continue;
				}
				RID base = li->light;

				ERR_CONTINUE(base.is_null());

				RS::LightType type = light_storage->light_get_type(base);
				if (type == RS::LIGHT_DIRECTIONAL && light_storage->light_directional_get_sky_mode(base) != RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_ONLY) {
					DirectionalLightData &sky_light_data = sky_globals.directional_lights[sky_globals.directional_light_count];
					Transform3D light_transform = li->transform;
					Vector3 world_direction = light_transform.basis.xform(Vector3(0, 0, 1)).normalized();

					sky_light_data.direction[0] = world_direction.x;
					sky_light_data.direction[1] = world_direction.y;
					sky_light_data.direction[2] = world_direction.z;

					float sign = light_storage->light_is_negative(base) ? -1 : 1;
					sky_light_data.energy = sign * light_storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY);

					Color linear_col = light_storage->light_get_color(base);
					sky_light_data.color[0] = linear_col.r;
					sky_light_data.color[1] = linear_col.g;
					sky_light_data.color[2] = linear_col.b;

					sky_light_data.enabled = true;

					float angular_diameter = light_storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);
					if (angular_diameter > 0.0) {
						angular_diameter = Math::tan(Math::deg2rad(angular_diameter));
					} else {
						angular_diameter = 0.0;
					}
					sky_light_data.size = angular_diameter;
					sky_globals.directional_light_count++;
					if (sky_globals.directional_light_count >= sky_globals.max_directional_lights) {
						break;
					}
				}
			}
			// Check whether the directional_light_buffer changes
			bool light_data_dirty = false;

			// Light buffer is dirty if we have fewer or more lights
			// If we have fewer lights, make sure that old lights are disabled
			if (sky_globals.directional_light_count != sky_globals.last_frame_directional_light_count) {
				light_data_dirty = true;
				for (uint32_t i = sky_globals.directional_light_count; i < sky_globals.max_directional_lights; i++) {
					sky_globals.directional_lights[i].enabled = false;
				}
			}

			if (!light_data_dirty) {
				for (uint32_t i = 0; i < sky_globals.directional_light_count; i++) {
					if (sky_globals.directional_lights[i].direction[0] != sky_globals.last_frame_directional_lights[i].direction[0] ||
							sky_globals.directional_lights[i].direction[1] != sky_globals.last_frame_directional_lights[i].direction[1] ||
							sky_globals.directional_lights[i].direction[2] != sky_globals.last_frame_directional_lights[i].direction[2] ||
							sky_globals.directional_lights[i].energy != sky_globals.last_frame_directional_lights[i].energy ||
							sky_globals.directional_lights[i].color[0] != sky_globals.last_frame_directional_lights[i].color[0] ||
							sky_globals.directional_lights[i].color[1] != sky_globals.last_frame_directional_lights[i].color[1] ||
							sky_globals.directional_lights[i].color[2] != sky_globals.last_frame_directional_lights[i].color[2] ||
							sky_globals.directional_lights[i].enabled != sky_globals.last_frame_directional_lights[i].enabled ||
							sky_globals.directional_lights[i].size != sky_globals.last_frame_directional_lights[i].size) {
						light_data_dirty = true;
						break;
					}
				}
			}

			if (light_data_dirty) {
				glBindBufferBase(GL_UNIFORM_BUFFER, SKY_DIRECTIONAL_LIGHT_UNIFORM_LOCATION, sky_globals.directional_light_buffer);
				glBufferData(GL_UNIFORM_BUFFER, sizeof(DirectionalLightData) * sky_globals.max_directional_lights, sky_globals.directional_lights, GL_STREAM_DRAW);
				glBindBuffer(GL_UNIFORM_BUFFER, 0);

				DirectionalLightData *temp = sky_globals.last_frame_directional_lights;
				sky_globals.last_frame_directional_lights = sky_globals.directional_lights;
				sky_globals.directional_lights = temp;
				sky_globals.last_frame_directional_light_count = sky_globals.directional_light_count;
				sky->reflection_dirty = true;
			}
		}

		if (!sky->radiance) {
			_update_dirty_skys();
		}
	}
}

void RasterizerSceneGLES3::_draw_sky(Environment *p_env, const CameraMatrix &p_projection, const Transform3D &p_transform) {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	ERR_FAIL_COND(!p_env);

	Sky *sky = sky_owner.get_or_null(p_env->sky);
	ERR_FAIL_COND(!sky);

	GLES3::SkyMaterialData *material_data = nullptr;
	RID sky_material;

	RS::EnvironmentBG background = p_env->background;

	if (sky) {
		ERR_FAIL_COND(!sky);
		sky_material = sky->material;

		if (sky_material.is_valid()) {
			material_data = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
			if (!material_data || !material_data->shader_data->valid) {
				material_data = nullptr;
			}
		}

		if (!material_data) {
			sky_material = sky_globals.default_material;
			material_data = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
		}
	} else if (background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) {
		sky_material = sky_globals.fog_material;
		material_data = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
	}

	ERR_FAIL_COND(!material_data);
	material_data->bind_uniforms();

	GLES3::SkyShaderData *shader_data = material_data->shader_data;

	ERR_FAIL_COND(!shader_data);

	// Camera
	CameraMatrix camera;

	if (p_env->sky_custom_fov) {
		float near_plane = p_projection.get_z_near();
		float far_plane = p_projection.get_z_far();
		float aspect = p_projection.get_aspect();

		camera.set_perspective(p_env->sky_custom_fov, aspect, near_plane, far_plane);
	} else {
		camera = p_projection;
	}
	Basis sky_transform = p_env->sky_orientation;
	sky_transform.invert();
	sky_transform = p_transform.basis * sky_transform;

	GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_bind_shader(shader_data->version, SkyShaderGLES3::MODE_BACKGROUND);
	GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::ORIENTATION, sky_transform, shader_data->version, SkyShaderGLES3::MODE_BACKGROUND);
	GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::PROJECTION, camera.matrix[2][0], camera.matrix[0][0], camera.matrix[2][1], camera.matrix[1][1], shader_data->version, SkyShaderGLES3::MODE_BACKGROUND);
	GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::POSITION, p_transform.origin, shader_data->version, SkyShaderGLES3::MODE_BACKGROUND);
	GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::TIME, time, shader_data->version, SkyShaderGLES3::MODE_BACKGROUND);

	glBindVertexArray(sky_globals.screen_triangle_array);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void RasterizerSceneGLES3::_update_sky_radiance(Environment *p_env, const CameraMatrix &p_projection, const Transform3D &p_transform) {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	ERR_FAIL_COND(!p_env);

	Sky *sky = sky_owner.get_or_null(p_env->sky);
	ERR_FAIL_COND(!sky);

	GLES3::SkyMaterialData *material_data = nullptr;
	RID sky_material;

	RS::EnvironmentBG background = p_env->background;

	if (sky) {
		ERR_FAIL_COND(!sky);
		sky_material = sky->material;

		if (sky_material.is_valid()) {
			material_data = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
			if (!material_data || !material_data->shader_data->valid) {
				material_data = nullptr;
			}
		}

		if (!material_data) {
			sky_material = sky_globals.default_material;
			material_data = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
		}
	} else if (background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) {
		sky_material = sky_globals.fog_material;
		material_data = static_cast<GLES3::SkyMaterialData *>(material_storage->material_get_data(sky_material, RS::SHADER_SKY));
	}

	ERR_FAIL_COND(!material_data);
	material_data->bind_uniforms();

	GLES3::SkyShaderData *shader_data = material_data->shader_data;

	ERR_FAIL_COND(!shader_data);

	bool update_single_frame = sky->mode == RS::SKY_MODE_REALTIME || sky->mode == RS::SKY_MODE_QUALITY;
	RS::SkyMode sky_mode = sky->mode;

	if (sky_mode == RS::SKY_MODE_AUTOMATIC) {
		if (shader_data->uses_time || shader_data->uses_position) {
			update_single_frame = true;
			sky_mode = RS::SKY_MODE_REALTIME;
		} else if (shader_data->uses_light || shader_data->ubo_size > 0) {
			update_single_frame = false;
			sky_mode = RS::SKY_MODE_INCREMENTAL;
		} else {
			update_single_frame = true;
			sky_mode = RS::SKY_MODE_QUALITY;
		}
	}

	if (sky->processing_layer == 0 && sky_mode == RS::SKY_MODE_INCREMENTAL) {
		// On the first frame after creating sky, rebuild in single frame
		update_single_frame = true;
		sky_mode = RS::SKY_MODE_QUALITY;
	}

	int max_processing_layer = sky->mipmap_count;

	// Update radiance cubemap
	if (sky->reflection_dirty && (sky->processing_layer >= max_processing_layer || update_single_frame)) {
		static const Vector3 view_normals[6] = {
			Vector3(+1, 0, 0),
			Vector3(-1, 0, 0),
			Vector3(0, +1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, +1),
			Vector3(0, 0, -1)
		};
		static const Vector3 view_up[6] = {
			Vector3(0, -1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, +1),
			Vector3(0, 0, -1),
			Vector3(0, -1, 0),
			Vector3(0, -1, 0)
		};

		CameraMatrix cm;
		cm.set_perspective(90, 1, 0.01, 10.0);
		CameraMatrix correction;
		correction.set_depth_correction(true);
		cm = correction * cm;

		GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_bind_shader(shader_data->version, SkyShaderGLES3::MODE_CUBEMAP);

		GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::POSITION, p_transform.origin, shader_data->version, SkyShaderGLES3::MODE_CUBEMAP);
		GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::TIME, time, shader_data->version, SkyShaderGLES3::MODE_CUBEMAP);
		GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::PROJECTION, cm.matrix[2][0], cm.matrix[0][0], cm.matrix[2][1], cm.matrix[1][1], shader_data->version, SkyShaderGLES3::MODE_CUBEMAP);

		// Bind a vertex array or else OpenGL complains. We won't actually use it
		glBindVertexArray(sky_globals.screen_triangle_array);

		glViewport(0, 0, sky->radiance_size, sky->radiance_size);
		glBindFramebuffer(GL_FRAMEBUFFER, sky->radiance_framebuffer);

		for (int i = 0; i < 6; i++) {
			Basis local_view = Basis::looking_at(view_normals[i], view_up[i]);
			GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_set_uniform(SkyShaderGLES3::ORIENTATION, local_view, shader_data->version, SkyShaderGLES3::MODE_CUBEMAP);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, sky->raw_radiance, 0);
			glDrawArrays(GL_TRIANGLES, 0, 3);
		}

		if (update_single_frame) {
			for (int i = 0; i < max_processing_layer; i++) {
				_filter_sky_radiance(sky, i);
			}
		} else {
			_filter_sky_radiance(sky, 0); //Just copy over the first mipmap
		}
		sky->processing_layer = 1;

		sky->reflection_dirty = false;
	} else {
		if (sky_mode == RS::SKY_MODE_INCREMENTAL && sky->processing_layer < max_processing_layer) {
			_filter_sky_radiance(sky, sky->processing_layer);
			sky->processing_layer++;
		}
	}
}

void RasterizerSceneGLES3::_filter_sky_radiance(Sky *p_sky, int p_base_layer) {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, p_sky->raw_radiance);
	glBindFramebuffer(GL_FRAMEBUFFER, p_sky->radiance_framebuffer);

	CubemapFilterShaderGLES3::ShaderVariant mode = CubemapFilterShaderGLES3::MODE_DEFAULT;

	if (p_base_layer == 0) {
		glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
		mode = CubemapFilterShaderGLES3::MODE_COPY;

		//Copy over base layer
	}
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, sky_globals.radical_inverse_vdc_cache_tex);

	int size = p_sky->radiance_size >> p_base_layer;
	glViewport(0, 0, size, size);
	glBindVertexArray(sky_globals.screen_triangle_array);

	material_storage->shaders.cubemap_filter_shader.version_bind_shader(scene_globals.cubemap_filter_shader_version, mode);
	material_storage->shaders.cubemap_filter_shader.version_set_uniform(CubemapFilterShaderGLES3::SAMPLE_COUNT, sky_globals.ggx_samples, scene_globals.cubemap_filter_shader_version, mode);
	material_storage->shaders.cubemap_filter_shader.version_set_uniform(CubemapFilterShaderGLES3::ROUGHNESS, float(p_base_layer) / (p_sky->mipmap_count - 1.0), scene_globals.cubemap_filter_shader_version, mode);
	material_storage->shaders.cubemap_filter_shader.version_set_uniform(CubemapFilterShaderGLES3::FACE_SIZE, float(size), scene_globals.cubemap_filter_shader_version, mode);

	for (int i = 0; i < 6; i++) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, p_sky->radiance, p_base_layer);
#ifdef DEBUG_ENABLED
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		ERR_CONTINUE(status != GL_FRAMEBUFFER_COMPLETE);
#endif
		material_storage->shaders.cubemap_filter_shader.version_set_uniform(CubemapFilterShaderGLES3::FACE_ID, i, scene_globals.cubemap_filter_shader_version, mode);

		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
	glBindVertexArray(0);
	glViewport(0, 0, p_sky->screen_size.x, p_sky->screen_size.y);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Ref<Image> RasterizerSceneGLES3::sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) {
	return Ref<Image>();
}

/* ENVIRONMENT API */

RID RasterizerSceneGLES3::environment_allocate() {
	return environment_owner.allocate_rid();
}

void RasterizerSceneGLES3::environment_initialize(RID p_rid) {
	environment_owner.initialize_rid(p_rid);
}

void RasterizerSceneGLES3::environment_set_background(RID p_env, RS::EnvironmentBG p_bg) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->background = p_bg;
}

void RasterizerSceneGLES3::environment_set_sky(RID p_env, RID p_sky) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky = p_sky;
}

void RasterizerSceneGLES3::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky_custom_fov = p_scale;
}

void RasterizerSceneGLES3::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky_orientation = p_orientation;
}

void RasterizerSceneGLES3::environment_set_bg_color(RID p_env, const Color &p_color) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->bg_color = p_color;
}

void RasterizerSceneGLES3::environment_set_bg_energy(RID p_env, float p_energy) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->bg_energy = p_energy;
}

void RasterizerSceneGLES3::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->canvas_max_layer = p_max_layer;
}

void RasterizerSceneGLES3::environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->ambient_light = p_color;
	env->ambient_source = p_ambient;
	env->ambient_light_energy = p_energy;
	env->ambient_sky_contribution = p_sky_contribution;
	env->reflection_source = p_reflection_source;
}

void RasterizerSceneGLES3::environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	ERR_FAIL_COND_MSG(p_levels.size() != 7, "Size of array of glow levels must be 7");
	env->glow_enabled = p_enable;
	env->glow_levels = p_levels;
	env->glow_intensity = p_intensity;
	env->glow_strength = p_strength;
	env->glow_mix = p_mix;
	env->glow_bloom = p_bloom_threshold;
	env->glow_blend_mode = p_blend_mode;
	env->glow_hdr_bleed_threshold = p_hdr_bleed_threshold;
	env->glow_hdr_bleed_scale = p_hdr_bleed_scale;
	env->glow_hdr_luminance_cap = p_hdr_luminance_cap;
	env->glow_map_strength = p_glow_map_strength;
	env->glow_map = p_glow_map;
}

void RasterizerSceneGLES3::environment_glow_set_use_bicubic_upscale(bool p_enable) {
	glow_bicubic_upscale = p_enable;
}

void RasterizerSceneGLES3::environment_glow_set_use_high_quality(bool p_enable) {
	glow_high_quality = p_enable;
}

void RasterizerSceneGLES3::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->ssr_enabled = p_enable;
	env->ssr_max_steps = p_max_steps;
	env->ssr_fade_in = p_fade_int;
	env->ssr_fade_out = p_fade_out;
	env->ssr_depth_tolerance = p_depth_tolerance;
}

void RasterizerSceneGLES3::environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) {
}

void RasterizerSceneGLES3::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES3::environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
}

void RasterizerSceneGLES3::environment_set_ssil(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_sharpness, float p_normal_rejection) {
}
void RasterizerSceneGLES3::environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
}

void RasterizerSceneGLES3::environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
}

void RasterizerSceneGLES3::environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) {
}

void RasterizerSceneGLES3::environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) {
}

void RasterizerSceneGLES3::environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) {
}

void RasterizerSceneGLES3::environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->exposure = p_exposure;
	env->tone_mapper = p_tone_mapper;
	if (!env->auto_exposure && p_auto_exposure) {
		env->auto_exposure_version = ++auto_exposure_counter;
	}
	env->auto_exposure = p_auto_exposure;
	env->white = p_white;
	env->min_luminance = p_min_luminance;
	env->max_luminance = p_max_luminance;
	env->auto_exp_speed = p_auto_exp_speed;
	env->auto_exp_scale = p_auto_exp_scale;
}

void RasterizerSceneGLES3::environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->adjustments_enabled = p_enable;
	env->adjustments_brightness = p_brightness;
	env->adjustments_contrast = p_contrast;
	env->adjustments_saturation = p_saturation;
	env->use_1d_color_correction = p_use_1d_color_correction;
	env->color_correction = p_color_correction;
}

void RasterizerSceneGLES3::environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->fog_enabled = p_enable;
	env->fog_light_color = p_light_color;
	env->fog_light_energy = p_light_energy;
	env->fog_sun_scatter = p_sun_scatter;
	env->fog_density = p_density;
	env->fog_height = p_height;
	env->fog_height_density = p_height_density;
	env->fog_aerial_perspective = p_aerial_perspective;
}

void RasterizerSceneGLES3::environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog_volume_size(int p_size, int p_depth) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog_filter_active(bool p_enable) {
}

Ref<Image> RasterizerSceneGLES3::environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Ref<Image>());
	return Ref<Image>();
}

bool RasterizerSceneGLES3::is_environment(RID p_env) const {
	return environment_owner.owns(p_env);
}

RS::EnvironmentBG RasterizerSceneGLES3::environment_get_background(RID p_env) const {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_BG_MAX);
	return env->background;
}

int RasterizerSceneGLES3::environment_get_canvas_max_layer(RID p_env) const {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->canvas_max_layer;
}

RID RasterizerSceneGLES3::camera_effects_allocate() {
	return RID();
}

void RasterizerSceneGLES3::camera_effects_initialize(RID p_rid) {
}

void RasterizerSceneGLES3::camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) {
}

void RasterizerSceneGLES3::camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) {
}

void RasterizerSceneGLES3::camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {
}

void RasterizerSceneGLES3::camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) {
}

void RasterizerSceneGLES3::shadows_quality_set(RS::ShadowQuality p_quality) {
}

void RasterizerSceneGLES3::directional_shadow_quality_set(RS::ShadowQuality p_quality) {
}

RID RasterizerSceneGLES3::light_instance_create(RID p_light) {
	RID li = light_instance_owner.make_rid(LightInstance());

	LightInstance *light_instance = light_instance_owner.get_or_null(li);

	light_instance->self = li;
	light_instance->light = p_light;
	light_instance->light_type = RSG::light_storage->light_get_type(p_light);

	return li;
}

void RasterizerSceneGLES3::light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->transform = p_transform;
}

void RasterizerSceneGLES3::light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->aabb = p_aabb;
}

void RasterizerSceneGLES3::light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale, float p_range_begin, const Vector2 &p_uv_scale) {
}

void RasterizerSceneGLES3::light_instance_mark_visible(RID p_light_instance) {
}

RID RasterizerSceneGLES3::fog_volume_instance_create(RID p_fog_volume) {
	return RID();
}

void RasterizerSceneGLES3::fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) {
}

void RasterizerSceneGLES3::fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) {
}

RID RasterizerSceneGLES3::fog_volume_instance_get_volume(RID p_fog_volume_instance) const {
	return RID();
}

Vector3 RasterizerSceneGLES3::fog_volume_instance_get_position(RID p_fog_volume_instance) const {
	return Vector3();
}

RID RasterizerSceneGLES3::reflection_atlas_create() {
	return RID();
}

int RasterizerSceneGLES3::reflection_atlas_get_size(RID p_ref_atlas) const {
	return 0;
}

void RasterizerSceneGLES3::reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {
}

RID RasterizerSceneGLES3::reflection_probe_instance_create(RID p_probe) {
	return RID();
}

void RasterizerSceneGLES3::reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) {
}

void RasterizerSceneGLES3::reflection_probe_release_atlas_index(RID p_instance) {
}

bool RasterizerSceneGLES3::reflection_probe_instance_needs_redraw(RID p_instance) {
	return false;
}

bool RasterizerSceneGLES3::reflection_probe_instance_has_reflection(RID p_instance) {
	return false;
}

bool RasterizerSceneGLES3::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {
	return false;
}

bool RasterizerSceneGLES3::reflection_probe_instance_postprocess_step(RID p_instance) {
	return true;
}

RID RasterizerSceneGLES3::decal_instance_create(RID p_decal) {
	return RID();
}

void RasterizerSceneGLES3::decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) {
}

RID RasterizerSceneGLES3::lightmap_instance_create(RID p_lightmap) {
	return RID();
}

void RasterizerSceneGLES3::lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) {
}

RID RasterizerSceneGLES3::voxel_gi_instance_create(RID p_voxel_gi) {
	return RID();
}

void RasterizerSceneGLES3::voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) {
}

bool RasterizerSceneGLES3::voxel_gi_needs_update(RID p_probe) const {
	return false;
}

void RasterizerSceneGLES3::voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects) {
}

void RasterizerSceneGLES3::voxel_gi_set_quality(RS::VoxelGIQuality) {
}

void RasterizerSceneGLES3::_fill_render_list(RenderListType p_render_list, const RenderDataGLES3 *p_render_data, PassMode p_pass_mode, bool p_append) {
	GLES3::MeshStorage *mesh_storage = GLES3::MeshStorage::get_singleton();

	if (p_render_list == RENDER_LIST_OPAQUE) {
		scene_state.used_screen_texture = false;
		scene_state.used_normal_texture = false;
		scene_state.used_depth_texture = false;
	}

	Plane near_plane;
	if (p_render_data->cam_orthogonal) {
		near_plane = Plane(-p_render_data->cam_transform.basis.get_column(Vector3::AXIS_Z), p_render_data->cam_transform.origin);
		near_plane.d += p_render_data->cam_projection.get_z_near();
	}
	float z_max = p_render_data->cam_projection.get_z_far() - p_render_data->cam_projection.get_z_near();

	RenderList *rl = &render_list[p_render_list];

	// Parse any updates on our geometry, updates surface caches and such
	_update_dirty_geometry_instances();

	if (!p_append) {
		rl->clear();
		if (p_render_list == RENDER_LIST_OPAQUE) {
			render_list[RENDER_LIST_ALPHA].clear(); //opaque fills alpha too
		}
	}

	//fill list

	for (int i = 0; i < (int)p_render_data->instances->size(); i++) {
		GeometryInstanceGLES3 *inst = static_cast<GeometryInstanceGLES3 *>((*p_render_data->instances)[i]);

		if (p_render_data->cam_orthogonal) {
			Vector3 support_min = inst->transformed_aabb.get_support(-near_plane.normal);
			inst->depth = near_plane.distance_to(support_min);
		} else {
			Vector3 aabb_center = inst->transformed_aabb.position + (inst->transformed_aabb.size * 0.5);
			inst->depth = p_render_data->cam_transform.origin.distance_to(aabb_center);
		}
		uint32_t depth_layer = CLAMP(int(inst->depth * 16 / z_max), 0, 15);

		uint32_t flags = inst->base_flags; //fill flags if appropriate

		if (inst->non_uniform_scale) {
			flags |= INSTANCE_DATA_FLAGS_NON_UNIFORM_SCALE;
		}

		// Sets the index values for lookup in the shader
		// This has to be done after _setup_lights was called this frame
		// TODO, check shadow status of lights here, if using shadows, skip here and add below
		if (p_pass_mode == PASS_MODE_COLOR) {
			if (inst->omni_light_count) {
				inst->omni_light_gl_cache.resize(inst->omni_light_count);
				for (uint32_t j = 0; j < inst->omni_light_count; j++) {
					inst->omni_light_gl_cache[j] = light_instance_get_gl_id(inst->omni_lights[j]);
				}
			}
			if (inst->spot_light_count) {
				inst->spot_light_gl_cache.resize(inst->spot_light_count);
				for (uint32_t j = 0; j < inst->spot_light_count; j++) {
					inst->spot_light_gl_cache[j] = light_instance_get_gl_id(inst->spot_lights[j]);
				}
			}
		}

		inst->flags_cache = flags;

		GeometryInstanceSurface *surf = inst->surface_caches;

		while (surf) {
			// LOD

			if (p_render_data->screen_mesh_lod_threshold > 0.0 && mesh_storage->mesh_surface_has_lod(surf->surface)) {
				//lod
				Vector3 lod_support_min = inst->transformed_aabb.get_support(-p_render_data->lod_camera_plane.normal);
				Vector3 lod_support_max = inst->transformed_aabb.get_support(p_render_data->lod_camera_plane.normal);

				float distance_min = p_render_data->lod_camera_plane.distance_to(lod_support_min);
				float distance_max = p_render_data->lod_camera_plane.distance_to(lod_support_max);

				float distance = 0.0;

				if (distance_min * distance_max < 0.0) {
					//crossing plane
					distance = 0.0;
				} else if (distance_min >= 0.0) {
					distance = distance_min;
				} else if (distance_max <= 0.0) {
					distance = -distance_max;
				}

				if (p_render_data->cam_orthogonal) {
					distance = 1.0;
				}

				uint32_t indices;
				surf->lod_index = mesh_storage->mesh_surface_get_lod(surf->surface, inst->lod_model_scale * inst->lod_bias, distance * p_render_data->lod_distance_multiplier, p_render_data->screen_mesh_lod_threshold, &indices);
				/*
				if (p_render_data->render_info) {
					indices = _indices_to_primitives(surf->primitive, indices);
					if (p_render_list == RENDER_LIST_OPAQUE) { //opaque
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += indices;
					} else if (p_render_list == RENDER_LIST_SECONDARY) { //shadow
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_SHADOW][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += indices;
					}
				}
				*/
			} else {
				surf->lod_index = 0;
				/*
				if (p_render_data->render_info) {
					uint32_t to_draw = mesh_storage->mesh_surface_get_vertices_drawn_count(surf->surface);
					to_draw = _indices_to_primitives(surf->primitive, to_draw);
					to_draw *= inst->instance_count;
					if (p_render_list == RENDER_LIST_OPAQUE) { //opaque
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += mesh_storage->mesh_surface_get_vertices_drawn_count(surf->surface);
					} else if (p_render_list == RENDER_LIST_SECONDARY) { //shadow
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_SHADOW][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += mesh_storage->mesh_surface_get_vertices_drawn_count(surf->surface);
					}
				}
				*/
			}

			// ADD Element
			if (p_pass_mode == PASS_MODE_COLOR) {
#ifdef DEBUG_ENABLED
				bool force_alpha = unlikely(get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW);
#else
				bool force_alpha = false;
#endif
				if (!force_alpha && (surf->flags & GeometryInstanceSurface::FLAG_PASS_OPAQUE)) {
					rl->add_element(surf);
				}
				if (force_alpha || (surf->flags & GeometryInstanceSurface::FLAG_PASS_ALPHA)) {
					render_list[RENDER_LIST_ALPHA].add_element(surf);
				}

				if (surf->flags & GeometryInstanceSurface::FLAG_USES_SCREEN_TEXTURE) {
					scene_state.used_screen_texture = true;
				}
				if (surf->flags & GeometryInstanceSurface::FLAG_USES_NORMAL_TEXTURE) {
					scene_state.used_normal_texture = true;
				}
				if (surf->flags & GeometryInstanceSurface::FLAG_USES_DEPTH_TEXTURE) {
					scene_state.used_depth_texture = true;
				}

				/*
					Add elements here if there are shadows
				*/

			} else if (p_pass_mode == PASS_MODE_SHADOW) {
				if (surf->flags & GeometryInstanceSurface::FLAG_PASS_SHADOW) {
					rl->add_element(surf);
				}
			} else {
				if (surf->flags & (GeometryInstanceSurface::FLAG_PASS_DEPTH | GeometryInstanceSurface::FLAG_PASS_OPAQUE)) {
					rl->add_element(surf);
				}
			}

			surf->sort.depth_layer = depth_layer;

			surf = surf->next;
		}
	}
}

// Needs to be called after _setup_lights so that directional_light_count is accurate.
void RasterizerSceneGLES3::_setup_environment(const RenderDataGLES3 *p_render_data, bool p_no_fog, const Size2i &p_screen_size, bool p_flip_y, const Color &p_default_bg_color, bool p_pancake_shadows) {
	CameraMatrix correction;
	correction.set_depth_correction(p_flip_y);
	CameraMatrix projection = correction * p_render_data->cam_projection;
	//store camera into ubo
	RasterizerStorageGLES3::store_camera(projection, scene_state.ubo.projection_matrix);
	RasterizerStorageGLES3::store_camera(projection.inverse(), scene_state.ubo.inv_projection_matrix);
	RasterizerStorageGLES3::store_transform(p_render_data->cam_transform, scene_state.ubo.inv_view_matrix);
	RasterizerStorageGLES3::store_transform(p_render_data->inv_cam_transform, scene_state.ubo.view_matrix);

	scene_state.ubo.directional_light_count = p_render_data->directional_light_count;

	scene_state.ubo.z_far = p_render_data->z_far;
	scene_state.ubo.z_near = p_render_data->z_near;

	scene_state.ubo.viewport_size[0] = p_screen_size.x;
	scene_state.ubo.viewport_size[1] = p_screen_size.y;

	Size2 screen_pixel_size = Vector2(1.0, 1.0) / Size2(p_screen_size);
	scene_state.ubo.screen_pixel_size[0] = screen_pixel_size.x;
	scene_state.ubo.screen_pixel_size[1] = screen_pixel_size.y;

	//time global variables
	scene_state.ubo.time = time;

	if (is_environment(p_render_data->environment)) {
		Environment *env = environment_owner.get_or_null(p_render_data->environment);
		RS::EnvironmentBG env_bg = env->background;
		RS::EnvironmentAmbientSource ambient_src = env->ambient_source;

		float bg_energy = env->bg_energy;
		scene_state.ubo.ambient_light_color_energy[3] = bg_energy;

		scene_state.ubo.ambient_color_sky_mix = env->ambient_sky_contribution;

		//ambient
		if (ambient_src == RS::ENV_AMBIENT_SOURCE_BG && (env_bg == RS::ENV_BG_CLEAR_COLOR || env_bg == RS::ENV_BG_COLOR)) {
			Color color = env_bg == RS::ENV_BG_CLEAR_COLOR ? p_default_bg_color : env->bg_color;
			color = color.srgb_to_linear();

			scene_state.ubo.ambient_light_color_energy[0] = color.r * bg_energy;
			scene_state.ubo.ambient_light_color_energy[1] = color.g * bg_energy;
			scene_state.ubo.ambient_light_color_energy[2] = color.b * bg_energy;
			scene_state.ubo.use_ambient_light = true;
			scene_state.ubo.use_ambient_cubemap = false;
		} else {
			float energy = env->ambient_light_energy;
			Color color = env->ambient_light;
			color = color.srgb_to_linear();
			scene_state.ubo.ambient_light_color_energy[0] = color.r * energy;
			scene_state.ubo.ambient_light_color_energy[1] = color.g * energy;
			scene_state.ubo.ambient_light_color_energy[2] = color.b * energy;

			Basis sky_transform = env->sky_orientation;
			sky_transform = sky_transform.inverse() * p_render_data->cam_transform.basis;
			RasterizerStorageGLES3::store_transform_3x3(sky_transform, scene_state.ubo.radiance_inverse_xform);
			scene_state.ubo.use_ambient_cubemap = (ambient_src == RS::ENV_AMBIENT_SOURCE_BG && env_bg == RS::ENV_BG_SKY) || ambient_src == RS::ENV_AMBIENT_SOURCE_SKY;
			scene_state.ubo.use_ambient_light = scene_state.ubo.use_ambient_cubemap || ambient_src == RS::ENV_AMBIENT_SOURCE_COLOR;
		}

		//specular
		RS::EnvironmentReflectionSource ref_src = env->reflection_source;
		if ((ref_src == RS::ENV_REFLECTION_SOURCE_BG && env_bg == RS::ENV_BG_SKY) || ref_src == RS::ENV_REFLECTION_SOURCE_SKY) {
			scene_state.ubo.use_reflection_cubemap = true;
		} else {
			scene_state.ubo.use_reflection_cubemap = false;
		}

		scene_state.ubo.fog_enabled = env->fog_enabled;
		scene_state.ubo.fog_density = env->fog_density;
		scene_state.ubo.fog_height = env->fog_height;
		scene_state.ubo.fog_height_density = env->fog_height_density;
		scene_state.ubo.fog_aerial_perspective = env->fog_aerial_perspective;

		Color fog_color = env->fog_light_color.srgb_to_linear();
		float fog_energy = env->fog_light_energy;

		scene_state.ubo.fog_light_color[0] = fog_color.r * fog_energy;
		scene_state.ubo.fog_light_color[1] = fog_color.g * fog_energy;
		scene_state.ubo.fog_light_color[2] = fog_color.b * fog_energy;

		scene_state.ubo.fog_sun_scatter = env->fog_sun_scatter;

	} else {
	}

	if (scene_state.ubo_buffer == 0) {
		glGenBuffers(1, &scene_state.ubo_buffer);
	}
	glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_DATA_UNIFORM_LOCATION, scene_state.ubo_buffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(SceneState::UBO), &scene_state.ubo, GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

// Puts lights into Uniform Buffers. Needs to be called before _fill_list as this caches the index of each light in the Uniform Buffer
void RasterizerSceneGLES3::_setup_lights(const RenderDataGLES3 *p_render_data, bool p_using_shadows, uint32_t &r_directional_light_count, uint32_t &r_omni_light_count, uint32_t &r_spot_light_count) {
	GLES3::LightStorage *light_storage = GLES3::LightStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

	const Transform3D inverse_transform = p_render_data->inv_cam_transform;

	const PagedArray<RID> &lights = *p_render_data->lights;

	r_directional_light_count = 0;
	r_omni_light_count = 0;
	r_spot_light_count = 0;

	int num_lights = lights.size();

	for (int i = 0; i < num_lights; i++) {
		LightInstance *li = light_instance_owner.get_or_null(lights[i]);
		if (!li) {
			continue;
		}
		RID base = li->light;

		ERR_CONTINUE(base.is_null());

		RS::LightType type = light_storage->light_get_type(base);
		switch (type) {
			case RS::LIGHT_DIRECTIONAL: {
				if (r_directional_light_count >= RendererSceneRender::MAX_DIRECTIONAL_LIGHTS || light_storage->light_directional_get_sky_mode(base) == RS::LIGHT_DIRECTIONAL_SKY_MODE_SKY_ONLY) {
					continue;
				}

				DirectionalLightData &light_data = scene_state.directional_lights[r_directional_light_count];

				Transform3D light_transform = li->transform;

				Vector3 direction = inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, 1))).normalized();

				light_data.direction[0] = direction.x;
				light_data.direction[1] = direction.y;
				light_data.direction[2] = direction.z;

				float sign = light_storage->light_is_negative(base) ? -1 : 1;

				light_data.energy = sign * light_storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI;

				Color linear_col = light_storage->light_get_color(base).srgb_to_linear();
				light_data.color[0] = linear_col.r;
				light_data.color[1] = linear_col.g;
				light_data.color[2] = linear_col.b;

				float size = light_storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);
				light_data.size = 1.0 - Math::cos(Math::deg2rad(size)); //angle to cosine offset

				light_data.specular = light_storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR);

				r_directional_light_count++;
			} break;
			case RS::LIGHT_OMNI: {
				if (r_omni_light_count >= (uint32_t)config->max_renderable_lights) {
					continue;
				}

				const real_t distance = p_render_data->cam_transform.origin.distance_to(li->transform.origin);

				if (light_storage->light_is_distance_fade_enabled(li->light)) {
					const float fade_begin = light_storage->light_get_distance_fade_begin(li->light);
					const float fade_length = light_storage->light_get_distance_fade_length(li->light);

					if (distance > fade_begin) {
						if (distance > fade_begin + fade_length) {
							// Out of range, don't draw this light to improve performance.
							continue;
						}
					}
				}

				li->gl_id = r_omni_light_count;

				scene_state.omni_light_sort[r_omni_light_count].instance = li;
				scene_state.omni_light_sort[r_omni_light_count].depth = distance;
				r_omni_light_count++;
			} break;
			case RS::LIGHT_SPOT: {
				if (r_spot_light_count >= (uint32_t)config->max_renderable_lights) {
					continue;
				}

				const real_t distance = p_render_data->cam_transform.origin.distance_to(li->transform.origin);

				if (light_storage->light_is_distance_fade_enabled(li->light)) {
					const float fade_begin = light_storage->light_get_distance_fade_begin(li->light);
					const float fade_length = light_storage->light_get_distance_fade_length(li->light);

					if (distance > fade_begin) {
						if (distance > fade_begin + fade_length) {
							// Out of range, don't draw this light to improve performance.
							continue;
						}
					}
				}

				li->gl_id = r_spot_light_count;

				scene_state.spot_light_sort[r_spot_light_count].instance = li;
				scene_state.spot_light_sort[r_spot_light_count].depth = distance;
				r_spot_light_count++;
			} break;
		}
	}

	if (r_omni_light_count) {
		SortArray<InstanceSort<LightInstance>> sorter;
		sorter.sort(scene_state.omni_light_sort, r_omni_light_count);
	}

	if (r_spot_light_count) {
		SortArray<InstanceSort<LightInstance>> sorter;
		sorter.sort(scene_state.spot_light_sort, r_spot_light_count);
	}

	for (uint32_t i = 0; i < (r_omni_light_count + r_spot_light_count); i++) {
		uint32_t index = (i < r_omni_light_count) ? i : i - (r_omni_light_count);
		LightData &light_data = (i < r_omni_light_count) ? scene_state.omni_lights[index] : scene_state.spot_lights[index];
		//RS::LightType type = (i < omni_light_count) ? RS::LIGHT_OMNI : RS::LIGHT_SPOT;
		LightInstance *li = (i < r_omni_light_count) ? scene_state.omni_light_sort[index].instance : scene_state.spot_light_sort[index].instance;
		RID base = li->light;

		Transform3D light_transform = li->transform;
		Vector3 pos = inverse_transform.xform(light_transform.origin);

		light_data.position[0] = pos.x;
		light_data.position[1] = pos.y;
		light_data.position[2] = pos.z;

		float radius = MAX(0.001, light_storage->light_get_param(base, RS::LIGHT_PARAM_RANGE));
		light_data.inv_radius = 1.0 / radius;

		Vector3 direction = inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, -1))).normalized();

		light_data.direction[0] = direction.x;
		light_data.direction[1] = direction.y;
		light_data.direction[2] = direction.z;

		float size = light_storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

		light_data.size = size;

		float sign = light_storage->light_is_negative(base) ? -1 : 1;
		Color linear_col = light_storage->light_get_color(base).srgb_to_linear();

		// Reuse fade begin, fade length and distance for shadow LOD determination later.
		float fade_begin = 0.0;
		float fade_length = 0.0;
		real_t distance = 0.0;

		float fade = 1.0;
		if (light_storage->light_is_distance_fade_enabled(li->light)) {
			fade_begin = light_storage->light_get_distance_fade_begin(li->light);
			fade_length = light_storage->light_get_distance_fade_length(li->light);
			distance = p_render_data->cam_transform.origin.distance_to(li->transform.origin);

			if (distance > fade_begin) {
				// Use `smoothstep()` to make opacity changes more gradual and less noticeable to the player.
				fade = Math::smoothstep(0.0f, 1.0f, 1.0f - float(distance - fade_begin) / fade_length);
			}
		}

		float energy = sign * light_storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI * fade;

		light_data.color[0] = linear_col.r * energy;
		light_data.color[1] = linear_col.g * energy;
		light_data.color[2] = linear_col.b * energy;

		light_data.attenuation = light_storage->light_get_param(base, RS::LIGHT_PARAM_ATTENUATION);

		light_data.inv_spot_attenuation = 1.0f / light_storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ATTENUATION);

		float spot_angle = light_storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ANGLE);
		light_data.cos_spot_angle = Math::cos(Math::deg2rad(spot_angle));

		light_data.specular_amount = light_storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR) * 2.0;

		light_data.shadow_enabled = false;
	}

	// TODO, to avoid stalls, should rotate between 3 buffers based on frame index.
	// TODO, consider mapping the buffer as in 2D
	if (r_omni_light_count) {
		glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_OMNILIGHT_UNIFORM_LOCATION, scene_state.omni_light_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightData) * r_omni_light_count, scene_state.omni_lights);
	}

	if (r_spot_light_count) {
		glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_SPOTLIGHT_UNIFORM_LOCATION, scene_state.spot_light_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightData) * r_spot_light_count, scene_state.spot_lights);
	}

	if (r_directional_light_count) {
		glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_DIRECTIONAL_LIGHT_UNIFORM_LOCATION, scene_state.directional_light_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(DirectionalLightData) * r_directional_light_count, scene_state.directional_lights);
	}
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void RasterizerSceneGLES3::render_scene(RID p_render_buffers, const CameraData *p_camera_data, const CameraData *p_prev_camera_data, const PagedArray<GeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_mesh_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data, RendererScene::RenderInfo *r_render_info) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();
	RENDER_TIMESTAMP("Setup 3D Scene");

	RenderBuffers *rb = nullptr;
	if (p_render_buffers.is_valid()) {
		rb = render_buffers_owner.get_or_null(p_render_buffers);
		ERR_FAIL_COND(!rb);
	}

	// Assign render data
	// Use the format from rendererRD
	RenderDataGLES3 render_data;
	{
		render_data.render_buffers = p_render_buffers;
		render_data.transparent_bg = rb->is_transparent;
		// Our first camera is used by default
		render_data.cam_transform = p_camera_data->main_transform;
		render_data.inv_cam_transform = render_data.cam_transform.affine_inverse();
		render_data.cam_projection = p_camera_data->main_projection;
		render_data.view_projection[0] = p_camera_data->main_projection;
		render_data.cam_orthogonal = p_camera_data->is_orthogonal;

		render_data.view_count = p_camera_data->view_count;
		for (uint32_t v = 0; v < p_camera_data->view_count; v++) {
			render_data.view_projection[v] = p_camera_data->view_projection[v];
		}

		render_data.z_near = p_camera_data->main_projection.get_z_near();
		render_data.z_far = p_camera_data->main_projection.get_z_far();

		render_data.instances = &p_instances;
		render_data.lights = &p_lights;
		render_data.reflection_probes = &p_reflection_probes;
		render_data.environment = p_environment;
		render_data.camera_effects = p_camera_effects;
		render_data.reflection_probe = p_reflection_probe;
		render_data.reflection_probe_pass = p_reflection_probe_pass;

		// this should be the same for all cameras..
		render_data.lod_distance_multiplier = p_camera_data->main_projection.get_lod_multiplier();
		render_data.lod_camera_plane = Plane(-p_camera_data->main_transform.basis.get_column(Vector3::AXIS_Z), p_camera_data->main_transform.get_origin());

		if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_DISABLE_LOD) {
			render_data.screen_mesh_lod_threshold = 0.0;
		} else {
			render_data.screen_mesh_lod_threshold = p_screen_mesh_lod_threshold;
		}
		render_data.render_info = r_render_info;
	}

	PagedArray<RID> empty;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {
		render_data.lights = &empty;
		render_data.reflection_probes = &empty;
	}

	bool reverse_cull = false;

	///////////
	// Fill Light lists here
	//////////

	GLuint global_buffer = GLES3::MaterialStorage::get_singleton()->global_variables_get_uniform_buffer();
	glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_GLOBALS_UNIFORM_LOCATION, global_buffer);

	Color clear_color;
	if (p_render_buffers.is_valid()) {
		clear_color = texture_storage->render_target_get_clear_request_color(rb->render_target);
	} else {
		clear_color = storage->get_default_clear_color();
	}

	Environment *env = environment_owner.get_or_null(p_environment);

	bool fb_cleared = false;

	Size2i screen_size;
	screen_size.x = rb->width;
	screen_size.y = rb->height;

	bool use_wireframe = get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_WIREFRAME;

	SceneState::TonemapUBO tonemap_ubo;
	if (env) {
		tonemap_ubo.exposure = env->exposure;
		tonemap_ubo.white = env->white;
		tonemap_ubo.tonemapper = int32_t(env->tone_mapper);
	}

	if (scene_state.tonemap_buffer == 0) {
		// Only create if using 3D
		glGenBuffers(1, &scene_state.tonemap_buffer);
	}
	glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_TONEMAP_UNIFORM_LOCATION, scene_state.tonemap_buffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(SceneState::TonemapUBO), &tonemap_ubo, GL_STREAM_DRAW);

	_setup_lights(&render_data, false, render_data.directional_light_count, render_data.omni_light_count, render_data.spot_light_count);
	_setup_environment(&render_data, render_data.reflection_probe.is_valid(), screen_size, !render_data.reflection_probe.is_valid(), clear_color, false);

	_fill_render_list(RENDER_LIST_OPAQUE, &render_data, PASS_MODE_COLOR);
	render_list[RENDER_LIST_OPAQUE].sort_by_key();
	render_list[RENDER_LIST_ALPHA].sort_by_reverse_depth_and_priority();

	bool draw_sky = false;
	bool draw_sky_fog_only = false;
	bool keep_color = false;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW) {
		clear_color = Color(0, 0, 0, 1); //in overdraw mode, BG should always be black
	} else if (env) {
		RS::EnvironmentBG bg_mode = env->background;
		float bg_energy = env->bg_energy;
		switch (bg_mode) {
			case RS::ENV_BG_CLEAR_COLOR: {
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
				if (env->fog_enabled) {
					draw_sky_fog_only = true;
					GLES3::MaterialStorage::get_singleton()->material_set_param(sky_globals.fog_material, "clear_color", Variant(clear_color));
				}
			} break;
			case RS::ENV_BG_COLOR: {
				clear_color = env->bg_color;
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
				if (env->fog_enabled) {
					draw_sky_fog_only = true;
					GLES3::MaterialStorage::get_singleton()->material_set_param(sky_globals.fog_material, "clear_color", Variant(clear_color));
				}
			} break;
			case RS::ENV_BG_SKY: {
				draw_sky = true;
			} break;
			case RS::ENV_BG_CANVAS: {
				keep_color = true;
			} break;
			case RS::ENV_BG_KEEP: {
				keep_color = true;
			} break;
			case RS::ENV_BG_CAMERA_FEED: {
			} break;
			default: {
			}
		}
		// setup sky if used for ambient, reflections, or background
		if (draw_sky || draw_sky_fog_only || env->reflection_source == RS::ENV_REFLECTION_SOURCE_SKY || env->ambient_source == RS::ENV_AMBIENT_SOURCE_SKY) {
			RENDER_TIMESTAMP("Setup Sky");
			CameraMatrix projection = render_data.cam_projection;
			if (render_data.reflection_probe.is_valid()) {
				CameraMatrix correction;
				correction.set_depth_correction(true);
				projection = correction * render_data.cam_projection;
			}

			_setup_sky(env, p_render_buffers, *render_data.lights, projection, render_data.cam_transform, screen_size);

			if (env->sky.is_valid()) {
				if (env->reflection_source == RS::ENV_REFLECTION_SOURCE_SKY || env->ambient_source == RS::ENV_AMBIENT_SOURCE_SKY || (env->reflection_source == RS::ENV_REFLECTION_SOURCE_BG && env->background == RS::ENV_BG_SKY)) {
					_update_sky_radiance(env, projection, render_data.cam_transform);
				}
			} else {
				// do not try to draw sky if invalid
				draw_sky = false;
			}
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, rb->framebuffer);
	glViewport(0, 0, rb->width, rb->height);

	// Do depth prepass if it's explicitly enabled
	bool use_depth_prepass = config->use_depth_prepass;

	// Don't do depth prepass we are rendering overdraw
	use_depth_prepass = use_depth_prepass && get_debug_draw_mode() != RS::VIEWPORT_DEBUG_DRAW_OVERDRAW;

	if (use_depth_prepass) {
		RENDER_TIMESTAMP("Depth Prepass");
		//pre z pass

		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_SCISSOR_TEST);
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
		scene_state.cull_mode = GLES3::SceneShaderData::CULL_BACK;

		glColorMask(0, 0, 0, 0);
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		RenderListParameters render_list_params(render_list[RENDER_LIST_OPAQUE].elements.ptr(), render_list[RENDER_LIST_OPAQUE].elements.size(), reverse_cull, 0, use_wireframe);
		_render_list_template<PASS_MODE_DEPTH>(&render_list_params, &render_data, 0, render_list[RENDER_LIST_OPAQUE].elements.size());

		glColorMask(1, 1, 1, 1);

		fb_cleared = true;
		scene_state.used_depth_prepass = true;
	} else {
		scene_state.used_depth_prepass = false;
	}

	glBlendEquation(GL_FUNC_ADD);

	if (render_data.transparent_bg) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
	} else {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
		glDisable(GL_BLEND);
	}
	scene_state.current_blend_mode = GLES3::SceneShaderData::BLEND_MODE_MIX;

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	scene_state.current_depth_test = GLES3::SceneShaderData::DEPTH_TEST_ENABLED;
	scene_state.current_depth_draw = GLES3::SceneShaderData::DEPTH_DRAW_OPAQUE;

	if (!fb_cleared) {
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	if (!keep_color) {
		glClearBufferfv(GL_COLOR, 0, clear_color.components);
	}
	RENDER_TIMESTAMP("Render Opaque Pass");
	uint32_t spec_constant_base_flags = 0;

	{
		// Specialization Constants that apply for entire rendering pass.
		if (render_data.directional_light_count == 0) {
			spec_constant_base_flags |= 1 << SPEC_CONSTANT_DISABLE_DIRECTIONAL_LIGHTS;
		}

		if (!env || (env && !env->fog_enabled)) {
			spec_constant_base_flags |= 1 << SPEC_CONSTANT_DISABLE_FOG;
		}
	}
	// Render Opaque Objects.
	RenderListParameters render_list_params(render_list[RENDER_LIST_OPAQUE].elements.ptr(), render_list[RENDER_LIST_OPAQUE].elements.size(), reverse_cull, spec_constant_base_flags, use_wireframe);

	_render_list_template<PASS_MODE_COLOR>(&render_list_params, &render_data, 0, render_list[RENDER_LIST_OPAQUE].elements.size());

	if (draw_sky) {
		RENDER_TIMESTAMP("Render Sky");
		if (scene_state.current_depth_test != GLES3::SceneShaderData::DEPTH_TEST_ENABLED) {
			glEnable(GL_DEPTH_TEST);
			scene_state.current_depth_test = GLES3::SceneShaderData::DEPTH_TEST_ENABLED;
		}
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glDisable(GL_BLEND);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		scene_state.current_depth_test = GLES3::SceneShaderData::DEPTH_TEST_ENABLED;
		scene_state.current_depth_draw = GLES3::SceneShaderData::DEPTH_DRAW_DISABLED;
		scene_state.cull_mode = GLES3::SceneShaderData::CULL_BACK;

		_draw_sky(env, render_data.cam_projection, render_data.cam_transform);
	}

	RENDER_TIMESTAMP("Render 3D Transparent Pass");
	glEnable(GL_BLEND);

	//Render transparent pass
	RenderListParameters render_list_params_alpha(render_list[RENDER_LIST_ALPHA].elements.ptr(), render_list[RENDER_LIST_ALPHA].elements.size(), reverse_cull, spec_constant_base_flags, use_wireframe);

	_render_list_template<PASS_MODE_COLOR_TRANSPARENT>(&render_list_params_alpha, &render_data, 0, render_list[RENDER_LIST_ALPHA].elements.size(), true);

	if (p_render_buffers.is_valid()) {
		_render_buffers_debug_draw(p_render_buffers, p_shadow_atlas, p_occluder_debug_tex);
	}
	glDisable(GL_BLEND);
	texture_storage->render_target_disable_clear_request(rb->render_target);
}

template <PassMode p_pass_mode>
void RasterizerSceneGLES3::_render_list_template(RenderListParameters *p_params, const RenderDataGLES3 *p_render_data, uint32_t p_from_element, uint32_t p_to_element, bool p_alpha_pass) {
	GLES3::MeshStorage *mesh_storage = GLES3::MeshStorage::get_singleton();
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

	GLuint prev_vertex_array_gl = 0;
	GLuint prev_index_array_gl = 0;

	GLES3::SceneMaterialData *prev_material_data = nullptr;
	GLES3::SceneShaderData *prev_shader = nullptr;
	GeometryInstanceGLES3 *prev_inst = nullptr;

	SceneShaderGLES3::ShaderVariant shader_variant = SceneShaderGLES3::MODE_COLOR; // Assigned to silence wrong -Wmaybe-initialized.

	switch (p_pass_mode) {
		case PASS_MODE_COLOR:
		case PASS_MODE_COLOR_TRANSPARENT: {
		} break;
		case PASS_MODE_COLOR_ADDITIVE: {
			shader_variant = SceneShaderGLES3::MODE_ADDITIVE;
		} break;
		case PASS_MODE_SHADOW:
		case PASS_MODE_DEPTH: {
			shader_variant = SceneShaderGLES3::MODE_DEPTH;
		} break;
	}

	if (p_pass_mode == PASS_MODE_COLOR || p_pass_mode == PASS_MODE_COLOR_TRANSPARENT) {
		Environment *env = environment_owner.get_or_null(p_render_data->environment);
		glActiveTexture(GL_TEXTURE0 + config->max_texture_image_units - 2);
		GLuint texture_to_bind = texture_storage->get_texture(texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_CUBEMAP_BLACK))->tex_id;
		if (env) {
			Sky *sky = sky_owner.get_or_null(env->sky);
			if (sky && sky->radiance != 0) {
				texture_to_bind = sky->radiance;
				// base_spec_constant |= USE_RADIANCE_MAP;
			}
			glBindTexture(GL_TEXTURE_CUBE_MAP, texture_to_bind);
		}
	}

	for (uint32_t i = p_from_element; i < p_to_element; i++) {
		const GeometryInstanceSurface *surf = p_params->elements[i];
		GeometryInstanceGLES3 *inst = surf->owner;

		if (p_pass_mode == PASS_MODE_COLOR && !(surf->flags & GeometryInstanceSurface::FLAG_PASS_OPAQUE)) {
			continue; // Objects with "Depth-prepass" transparency are included in both render lists, but should only be rendered in the transparent pass
		}

		if (inst->instance_count == 0) {
			continue;
		}

		//uint32_t base_spec_constants = p_params->spec_constant_base_flags;

		GLES3::SceneShaderData *shader;
		GLES3::SceneMaterialData *material_data;
		void *mesh_surface;

		if (p_pass_mode == PASS_MODE_SHADOW) {
			shader = surf->shader_shadow;
			material_data = surf->material_shadow;
			mesh_surface = surf->surface_shadow;
		} else {
			shader = surf->shader;
			material_data = surf->material;
			mesh_surface = surf->surface;
		}

		if (!mesh_surface) {
			continue;
		}

		if (p_pass_mode == PASS_MODE_COLOR_TRANSPARENT) {
			if (scene_state.current_depth_test != shader->depth_test) {
				if (shader->depth_test == GLES3::SceneShaderData::DEPTH_TEST_DISABLED) {
					glDisable(GL_DEPTH_TEST);
				} else {
					glEnable(GL_DEPTH_TEST);
				}
				scene_state.current_depth_test = shader->depth_test;
			}
		}

		if (scene_state.current_depth_draw != shader->depth_draw) {
			switch (shader->depth_draw) {
				case GLES3::SceneShaderData::DEPTH_DRAW_OPAQUE: {
					glDepthMask(p_pass_mode == PASS_MODE_COLOR);
				} break;
				case GLES3::SceneShaderData::DEPTH_DRAW_ALWAYS: {
					glDepthMask(GL_TRUE);
				} break;
				case GLES3::SceneShaderData::DEPTH_DRAW_DISABLED: {
					glDepthMask(GL_FALSE);
				} break;
			}

			scene_state.current_depth_draw = shader->depth_draw;
		}

		if (p_pass_mode == PASS_MODE_COLOR_TRANSPARENT || p_pass_mode == PASS_MODE_COLOR_ADDITIVE) {
			GLES3::SceneShaderData::BlendMode desired_blend_mode;
			if (p_pass_mode == PASS_MODE_COLOR_ADDITIVE) {
				desired_blend_mode = GLES3::SceneShaderData::BLEND_MODE_ADD;
			} else {
				desired_blend_mode = shader->blend_mode;
			}

			if (desired_blend_mode != scene_state.current_blend_mode) {
				switch (desired_blend_mode) {
					case GLES3::SceneShaderData::BLEND_MODE_MIX: {
						glBlendEquation(GL_FUNC_ADD);
						if (p_render_data->transparent_bg) {
							glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
						} else {
							glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
						}

					} break;
					case GLES3::SceneShaderData::BLEND_MODE_ADD: {
						glBlendEquation(GL_FUNC_ADD);
						glBlendFunc(p_pass_mode == PASS_MODE_COLOR_TRANSPARENT ? GL_SRC_ALPHA : GL_ONE, GL_ONE);

					} break;
					case GLES3::SceneShaderData::BLEND_MODE_SUB: {
						glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
						glBlendFunc(GL_SRC_ALPHA, GL_ONE);

					} break;
					case GLES3::SceneShaderData::BLEND_MODE_MUL: {
						glBlendEquation(GL_FUNC_ADD);
						if (p_render_data->transparent_bg) {
							glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
						} else {
							glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
						}

					} break;
					case GLES3::SceneShaderData::BLEND_MODE_ALPHA_TO_COVERAGE: {
						// Do nothing for now.
					} break;
				}
				scene_state.current_blend_mode = desired_blend_mode;
			}
		}

		//find cull variant
		GLES3::SceneShaderData::Cull cull_mode = shader->cull_mode;

		if ((surf->flags & GeometryInstanceSurface::FLAG_USES_DOUBLE_SIDED_SHADOWS)) {
			cull_mode = GLES3::SceneShaderData::CULL_DISABLED;
		} else {
			bool mirror = inst->mirror;
			if (p_params->reverse_cull) {
				mirror = !mirror;
			}
			if (cull_mode == GLES3::SceneShaderData::CULL_FRONT && mirror) {
				cull_mode = GLES3::SceneShaderData::CULL_BACK;
			} else if (cull_mode == GLES3::SceneShaderData::CULL_BACK && mirror) {
				cull_mode = GLES3::SceneShaderData::CULL_FRONT;
			}
		}

		if (scene_state.cull_mode != cull_mode) {
			if (cull_mode == GLES3::SceneShaderData::CULL_DISABLED) {
				glDisable(GL_CULL_FACE);
			} else {
				if (scene_state.cull_mode == GLES3::SceneShaderData::CULL_DISABLED) {
					// Last time was disabled, so enable and set proper face.
					glEnable(GL_CULL_FACE);
				}
				glCullFace(cull_mode == GLES3::SceneShaderData::CULL_FRONT ? GL_FRONT : GL_BACK);
			}
			scene_state.cull_mode = cull_mode;
		}

		RS::PrimitiveType primitive = surf->primitive;
		static const GLenum prim[5] = { GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_TRIANGLES, GL_TRIANGLE_STRIP };
		GLenum primitive_gl = prim[int(primitive)];

		GLuint vertex_array_gl = 0;
		GLuint index_array_gl = 0;

		//skeleton and blend shape
		if (surf->owner->mesh_instance.is_valid()) {
			mesh_storage->mesh_instance_surface_get_vertex_arrays_and_format(surf->owner->mesh_instance, surf->surface_index, shader->vertex_input_mask, vertex_array_gl);
		} else {
			mesh_storage->mesh_surface_get_vertex_arrays_and_format(mesh_surface, shader->vertex_input_mask, vertex_array_gl);
		}

		index_array_gl = mesh_storage->mesh_surface_get_index_buffer(mesh_surface, surf->lod_index);

		if (prev_vertex_array_gl != vertex_array_gl) {
			glBindVertexArray(vertex_array_gl);
			prev_vertex_array_gl = vertex_array_gl;
		}

		bool use_index_buffer = false;
		if (prev_index_array_gl != index_array_gl) {
			if (index_array_gl != 0) {
				// Bind index each time so we can use LODs
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_array_gl);
				use_index_buffer = true;
			}
			prev_index_array_gl = index_array_gl;
		}

		Transform3D world_transform;
		if (inst->store_transform_cache) {
			world_transform = inst->transform;
		}

		if (prev_material_data != material_data) {
			material_data->bind_uniforms();
			prev_material_data = material_data;
		}

		if (prev_shader != shader) {
			material_storage->shaders.scene_shader.version_bind_shader(shader->version, shader_variant);
			float opaque_prepass_threshold = 0.0;
			if (p_pass_mode == PASS_MODE_DEPTH) {
				opaque_prepass_threshold = 0.99;
			} else if (p_pass_mode == PASS_MODE_SHADOW) {
				opaque_prepass_threshold = 0.1;
			}

			material_storage->shaders.scene_shader.version_set_uniform(SceneShaderGLES3::OPAQUE_PREPASS_THRESHOLD, opaque_prepass_threshold, shader->version, shader_variant);

			prev_shader = shader;
		}

		if (prev_inst != inst) {
			// Rebind the light indices.
			material_storage->shaders.scene_shader.version_set_uniform(SceneShaderGLES3::OMNI_LIGHT_COUNT, inst->omni_light_count, shader->version, shader_variant);
			material_storage->shaders.scene_shader.version_set_uniform(SceneShaderGLES3::SPOT_LIGHT_COUNT, inst->spot_light_count, shader->version, shader_variant);

			if (inst->omni_light_count) {
				glUniform1uiv(material_storage->shaders.scene_shader.version_get_uniform(SceneShaderGLES3::OMNI_LIGHT_INDICES, shader->version, shader_variant), inst->omni_light_count, inst->omni_light_gl_cache.ptr());
			}

			if (inst->spot_light_count) {
				glUniform1uiv(material_storage->shaders.scene_shader.version_get_uniform(SceneShaderGLES3::SPOT_LIGHT_INDICES, shader->version, shader_variant), inst->spot_light_count, inst->spot_light_gl_cache.ptr());
			}

			prev_inst = inst;
		}

		material_storage->shaders.scene_shader.version_set_uniform(SceneShaderGLES3::WORLD_TRANSFORM, world_transform, shader->version, shader_variant);

		if (use_index_buffer) {
			glDrawElements(primitive_gl, mesh_storage->mesh_surface_get_vertices_drawn_count(mesh_surface), mesh_storage->mesh_surface_get_index_type(mesh_surface), 0);
		} else {
			glDrawArrays(primitive_gl, 0, mesh_storage->mesh_surface_get_vertices_drawn_count(mesh_surface));
		}
	}
}

void RasterizerSceneGLES3::render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) {
}

void RasterizerSceneGLES3::render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) {
}

void RasterizerSceneGLES3::set_time(double p_time, double p_step) {
	time = p_time;
	time_step = p_step;
}

void RasterizerSceneGLES3::set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
}

RID RasterizerSceneGLES3::render_buffers_create() {
	RenderBuffers rb;
	return render_buffers_owner.make_rid(rb);
}

void RasterizerSceneGLES3::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_internal_width, int p_internal_height, int p_width, int p_height, float p_fsr_sharpness, float p_fsr_mipmap_bias, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();

	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	//rb->internal_width = p_internal_width; // ignore for now
	//rb->internal_height = p_internal_height;
	rb->width = p_width;
	rb->height = p_height;
	//rb->fsr_sharpness = p_fsr_sharpness;
	rb->render_target = p_render_target;
	//rb->msaa = p_msaa;
	//rb->screen_space_aa = p_screen_space_aa;
	//rb->use_debanding = p_use_debanding;
	//rb->view_count = p_view_count;

	_free_render_buffer_data(rb);

	GLES3::RenderTarget *rt = texture_storage->get_render_target(p_render_target);

	rb->is_transparent = rt->is_transparent;

	// framebuffer
	glGenFramebuffers(1, &rb->framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, rb->framebuffer);

	glBindTexture(GL_TEXTURE_2D, rt->color);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

	glGenTextures(1, &rb->depth_texture);
	glBindTexture(GL_TEXTURE_2D, rb->depth_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, rt->size.x, rt->size.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rb->depth_texture, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, texture_storage->system_fbo);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		_free_render_buffer_data(rb);
		WARN_PRINT("Could not create 3D renderbuffer, status: " + texture_storage->get_framebuffer_error(status));
		return;
	}
}

void RasterizerSceneGLES3::_free_render_buffer_data(RenderBuffers *rb) {
	if (rb->depth_texture) {
		glDeleteTextures(1, &rb->depth_texture);
		rb->depth_texture = 0;
	}
	if (rb->framebuffer) {
		glDeleteFramebuffers(1, &rb->framebuffer);
		rb->framebuffer = 0;
	}
}

//clear render buffers
/*


		if (rt->copy_screen_effect.color) {
		glDeleteFramebuffers(1, &rt->copy_screen_effect.fbo);
		rt->copy_screen_effect.fbo = 0;

		glDeleteTextures(1, &rt->copy_screen_effect.color);
		rt->copy_screen_effect.color = 0;
	}

	if (rt->multisample_active) {
		glDeleteFramebuffers(1, &rt->multisample_fbo);
		rt->multisample_fbo = 0;

		glDeleteRenderbuffers(1, &rt->multisample_depth);
		rt->multisample_depth = 0;

		glDeleteRenderbuffers(1, &rt->multisample_color);

		rt->multisample_color = 0;
	}
*/

void RasterizerSceneGLES3::_render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas, RID p_occlusion_buffer) {
}

void RasterizerSceneGLES3::gi_set_use_half_resolution(bool p_enable) {
}

void RasterizerSceneGLES3::screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_curve) {
}

bool RasterizerSceneGLES3::screen_space_roughness_limiter_is_active() const {
	return false;
}

void RasterizerSceneGLES3::sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) {
}

void RasterizerSceneGLES3::sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) {
}

TypedArray<Image> RasterizerSceneGLES3::bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) {
	return TypedArray<Image>();
}

bool RasterizerSceneGLES3::free(RID p_rid) {
	if (environment_owner.owns(p_rid)) {
		environment_owner.free(p_rid);
	} else if (sky_owner.owns(p_rid)) {
		Sky *sky = sky_owner.get_or_null(p_rid);
		ERR_FAIL_COND_V(!sky, false);
		_free_sky_data(sky);
		sky_owner.free(p_rid);
	} else if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.get_or_null(p_rid);
		ERR_FAIL_COND_V(!rb, false);
		_free_render_buffer_data(rb);
		render_buffers_owner.free(p_rid);

	} else if (light_instance_owner.owns(p_rid)) {
		LightInstance *light_instance = light_instance_owner.get_or_null(p_rid);
		ERR_FAIL_COND_V(!light_instance, false);
		light_instance_owner.free(p_rid);
	} else {
		return false;
	}
	return true;
}

void RasterizerSceneGLES3::update() {
	_update_dirty_skys();
}

void RasterizerSceneGLES3::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
}

void RasterizerSceneGLES3::decals_set_filter(RS::DecalFilter p_filter) {
}

void RasterizerSceneGLES3::light_projectors_set_filter(RS::LightProjectorFilter p_filter) {
}

RasterizerSceneGLES3::RasterizerSceneGLES3(RasterizerStorageGLES3 *p_storage) {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

	storage = p_storage;

	{
		// Setup Lights

		config->max_renderable_lights = MIN(config->max_renderable_lights, config->max_uniform_buffer_size / (int)sizeof(RasterizerSceneGLES3::LightData));
		config->max_lights_per_object = MIN(config->max_lights_per_object, config->max_renderable_lights);

		uint32_t light_buffer_size = config->max_renderable_lights * sizeof(LightData);
		scene_state.omni_lights = memnew_arr(LightData, config->max_renderable_lights);
		scene_state.omni_light_sort = memnew_arr(InstanceSort<LightInstance>, config->max_renderable_lights);
		glGenBuffers(1, &scene_state.omni_light_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, scene_state.omni_light_buffer);
		glBufferData(GL_UNIFORM_BUFFER, light_buffer_size, nullptr, GL_STREAM_DRAW);

		scene_state.spot_lights = memnew_arr(LightData, config->max_renderable_lights);
		scene_state.spot_light_sort = memnew_arr(InstanceSort<LightInstance>, config->max_renderable_lights);
		glGenBuffers(1, &scene_state.spot_light_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, scene_state.spot_light_buffer);
		glBufferData(GL_UNIFORM_BUFFER, light_buffer_size, nullptr, GL_STREAM_DRAW);

		uint32_t directional_light_buffer_size = MAX_DIRECTIONAL_LIGHTS * sizeof(DirectionalLightData);
		scene_state.directional_lights = memnew_arr(DirectionalLightData, MAX_DIRECTIONAL_LIGHTS);
		glGenBuffers(1, &scene_state.directional_light_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, scene_state.directional_light_buffer);
		glBufferData(GL_UNIFORM_BUFFER, directional_light_buffer_size, nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	{
		sky_globals.max_directional_lights = 4;
		uint32_t directional_light_buffer_size = sky_globals.max_directional_lights * sizeof(DirectionalLightData);
		sky_globals.directional_lights = memnew_arr(DirectionalLightData, sky_globals.max_directional_lights);
		sky_globals.last_frame_directional_lights = memnew_arr(DirectionalLightData, sky_globals.max_directional_lights);
		sky_globals.last_frame_directional_light_count = sky_globals.max_directional_lights + 1;
		glGenBuffers(1, &sky_globals.directional_light_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, sky_globals.directional_light_buffer);
		glBufferData(GL_UNIFORM_BUFFER, directional_light_buffer_size, nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	{
		String global_defines;
		global_defines += "#define MAX_GLOBAL_VARIABLES 256\n"; // TODO: this is arbitrary for now
		global_defines += "\n#define MAX_LIGHT_DATA_STRUCTS " + itos(config->max_renderable_lights) + "\n";
		global_defines += "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(MAX_DIRECTIONAL_LIGHTS) + "\n";
		global_defines += "\n#define MAX_FORWARD_LIGHTS " + itos(config->max_lights_per_object) + "\n";
		material_storage->shaders.scene_shader.initialize(global_defines);
		scene_globals.shader_default_version = material_storage->shaders.scene_shader.version_create();
		material_storage->shaders.scene_shader.version_bind_shader(scene_globals.shader_default_version, SceneShaderGLES3::MODE_COLOR);
	}

	{
		//default material and shader
		scene_globals.default_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(scene_globals.default_shader);
		material_storage->shader_set_code(scene_globals.default_shader, R"(
// Default 3D material shader (clustered).

shader_type spatial;

void vertex() {
	ROUGHNESS = 0.8;
}

void fragment() {
	ALBEDO = vec3(0.6);
	ROUGHNESS = 0.8;
	METALLIC = 0.2;
}
)");
		scene_globals.default_material = material_storage->material_allocate();
		material_storage->material_initialize(scene_globals.default_material);
		material_storage->material_set_shader(scene_globals.default_material, scene_globals.default_shader);
	}

	{
		// Initialize Sky stuff
		sky_globals.roughness_layers = GLOBAL_GET("rendering/reflections/sky_reflections/roughness_layers");
		sky_globals.ggx_samples = GLOBAL_GET("rendering/reflections/sky_reflections/ggx_samples");

		String global_defines;
		global_defines += "#define MAX_GLOBAL_VARIABLES 256\n"; // TODO: this is arbitrary for now
		global_defines += "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(sky_globals.max_directional_lights) + "\n";
		material_storage->shaders.sky_shader.initialize(global_defines);
		sky_globals.shader_default_version = material_storage->shaders.sky_shader.version_create();
		material_storage->shaders.sky_shader.version_bind_shader(sky_globals.shader_default_version, SkyShaderGLES3::MODE_BACKGROUND);

		material_storage->shaders.cubemap_filter_shader.initialize();
		scene_globals.cubemap_filter_shader_version = material_storage->shaders.cubemap_filter_shader.version_create();
		material_storage->shaders.cubemap_filter_shader.version_bind_shader(scene_globals.cubemap_filter_shader_version, CubemapFilterShaderGLES3::MODE_DEFAULT);
	}

	{
		sky_globals.default_shader = material_storage->shader_allocate();

		material_storage->shader_initialize(sky_globals.default_shader);

		material_storage->shader_set_code(sky_globals.default_shader, R"(
// Default sky shader.

shader_type sky;

void sky() {
	COLOR = vec3(0.0);
}
)");
		sky_globals.default_material = material_storage->material_allocate();
		material_storage->material_initialize(sky_globals.default_material);

		material_storage->material_set_shader(sky_globals.default_material, sky_globals.default_shader);
	}
	{
		sky_globals.fog_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(sky_globals.fog_shader);

		material_storage->shader_set_code(sky_globals.fog_shader, R"(
// Default clear color sky shader.

shader_type sky;

uniform vec4 clear_color;

void sky() {
	COLOR = clear_color.rgb;
}
)");
		sky_globals.fog_material = material_storage->material_allocate();
		material_storage->material_initialize(sky_globals.fog_material);

		material_storage->material_set_shader(sky_globals.fog_material, sky_globals.fog_shader);
	}

	{
		glGenBuffers(1, &sky_globals.screen_triangle);
		glBindBuffer(GL_ARRAY_BUFFER, sky_globals.screen_triangle);

		const float qv[6] = {
			-1.0f,
			-1.0f,
			3.0f,
			-1.0f,
			-1.0f,
			3.0f,
		};

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, qv, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		glGenVertexArrays(1, &sky_globals.screen_triangle_array);
		glBindVertexArray(sky_globals.screen_triangle_array);
		glBindBuffer(GL_ARRAY_BUFFER, sky_globals.screen_triangle);
		glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);
		glEnableVertexAttribArray(RS::ARRAY_VERTEX);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}

	// Radical inverse vdc cache texture used for cubemap filtering.
	{
		glGenTextures(1, &sky_globals.radical_inverse_vdc_cache_tex);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, sky_globals.radical_inverse_vdc_cache_tex);

		uint8_t radical_inverse[512];

		for (uint32_t i = 0; i < 512; i++) {
			uint32_t bits = i;

			bits = (bits << 16) | (bits >> 16);
			bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
			bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
			bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
			bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);

			float value = float(bits) * 2.3283064365386963e-10;
			radical_inverse[i] = uint8_t(CLAMP(value * 255.0, 0, 255));
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 512, 1, 0, GL_RED, GL_UNSIGNED_BYTE, radical_inverse);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //need this for proper sampling

		glBindTexture(GL_TEXTURE_2D, 0);
	}
#ifdef GLES_OVER_GL
	glEnable(_EXT_TEXTURE_CUBE_MAP_SEAMLESS);
#endif
}

RasterizerSceneGLES3::~RasterizerSceneGLES3() {
	glDeleteBuffers(1, &scene_state.directional_light_buffer);
	glDeleteBuffers(1, &scene_state.omni_light_buffer);
	glDeleteBuffers(1, &scene_state.spot_light_buffer);
	memdelete_arr(scene_state.directional_lights);
	memdelete_arr(scene_state.omni_lights);
	memdelete_arr(scene_state.spot_lights);
	memdelete_arr(scene_state.omni_light_sort);
	memdelete_arr(scene_state.spot_light_sort);

	// Scene Shader
	GLES3::MaterialStorage::get_singleton()->shaders.scene_shader.version_free(scene_globals.shader_default_version);
	GLES3::MaterialStorage::get_singleton()->shaders.cubemap_filter_shader.version_free(scene_globals.cubemap_filter_shader_version);
	storage->free(scene_globals.default_material);
	storage->free(scene_globals.default_shader);

	// Sky Shader
	GLES3::MaterialStorage::get_singleton()->shaders.sky_shader.version_free(sky_globals.shader_default_version);
	storage->free(sky_globals.default_material);
	storage->free(sky_globals.default_shader);
	storage->free(sky_globals.fog_material);
	storage->free(sky_globals.fog_shader);
	glDeleteBuffers(1, &sky_globals.screen_triangle);
	glDeleteVertexArrays(1, &sky_globals.screen_triangle_array);
	glDeleteTextures(1, &sky_globals.radical_inverse_vdc_cache_tex);
	glDeleteBuffers(1, &sky_globals.directional_light_buffer);
	memdelete_arr(sky_globals.directional_lights);
	memdelete_arr(sky_globals.last_frame_directional_lights);
}

#endif // GLES3_ENABLED
