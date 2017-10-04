/*************************************************************************/
/*  visual_server_scene.cpp                                              */
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
#include "visual_server_scene.h"
#include "os/os.h"
#include "visual_server_global.h"
#include "visual_server_raster.h"
/* CAMERA API */

RID VisualServerScene::camera_create() {

	Camera *camera = memnew(Camera);
	return camera_owner.make_rid(camera);
}

void VisualServerScene::camera_set_perspective(RID p_camera, float p_fovy_degrees, float p_z_near, float p_z_far) {

	Camera *camera = camera_owner.get(p_camera);
	ERR_FAIL_COND(!camera);
	camera->type = Camera::PERSPECTIVE;
	camera->fov = p_fovy_degrees;
	camera->znear = p_z_near;
	camera->zfar = p_z_far;
}

void VisualServerScene::camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far) {

	Camera *camera = camera_owner.get(p_camera);
	ERR_FAIL_COND(!camera);
	camera->type = Camera::ORTHOGONAL;
	camera->size = p_size;
	camera->znear = p_z_near;
	camera->zfar = p_z_far;
}

void VisualServerScene::camera_set_transform(RID p_camera, const Transform &p_transform) {

	Camera *camera = camera_owner.get(p_camera);
	ERR_FAIL_COND(!camera);
	camera->transform = p_transform.orthonormalized();
}

void VisualServerScene::camera_set_cull_mask(RID p_camera, uint32_t p_layers) {

	Camera *camera = camera_owner.get(p_camera);
	ERR_FAIL_COND(!camera);

	camera->visible_layers = p_layers;
}

void VisualServerScene::camera_set_environment(RID p_camera, RID p_env) {

	Camera *camera = camera_owner.get(p_camera);
	ERR_FAIL_COND(!camera);
	camera->env = p_env;
}

void VisualServerScene::camera_set_use_vertical_aspect(RID p_camera, bool p_enable) {

	Camera *camera = camera_owner.get(p_camera);
	ERR_FAIL_COND(!camera);
	camera->vaspect = p_enable;
}

/* SCENARIO API */

void *VisualServerScene::_instance_pair(void *p_self, OctreeElementID, Instance *p_A, int, OctreeElementID, Instance *p_B, int) {

	//VisualServerScene *self = (VisualServerScene*)p_self;
	Instance *A = p_A;
	Instance *B = p_B;

	//instance indices are designed so greater always contains lesser
	if (A->base_type > B->base_type) {
		SWAP(A, B); //lesser always first
	}

	if (B->base_type == VS::INSTANCE_LIGHT && ((1 << A->base_type) & VS::INSTANCE_GEOMETRY_MASK)) {

		InstanceLightData *light = static_cast<InstanceLightData *>(B->base_data);
		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(A->base_data);

		InstanceLightData::PairInfo pinfo;
		pinfo.geometry = A;
		pinfo.L = geom->lighting.push_back(B);

		List<InstanceLightData::PairInfo>::Element *E = light->geometries.push_back(pinfo);

		if (geom->can_cast_shadows) {

			light->shadow_dirty = true;
		}
		geom->lighting_dirty = true;

		return E; //this element should make freeing faster
	} else if (B->base_type == VS::INSTANCE_REFLECTION_PROBE && ((1 << A->base_type) & VS::INSTANCE_GEOMETRY_MASK)) {

		InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(B->base_data);
		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(A->base_data);

		InstanceReflectionProbeData::PairInfo pinfo;
		pinfo.geometry = A;
		pinfo.L = geom->reflection_probes.push_back(B);

		List<InstanceReflectionProbeData::PairInfo>::Element *E = reflection_probe->geometries.push_back(pinfo);

		geom->reflection_dirty = true;

		return E; //this element should make freeing faster
	} else if (B->base_type == VS::INSTANCE_GI_PROBE && ((1 << A->base_type) & VS::INSTANCE_GEOMETRY_MASK)) {

		InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(B->base_data);
		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(A->base_data);

		InstanceGIProbeData::PairInfo pinfo;
		pinfo.geometry = A;
		pinfo.L = geom->gi_probes.push_back(B);

		List<InstanceGIProbeData::PairInfo>::Element *E = gi_probe->geometries.push_back(pinfo);

		geom->gi_probes_dirty = true;

		return E; //this element should make freeing faster

	} else if (B->base_type == VS::INSTANCE_GI_PROBE && A->base_type == VS::INSTANCE_LIGHT) {

		InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(B->base_data);
		return gi_probe->lights.insert(A);
	}

	return NULL;
}
void VisualServerScene::_instance_unpair(void *p_self, OctreeElementID, Instance *p_A, int, OctreeElementID, Instance *p_B, int, void *udata) {

	//VisualServerScene *self = (VisualServerScene*)p_self;
	Instance *A = p_A;
	Instance *B = p_B;

	//instance indices are designed so greater always contains lesser
	if (A->base_type > B->base_type) {
		SWAP(A, B); //lesser always first
	}

	if (B->base_type == VS::INSTANCE_LIGHT && ((1 << A->base_type) & VS::INSTANCE_GEOMETRY_MASK)) {

		InstanceLightData *light = static_cast<InstanceLightData *>(B->base_data);
		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(A->base_data);

		List<InstanceLightData::PairInfo>::Element *E = reinterpret_cast<List<InstanceLightData::PairInfo>::Element *>(udata);

		geom->lighting.erase(E->get().L);
		light->geometries.erase(E);

		if (geom->can_cast_shadows) {
			light->shadow_dirty = true;
		}
		geom->lighting_dirty = true;

	} else if (B->base_type == VS::INSTANCE_REFLECTION_PROBE && ((1 << A->base_type) & VS::INSTANCE_GEOMETRY_MASK)) {

		InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(B->base_data);
		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(A->base_data);

		List<InstanceReflectionProbeData::PairInfo>::Element *E = reinterpret_cast<List<InstanceReflectionProbeData::PairInfo>::Element *>(udata);

		geom->reflection_probes.erase(E->get().L);
		reflection_probe->geometries.erase(E);

		geom->reflection_dirty = true;

	} else if (B->base_type == VS::INSTANCE_GI_PROBE && ((1 << A->base_type) & VS::INSTANCE_GEOMETRY_MASK)) {

		InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(B->base_data);
		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(A->base_data);

		List<InstanceGIProbeData::PairInfo>::Element *E = reinterpret_cast<List<InstanceGIProbeData::PairInfo>::Element *>(udata);

		geom->gi_probes.erase(E->get().L);
		gi_probe->geometries.erase(E);

		geom->gi_probes_dirty = true;

	} else if (B->base_type == VS::INSTANCE_GI_PROBE && A->base_type == VS::INSTANCE_LIGHT) {

		InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(B->base_data);
		Set<Instance *>::Element *E = reinterpret_cast<Set<Instance *>::Element *>(udata);

		gi_probe->lights.erase(E);
	}
}

RID VisualServerScene::scenario_create() {

	Scenario *scenario = memnew(Scenario);
	ERR_FAIL_COND_V(!scenario, RID());
	RID scenario_rid = scenario_owner.make_rid(scenario);
	scenario->self = scenario_rid;

	scenario->octree.set_pair_callback(_instance_pair, this);
	scenario->octree.set_unpair_callback(_instance_unpair, this);
	scenario->reflection_probe_shadow_atlas = VSG::scene_render->shadow_atlas_create();
	VSG::scene_render->shadow_atlas_set_size(scenario->reflection_probe_shadow_atlas, 1024); //make enough shadows for close distance, don't bother with rest
	VSG::scene_render->shadow_atlas_set_quadrant_subdivision(scenario->reflection_probe_shadow_atlas, 0, 4);
	VSG::scene_render->shadow_atlas_set_quadrant_subdivision(scenario->reflection_probe_shadow_atlas, 1, 4);
	VSG::scene_render->shadow_atlas_set_quadrant_subdivision(scenario->reflection_probe_shadow_atlas, 2, 4);
	VSG::scene_render->shadow_atlas_set_quadrant_subdivision(scenario->reflection_probe_shadow_atlas, 3, 8);
	scenario->reflection_atlas = VSG::scene_render->reflection_atlas_create();

	return scenario_rid;
}

void VisualServerScene::scenario_set_debug(RID p_scenario, VS::ScenarioDebugMode p_debug_mode) {

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->debug = p_debug_mode;
}

void VisualServerScene::scenario_set_environment(RID p_scenario, RID p_environment) {

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->environment = p_environment;
}

void VisualServerScene::scenario_set_fallback_environment(RID p_scenario, RID p_environment) {

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->fallback_environment = p_environment;
}

void VisualServerScene::scenario_set_reflection_atlas_size(RID p_scenario, int p_size, int p_subdiv) {

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	VSG::scene_render->reflection_atlas_set_size(scenario->reflection_atlas, p_size);
	VSG::scene_render->reflection_atlas_set_subdivision(scenario->reflection_atlas, p_subdiv);
}

/* INSTANCING API */

void VisualServerScene::_instance_queue_update(Instance *p_instance, bool p_update_aabb, bool p_update_materials) {

	if (p_update_aabb)
		p_instance->update_aabb = true;
	if (p_update_materials)
		p_instance->update_materials = true;

	if (p_instance->update_item.in_list())
		return;

	_instance_update_list.add(&p_instance->update_item);
}

// from can be mesh, light,  area and portal so far.
RID VisualServerScene::instance_create() {

	Instance *instance = memnew(Instance);
	ERR_FAIL_COND_V(!instance, RID());

	RID instance_rid = instance_owner.make_rid(instance);
	instance->self = instance_rid;

	return instance_rid;
}

void VisualServerScene::instance_set_base(RID p_instance, RID p_base) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	Scenario *scenario = instance->scenario;

	if (instance->base_type != VS::INSTANCE_NONE) {
		//free anything related to that base

		VSG::storage->instance_remove_dependency(instance->base, instance);

		if (scenario && instance->octree_id) {
			scenario->octree.erase(instance->octree_id); //make dependencies generated by the octree go away
			instance->octree_id = 0;
		}

		switch (instance->base_type) {
			case VS::INSTANCE_LIGHT: {

				InstanceLightData *light = static_cast<InstanceLightData *>(instance->base_data);

				if (instance->scenario && light->D) {
					instance->scenario->directional_lights.erase(light->D);
					light->D = NULL;
				}
				VSG::scene_render->free(light->instance);
			} break;
			case VS::INSTANCE_REFLECTION_PROBE: {

				InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(instance->base_data);
				VSG::scene_render->free(reflection_probe->instance);
				if (reflection_probe->update_list.in_list()) {
					reflection_probe_render_list.remove(&reflection_probe->update_list);
				}
			} break;
			case VS::INSTANCE_GI_PROBE: {

				InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(instance->base_data);

				while (gi_probe->dynamic.updating_stage == GI_UPDATE_STAGE_LIGHTING) {
					//wait until bake is done if it's baking
					OS::get_singleton()->delay_usec(1);
				}
				if (gi_probe->update_element.in_list()) {
					gi_probe_update_list.remove(&gi_probe->update_element);
				}
				if (gi_probe->dynamic.probe_data.is_valid()) {
					VSG::storage->free(gi_probe->dynamic.probe_data);
				}

				VSG::scene_render->free(gi_probe->probe_instance);

			} break;
		}

		if (instance->base_data) {
			memdelete(instance->base_data);
			instance->base_data = NULL;
		}

		instance->blend_values.clear();

		for (int i = 0; i < instance->materials.size(); i++) {
			if (instance->materials[i].is_valid()) {
				VSG::storage->material_remove_instance_owner(instance->materials[i], instance);
			}
		}
		instance->materials.clear();
	}

	instance->base_type = VS::INSTANCE_NONE;
	instance->base = RID();

	if (p_base.is_valid()) {

		instance->base_type = VSG::storage->get_base_type(p_base);
		ERR_FAIL_COND(instance->base_type == VS::INSTANCE_NONE);

		switch (instance->base_type) {
			case VS::INSTANCE_LIGHT: {

				InstanceLightData *light = memnew(InstanceLightData);

				if (scenario && VSG::storage->light_get_type(p_base) == VS::LIGHT_DIRECTIONAL) {
					light->D = scenario->directional_lights.push_back(instance);
				}

				light->instance = VSG::scene_render->light_instance_create(p_base);

				instance->base_data = light;
			} break;
			case VS::INSTANCE_MESH:
			case VS::INSTANCE_MULTIMESH:
			case VS::INSTANCE_IMMEDIATE:
			case VS::INSTANCE_PARTICLES: {

				InstanceGeometryData *geom = memnew(InstanceGeometryData);
				instance->base_data = geom;
			} break;
			case VS::INSTANCE_REFLECTION_PROBE: {

				InstanceReflectionProbeData *reflection_probe = memnew(InstanceReflectionProbeData);
				reflection_probe->owner = instance;
				instance->base_data = reflection_probe;

				reflection_probe->instance = VSG::scene_render->reflection_probe_instance_create(p_base);
			} break;
			case VS::INSTANCE_GI_PROBE: {

				InstanceGIProbeData *gi_probe = memnew(InstanceGIProbeData);
				instance->base_data = gi_probe;
				gi_probe->owner = instance;

				if (scenario && !gi_probe->update_element.in_list()) {
					gi_probe_update_list.add(&gi_probe->update_element);
				}

				gi_probe->probe_instance = VSG::scene_render->gi_probe_instance_create();

			} break;
		}

		VSG::storage->instance_add_dependency(p_base, instance);

		instance->base = p_base;

		if (scenario)
			_instance_queue_update(instance, true, true);
	}
}
void VisualServerScene::instance_set_scenario(RID p_instance, RID p_scenario) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->scenario) {

		instance->scenario->instances.remove(&instance->scenario_item);

		if (instance->octree_id) {
			instance->scenario->octree.erase(instance->octree_id); //make dependencies generated by the octree go away
			instance->octree_id = 0;
		}

		switch (instance->base_type) {

			case VS::INSTANCE_LIGHT: {

				InstanceLightData *light = static_cast<InstanceLightData *>(instance->base_data);

				if (light->D) {
					instance->scenario->directional_lights.erase(light->D);
					light->D = NULL;
				}
			} break;
			case VS::INSTANCE_REFLECTION_PROBE: {

				InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(instance->base_data);
				VSG::scene_render->reflection_probe_release_atlas_index(reflection_probe->instance);
			} break;
			case VS::INSTANCE_GI_PROBE: {

				InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(instance->base_data);
				if (gi_probe->update_element.in_list()) {
					gi_probe_update_list.remove(&gi_probe->update_element);
				}
			} break;
		}

		instance->scenario = NULL;
	}

	if (p_scenario.is_valid()) {

		Scenario *scenario = scenario_owner.get(p_scenario);
		ERR_FAIL_COND(!scenario);

		instance->scenario = scenario;

		scenario->instances.add(&instance->scenario_item);

		switch (instance->base_type) {

			case VS::INSTANCE_LIGHT: {

				InstanceLightData *light = static_cast<InstanceLightData *>(instance->base_data);

				if (VSG::storage->light_get_type(instance->base) == VS::LIGHT_DIRECTIONAL) {
					light->D = scenario->directional_lights.push_back(instance);
				}
			} break;
			case VS::INSTANCE_GI_PROBE: {

				InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(instance->base_data);
				if (!gi_probe->update_element.in_list()) {
					gi_probe_update_list.add(&gi_probe->update_element);
				}
			} break;
		}

		_instance_queue_update(instance, true, true);
	}
}
void VisualServerScene::instance_set_layer_mask(RID p_instance, uint32_t p_mask) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	instance->layer_mask = p_mask;
}
void VisualServerScene::instance_set_transform(RID p_instance, const Transform &p_transform) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->transform == p_transform)
		return; //must be checked to avoid worst evil

	instance->transform = p_transform;
	_instance_queue_update(instance, true);
}
void VisualServerScene::instance_attach_object_instance_id(RID p_instance, ObjectID p_ID) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	instance->object_ID = p_ID;
}
void VisualServerScene::instance_set_blend_shape_weight(RID p_instance, int p_shape, float p_weight) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->update_item.in_list()) {
		_update_dirty_instance(instance);
	}

	ERR_FAIL_INDEX(p_shape, instance->blend_values.size());
	instance->blend_values[p_shape] = p_weight;
}

void VisualServerScene::instance_set_surface_material(RID p_instance, int p_surface, RID p_material) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->update_item.in_list()) {
		_update_dirty_instance(instance);
	}

	ERR_FAIL_INDEX(p_surface, instance->materials.size());

	if (instance->materials[p_surface].is_valid()) {
		VSG::storage->material_remove_instance_owner(instance->materials[p_surface], instance);
	}
	instance->materials[p_surface] = p_material;
	instance->base_material_changed();

	if (instance->materials[p_surface].is_valid()) {
		VSG::storage->material_add_instance_owner(instance->materials[p_surface], instance);
	}
}

void VisualServerScene::instance_set_visible(RID p_instance, bool p_visible) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->visible == p_visible)
		return;

	instance->visible = p_visible;

	switch (instance->base_type) {
		case VS::INSTANCE_LIGHT: {
			if (VSG::storage->light_get_type(instance->base) != VS::LIGHT_DIRECTIONAL && instance->octree_id && instance->scenario) {
				instance->scenario->octree.set_pairable(instance->octree_id, p_visible, 1 << VS::INSTANCE_LIGHT, p_visible ? VS::INSTANCE_GEOMETRY_MASK : 0);
			}

		} break;
		case VS::INSTANCE_REFLECTION_PROBE: {
			if (instance->octree_id && instance->scenario) {
				instance->scenario->octree.set_pairable(instance->octree_id, p_visible, 1 << VS::INSTANCE_REFLECTION_PROBE, p_visible ? VS::INSTANCE_GEOMETRY_MASK : 0);
			}

		} break;
		case VS::INSTANCE_GI_PROBE: {
			if (instance->octree_id && instance->scenario) {
				instance->scenario->octree.set_pairable(instance->octree_id, p_visible, 1 << VS::INSTANCE_GI_PROBE, p_visible ? (VS::INSTANCE_GEOMETRY_MASK | (1 << VS::INSTANCE_LIGHT)) : 0);
			}

		} break;
	}
}

void VisualServerScene::instance_attach_skeleton(RID p_instance, RID p_skeleton) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->skeleton == p_skeleton)
		return;

	if (instance->skeleton.is_valid()) {
		VSG::storage->instance_remove_skeleton(instance->skeleton, instance);
	}

	instance->skeleton = p_skeleton;

	if (instance->skeleton.is_valid()) {
		VSG::storage->instance_add_skeleton(instance->skeleton, instance);
	}

	_instance_queue_update(instance, true);
}

void VisualServerScene::instance_set_exterior(RID p_instance, bool p_enabled) {
}

void VisualServerScene::instance_set_extra_visibility_margin(RID p_instance, real_t p_margin) {
}

Vector<ObjectID> VisualServerScene::instances_cull_aabb(const Rect3 &p_aabb, RID p_scenario) const {

	Vector<ObjectID> instances;
	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario, instances);

	const_cast<VisualServerScene *>(this)->update_dirty_instances(); // check dirty instances before culling

	int culled = 0;
	Instance *cull[1024];
	culled = scenario->octree.cull_aabb(p_aabb, cull, 1024);

	for (int i = 0; i < culled; i++) {

		Instance *instance = cull[i];
		ERR_CONTINUE(!instance);
		if (instance->object_ID == 0)
			continue;

		instances.push_back(instance->object_ID);
	}

	return instances;
}
Vector<ObjectID> VisualServerScene::instances_cull_ray(const Vector3 &p_from, const Vector3 &p_to, RID p_scenario) const {

	Vector<ObjectID> instances;
	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario, instances);
	const_cast<VisualServerScene *>(this)->update_dirty_instances(); // check dirty instances before culling

	int culled = 0;
	Instance *cull[1024];
	culled = scenario->octree.cull_segment(p_from, p_to * 10000, cull, 1024);

	for (int i = 0; i < culled; i++) {
		Instance *instance = cull[i];
		ERR_CONTINUE(!instance);
		if (instance->object_ID == 0)
			continue;

		instances.push_back(instance->object_ID);
	}

	return instances;
}
Vector<ObjectID> VisualServerScene::instances_cull_convex(const Vector<Plane> &p_convex, RID p_scenario) const {

	Vector<ObjectID> instances;
	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario, instances);
	const_cast<VisualServerScene *>(this)->update_dirty_instances(); // check dirty instances before culling

	int culled = 0;
	Instance *cull[1024];

	culled = scenario->octree.cull_convex(p_convex, cull, 1024);

	for (int i = 0; i < culled; i++) {

		Instance *instance = cull[i];
		ERR_CONTINUE(!instance);
		if (instance->object_ID == 0)
			continue;

		instances.push_back(instance->object_ID);
	}

	return instances;
}

void VisualServerScene::instance_geometry_set_flag(RID p_instance, VS::InstanceFlags p_flags, bool p_enabled) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	switch (p_flags) {

		case VS::INSTANCE_FLAG_USE_BAKED_LIGHT: {

			instance->baked_light = p_enabled;

		} break;
	}
}
void VisualServerScene::instance_geometry_set_cast_shadows_setting(RID p_instance, VS::ShadowCastingSetting p_shadow_casting_setting) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	instance->cast_shadows = p_shadow_casting_setting;
	instance->base_material_changed(); // to actually compute if shadows are visible or not
}
void VisualServerScene::instance_geometry_set_material_override(RID p_instance, RID p_material) {

	Instance *instance = instance_owner.get(p_instance);
	ERR_FAIL_COND(!instance);

	if (instance->material_override.is_valid()) {
		VSG::storage->material_remove_instance_owner(instance->material_override, instance);
	}
	instance->material_override = p_material;
	instance->base_material_changed();

	if (instance->material_override.is_valid()) {
		VSG::storage->material_add_instance_owner(instance->material_override, instance);
	}
}

void VisualServerScene::instance_geometry_set_draw_range(RID p_instance, float p_min, float p_max, float p_min_margin, float p_max_margin) {
}
void VisualServerScene::instance_geometry_set_as_instance_lod(RID p_instance, RID p_as_lod_of_instance) {
}

void VisualServerScene::_update_instance(Instance *p_instance) {

	p_instance->version++;

	if (p_instance->base_type == VS::INSTANCE_LIGHT) {

		InstanceLightData *light = static_cast<InstanceLightData *>(p_instance->base_data);

		VSG::scene_render->light_instance_set_transform(light->instance, p_instance->transform);
		light->shadow_dirty = true;
	}

	if (p_instance->base_type == VS::INSTANCE_REFLECTION_PROBE) {

		InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(p_instance->base_data);

		VSG::scene_render->reflection_probe_instance_set_transform(reflection_probe->instance, p_instance->transform);
		reflection_probe->reflection_dirty = true;
	}

	if (p_instance->base_type == VS::INSTANCE_PARTICLES) {

		VSG::storage->particles_set_emission_transform(p_instance->base, p_instance->transform);
	}

	if (p_instance->aabb.has_no_surface()) {
		return;
	}

	if ((1 << p_instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) {

		InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(p_instance->base_data);
		//make sure lights are updated if it casts shadow

		if (geom->can_cast_shadows) {
			for (List<Instance *>::Element *E = geom->lighting.front(); E; E = E->next()) {
				InstanceLightData *light = static_cast<InstanceLightData *>(E->get()->base_data);
				light->shadow_dirty = true;
			}
		}
	}

	p_instance->mirror = p_instance->transform.basis.determinant() < 0.0;

	Rect3 new_aabb;

	new_aabb = p_instance->transform.xform(p_instance->aabb);

	p_instance->transformed_aabb = new_aabb;

	if (!p_instance->scenario) {

		return;
	}

	if (p_instance->octree_id == 0) {

		uint32_t base_type = 1 << p_instance->base_type;
		uint32_t pairable_mask = 0;
		bool pairable = false;

		if (p_instance->base_type == VS::INSTANCE_LIGHT || p_instance->base_type == VS::INSTANCE_REFLECTION_PROBE) {

			pairable_mask = p_instance->visible ? VS::INSTANCE_GEOMETRY_MASK : 0;
			pairable = true;
		}

		if (p_instance->base_type == VS::INSTANCE_GI_PROBE) {
			//lights and geometries
			pairable_mask = p_instance->visible ? VS::INSTANCE_GEOMETRY_MASK | (1 << VS::INSTANCE_LIGHT) : 0;
			pairable = true;
		}

		// not inside octree
		p_instance->octree_id = p_instance->scenario->octree.create(p_instance, new_aabb, 0, pairable, base_type, pairable_mask);

	} else {

		/*
		if (new_aabb==p_instance->data.transformed_aabb)
			return;
		*/

		p_instance->scenario->octree.move(p_instance->octree_id, new_aabb);
	}
}

void VisualServerScene::_update_instance_aabb(Instance *p_instance) {

	Rect3 new_aabb;

	ERR_FAIL_COND(p_instance->base_type != VS::INSTANCE_NONE && !p_instance->base.is_valid());

	switch (p_instance->base_type) {
		case VisualServer::INSTANCE_NONE: {

			// do nothing
		} break;
		case VisualServer::INSTANCE_MESH: {

			new_aabb = VSG::storage->mesh_get_aabb(p_instance->base, p_instance->skeleton);

		} break;

		case VisualServer::INSTANCE_MULTIMESH: {

			new_aabb = VSG::storage->multimesh_get_aabb(p_instance->base);

		} break;
		case VisualServer::INSTANCE_IMMEDIATE: {

			new_aabb = VSG::storage->immediate_get_aabb(p_instance->base);

		} break;
		case VisualServer::INSTANCE_PARTICLES: {

			new_aabb = VSG::storage->particles_get_aabb(p_instance->base);

		} break;
		case VisualServer::INSTANCE_LIGHT: {

			new_aabb = VSG::storage->light_get_aabb(p_instance->base);

		} break;
		case VisualServer::INSTANCE_REFLECTION_PROBE: {

			new_aabb = VSG::storage->reflection_probe_get_aabb(p_instance->base);

		} break;
		case VisualServer::INSTANCE_GI_PROBE: {

			new_aabb = VSG::storage->gi_probe_get_bounds(p_instance->base);

		} break;

		default: {}
	}

	if (p_instance->extra_margin)
		new_aabb.grow_by(p_instance->extra_margin);

	p_instance->aabb = new_aabb;
}

void VisualServerScene::_light_instance_update_shadow(Instance *p_instance, const Transform p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, RID p_shadow_atlas, Scenario *p_scenario) {

	InstanceLightData *light = static_cast<InstanceLightData *>(p_instance->base_data);

	switch (VSG::storage->light_get_type(p_instance->base)) {

		case VS::LIGHT_DIRECTIONAL: {

			float max_distance = p_cam_projection.get_z_far();
			float shadow_max = VSG::storage->light_get_param(p_instance->base, VS::LIGHT_PARAM_SHADOW_MAX_DISTANCE);
			if (shadow_max > 0 && !p_cam_orthogonal) { //its impractical (and leads to unwanted behaviors) to set max distance in orthogonal camera
				max_distance = MIN(shadow_max, max_distance);
			}
			max_distance = MAX(max_distance, p_cam_projection.get_z_near() + 0.001);
			float min_distance = MIN(p_cam_projection.get_z_near(), max_distance);

			VS::LightDirectionalShadowDepthRangeMode depth_range_mode = VSG::storage->light_directional_get_shadow_depth_range_mode(p_instance->base);

			if (depth_range_mode == VS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_OPTIMIZED) {
				//optimize min/max
				Vector<Plane> planes = p_cam_projection.get_projection_planes(p_cam_transform);
				int cull_count = p_scenario->octree.cull_convex(planes, instance_shadow_cull_result, MAX_INSTANCE_CULL, VS::INSTANCE_GEOMETRY_MASK);
				Plane base(p_cam_transform.origin, -p_cam_transform.basis.get_axis(2));
				//check distance max and min

				bool found_items = false;
				float z_max = -1e20;
				float z_min = 1e20;

				for (int i = 0; i < cull_count; i++) {

					Instance *instance = instance_shadow_cull_result[i];
					if (!instance->visible || !((1 << instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData *>(instance->base_data)->can_cast_shadows) {
						continue;
					}

					float max, min;
					instance->transformed_aabb.project_range_in_plane(base, min, max);

					if (max > z_max) {
						z_max = max;
					}

					if (min < z_min) {
						z_min = min;
					}

					found_items = true;
				}

				if (found_items) {
					min_distance = MAX(min_distance, z_min);
					max_distance = MIN(max_distance, z_max);
				}
			}

			float range = max_distance - min_distance;

			int splits = 0;
			switch (VSG::storage->light_directional_get_shadow_mode(p_instance->base)) {
				case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL: splits = 1; break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS: splits = 2; break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: splits = 4; break;
			}

			float distances[5];

			distances[0] = min_distance;
			for (int i = 0; i < splits; i++) {
				distances[i + 1] = min_distance + VSG::storage->light_get_param(p_instance->base, VS::LightParam(VS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET + i)) * range;
			};

			distances[splits] = max_distance;

			float texture_size = VSG::scene_render->get_directional_light_shadow_size(light->instance);

			bool overlap = VSG::storage->light_directional_get_blend_splits(p_instance->base);

			float first_radius = 0.0;

			for (int i = 0; i < splits; i++) {

				// setup a camera matrix for that range!
				CameraMatrix camera_matrix;

				float aspect = p_cam_projection.get_aspect();

				if (p_cam_orthogonal) {

					float w, h;
					p_cam_projection.get_viewport_size(w, h);
					camera_matrix.set_orthogonal(w, aspect, distances[(i == 0 || !overlap) ? i : i - 1], distances[i + 1], false);
				} else {

					float fov = p_cam_projection.get_fov();
					camera_matrix.set_perspective(fov, aspect, distances[(i == 0 || !overlap) ? i : i - 1], distances[i + 1], false);
				}

				//obtain the frustum endpoints

				Vector3 endpoints[8]; // frustum plane endpoints
				bool res = camera_matrix.get_endpoints(p_cam_transform, endpoints);
				ERR_CONTINUE(!res);

				// obtain the light frustm ranges (given endpoints)

				Transform transform = p_instance->transform.orthonormalized(); //discard scale and stabilize light

				Vector3 x_vec = transform.basis.get_axis(Vector3::AXIS_X).normalized();
				Vector3 y_vec = transform.basis.get_axis(Vector3::AXIS_Y).normalized();
				Vector3 z_vec = transform.basis.get_axis(Vector3::AXIS_Z).normalized();
				//z_vec points agsint the camera, like in default opengl

				float x_min = 0.f, x_max = 0.f;
				float y_min = 0.f, y_max = 0.f;
				float z_min = 0.f, z_max = 0.f;

				float x_min_cam = 0.f, x_max_cam = 0.f;
				float y_min_cam = 0.f, y_max_cam = 0.f;
				float z_min_cam = 0.f, z_max_cam = 0.f;

				float bias_scale = 1.0;

				//used for culling

				for (int j = 0; j < 8; j++) {

					float d_x = x_vec.dot(endpoints[j]);
					float d_y = y_vec.dot(endpoints[j]);
					float d_z = z_vec.dot(endpoints[j]);

					if (j == 0 || d_x < x_min)
						x_min = d_x;
					if (j == 0 || d_x > x_max)
						x_max = d_x;

					if (j == 0 || d_y < y_min)
						y_min = d_y;
					if (j == 0 || d_y > y_max)
						y_max = d_y;

					if (j == 0 || d_z < z_min)
						z_min = d_z;
					if (j == 0 || d_z > z_max)
						z_max = d_z;
				}

				{
					//camera viewport stuff

					Vector3 center;

					for (int j = 0; j < 8; j++) {

						center += endpoints[j];
					}
					center /= 8.0;

					//center=x_vec*(x_max-x_min)*0.5 + y_vec*(y_max-y_min)*0.5 + z_vec*(z_max-z_min)*0.5;

					float radius = 0;

					for (int j = 0; j < 8; j++) {

						float d = center.distance_to(endpoints[j]);
						if (d > radius)
							radius = d;
					}

					radius *= texture_size / (texture_size - 2.0); //add a texel by each side

					if (i == 0) {
						first_radius = radius;
					} else {
						bias_scale = radius / first_radius;
					}

					x_max_cam = x_vec.dot(center) + radius;
					x_min_cam = x_vec.dot(center) - radius;
					y_max_cam = y_vec.dot(center) + radius;
					y_min_cam = y_vec.dot(center) - radius;
					z_max_cam = z_vec.dot(center) + radius;
					z_min_cam = z_vec.dot(center) - radius;

					if (depth_range_mode == VS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE) {
						//this trick here is what stabilizes the shadow (make potential jaggies to not move)
						//at the cost of some wasted resolution. Still the quality increase is very well worth it

						float unit = radius * 2.0 / texture_size;

						x_max_cam = Math::stepify(x_max_cam, unit);
						x_min_cam = Math::stepify(x_min_cam, unit);
						y_max_cam = Math::stepify(y_max_cam, unit);
						y_min_cam = Math::stepify(y_min_cam, unit);
					}
				}

				//now that we now all ranges, we can proceed to make the light frustum planes, for culling octree

				Vector<Plane> light_frustum_planes;
				light_frustum_planes.resize(6);

				//right/left
				light_frustum_planes[0] = Plane(x_vec, x_max);
				light_frustum_planes[1] = Plane(-x_vec, -x_min);
				//top/bottom
				light_frustum_planes[2] = Plane(y_vec, y_max);
				light_frustum_planes[3] = Plane(-y_vec, -y_min);
				//near/far
				light_frustum_planes[4] = Plane(z_vec, z_max + 1e6);
				light_frustum_planes[5] = Plane(-z_vec, -z_min); // z_min is ok, since casters further than far-light plane are not needed

				int cull_count = p_scenario->octree.cull_convex(light_frustum_planes, instance_shadow_cull_result, MAX_INSTANCE_CULL, VS::INSTANCE_GEOMETRY_MASK);

				// a pre pass will need to be needed to determine the actual z-near to be used

				Plane near_plane(p_instance->transform.origin, -p_instance->transform.basis.get_axis(2));

				for (int j = 0; j < cull_count; j++) {

					float min, max;
					Instance *instance = instance_shadow_cull_result[j];
					if (!instance->visible || !((1 << instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData *>(instance->base_data)->can_cast_shadows) {
						cull_count--;
						SWAP(instance_shadow_cull_result[j], instance_shadow_cull_result[cull_count]);
						j--;
						continue;
					}

					instance->transformed_aabb.project_range_in_plane(Plane(z_vec, 0), min, max);
					instance->depth = near_plane.distance_to(instance->transform.origin);
					instance->depth_layer = 0;
					if (max > z_max)
						z_max = max;
				}

				{

					CameraMatrix ortho_camera;
					real_t half_x = (x_max_cam - x_min_cam) * 0.5;
					real_t half_y = (y_max_cam - y_min_cam) * 0.5;

					ortho_camera.set_orthogonal(-half_x, half_x, -half_y, half_y, 0, (z_max - z_min_cam));

					Transform ortho_transform;
					ortho_transform.basis = transform.basis;
					ortho_transform.origin = x_vec * (x_min_cam + half_x) + y_vec * (y_min_cam + half_y) + z_vec * z_max;

					VSG::scene_render->light_instance_set_shadow_transform(light->instance, ortho_camera, ortho_transform, 0, distances[i + 1], i, bias_scale);
				}

				VSG::scene_render->render_shadow(light->instance, p_shadow_atlas, i, (RasterizerScene::InstanceBase **)instance_shadow_cull_result, cull_count);
			}

		} break;
		case VS::LIGHT_OMNI: {

			VS::LightOmniShadowMode shadow_mode = VSG::storage->light_omni_get_shadow_mode(p_instance->base);

			switch (shadow_mode) {
				case VS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID: {

					for (int i = 0; i < 2; i++) {

						//using this one ensures that raster deferred will have it

						float radius = VSG::storage->light_get_param(p_instance->base, VS::LIGHT_PARAM_RANGE);

						float z = i == 0 ? -1 : 1;
						Vector<Plane> planes;
						planes.resize(5);
						planes[0] = p_instance->transform.xform(Plane(Vector3(0, 0, z), radius));
						planes[1] = p_instance->transform.xform(Plane(Vector3(1, 0, z).normalized(), radius));
						planes[2] = p_instance->transform.xform(Plane(Vector3(-1, 0, z).normalized(), radius));
						planes[3] = p_instance->transform.xform(Plane(Vector3(0, 1, z).normalized(), radius));
						planes[4] = p_instance->transform.xform(Plane(Vector3(0, -1, z).normalized(), radius));

						int cull_count = p_scenario->octree.cull_convex(planes, instance_shadow_cull_result, MAX_INSTANCE_CULL, VS::INSTANCE_GEOMETRY_MASK);
						Plane near_plane(p_instance->transform.origin, p_instance->transform.basis.get_axis(2) * z);

						for (int j = 0; j < cull_count; j++) {

							Instance *instance = instance_shadow_cull_result[j];
							if (!instance->visible || !((1 << instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData *>(instance->base_data)->can_cast_shadows) {
								cull_count--;
								SWAP(instance_shadow_cull_result[j], instance_shadow_cull_result[cull_count]);
								j--;
							} else {
								instance->depth = near_plane.distance_to(instance->transform.origin);
								instance->depth_layer = 0;
							}
						}

						VSG::scene_render->light_instance_set_shadow_transform(light->instance, CameraMatrix(), p_instance->transform, radius, 0, i);
						VSG::scene_render->render_shadow(light->instance, p_shadow_atlas, i, (RasterizerScene::InstanceBase **)instance_shadow_cull_result, cull_count);
					}
				} break;
				case VS::LIGHT_OMNI_SHADOW_CUBE: {

					float radius = VSG::storage->light_get_param(p_instance->base, VS::LIGHT_PARAM_RANGE);
					CameraMatrix cm;
					cm.set_perspective(90, 1, 0.01, radius);

					for (int i = 0; i < 6; i++) {

						//using this one ensures that raster deferred will have it

						static const Vector3 view_normals[6] = {
							Vector3(-1, 0, 0),
							Vector3(+1, 0, 0),
							Vector3(0, -1, 0),
							Vector3(0, +1, 0),
							Vector3(0, 0, -1),
							Vector3(0, 0, +1)
						};
						static const Vector3 view_up[6] = {
							Vector3(0, -1, 0),
							Vector3(0, -1, 0),
							Vector3(0, 0, -1),
							Vector3(0, 0, +1),
							Vector3(0, -1, 0),
							Vector3(0, -1, 0)
						};

						Transform xform = p_instance->transform * Transform().looking_at(view_normals[i], view_up[i]);

						Vector<Plane> planes = cm.get_projection_planes(xform);

						int cull_count = p_scenario->octree.cull_convex(planes, instance_shadow_cull_result, MAX_INSTANCE_CULL, VS::INSTANCE_GEOMETRY_MASK);

						Plane near_plane(xform.origin, -xform.basis.get_axis(2));
						for (int j = 0; j < cull_count; j++) {

							Instance *instance = instance_shadow_cull_result[j];
							if (!instance->visible || !((1 << instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData *>(instance->base_data)->can_cast_shadows) {
								cull_count--;
								SWAP(instance_shadow_cull_result[j], instance_shadow_cull_result[cull_count]);
								j--;
							} else {
								instance->depth = near_plane.distance_to(instance->transform.origin);
								instance->depth_layer = 0;
							}
						}

						VSG::scene_render->light_instance_set_shadow_transform(light->instance, cm, xform, radius, 0, i);
						VSG::scene_render->render_shadow(light->instance, p_shadow_atlas, i, (RasterizerScene::InstanceBase **)instance_shadow_cull_result, cull_count);
					}

					//restore the regular DP matrix
					VSG::scene_render->light_instance_set_shadow_transform(light->instance, CameraMatrix(), p_instance->transform, radius, 0, 0);

				} break;
			}

		} break;
		case VS::LIGHT_SPOT: {

			float radius = VSG::storage->light_get_param(p_instance->base, VS::LIGHT_PARAM_RANGE);
			float angle = VSG::storage->light_get_param(p_instance->base, VS::LIGHT_PARAM_SPOT_ANGLE);

			CameraMatrix cm;
			cm.set_perspective(angle * 2.0, 1.0, 0.01, radius);

			Vector<Plane> planes = cm.get_projection_planes(p_instance->transform);
			int cull_count = p_scenario->octree.cull_convex(planes, instance_shadow_cull_result, MAX_INSTANCE_CULL, VS::INSTANCE_GEOMETRY_MASK);

			Plane near_plane(p_instance->transform.origin, -p_instance->transform.basis.get_axis(2));
			for (int j = 0; j < cull_count; j++) {

				Instance *instance = instance_shadow_cull_result[j];
				if (!instance->visible || !((1 << instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData *>(instance->base_data)->can_cast_shadows) {
					cull_count--;
					SWAP(instance_shadow_cull_result[j], instance_shadow_cull_result[cull_count]);
					j--;
				} else {
					instance->depth = near_plane.distance_to(instance->transform.origin);
					instance->depth_layer = 0;
				}
			}

			VSG::scene_render->light_instance_set_shadow_transform(light->instance, cm, p_instance->transform, radius, 0, 0);
			VSG::scene_render->render_shadow(light->instance, p_shadow_atlas, 0, (RasterizerScene::InstanceBase **)instance_shadow_cull_result, cull_count);

		} break;
	}
}

void VisualServerScene::render_camera(RID p_camera, RID p_scenario, Size2 p_viewport_size, RID p_shadow_atlas) {
	// render to mono camera

	Camera *camera = camera_owner.getornull(p_camera);
	ERR_FAIL_COND(!camera);

	/* STEP 1 - SETUP CAMERA */
	CameraMatrix camera_matrix;
	bool ortho = false;

	switch (camera->type) {
		case Camera::ORTHOGONAL: {

			camera_matrix.set_orthogonal(
					camera->size,
					p_viewport_size.width / (float)p_viewport_size.height,
					camera->znear,
					camera->zfar,
					camera->vaspect

					);
			ortho = true;
		} break;
		case Camera::PERSPECTIVE: {

			camera_matrix.set_perspective(
					camera->fov,
					p_viewport_size.width / (float)p_viewport_size.height,
					camera->znear,
					camera->zfar,
					camera->vaspect

					);
			ortho = false;

		} break;
	}

	_render_scene(camera->transform, camera_matrix, ortho, camera->env, camera->visible_layers, p_scenario, p_shadow_atlas, RID(), -1);
}

void VisualServerScene::render_camera(Ref<ARVRInterface> &p_interface, ARVRInterface::Eyes p_eye, RID p_camera, RID p_scenario, Size2 p_viewport_size, RID p_shadow_atlas) {
	// render for AR/VR interface

	Camera *camera = camera_owner.getornull(p_camera);
	ERR_FAIL_COND(!camera);

	/* SETUP CAMERA, we are ignoring type and FOV here */
	bool ortho = false;
	float aspect = p_viewport_size.width / (float)p_viewport_size.height;
	CameraMatrix camera_matrix = p_interface->get_projection_for_eye(p_eye, aspect, camera->znear, camera->zfar);

	// We also ignore our camera position, it will have been positioned with a slightly old tracking position.
	// Instead we take our origin point and have our ar/vr interface add fresh tracking data! Whoohoo!
	Transform world_origin = ARVRServer::get_singleton()->get_world_origin();
	Transform cam_transform = p_interface->get_transform_for_eye(p_eye, world_origin);

	_render_scene(cam_transform, camera_matrix, ortho, camera->env, camera->visible_layers, p_scenario, p_shadow_atlas, RID(), -1);
};

void VisualServerScene::_render_scene(const Transform p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, RID p_force_environment, uint32_t p_visible_layers, RID p_scenario, RID p_shadow_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	Scenario *scenario = scenario_owner.getornull(p_scenario);

	render_pass++;
	uint32_t camera_layer_mask = p_visible_layers;

	VSG::scene_render->set_scene_pass(render_pass);

	//rasterizer->set_camera(camera->transform, camera_matrix,ortho);

	Vector<Plane> planes = p_cam_projection.get_projection_planes(p_cam_transform);

	Plane near_plane(p_cam_transform.origin, -p_cam_transform.basis.get_axis(2).normalized());
	float z_far = p_cam_projection.get_z_far();

	/* STEP 2 - CULL */
	int cull_count = scenario->octree.cull_convex(planes, instance_cull_result, MAX_INSTANCE_CULL);
	light_cull_count = 0;

	reflection_probe_cull_count = 0;

	//light_samplers_culled=0;

	/*	print_line("OT: "+rtos( (OS::get_singleton()->get_ticks_usec()-t)/1000.0));
	print_line("OTO: "+itos(p_scenario->octree.get_octant_count()));
	//print_line("OTE: "+itos(p_scenario->octree.get_elem_count()));
	print_line("OTP: "+itos(p_scenario->octree.get_pair_count()));
*/

	/* STEP 3 - PROCESS PORTALS, VALIDATE ROOMS */
	//removed, will replace with culling

	/* STEP 4 - REMOVE FURTHER CULLED OBJECTS, ADD LIGHTS */

	for (int i = 0; i < cull_count; i++) {

		Instance *ins = instance_cull_result[i];

		bool keep = false;

		if ((camera_layer_mask & ins->layer_mask) == 0) {

			//failure
		} else if (ins->base_type == VS::INSTANCE_LIGHT && ins->visible) {

			if (ins->visible && light_cull_count < MAX_LIGHTS_CULLED) {

				InstanceLightData *light = static_cast<InstanceLightData *>(ins->base_data);

				if (!light->geometries.empty()) {
					//do not add this light if no geometry is affected by it..
					light_cull_result[light_cull_count] = ins;
					light_instance_cull_result[light_cull_count] = light->instance;
					if (p_shadow_atlas.is_valid() && VSG::storage->light_has_shadow(ins->base)) {
						VSG::scene_render->light_instance_mark_visible(light->instance); //mark it visible for shadow allocation later
					}

					light_cull_count++;
				}
			}
		} else if (ins->base_type == VS::INSTANCE_REFLECTION_PROBE && ins->visible) {

			if (ins->visible && reflection_probe_cull_count < MAX_REFLECTION_PROBES_CULLED) {

				InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(ins->base_data);

				if (p_reflection_probe != reflection_probe->instance) {
					//avoid entering The Matrix

					if (!reflection_probe->geometries.empty()) {
						//do not add this light if no geometry is affected by it..

						if (reflection_probe->reflection_dirty || VSG::scene_render->reflection_probe_instance_needs_redraw(reflection_probe->instance)) {
							if (!reflection_probe->update_list.in_list()) {
								reflection_probe->render_step = 0;
								reflection_probe_render_list.add_last(&reflection_probe->update_list);
							}

							reflection_probe->reflection_dirty = false;
						}

						if (VSG::scene_render->reflection_probe_instance_has_reflection(reflection_probe->instance)) {
							reflection_probe_instance_cull_result[reflection_probe_cull_count] = reflection_probe->instance;
							reflection_probe_cull_count++;
						}
					}
				}
			}

		} else if (ins->base_type == VS::INSTANCE_GI_PROBE && ins->visible) {

			InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(ins->base_data);
			if (!gi_probe->update_element.in_list()) {
				gi_probe_update_list.add(&gi_probe->update_element);
			}

		} else if (((1 << ins->base_type) & VS::INSTANCE_GEOMETRY_MASK) && ins->visible && ins->cast_shadows != VS::SHADOW_CASTING_SETTING_SHADOWS_ONLY) {

			keep = true;

			InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(ins->base_data);

			if (ins->base_type == VS::INSTANCE_PARTICLES) {
				//particles visible? process them
				VSG::storage->particles_request_process(ins->base);
				//particles visible? request redraw
				VisualServerRaster::redraw_request();
			}

			if (geom->lighting_dirty) {
				int l = 0;
				//only called when lights AABB enter/exit this geometry
				ins->light_instances.resize(geom->lighting.size());

				for (List<Instance *>::Element *E = geom->lighting.front(); E; E = E->next()) {

					InstanceLightData *light = static_cast<InstanceLightData *>(E->get()->base_data);

					ins->light_instances[l++] = light->instance;
				}

				geom->lighting_dirty = false;
			}

			if (geom->reflection_dirty) {
				int l = 0;
				//only called when reflection probe AABB enter/exit this geometry
				ins->reflection_probe_instances.resize(geom->reflection_probes.size());

				for (List<Instance *>::Element *E = geom->reflection_probes.front(); E; E = E->next()) {

					InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(E->get()->base_data);

					ins->reflection_probe_instances[l++] = reflection_probe->instance;
				}

				geom->reflection_dirty = false;
			}

			if (geom->gi_probes_dirty) {
				int l = 0;
				//only called when reflection probe AABB enter/exit this geometry
				ins->gi_probe_instances.resize(geom->gi_probes.size());

				for (List<Instance *>::Element *E = geom->gi_probes.front(); E; E = E->next()) {

					InstanceGIProbeData *gi_probe = static_cast<InstanceGIProbeData *>(E->get()->base_data);

					ins->gi_probe_instances[l++] = gi_probe->probe_instance;
				}

				geom->gi_probes_dirty = false;
			}

			ins->depth = near_plane.distance_to(ins->transform.origin);
			ins->depth_layer = CLAMP(int(ins->depth * 16 / z_far), 0, 15);
		}

		if (!keep) {
			// remove, no reason to keep
			cull_count--;
			SWAP(instance_cull_result[i], instance_cull_result[cull_count]);
			i--;
			ins->last_render_pass = 0; // make invalid
		} else {

			ins->last_render_pass = render_pass;
		}
	}

	/* STEP 5 - PROCESS LIGHTS */

	RID *directional_light_ptr = &light_instance_cull_result[light_cull_count];
	int directional_light_count = 0;

	// directional lights
	{

		Instance **lights_with_shadow = (Instance **)alloca(sizeof(Instance *) * scenario->directional_lights.size());
		int directional_shadow_count = 0;

		for (List<Instance *>::Element *E = scenario->directional_lights.front(); E; E = E->next()) {

			if (light_cull_count + directional_light_count >= MAX_LIGHTS_CULLED) {
				break;
			}

			if (!E->get()->visible)
				continue;

			InstanceLightData *light = static_cast<InstanceLightData *>(E->get()->base_data);

			//check shadow..

			if (light) {
				if (p_shadow_atlas.is_valid() && VSG::storage->light_has_shadow(E->get()->base)) {
					lights_with_shadow[directional_shadow_count++] = E->get();
				}
				//add to list
				directional_light_ptr[directional_light_count++] = light->instance;
			}
		}

		VSG::scene_render->set_directional_shadow_count(directional_shadow_count);

		for (int i = 0; i < directional_shadow_count; i++) {

			_light_instance_update_shadow(lights_with_shadow[i], p_cam_transform, p_cam_projection, p_cam_orthogonal, p_shadow_atlas, scenario);
		}
	}

	{ //setup shadow maps

		//SortArray<Instance*,_InstanceLightsort> sorter;
		//sorter.sort(light_cull_result,light_cull_count);
		for (int i = 0; i < light_cull_count; i++) {

			Instance *ins = light_cull_result[i];

			if (!p_shadow_atlas.is_valid() || !VSG::storage->light_has_shadow(ins->base))
				continue;

			InstanceLightData *light = static_cast<InstanceLightData *>(ins->base_data);

			float coverage = 0.f;

			{ //compute coverage

				Transform cam_xf = p_cam_transform;
				float zn = p_cam_projection.get_z_near();
				Plane p(cam_xf.origin + cam_xf.basis.get_axis(2) * -zn, -cam_xf.basis.get_axis(2)); //camera near plane

				float vp_w, vp_h; //near plane size in screen coordinates
				p_cam_projection.get_viewport_size(vp_w, vp_h);

				switch (VSG::storage->light_get_type(ins->base)) {

					case VS::LIGHT_OMNI: {

						float radius = VSG::storage->light_get_param(ins->base, VS::LIGHT_PARAM_RANGE);

						//get two points parallel to near plane
						Vector3 points[2] = {
							ins->transform.origin,
							ins->transform.origin + cam_xf.basis.get_axis(0) * radius
						};

						if (!p_cam_orthogonal) {
							//if using perspetive, map them to near plane
							for (int j = 0; j < 2; j++) {
								if (p.distance_to(points[j]) < 0) {
									points[j].z = -zn; //small hack to keep size constant when hitting the screen
								}

								p.intersects_segment(cam_xf.origin, points[j], &points[j]); //map to plane
							}
						}

						float screen_diameter = points[0].distance_to(points[1]) * 2;
						coverage = screen_diameter / (vp_w + vp_h);
					} break;
					case VS::LIGHT_SPOT: {

						float radius = VSG::storage->light_get_param(ins->base, VS::LIGHT_PARAM_RANGE);
						float angle = VSG::storage->light_get_param(ins->base, VS::LIGHT_PARAM_SPOT_ANGLE);

						float w = radius * Math::sin(Math::deg2rad(angle));
						float d = radius * Math::cos(Math::deg2rad(angle));

						Vector3 base = ins->transform.origin - ins->transform.basis.get_axis(2).normalized() * d;

						Vector3 points[2] = {
							base,
							base + cam_xf.basis.get_axis(0) * w
						};

						if (!p_cam_orthogonal) {
							//if using perspetive, map them to near plane
							for (int j = 0; j < 2; j++) {
								if (p.distance_to(points[j]) < 0) {
									points[j].z = -zn; //small hack to keep size constant when hitting the screen
								}

								p.intersects_segment(cam_xf.origin, points[j], &points[j]); //map to plane
							}
						}

						float screen_diameter = points[0].distance_to(points[1]) * 2;
						coverage = screen_diameter / (vp_w + vp_h);

					} break;
					default: {
						ERR_PRINT("Invalid Light Type");
					}
				}
			}

			if (light->shadow_dirty) {
				light->last_version++;
				light->shadow_dirty = false;
			}

			bool redraw = VSG::scene_render->shadow_atlas_update_light(p_shadow_atlas, light->instance, coverage, light->last_version);

			if (redraw) {
				//must redraw!
				_light_instance_update_shadow(ins, p_cam_transform, p_cam_projection, p_cam_orthogonal, p_shadow_atlas, scenario);
			}
		}
	}

	/* ENVIRONMENT */

	RID environment;
	if (p_force_environment.is_valid()) //camera has more environment priority
		environment = p_force_environment;
	else if (scenario->environment.is_valid())
		environment = scenario->environment;
	else
		environment = scenario->fallback_environment;

	/* STEP 6 - PROCESS GEOMETRY AND DRAW SCENE*/

	VSG::scene_render->render_scene(p_cam_transform, p_cam_projection, p_cam_orthogonal, (RasterizerScene::InstanceBase **)instance_cull_result, cull_count, light_instance_cull_result, light_cull_count + directional_light_count, reflection_probe_instance_cull_result, reflection_probe_cull_count, environment, p_shadow_atlas, scenario->reflection_atlas, p_reflection_probe, p_reflection_probe_pass);
}

void VisualServerScene::render_empty_scene(RID p_scenario, RID p_shadow_atlas) {

	Scenario *scenario = scenario_owner.getornull(p_scenario);

	RID environment;
	if (scenario->environment.is_valid())
		environment = scenario->environment;
	else
		environment = scenario->fallback_environment;
	VSG::scene_render->render_scene(Transform(), CameraMatrix(), true, NULL, 0, NULL, 0, NULL, 0, environment, p_shadow_atlas, scenario->reflection_atlas, RID(), 0);
}

bool VisualServerScene::_render_reflection_probe_step(Instance *p_instance, int p_step) {

	InstanceReflectionProbeData *reflection_probe = static_cast<InstanceReflectionProbeData *>(p_instance->base_data);
	Scenario *scenario = p_instance->scenario;
	ERR_FAIL_COND_V(!scenario, true);

	if (p_step == 0) {

		if (!VSG::scene_render->reflection_probe_instance_begin_render(reflection_probe->instance, scenario->reflection_atlas)) {
			return true; //sorry, all full :(
		}
	}

	if (p_step >= 0 && p_step < 6) {

		static const Vector3 view_normals[6] = {
			Vector3(-1, 0, 0),
			Vector3(+1, 0, 0),
			Vector3(0, -1, 0),
			Vector3(0, +1, 0),
			Vector3(0, 0, -1),
			Vector3(0, 0, +1)
		};

		Vector3 extents = VSG::storage->reflection_probe_get_extents(p_instance->base);
		Vector3 origin_offset = VSG::storage->reflection_probe_get_origin_offset(p_instance->base);
		float max_distance = VSG::storage->reflection_probe_get_origin_max_distance(p_instance->base);

		Vector3 edge = view_normals[p_step] * extents;
		float distance = ABS(view_normals[p_step].dot(edge) - view_normals[p_step].dot(origin_offset)); //distance from origin offset to actual view distance limit

		max_distance = MAX(max_distance, distance);

		//render cubemap side
		CameraMatrix cm;
		cm.set_perspective(90, 1, 0.01, max_distance);

		static const Vector3 view_up[6] = {
			Vector3(0, -1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, -1),
			Vector3(0, 0, +1),
			Vector3(0, -1, 0),
			Vector3(0, -1, 0)
		};

		Transform local_view;
		local_view.set_look_at(origin_offset, origin_offset + view_normals[p_step], view_up[p_step]);

		Transform xform = p_instance->transform * local_view;

		RID shadow_atlas;

		if (VSG::storage->reflection_probe_renders_shadows(p_instance->base)) {

			shadow_atlas = scenario->reflection_probe_shadow_atlas;
		}

		_render_scene(xform, cm, false, RID(), VSG::storage->reflection_probe_get_cull_mask(p_instance->base), p_instance->scenario->self, shadow_atlas, reflection_probe->instance, p_step);

	} else {
		//do roughness postprocess step until it belives it's done
		return VSG::scene_render->reflection_probe_instance_postprocess_step(reflection_probe->instance);
	}

	return false;
}

void VisualServerScene::_gi_probe_fill_local_data(int p_idx, int p_level, int p_x, int p_y, int p_z, const GIProbeDataCell *p_cell, const GIProbeDataHeader *p_header, InstanceGIProbeData::LocalData *p_local_data, Vector<uint32_t> *prev_cell) {

	if ((uint32_t)p_level == p_header->cell_subdiv - 1) {

		Vector3 emission;
		emission.x = (p_cell[p_idx].emission >> 24) / 255.0;
		emission.y = ((p_cell[p_idx].emission >> 16) & 0xFF) / 255.0;
		emission.z = ((p_cell[p_idx].emission >> 8) & 0xFF) / 255.0;
		float l = (p_cell[p_idx].emission & 0xFF) / 255.0;
		l *= 8.0;

		emission *= l;

		p_local_data[p_idx].energy[0] = uint16_t(emission.x * 1024); //go from 0 to 1024 for light
		p_local_data[p_idx].energy[1] = uint16_t(emission.y * 1024); //go from 0 to 1024 for light
		p_local_data[p_idx].energy[2] = uint16_t(emission.z * 1024); //go from 0 to 1024 for light
	} else {

		p_local_data[p_idx].energy[0] = 0;
		p_local_data[p_idx].energy[1] = 0;
		p_local_data[p_idx].energy[2] = 0;

		int half = (1 << (p_header->cell_subdiv - 1)) >> (p_level + 1);

		for (int i = 0; i < 8; i++) {

			uint32_t child = p_cell[p_idx].children[i];

			if (child == 0xFFFFFFFF)
				continue;

			int x = p_x;
			int y = p_y;
			int z = p_z;

			if (i & 1)
				x += half;
			if (i & 2)
				y += half;
			if (i & 4)
				z += half;

			_gi_probe_fill_local_data(child, p_level + 1, x, y, z, p_cell, p_header, p_local_data, prev_cell);
		}
	}

	//position for each part of the mipmaped texture
	p_local_data[p_idx].pos[0] = p_x >> (p_header->cell_subdiv - p_level - 1);
	p_local_data[p_idx].pos[1] = p_y >> (p_header->cell_subdiv - p_level - 1);
	p_local_data[p_idx].pos[2] = p_z >> (p_header->cell_subdiv - p_level - 1);

	prev_cell[p_level].push_back(p_idx);
}

void VisualServerScene::_gi_probe_bake_threads(void *self) {

	VisualServerScene *vss = (VisualServerScene *)self;
	vss->_gi_probe_bake_thread();
}

void VisualServerScene::_setup_gi_probe(Instance *p_instance) {

	InstanceGIProbeData *probe = static_cast<InstanceGIProbeData *>(p_instance->base_data);

	if (probe->dynamic.probe_data.is_valid()) {
		VSG::storage->free(probe->dynamic.probe_data);
		probe->dynamic.probe_data = RID();
	}

	probe->dynamic.light_data = VSG::storage->gi_probe_get_dynamic_data(p_instance->base);

	if (probe->dynamic.light_data.size() == 0)
		return;
	//using dynamic data
	PoolVector<int>::Read r = probe->dynamic.light_data.read();

	const GIProbeDataHeader *header = (GIProbeDataHeader *)r.ptr();

	probe->dynamic.local_data.resize(header->cell_count);

	int cell_count = probe->dynamic.local_data.size();
	PoolVector<InstanceGIProbeData::LocalData>::Write ldw = probe->dynamic.local_data.write();
	const GIProbeDataCell *cells = (GIProbeDataCell *)&r[16];

	probe->dynamic.level_cell_lists.resize(header->cell_subdiv);

	_gi_probe_fill_local_data(0, 0, 0, 0, 0, cells, header, ldw.ptr(), probe->dynamic.level_cell_lists.ptr());

	bool compress = VSG::storage->gi_probe_is_compressed(p_instance->base);

	probe->dynamic.compression = compress ? VSG::storage->gi_probe_get_dynamic_data_get_preferred_compression() : RasterizerStorage::GI_PROBE_UNCOMPRESSED;

	probe->dynamic.probe_data = VSG::storage->gi_probe_dynamic_data_create(header->width, header->height, header->depth, probe->dynamic.compression);

	probe->dynamic.bake_dynamic_range = VSG::storage->gi_probe_get_dynamic_range(p_instance->base);

	probe->dynamic.mipmaps_3d.clear();
	probe->dynamic.propagate = VSG::storage->gi_probe_get_propagation(p_instance->base);

	probe->dynamic.grid_size[0] = header->width;
	probe->dynamic.grid_size[1] = header->height;
	probe->dynamic.grid_size[2] = header->depth;

	int size_limit = 1;
	int size_divisor = 1;

	if (probe->dynamic.compression == RasterizerStorage::GI_PROBE_S3TC) {
		print_line("S3TC");
		size_limit = 4;
		size_divisor = 4;
	}
	for (int i = 0; i < (int)header->cell_subdiv; i++) {

		int x = header->width >> i;
		int y = header->height >> i;
		int z = header->depth >> i;

		//create and clear mipmap
		PoolVector<uint8_t> mipmap;
		int size = x * y * z * 4;
		size /= size_divisor;
		mipmap.resize(size);
		PoolVector<uint8_t>::Write w = mipmap.write();
		zeromem(w.ptr(), size);
		w = PoolVector<uint8_t>::Write();

		probe->dynamic.mipmaps_3d.push_back(mipmap);

		if (x <= size_limit || y <= size_limit || z <= size_limit)
			break;
	}

	probe->dynamic.updating_stage = GI_UPDATE_STAGE_CHECK;
	probe->invalid = false;
	probe->dynamic.enabled = true;

	Transform cell_to_xform = VSG::storage->gi_probe_get_to_cell_xform(p_instance->base);
	Rect3 bounds = VSG::storage->gi_probe_get_bounds(p_instance->base);
	float cell_size = VSG::storage->gi_probe_get_cell_size(p_instance->base);

	probe->dynamic.light_to_cell_xform = cell_to_xform * p_instance->transform.affine_inverse();

	VSG::scene_render->gi_probe_instance_set_light_data(probe->probe_instance, p_instance->base, probe->dynamic.probe_data);
	VSG::scene_render->gi_probe_instance_set_transform_to_data(probe->probe_instance, probe->dynamic.light_to_cell_xform);

	VSG::scene_render->gi_probe_instance_set_bounds(probe->probe_instance, bounds.size / cell_size);

	probe->base_version = VSG::storage->gi_probe_get_version(p_instance->base);

	//if compression is S3TC, fill it up
	if (probe->dynamic.compression == RasterizerStorage::GI_PROBE_S3TC) {

		//create all blocks
		Vector<Map<uint32_t, InstanceGIProbeData::CompBlockS3TC> > comp_blocks;
		int mipmap_count = probe->dynamic.mipmaps_3d.size();
		comp_blocks.resize(mipmap_count);

		for (int i = 0; i < cell_count; i++) {

			const GIProbeDataCell &c = cells[i];
			const InstanceGIProbeData::LocalData &ld = ldw[i];
			int level = c.level_alpha >> 16;
			int mipmap = header->cell_subdiv - level - 1;
			if (mipmap >= mipmap_count)
				continue; //uninteresting

			int blockx = (ld.pos[0] >> 2);
			int blocky = (ld.pos[1] >> 2);
			int blockz = (ld.pos[2]); //compression is x/y only

			int blockw = (header->width >> mipmap) >> 2;
			int blockh = (header->height >> mipmap) >> 2;

			//print_line("cell "+itos(i)+" level "+itos(level)+"mipmap: "+itos(mipmap)+" pos: "+Vector3(blockx,blocky,blockz)+" size "+Vector2(blockw,blockh));

			uint32_t key = blockz * blockw * blockh + blocky * blockw + blockx;

			Map<uint32_t, InstanceGIProbeData::CompBlockS3TC> &cmap = comp_blocks[mipmap];

			if (!cmap.has(key)) {

				InstanceGIProbeData::CompBlockS3TC k;
				k.offset = key; //use offset as counter first
				k.source_count = 0;
				cmap[key] = k;
			}

			InstanceGIProbeData::CompBlockS3TC &k = cmap[key];
			ERR_CONTINUE(k.source_count == 16);
			k.sources[k.source_count++] = i;
		}

		//fix the blocks, precomputing what is needed
		probe->dynamic.mipmaps_s3tc.resize(mipmap_count);

		for (int i = 0; i < mipmap_count; i++) {
			print_line("S3TC level: " + itos(i) + " blocks: " + itos(comp_blocks[i].size()));
			probe->dynamic.mipmaps_s3tc[i].resize(comp_blocks[i].size());
			PoolVector<InstanceGIProbeData::CompBlockS3TC>::Write w = probe->dynamic.mipmaps_s3tc[i].write();
			int block_idx = 0;

			for (Map<uint32_t, InstanceGIProbeData::CompBlockS3TC>::Element *E = comp_blocks[i].front(); E; E = E->next()) {

				InstanceGIProbeData::CompBlockS3TC k = E->get();

				//PRECOMPUTE ALPHA
				int max_alpha = -100000;
				int min_alpha = k.source_count == 16 ? 100000 : 0; //if the block is not completely full, minimum is always 0, (and those blocks will map to 1, which will be zero)

				uint8_t alpha_block[4][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };

				for (uint32_t j = 0; j < k.source_count; j++) {

					int alpha = (cells[k.sources[j]].level_alpha >> 8) & 0xFF;
					if (alpha < min_alpha)
						min_alpha = alpha;
					if (alpha > max_alpha)
						max_alpha = alpha;
					//fill up alpha block
					alpha_block[ldw[k.sources[j]].pos[0] % 4][ldw[k.sources[j]].pos[1] % 4] = alpha;
				}

				//use the first mode (8 adjustable levels)
				k.alpha[0] = max_alpha;
				k.alpha[1] = min_alpha;

				uint64_t alpha_bits = 0;

				if (max_alpha != min_alpha) {

					int idx = 0;

					for (int y = 0; y < 4; y++) {
						for (int x = 0; x < 4; x++) {

							//subtract minimum
							uint32_t a = uint32_t(alpha_block[x][y]) - min_alpha;
							//convert range to 3 bits
							a = int((a * 7.0 / (max_alpha - min_alpha)) + 0.5);
							a = CLAMP(a, 0, 7); //just to be sure
							a = 7 - a; //because range is inverted in this mode
							if (a == 0) {
								//do none, remain
							} else if (a == 7) {
								a = 1;
							} else {
								a = a + 1;
							}

							alpha_bits |= uint64_t(a) << (idx * 3);
							idx++;
						}
					}
				}

				k.alpha[2] = (alpha_bits >> 0) & 0xFF;
				k.alpha[3] = (alpha_bits >> 8) & 0xFF;
				k.alpha[4] = (alpha_bits >> 16) & 0xFF;
				k.alpha[5] = (alpha_bits >> 24) & 0xFF;
				k.alpha[6] = (alpha_bits >> 32) & 0xFF;
				k.alpha[7] = (alpha_bits >> 40) & 0xFF;

				w[block_idx++] = k;
			}
		}
	}
}

void VisualServerScene::_gi_probe_bake_thread() {

	while (true) {

		probe_bake_sem->wait();
		if (probe_bake_thread_exit) {
			break;
		}

		Instance *to_bake = NULL;

		probe_bake_mutex->lock();

		if (!probe_bake_list.empty()) {
			to_bake = probe_bake_list.front()->get();
			probe_bake_list.pop_front();
		}
		probe_bake_mutex->unlock();

		if (!to_bake)
			continue;

		_bake_gi_probe(to_bake);
	}
}

uint32_t VisualServerScene::_gi_bake_find_cell(const GIProbeDataCell *cells, int x, int y, int z, int p_cell_subdiv) {

	uint32_t cell = 0;

	int ofs_x = 0;
	int ofs_y = 0;
	int ofs_z = 0;
	int size = 1 << (p_cell_subdiv - 1);
	int half = size / 2;

	if (x < 0 || x >= size)
		return -1;
	if (y < 0 || y >= size)
		return -1;
	if (z < 0 || z >= size)
		return -1;

	for (int i = 0; i < p_cell_subdiv - 1; i++) {

		const GIProbeDataCell *bc = &cells[cell];

		int child = 0;
		if (x >= ofs_x + half) {
			child |= 1;
			ofs_x += half;
		}
		if (y >= ofs_y + half) {
			child |= 2;
			ofs_y += half;
		}
		if (z >= ofs_z + half) {
			child |= 4;
			ofs_z += half;
		}

		cell = bc->children[child];
		if (cell == 0xFFFFFFFF)
			return 0xFFFFFFFF;

		half >>= 1;
	}

	return cell;
}

static float _get_normal_advance(const Vector3 &p_normal) {

	Vector3 normal = p_normal;
	Vector3 unorm = normal.abs();

	if ((unorm.x >= unorm.y) && (unorm.x >= unorm.z)) {
		// x code
		unorm = normal.x > 0.0 ? Vector3(1.0, 0.0, 0.0) : Vector3(-1.0, 0.0, 0.0);
	} else if ((unorm.y > unorm.x) && (unorm.y >= unorm.z)) {
		// y code
		unorm = normal.y > 0.0 ? Vector3(0.0, 1.0, 0.0) : Vector3(0.0, -1.0, 0.0);
	} else if ((unorm.z > unorm.x) && (unorm.z > unorm.y)) {
		// z code
		unorm = normal.z > 0.0 ? Vector3(0.0, 0.0, 1.0) : Vector3(0.0, 0.0, -1.0);
	} else {
		// oh-no we messed up code
		// has to be
		unorm = Vector3(1.0, 0.0, 0.0);
	}

	return 1.0 / normal.dot(unorm);
}

void VisualServerScene::_bake_gi_probe_light(const GIProbeDataHeader *header, const GIProbeDataCell *cells, InstanceGIProbeData::LocalData *local_data, const uint32_t *leaves, int p_leaf_count, const InstanceGIProbeData::LightCache &light_cache, int p_sign) {

	int light_r = int(light_cache.color.r * light_cache.energy * 1024.0) * p_sign;
	int light_g = int(light_cache.color.g * light_cache.energy * 1024.0) * p_sign;
	int light_b = int(light_cache.color.b * light_cache.energy * 1024.0) * p_sign;

	float limits[3] = { float(header->width), float(header->height), float(header->depth) };
	Plane clip[3];
	int clip_planes = 0;

	switch (light_cache.type) {

		case VS::LIGHT_DIRECTIONAL: {

			float max_len = Vector3(limits[0], limits[1], limits[2]).length() * 1.1;

			Vector3 light_axis = -light_cache.transform.basis.get_axis(2).normalized();

			for (int i = 0; i < 3; i++) {

				if (ABS(light_axis[i]) < CMP_EPSILON)
					continue;
				clip[clip_planes].normal[i] = 1.0;

				if (light_axis[i] < 0) {

					clip[clip_planes].d = limits[i] + 1;
				} else {
					clip[clip_planes].d -= 1.0;
				}

				clip_planes++;
			}

			float distance_adv = _get_normal_advance(light_axis);

			int success_count = 0;

			// uint64_t us = OS::get_singleton()->get_ticks_usec();

			for (int i = 0; i < p_leaf_count; i++) {

				uint32_t idx = leaves[i];

				const GIProbeDataCell *cell = &cells[idx];
				InstanceGIProbeData::LocalData *light = &local_data[idx];

				Vector3 to(light->pos[0] + 0.5, light->pos[1] + 0.5, light->pos[2] + 0.5);
				Vector3 norm(
						(((cells[idx].normal >> 16) & 0xFF) / 255.0) * 2.0 - 1.0,
						(((cells[idx].normal >> 8) & 0xFF) / 255.0) * 2.0 - 1.0,
						(((cells[idx].normal >> 0) & 0xFF) / 255.0) * 2.0 - 1.0);

				float att = norm.dot(-light_axis);
				if (att < 0.001) {
					//not lighting towards this
					continue;
				}

				Vector3 from = to - max_len * light_axis;

				for (int j = 0; j < clip_planes; j++) {

					clip[j].intersects_segment(from, to, &from);
				}

				float distance = (to - from).length();
				distance += distance_adv - Math::fmod(distance, distance_adv); //make it reach the center of the box always
				from = to - light_axis * distance;

				uint32_t result = 0xFFFFFFFF;

				while (distance > -distance_adv) { //use this to avoid precision errors

					result = _gi_bake_find_cell(cells, int(floor(from.x)), int(floor(from.y)), int(floor(from.z)), header->cell_subdiv);
					if (result != 0xFFFFFFFF) {
						break;
					}

					from += light_axis * distance_adv;
					distance -= distance_adv;
				}

				if (result == idx) {
					//cell hit itself! hooray!
					light->energy[0] += int32_t(light_r * att * ((cell->albedo >> 16) & 0xFF) / 255.0);
					light->energy[1] += int32_t(light_g * att * ((cell->albedo >> 8) & 0xFF) / 255.0);
					light->energy[2] += int32_t(light_b * att * ((cell->albedo) & 0xFF) / 255.0);
					success_count++;
				}
			}

			// print_line("BAKE TIME: " + rtos((OS::get_singleton()->get_ticks_usec() - us) / 1000000.0));
			// print_line("valid cells: " + itos(success_count));

		} break;
		case VS::LIGHT_OMNI:
		case VS::LIGHT_SPOT: {

			// uint64_t us = OS::get_singleton()->get_ticks_usec();

			Vector3 light_pos = light_cache.transform.origin;
			Vector3 spot_axis = -light_cache.transform.basis.get_axis(2).normalized();

			float local_radius = light_cache.radius * light_cache.transform.basis.get_axis(2).length();

			for (int i = 0; i < p_leaf_count; i++) {

				uint32_t idx = leaves[i];

				const GIProbeDataCell *cell = &cells[idx];
				InstanceGIProbeData::LocalData *light = &local_data[idx];

				Vector3 to(light->pos[0] + 0.5, light->pos[1] + 0.5, light->pos[2] + 0.5);
				Vector3 norm(
						(((cells[idx].normal >> 16) & 0xFF) / 255.0) * 2.0 - 1.0,
						(((cells[idx].normal >> 8) & 0xFF) / 255.0) * 2.0 - 1.0,
						(((cells[idx].normal >> 0) & 0xFF) / 255.0) * 2.0 - 1.0);

				Vector3 light_axis = (to - light_pos).normalized();
				float distance_adv = _get_normal_advance(light_axis);

				float att = norm.dot(-light_axis);
				if (att < 0.001) {
					//not lighting towards this
					continue;
				}

				{
					float d = light_pos.distance_to(to);
					if (d + distance_adv > local_radius)
						continue; // too far away

					float dt = CLAMP((d + distance_adv) / local_radius, 0, 1);
					att *= powf(1.0 - dt, light_cache.attenuation);
				}

				if (light_cache.type == VS::LIGHT_SPOT) {

					float angle = Math::rad2deg(acos(light_axis.dot(spot_axis)));
					if (angle > light_cache.spot_angle)
						continue;

					float d = CLAMP(angle / light_cache.spot_angle, 1, 0);
					att *= powf(1.0 - d, light_cache.spot_attenuation);
				}

				clip_planes = 0;

				for (int c = 0; c < 3; c++) {

					if (ABS(light_axis[c]) < CMP_EPSILON)
						continue;
					clip[clip_planes].normal[c] = 1.0;

					if (light_axis[c] < 0) {

						clip[clip_planes].d = limits[c] + 1;
					} else {
						clip[clip_planes].d -= 1.0;
					}

					clip_planes++;
				}

				Vector3 from = light_pos;

				for (int j = 0; j < clip_planes; j++) {

					clip[j].intersects_segment(from, to, &from);
				}

				float distance = (to - from).length();

				distance -= Math::fmod(distance, distance_adv); //make it reach the center of the box always, but this tame make it closer
				from = to - light_axis * distance;

				uint32_t result = 0xFFFFFFFF;

				while (distance > -distance_adv) { //use this to avoid precision errors

					result = _gi_bake_find_cell(cells, int(floor(from.x)), int(floor(from.y)), int(floor(from.z)), header->cell_subdiv);
					if (result != 0xFFFFFFFF) {
						break;
					}

					from += light_axis * distance_adv;
					distance -= distance_adv;
				}

				if (result == idx) {
					//cell hit itself! hooray!

					light->energy[0] += int32_t(light_r * att * ((cell->albedo >> 16) & 0xFF) / 255.0);
					light->energy[1] += int32_t(light_g * att * ((cell->albedo >> 8) & 0xFF) / 255.0);
					light->energy[2] += int32_t(light_b * att * ((cell->albedo) & 0xFF) / 255.0);
				}
			}
			// print_line("BAKE TIME: " + rtos((OS::get_singleton()->get_ticks_usec() - us) / 1000000.0));
		} break;
	}
}

void VisualServerScene::_bake_gi_downscale_light(int p_idx, int p_level, const GIProbeDataCell *p_cells, const GIProbeDataHeader *p_header, InstanceGIProbeData::LocalData *p_local_data, float p_propagate) {

	//average light to upper level

	float divisor = 0;
	float sum[3] = { 0.0, 0.0, 0.0 };

	for (int i = 0; i < 8; i++) {

		uint32_t child = p_cells[p_idx].children[i];

		if (child == 0xFFFFFFFF)
			continue;

		if (p_level + 1 < (int)p_header->cell_subdiv - 1) {
			_bake_gi_downscale_light(child, p_level + 1, p_cells, p_header, p_local_data, p_propagate);
		}

		sum[0] += p_local_data[child].energy[0];
		sum[1] += p_local_data[child].energy[1];
		sum[2] += p_local_data[child].energy[2];
		divisor += 1.0;
	}

	divisor = Math::lerp((float)8.0, divisor, p_propagate);
	sum[0] /= divisor;
	sum[1] /= divisor;
	sum[2] /= divisor;

	//divide by eight for average
	p_local_data[p_idx].energy[0] = Math::fast_ftoi(sum[0]);
	p_local_data[p_idx].energy[1] = Math::fast_ftoi(sum[1]);
	p_local_data[p_idx].energy[2] = Math::fast_ftoi(sum[2]);
}

void VisualServerScene::_bake_gi_probe(Instance *p_gi_probe) {

	InstanceGIProbeData *probe_data = static_cast<InstanceGIProbeData *>(p_gi_probe->base_data);

	PoolVector<int>::Read r = probe_data->dynamic.light_data.read();

	const GIProbeDataHeader *header = (const GIProbeDataHeader *)r.ptr();
	const GIProbeDataCell *cells = (const GIProbeDataCell *)&r[16];

	int leaf_count = probe_data->dynamic.level_cell_lists[header->cell_subdiv - 1].size();
	const uint32_t *leaves = probe_data->dynamic.level_cell_lists[header->cell_subdiv - 1].ptr();

	PoolVector<InstanceGIProbeData::LocalData>::Write ldw = probe_data->dynamic.local_data.write();

	InstanceGIProbeData::LocalData *local_data = ldw.ptr();

	//remove what must be removed
	for (Map<RID, InstanceGIProbeData::LightCache>::Element *E = probe_data->dynamic.light_cache.front(); E; E = E->next()) {

		RID rid = E->key();
		const InstanceGIProbeData::LightCache &lc = E->get();

		if ((!probe_data->dynamic.light_cache_changes.has(rid) || !(probe_data->dynamic.light_cache_changes[rid] == lc)) && lc.visible) {
			//erase light data

			_bake_gi_probe_light(header, cells, local_data, leaves, leaf_count, lc, -1);
		}
	}

	//add what must be added
	for (Map<RID, InstanceGIProbeData::LightCache>::Element *E = probe_data->dynamic.light_cache_changes.front(); E; E = E->next()) {

		RID rid = E->key();
		const InstanceGIProbeData::LightCache &lc = E->get();

		if ((!probe_data->dynamic.light_cache.has(rid) || !(probe_data->dynamic.light_cache[rid] == lc)) && lc.visible) {
			//add light data

			_bake_gi_probe_light(header, cells, local_data, leaves, leaf_count, lc, 1);
		}
	}

	SWAP(probe_data->dynamic.light_cache_changes, probe_data->dynamic.light_cache);

	//downscale to lower res levels
	_bake_gi_downscale_light(0, 0, cells, header, local_data, probe_data->dynamic.propagate);

	//plot result to 3D texture!

	if (probe_data->dynamic.compression == RasterizerStorage::GI_PROBE_UNCOMPRESSED) {

		for (int i = 0; i < (int)header->cell_subdiv; i++) {

			int stage = header->cell_subdiv - i - 1;

			if (stage >= probe_data->dynamic.mipmaps_3d.size())
				continue; //no mipmap for this one

			//print_line("generating mipmap stage: " + itos(stage));
			int level_cell_count = probe_data->dynamic.level_cell_lists[i].size();
			const uint32_t *level_cells = probe_data->dynamic.level_cell_lists[i].ptr();

			PoolVector<uint8_t>::Write lw = probe_data->dynamic.mipmaps_3d[stage].write();
			uint8_t *mipmapw = lw.ptr();

			uint32_t sizes[3] = { header->width >> stage, header->height >> stage, header->depth >> stage };

			for (int j = 0; j < level_cell_count; j++) {

				uint32_t idx = level_cells[j];

				uint32_t r = (uint32_t(local_data[idx].energy[0]) / probe_data->dynamic.bake_dynamic_range) >> 2;
				uint32_t g = (uint32_t(local_data[idx].energy[1]) / probe_data->dynamic.bake_dynamic_range) >> 2;
				uint32_t b = (uint32_t(local_data[idx].energy[2]) / probe_data->dynamic.bake_dynamic_range) >> 2;
				uint32_t a = (cells[idx].level_alpha >> 8) & 0xFF;

				uint32_t mm_ofs = sizes[0] * sizes[1] * (local_data[idx].pos[2]) + sizes[0] * (local_data[idx].pos[1]) + (local_data[idx].pos[0]);
				mm_ofs *= 4; //for RGBA (4 bytes)

				mipmapw[mm_ofs + 0] = uint8_t(CLAMP(r, 0, 255));
				mipmapw[mm_ofs + 1] = uint8_t(CLAMP(g, 0, 255));
				mipmapw[mm_ofs + 2] = uint8_t(CLAMP(b, 0, 255));
				mipmapw[mm_ofs + 3] = uint8_t(CLAMP(a, 0, 255));
			}
		}
	} else if (probe_data->dynamic.compression == RasterizerStorage::GI_PROBE_S3TC) {

		int mipmap_count = probe_data->dynamic.mipmaps_3d.size();

		for (int mmi = 0; mmi < mipmap_count; mmi++) {

			PoolVector<uint8_t>::Write mmw = probe_data->dynamic.mipmaps_3d[mmi].write();
			int block_count = probe_data->dynamic.mipmaps_s3tc[mmi].size();
			PoolVector<InstanceGIProbeData::CompBlockS3TC>::Read mmr = probe_data->dynamic.mipmaps_s3tc[mmi].read();

			for (int i = 0; i < block_count; i++) {

				const InstanceGIProbeData::CompBlockS3TC &b = mmr[i];

				uint8_t *blockptr = &mmw[b.offset * 16];
				copymem(blockptr, b.alpha, 8); //copy alpha part, which is precomputed

				Vector3 colors[16];

				for (uint32_t j = 0; j < b.source_count; j++) {

					colors[j].x = (local_data[b.sources[j]].energy[0] / float(probe_data->dynamic.bake_dynamic_range)) / 1024.0;
					colors[j].y = (local_data[b.sources[j]].energy[1] / float(probe_data->dynamic.bake_dynamic_range)) / 1024.0;
					colors[j].z = (local_data[b.sources[j]].energy[2] / float(probe_data->dynamic.bake_dynamic_range)) / 1024.0;
				}
				//super quick and dirty compression
				//find 2 most further apart
				float distance = 0;
				Vector3 from, to;

				if (b.source_count == 16) {
					//all cells are used so, find minmax between them
					int further_apart[2] = { 0, 0 };
					for (uint32_t j = 0; j < b.source_count; j++) {
						for (uint32_t k = j + 1; k < b.source_count; k++) {
							float d = colors[j].distance_squared_to(colors[k]);
							if (d > distance) {
								distance = d;
								further_apart[0] = j;
								further_apart[1] = k;
							}
						}
					}

					from = colors[further_apart[0]];
					to = colors[further_apart[1]];

				} else {
					//if a block is missing, the priority is that this block remains black,
					//otherwise the geometry will appear deformed
					//correct shape wins over correct color in this case
					//average all colors first
					Vector3 average;

					for (uint32_t j = 0; j < b.source_count; j++) {
						average += colors[j];
					}
					average.normalize();
					//find max distance in normal from average
					for (uint32_t j = 0; j < b.source_count; j++) {
						float d = average.dot(colors[j]);
						distance = MAX(d, distance);
					}

					from = Vector3(); //from black
					to = average * distance;
					//find max distance
				}

				int indices[16];
				uint16_t color_0 = 0;
				color_0 = CLAMP(int(from.x * 31), 0, 31) << 11;
				color_0 |= CLAMP(int(from.y * 63), 0, 63) << 5;
				color_0 |= CLAMP(int(from.z * 31), 0, 31);

				uint16_t color_1 = 0;
				color_1 = CLAMP(int(to.x * 31), 0, 31) << 11;
				color_1 |= CLAMP(int(to.y * 63), 0, 63) << 5;
				color_1 |= CLAMP(int(to.z * 31), 0, 31);

				if (color_1 > color_0) {
					SWAP(color_1, color_0);
					SWAP(from, to);
				}

				if (distance > 0) {

					Vector3 dir = (to - from).normalized();

					for (uint32_t j = 0; j < b.source_count; j++) {

						float d = (colors[j] - from).dot(dir) / distance;
						indices[j] = int(d * 3 + 0.5);

						static const int index_swap[4] = { 0, 3, 1, 2 };

						indices[j] = index_swap[CLAMP(indices[j], 0, 3)];
					}
				} else {
					for (uint32_t j = 0; j < b.source_count; j++) {
						indices[j] = 0;
					}
				}

				//by default, 1 is black, otherwise it will be overridden by source

				uint32_t index_block[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

				for (uint32_t j = 0; j < b.source_count; j++) {

					int x = local_data[b.sources[j]].pos[0] % 4;
					int y = local_data[b.sources[j]].pos[1] % 4;

					index_block[y * 4 + x] = indices[j];
				}

				uint32_t encode = 0;

				for (int j = 0; j < 16; j++) {
					encode |= index_block[j] << (j * 2);
				}

				blockptr[8] = color_0 & 0xFF;
				blockptr[9] = (color_0 >> 8) & 0xFF;
				blockptr[10] = color_1 & 0xFF;
				blockptr[11] = (color_1 >> 8) & 0xFF;
				blockptr[12] = encode & 0xFF;
				blockptr[13] = (encode >> 8) & 0xFF;
				blockptr[14] = (encode >> 16) & 0xFF;
				blockptr[15] = (encode >> 24) & 0xFF;
			}
		}
	}

	//send back to main thread to update un little chunks
	probe_data->dynamic.updating_stage = GI_UPDATE_STAGE_UPLOADING;
}

bool VisualServerScene::_check_gi_probe(Instance *p_gi_probe) {

	InstanceGIProbeData *probe_data = static_cast<InstanceGIProbeData *>(p_gi_probe->base_data);

	probe_data->dynamic.light_cache_changes.clear();

	bool all_equal = true;

	for (List<Instance *>::Element *E = p_gi_probe->scenario->directional_lights.front(); E; E = E->next()) {

		InstanceGIProbeData::LightCache lc;
		lc.type = VSG::storage->light_get_type(E->get()->base);
		lc.color = VSG::storage->light_get_color(E->get()->base);
		lc.energy = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_ENERGY);
		lc.radius = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_RANGE);
		lc.attenuation = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_ATTENUATION);
		lc.spot_angle = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_SPOT_ANGLE);
		lc.spot_attenuation = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_SPOT_ATTENUATION);
		lc.transform = probe_data->dynamic.light_to_cell_xform * E->get()->transform;
		lc.visible = E->get()->visible;

		if (!probe_data->dynamic.light_cache.has(E->get()->self) || !(probe_data->dynamic.light_cache[E->get()->self] == lc)) {
			all_equal = false;
		}

		probe_data->dynamic.light_cache_changes[E->get()->self] = lc;
	}

	for (Set<Instance *>::Element *E = probe_data->lights.front(); E; E = E->next()) {

		InstanceGIProbeData::LightCache lc;
		lc.type = VSG::storage->light_get_type(E->get()->base);
		lc.color = VSG::storage->light_get_color(E->get()->base);
		lc.energy = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_ENERGY);
		lc.radius = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_RANGE);
		lc.attenuation = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_ATTENUATION);
		lc.spot_angle = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_SPOT_ANGLE);
		lc.spot_attenuation = VSG::storage->light_get_param(E->get()->base, VS::LIGHT_PARAM_SPOT_ATTENUATION);
		lc.transform = probe_data->dynamic.light_to_cell_xform * E->get()->transform;
		lc.visible = E->get()->visible;

		if (!probe_data->dynamic.light_cache.has(E->get()->self) || !(probe_data->dynamic.light_cache[E->get()->self] == lc)) {
			all_equal = false;
		}

		probe_data->dynamic.light_cache_changes[E->get()->self] = lc;
	}

	//lighting changed from after to before, must do some updating
	return !all_equal || probe_data->dynamic.light_cache_changes.size() != probe_data->dynamic.light_cache.size();
}

void VisualServerScene::render_probes() {

	/* REFLECTION PROBES */

	SelfList<InstanceReflectionProbeData> *ref_probe = reflection_probe_render_list.first();

	bool busy = false;

	while (ref_probe) {

		SelfList<InstanceReflectionProbeData> *next = ref_probe->next();
		RID base = ref_probe->self()->owner->base;

		switch (VSG::storage->reflection_probe_get_update_mode(base)) {

			case VS::REFLECTION_PROBE_UPDATE_ONCE: {
				if (busy) //already rendering something
					break;

				bool done = _render_reflection_probe_step(ref_probe->self()->owner, ref_probe->self()->render_step);
				if (done) {
					reflection_probe_render_list.remove(ref_probe);
				} else {
					ref_probe->self()->render_step++;
				}

				busy = true; //do not render another one of this kind
			} break;
			case VS::REFLECTION_PROBE_UPDATE_ALWAYS: {

				int step = 0;
				bool done = false;
				while (!done) {
					done = _render_reflection_probe_step(ref_probe->self()->owner, step);
					step++;
				}

				reflection_probe_render_list.remove(ref_probe);
			} break;
		}

		ref_probe = next;
	}

	/* GI PROBES */

	SelfList<InstanceGIProbeData> *gi_probe = gi_probe_update_list.first();

	while (gi_probe) {

		SelfList<InstanceGIProbeData> *next = gi_probe->next();

		InstanceGIProbeData *probe = gi_probe->self();
		Instance *instance_probe = probe->owner;

		//check if probe must be setup, but don't do if on the lighting thread

		bool force_lighting = false;

		if (probe->invalid || (probe->dynamic.updating_stage == GI_UPDATE_STAGE_CHECK && probe->base_version != VSG::storage->gi_probe_get_version(instance_probe->base))) {

			_setup_gi_probe(instance_probe);
			force_lighting = true;
		}

		float propagate = VSG::storage->gi_probe_get_propagation(instance_probe->base);

		if (probe->dynamic.propagate != propagate) {
			probe->dynamic.propagate = propagate;
			force_lighting = true;
		}

		if (probe->invalid == false && probe->dynamic.enabled) {

			switch (probe->dynamic.updating_stage) {
				case GI_UPDATE_STAGE_CHECK: {

					if (_check_gi_probe(instance_probe) || force_lighting) {
						//send to lighting thread
						probe->dynamic.updating_stage = GI_UPDATE_STAGE_LIGHTING;

#ifndef NO_THREADS
						probe_bake_mutex->lock();
						probe_bake_list.push_back(instance_probe);
						probe_bake_mutex->unlock();
						probe_bake_sem->post();

#else

						_bake_gi_probe(instance_probe);
#endif
					}
				} break;
				case GI_UPDATE_STAGE_LIGHTING: {
					//do none, wait til done!

				} break;
				case GI_UPDATE_STAGE_UPLOADING: {

					// uint64_t us = OS::get_singleton()->get_ticks_usec();

					for (int i = 0; i < (int)probe->dynamic.mipmaps_3d.size(); i++) {

						PoolVector<uint8_t>::Read r = probe->dynamic.mipmaps_3d[i].read();
						VSG::storage->gi_probe_dynamic_data_update(probe->dynamic.probe_data, 0, probe->dynamic.grid_size[2] >> i, i, r.ptr());
					}

					probe->dynamic.updating_stage = GI_UPDATE_STAGE_CHECK;

					// print_line("UPLOAD TIME: " + rtos((OS::get_singleton()->get_ticks_usec() - us) / 1000000.0));
				} break;
			}
		}
		//_update_gi_probe(gi_probe->self()->owner);

		gi_probe = next;
	}
}

void VisualServerScene::_update_dirty_instance(Instance *p_instance) {

	if (p_instance->update_aabb) {
		_update_instance_aabb(p_instance);
	}

	if (p_instance->update_materials) {

		if (p_instance->base_type == VS::INSTANCE_MESH) {
			//remove materials no longer used and un-own them

			int new_mat_count = VSG::storage->mesh_get_surface_count(p_instance->base);
			for (int i = p_instance->materials.size() - 1; i >= new_mat_count; i--) {
				if (p_instance->materials[i].is_valid()) {
					VSG::storage->material_remove_instance_owner(p_instance->materials[i], p_instance);
				}
			}
			p_instance->materials.resize(new_mat_count);

			int new_blend_shape_count = VSG::storage->mesh_get_blend_shape_count(p_instance->base);
			if (new_blend_shape_count != p_instance->blend_values.size()) {
				p_instance->blend_values.resize(new_blend_shape_count);
				for (int i = 0; i < new_blend_shape_count; i++) {
					p_instance->blend_values[i] = 0;
				}
			}
		}

		if ((1 << p_instance->base_type) & VS::INSTANCE_GEOMETRY_MASK) {

			InstanceGeometryData *geom = static_cast<InstanceGeometryData *>(p_instance->base_data);

			bool can_cast_shadows = true;

			if (p_instance->cast_shadows == VS::SHADOW_CASTING_SETTING_OFF) {
				can_cast_shadows = false;
			} else if (p_instance->material_override.is_valid()) {
				can_cast_shadows = VSG::storage->material_casts_shadows(p_instance->material_override);
			} else {

				if (p_instance->base_type == VS::INSTANCE_MESH) {
					RID mesh = p_instance->base;

					if (mesh.is_valid()) {
						bool cast_shadows = false;

						for (int i = 0; i < p_instance->materials.size(); i++) {

							RID mat = p_instance->materials[i].is_valid() ? p_instance->materials[i] : VSG::storage->mesh_surface_get_material(mesh, i);

							if (!mat.is_valid()) {
								cast_shadows = true;
								break;
							}

							if (VSG::storage->material_casts_shadows(mat)) {
								cast_shadows = true;
								break;
							}
						}

						if (!cast_shadows) {
							can_cast_shadows = false;
						}
					}

				} else if (p_instance->base_type == VS::INSTANCE_MULTIMESH) {
					RID mesh = VSG::storage->multimesh_get_mesh(p_instance->base);
					if (mesh.is_valid()) {

						bool cast_shadows = false;

						int sc = VSG::storage->mesh_get_surface_count(mesh);
						for (int i = 0; i < sc; i++) {

							RID mat = VSG::storage->mesh_surface_get_material(mesh, i);

							if (!mat.is_valid()) {
								cast_shadows = true;
								break;
							}

							if (VSG::storage->material_casts_shadows(mat)) {
								cast_shadows = true;
								break;
							}
						}

						if (!cast_shadows) {
							can_cast_shadows = false;
						}
					}
				} else if (p_instance->base_type == VS::INSTANCE_IMMEDIATE) {

					RID mat = VSG::storage->immediate_get_material(p_instance->base);

					if (!mat.is_valid() || VSG::storage->material_casts_shadows(mat)) {
						can_cast_shadows = true;
					} else {
						can_cast_shadows = false;
					}
				} else if (p_instance->base_type == VS::INSTANCE_PARTICLES) {

					bool cast_shadows = false;

					int dp = VSG::storage->particles_get_draw_passes(p_instance->base);

					for (int i = 0; i < dp; i++) {

						RID mesh = VSG::storage->particles_get_draw_pass_mesh(p_instance->base, i);
						if (!mesh.is_valid())
							continue;

						int sc = VSG::storage->mesh_get_surface_count(mesh);
						for (int j = 0; j < sc; j++) {

							RID mat = VSG::storage->mesh_surface_get_material(mesh, j);

							if (!mat.is_valid()) {
								cast_shadows = true;
								break;
							}

							if (VSG::storage->material_casts_shadows(mat)) {
								cast_shadows = true;
								break;
							}
						}
					}

					if (!cast_shadows) {
						can_cast_shadows = false;
					}
				}
			}

			if (can_cast_shadows != geom->can_cast_shadows) {
				//ability to cast shadows change, let lights now
				for (List<Instance *>::Element *E = geom->lighting.front(); E; E = E->next()) {
					InstanceLightData *light = static_cast<InstanceLightData *>(E->get()->base_data);
					light->shadow_dirty = true;
				}

				geom->can_cast_shadows = can_cast_shadows;
			}
		}
	}

	_update_instance(p_instance);

	p_instance->update_aabb = false;
	p_instance->update_materials = false;

	_instance_update_list.remove(&p_instance->update_item);
}

void VisualServerScene::update_dirty_instances() {

	VSG::storage->update_dirty_resources();

	while (_instance_update_list.first()) {

		_update_dirty_instance(_instance_update_list.first()->self());
	}
}

bool VisualServerScene::free(RID p_rid) {

	if (camera_owner.owns(p_rid)) {

		Camera *camera = camera_owner.get(p_rid);

		camera_owner.free(p_rid);
		memdelete(camera);

	} else if (scenario_owner.owns(p_rid)) {

		Scenario *scenario = scenario_owner.get(p_rid);

		while (scenario->instances.first()) {
			instance_set_scenario(scenario->instances.first()->self()->self, RID());
		}
		VSG::scene_render->free(scenario->reflection_probe_shadow_atlas);
		VSG::scene_render->free(scenario->reflection_atlas);
		scenario_owner.free(p_rid);
		memdelete(scenario);

	} else if (instance_owner.owns(p_rid)) {
		// delete the instance

		update_dirty_instances();

		Instance *instance = instance_owner.get(p_rid);

		instance_set_scenario(p_rid, RID());
		instance_set_base(p_rid, RID());
		instance_geometry_set_material_override(p_rid, RID());
		instance_attach_skeleton(p_rid, RID());

		update_dirty_instances(); //in case something changed this

		instance_owner.free(p_rid);
		memdelete(instance);
	} else {
		return false;
	}

	return true;
}

VisualServerScene *VisualServerScene::singleton = NULL;

VisualServerScene::VisualServerScene() {

#ifndef NO_THREADS
	probe_bake_sem = Semaphore::create();
	probe_bake_mutex = Mutex::create();
	probe_bake_thread = Thread::create(_gi_probe_bake_threads, this);
	probe_bake_thread_exit = false;
#endif

	render_pass = 1;
	singleton = this;
}

VisualServerScene::~VisualServerScene() {

#ifndef NO_THREADS
	probe_bake_thread_exit = true;
	Thread::wait_to_finish(probe_bake_thread);
	memdelete(probe_bake_thread);
	memdelete(probe_bake_sem);
	memdelete(probe_bake_mutex);

#endif
}
