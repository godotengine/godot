/*************************************************************************/
/*  rendering_server_scene.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VISUALSERVERSCENE_H
#define VISUALSERVERSCENE_H

#include "servers/rendering/rasterizer.h"

#include "core/local_vector.h"
#include "core/math/geometry_3d.h"
#include "core/math/octree.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/rid_owner.h"
#include "core/self_list.h"
#include "servers/xr/xr_interface.h"

class RenderingServerScene {
public:
	enum {

		MAX_INSTANCE_CULL = 65536,
		MAX_LIGHTS_CULLED = 4096,
		MAX_REFLECTION_PROBES_CULLED = 4096,
		MAX_DECALS_CULLED = 4096,
		MAX_GI_PROBES_CULLED = 4096,
		MAX_ROOM_CULL = 32,
		MAX_LIGHTMAPS_CULLED = 4096,
		MAX_EXTERIOR_PORTALS = 128,
	};

	uint64_t render_pass;

	static RenderingServerScene *singleton;

	/* CAMERA API */

	struct Camera {
		enum Type {
			PERSPECTIVE,
			ORTHOGONAL,
			FRUSTUM
		};
		Type type;
		float fov;
		float znear, zfar;
		float size;
		Vector2 offset;
		uint32_t visible_layers;
		bool vaspect;
		RID env;
		RID effects;

		Transform transform;

		Camera() {
			visible_layers = 0xFFFFFFFF;
			fov = 75;
			type = PERSPECTIVE;
			znear = 0.05;
			zfar = 100;
			size = 1.0;
			offset = Vector2();
			vaspect = false;
		}
	};

	mutable RID_PtrOwner<Camera> camera_owner;

	virtual RID camera_create();
	virtual void camera_set_perspective(RID p_camera, float p_fovy_degrees, float p_z_near, float p_z_far);
	virtual void camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far);
	virtual void camera_set_frustum(RID p_camera, float p_size, Vector2 p_offset, float p_z_near, float p_z_far);
	virtual void camera_set_transform(RID p_camera, const Transform &p_transform);
	virtual void camera_set_cull_mask(RID p_camera, uint32_t p_layers);
	virtual void camera_set_environment(RID p_camera, RID p_env);
	virtual void camera_set_camera_effects(RID p_camera, RID p_fx);
	virtual void camera_set_use_vertical_aspect(RID p_camera, bool p_enable);

	/* SCENARIO API */

	struct Instance;

	struct Scenario {
		RS::ScenarioDebugMode debug;
		RID self;

		Octree<Instance, true> octree;

		List<Instance *> directional_lights;
		RID environment;
		RID fallback_environment;
		RID camera_effects;
		RID reflection_probe_shadow_atlas;
		RID reflection_atlas;

		SelfList<Instance>::List instances;

		LocalVector<RID> dynamic_lights;

		Scenario() { debug = RS::SCENARIO_DEBUG_DISABLED; }
	};

	mutable RID_PtrOwner<Scenario> scenario_owner;

	static void *_instance_pair(void *p_self, OctreeElementID, Instance *p_A, int, OctreeElementID, Instance *p_B, int);
	static void _instance_unpair(void *p_self, OctreeElementID, Instance *p_A, int, OctreeElementID, Instance *p_B, int, void *);

	virtual RID scenario_create();

	virtual void scenario_set_debug(RID p_scenario, RS::ScenarioDebugMode p_debug_mode);
	virtual void scenario_set_environment(RID p_scenario, RID p_environment);
	virtual void scenario_set_camera_effects(RID p_scenario, RID p_fx);
	virtual void scenario_set_fallback_environment(RID p_scenario, RID p_environment);
	virtual void scenario_set_reflection_atlas_size(RID p_scenario, int p_reflection_size, int p_reflection_count);

	/* INSTANCING API */

	struct InstanceBaseData {
		virtual ~InstanceBaseData() {}
	};

	struct Instance : RasterizerScene::InstanceBase {
		RID self;
		//scenario stuff
		OctreeElementID octree_id;
		Scenario *scenario;
		SelfList<Instance> scenario_item;

		//aabb stuff
		bool update_aabb;
		bool update_dependencies;

		SelfList<Instance> update_item;

		AABB *custom_aabb; // <Zylann> would using aabb directly with a bool be better?
		float extra_margin;
		ObjectID object_id;

		float lod_begin;
		float lod_end;
		float lod_begin_hysteresis;
		float lod_end_hysteresis;
		RID lod_instance;

		Vector<Color> lightmap_target_sh; //target is used for incrementally changing the SH over time, this avoids pops in some corner cases and when going interior <-> exterior

		uint64_t last_render_pass;
		uint64_t last_frame_pass;

		uint64_t version; // changes to this, and changes to base increase version

		InstanceBaseData *base_data;

		virtual void dependency_deleted(RID p_dependency) {
			if (p_dependency == base) {
				singleton->instance_set_base(self, RID());
			} else if (p_dependency == skeleton) {
				singleton->instance_attach_skeleton(self, RID());
			} else {
				singleton->_instance_queue_update(this, false, true);
			}
		}

		virtual void dependency_changed(bool p_aabb, bool p_dependencies) {
			singleton->_instance_queue_update(this, p_aabb, p_dependencies);
		}

		Instance() :
				scenario_item(this),
				update_item(this) {
			octree_id = 0;
			scenario = nullptr;

			update_aabb = false;
			update_dependencies = false;

			extra_margin = 0;

			visible = true;

			lod_begin = 0;
			lod_end = 0;
			lod_begin_hysteresis = 0;
			lod_end_hysteresis = 0;

			last_render_pass = 0;
			last_frame_pass = 0;
			version = 1;
			base_data = nullptr;

			custom_aabb = nullptr;
		}

		~Instance() {
			if (base_data) {
				memdelete(base_data);
			}
			if (custom_aabb) {
				memdelete(custom_aabb);
			}
		}
	};

	SelfList<Instance>::List _instance_update_list;
	void _instance_queue_update(Instance *p_instance, bool p_update_aabb, bool p_update_dependencies = false);

	struct InstanceGeometryData : public InstanceBaseData {
		List<Instance *> lighting;
		bool lighting_dirty;
		bool can_cast_shadows;
		bool material_is_animated;

		List<Instance *> decals;
		bool decal_dirty;

		List<Instance *> reflection_probes;
		bool reflection_dirty;

		List<Instance *> gi_probes;
		bool gi_probes_dirty;

		List<Instance *> lightmap_captures;

		InstanceGeometryData() {
			lighting_dirty = false;
			reflection_dirty = true;
			can_cast_shadows = true;
			material_is_animated = true;
			gi_probes_dirty = true;
			decal_dirty = true;
		}
	};

	struct InstanceReflectionProbeData : public InstanceBaseData {
		Instance *owner;

		struct PairInfo {
			List<Instance *>::Element *L; //reflection iterator in geometry
			Instance *geometry;
		};
		List<PairInfo> geometries;

		RID instance;
		bool reflection_dirty;
		SelfList<InstanceReflectionProbeData> update_list;

		int render_step;

		InstanceReflectionProbeData() :
				update_list(this) {
			reflection_dirty = true;
			render_step = -1;
		}
	};

	struct InstanceDecalData : public InstanceBaseData {
		Instance *owner;
		RID instance;

		struct PairInfo {
			List<Instance *>::Element *L; //reflection iterator in geometry
			Instance *geometry;
		};
		List<PairInfo> geometries;

		InstanceDecalData() {
		}
	};

	SelfList<InstanceReflectionProbeData>::List reflection_probe_render_list;

	struct InstanceLightData : public InstanceBaseData {
		struct PairInfo {
			List<Instance *>::Element *L; //light iterator in geometry
			Instance *geometry;
		};

		RID instance;
		uint64_t last_version;
		List<Instance *>::Element *D; // directional light in scenario

		bool shadow_dirty;

		List<PairInfo> geometries;

		Instance *baked_light;

		RS::LightBakeMode bake_mode;
		uint32_t max_sdfgi_cascade = 2;

		uint64_t sdfgi_cascade_light_pass = 0;

		InstanceLightData() {
			bake_mode = RS::LIGHT_BAKE_DISABLED;
			shadow_dirty = true;
			D = nullptr;
			last_version = 0;
			baked_light = nullptr;
		}
	};

	struct InstanceGIProbeData : public InstanceBaseData {
		Instance *owner;

		struct PairInfo {
			List<Instance *>::Element *L; //gi probe iterator in geometry
			Instance *geometry;
		};

		List<PairInfo> geometries;
		List<PairInfo> dynamic_geometries;

		Set<Instance *> lights;

		struct LightCache {
			RS::LightType type;
			Transform transform;
			Color color;
			float energy;
			float bake_energy;
			float radius;
			float attenuation;
			float spot_angle;
			float spot_attenuation;
			bool has_shadow;
		};

		Vector<LightCache> light_cache;
		Vector<RID> light_instances;

		RID probe_instance;

		bool invalid;
		uint32_t base_version;

		SelfList<InstanceGIProbeData> update_element;

		InstanceGIProbeData() :
				update_element(this) {
			invalid = true;
			base_version = 0;
		}
	};

	SelfList<InstanceGIProbeData>::List gi_probe_update_list;

	struct InstanceLightmapData : public InstanceBaseData {
		struct PairInfo {
			List<Instance *>::Element *L; //iterator in geometry
			Instance *geometry;
		};
		List<PairInfo> geometries;

		Set<Instance *> users;

		InstanceLightmapData() {
		}
	};

	Set<Instance *> heightfield_particle_colliders_update_list;

	int instance_cull_count;
	Instance *instance_cull_result[MAX_INSTANCE_CULL];
	Instance *instance_shadow_cull_result[MAX_INSTANCE_CULL]; //used for generating shadowmaps
	Instance *light_cull_result[MAX_LIGHTS_CULLED];
	RID sdfgi_light_cull_result[MAX_LIGHTS_CULLED];
	RID light_instance_cull_result[MAX_LIGHTS_CULLED];
	uint64_t sdfgi_light_cull_pass = 0;
	int light_cull_count;
	int directional_light_count;
	RID reflection_probe_instance_cull_result[MAX_REFLECTION_PROBES_CULLED];
	RID decal_instance_cull_result[MAX_DECALS_CULLED];
	int reflection_probe_cull_count;
	int decal_cull_count;
	RID gi_probe_instance_cull_result[MAX_GI_PROBES_CULLED];
	int gi_probe_cull_count;
	Instance *lightmap_cull_result[MAX_LIGHTS_CULLED];
	int lightmap_cull_count;

	RID_PtrOwner<Instance> instance_owner;

	virtual RID instance_create();

	virtual void instance_set_base(RID p_instance, RID p_base);
	virtual void instance_set_scenario(RID p_instance, RID p_scenario);
	virtual void instance_set_layer_mask(RID p_instance, uint32_t p_mask);
	virtual void instance_set_transform(RID p_instance, const Transform &p_transform);
	virtual void instance_attach_object_instance_id(RID p_instance, ObjectID p_id);
	virtual void instance_set_blend_shape_weight(RID p_instance, int p_shape, float p_weight);
	virtual void instance_set_surface_material(RID p_instance, int p_surface, RID p_material);
	virtual void instance_set_visible(RID p_instance, bool p_visible);

	virtual void instance_set_custom_aabb(RID p_instance, AABB p_aabb);

	virtual void instance_attach_skeleton(RID p_instance, RID p_skeleton);
	virtual void instance_set_exterior(RID p_instance, bool p_enabled);

	virtual void instance_set_extra_visibility_margin(RID p_instance, real_t p_margin);

	// don't use these in a game!
	virtual Vector<ObjectID> instances_cull_aabb(const AABB &p_aabb, RID p_scenario = RID()) const;
	virtual Vector<ObjectID> instances_cull_ray(const Vector3 &p_from, const Vector3 &p_to, RID p_scenario = RID()) const;
	virtual Vector<ObjectID> instances_cull_convex(const Vector<Plane> &p_convex, RID p_scenario = RID()) const;

	virtual void instance_geometry_set_flag(RID p_instance, RS::InstanceFlags p_flags, bool p_enabled);
	virtual void instance_geometry_set_cast_shadows_setting(RID p_instance, RS::ShadowCastingSetting p_shadow_casting_setting);
	virtual void instance_geometry_set_material_override(RID p_instance, RID p_material);

	virtual void instance_geometry_set_draw_range(RID p_instance, float p_min, float p_max, float p_min_margin, float p_max_margin);
	virtual void instance_geometry_set_as_instance_lod(RID p_instance, RID p_as_lod_of_instance);
	virtual void instance_geometry_set_lightmap(RID p_instance, RID p_lightmap, const Rect2 &p_lightmap_uv_scale, int p_slice_index);

	void _update_instance_shader_parameters_from_material(Map<StringName, RasterizerScene::InstanceBase::InstanceShaderParameter> &isparams, const Map<StringName, RasterizerScene::InstanceBase::InstanceShaderParameter> &existing_isparams, RID p_material);

	virtual void instance_geometry_set_shader_parameter(RID p_instance, const StringName &p_parameter, const Variant &p_value);
	virtual void instance_geometry_get_shader_parameter_list(RID p_instance, List<PropertyInfo> *p_parameters) const;
	virtual Variant instance_geometry_get_shader_parameter(RID p_instance, const StringName &p_parameter) const;
	virtual Variant instance_geometry_get_shader_parameter_default_value(RID p_instance, const StringName &p_parameter) const;

	_FORCE_INLINE_ void _update_instance(Instance *p_instance);
	_FORCE_INLINE_ void _update_instance_aabb(Instance *p_instance);
	_FORCE_INLINE_ void _update_dirty_instance(Instance *p_instance);
	_FORCE_INLINE_ void _update_instance_lightmap_captures(Instance *p_instance);

	_FORCE_INLINE_ bool _light_instance_update_shadow(Instance *p_instance, const Transform p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, bool p_cam_vaspect, RID p_shadow_atlas, Scenario *p_scenario);

	RID _render_get_environment(RID p_camera, RID p_scenario);

	bool _render_reflection_probe_step(Instance *p_instance, int p_step);
	void _prepare_scene(const Transform p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, bool p_cam_vaspect, RID p_render_buffers, RID p_environment, uint32_t p_visible_layers, RID p_scenario, RID p_shadow_atlas, RID p_reflection_probe, bool p_using_shadows = true);
	void _render_scene(RID p_render_buffers, const Transform p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, RID p_environment, RID p_force_camera_effects, RID p_scenario, RID p_shadow_atlas, RID p_reflection_probe, int p_reflection_probe_pass);
	void render_empty_scene(RID p_render_buffers, RID p_scenario, RID p_shadow_atlas);

	void render_camera(RID p_render_buffers, RID p_camera, RID p_scenario, Size2 p_viewport_size, RID p_shadow_atlas);
	void render_camera(RID p_render_buffers, Ref<XRInterface> &p_interface, XRInterface::Eyes p_eye, RID p_camera, RID p_scenario, Size2 p_viewport_size, RID p_shadow_atlas);
	void update_dirty_instances();

	void render_particle_colliders();
	void render_probes();

	TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size);

	bool free(RID p_rid);

	RenderingServerScene();
	virtual ~RenderingServerScene();
};

#endif // VISUALSERVERSCENE_H
