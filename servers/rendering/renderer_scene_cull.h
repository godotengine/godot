/*************************************************************************/
/*  renderer_scene_cull.h                                                */
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

#ifndef RENDERING_SERVER_SCENE_CULL_H
#define RENDERING_SERVER_SCENE_CULL_H

#include "core/templates/bin_sorted_array.h"
#include "core/templates/pass_func.h"
#include "servers/rendering/renderer_compositor.h"

#include "core/math/dynamic_bvh.h"
#include "core/math/geometry_3d.h"
#include "core/math/octree.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/local_vector.h"
#include "core/templates/paged_allocator.h"
#include "core/templates/paged_array.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/renderer_scene.h"
#include "servers/rendering/renderer_scene_occlusion_cull.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/xr/xr_interface.h"

class RendererSceneCull : public RendererScene {
public:
	RendererSceneRender *scene_render;

	enum {
		SDFGI_MAX_CASCADES = 8,
		SDFGI_MAX_REGIONS_PER_CASCADE = 3,
		MAX_INSTANCE_PAIRS = 32,
		MAX_UPDATE_SHADOWS = 512
	};

	uint64_t render_pass;

	static RendererSceneCull *singleton;

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

		Transform3D transform;

		Camera() {
			visible_layers = 0xFFFFFFFF;
			fov = 75;
			type = PERSPECTIVE;
			znear = 0.05;
			zfar = 4000;
			size = 1.0;
			offset = Vector2();
			vaspect = false;
		}
	};

	mutable RID_Owner<Camera, true> camera_owner;

	virtual RID camera_allocate();
	virtual void camera_initialize(RID p_rid);

	virtual void camera_set_perspective(RID p_camera, float p_fovy_degrees, float p_z_near, float p_z_far);
	virtual void camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far);
	virtual void camera_set_frustum(RID p_camera, float p_size, Vector2 p_offset, float p_z_near, float p_z_far);
	virtual void camera_set_transform(RID p_camera, const Transform3D &p_transform);
	virtual void camera_set_cull_mask(RID p_camera, uint32_t p_layers);
	virtual void camera_set_environment(RID p_camera, RID p_env);
	virtual void camera_set_camera_effects(RID p_camera, RID p_fx);
	virtual void camera_set_use_vertical_aspect(RID p_camera, bool p_enable);
	virtual bool is_camera(RID p_camera) const;

	/* OCCLUDER API */

	virtual RID occluder_allocate();
	virtual void occluder_initialize(RID p_occluder);
	virtual void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices);

	/* VISIBILITY NOTIFIER API */

	RendererSceneOcclusionCull *dummy_occlusion_culling;

	/* SCENARIO API */

	struct Instance;

	struct PlaneSign {
		_ALWAYS_INLINE_ PlaneSign() {}
		_ALWAYS_INLINE_ PlaneSign(const Plane &p_plane) {
			if (p_plane.normal.x > 0) {
				signs[0] = 0;
			} else {
				signs[0] = 3;
			}
			if (p_plane.normal.y > 0) {
				signs[1] = 1;
			} else {
				signs[1] = 4;
			}
			if (p_plane.normal.z > 0) {
				signs[2] = 2;
			} else {
				signs[2] = 5;
			}
		}

		uint32_t signs[3];
	};

	struct Frustum {
		Vector<Plane> planes;
		Vector<PlaneSign> plane_signs;
		const Plane *planes_ptr;
		const PlaneSign *plane_signs_ptr;
		uint32_t plane_count;

		_ALWAYS_INLINE_ Frustum() {}
		_ALWAYS_INLINE_ Frustum(const Frustum &p_frustum) {
			planes = p_frustum.planes;
			plane_signs = p_frustum.plane_signs;

			planes_ptr = planes.ptr();
			plane_signs_ptr = plane_signs.ptr();
			plane_count = p_frustum.plane_count;
		}
		_ALWAYS_INLINE_ void operator=(const Frustum &p_frustum) {
			planes = p_frustum.planes;
			plane_signs = p_frustum.plane_signs;

			planes_ptr = planes.ptr();
			plane_signs_ptr = plane_signs.ptr();
			plane_count = p_frustum.plane_count;
		}
		_ALWAYS_INLINE_ Frustum(const Vector<Plane> &p_planes) {
			planes = p_planes;
			planes_ptr = planes.ptrw();
			plane_count = planes.size();
			for (int i = 0; i < planes.size(); i++) {
				PlaneSign ps(p_planes[i]);
				plane_signs.push_back(ps);
			}

			plane_signs_ptr = plane_signs.ptr();
		}
	};

	struct InstanceBounds {
		// Efficiently store instance bounds.
		// Because bounds checking is performed first,
		// keep it separated from data.

		real_t bounds[6];
		_ALWAYS_INLINE_ InstanceBounds() {}

		_ALWAYS_INLINE_ InstanceBounds(const AABB &p_aabb) {
			bounds[0] = p_aabb.position.x;
			bounds[1] = p_aabb.position.y;
			bounds[2] = p_aabb.position.z;
			bounds[3] = p_aabb.position.x + p_aabb.size.x;
			bounds[4] = p_aabb.position.y + p_aabb.size.y;
			bounds[5] = p_aabb.position.z + p_aabb.size.z;
		}
		_ALWAYS_INLINE_ bool in_frustum(const Frustum &p_frustum) const {
			// This is not a full SAT check and the possibility of false positives exist,
			// but the tradeoff vs performance is still very good.

			for (uint32_t i = 0; i < p_frustum.plane_count; i++) {
				Vector3 min(
						bounds[p_frustum.plane_signs_ptr[i].signs[0]],
						bounds[p_frustum.plane_signs_ptr[i].signs[1]],
						bounds[p_frustum.plane_signs_ptr[i].signs[2]]);

				if (p_frustum.planes_ptr[i].distance_to(min) >= 0.0) {
					return false;
				}
			}

			return true;
		}
		_ALWAYS_INLINE_ bool in_aabb(const AABB &p_aabb) const {
			Vector3 end = p_aabb.position + p_aabb.size;

			if (bounds[0] >= end.x) {
				return false;
			}
			if (bounds[3] <= p_aabb.position.x) {
				return false;
			}
			if (bounds[1] >= end.y) {
				return false;
			}
			if (bounds[4] <= p_aabb.position.y) {
				return false;
			}
			if (bounds[2] >= end.z) {
				return false;
			}
			if (bounds[5] <= p_aabb.position.z) {
				return false;
			}

			return true;
		}
	};

	struct InstanceVisibilityNotifierData;

	struct InstanceData {
		// Store instance pointer as well as common instance processing information,
		// to make processing more cache friendly.
		enum Flags {
			FLAG_BASE_TYPE_MASK = 0xFF,
			FLAG_CAST_SHADOWS = (1 << 8),
			FLAG_CAST_SHADOWS_ONLY = (1 << 9),
			FLAG_REDRAW_IF_VISIBLE = (1 << 10),
			FLAG_GEOM_LIGHTING_DIRTY = (1 << 11),
			FLAG_GEOM_REFLECTION_DIRTY = (1 << 12),
			FLAG_GEOM_DECAL_DIRTY = (1 << 13),
			FLAG_GEOM_VOXEL_GI_DIRTY = (1 << 14),
			FLAG_LIGHTMAP_CAPTURE = (1 << 15),
			FLAG_USES_BAKED_LIGHT = (1 << 16),
			FLAG_USES_MESH_INSTANCE = (1 << 17),
			FLAG_REFLECTION_PROBE_DIRTY = (1 << 18),
			FLAG_IGNORE_OCCLUSION_CULLING = (1 << 19),
			FLAG_VISIBILITY_DEPENDENCY_NEEDS_CHECK = (3 << 20), // 2 bits, overlaps with the other vis. dependency flags
			FLAG_VISIBILITY_DEPENDENCY_HIDDEN_CLOSE_RANGE = (1 << 20),
			FLAG_VISIBILITY_DEPENDENCY_HIDDEN = (1 << 21),
			FLAG_VISIBILITY_DEPENDENCY_FADE_CHILDREN = (1 << 22),
			FLAG_GEOM_PROJECTOR_SOFTSHADOW_DIRTY = (1 << 23),
			FLAG_IGNORE_ALL_CULLING = (1 << 24),
		};

		uint32_t flags = 0;
		uint32_t layer_mask = 0; //for fast layer-mask discard
		RID base_rid;
		union {
			uint64_t instance_data_rid;
			RendererSceneRender::GeometryInstance *instance_geometry;
			InstanceVisibilityNotifierData *visibility_notifier;
		};
		Instance *instance = nullptr;
		int32_t parent_array_index = -1;
		int32_t visibility_index = -1;
	};

	struct InstanceVisibilityData {
		uint64_t viewport_state = 0;
		int32_t array_index = -1;
		RS::VisibilityRangeFadeMode fade_mode = RS::VISIBILITY_RANGE_FADE_DISABLED;
		Vector3 position;
		Instance *instance = nullptr;
		float range_begin = 0.0f;
		float range_end = 0.0f;
		float range_begin_margin = 0.0f;
		float range_end_margin = 0.0f;
		float children_fade_alpha = 1.0f;
	};

	class VisibilityArray : public BinSortedArray<InstanceVisibilityData> {
		_FORCE_INLINE_ virtual void _update_idx(InstanceVisibilityData &r_element, uint64_t p_idx) {
			r_element.instance->visibility_index = p_idx;
			if (r_element.instance->scenario && r_element.instance->array_index != -1) {
				r_element.instance->scenario->instance_data[r_element.instance->array_index].visibility_index = p_idx;
			}
		}
	};

	PagedArrayPool<InstanceBounds> instance_aabb_page_pool;
	PagedArrayPool<InstanceData> instance_data_page_pool;
	PagedArrayPool<InstanceVisibilityData> instance_visibility_data_page_pool;

	struct Scenario {
		enum IndexerType {
			INDEXER_GEOMETRY, //for geometry
			INDEXER_VOLUMES, //for everything else
			INDEXER_MAX
		};

		DynamicBVH indexers[INDEXER_MAX];

		RID self;

		List<Instance *> directional_lights;
		RID environment;
		RID fallback_environment;
		RID camera_effects;
		RID reflection_probe_shadow_atlas;
		RID reflection_atlas;
		uint64_t used_viewport_visibility_bits;
		Map<RID, uint64_t> viewport_visibility_masks;

		SelfList<Instance>::List instances;

		LocalVector<RID> dynamic_lights;

		PagedArray<InstanceBounds> instance_aabbs;
		PagedArray<InstanceData> instance_data;
		VisibilityArray instance_visibility;

		Scenario() {
			indexers[INDEXER_GEOMETRY].set_index(INDEXER_GEOMETRY);
			indexers[INDEXER_VOLUMES].set_index(INDEXER_VOLUMES);
			used_viewport_visibility_bits = 0;
		}
	};

	int indexer_update_iterations = 0;

	mutable RID_Owner<Scenario, true> scenario_owner;

	static void _instance_pair(Instance *p_A, Instance *p_B);
	static void _instance_unpair(Instance *p_A, Instance *p_B);

	void _instance_update_mesh_instance(Instance *p_instance);

	virtual RID scenario_allocate();
	virtual void scenario_initialize(RID p_rid);

	virtual void scenario_set_environment(RID p_scenario, RID p_environment);
	virtual void scenario_set_camera_effects(RID p_scenario, RID p_fx);
	virtual void scenario_set_fallback_environment(RID p_scenario, RID p_environment);
	virtual void scenario_set_reflection_atlas_size(RID p_scenario, int p_reflection_size, int p_reflection_count);
	virtual bool is_scenario(RID p_scenario) const;
	virtual RID scenario_get_environment(RID p_scenario);
	virtual void scenario_add_viewport_visibility_mask(RID p_scenario, RID p_viewport);
	virtual void scenario_remove_viewport_visibility_mask(RID p_scenario, RID p_viewport);

	/* INSTANCING API */

	struct InstancePair {
		Instance *a;
		Instance *b;
		SelfList<InstancePair> list_a;
		SelfList<InstancePair> list_b;
		InstancePair() :
				list_a(this), list_b(this) {}
	};

	PagedAllocator<InstancePair> pair_allocator;

	struct InstanceBaseData {
		virtual ~InstanceBaseData() {}
	};

	struct Instance {
		RS::InstanceType base_type;
		RID base;

		RID skeleton;
		RID material_override;

		RID mesh_instance; //only used for meshes and when skeleton/blendshapes exist

		Transform3D transform;

		float lod_bias;

		bool ignore_occlusion_culling;
		bool ignore_all_culling;

		Vector<RID> materials;

		RS::ShadowCastingSetting cast_shadows;

		uint32_t layer_mask;
		//fit in 32 bits
		bool mirror : 8;
		bool receive_shadows : 8;
		bool visible : 8;
		bool baked_light : 2; //this flag is only to know if it actually did use baked light
		bool dynamic_gi : 2; //same above for dynamic objects
		bool redraw_if_visible : 4;

		Instance *lightmap;
		Rect2 lightmap_uv_scale;
		int lightmap_slice_index;
		uint32_t lightmap_cull_index;
		Vector<Color> lightmap_sh; //spherical harmonic

		AABB aabb;
		AABB transformed_aabb;
		AABB prev_transformed_aabb;

		struct InstanceShaderParameter {
			int32_t index = -1;
			Variant value;
			Variant default_value;
			PropertyInfo info;
		};

		Map<StringName, InstanceShaderParameter> instance_shader_parameters;
		bool instance_allocated_shader_parameters = false;
		int32_t instance_allocated_shader_parameters_offset = -1;

		//

		RID self;
		//scenario stuff
		DynamicBVH::ID indexer_id;
		int32_t array_index = -1;
		int32_t visibility_index = -1;
		float visibility_range_begin = 0.0f;
		float visibility_range_end = 0.0f;
		float visibility_range_begin_margin = 0.0f;
		float visibility_range_end_margin = 0.0f;
		RS::VisibilityRangeFadeMode visibility_range_fade_mode = RS::VISIBILITY_RANGE_FADE_DISABLED;
		Instance *visibility_parent = nullptr;
		Set<Instance *> visibility_dependencies;
		uint32_t visibility_dependencies_depth = 0;
		float transparency = 0.0f;
		Scenario *scenario = nullptr;
		SelfList<Instance> scenario_item;

		//aabb stuff
		bool update_aabb;
		bool update_dependencies;

		SelfList<Instance> update_item;

		AABB *custom_aabb; // <Zylann> would using aabb directly with a bool be better?
		float extra_margin;
		ObjectID object_id;

		Vector<Color> lightmap_target_sh; //target is used for incrementally changing the SH over time, this avoids pops in some corner cases and when going interior <-> exterior

		uint64_t last_frame_pass;

		uint64_t version; // changes to this, and changes to base increase version

		InstanceBaseData *base_data;

		SelfList<InstancePair>::List pairs;
		uint64_t pair_check;

		RendererStorage::DependencyTracker dependency_tracker;

		static void dependency_changed(RendererStorage::DependencyChangedNotification p_notification, RendererStorage::DependencyTracker *tracker) {
			Instance *instance = (Instance *)tracker->userdata;
			switch (p_notification) {
				case RendererStorage::DEPENDENCY_CHANGED_SKELETON_DATA:
				case RendererStorage::DEPENDENCY_CHANGED_AABB: {
					singleton->_instance_queue_update(instance, true, false);

				} break;
				case RendererStorage::DEPENDENCY_CHANGED_MATERIAL: {
					singleton->_instance_queue_update(instance, false, true);
				} break;
				case RendererStorage::DEPENDENCY_CHANGED_MESH:
				case RendererStorage::DEPENDENCY_CHANGED_PARTICLES:
				case RendererStorage::DEPENDENCY_CHANGED_MULTIMESH:
				case RendererStorage::DEPENDENCY_CHANGED_DECAL:
				case RendererStorage::DEPENDENCY_CHANGED_LIGHT:
				case RendererStorage::DEPENDENCY_CHANGED_REFLECTION_PROBE: {
					singleton->_instance_queue_update(instance, true, true);
				} break;
				case RendererStorage::DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES:
				case RendererStorage::DEPENDENCY_CHANGED_SKELETON_BONES: {
					//ignored
				} break;
				case RendererStorage::DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR: {
					//requires repairing
					if (instance->indexer_id.is_valid()) {
						singleton->_unpair_instance(instance);
						singleton->_instance_queue_update(instance, true, true);
					}

				} break;
			}
		}

		static void dependency_deleted(const RID &p_dependency, RendererStorage::DependencyTracker *tracker) {
			Instance *instance = (Instance *)tracker->userdata;

			if (p_dependency == instance->base) {
				singleton->instance_set_base(instance->self, RID());
			} else if (p_dependency == instance->skeleton) {
				singleton->instance_attach_skeleton(instance->self, RID());
			} else {
				singleton->_instance_queue_update(instance, false, true);
			}
		}

		Instance() :
				scenario_item(this),
				update_item(this) {
			base_type = RS::INSTANCE_NONE;
			cast_shadows = RS::SHADOW_CASTING_SETTING_ON;
			receive_shadows = true;
			visible = true;
			layer_mask = 1;
			baked_light = false;
			dynamic_gi = false;
			redraw_if_visible = false;
			lightmap_slice_index = 0;
			lightmap = nullptr;
			lightmap_cull_index = 0;
			lod_bias = 1.0;
			ignore_occlusion_culling = false;
			ignore_all_culling = false;

			scenario = nullptr;

			update_aabb = false;
			update_dependencies = false;

			extra_margin = 0;

			visible = true;

			visibility_range_begin = 0;
			visibility_range_end = 0;
			visibility_range_begin_margin = 0;
			visibility_range_end_margin = 0;

			last_frame_pass = 0;
			version = 1;
			base_data = nullptr;

			custom_aabb = nullptr;

			pair_check = 0;
			array_index = -1;

			dependency_tracker.userdata = this;
			dependency_tracker.changed_callback = dependency_changed;
			dependency_tracker.deleted_callback = dependency_deleted;
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
		RendererSceneRender::GeometryInstance *geometry_instance = nullptr;
		Set<Instance *> lights;
		bool can_cast_shadows;
		bool material_is_animated;
		uint32_t projector_count = 0;
		uint32_t softshadow_count = 0;

		Set<Instance *> decals;
		Set<Instance *> reflection_probes;
		Set<Instance *> voxel_gi_instances;
		Set<Instance *> lightmap_captures;

		InstanceGeometryData() {
			can_cast_shadows = true;
			material_is_animated = true;
		}
	};

	struct InstanceReflectionProbeData : public InstanceBaseData {
		Instance *owner;

		Set<Instance *> geometries;

		RID instance;
		SelfList<InstanceReflectionProbeData> update_list;

		int render_step;

		InstanceReflectionProbeData() :
				update_list(this) {
			render_step = -1;
		}
	};

	struct InstanceDecalData : public InstanceBaseData {
		Instance *owner;
		RID instance;

		Set<Instance *> geometries;

		InstanceDecalData() {
		}
	};

	SelfList<InstanceReflectionProbeData>::List reflection_probe_render_list;

	struct InstanceParticlesCollisionData : public InstanceBaseData {
		RID instance;
	};

	struct InstanceFogVolumeData : public InstanceBaseData {
		RID instance;
		bool is_global;
	};

	struct InstanceVisibilityNotifierData : public InstanceBaseData {
		bool just_visible = false;
		uint64_t visible_in_frame = 0;
		RID base;
		SelfList<InstanceVisibilityNotifierData> list_element;
		InstanceVisibilityNotifierData() :
				list_element(this) {}
	};

	SpinLock visible_notifier_list_lock;
	SelfList<InstanceVisibilityNotifierData>::List visible_notifier_list;

	struct InstanceLightData : public InstanceBaseData {
		RID instance;
		uint64_t last_version;
		List<Instance *>::Element *D; // directional light in scenario

		bool shadow_dirty;
		bool uses_projector = false;
		bool uses_softshadow = false;

		Set<Instance *> geometries;

		Instance *baked_light;

		RS::LightBakeMode bake_mode;
		uint32_t max_sdfgi_cascade = 2;

		InstanceLightData() {
			bake_mode = RS::LIGHT_BAKE_DISABLED;
			shadow_dirty = true;
			D = nullptr;
			last_version = 0;
			baked_light = nullptr;
		}
	};

	struct InstanceVoxelGIData : public InstanceBaseData {
		Instance *owner;

		Set<Instance *> geometries;
		Set<Instance *> dynamic_geometries;

		Set<Instance *> lights;

		struct LightCache {
			RS::LightType type;
			Transform3D transform;
			Color color;
			float energy;
			float bake_energy;
			float radius;
			float attenuation;
			float spot_angle;
			float spot_attenuation;
			bool has_shadow;
			bool sky_only;
		};

		Vector<LightCache> light_cache;
		Vector<RID> light_instances;

		RID probe_instance;

		bool invalid;
		uint32_t base_version;

		SelfList<InstanceVoxelGIData> update_element;

		InstanceVoxelGIData() :
				update_element(this) {
			invalid = true;
			base_version = 0;
		}
	};

	SelfList<InstanceVoxelGIData>::List voxel_gi_update_list;

	struct InstanceLightmapData : public InstanceBaseData {
		RID instance;
		Set<Instance *> geometries;
		Set<Instance *> users;

		InstanceLightmapData() {
		}
	};

	uint64_t pair_pass = 1;

	struct PairInstances {
		Instance *instance = nullptr;
		PagedAllocator<InstancePair> *pair_allocator = nullptr;
		SelfList<InstancePair>::List pairs_found;
		DynamicBVH *bvh = nullptr;
		DynamicBVH *bvh2 = nullptr; //some may need to cull in two
		uint32_t pair_mask;
		uint64_t pair_pass;

		_FORCE_INLINE_ bool operator()(void *p_data) {
			Instance *p_instance = (Instance *)p_data;

			if (instance != p_instance && instance->transformed_aabb.intersects(p_instance->transformed_aabb) && (pair_mask & (1 << p_instance->base_type))) {
				//test is more coarse in indexer
				p_instance->pair_check = pair_pass;
				InstancePair *pair = pair_allocator->alloc();
				pair->a = instance;
				pair->b = p_instance;
				pairs_found.add(&pair->list_a);
			}
			return false;
		}

		void pair() {
			if (bvh) {
				bvh->aabb_query(instance->transformed_aabb, *this);
			}
			if (bvh2) {
				bvh2->aabb_query(instance->transformed_aabb, *this);
			}
			while (instance->pairs.first()) {
				InstancePair *pair = instance->pairs.first()->self();
				Instance *other_instance = instance == pair->a ? pair->b : pair->a;
				if (other_instance->pair_check != pair_pass) {
					//unpaired
					_instance_unpair(instance, other_instance);
				} else {
					//kept
					other_instance->pair_check = 0; // if kept, then put pair check to zero, so we can distinguish with the newly added ones
				}

				pair_allocator->free(pair);
			}
			while (pairs_found.first()) {
				InstancePair *pair = pairs_found.first()->self();
				pairs_found.remove(pairs_found.first());

				if (pair->b->pair_check == pair_pass) {
					//paired
					_instance_pair(instance, pair->b);
				}
				pair->a->pairs.add(&pair->list_a);
				pair->b->pairs.add(&pair->list_b);
			}
		}
	};

	Set<Instance *> heightfield_particle_colliders_update_list;

	PagedArrayPool<Instance *> instance_cull_page_pool;
	PagedArrayPool<RendererSceneRender::GeometryInstance *> geometry_instance_cull_page_pool;
	PagedArrayPool<RID> rid_cull_page_pool;

	PagedArray<Instance *> instance_cull_result;
	PagedArray<Instance *> instance_shadow_cull_result;

	struct InstanceCullResult {
		PagedArray<RendererSceneRender::GeometryInstance *> geometry_instances;
		PagedArray<Instance *> lights;
		PagedArray<RID> light_instances;
		PagedArray<RID> lightmaps;
		PagedArray<RID> reflections;
		PagedArray<RID> decals;
		PagedArray<RID> voxel_gi_instances;
		PagedArray<RID> mesh_instances;
		PagedArray<RID> fog_volumes;

		struct DirectionalShadow {
			PagedArray<RendererSceneRender::GeometryInstance *> cascade_geometry_instances[RendererSceneRender::MAX_DIRECTIONAL_LIGHT_CASCADES];
		} directional_shadows[RendererSceneRender::MAX_DIRECTIONAL_LIGHTS];

		PagedArray<RendererSceneRender::GeometryInstance *> sdfgi_region_geometry_instances[SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE];
		PagedArray<RID> sdfgi_cascade_lights[SDFGI_MAX_CASCADES];

		void clear() {
			geometry_instances.clear();
			lights.clear();
			light_instances.clear();
			lightmaps.clear();
			reflections.clear();
			decals.clear();
			voxel_gi_instances.clear();
			mesh_instances.clear();
			fog_volumes.clear();
			for (int i = 0; i < RendererSceneRender::MAX_DIRECTIONAL_LIGHTS; i++) {
				for (int j = 0; j < RendererSceneRender::MAX_DIRECTIONAL_LIGHT_CASCADES; j++) {
					directional_shadows[i].cascade_geometry_instances[j].clear();
				}
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE; i++) {
				sdfgi_region_geometry_instances[i].clear();
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES; i++) {
				sdfgi_cascade_lights[i].clear();
			}
		}

		void reset() {
			geometry_instances.reset();
			lights.reset();
			light_instances.reset();
			lightmaps.reset();
			reflections.reset();
			decals.reset();
			voxel_gi_instances.reset();
			mesh_instances.reset();
			fog_volumes.reset();
			for (int i = 0; i < RendererSceneRender::MAX_DIRECTIONAL_LIGHTS; i++) {
				for (int j = 0; j < RendererSceneRender::MAX_DIRECTIONAL_LIGHT_CASCADES; j++) {
					directional_shadows[i].cascade_geometry_instances[j].reset();
				}
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE; i++) {
				sdfgi_region_geometry_instances[i].reset();
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES; i++) {
				sdfgi_cascade_lights[i].reset();
			}
		}

		void append_from(InstanceCullResult &p_cull_result) {
			geometry_instances.merge_unordered(p_cull_result.geometry_instances);
			lights.merge_unordered(p_cull_result.lights);
			light_instances.merge_unordered(p_cull_result.light_instances);
			lightmaps.merge_unordered(p_cull_result.lightmaps);
			reflections.merge_unordered(p_cull_result.reflections);
			decals.merge_unordered(p_cull_result.decals);
			voxel_gi_instances.merge_unordered(p_cull_result.voxel_gi_instances);
			mesh_instances.merge_unordered(p_cull_result.mesh_instances);
			fog_volumes.merge_unordered(p_cull_result.fog_volumes);

			for (int i = 0; i < RendererSceneRender::MAX_DIRECTIONAL_LIGHTS; i++) {
				for (int j = 0; j < RendererSceneRender::MAX_DIRECTIONAL_LIGHT_CASCADES; j++) {
					directional_shadows[i].cascade_geometry_instances[j].merge_unordered(p_cull_result.directional_shadows[i].cascade_geometry_instances[j]);
				}
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE; i++) {
				sdfgi_region_geometry_instances[i].merge_unordered(p_cull_result.sdfgi_region_geometry_instances[i]);
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES; i++) {
				sdfgi_cascade_lights[i].merge_unordered(p_cull_result.sdfgi_cascade_lights[i]);
			}
		}

		void init(PagedArrayPool<RID> *p_rid_pool, PagedArrayPool<RendererSceneRender::GeometryInstance *> *p_geometry_instance_pool, PagedArrayPool<Instance *> *p_instance_pool) {
			geometry_instances.set_page_pool(p_geometry_instance_pool);
			light_instances.set_page_pool(p_rid_pool);
			lights.set_page_pool(p_instance_pool);
			lightmaps.set_page_pool(p_rid_pool);
			reflections.set_page_pool(p_rid_pool);
			decals.set_page_pool(p_rid_pool);
			voxel_gi_instances.set_page_pool(p_rid_pool);
			mesh_instances.set_page_pool(p_rid_pool);
			fog_volumes.set_page_pool(p_rid_pool);
			for (int i = 0; i < RendererSceneRender::MAX_DIRECTIONAL_LIGHTS; i++) {
				for (int j = 0; j < RendererSceneRender::MAX_DIRECTIONAL_LIGHT_CASCADES; j++) {
					directional_shadows[i].cascade_geometry_instances[j].set_page_pool(p_geometry_instance_pool);
				}
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE; i++) {
				sdfgi_region_geometry_instances[i].set_page_pool(p_geometry_instance_pool);
			}

			for (int i = 0; i < SDFGI_MAX_CASCADES; i++) {
				sdfgi_cascade_lights[i].set_page_pool(p_rid_pool);
			}
		}
	};

	InstanceCullResult scene_cull_result;
	LocalVector<InstanceCullResult> scene_cull_result_threads;

	RendererSceneRender::RenderShadowData render_shadow_data[MAX_UPDATE_SHADOWS];
	uint32_t max_shadows_used = 0;

	RendererSceneRender::RenderSDFGIData render_sdfgi_data[SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE];
	RendererSceneRender::RenderSDFGIUpdateData sdfgi_update_data;

	uint32_t thread_cull_threshold = 200;

	RID_Owner<Instance, true> instance_owner;

	uint32_t geometry_instance_pair_mask; // used in traditional forward, unnecessary on clustered

	virtual RID instance_allocate();
	virtual void instance_initialize(RID p_rid);

	virtual void instance_set_base(RID p_instance, RID p_base);
	virtual void instance_set_scenario(RID p_instance, RID p_scenario);
	virtual void instance_set_layer_mask(RID p_instance, uint32_t p_mask);
	virtual void instance_set_transform(RID p_instance, const Transform3D &p_transform);
	virtual void instance_attach_object_instance_id(RID p_instance, ObjectID p_id);
	virtual void instance_set_blend_shape_weight(RID p_instance, int p_shape, float p_weight);
	virtual void instance_set_surface_override_material(RID p_instance, int p_surface, RID p_material);
	virtual void instance_set_visible(RID p_instance, bool p_visible);
	virtual void instance_geometry_set_transparency(RID p_instance, float p_transparency);

	virtual void instance_set_custom_aabb(RID p_instance, AABB p_aabb);

	virtual void instance_attach_skeleton(RID p_instance, RID p_skeleton);

	virtual void instance_set_extra_visibility_margin(RID p_instance, real_t p_margin);

	virtual void instance_set_visibility_parent(RID p_instance, RID p_parent_instance);

	virtual void instance_set_ignore_culling(RID p_instance, bool p_enabled);

	bool _update_instance_visibility_depth(Instance *p_instance);
	void _update_instance_visibility_dependencies(Instance *p_instance);

	// don't use these in a game!
	virtual Vector<ObjectID> instances_cull_aabb(const AABB &p_aabb, RID p_scenario = RID()) const;
	virtual Vector<ObjectID> instances_cull_ray(const Vector3 &p_from, const Vector3 &p_to, RID p_scenario = RID()) const;
	virtual Vector<ObjectID> instances_cull_convex(const Vector<Plane> &p_convex, RID p_scenario = RID()) const;

	virtual void instance_geometry_set_flag(RID p_instance, RS::InstanceFlags p_flags, bool p_enabled);
	virtual void instance_geometry_set_cast_shadows_setting(RID p_instance, RS::ShadowCastingSetting p_shadow_casting_setting);
	virtual void instance_geometry_set_material_override(RID p_instance, RID p_material);

	virtual void instance_geometry_set_visibility_range(RID p_instance, float p_min, float p_max, float p_min_margin, float p_max_margin, RS::VisibilityRangeFadeMode p_fade_mode);

	virtual void instance_geometry_set_lightmap(RID p_instance, RID p_lightmap, const Rect2 &p_lightmap_uv_scale, int p_slice_index);
	virtual void instance_geometry_set_lod_bias(RID p_instance, float p_lod_bias);

	void _update_instance_shader_parameters_from_material(Map<StringName, Instance::InstanceShaderParameter> &isparams, const Map<StringName, Instance::InstanceShaderParameter> &existing_isparams, RID p_material);

	virtual void instance_geometry_set_shader_parameter(RID p_instance, const StringName &p_parameter, const Variant &p_value);
	virtual void instance_geometry_get_shader_parameter_list(RID p_instance, List<PropertyInfo> *p_parameters) const;
	virtual Variant instance_geometry_get_shader_parameter(RID p_instance, const StringName &p_parameter) const;
	virtual Variant instance_geometry_get_shader_parameter_default_value(RID p_instance, const StringName &p_parameter) const;

	_FORCE_INLINE_ void _update_instance(Instance *p_instance);
	_FORCE_INLINE_ void _update_instance_aabb(Instance *p_instance);
	_FORCE_INLINE_ void _update_dirty_instance(Instance *p_instance);
	_FORCE_INLINE_ void _update_instance_lightmap_captures(Instance *p_instance);
	void _unpair_instance(Instance *p_instance);

	void _light_instance_setup_directional_shadow(int p_shadow_index, Instance *p_instance, const Transform3D p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, bool p_cam_vaspect);

	_FORCE_INLINE_ bool _light_instance_update_shadow(Instance *p_instance, const Transform3D p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, bool p_cam_vaspect, RID p_shadow_atlas, Scenario *p_scenario, float p_scren_lod_threshold);

	RID _render_get_environment(RID p_camera, RID p_scenario);

	struct Cull {
		struct Shadow {
			RID light_instance;
			struct Cascade {
				Frustum frustum;

				CameraMatrix projection;
				Transform3D transform;
				real_t zfar;
				real_t split;
				real_t shadow_texel_size;
				real_t bias_scale;
				real_t range_begin;
				Vector2 uv_scale;

			} cascades[RendererSceneRender::MAX_DIRECTIONAL_LIGHT_CASCADES]; //max 4 cascades
			uint32_t cascade_count;

		} shadows[RendererSceneRender::MAX_DIRECTIONAL_LIGHTS];

		uint32_t shadow_count;

		struct SDFGI {
			//have arrays here because SDFGI functions expects this, plus regions can have areas
			AABB region_aabb[SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE]; //max 3 regions per cascade
			uint32_t region_cascade[SDFGI_MAX_CASCADES * SDFGI_MAX_REGIONS_PER_CASCADE]; //max 3 regions per cascade
			uint32_t region_count = 0;

			uint32_t cascade_light_index[SDFGI_MAX_CASCADES];
			uint32_t cascade_light_count = 0;

		} sdfgi;

		SpinLock lock;

		Frustum frustum;
	} cull;

	struct VisibilityCullData {
		uint64_t viewport_mask;
		Scenario *scenario;
		Vector3 camera_position;
		uint32_t cull_offset;
		uint32_t cull_count;
	};

	void _visibility_cull_threaded(uint32_t p_thread, VisibilityCullData *cull_data);
	void _visibility_cull(const VisibilityCullData &cull_data, uint64_t p_from, uint64_t p_to);
	template <bool p_fade_check>
	_FORCE_INLINE_ int _visibility_range_check(InstanceVisibilityData &r_vis_data, const Vector3 &p_camera_pos, uint64_t p_viewport_mask);

	struct CullData {
		Cull *cull;
		Scenario *scenario;
		RID shadow_atlas;
		Transform3D cam_transform;
		uint32_t visible_layers;
		Instance *render_reflection_probe;
		const RendererSceneOcclusionCull::HZBuffer *occlusion_buffer;
		const CameraMatrix *camera_matrix;
		uint64_t visibility_viewport_mask;
	};

	void _scene_cull_threaded(uint32_t p_thread, CullData *cull_data);
	void _scene_cull(CullData &cull_data, InstanceCullResult &cull_result, uint64_t p_from, uint64_t p_to);
	_FORCE_INLINE_ bool _visibility_parent_check(const CullData &p_cull_data, const InstanceData &p_instance_data);

	bool _render_reflection_probe_step(Instance *p_instance, int p_step);
	void _render_scene(const RendererSceneRender::CameraData *p_camera_data, RID p_render_buffers, RID p_environment, RID p_force_camera_effects, uint32_t p_visible_layers, RID p_scenario, RID p_viewport, RID p_shadow_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_lod_threshold, bool p_using_shadows = true, RenderInfo *r_render_info = nullptr);
	void render_empty_scene(RID p_render_buffers, RID p_scenario, RID p_shadow_atlas);

	void render_camera(RID p_render_buffers, RID p_camera, RID p_scenario, RID p_viewport, Size2 p_viewport_size, float p_screen_lod_threshold, RID p_shadow_atlas, Ref<XRInterface> &p_xr_interface, RendererScene::RenderInfo *r_render_info = nullptr);
	void update_dirty_instances();

	void render_particle_colliders();
	virtual void render_probes();

	TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size);

	//pass to scene render

	/* ENVIRONMENT API */

#ifdef PASSBASE
#undef PASSBASE
#endif

#define PASSBASE scene_render

	PASS2(directional_shadow_atlas_set_size, int, bool)
	PASS1(voxel_gi_set_quality, RS::VoxelGIQuality)

	/* SKY API */

	PASS0R(RID, sky_allocate)
	PASS1(sky_initialize, RID)

	PASS2(sky_set_radiance_size, RID, int)
	PASS2(sky_set_mode, RID, RS::SkyMode)
	PASS2(sky_set_material, RID, RID)
	PASS4R(Ref<Image>, sky_bake_panorama, RID, float, bool, const Size2i &)

	PASS0R(RID, environment_allocate)
	PASS1(environment_initialize, RID)

	PASS1RC(bool, is_environment, RID)

	PASS2(environment_set_background, RID, RS::EnvironmentBG)
	PASS2(environment_set_sky, RID, RID)
	PASS2(environment_set_sky_custom_fov, RID, float)
	PASS2(environment_set_sky_orientation, RID, const Basis &)
	PASS2(environment_set_bg_color, RID, const Color &)
	PASS2(environment_set_bg_energy, RID, float)
	PASS2(environment_set_canvas_max_layer, RID, int)
	PASS6(environment_set_ambient_light, RID, const Color &, RS::EnvironmentAmbientSource, float, float, RS::EnvironmentReflectionSource)

	PASS6(environment_set_ssr, RID, bool, int, float, float, float)
	PASS1(environment_set_ssr_roughness_quality, RS::EnvironmentSSRRoughnessQuality)

	PASS10(environment_set_ssao, RID, bool, float, float, float, float, float, float, float, float)
	PASS6(environment_set_ssao_quality, RS::EnvironmentSSAOQuality, bool, float, int, float, float)

	PASS11(environment_set_glow, RID, bool, Vector<float>, float, float, float, float, RS::EnvironmentGlowBlendMode, float, float, float)
	PASS1(environment_glow_set_use_bicubic_upscale, bool)
	PASS1(environment_glow_set_use_high_quality, bool)

	PASS9(environment_set_tonemap, RID, RS::EnvironmentToneMapper, float, float, bool, float, float, float, float)

	PASS7(environment_set_adjustment, RID, bool, float, float, float, bool, RID)

	PASS9(environment_set_fog, RID, bool, const Color &, float, float, float, float, float, float)
	PASS13(environment_set_volumetric_fog, RID, bool, float, const Color &, const Color &, float, float, float, float, float, bool, float, float)

	PASS2(environment_set_volumetric_fog_volume_size, int, int)
	PASS1(environment_set_volumetric_fog_filter_active, bool)

	PASS11(environment_set_sdfgi, RID, bool, RS::EnvironmentSDFGICascades, float, RS::EnvironmentSDFGIYScale, bool, float, bool, float, float, float)
	PASS1(environment_set_sdfgi_ray_count, RS::EnvironmentSDFGIRayCount)
	PASS1(environment_set_sdfgi_frames_to_converge, RS::EnvironmentSDFGIFramesToConverge)
	PASS1(environment_set_sdfgi_frames_to_update_light, RS::EnvironmentSDFGIFramesToUpdateLight)

	PASS1RC(RS::EnvironmentBG, environment_get_background, RID)
	PASS1RC(int, environment_get_canvas_max_layer, RID)

	PASS3R(Ref<Image>, environment_bake_panorama, RID, bool, const Size2i &)

	PASS3(screen_space_roughness_limiter_set_active, bool, float, float)
	PASS1(sub_surface_scattering_set_quality, RS::SubSurfaceScatteringQuality)
	PASS2(sub_surface_scattering_set_scale, float, float)

	/* CAMERA EFFECTS */

	PASS0R(RID, camera_effects_allocate)
	PASS1(camera_effects_initialize, RID)

	PASS2(camera_effects_set_dof_blur_quality, RS::DOFBlurQuality, bool)
	PASS1(camera_effects_set_dof_blur_bokeh_shape, RS::DOFBokehShape)

	PASS8(camera_effects_set_dof_blur, RID, bool, float, float, bool, float, float, float)
	PASS3(camera_effects_set_custom_exposure, RID, bool, float)

	PASS1(shadows_quality_set, RS::ShadowQuality)
	PASS1(directional_shadow_quality_set, RS::ShadowQuality)

	PASS2(sdfgi_set_debug_probe_select, const Vector3 &, const Vector3 &)

	/* Render Buffers */

	PASS0R(RID, render_buffers_create)
	PASS8(render_buffers_configure, RID, RID, int, int, RS::ViewportMSAA, RS::ViewportScreenSpaceAA, bool, uint32_t)
	PASS1(gi_set_use_half_resolution, bool)

	/* Shadow Atlas */
	PASS0R(RID, shadow_atlas_create)
	PASS3(shadow_atlas_set_size, RID, int, bool)
	PASS3(shadow_atlas_set_quadrant_subdivision, RID, int, int)

	PASS1(set_debug_draw_mode, RS::ViewportDebugDraw)

	PASS1(decals_set_filter, RS::DecalFilter)
	PASS1(light_projectors_set_filter, RS::LightProjectorFilter)

	virtual void update();

	bool free(RID p_rid);

	void set_scene_render(RendererSceneRender *p_scene_render);

	virtual void update_visibility_notifiers();

	RendererSceneCull();
	virtual ~RendererSceneCull();
};

#endif // RENDERING_SERVER_SCENE_CULL_H
