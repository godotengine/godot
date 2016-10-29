#ifndef VISUALSERVERSCENE_H
#define VISUALSERVERSCENE_H

#include "servers/visual/rasterizer.h"

#include "geometry.h"
#include "allocators.h"
#include "octree.h"
#include "self_list.h"

class VisualServerScene {
public:


	enum {

		MAX_INSTANCE_CULL=65536,
		MAX_LIGHTS_CULLED=4096,
		MAX_ROOM_CULL=32,
		MAX_EXTERIOR_PORTALS=128,
	};

	uint64_t render_pass;


	static VisualServerScene *singleton;
#if 0
	struct Portal {

		bool enabled;
		float disable_distance;
		Color disable_color;
		float connect_range;
		Vector<Point2> shape;
		Rect2 bounds;


		Portal() { enabled=true; disable_distance=50; disable_color=Color(); connect_range=0.8; }
	};

	struct BakedLight {

		Rasterizer::BakedLightData data;
		DVector<int> sampler;
		AABB octree_aabb;
		Size2i octree_tex_size;
		Size2i light_tex_size;

	};

	struct BakedLightSampler {

		float params[BAKED_LIGHT_SAMPLER_MAX];
		int resolution;
		Vector<Vector3> dp_cache;

		BakedLightSampler() {
			params[BAKED_LIGHT_SAMPLER_STRENGTH]=1.0;
			params[BAKED_LIGHT_SAMPLER_ATTENUATION]=1.0;
			params[BAKED_LIGHT_SAMPLER_RADIUS]=1.0;
			params[BAKED_LIGHT_SAMPLER_DETAIL_RATIO]=0.1;
			resolution=16;
		}
	};

	void _update_baked_light_sampler_dp_cache(BakedLightSampler * blsamp);

#endif


	struct Camera  : public RID_Data {

		enum Type {
			PERSPECTIVE,
			ORTHOGONAL
		};
		Type type;
		float fov;
		float znear,zfar;
		float size;
		uint32_t visible_layers;
		bool vaspect;
		RID env;

		Transform transform;

		Camera() {

			visible_layers=0xFFFFFFFF;
			fov=60;
			type=PERSPECTIVE;
			znear=0.1; zfar=100;
			size=1.0;
			vaspect=false;

		}
	};

	mutable RID_Owner<Camera> camera_owner;

	virtual RID camera_create();
	virtual void camera_set_perspective(RID p_camera,float p_fovy_degrees, float p_z_near, float p_z_far);
	virtual void camera_set_orthogonal(RID p_camera,float p_size, float p_z_near, float p_z_far);
	virtual void camera_set_transform(RID p_camera,const Transform& p_transform);
	virtual void camera_set_cull_mask(RID p_camera,uint32_t p_layers);
	virtual void camera_set_environment(RID p_camera,RID p_env);
	virtual void camera_set_use_vertical_aspect(RID p_camera,bool p_enable);


	/*

	struct RoomInfo {

		Transform affine_inverse;
		Room *room;
		List<Instance*> owned_geometry_instances;
		List<Instance*> owned_portal_instances;
		List<Instance*> owned_room_instances;
		List<Instance*> owned_light_instances; //not used, but just for the sake of it
		Set<Instance*> disconnected_child_portals;
		Set<Instance*> owned_autoroom_geometry;
		uint64_t last_visited_pass;
		RoomInfo() { last_visited_pass=0; }

	};

	struct InstancePortal {

		Portal *portal;
		Set<Instance*> candidate_set;
		Instance *connected;
		uint64_t last_visited_pass;

		Plane plane_cache;
		Vector<Vector3> transformed_point_cache;


		PortalInfo() { connected=NULL; last_visited_pass=0;}
	};
*/


	/* SCENARIO API */

	struct Instance;

	struct Scenario  : RID_Data {


		VS::ScenarioDebugMode debug;
		RID self;
		// well wtf, balloon allocator is slower?

		Octree<Instance,true> octree;

		List<Instance*> directional_lights;
		RID environment;
		RID fallback_environment;

		SelfList<Instance>::List instances;

		Scenario() {  debug=VS::SCENARIO_DEBUG_DISABLED; }
	};

	mutable RID_Owner<Scenario> scenario_owner;

	static void* _instance_pair(void *p_self, OctreeElementID, Instance *p_A,int, OctreeElementID, Instance *p_B,int);
	static void _instance_unpair(void *p_self, OctreeElementID, Instance *p_A,int, OctreeElementID, Instance *p_B,int,void*);

	virtual RID scenario_create();

	virtual void scenario_set_debug(RID p_scenario,VS::ScenarioDebugMode p_debug_mode);
	virtual void scenario_set_environment(RID p_scenario, RID p_environment);
	virtual void scenario_set_fallback_environment(RID p_scenario, RID p_environment);


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
		bool update_materials;
		SelfList<Instance> update_item;


		AABB aabb;
		AABB transformed_aabb;
		float extra_margin;
		uint32_t object_ID;
		bool visible;
		uint32_t layer_mask;

		float lod_begin;
		float lod_end;
		float lod_begin_hysteresis;
		float lod_end_hysteresis;
		RID lod_instance;

		Instance *room;
		SelfList<Instance> room_item;
		bool visible_in_all_rooms;

		uint64_t last_render_pass;
		uint64_t last_frame_pass;

		uint64_t version; // changes to this, and changes to base increase version

		InstanceBaseData *base_data;

		virtual void base_removed() {

			singleton->instance_set_base(self,RID());
		}

		virtual void base_changed() {

			singleton->_instance_queue_update(this,true,true);
		}


		Instance() : scenario_item(this), update_item(this), room_item(this) {

			octree_id=0;
			scenario=NULL;


			update_aabb=false;
			update_materials=false;

			extra_margin=0;


			object_ID=0;
			visible=true;
			layer_mask=1;

			lod_begin=0;
			lod_end=0;
			lod_begin_hysteresis=0;
			lod_end_hysteresis=0;

			room=NULL;
			visible_in_all_rooms=false;



			last_render_pass=0;
			last_frame_pass=0;
			version=1;
			base_data=NULL;

		}

		~Instance() {

			if (base_data)
				memdelete(base_data);

		}
	};

	SelfList<Instance>::List _instance_update_list;
	void _instance_queue_update(Instance *p_instance,bool p_update_aabb,bool p_update_materials=false);


	struct InstanceGeometryData : public InstanceBaseData {

		List<Instance*> lighting;
		bool lighting_dirty;

		InstanceGeometryData() {

			lighting_dirty=false;
		}
	};


	struct InstanceLightData : public InstanceBaseData {

		struct PairInfo {
			List<Instance*>::Element *L; //light iterator in geometry
			Instance *geometry;
		};

		RID instance;
		uint64_t last_hash;
		List<Instance*>::Element *D; // directional light in scenario

		bool shadow_sirty;

		List<PairInfo> geometries;

		InstanceLightData() {

			shadow_sirty=true;
			D=NULL;
			last_hash=0;
		}
	};


	Instance *instance_cull_result[MAX_INSTANCE_CULL];
	Instance *instance_shadow_cull_result[MAX_INSTANCE_CULL]; //used for generating shadowmaps
	Instance *light_cull_result[MAX_LIGHTS_CULLED];
	RID light_instance_cull_result[MAX_LIGHTS_CULLED];
	int light_cull_count;


	RID_Owner<Instance> instance_owner;

 // from can be mesh, light,  area and portal so far.
	virtual RID instance_create(); // from can be mesh, light, poly, area and portal so far.

	virtual void instance_set_base(RID p_instance, RID p_base); // from can be mesh, light, poly, area and portal so far.
	virtual void instance_set_scenario(RID p_instance, RID p_scenario); // from can be mesh, light, poly, area and portal so far.
	virtual void instance_set_layer_mask(RID p_instance, uint32_t p_mask);
	virtual void instance_set_transform(RID p_instance, const Transform& p_transform);
	virtual void instance_attach_object_instance_ID(RID p_instance,ObjectID p_ID);
	virtual void instance_set_morph_target_weight(RID p_instance,int p_shape, float p_weight);
	virtual void instance_set_surface_material(RID p_instance,int p_surface, RID p_material);

	virtual void instance_attach_skeleton(RID p_instance,RID p_skeleton);
	virtual void instance_set_exterior( RID p_instance, bool p_enabled );
	virtual void instance_set_room( RID p_instance, RID p_room );

	virtual void instance_set_extra_visibility_margin( RID p_instance, real_t p_margin );


	// don't use these in a game!
	virtual Vector<ObjectID> instances_cull_aabb(const AABB& p_aabb, RID p_scenario=RID()) const;
	virtual Vector<ObjectID> instances_cull_ray(const Vector3& p_from, const Vector3& p_to, RID p_scenario=RID()) const;
	virtual Vector<ObjectID> instances_cull_convex(const Vector<Plane>& p_convex, RID p_scenario=RID()) const;


	virtual void instance_geometry_set_flag(RID p_instance,VS::InstanceFlags p_flags,bool p_enabled);
	virtual void instance_geometry_set_cast_shadows_setting(RID p_instance, VS::ShadowCastingSetting p_shadow_casting_setting);
	virtual void instance_geometry_set_material_override(RID p_instance, RID p_material);


	virtual void instance_geometry_set_draw_range(RID p_instance,float p_min,float p_max,float p_min_margin,float p_max_margin);
	virtual void instance_geometry_set_as_instance_lod(RID p_instance,RID p_as_lod_of_instance);


	_FORCE_INLINE_ void _update_instance(Instance *p_instance);
	_FORCE_INLINE_ void _update_instance_aabb(Instance *p_instance);
	_FORCE_INLINE_ void _update_dirty_instance(Instance *p_instance);


	void render_camera(RID p_camera, RID p_scenario, Size2 p_viewport_size);
	void update_dirty_instances();
	bool free(RID p_rid);

	VisualServerScene();
};

#endif // VISUALSERVERSCENE_H
