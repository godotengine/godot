/*************************************************************************/
/*  visual_server_raster.h                                               */
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
#ifndef VISUAL_SERVER_RASTER_H
#define VISUAL_SERVER_RASTER_H

#include "allocators.h"
#include "octree.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"
#include "visual_server_canvas.h"
#include "visual_server_global.h"
#include "visual_server_scene.h"
#include "visual_server_viewport.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class VisualServerRaster : public VisualServer {

	enum {

		MAX_INSTANCE_CULL = 8192,
		MAX_INSTANCE_LIGHTS = 4,
		LIGHT_CACHE_DIRTY = -1,
		MAX_LIGHTS_CULLED = 256,
		MAX_ROOM_CULL = 32,
		MAX_EXTERIOR_PORTALS = 128,
		MAX_LIGHT_SAMPLERS = 256,
		INSTANCE_ROOMLESS_MASK = (1 << 20)

	};

	static int changes;
	bool draw_extra_frame;
	RID test_cube;

	int black_margin[4];
	RID black_image[4];

	struct FrameDrawnCallbacks {

		ObjectID object;
		StringName method;
		Variant param;
	};

	List<FrameDrawnCallbacks> frame_drawn_callbacks;

// FIXME: Kept as reference for future implementation
#if 0
	struct Room {

		bool occlude_exterior;
		BSP_Tree bounds;
		Room() { occlude_exterior=true; }
	};


	BalloonAllocator<> octree_allocator;

	struct OctreeAllocator {

		static BalloonAllocator<> *allocator;

		_FORCE_INLINE_ static void *alloc(size_t p_size) { return allocator->alloc(p_size); }
		_FORCE_INLINE_ static void free(void *p_ptr) { return allocator->free(p_ptr); }

	};

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
		PoolVector<int> sampler;
		Rect3 octree_aabb;
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
	struct Camera  {

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


	struct Instance;
	typedef Set<Instance*,Comparator<Instance*>,OctreeAllocator> InstanceSet;
	struct Scenario;

	struct Instance {

		enum {

			MAX_LIGHTS=4
		};

		RID self;
		OctreeElementID octree_id;
		Scenario *scenario;
		bool update;
		bool update_aabb;
		bool update_materials;
		Instance *update_next;
		InstanceType base_type;

		RID base_rid;

		Rect3 aabb;
		Rect3 transformed_aabb;
		uint32_t object_ID;
		bool visible;
		bool visible_in_all_rooms;
		uint32_t layer_mask;
		float draw_range_begin;
		float draw_range_end;
		float extra_margin;



		Rasterizer::InstanceData data;


		Set<Instance*> auto_rooms;
		Set<Instance*> valid_auto_rooms;
		Instance *room;
		List<Instance*>::Element *RE;
		Instance *baked_light;
		List<Instance*>::Element *BLE;
		Instance *sampled_light;
		bool exterior;

		uint64_t last_render_pass;
		uint64_t last_frame_pass;

		uint64_t version; // changes to this, and changes to base increase version

		InstanceSet lights;
		bool light_cache_dirty;



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

		struct PortalInfo {

			Portal *portal;
			Set<Instance*> candidate_set;
			Instance *connected;
			uint64_t last_visited_pass;

			Plane plane_cache;
			Vector<Vector3> transformed_point_cache;


			PortalInfo() { connected=NULL; last_visited_pass=0;}
		};

		struct LightInfo {

			RID instance;
			int light_set_index;
			uint64_t last_version;
			uint64_t last_add_pass;
			List<RID>::Element *D; // directional light in scenario
			InstanceSet affected;
			bool enabled;
			float dtc; //distance to camera, used for sorting


			LightInfo() {

				D=NULL;
				light_set_index=-1;
				last_add_pass=0;
				enabled=true;
			}
		};

		struct BakedLightInfo {

			BakedLight *baked_light;
			Transform affine_inverse;
			List<Instance*> owned_instances;
		};

		struct BakedLightSamplerInfo {

			Set<Instance*> baked_lights;
			Set<Instance*> owned_instances;
			BakedLightSampler *sampler;
			int resolution;
			Vector<Color> light_buffer;
			RID sampled_light;
			uint64_t last_pass;
			Transform xform; // viewspace normal to lightspace, might not use one.
			BakedLightSamplerInfo() {
				sampler=NULL;
				last_pass=0;
				resolution=0;
			}
		};

		struct ParticlesInfo {

			RID instance;
		};


		RoomInfo *room_info;
		LightInfo *light_info;
		ParticlesInfo *particles_info;
		PortalInfo * portal_info;
		BakedLightInfo * baked_light_info;
		BakedLightSamplerInfo * baked_light_sampler_info;


		Instance() {
			octree_id=0;
			update_next=0;
			object_ID=0;
			last_render_pass=0;
			last_frame_pass=0;
			light_info=0;
			particles_info=0;
			update_next=NULL;
			update=false;
			visible=true;
			data.cast_shadows=SHADOW_CASTING_SETTING_ON;
			data.receive_shadows=true;
			data.depth_scale=false;
			data.billboard=false;
			data.billboard_y=false;
			data.baked_light=NULL;
			data.baked_light_octree_xform=NULL;
			data.baked_lightmap_id=-1;
			version=1;
			room_info=NULL;
			room=NULL;
			RE=NULL;
			portal_info=NULL;
			exterior=false;
			layer_mask=1;
			draw_range_begin=0;
			draw_range_end=0;
			extra_margin=0;
			visible_in_all_rooms=false;
			update_aabb=false;
			update_materials=false;

			baked_light=NULL;
			baked_light_info=NULL;
			baked_light_sampler_info=NULL;
			sampled_light=NULL;
			BLE=NULL;

			light_cache_dirty=true;

		}

		~Instance() {

			if (light_info)
				memdelete(light_info);
			if (particles_info)
				memdelete(particles_info);
			if (room_info)
				memdelete(room_info);
			if (portal_info)
				memdelete(portal_info);
			if (baked_light_info)
				memdelete(baked_light_info);
		};
	};

	struct _InstanceLightsort {

		bool operator()(const Instance* p_A, const Instance* p_B) const { return p_A->light_info->dtc < p_B->light_info->dtc; }
	};

	struct Scenario {


		ScenarioDebugMode debug;
		RID self;
		// well wtf, balloon allocator is slower?
		typedef ::Octree<Instance,true> Octree;

		Octree octree;

		List<RID> directional_lights;
		RID environment;
		RID fallback_environment;

		Instance *dirty_instances;

		Scenario() { dirty_instances=NULL; debug=SCENARIO_DEBUG_DISABLED; }
	};



	mutable RID_Owner<Rasterizer::ShaderMaterial> canvas_item_material_owner;



	struct Viewport {

		RID self;
		RID parent;

		VisualServer::ViewportRect rect;
		RID camera;
		RID scenario;
		RID viewport_data;

		RenderTargetUpdateMode render_target_update_mode;
		RID render_target;
		RID render_target_texture;

		Rect2 rt_to_screen_rect;

		bool hide_scenario;
		bool hide_canvas;
		bool transparent_bg;
		bool queue_capture;
		bool render_target_vflip;
		bool render_target_clear_on_new_frame;
		bool render_target_clear;
		bool disable_environment;

		Image capture;

		bool rendered_in_prev_frame;

		struct CanvasKey {

			int layer;
			RID canvas;
			bool operator<(const CanvasKey& p_canvas) const { if (layer==p_canvas.layer) return canvas < p_canvas.canvas; return layer<p_canvas.layer; }
			CanvasKey() { layer=0; }
			CanvasKey(const RID& p_canvas, int p_layer) { canvas=p_canvas; layer=p_layer; }
		};

		struct CanvasData {

			Canvas *canvas;
			Transform2D transform;
			int layer;
		};

		Transform2D global_transform;

		Map<RID,CanvasData> canvas_map;

		SelfList<Viewport> update_list;

		Viewport() : update_list(this) { transparent_bg=false; render_target_update_mode=RENDER_TARGET_UPDATE_WHEN_VISIBLE; queue_capture=false; rendered_in_prev_frame=false; render_target_vflip=false; render_target_clear_on_new_frame=true; render_target_clear=true; disable_environment=false; }
	};

	SelfList<Viewport>::List viewport_update_list;

	Map<RID,int> screen_viewports;

	struct CullRange {

		Plane nearp;
		float min,max;
		float z_near,z_far;

		void add_aabb(const Rect3& p_aabb) {


		}
	};

	struct Cursor {

		Point2 pos;
		float rot;
		RID texture;
		Point2 center;
		bool visible;
		Rect2 region;
		Cursor() {

			rot = 0;
			visible = false;
			region = Rect2();
		};
	};

	Rect2 canvas_clip;
	Color clear_color;
	Cursor cursors[MAX_CURSORS];
	RID default_cursor_texture;

	static void* instance_pair(void *p_self, OctreeElementID,Instance *p_A,int, OctreeElementID,Instance *p_B,int);
	static void instance_unpair(void *p_self, OctreeElementID,Instance *p_A,int, OctreeElementID,Instance *p_B,int,void*);

	Instance *instance_cull_result[MAX_INSTANCE_CULL];
	Instance *instance_shadow_cull_result[MAX_INSTANCE_CULL]; //used for generating shadowmaps
	Instance *light_cull_result[MAX_LIGHTS_CULLED];
	int light_cull_count;

	Instance *exterior_portal_cull_result[MAX_EXTERIOR_PORTALS];
	int exterior_portal_cull_count;
	bool exterior_visited;

	Instance *light_sampler_cull_result[MAX_LIGHT_SAMPLERS];
	int light_samplers_culled;

	Instance *room_cull_result[MAX_ROOM_CULL];
	int room_cull_count;
	bool room_cull_enabled;
	bool light_discard_enabled;
	bool shadows_enabled;
	int black_margin[4];
	RID black_image[4];

	Vector<Vector3> aabb_random_points;
	Vector<Vector3> transformed_aabb_random_points;

	void _instance_validate_autorooms(Instance *p_geometry);

	void _portal_disconnect(Instance *p_portal,bool p_cleanup=false);
	void _portal_attempt_connect(Instance *p_portal);
	void _dependency_queue_update(RID p_rid, bool p_update_aabb=false, bool p_update_materials=false);
	_FORCE_INLINE_ void _instance_queue_update(Instance *p_instance,bool p_update_aabb=false,bool p_update_materials=false);
	void _update_instances();
	void _update_instance_aabb(Instance *p_instance);
	void _update_instance(Instance *p_instance);
	void _free_attached_instances(RID p_rid,bool p_free_scenario=false);
	void _clean_up_owner(RID_OwnerBase *p_owner,String p_type);

	Instance *instance_update_list;

	//RID default_scenario;
	//RID default_viewport;

	RID test_cube;


	mutable RID_Owner<Room> room_owner;
	mutable RID_Owner<Portal> portal_owner;

	mutable RID_Owner<BakedLight> baked_light_owner;
	mutable RID_Owner<BakedLightSampler> baked_light_sampler_owner;

	mutable RID_Owner<Camera> camera_owner;
	mutable RID_Owner<Viewport> viewport_owner;

	mutable RID_Owner<Scenario> scenario_owner;
	mutable RID_Owner<Instance> instance_owner;

	mutable RID_Owner<Canvas> canvas_owner;
	mutable RID_Owner<CanvasItem> canvas_item_owner;

	Map< RID, Set<RID> > instance_dependency_map;
	Map< RID, Set<Instance*> > skeleton_dependency_map;


	ViewportRect viewport_rect;
	_FORCE_INLINE_ void _instance_draw(Instance *p_instance);

	bool _test_portal_cull(Camera *p_camera, Instance *p_portal_from, Instance *p_portal_to);
	void _cull_portal(Camera *p_camera, Instance *p_portal,Instance *p_from_portal);
	void _cull_room(Camera *p_camera, Instance *p_room,Instance *p_from_portal=NULL);
	void _process_sampled_light(const Transform &p_camera, Instance *p_sampled_light, bool p_linear_colorspace);

	void _render_no_camera(Viewport *p_viewport,Camera *p_camera, Scenario *p_scenario);
	void _render_camera(Viewport *p_viewport,Camera *p_camera, Scenario *p_scenario);
	static void _render_canvas_item_viewport(VisualServer* p_self,void *p_vp,const Rect2& p_rect);
	void _render_canvas_item_tree(CanvasItem *p_canvas_item, const Transform2D& p_transform, const Rect2& p_clip_rect, const Color &p_modulate, Rasterizer::CanvasLight *p_lights);
	void _render_canvas_item(CanvasItem *p_canvas_item, const Transform2D& p_transform, const Rect2& p_clip_rect, float p_opacity, int p_z, Rasterizer::CanvasItem **z_list, Rasterizer::CanvasItem **z_last_list, CanvasItem *p_canvas_clip, CanvasItem *p_material_owner);
	void _render_canvas(Canvas *p_canvas, const Transform2D &p_transform, Rasterizer::CanvasLight *p_lights, Rasterizer::CanvasLight *p_masked_lights);
	void _light_mask_canvas_items(int p_z,Rasterizer::CanvasItem *p_canvas_item,Rasterizer::CanvasLight *p_masked_lights);

	Vector<Vector3> _camera_generate_endpoints(Instance *p_light,Camera *p_camera,float p_range_min, float p_range_max);
	Vector<Plane> _camera_generate_orthogonal_planes(Instance *p_light,Camera *p_camera,float p_range_min, float p_range_max);

	void _light_instance_update_lispsm_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range);
	void _light_instance_update_pssm_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range);

	void _light_instance_update_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range);

	uint64_t render_pass;
	int changes;
	bool draw_extra_frame;

	void _draw_viewport_camera(Viewport *p_viewport, bool p_ignore_camera);
	void _draw_viewport(Viewport *p_viewport,int p_ofs_x, int p_ofs_y,int p_parent_w,int p_parent_h);
	void _draw_viewports();
	void _draw_cursors_and_margins();


	Rasterizer *rasterizer;

#endif

	void _draw_margins();
	static void _changes_changed() {}

public:
//if editor is redrawing when it shouldn't, enable this and put a breakpoint in _changes_changed()
//#define DEBUG_CHANGES

#ifdef DEBUG_CHANGES
	_FORCE_INLINE_ static void redraw_request() {
		changes++;
		_changes_changed();
	}

#define DISPLAY_CHANGED \
	changes++;          \
	_changes_changed();

#else
	_FORCE_INLINE_ static void redraw_request() { changes++; }

#define DISPLAY_CHANGED \
	changes++;
#endif
//	print_line(String("CHANGED: ") + __FUNCTION__);

#define BIND0R(m_r, m_name) \
	m_r m_name() { return BINDBASE->m_name(); }
#define BIND1R(m_r, m_name, m_type1) \
	m_r m_name(m_type1 arg1) { return BINDBASE->m_name(arg1); }
#define BIND1RC(m_r, m_name, m_type1) \
	m_r m_name(m_type1 arg1) const { return BINDBASE->m_name(arg1); }
#define BIND2R(m_r, m_name, m_type1, m_type2) \
	m_r m_name(m_type1 arg1, m_type2 arg2) { return BINDBASE->m_name(arg1, arg2); }
#define BIND2RC(m_r, m_name, m_type1, m_type2) \
	m_r m_name(m_type1 arg1, m_type2 arg2) const { return BINDBASE->m_name(arg1, arg2); }
#define BIND3RC(m_r, m_name, m_type1, m_type2, m_type3) \
	m_r m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3) const { return BINDBASE->m_name(arg1, arg2, arg3); }
#define BIND4RC(m_r, m_name, m_type1, m_type2, m_type3, m_type4) \
	m_r m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4) const { return BINDBASE->m_name(arg1, arg2, arg3, arg4); }

#define BIND1(m_name, m_type1) \
	void m_name(m_type1 arg1) { DISPLAY_CHANGED BINDBASE->m_name(arg1); }
#define BIND2(m_name, m_type1, m_type2) \
	void m_name(m_type1 arg1, m_type2 arg2) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2); }
#define BIND2C(m_name, m_type1, m_type2) \
	void m_name(m_type1 arg1, m_type2 arg2) const { BINDBASE->m_name(arg1, arg2); }
#define BIND3(m_name, m_type1, m_type2, m_type3) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3); }
#define BIND4(m_name, m_type1, m_type2, m_type3, m_type4) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4); }
#define BIND5(m_name, m_type1, m_type2, m_type3, m_type4, m_type5) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5); }
#define BIND6(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6); }
#define BIND7(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6, m_type7) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6, m_type7 arg7) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7); }
#define BIND8(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6, m_type7, m_type8) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6, m_type7 arg7, m_type8 arg8) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); }
#define BIND9(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6, m_type7, m_type8, m_type9) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6, m_type7 arg7, m_type8 arg8, m_type9 arg9) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); }
#define BIND10(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6, m_type7, m_type8, m_type9, m_type10) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6, m_type7 arg7, m_type8 arg8, m_type9 arg9, m_type10 arg10) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10); }
#define BIND11(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6, m_type7, m_type8, m_type9, m_type10, m_type11) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6, m_type7 arg7, m_type8 arg8, m_type9 arg9, m_type10 arg10, m_type11 arg11) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11); }
#define BIND12(m_name, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6, m_type7, m_type8, m_type9, m_type10, m_type11, m_type12) \
	void m_name(m_type1 arg1, m_type2 arg2, m_type3 arg3, m_type4 arg4, m_type5 arg5, m_type6 arg6, m_type7 arg7, m_type8 arg8, m_type9 arg9, m_type10 arg10, m_type11 arg11, m_type12 arg12) { DISPLAY_CHANGED BINDBASE->m_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12); }

//from now on, calls forwarded to this singleton
#define BINDBASE VSG::storage

	/* TEXTURE API */

	BIND0R(RID, texture_create)
	BIND5(texture_allocate, RID, int, int, Image::Format, uint32_t)
	BIND3(texture_set_data, RID, const Ref<Image> &, CubeMapSide)
	BIND2RC(Ref<Image>, texture_get_data, RID, CubeMapSide)
	BIND2(texture_set_flags, RID, uint32_t)
	BIND1RC(uint32_t, texture_get_flags, RID)
	BIND1RC(Image::Format, texture_get_format, RID)
	BIND1RC(uint32_t, texture_get_texid, RID)
	BIND1RC(uint32_t, texture_get_width, RID)
	BIND1RC(uint32_t, texture_get_height, RID)
	BIND3(texture_set_size_override, RID, int, int)

	BIND3(texture_set_detect_3d_callback, RID, TextureDetectCallback, void *)
	BIND3(texture_set_detect_srgb_callback, RID, TextureDetectCallback, void *)
	BIND3(texture_set_detect_normal_callback, RID, TextureDetectCallback, void *)

	BIND2(texture_set_path, RID, const String &)
	BIND1RC(String, texture_get_path, RID)
	BIND1(texture_set_shrink_all_x2_on_set_data, bool)
	BIND1(texture_debug_usage, List<TextureInfo> *)

	BIND1(textures_keep_original, bool)

	/* SKY API */

	BIND0R(RID, sky_create)
	BIND3(sky_set_texture, RID, RID, int)

	/* SHADER API */

	BIND0R(RID, shader_create)

	BIND2(shader_set_code, RID, const String &)
	BIND1RC(String, shader_get_code, RID)

	BIND2C(shader_get_param_list, RID, List<PropertyInfo> *)

	BIND3(shader_set_default_texture_param, RID, const StringName &, RID)
	BIND2RC(RID, shader_get_default_texture_param, RID, const StringName &)

	/* COMMON MATERIAL API */

	BIND0R(RID, material_create)

	BIND2(material_set_shader, RID, RID)
	BIND1RC(RID, material_get_shader, RID)

	BIND3(material_set_param, RID, const StringName &, const Variant &)
	BIND2RC(Variant, material_get_param, RID, const StringName &)

	BIND2(material_set_render_priority, RID, int)
	BIND2(material_set_line_width, RID, float)
	BIND2(material_set_next_pass, RID, RID)

	/* MESH API */

	BIND0R(RID, mesh_create)

	BIND10(mesh_add_surface, RID, uint32_t, PrimitiveType, const PoolVector<uint8_t> &, int, const PoolVector<uint8_t> &, int, const Rect3 &, const Vector<PoolVector<uint8_t> > &, const Vector<Rect3> &)

	BIND2(mesh_set_blend_shape_count, RID, int)
	BIND1RC(int, mesh_get_blend_shape_count, RID)

	BIND2(mesh_set_blend_shape_mode, RID, BlendShapeMode)
	BIND1RC(BlendShapeMode, mesh_get_blend_shape_mode, RID)

	BIND3(mesh_surface_set_material, RID, int, RID)
	BIND2RC(RID, mesh_surface_get_material, RID, int)

	BIND2RC(int, mesh_surface_get_array_len, RID, int)
	BIND2RC(int, mesh_surface_get_array_index_len, RID, int)

	BIND2RC(PoolVector<uint8_t>, mesh_surface_get_array, RID, int)
	BIND2RC(PoolVector<uint8_t>, mesh_surface_get_index_array, RID, int)

	BIND2RC(uint32_t, mesh_surface_get_format, RID, int)
	BIND2RC(PrimitiveType, mesh_surface_get_primitive_type, RID, int)

	BIND2RC(Rect3, mesh_surface_get_aabb, RID, int)
	BIND2RC(Vector<PoolVector<uint8_t> >, mesh_surface_get_blend_shapes, RID, int)
	BIND2RC(Vector<Rect3>, mesh_surface_get_skeleton_aabb, RID, int)

	BIND2(mesh_remove_surface, RID, int)
	BIND1RC(int, mesh_get_surface_count, RID)

	BIND2(mesh_set_custom_aabb, RID, const Rect3 &)
	BIND1RC(Rect3, mesh_get_custom_aabb, RID)

	BIND1(mesh_clear, RID)

	/* MULTIMESH API */

	BIND0R(RID, multimesh_create)

	BIND4(multimesh_allocate, RID, int, MultimeshTransformFormat, MultimeshColorFormat)
	BIND1RC(int, multimesh_get_instance_count, RID)

	BIND2(multimesh_set_mesh, RID, RID)
	BIND3(multimesh_instance_set_transform, RID, int, const Transform &)
	BIND3(multimesh_instance_set_transform_2d, RID, int, const Transform2D &)
	BIND3(multimesh_instance_set_color, RID, int, const Color &)

	BIND1RC(RID, multimesh_get_mesh, RID)
	BIND1RC(Rect3, multimesh_get_aabb, RID)

	BIND2RC(Transform, multimesh_instance_get_transform, RID, int)
	BIND2RC(Transform2D, multimesh_instance_get_transform_2d, RID, int)
	BIND2RC(Color, multimesh_instance_get_color, RID, int)

	BIND2(multimesh_set_visible_instances, RID, int)
	BIND1RC(int, multimesh_get_visible_instances, RID)

	/* IMMEDIATE API */

	BIND0R(RID, immediate_create)
	BIND3(immediate_begin, RID, PrimitiveType, RID)
	BIND2(immediate_vertex, RID, const Vector3 &)
	BIND2(immediate_normal, RID, const Vector3 &)
	BIND2(immediate_tangent, RID, const Plane &)
	BIND2(immediate_color, RID, const Color &)
	BIND2(immediate_uv, RID, const Vector2 &)
	BIND2(immediate_uv2, RID, const Vector2 &)
	BIND1(immediate_end, RID)
	BIND1(immediate_clear, RID)
	BIND2(immediate_set_material, RID, RID)
	BIND1RC(RID, immediate_get_material, RID)

	/* SKELETON API */

	BIND0R(RID, skeleton_create)
	BIND3(skeleton_allocate, RID, int, bool)
	BIND1RC(int, skeleton_get_bone_count, RID)
	BIND3(skeleton_bone_set_transform, RID, int, const Transform &)
	BIND2RC(Transform, skeleton_bone_get_transform, RID, int)
	BIND3(skeleton_bone_set_transform_2d, RID, int, const Transform2D &)
	BIND2RC(Transform2D, skeleton_bone_get_transform_2d, RID, int)

	/* Light API */

	BIND1R(RID, light_create, LightType)

	BIND2(light_set_color, RID, const Color &)
	BIND3(light_set_param, RID, LightParam, float)
	BIND2(light_set_shadow, RID, bool)
	BIND2(light_set_shadow_color, RID, const Color &)
	BIND2(light_set_projector, RID, RID)
	BIND2(light_set_negative, RID, bool)
	BIND2(light_set_cull_mask, RID, uint32_t)
	BIND2(light_set_reverse_cull_face_mode, RID, bool)

	BIND2(light_omni_set_shadow_mode, RID, LightOmniShadowMode)
	BIND2(light_omni_set_shadow_detail, RID, LightOmniShadowDetail)

	BIND2(light_directional_set_shadow_mode, RID, LightDirectionalShadowMode)
	BIND2(light_directional_set_blend_splits, RID, bool)
	BIND2(light_directional_set_shadow_depth_range_mode, RID, LightDirectionalShadowDepthRangeMode)

	/* PROBE API */

	BIND0R(RID, reflection_probe_create)

	BIND2(reflection_probe_set_update_mode, RID, ReflectionProbeUpdateMode)
	BIND2(reflection_probe_set_intensity, RID, float)
	BIND2(reflection_probe_set_interior_ambient, RID, const Color &)
	BIND2(reflection_probe_set_interior_ambient_energy, RID, float)
	BIND2(reflection_probe_set_interior_ambient_probe_contribution, RID, float)
	BIND2(reflection_probe_set_max_distance, RID, float)
	BIND2(reflection_probe_set_extents, RID, const Vector3 &)
	BIND2(reflection_probe_set_origin_offset, RID, const Vector3 &)
	BIND2(reflection_probe_set_as_interior, RID, bool)
	BIND2(reflection_probe_set_enable_box_projection, RID, bool)
	BIND2(reflection_probe_set_enable_shadows, RID, bool)
	BIND2(reflection_probe_set_cull_mask, RID, uint32_t)

	/* BAKED LIGHT API */

	BIND0R(RID, gi_probe_create)

	BIND2(gi_probe_set_bounds, RID, const Rect3 &)
	BIND1RC(Rect3, gi_probe_get_bounds, RID)

	BIND2(gi_probe_set_cell_size, RID, float)
	BIND1RC(float, gi_probe_get_cell_size, RID)

	BIND2(gi_probe_set_to_cell_xform, RID, const Transform &)
	BIND1RC(Transform, gi_probe_get_to_cell_xform, RID)

	BIND2(gi_probe_set_dynamic_range, RID, int)
	BIND1RC(int, gi_probe_get_dynamic_range, RID)

	BIND2(gi_probe_set_energy, RID, float)
	BIND1RC(float, gi_probe_get_energy, RID)

	BIND2(gi_probe_set_bias, RID, float)
	BIND1RC(float, gi_probe_get_bias, RID)

	BIND2(gi_probe_set_normal_bias, RID, float)
	BIND1RC(float, gi_probe_get_normal_bias, RID)

	BIND2(gi_probe_set_propagation, RID, float)
	BIND1RC(float, gi_probe_get_propagation, RID)

	BIND2(gi_probe_set_interior, RID, bool)
	BIND1RC(bool, gi_probe_is_interior, RID)

	BIND2(gi_probe_set_compress, RID, bool)
	BIND1RC(bool, gi_probe_is_compressed, RID)

	BIND2(gi_probe_set_dynamic_data, RID, const PoolVector<int> &)
	BIND1RC(PoolVector<int>, gi_probe_get_dynamic_data, RID)

	/* PARTICLES */

	BIND0R(RID, particles_create)

	BIND2(particles_set_emitting, RID, bool)
	BIND2(particles_set_amount, RID, int)
	BIND2(particles_set_lifetime, RID, float)
	BIND2(particles_set_one_shot, RID, bool)
	BIND2(particles_set_pre_process_time, RID, float)
	BIND2(particles_set_explosiveness_ratio, RID, float)
	BIND2(particles_set_randomness_ratio, RID, float)
	BIND2(particles_set_custom_aabb, RID, const Rect3 &)
	BIND2(particles_set_speed_scale, RID, float)
	BIND2(particles_set_use_local_coordinates, RID, bool)
	BIND2(particles_set_process_material, RID, RID)
	BIND2(particles_set_fixed_fps, RID, int)
	BIND2(particles_set_fractional_delta, RID, bool)
	BIND1(particles_restart, RID)

	BIND2(particles_set_draw_order, RID, VS::ParticlesDrawOrder)

	BIND2(particles_set_draw_passes, RID, int)
	BIND3(particles_set_draw_pass_mesh, RID, int, RID)

	BIND1R(Rect3, particles_get_current_aabb, RID)
	BIND2(particles_set_emission_transform, RID, const Transform &)

#undef BINDBASE
//from now on, calls forwarded to this singleton
#define BINDBASE VSG::scene

	/* CAMERA API */

	BIND0R(RID, camera_create)
	BIND4(camera_set_perspective, RID, float, float, float)
	BIND4(camera_set_orthogonal, RID, float, float, float)
	BIND2(camera_set_transform, RID, const Transform &)
	BIND2(camera_set_cull_mask, RID, uint32_t)
	BIND2(camera_set_environment, RID, RID)
	BIND2(camera_set_use_vertical_aspect, RID, bool)

#undef BINDBASE
//from now on, calls forwarded to this singleton
#define BINDBASE VSG::viewport

	/* VIEWPORT TARGET API */

	BIND0R(RID, viewport_create)

	BIND2(viewport_set_use_arvr, RID, bool)
	BIND3(viewport_set_size, RID, int, int)

	BIND2(viewport_set_active, RID, bool)
	BIND2(viewport_set_parent_viewport, RID, RID)

	BIND2(viewport_set_clear_mode, RID, ViewportClearMode)

	BIND3(viewport_attach_to_screen, RID, const Rect2 &, int)
	BIND1(viewport_detach, RID)

	BIND2(viewport_set_update_mode, RID, ViewportUpdateMode)
	BIND2(viewport_set_vflip, RID, bool)

	BIND1RC(RID, viewport_get_texture, RID)

	BIND2(viewport_set_hide_scenario, RID, bool)
	BIND2(viewport_set_hide_canvas, RID, bool)
	BIND2(viewport_set_disable_environment, RID, bool)
	BIND2(viewport_set_disable_3d, RID, bool)

	BIND2(viewport_attach_camera, RID, RID)
	BIND2(viewport_set_scenario, RID, RID)
	BIND2(viewport_attach_canvas, RID, RID)

	BIND2(viewport_remove_canvas, RID, RID)
	BIND3(viewport_set_canvas_transform, RID, RID, const Transform2D &)
	BIND2(viewport_set_transparent_background, RID, bool)

	BIND2(viewport_set_global_canvas_transform, RID, const Transform2D &)
	BIND3(viewport_set_canvas_layer, RID, RID, int)
	BIND2(viewport_set_shadow_atlas_size, RID, int)
	BIND3(viewport_set_shadow_atlas_quadrant_subdivision, RID, int, int)
	BIND2(viewport_set_msaa, RID, ViewportMSAA)
	BIND2(viewport_set_hdr, RID, bool)
	BIND2(viewport_set_usage, RID, ViewportUsage)

	BIND2R(int, viewport_get_render_info, RID, ViewportRenderInfo)
	BIND2(viewport_set_debug_draw, RID, ViewportDebugDraw)

/* ENVIRONMENT API */

#undef BINDBASE
//from now on, calls forwarded to this singleton
#define BINDBASE VSG::scene_render

	BIND0R(RID, environment_create)

	BIND2(environment_set_background, RID, EnvironmentBG)
	BIND2(environment_set_sky, RID, RID)
	BIND2(environment_set_sky_custom_fov, RID, float)
	BIND2(environment_set_bg_color, RID, const Color &)
	BIND2(environment_set_bg_energy, RID, float)
	BIND2(environment_set_canvas_max_layer, RID, int)
	BIND4(environment_set_ambient_light, RID, const Color &, float, float)
	BIND7(environment_set_ssr, RID, bool, int, float, float, float, bool)
	BIND12(environment_set_ssao, RID, bool, float, float, float, float, float, float, const Color &, EnvironmentSSAOQuality, EnvironmentSSAOBlur, float)

	BIND6(environment_set_dof_blur_near, RID, bool, float, float, float, EnvironmentDOFBlurQuality)
	BIND6(environment_set_dof_blur_far, RID, bool, float, float, float, EnvironmentDOFBlurQuality)
	BIND10(environment_set_glow, RID, bool, int, float, float, float, EnvironmentGlowBlendMode, float, float, bool)

	BIND9(environment_set_tonemap, RID, EnvironmentToneMapper, float, float, bool, float, float, float, float)

	BIND6(environment_set_adjustment, RID, bool, float, float, float, RID)

	BIND5(environment_set_fog, RID, bool, const Color &, const Color &, float)
	BIND6(environment_set_fog_depth, RID, bool, float, float, bool, float)
	BIND5(environment_set_fog_height, RID, bool, float, float, float)

/* SCENARIO API */

#undef BINDBASE
#define BINDBASE VSG::scene

	BIND0R(RID, scenario_create)

	BIND2(scenario_set_debug, RID, ScenarioDebugMode)
	BIND2(scenario_set_environment, RID, RID)
	BIND3(scenario_set_reflection_atlas_size, RID, int, int)
	BIND2(scenario_set_fallback_environment, RID, RID)

	/* INSTANCING API */
	// from can be mesh, light,  area and portal so far.
	BIND0R(RID, instance_create)

	BIND2(instance_set_base, RID, RID) // from can be mesh, light, poly, area and portal so far.
	BIND2(instance_set_scenario, RID, RID) // from can be mesh, light, poly, area and portal so far.
	BIND2(instance_set_layer_mask, RID, uint32_t)
	BIND2(instance_set_transform, RID, const Transform &)
	BIND2(instance_attach_object_instance_id, RID, ObjectID)
	BIND3(instance_set_blend_shape_weight, RID, int, float)
	BIND3(instance_set_surface_material, RID, int, RID)
	BIND2(instance_set_visible, RID, bool)

	BIND2(instance_attach_skeleton, RID, RID)
	BIND2(instance_set_exterior, RID, bool)

	BIND2(instance_set_extra_visibility_margin, RID, real_t)

	// don't use these in a game!
	BIND2RC(Vector<ObjectID>, instances_cull_aabb, const Rect3 &, RID)
	BIND3RC(Vector<ObjectID>, instances_cull_ray, const Vector3 &, const Vector3 &, RID)
	BIND2RC(Vector<ObjectID>, instances_cull_convex, const Vector<Plane> &, RID)

	BIND3(instance_geometry_set_flag, RID, InstanceFlags, bool)
	BIND2(instance_geometry_set_cast_shadows_setting, RID, ShadowCastingSetting)
	BIND2(instance_geometry_set_material_override, RID, RID)

	BIND5(instance_geometry_set_draw_range, RID, float, float, float, float)
	BIND2(instance_geometry_set_as_instance_lod, RID, RID)

#undef BINDBASE
//from now on, calls forwarded to this singleton
#define BINDBASE VSG::canvas

	/* CANVAS (2D) */

	BIND0R(RID, canvas_create)
	BIND3(canvas_set_item_mirroring, RID, RID, const Point2 &)
	BIND2(canvas_set_modulate, RID, const Color &)

	BIND0R(RID, canvas_item_create)
	BIND2(canvas_item_set_parent, RID, RID)

	BIND2(canvas_item_set_visible, RID, bool)
	BIND2(canvas_item_set_light_mask, RID, int)

	BIND2(canvas_item_set_transform, RID, const Transform2D &)
	BIND2(canvas_item_set_clip, RID, bool)
	BIND2(canvas_item_set_distance_field_mode, RID, bool)
	BIND3(canvas_item_set_custom_rect, RID, bool, const Rect2 &)
	BIND2(canvas_item_set_modulate, RID, const Color &)
	BIND2(canvas_item_set_self_modulate, RID, const Color &)

	BIND2(canvas_item_set_draw_behind_parent, RID, bool)

	BIND6(canvas_item_add_line, RID, const Point2 &, const Point2 &, const Color &, float, bool)
	BIND5(canvas_item_add_polyline, RID, const Vector<Point2> &, const Vector<Color> &, float, bool)
	BIND3(canvas_item_add_rect, RID, const Rect2 &, const Color &)
	BIND4(canvas_item_add_circle, RID, const Point2 &, float, const Color &)
	BIND7(canvas_item_add_texture_rect, RID, const Rect2 &, RID, bool, const Color &, bool, RID)
	BIND8(canvas_item_add_texture_rect_region, RID, const Rect2 &, RID, const Rect2 &, const Color &, bool, RID, bool)
	BIND11(canvas_item_add_nine_patch, RID, const Rect2 &, const Rect2 &, RID, const Vector2 &, const Vector2 &, NinePatchAxisMode, NinePatchAxisMode, bool, const Color &, RID)
	BIND7(canvas_item_add_primitive, RID, const Vector<Point2> &, const Vector<Color> &, const Vector<Point2> &, RID, float, RID)
	BIND7(canvas_item_add_polygon, RID, const Vector<Point2> &, const Vector<Color> &, const Vector<Point2> &, RID, RID, bool)
	BIND8(canvas_item_add_triangle_array, RID, const Vector<int> &, const Vector<Point2> &, const Vector<Color> &, const Vector<Point2> &, RID, int, RID)
	BIND3(canvas_item_add_mesh, RID, const RID &, RID)
	BIND3(canvas_item_add_multimesh, RID, RID, RID)
	BIND6(canvas_item_add_particles, RID, RID, RID, RID, int, int)
	BIND2(canvas_item_add_set_transform, RID, const Transform2D &)
	BIND2(canvas_item_add_clip_ignore, RID, bool)
	BIND2(canvas_item_set_sort_children_by_y, RID, bool)
	BIND2(canvas_item_set_z, RID, int)
	BIND2(canvas_item_set_z_as_relative_to_parent, RID, bool)
	BIND3(canvas_item_set_copy_to_backbuffer, RID, bool, const Rect2 &)

	BIND1(canvas_item_clear, RID)
	BIND2(canvas_item_set_draw_index, RID, int)

	BIND2(canvas_item_set_material, RID, RID)

	BIND2(canvas_item_set_use_parent_material, RID, bool)

	BIND0R(RID, canvas_light_create)
	BIND2(canvas_light_attach_to_canvas, RID, RID)
	BIND2(canvas_light_set_enabled, RID, bool)
	BIND2(canvas_light_set_scale, RID, float)
	BIND2(canvas_light_set_transform, RID, const Transform2D &)
	BIND2(canvas_light_set_texture, RID, RID)
	BIND2(canvas_light_set_texture_offset, RID, const Vector2 &)
	BIND2(canvas_light_set_color, RID, const Color &)
	BIND2(canvas_light_set_height, RID, float)
	BIND2(canvas_light_set_energy, RID, float)
	BIND3(canvas_light_set_z_range, RID, int, int)
	BIND3(canvas_light_set_layer_range, RID, int, int)
	BIND2(canvas_light_set_item_cull_mask, RID, int)
	BIND2(canvas_light_set_item_shadow_cull_mask, RID, int)

	BIND2(canvas_light_set_mode, RID, CanvasLightMode)

	BIND2(canvas_light_set_shadow_enabled, RID, bool)
	BIND2(canvas_light_set_shadow_buffer_size, RID, int)
	BIND2(canvas_light_set_shadow_gradient_length, RID, float)
	BIND2(canvas_light_set_shadow_filter, RID, CanvasLightShadowFilter)
	BIND2(canvas_light_set_shadow_color, RID, const Color &)
	BIND2(canvas_light_set_shadow_smooth, RID, float)

	BIND0R(RID, canvas_light_occluder_create)
	BIND2(canvas_light_occluder_attach_to_canvas, RID, RID)
	BIND2(canvas_light_occluder_set_enabled, RID, bool)
	BIND2(canvas_light_occluder_set_polygon, RID, RID)
	BIND2(canvas_light_occluder_set_transform, RID, const Transform2D &)
	BIND2(canvas_light_occluder_set_light_mask, RID, int)

	BIND0R(RID, canvas_occluder_polygon_create)
	BIND3(canvas_occluder_polygon_set_shape, RID, const PoolVector<Vector2> &, bool)
	BIND2(canvas_occluder_polygon_set_shape_as_lines, RID, const PoolVector<Vector2> &)

	BIND2(canvas_occluder_polygon_set_cull_mode, RID, CanvasOccluderPolygonCullMode)

	/* BLACK BARS */

	virtual void black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom);
	virtual void black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom);

	/* FREE */

	virtual void free(RID p_rid); ///< free RIDs associated with the visual server

	/* EVENT QUEUING */

	virtual void request_frame_drawn_callback(Object *p_where, const StringName &p_method, const Variant &p_userdata);

	virtual void draw();
	virtual void sync();
	virtual bool has_changed() const;
	virtual void init();
	virtual void finish();

	/* STATUS INFORMATION */

	virtual int get_render_info(RenderInfo p_info);

	virtual RID get_test_cube();

	/* TESTING */

	virtual void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale);
	virtual void set_default_clear_color(const Color &p_color);

	virtual bool has_feature(Features p_feature) const;

	virtual bool has_os_feature(const String &p_feature) const;
	virtual void set_debug_generate_wireframes(bool p_generate);

	VisualServerRaster();
	~VisualServerRaster();

#undef DISPLAY_CHANGED

#undef BIND0R
#undef BIND1RC
#undef BIND2RC
#undef BIND3RC
#undef BIND4RC

#undef BIND1
#undef BIND2
#undef BIND3
#undef BIND4
#undef BIND5
#undef BIND6
#undef BIND7
#undef BIND8
#undef BIND9
#undef BIND10
};

#endif
