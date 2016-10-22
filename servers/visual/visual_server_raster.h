/*************************************************************************/
/*  visual_server_raster.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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


#include "servers/visual_server.h"
#include "servers/visual/rasterizer.h"
#include "allocators.h"
#include "octree.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class VisualServerRaster : public VisualServer {


	enum {

		MAX_INSTANCE_CULL=8192,
		MAX_INSTANCE_LIGHTS=4,
		LIGHT_CACHE_DIRTY=-1,
		MAX_LIGHTS_CULLED=256,
		MAX_ROOM_CULL=32,
		MAX_EXTERIOR_PORTALS=128,
		MAX_LIGHT_SAMPLERS=256,
		INSTANCE_ROOMLESS_MASK=(1<<20)


	};

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

		AABB aabb;
		AABB transformed_aabb;
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
			Vector<Color> light_bufer;
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



	mutable RID_Owner<Rasterizer::CanvasItemMaterial> canvas_item_material_owner;

	struct CanvasItem : public Rasterizer::CanvasItem {


		RID parent; // canvas it belongs to
		List<CanvasItem*>::Element *E;
		RID viewport;
		int z;
		bool z_relative;
		bool sort_y;
		float opacity;
		float self_opacity;
		bool use_parent_material;


		Vector<CanvasItem*> child_items;


		CanvasItem() {
			E=NULL;
			z=0;
			opacity=1;
			self_opacity=1;
			sort_y=false;
			use_parent_material=false;
			z_relative=true;
		}
	};


	struct CanvasItemPtrSort {

		_FORCE_INLINE_ bool operator()(const CanvasItem* p_left,const CanvasItem* p_right) const {

			return p_left->xform.elements[2].y < p_right->xform.elements[2].y;
		}
	};

	struct CanvasLightOccluder;

	struct CanvasLightOccluderPolygon {

		bool active;
		Rect2 aabb;
		CanvasOccluderPolygonCullMode cull_mode;
		RID occluder;
		Set<Rasterizer::CanvasLightOccluderInstance*> owners;

		CanvasLightOccluderPolygon() { active=false; cull_mode=CANVAS_OCCLUDER_POLYGON_CULL_DISABLED; }
	};


	RID_Owner<CanvasLightOccluderPolygon> canvas_light_occluder_polygon_owner;

	RID_Owner<Rasterizer::CanvasLightOccluderInstance> canvas_light_occluder_owner;

	struct CanvasLight;

	struct Canvas {

		Set<RID> viewports;
		struct ChildItem {

			Point2 mirror;
			CanvasItem *item;
		};

		Set<Rasterizer::CanvasLight*> lights;
		Set<Rasterizer::CanvasLightOccluderInstance*> occluders;

		Vector<ChildItem> child_items;
		Color modulate;

		int find_item(CanvasItem *p_item) {
			for(int i=0;i<child_items.size();i++) {
				if (child_items[i].item==p_item)
					return i;
			}
			return -1;
		}
		void erase_item(CanvasItem *p_item) {
			int idx=find_item(p_item);
			if (idx>=0)
				child_items.remove(idx);
		}

		Canvas() { modulate=Color(1,1,1,1); }

	};


	RID_Owner<Rasterizer::CanvasLight> canvas_light_owner;


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
			Matrix32 transform;
			int layer;
		};

		Matrix32 global_transform;

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

		void add_aabb(const AABB& p_aabb) {


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
	void _render_canvas_item_tree(CanvasItem *p_canvas_item, const Matrix32& p_transform, const Rect2& p_clip_rect, const Color &p_modulate, Rasterizer::CanvasLight *p_lights);
	void _render_canvas_item(CanvasItem *p_canvas_item, const Matrix32& p_transform, const Rect2& p_clip_rect, float p_opacity, int p_z, Rasterizer::CanvasItem **z_list, Rasterizer::CanvasItem **z_last_list, CanvasItem *p_canvas_clip, CanvasItem *p_material_owner);
	void _render_canvas(Canvas *p_canvas, const Matrix32 &p_transform, Rasterizer::CanvasLight *p_lights, Rasterizer::CanvasLight *p_masked_lights);
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
public:

	virtual RID texture_create();
	virtual void texture_allocate(RID p_texture,int p_width, int p_height,Image::Format p_format,uint32_t p_flags=TEXTURE_FLAGS_DEFAULT);
	virtual void texture_set_data(RID p_texture,const Image& p_image,CubeMapSide p_cube_side=CUBEMAP_LEFT);
	virtual Image texture_get_data(RID p_texture,CubeMapSide p_cube_side=CUBEMAP_LEFT) const;
	virtual void texture_set_flags(RID p_texture,uint32_t p_flags) ;
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;
	virtual void texture_set_size_override(RID p_texture,int p_width, int p_height);
	virtual bool texture_can_stream(RID p_texture) const;
	virtual void texture_set_reload_hook(RID p_texture,ObjectID p_owner,const StringName& p_function) const;

	virtual void texture_set_path(RID p_texture,const String& p_path);
	virtual String texture_get_path(RID p_texture) const;

	virtual void texture_debug_usage(List<TextureInfo> *r_info);

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable);


	/* SHADER API */

	virtual RID shader_create(ShaderMode p_mode=SHADER_MATERIAL);

	virtual void shader_set_mode(RID p_shader,ShaderMode p_mode);
	virtual ShaderMode shader_get_mode(RID p_shader) const;

	virtual void shader_set_code(RID p_shader, const String& p_vertex, const String& p_fragment,const String& p_light,int p_vertex_ofs=0,int p_fragment_ofs=0,int p_light_ofs=0);
	virtual String shader_get_vertex_code(RID p_shader) const;
	virtual String shader_get_fragment_code(RID p_shader) const;
	virtual String shader_get_light_code(RID p_shader) const;

	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName& p_name, RID p_texture);
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName& p_name) const;


	/* COMMON MATERIAL API */

	virtual RID material_create();

	virtual void material_set_shader(RID p_shader_material, RID p_shader);
	virtual RID material_get_shader(RID p_shader_material) const;

	virtual void material_set_param(RID p_material, const StringName& p_param, const Variant& p_value);
	virtual Variant material_get_param(RID p_material, const StringName& p_param) const;

	virtual void material_set_flag(RID p_material, MaterialFlag p_flag,bool p_enabled);
	virtual bool material_get_flag(RID p_material,MaterialFlag p_flag) const;

	virtual void material_set_depth_draw_mode(RID p_material, MaterialDepthDrawMode p_mode);
	virtual MaterialDepthDrawMode material_get_depth_draw_mode(RID p_material) const;

	virtual void material_set_blend_mode(RID p_material,MaterialBlendMode p_mode);
	virtual MaterialBlendMode material_get_blend_mode(RID p_material) const;

	virtual void material_set_line_width(RID p_material,float p_line_width);
	virtual float material_get_line_width(RID p_material) const;

	/* FIXED MATERIAL */


	virtual RID fixed_material_create();

	virtual void fixed_material_set_flag(RID p_material, FixedMaterialFlags p_flag, bool p_enabled);
	virtual bool fixed_material_get_flag(RID p_material, FixedMaterialFlags p_flag) const;

	virtual void fixed_material_set_param(RID p_material, FixedMaterialParam p_parameter, const Variant& p_value);
	virtual Variant fixed_material_get_param(RID p_material,FixedMaterialParam p_parameter) const;

	virtual void fixed_material_set_texture(RID p_material,FixedMaterialParam p_parameter, RID p_texture);
	virtual RID fixed_material_get_texture(RID p_material,FixedMaterialParam p_parameter) const;

	virtual void fixed_material_set_texcoord_mode(RID p_material,FixedMaterialParam p_parameter, FixedMaterialTexCoordMode p_mode);
	virtual FixedMaterialTexCoordMode fixed_material_get_texcoord_mode(RID p_material,FixedMaterialParam p_parameter) const;


	virtual void fixed_material_set_uv_transform(RID p_material,const Transform& p_transform);
	virtual Transform fixed_material_get_uv_transform(RID p_material) const;

	virtual void fixed_material_set_light_shader(RID p_material,FixedMaterialLightShader p_shader);
	virtual FixedMaterialLightShader fixed_material_get_light_shader(RID p_material) const;

	virtual void fixed_material_set_point_size(RID p_material,float p_size);
	virtual float fixed_material_get_point_size(RID p_material) const;

	/* SURFACE API */
	virtual RID mesh_create();

	virtual void mesh_set_morph_target_count(RID p_mesh,int p_amount);
	virtual int mesh_get_morph_target_count(RID p_mesh) const;

	virtual void mesh_set_morph_target_mode(RID p_mesh,MorphTargetMode p_mode);
	virtual MorphTargetMode mesh_get_morph_target_mode(RID p_mesh) const;

	virtual void mesh_add_custom_surface(RID p_mesh,const Variant& p_dat); //this is used by each platform in a different way

	virtual void mesh_add_surface(RID p_mesh,PrimitiveType p_primitive,const Array& p_arrays,const Array& p_blend_shapes=Array(),bool p_alpha_sort=false);
	virtual Array mesh_get_surface_arrays(RID p_mesh,int p_surface) const;
	virtual Array mesh_get_surface_morph_arrays(RID p_mesh,int p_surface) const;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material,bool p_owned=false);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const;
	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const;
	virtual PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const;

	virtual void mesh_remove_surface(RID p_mesh,int p_index);
	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual void mesh_set_custom_aabb(RID p_mesh,const AABB& p_aabb);
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const;

	virtual void mesh_clear(RID p_mesh);

	/* MULTIMESH API */

	virtual RID multimesh_create();
	virtual void multimesh_set_instance_count(RID p_multimesh,int p_count);
	virtual int multimesh_get_instance_count(RID p_multimesh) const;

	virtual void multimesh_set_mesh(RID p_multimesh,RID p_mesh);
	virtual void multimesh_set_aabb(RID p_multimesh,const AABB& p_aabb);
	virtual void multimesh_instance_set_transform(RID p_multimesh,int p_index,const Transform& p_transform);
	virtual void multimesh_instance_set_color(RID p_multimesh,int p_index,const Color& p_color);

	virtual RID multimesh_get_mesh(RID p_multimesh) const;
	virtual AABB multimesh_get_aabb(RID p_multimesh,const AABB& p_aabb) const;
	virtual Transform multimesh_instance_get_transform(RID p_multimesh,int p_index) const;
	virtual Color multimesh_instance_get_color(RID p_multimesh,int p_index) const;

	virtual void multimesh_set_visible_instances(RID p_multimesh,int p_visible);
	virtual int multimesh_get_visible_instances(RID p_multimesh) const;

	/* IMMEDIATE API */

	virtual RID immediate_create();
	virtual void immediate_begin(RID p_immediate,PrimitiveType p_rimitive,RID p_texture=RID());
	virtual void immediate_vertex(RID p_immediate,const Vector3& p_vertex);
	virtual void immediate_normal(RID p_immediate,const Vector3& p_normal);
	virtual void immediate_tangent(RID p_immediate,const Plane& p_tangent);
	virtual void immediate_color(RID p_immediate,const Color& p_color);
	virtual void immediate_uv(RID p_immediate, const Vector2& p_uv);
	virtual void immediate_uv2(RID p_immediate,const Vector2& tex_uv);
	virtual void immediate_end(RID p_immediate);
	virtual void immediate_clear(RID p_immediate);
	virtual void immediate_set_material(RID p_immediate,RID p_material);
	virtual RID immediate_get_material(RID p_immediate) const;


	/* PARTICLES API */

	virtual RID particles_create();

	virtual void particles_set_amount(RID p_particles, int p_amount);
	virtual int particles_get_amount(RID p_particles) const;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting);
	virtual bool particles_is_emitting(RID p_particles) const;

	virtual void particles_set_visibility_aabb(RID p_particles, const AABB& p_visibility);
	virtual AABB particles_get_visibility_aabb(RID p_particles) const;

	virtual void particles_set_emission_half_extents(RID p_particles, const Vector3& p_half_extents);
	virtual Vector3 particles_get_emission_half_extents(RID p_particles) const;

	virtual void particles_set_emission_base_velocity(RID p_particles, const Vector3& p_base_velocity);
	virtual Vector3 particles_get_emission_base_velocity(RID p_particles) const;

	virtual void particles_set_emission_points(RID p_particles, const DVector<Vector3>& p_points);
	virtual DVector<Vector3> particles_get_emission_points(RID p_particles) const;

	virtual void particles_set_gravity_normal(RID p_particles, const Vector3& p_normal);
	virtual Vector3 particles_get_gravity_normal(RID p_particles) const;

	virtual void particles_set_variable(RID p_particles, ParticleVariable p_variable,float p_value);
	virtual float particles_get_variable(RID p_particles, ParticleVariable p_variable) const;

	virtual void particles_set_randomness(RID p_particles, ParticleVariable p_variable,float p_randomness);
	virtual float particles_get_randomness(RID p_particles, ParticleVariable p_variable) const;

	virtual void particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos);
	virtual float particles_get_color_phase_pos(RID p_particles, int p_phase) const;

	virtual void particles_set_color_phases(RID p_particles, int p_phases);
	virtual int particles_get_color_phases(RID p_particles) const;

	virtual void particles_set_color_phase_color(RID p_particles, int p_phase, const Color& p_color);
	virtual Color particles_get_color_phase_color(RID p_particles, int p_phase) const;

	virtual void particles_set_attractors(RID p_particles, int p_attractors);
	virtual int particles_get_attractors(RID p_particles) const;

	virtual void particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3& p_pos);
	virtual Vector3 particles_get_attractor_pos(RID p_particles,int p_attractor) const;

	virtual void particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force);
	virtual float particles_get_attractor_strength(RID p_particles,int p_attractor) const;

	virtual void particles_set_material(RID p_particles, RID p_material,bool p_owned=false);
	virtual RID particles_get_material(RID p_particles) const;

	virtual void particles_set_height_from_velocity(RID p_particles, bool p_enable);
	virtual bool particles_has_height_from_velocity(RID p_particles) const;

	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable);
	virtual bool particles_is_using_local_coordinates(RID p_particles) const;


	/* Light API */

	virtual RID light_create(LightType p_type);
	virtual LightType light_get_type(RID p_light) const;

	virtual void light_set_color(RID p_light,LightColor p_type, const Color& p_color);
	virtual Color light_get_color(RID p_light,LightColor p_type) const;


	virtual void light_set_shadow(RID p_light,bool p_enabled);
	virtual bool light_has_shadow(RID p_light) const;

	virtual void light_set_volumetric(RID p_light,bool p_enabled);
	virtual bool light_is_volumetric(RID p_light) const;

	virtual void light_set_projector(RID p_light,RID p_texture);
	virtual RID light_get_projector(RID p_light) const;

	virtual void light_set_param(RID p_light, LightParam p_var, float p_value);
	virtual float light_get_param(RID p_light, LightParam p_var) const;

	virtual void light_set_operator(RID p_light,LightOp p_op);
	virtual LightOp light_get_operator(RID p_light) const;

	virtual void light_omni_set_shadow_mode(RID p_light,LightOmniShadowMode p_mode);
	virtual LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) const;

	virtual void light_directional_set_shadow_mode(RID p_light,LightDirectionalShadowMode p_mode);
	virtual LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) const;
	virtual void light_directional_set_shadow_param(RID p_light,LightDirectionalShadowParam p_param, float p_value);
	virtual float light_directional_get_shadow_param(RID p_light,LightDirectionalShadowParam p_param) const;


	/* SKELETON API */

	virtual RID skeleton_create();
	virtual void skeleton_resize(RID p_skeleton,int p_bones);
	virtual int skeleton_get_bone_count(RID p_skeleton) const;
	virtual void skeleton_bone_set_transform(RID p_skeleton,int p_bone, const Transform& p_transform);
	virtual Transform skeleton_bone_get_transform(RID p_skeleton,int p_bone);

	/* ROOM API */

	virtual RID room_create();
	virtual void room_set_bounds(RID p_room, const BSP_Tree& p_bounds);
	virtual BSP_Tree room_get_bounds(RID p_room) const;

	/* PORTAL API */

	virtual RID portal_create();
	virtual void portal_set_shape(RID p_portal, const Vector<Point2>& p_shape);
	virtual Vector<Point2> portal_get_shape(RID p_portal) const;
	virtual void portal_set_enabled(RID p_portal, bool p_enabled);
	virtual bool portal_is_enabled(RID p_portal) const;
	virtual void portal_set_disable_distance(RID p_portal, float p_distance);
	virtual float portal_get_disable_distance(RID p_portal) const;
	virtual void portal_set_disabled_color(RID p_portal, const Color& p_color);
	virtual Color portal_get_disabled_color(RID p_portal) const;
	virtual void portal_set_connect_range(RID p_portal, float p_range);
	virtual float portal_get_connect_range(RID p_portal) const;

	/* BAKED LIGHT */

	virtual RID baked_light_create();

	virtual void baked_light_set_mode(RID p_baked_light,BakedLightMode p_mode);
	virtual BakedLightMode baked_light_get_mode(RID p_baked_light) const;

	virtual void baked_light_set_octree(RID p_baked_light,const DVector<uint8_t> p_octree);
	virtual DVector<uint8_t> baked_light_get_octree(RID p_baked_light) const;

	virtual void baked_light_set_light(RID p_baked_light,const DVector<uint8_t> p_light);
	virtual DVector<uint8_t> baked_light_get_light(RID p_baked_light) const;

	virtual void baked_light_set_sampler_octree(RID p_baked_light,const DVector<int> &p_sampler);
	virtual DVector<int> baked_light_get_sampler_octree(RID p_baked_light) const;

	virtual void baked_light_set_lightmap_multiplier(RID p_baked_light,float p_multiplier);
	virtual float baked_light_get_lightmap_multiplier(RID p_baked_light) const;

	virtual void baked_light_add_lightmap(RID p_baked_light,const RID p_texture,int p_id);
	virtual void baked_light_clear_lightmaps(RID p_baked_light);

	virtual void baked_light_set_realtime_color_enabled(RID p_baked_light, const bool p_enabled);
	virtual bool baked_light_get_realtime_color_enabled(RID p_baked_light) const;

	virtual void baked_light_set_realtime_color(RID p_baked_light, const Color& p_color);
	virtual Color baked_light_get_realtime_color(RID p_baked_light) const;

	virtual void baked_light_set_realtime_energy(RID p_baked_light, const float p_energy);
	virtual float baked_light_get_realtime_energy(RID p_baked_light) const;

	/* BAKED LIGHT SAMPLER */

	virtual RID baked_light_sampler_create();

	virtual void baked_light_sampler_set_param(RID p_baked_light_sampler,BakedLightSamplerParam p_param,float p_value);
	virtual float baked_light_sampler_get_param(RID p_baked_light_sampler,BakedLightSamplerParam p_param) const;

	virtual void baked_light_sampler_set_resolution(RID p_baked_light_sampler,int p_resolution);
	virtual int baked_light_sampler_get_resolution(RID p_baked_light_sampler) const;

	/* CAMERA API */

	virtual RID camera_create();
	virtual void camera_set_perspective(RID p_camera,float p_fovy_degrees, float p_z_near, float p_z_far);
	virtual void camera_set_orthogonal(RID p_camera,float p_size, float p_z_near, float p_z_far);
	virtual void camera_set_transform(RID p_camera,const Transform& p_transform);

	virtual void camera_set_visible_layers(RID p_camera,uint32_t p_layers);
	virtual uint32_t camera_get_visible_layers(RID p_camera) const;

	virtual void camera_set_environment(RID p_camera,RID p_env);
	virtual RID camera_get_environment(RID p_camera) const;

	virtual void camera_set_use_vertical_aspect(RID p_camera,bool p_enable);
	virtual bool camera_is_using_vertical_aspect(RID p_camera,bool p_enable) const;

	/* VIEWPORT API */

	virtual RID viewport_create();

	virtual void viewport_attach_to_screen(RID p_viewport,int p_screen=0);
	virtual void viewport_detach(RID p_viewport);

	virtual void viewport_set_as_render_target(RID p_viewport,bool p_enable);
	virtual void viewport_set_render_target_update_mode(RID p_viewport,RenderTargetUpdateMode p_mode);
	virtual RenderTargetUpdateMode viewport_get_render_target_update_mode(RID p_viewport) const;
	virtual RID viewport_get_render_target_texture(RID p_viewport) const;
	virtual void viewport_set_render_target_vflip(RID p_viewport,bool p_enable);
	virtual bool viewport_get_render_target_vflip(RID p_viewport) const;
	virtual void viewport_set_render_target_clear_on_new_frame(RID p_viewport,bool p_enable);
	virtual bool viewport_get_render_target_clear_on_new_frame(RID p_viewport) const;
	virtual void viewport_render_target_clear(RID p_viewport);
	virtual void viewport_set_render_target_to_screen_rect(RID p_viewport,const Rect2& p_rect);


	virtual void viewport_queue_screen_capture(RID p_viewport);
	virtual Image viewport_get_screen_capture(RID p_viewport) const;

	virtual void viewport_set_rect(RID p_viewport,const ViewportRect& p_rect);
	virtual ViewportRect viewport_get_rect(RID p_viewport) const;

	virtual void viewport_set_hide_scenario(RID p_viewport,bool p_hide);
	virtual void viewport_set_hide_canvas(RID p_viewport,bool p_hide);
	virtual void viewport_set_disable_environment(RID p_viewport,bool p_disable);
	virtual void viewport_attach_camera(RID p_viewport,RID p_camera);
	virtual void viewport_set_scenario(RID p_viewport,RID p_scenario);

	virtual RID viewport_get_attached_camera(RID  p_viewport) const;
	virtual RID viewport_get_scenario(RID  p_viewport) const;
	virtual void viewport_attach_canvas(RID p_viewport,RID p_canvas);
	virtual void viewport_remove_canvas(RID p_viewport,RID p_canvas);
	virtual void viewport_set_canvas_transform(RID p_viewport,RID p_canvas,const Matrix32& p_offset);
	virtual Matrix32 viewport_get_canvas_transform(RID p_viewport,RID p_canvas) const;
	virtual void viewport_set_global_canvas_transform(RID p_viewport,const Matrix32& p_transform);
	virtual Matrix32 viewport_get_global_canvas_transform(RID p_viewport) const;
	virtual void viewport_set_canvas_layer(RID p_viewport,RID p_canvas,int p_layer);
	virtual void viewport_set_transparent_background(RID p_viewport,bool p_enabled);
	virtual bool viewport_has_transparent_background(RID p_viewport) const;


	/* ENVIRONMENT API */

	virtual RID environment_create();

	virtual void environment_set_background(RID p_env,EnvironmentBG p_bg);
	virtual EnvironmentBG environment_get_background(RID p_env) const;

	virtual void environment_set_background_param(RID p_env,EnvironmentBGParam p_param, const Variant& p_value);
	virtual Variant environment_get_background_param(RID p_env,EnvironmentBGParam p_param) const;

	virtual void environment_set_enable_fx(RID p_env,EnvironmentFx p_effect,bool p_enabled);
	virtual bool environment_is_fx_enabled(RID p_env,EnvironmentFx p_effect) const;


	virtual void environment_fx_set_param(RID p_env,EnvironmentFxParam p_effect,const Variant& p_param);
	virtual Variant environment_fx_get_param(RID p_env,EnvironmentFxParam p_effect) const;


	/* SCENARIO API */

	virtual RID scenario_create();

	virtual void scenario_set_debug(RID p_scenario,ScenarioDebugMode p_debug_mode);
	virtual void scenario_set_environment(RID p_scenario, RID p_environment);
	virtual RID scenario_get_environment(RID p_scenario, RID p_environment) const;
	virtual void scenario_set_fallback_environment(RID p_scenario, RID p_environment);



	/* INSTANCING API */

	virtual RID instance_create();

	virtual void instance_set_base(RID p_instance, RID p_base);
	virtual RID instance_get_base(RID p_instance) const;

	virtual void instance_set_scenario(RID p_instance, RID p_scenario);
	virtual RID instance_get_scenario(RID p_instance) const;

	virtual void instance_set_layer_mask(RID p_instance, uint32_t p_mask);
	virtual uint32_t instance_get_layer_mask(RID p_instance) const;

	virtual AABB instance_get_base_aabb(RID p_instance) const;

	virtual void instance_attach_object_instance_ID(RID p_instance,uint32_t p_ID);
	virtual uint32_t instance_get_object_instance_ID(RID p_instance) const;

	virtual void instance_attach_skeleton(RID p_instance,RID p_skeleton);
	virtual RID instance_get_skeleton(RID p_instance) const;

	virtual void instance_set_morph_target_weight(RID p_instance,int p_shape, float p_weight);
	virtual float instance_get_morph_target_weight(RID p_instance,int p_shape) const;

	virtual void instance_set_surface_material(RID p_instance,int p_surface, RID p_material);


	virtual void instance_set_transform(RID p_instance, const Transform& p_transform);
	virtual Transform instance_get_transform(RID p_instance) const;

	virtual void instance_set_exterior( RID p_instance, bool p_enabled );
	virtual bool instance_is_exterior( RID p_instance) const;

	virtual void instance_set_room( RID p_instance, RID p_room );
	virtual RID instance_get_room( RID p_instance ) const ;

	virtual void instance_set_extra_visibility_margin( RID p_instance, real_t p_margin );
	virtual real_t instance_get_extra_visibility_margin( RID p_instance ) const;

	virtual Vector<RID> instances_cull_aabb(const AABB& p_aabb, RID p_scenario=RID()) const;
	virtual Vector<RID> instances_cull_ray(const Vector3& p_from, const Vector3& p_to, RID p_scenario=RID()) const;
	virtual Vector<RID> instances_cull_convex(const Vector<Plane>& p_convex, RID p_scenario=RID()) const;

	virtual void instance_geometry_set_flag(RID p_instance,InstanceFlags p_flags,bool p_enabled);
	virtual bool instance_geometry_get_flag(RID p_instance,InstanceFlags p_flags) const;

	virtual void instance_geometry_set_cast_shadows_setting(RID p_instance, VS::ShadowCastingSetting p_shadow_casting_setting);
	virtual VS::ShadowCastingSetting instance_geometry_get_cast_shadows_setting(RID p_instance) const;

	virtual void instance_geometry_set_material_override(RID p_instance, RID p_material);
	virtual RID instance_geometry_get_material_override(RID p_instance) const;

	virtual void instance_geometry_set_draw_range(RID p_instance,float p_min,float p_max);
	virtual float instance_geometry_get_draw_range_max(RID p_instance) const;
	virtual float instance_geometry_get_draw_range_min(RID p_instance) const;

	virtual void instance_geometry_set_baked_light(RID p_instance,RID p_baked_light);
	virtual RID instance_geometry_get_baked_light(RID p_instance) const;

	virtual void instance_geometry_set_baked_light_sampler(RID p_instance,RID p_baked_light_sampler);
	virtual RID instance_geometry_get_baked_light_sampler(RID p_instance) const;

	virtual void instance_geometry_set_baked_light_texture_index(RID p_instance,int p_tex_id);
	virtual int instance_geometry_get_baked_light_texture_index(RID p_instance) const;

	virtual void instance_light_set_enabled(RID p_instance,bool p_enabled);
	virtual bool instance_light_is_enabled(RID p_instance) const;

	/* CANVAS (2D) */

	virtual RID canvas_create();
	virtual void canvas_set_item_mirroring(RID p_canvas,RID p_item,const Point2& p_mirroring);
	virtual Point2 canvas_get_item_mirroring(RID p_canvas,RID p_item) const;
	virtual void canvas_set_modulate(RID p_canvas,const Color& p_color);


	virtual RID canvas_item_create();

	virtual void canvas_item_set_parent(RID p_item,RID p_parent_item);
	virtual RID canvas_item_get_parent(RID p_canvas_item) const;

	virtual void canvas_item_set_visible(RID p_item,bool p_visible);
	virtual bool canvas_item_is_visible(RID p_item) const;

	virtual void canvas_item_set_blend_mode(RID p_canvas_item,MaterialBlendMode p_blend);
	virtual void canvas_item_set_light_mask(RID p_canvas_item,int p_mask);



	//virtual void canvas_item_set_rect(RID p_item, const Rect2& p_rect);
	virtual void canvas_item_set_transform(RID p_item, const Matrix32& p_transform);
	virtual void canvas_item_set_clip(RID p_item, bool p_clip);
	virtual void canvas_item_set_distance_field_mode(RID p_item, bool p_enable);
	virtual void canvas_item_set_custom_rect(RID p_item, bool p_custom_rect,const Rect2& p_rect=Rect2());
	virtual void canvas_item_set_opacity(RID p_item, float p_opacity);
	virtual float canvas_item_get_opacity(RID p_item, float p_opacity) const;
	virtual void canvas_item_set_on_top(RID p_item, bool p_on_top);
	virtual bool canvas_item_is_on_top(RID p_item) const;

	virtual void canvas_item_set_self_opacity(RID p_item, float p_self_opacity);
	virtual float canvas_item_get_self_opacity(RID p_item, float p_self_opacity) const;

	virtual void canvas_item_attach_viewport(RID p_item, RID p_viewport);

	virtual void canvas_item_add_line(RID p_item, const Point2& p_from, const Point2& p_to, const Color& p_color, float p_width=1.0, bool p_antialiased=false);
	virtual void canvas_item_add_rect(RID p_item, const Rect2& p_rect, const Color& p_color);
	virtual void canvas_item_add_circle(RID p_item, const Point2& p_pos, float p_radius,const Color& p_color);
	virtual void canvas_item_add_texture_rect(RID p_item, const Rect2& p_rect, RID p_texture,bool p_tile=false,const Color& p_modulate=Color(1,1,1),bool p_transpose=false);
	virtual void canvas_item_add_texture_rect_region(RID p_item, const Rect2& p_rect, RID p_texture,const Rect2& p_src_rect,const Color& p_modulate=Color(1,1,1),bool p_transpose=false);
	virtual void canvas_item_add_style_box(RID p_item, const Rect2& p_rect, const Rect2& p_source, RID p_texture,const Vector2& p_topleft, const Vector2& p_bottomright, bool p_draw_center=true,const Color& p_modulate=Color(1,1,1));
	virtual void canvas_item_add_primitive(RID p_item, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture,float p_width=1.0);
	virtual void canvas_item_add_polygon(RID p_item, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs=Vector<Point2>(), RID p_texture=RID());
	virtual void canvas_item_add_triangle_array(RID p_item, const Vector<int>& p_indices, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs=Vector<Point2>(), RID p_texture=RID(), int p_count=-1);
	virtual void canvas_item_add_triangle_array_ptr(RID p_item, int p_count, const int* p_indices, const Point2* p_points, const Color* p_colors,const Point2* p_uvs=NULL, RID p_texture=RID());
	virtual void canvas_item_add_set_transform(RID p_item,const Matrix32& p_transform);
	virtual void canvas_item_add_set_blend_mode(RID p_item, MaterialBlendMode p_blend);
	virtual void canvas_item_add_clip_ignore(RID p_item, bool p_ignore);
	virtual void canvas_item_set_sort_children_by_y(RID p_item, bool p_enable);
	virtual void canvas_item_set_z(RID p_item, int p_z);
	virtual void canvas_item_set_z_as_relative_to_parent(RID p_item, bool p_enable);
	virtual void canvas_item_set_copy_to_backbuffer(RID p_item, bool p_enable,const Rect2& p_rect);

	virtual void canvas_item_set_material(RID p_item, RID p_material);
	virtual void canvas_item_set_use_parent_material(RID p_item, bool p_enable);

	virtual RID canvas_light_create();
	virtual void canvas_light_attach_to_canvas(RID p_light,RID p_canvas);
	virtual void canvas_light_set_enabled(RID p_light, bool p_enabled);
	virtual void canvas_light_set_transform(RID p_light, const Matrix32& p_transform);
	virtual void canvas_light_set_scale(RID p_light, float p_scale);
	virtual void canvas_light_set_texture(RID p_light, RID p_texture);
	virtual void canvas_light_set_texture_offset(RID p_light, const Vector2& p_offset);
	virtual void canvas_light_set_color(RID p_light, const Color& p_color);
	virtual void canvas_light_set_height(RID p_light, float p_height);
	virtual void canvas_light_set_energy(RID p_light, float p_energy);
	virtual void canvas_light_set_z_range(RID p_light, int p_min_z,int p_max_z);
	virtual void canvas_light_set_layer_range(RID p_light, int p_min_layer,int p_max_layer);
	virtual void canvas_light_set_item_mask(RID p_light, int p_mask);
	virtual void canvas_light_set_item_shadow_mask(RID p_light, int p_mask);

	virtual void canvas_light_set_mode(RID p_light, CanvasLightMode p_mode);
	virtual void canvas_light_set_shadow_enabled(RID p_light, bool p_enabled);
	virtual void canvas_light_set_shadow_buffer_size(RID p_light, int p_size);
	virtual void canvas_light_set_shadow_esm_multiplier(RID p_light, float p_multiplier);
	virtual void canvas_light_set_shadow_color(RID p_light, const Color& p_color);



	virtual RID canvas_light_occluder_create();
	virtual void canvas_light_occluder_attach_to_canvas(RID p_occluder,RID p_canvas);
	virtual void canvas_light_occluder_set_enabled(RID p_occluder,bool p_enabled);
	virtual void canvas_light_occluder_set_polygon(RID p_occluder,RID p_polygon);
	virtual void canvas_light_occluder_set_transform(RID p_occluder,const Matrix32& p_xform);
	virtual void canvas_light_occluder_set_light_mask(RID p_occluder,int p_mask);


	virtual RID canvas_occluder_polygon_create();
	virtual void canvas_occluder_polygon_set_shape(RID p_occluder_polygon,const DVector<Vector2>& p_shape,bool p_close);
	virtual void canvas_occluder_polygon_set_shape_as_lines(RID p_occluder_polygon,const DVector<Vector2>& p_shape);
	virtual void canvas_occluder_polygon_set_cull_mode(RID p_occluder_polygon,CanvasOccluderPolygonCullMode p_mode);



	virtual void canvas_item_clear(RID p_item);
	virtual void canvas_item_raise(RID p_item);

	/* CANVAS ITEM MATERIAL */

	virtual RID canvas_item_material_create();
	virtual void canvas_item_material_set_shader(RID p_material, RID p_shader);
	virtual void canvas_item_material_set_shader_param(RID p_material, const StringName& p_param, const Variant& p_value);
	virtual Variant canvas_item_material_get_shader_param(RID p_material, const StringName& p_param) const;
	virtual void canvas_item_material_set_shading_mode(RID p_material, CanvasItemShadingMode p_mode);



	/* CURSOR */
	virtual void cursor_set_rotation(float p_rotation, int p_cursor = 0); // radians
	virtual void cursor_set_texture(RID p_texture, const Point2 &p_center_offset, int p_cursor=0, const Rect2 &p_region=Rect2());
	virtual void cursor_set_visible(bool p_visible, int p_cursor = 0);
	virtual void cursor_set_pos(const Point2& p_pos, int p_cursor = 0);

	/* BLACK BARS */

	virtual void black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom);
	virtual void black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom);

	/* FREE */

	virtual void free( RID p_rid );

	/* CUSTOM SHADE MODEL */

	virtual void custom_shade_model_set_shader(int p_model, RID p_shader);
	virtual RID custom_shade_model_get_shader(int p_model) const;
	virtual void custom_shade_model_set_name(int p_model, const String& p_name);
	virtual String custom_shade_model_get_name(int p_model) const;
	virtual void custom_shade_model_set_param_info(int p_model, const List<PropertyInfo>& p_info);
	virtual void custom_shade_model_get_param_info(int p_model, List<PropertyInfo>* p_info) const;

	/* EVENT QUEUING */

	virtual void draw();
	virtual void sync();

	virtual void init();
	virtual void finish();

	virtual bool has_changed() const;

	/* RENDER INFO */

	virtual int get_render_info(RenderInfo p_info);
	virtual bool has_feature(Features p_feature) const;

	RID get_test_cube();

	virtual void set_boot_image(const Image& p_image, const Color& p_color, bool p_scale);
	virtual void set_default_clear_color(const Color& p_color);
	virtual Color get_default_clear_color() const;

	VisualServerRaster(Rasterizer *p_rasterizer);
	~VisualServerRaster();

};

#endif
