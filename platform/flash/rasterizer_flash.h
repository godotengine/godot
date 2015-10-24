/*************************************************/
/*  rasterizer_gles2.h                            */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef RASTERIZER_FLASH_H
#define RASTERIZER_FLASH_H

#include "servers/visual/rasterizer.h"

#include "servers/visual/shader_language.h"
#include "image.h"
#include "rid.h"
#include "servers/visual_server.h"
#include "list.h"
#include "map.h"
#include "camera_matrix.h"
#include "sort.h"
#include "self_list.h"

#include "platform_config.h"

#include "servers/visual/particle_system_sw.h"

/**
        @author Juan Linietsky <reduzio@gmail.com>
*/
class RasterizerFlash : public Rasterizer {

	enum {

		MAX_SCENE_LIGHTS=2048,
		LIGHT_SPOT_BIT=0x80,
		DEFAULT_SKINNED_BUFFER_SIZE = 1024 * 1024, // 10k vertices
		MAX_HW_LIGHTS = 1,
	};

	uint8_t *skinned_buffer;
	int skinned_buffer_size;
	bool pvr_supported;
	bool s3tc_supported;
	bool etc_supported;
	bool npo2_textures_available;

	struct Texture {

		uint32_t flags;
		int width,height;
		int alloc_width, alloc_height;
		Image::Format format;

		int mipmaps;
		int data_size; //original data size, useful for retrieving back
		bool has_alpha;
		bool format_has_alpha;
		bool compressed;
		bool disallow_mipmaps;
		int total_data_size;

		Image image[6];

		bool active;
		bool gen_mipmap;

		Texture() {

			flags=width=height=0;
			data_size=0;
			format=Image::FORMAT_GRAYSCALE;
			format_has_alpha=false;
			has_alpha=false;
			active=false;
			disallow_mipmaps=false;
			gen_mipmap=true;
			compressed=false;
			total_data_size=0;
		}

		~Texture() {

		}
	};

	mutable RID_Owner<Texture> texture_owner;

	struct Shader {

		String vertex_code;
		String fragment_code;
		int vertex_line;
		int fragment_line;
		VS::ShaderMode mode;

		uint32_t custom_code_id;
		uint32_t version;


		bool valid;
		bool has_alpha;

		Map<StringName,ShaderLanguage::Uniform> uniforms;


		SelfList<Shader> dirty_list;

		Shader() : dirty_list(this) {

			valid=false;
			custom_code_id=0;
			has_alpha=false;
			version=1;
			vertex_line=0;
			fragment_line=0;
		}


	};

	mutable RID_Owner<Shader> shader_owner;
	mutable SelfList<Shader>::List _shader_dirty_list;
	_FORCE_INLINE_ void _shader_make_dirty(Shader* p_shader);
	void _update_shader( Shader* p_shader) const;

	struct Material {

		bool flags[VS::MATERIAL_FLAG_MAX];
		bool hints[VS::MATERIAL_HINT_MAX];

		VS::MaterialShadeModel shade_model;
		VS::MaterialBlendMode blend_mode;

		float line_width;
		bool has_alpha;

		mutable uint32_t shader_version;

		RID shader; // shader material
		Shader *shader_cache;

		struct UniformData {

			bool istexture;
			Variant value;
			int index;
		};

		mutable Map<StringName,UniformData> shader_params;

		uint64_t last_pass;


		Material() {

			for(int i=0;i<VS::MATERIAL_FLAG_MAX;i++)
				flags[i]=false;
			flags[VS::MATERIAL_FLAG_VISIBLE]=true;
			for(int i=0;i<VS::MATERIAL_HINT_MAX;i++)
				hints[i]=false;

			line_width=1;
			has_alpha=false;
			blend_mode=VS::MATERIAL_BLEND_MODE_MIX;
			last_pass = 0;
			shader_version=0;
			shader_cache=NULL;

		}
	};

	_FORCE_INLINE_ void _update_material_shader_params(Material *p_material) const;
	mutable RID_Owner<Material> material_owner;



	struct Geometry {

		enum Type {
			GEOMETRY_INVALID,
			GEOMETRY_SURFACE,
			GEOMETRY_POLY,
			GEOMETRY_PARTICLES,
			GEOMETRY_MULTISURFACE,
		};

		Type type;
		RID material;
		bool has_alpha;
		bool material_owned;

		Geometry() { has_alpha=false; material_owned = false; }
		virtual ~Geometry() {};
	};

	struct GeometryOwner {

		virtual ~GeometryOwner() {}
	};

	class Mesh;

	struct Surface : public Geometry {

		struct ArrayData {

			uint32_t ofs,size,datatype,count;
			bool normalize;
			bool bind;

			ArrayData() { ofs=0; size=0; count=0; datatype=0; normalize=0; bind=false;}
		};

		Mesh *mesh;

		Array data;
		Array morph_data;
		ArrayData array[VS::ARRAY_MAX];

		// no support for the above, array in localmem.
		uint8_t *array_local;
		uint8_t *index_array_local;

		bool packed;

		struct MorphTarget {
			uint32_t configured_format;
			uint8_t *array;
		};

		MorphTarget* morph_targets_local;
		int morph_target_count;
		AABB aabb;

		int array_len;
		int index_array_len;
		int max_bone;

		float vertex_scale;
		float uv_scale;
		float uv2_scale;

		VS::PrimitiveType primitive;

		uint32_t format;
		uint32_t configured_format;

		int stride;
		int local_stride;
		uint32_t morph_format;

		bool active;

		Point2 uv_min;
		Point2 uv_max;

		Surface() {

			array_len=0;
			local_stride=0;
			morph_format=0;
			type=GEOMETRY_SURFACE;
			primitive=VS::PRIMITIVE_POINTS;
			index_array_len=0;
			vertex_scale=1.0;
			uv_scale=1.0;
			uv2_scale=1.0;

			format=0;
			stride=0;
			morph_targets_local=0;
			morph_target_count=0;

			array_local = index_array_local = 0;

			active=false;
			packed=false;
		}

		~Surface() {

		}
	};


	struct Mesh {

		bool active;
		Vector<Surface*> surfaces;
		int morph_target_count;
		VS::MorphTargetMode morph_target_mode;

		mutable uint64_t last_pass;
		Mesh() {
			morph_target_mode=VS::MORPH_MODE_NORMALIZED;
			morph_target_count=0;
			last_pass=0;
			active=false;
		}
	};
	mutable RID_Owner<Mesh> mesh_owner;

	Error _surface_set_arrays(Surface *p_surface, uint8_t *p_mem,uint8_t *p_index_mem,const Array& p_arrays,bool p_main);


	struct MultiMesh;

	struct MultiMeshSurface : public Geometry {

		Surface *surface;
		MultiMeshSurface() { type=GEOMETRY_MULTISURFACE; }
	};

	struct MultiMesh : public GeometryOwner {

		struct Element {

			float matrix[16];
			uint8_t color[4];
		};

		AABB aabb;
		RID mesh;
		int visible;

		//IDirect3DVertexBuffer9* instance_buffer;
		Vector<Element> elements;
		Vector<MultiMeshSurface> cache_surfaces;
		mutable uint64_t last_pass;

		MultiMesh() {

			last_pass=0;
			visible = -1;
		}
	};

	mutable RID_Owner<MultiMesh> multimesh_owner;

	struct Particles : public Geometry {

		ParticleSystemSW data; // software particle system

		Particles() {
			type=GEOMETRY_PARTICLES;

		}
	};

	mutable RID_Owner<Particles> particles_owner;

	struct ParticlesInstance : public GeometryOwner {

		RID particles;

		ParticleSystemProcessSW particles_process;
		Transform transform;

		ParticlesInstance() {  }
	};

	mutable RID_Owner<ParticlesInstance> particles_instance_owner;
	ParticleSystemDrawInfoSW particle_draw_info;

	struct Skeleton {

		Vector<Transform> bones;

	};

	mutable RID_Owner<Skeleton> skeleton_owner;


	struct Light {

		VS::LightType type;
		float vars[VS::LIGHT_PARAM_MAX];
		Color colors[3];
		bool shadow_enabled;
		RID projector;
		bool volumetric_enabled;
		Color volumetric_color;


		Light() {

			vars[VS::LIGHT_PARAM_SPOT_ATTENUATION]=1;
			vars[VS::LIGHT_PARAM_SPOT_ANGLE]=45;
			vars[VS::LIGHT_PARAM_ATTENUATION]=1.0;
			vars[VS::LIGHT_PARAM_ENERGY]=1.0;
			vars[VS::LIGHT_PARAM_RADIUS]=1.0;
			colors[VS::LIGHT_COLOR_AMBIENT]=Color(0,0,0);
			colors[VS::LIGHT_COLOR_DIFFUSE]=Color(1,1,1);
			colors[VS::LIGHT_COLOR_SPECULAR]=Color(1,1,1);
			shadow_enabled=false;
			volumetric_enabled=false;
		}
	};

	struct Environment {


		VS::EnvironmentBG bg_mode;
		Variant bg_param[VS::ENV_BG_PARAM_MAX];
		bool fx_enabled[VS::ENV_FX_MAX];
		Variant fx_param[VS::ENV_FX_PARAM_MAX];

		Environment() {

			bg_mode=VS::ENV_BG_DEFAULT_COLOR;
			bg_param[VS::ENV_BG_PARAM_COLOR]=Color(0,0,0);
			bg_param[VS::ENV_BG_PARAM_TEXTURE]=RID();
			bg_param[VS::ENV_BG_PARAM_CUBEMAP]=RID();
			bg_param[VS::ENV_BG_PARAM_ENERGY]=1.0;

			for(int i=0;i<VS::ENV_FX_MAX;i++)
				fx_enabled[i]=false;

			fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_PASSES]=1;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLOOM]=0.0;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLOOM_TRESHOLD]=0.5;
			fx_param[VS::ENV_FX_PARAM_DOF_BLUR_PASSES]=1;
			fx_param[VS::ENV_FX_PARAM_DOF_BLUR_BEGIN]=100.0;
			fx_param[VS::ENV_FX_PARAM_DOF_BLUR_RANGE]=10.0;
			fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE]=0.4;
			fx_param[VS::ENV_FX_PARAM_HDR_SCALAR]=1.0;
			fx_param[VS::ENV_FX_PARAM_HDR_GLOW_TRESHOLD]=0.95;
			fx_param[VS::ENV_FX_PARAM_HDR_GLOW_SCALE]=0.2;
			fx_param[VS::ENV_FX_PARAM_HDR_MIN_LUMINANCE]=0.4;
			fx_param[VS::ENV_FX_PARAM_HDR_MAX_LUMINANCE]=8.0;
			fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED]=0.5;
			fx_param[VS::ENV_FX_PARAM_FOG_BEGIN]=100.0;
			fx_param[VS::ENV_FX_PARAM_FOG_ATTENUATION]=1.0;
			fx_param[VS::ENV_FX_PARAM_FOG_BEGIN_COLOR]=Color(0,0,0);
			fx_param[VS::ENV_FX_PARAM_FOG_END_COLOR]=Color(0,0,0);
			fx_param[VS::ENV_FX_PARAM_FOG_BG]=true;
			fx_param[VS::ENV_FX_PARAM_BCS_BRIGHTNESS]=1.0;
			fx_param[VS::ENV_FX_PARAM_BCS_CONTRAST]=1.0;
			fx_param[VS::ENV_FX_PARAM_BCS_SATURATION]=1.0;
			fx_param[VS::ENV_FX_PARAM_SRGB_CONVERT]=1.0;

		}

	};

	mutable RID_Owner<Environment> environment_owner;


	struct ShadowBuffer;

	struct LightInstance {

		struct SplitInfo {

			CameraMatrix camera;
			Transform transform;
			float near;
			float far;
		};

		RID light;
		Light *base;
		Transform transform;
		CameraMatrix projection;

		Transform custom_transform;
		CameraMatrix custom_projection;

		Vector3 light_vector;
		Vector3 spot_vector;
		float linear_att;

		uint64_t shadow_pass;
		uint64_t last_pass;
		uint16_t sort_key;

		Vector<ShadowBuffer*> shadow_buffers;

		void clear_shadow_buffers() {

			for (int i=0;i<shadow_buffers.size();i++) {

				ShadowBuffer *sb=shadow_buffers[i];
				ERR_CONTINUE( sb->owner != this );

				sb->owner=NULL;
			}

			shadow_buffers.clear();
		}

		LightInstance() { shadow_pass=0; last_pass=0; sort_key=0; }

	};
	mutable RID_Owner<Light> light_owner;
	mutable RID_Owner<LightInstance> light_instance_owner;

	LightInstance *light_instances[MAX_SCENE_LIGHTS];
	LightInstance *directional_lights[4];
	int light_instance_count;
	int directional_light_count;
	int last_light_id;
	bool current_depth_test;
	bool current_depth_mask;

	bool fragment_lighting;


	struct RenderList {

		enum {
			MAX_ELEMENTS=4096,
			MAX_LIGHTS=4
		};

		struct Element {


			float depth;
			const InstanceData *instance;
			const Skeleton *skeleton;
			union {
				uint16_t lights[MAX_HW_LIGHTS];
				uint64_t light_key;
			};
			uint16_t light_count;
			const Geometry *geometry;
			const Geometry *geometry_cmp;
			const Material *material;
			const GeometryOwner *owner;
			bool *additive_ptr;
			uint8_t light_type;
			bool additive;
			bool mirror;
		};


		Element _elements[MAX_ELEMENTS];
		Element *elements[MAX_ELEMENTS];
		int element_count;

		void clear() {

			element_count=0;
		}

		struct SortZ {

			_FORCE_INLINE_ bool operator()(const Element* A,  const Element* B ) const {

				return A->depth > B->depth;
			}
		};

		void sort_z() {

			SortArray<Element*,SortZ> sorter;
			sorter.sort(elements,element_count);
		}


		struct SortMat {

			_FORCE_INLINE_ bool operator()(const Element* A,  const Element* B ) const {
				// TODO move to a single uint64 (one comparison)
				if (A->material->shader_cache == B->material->shader_cache) {
					if (A->material == B->material) {

						return A->light_key < B->light_key;
					} else {

						return (A->material < B->material);
					}
				} else {

					return A->material->shader_cache < B->material->shader_cache;
				}
			}
		};

		void sort_mat() {

			SortArray<Element*,SortMat> sorter;
			sorter.sort(elements,element_count);
		}

		struct SortMatLight {

			_FORCE_INLINE_ bool operator()(const Element* A,  const Element* B ) const {

				if (A->geometry_cmp == B->geometry_cmp) {

					if (A->material == B->material) {

						return A->light_key<B->light_key;
					} else {

						return (A->material < B->material);
					}
				} else {

					return (A->geometry_cmp < B->geometry_cmp);
				}
			}
		};

		void sort_mat_light() {

			SortArray<Element*,SortMatLight> sorter;
			sorter.sort(elements,element_count);
		}

		struct SortMatLightType {

			_FORCE_INLINE_ bool operator()(const Element* A,  const Element* B ) const {

				if (A->light_type == B->light_type) {
					if (A->geometry_cmp == B->geometry_cmp) {

						return (A->material < B->material);
					} else {

						return (A->geometry_cmp < B->geometry_cmp);
					}
				} else {

					return A->light_type < B->light_type;
				}
			}
		};

		void sort_mat_light_type() {

			SortArray<Element*,SortMatLightType> sorter;
			sorter.sort(elements,element_count);
		}

		_FORCE_INLINE_ Element* add_element() {

			if (element_count>MAX_ELEMENTS)
				return NULL;
			elements[element_count]=&_elements[element_count];
			return elements[element_count++];
		}

		RenderList() {

			element_count = 0;
			for (int i=0;i<MAX_ELEMENTS;i++)
				elements[i]=&_elements[i]; // assign elements
		}
	};

	RenderList opaque_render_list;
	RenderList alpha_render_list;

	RID default_material;

	struct FX {

		bool bgcolor_active;
		Color bgcolor;

		bool skybox_active;
		RID skybox_cubemap;

		bool antialias_active;
		float antialias_tolerance;

		bool glow_active;
		int glow_passes;
		float glow_attenuation;
		float glow_bloom;

		bool ssao_active;
		float ssao_attenuation;
		float ssao_radius;
		float ssao_max_distance;
		float ssao_range_max;
		float ssao_range_min;
		bool ssao_only;

		bool fog_active;
		float fog_near;
		float fog_far;
		float fog_attenuation;
		Color fog_color_near;
		Color fog_color_far;
		bool fog_bg;

		bool toon_active;
		float toon_treshold;
		float toon_soft;

		bool edge_active;
		Color edge_color;
		float edge_size;

		FX();

	};
	mutable RID_Owner<FX> fx_owner;


	FX *scene_fx;
	CameraMatrix camera_projection;
	Transform camera_transform;
	Transform camera_transform_inverse;
	float camera_z_near;
	float camera_z_far;
	Size2 camera_vp_size;

	Plane camera_plane;

	void _add_geometry( const Geometry* p_geometry, const InstanceData *p_instance, const Geometry *p_geometry_cmp, const GeometryOwner *p_owner);
	void _render_list_forward(RenderList *p_render_list,bool p_reverse_cull=false,bool p_fragment_light=false);

	//void _setup_light(LightInstance* p_instance, int p_idx);
	void _setup_lights(const uint16_t * p_lights,int p_light_count);

	_FORCE_INLINE_ void _setup_shader_params(const Material *p_material);
	bool _setup_material(const Geometry *p_geometry,const Material *p_material,bool p_vertexlit,bool p_no_const_light);

	Error _setup_geometry(const Geometry *p_geometry, const Material* p_material,const Skeleton *p_skeleton, const float *p_morphs);
	void _render(const Geometry *p_geometry,const Material *p_material, const Skeleton* p_skeleton, const GeometryOwner *p_owner);


	/***********/
	/* SHADOWS */
	/***********/

	struct ShadowBuffer {

		int size;
		int fbo;
		int depth;
		LightInstance *owner;
		void init(int p_size);
		ShadowBuffer() { size=0; depth=0; owner=NULL; }
	};

	Vector<ShadowBuffer> near_shadow_buffers;
	Vector<ShadowBuffer> far_shadow_buffers;

	LightInstance *shadow;
	int shadow_pass;
	void _init_shadow_buffers();

	float shadow_near_far_split_size_ratio;


	/***********/
	/*  FBOs   */
	/***********/


	struct FrameBuffer {

		int fbo;
		int color;
		int depth;
		int width,height;
		bool buff16;
		bool active;

		struct Blur {

			int fbo;
			int color;
		} blur[2];

	} framebuffer;

	void _update_framebuffer();
	void _process_glow_and_bloom();

	/*********/
	/* FRAME */
	/*********/

	struct _Rinfo {

		int texture_mem;
		int vertex_count;
		int object_count;
		int mat_change_count;
		int shader_change_count;

	} _rinfo;

	int white_tex;
	RID canvas_tex;
	float canvas_opacity;
	_FORCE_INLINE_ Texture* _bind_canvas_texture(const RID& p_texture);
	VS::MaterialBlendMode canvas_blend_mode;

	int _setup_geometry_vinfo;

	bool pack_arrays;
	bool keep_copies;
	bool cull_front;
	_FORCE_INLINE_ void _set_cull(bool p_front,bool p_reverse_cull=false);

	Size2 window_size;
	VS::ViewportRect viewport;
	double last_time;
	double time_delta;
	uint64_t frame;
	uint64_t scene_pass;

	void _draw_primitive(int p_points, const Vector3 *p_vertices, const Vector3 *p_normals, const Color* p_colors, const Vector3 *p_uvs,const Plane *p_tangents=NULL,int p_instanced=1);
	_FORCE_INLINE_ void _draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color* p_colors, const Vector2 *p_uvs);
	void _draw_textured_quad(const Rect2& p_rect, const Rect2& p_src_region, const Size2& p_tex_size,bool p_h_flip=false, bool p_v_flip=false );
	void _draw_quad(const Rect2& p_rect);

public:

	/* TEXTURE API */

	virtual RID texture_create();
	virtual void texture_allocate(RID p_texture,int p_width, int p_height,Image::Format p_format,uint32_t p_flags=VS::TEXTURE_FLAGS_DEFAULT,int p_mipmap_count=-1);
	virtual void texture_set_data(RID p_texture,const Image& p_image,VS::CubeMapSide p_cube_side=VS::CUBEMAP_LEFT);
	virtual Image texture_get_data(RID p_texture,VS::CubeMapSide p_cube_side=VS::CUBEMAP_LEFT) const;
	virtual void texture_set_flags(RID p_texture,uint32_t p_flags);
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;
	virtual bool texture_has_alpha(RID p_texture) const;
	virtual void texture_set_size_override(RID p_texture,int p_width, int p_height);

	virtual void texture_set_reload_hook(RID p_texture,ObjectID p_owner,const StringName& p_function) const {};

	/* SHADER API */

	virtual RID shader_create(VS::ShaderMode p_mode=VS::SHADER_MATERIAL);

	virtual void shader_set_mode(RID p_shader,VS::ShaderMode p_mode);
	virtual VS::ShaderMode shader_get_mode(RID p_shader) const;

	virtual void shader_set_code(RID p_shader, const String& p_vertex, const String& p_fragment,int p_vertex_ofs=0,int p_fragment_ofs=0);
	virtual String shader_get_fragment_code(RID p_shader) const;
	virtual String shader_get_vertex_code(RID p_shader) const;

	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const;


	/* COMMON MATERIAL API */

	virtual RID material_create();

	virtual void material_set_shader(RID p_shader_material, RID p_shader);
	virtual RID material_get_shader(RID p_shader_material) const;

	virtual void material_set_param(RID p_material, const StringName& p_param, const Variant& p_value);
	virtual Variant material_get_param(RID p_material, const StringName& p_param) const;

	virtual void material_set_flag(RID p_material, VS::MaterialFlag p_flag,bool p_enabled);
	virtual bool material_get_flag(RID p_material,VS::MaterialFlag p_flag) const;

	virtual void material_set_hint(RID p_material, VS::MaterialHint p_hint,bool p_enabled);
	virtual bool material_get_hint(RID p_material,VS::MaterialHint p_hint) const;

	virtual void material_set_shade_model(RID p_material, VS::MaterialShadeModel p_model);
	virtual VS::MaterialShadeModel material_get_shade_model(RID p_material) const;

	virtual void material_set_blend_mode(RID p_material,VS::MaterialBlendMode p_mode);
	virtual VS::MaterialBlendMode material_get_blend_mode(RID p_material) const;

	virtual void material_set_line_width(RID p_material,float p_line_width);
	virtual float material_get_line_width(RID p_material) const;


	/* MESH API */

	virtual RID mesh_create();

	virtual void mesh_add_surface(RID p_mesh,VS::PrimitiveType p_primitive,const Array& p_arrays,const Array& p_blend_shapes=Array());
	virtual Array mesh_get_surface_arrays(RID p_mesh,int p_surface) const;
	virtual Array mesh_get_surface_morph_arrays(RID p_mesh,int p_surface) const;
	virtual void mesh_add_custom_surface(RID p_mesh,const Variant& p_dat);

	virtual void mesh_set_morph_target_count(RID p_mesh,int p_amount);
	virtual int mesh_get_morph_target_count(RID p_mesh) const;

	virtual void mesh_set_morph_target_mode(RID p_mesh,VS::MorphTargetMode p_mode);
	virtual VS::MorphTargetMode mesh_get_morph_target_mode(RID p_mesh) const;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material,bool p_owned=false);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const;
	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const;

	virtual void mesh_remove_surface(RID p_mesh,int p_index);
	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual AABB mesh_get_aabb(RID p_mesh) const;

	/* MULTIMESH API */

	virtual RID multimesh_create();

	virtual void multimesh_set_instance_count(RID p_multimesh,int p_count);
	virtual int multimesh_get_instance_count(RID p_multimesh) const;

	virtual void multimesh_set_mesh(RID p_multimesh,RID p_mesh);
	virtual void multimesh_set_aabb(RID p_multimesh,const AABB& p_aabb);
	virtual void multimesh_instance_set_transform(RID p_multimesh,int p_index,const Transform& p_transform);
	virtual void multimesh_instance_set_color(RID p_multimesh,int p_index,const Color& p_color);

	virtual RID multimesh_get_mesh(RID p_multimesh) const;
	virtual AABB multimesh_get_aabb(RID p_multimesh) const;;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh,int p_index) const;
	virtual Color multimesh_instance_get_color(RID p_multimesh,int p_index) const;

	virtual void multimesh_set_visible_instances(RID p_multimesh,int p_visible);
	virtual int multimesh_get_visible_instances(RID p_multimesh) const;

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

	virtual void particles_set_variable(RID p_particles, VS::ParticleVariable p_variable,float p_value);
	virtual float particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const;

	virtual void particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable,float p_randomness);
	virtual float particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const;

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

	virtual AABB particles_get_aabb(RID p_particles) const;

	virtual void particles_set_height_from_velocity(RID p_particles, bool p_enable);
	virtual bool particles_has_height_from_velocity(RID p_particles) const;

	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable);
	virtual bool particles_is_using_local_coordinates(RID p_particles) const;

	/* SKELETON API */

	virtual RID skeleton_create();
	virtual void skeleton_resize(RID p_skeleton,int p_bones);
	virtual int skeleton_get_bone_count(RID p_skeleton) const;
	virtual void skeleton_bone_set_transform(RID p_skeleton,int p_bone, const Transform& p_transform);
	virtual Transform skeleton_bone_get_transform(RID p_skeleton,int p_bone);


	/* LIGHT API */

	virtual RID light_create(VS::LightType p_type);
	virtual VS::LightType light_get_type(RID p_light) const;

	virtual void light_set_color(RID p_light,VS::LightColor p_type, const Color& p_color);
	virtual Color light_get_color(RID p_light,VS::LightColor p_type) const;

	virtual void light_set_shadow(RID p_light,bool p_enabled);
	virtual bool light_has_shadow(RID p_light) const;

	virtual void light_set_volumetric(RID p_light,bool p_enabled);
	virtual bool light_is_volumetric(RID p_light) const;

	virtual void light_set_projector(RID p_light,RID p_texture);
	virtual RID light_get_projector(RID p_light) const;

	virtual void light_set_var(RID p_light, VS::LightParam p_var, float p_value);
	virtual float light_get_var(RID p_light, VS::LightParam p_var) const;

	virtual void light_set_operator(RID p_light,VS::LightOp p_op);
	virtual VS::LightOp light_get_operator(RID p_light) const;

	virtual void light_omni_set_shadow_mode(RID p_light,VS::LightOmniShadowMode p_mode);
	virtual VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) const;

	virtual void light_directional_set_shadow_mode(RID p_light,VS::LightDirectionalShadowMode p_mode);
	virtual VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) const;
	virtual void light_directional_set_shadow_max_distance(RID p_light,float p_distance);
	virtual float light_directional_get_shadow_max_distance(RID p_light) const;
	virtual void light_directional_set_pssm_split_weight(RID p_light,float p_weight);
	virtual float light_directional_get_pssm_split_weight(RID p_light) const;
	virtual void light_directional_set_shadow_param(RID p_light,VS::LightDirectionalShadowParam p_param, float p_value);
	virtual float light_directional_get_shadow_param(RID p_light,VS::LightDirectionalShadowParam p_param) const;

	virtual AABB light_get_aabb(RID p_poly) const;

	virtual RID light_instance_create(RID p_light);
	virtual void light_instance_set_transform(RID p_light_instance,const Transform& p_transform);

	virtual void light_instance_set_active_hint(RID p_light_instance);
	virtual bool light_instance_has_shadow(RID p_light_instance) const;
	virtual bool light_instance_assign_shadow(RID p_light_instance);
	virtual ShadowType light_instance_get_shadow_type(RID p_light_instance) const;
	virtual int light_instance_get_shadow_passes(RID p_light_instance) const;
	virtual void light_instance_set_custom_transform(RID p_light_instance, int p_index, const CameraMatrix& p_camera, const Transform& p_transform, float p_split_near=0,float p_split_far=0);
	virtual int light_instance_get_shadow_size(RID p_light_instance, int p_index=0) const { return 1; }

	virtual ShadowType light_instance_get_shadow_type(RID p_light_instance,bool p_far=false) const { return	SHADOW_NONE; };
	virtual void light_instance_set_shadow_transform(RID p_light_instance, int p_index, const CameraMatrix& p_camera, const Transform& p_transform, float p_split_near=0,float p_split_far=0) {};

	/* SHADOWS */

	virtual void shadow_clear_near() { };
	virtual bool shadow_allocate_near(RID p_light) { return false; }; //true on successful alloc
	virtual bool shadow_allocate_far(RID p_light) { return false; }; //true on successful alloc


	/* PARTICLES INSTANCE */

	virtual RID particles_instance_create(RID p_particles);
	virtual void particles_instance_set_transform(RID p_particles_instance,const Transform& p_transform);

	/* RENDER API */
	/* all calls (inside begin/end shadow) are always warranted to be in the following order: */

	virtual void begin_frame();

	virtual void set_viewport(const VS::ViewportRect& p_viewport);
	virtual void clear_viewport(const Color& p_color);

	virtual void begin_scene(bool p_copy_bg,RID p_fx=RID(),VS::ScenarioDebugMode p_debug=VS::SCENARIO_DEBUG_DISABLED);
	virtual void begin_shadow_map( RID p_light_instance, int p_shadow_pass );

	virtual void set_camera(const Transform& p_world,const CameraMatrix& p_projection);

	virtual void add_light( RID p_light_instance ); ///< all "add_light" calls happen before add_geometry calls

	typedef Map<StringName,Variant> ParamOverrideMap;

	virtual void add_mesh( const RID& p_mesh, const InstanceData *p_data);
	virtual void add_multimesh( const RID& p_multimesh, const InstanceData *p_data);
	virtual void add_particles( const RID& p_particle_instance, const InstanceData *p_data);

	virtual void end_scene();
	virtual void end_shadow_map();

	virtual void end_frame();

	/* CANVAS API */

	virtual void canvas_begin();
	virtual void canvas_set_opacity(float p_opacity);
	virtual void canvas_set_blend_mode(VS::MaterialBlendMode p_mode);
	virtual void canvas_begin_rect(const Matrix32& p_transform);
	virtual void canvas_set_clip(bool p_clip, const Rect2& p_rect);
	virtual void canvas_end_rect();
	virtual void canvas_draw_line(const Point2& p_from, const Point2& p_to,const Color& p_color,float p_width);
	virtual void canvas_draw_rect(const Rect2& p_rect, int p_flags, const Rect2& p_source,RID p_texture,const Color& p_modulate);
	virtual void canvas_draw_style_box(const Rect2& p_rect, RID p_texture,const float *p_margins, bool p_draw_center=true,const Color& p_modulate=Color(1,1,1));
	virtual void canvas_draw_primitive(const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture,float p_width);
	virtual void canvas_draw_polygon(int p_vertex_count, const int* p_indices, const Vector2* p_vertices, const Vector2* p_uvs, const Color* p_colors,const RID& p_texture,bool p_singlecolor);
	virtual void canvas_set_transform(const Matrix32& p_transform);

	/* ENVIRONMENT */


	virtual RID environment_create();

	virtual void environment_set_background(RID p_env,VS::EnvironmentBG p_bg);
	virtual VS::EnvironmentBG environment_get_background(RID p_env) const;

	virtual void environment_set_background_param(RID p_env,VS::EnvironmentBGParam p_param, const Variant& p_value);
	virtual Variant environment_get_background_param(RID p_env,VS::EnvironmentBGParam p_param) const;

	virtual void environment_set_enable_fx(RID p_env,VS::EnvironmentFx p_effect,bool p_enabled);
	virtual bool environment_is_fx_enabled(RID p_env,VS::EnvironmentFx p_effect) const;

	virtual void environment_fx_set_param(RID p_env,VS::EnvironmentFxParam p_param,const Variant& p_value);
	virtual Variant environment_fx_get_param(RID p_env,VS::EnvironmentFxParam p_param) const;

	/* FX */

	virtual RID fx_create();
	virtual void fx_get_effects(RID p_fx,List<String> *p_effects) const;
	virtual void fx_set_active(RID p_fx,const String& p_effect, bool p_active);
	virtual bool fx_is_active(RID p_fx,const String& p_effect) const;
	virtual void fx_get_effect_params(RID p_fx,const String& p_effect,List<PropertyInfo> *p_params) const;
	virtual Variant fx_get_effect_param(RID p_fx,const String& p_effect,const String& p_param) const;
	virtual void fx_set_effect_param(RID p_fx,const String& p_effect, const String& p_param, const Variant& p_pvalue);

	/*MISC*/

	virtual bool is_texture(const RID& p_rid) const;
	virtual bool is_material(const RID& p_rid) const;
	virtual bool is_mesh(const RID& p_rid) const;
	virtual bool is_multimesh(const RID& p_rid) const;
	virtual bool is_particles(const RID &p_beam) const;

	virtual bool is_light(const RID& p_rid) const;
	virtual bool is_light_instance(const RID& p_rid) const;
	virtual bool is_particles_instance(const RID& p_rid) const;
	virtual bool is_skeleton(const RID& p_rid) const;
	virtual bool is_environment(const RID& p_rid) const;
	virtual bool is_fx(const RID& p_rid) const;
	virtual bool is_shader(const RID& p_rid) const;

	virtual void free(const RID& p_rid);

	virtual void init();
	virtual void finish();

	virtual int get_render_info(VS::RenderInfo p_info);

	virtual void flush_frame(); //not necesary in most cases
	RasterizerFlash(bool p_compress_arrays=false,bool p_keep_ram_copy=true,bool p_default_fragment_lighting=true);
	virtual ~RasterizerFlash();
};

#endif
