#ifndef RASTERIZERSCENEGLES3_H
#define RASTERIZERSCENEGLES3_H

#include "rasterizer_storage_gles3.h"
#include "drivers/gles3/shaders/scene.glsl.h"

class RasterizerSceneGLES3 : public RasterizerScene {
public:

	uint64_t shadow_atlas_realloc_tolerance_msec;


	uint64_t render_pass;
	uint64_t scene_pass;
	uint32_t current_material_index;
	uint32_t current_geometry_index;

	RID default_material;
	RID default_shader;

	RasterizerStorageGLES3 *storage;

	struct State {


		bool texscreen_copied;
		int current_blend_mode;
		float current_line_width;
		int current_depth_draw;

		SceneShaderGLES3 scene_shader;


		struct SceneDataUBO {

			float projection_matrix[16];
			float camera_inverse_matrix[16];
			float camera_matrix[16];
			float time[4];
			float ambient_light_color[4];
			float bg_color[4];
			float ambient_energy;
			float bg_energy;

		} ubo_data;

		GLuint scene_ubo;

		struct EnvironmentRadianceUBO {

			float transform[16];
			float box_min[4]; //unused for now
			float box_max[4];
			float ambient_contribution;

		} env_radiance_data;

		GLuint env_radiance_ubo;

		GLuint brdf_texture;

		GLuint skybox_verts;
		GLuint skybox_array;



	} state;

	/* SHADOW ATLAS API */

	struct ShadowAtlas : public RID_Data {

		enum {
			SHADOW_INDEX_DIRTY_BIT=(1<<31),
			QUADRANT_SHIFT=27,
			SHADOW_INDEX_MASK=(1<<QUADRANT_SHIFT)-1,
			SHADOW_INVALID=0xFFFFFFFF
		};

		struct Quadrant {

			uint32_t subdivision;

			struct Shadow {
				RID owner;
				uint64_t version;
				uint64_t alloc_tick;

				Shadow() {
					version=0;
					alloc_tick=0;
				}
			};

			Vector<Shadow> shadows;

			Quadrant() {
				subdivision=0; //not in use
			}

		} quadrants[4];

		int size_order[4];
		uint32_t smallest_subdiv;

		int size;

		GLuint fbo;
		GLuint depth;

		Map<RID,uint32_t> shadow_owners;
	};

	RID_Owner<ShadowAtlas> shadow_atlas_owner;

	RID shadow_atlas_create();
	void shadow_atlas_set_size(RID p_atlas,int p_size);
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas,int p_quadrant,int p_subdivision);
	bool _shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow);
	uint32_t shadow_atlas_update_light(RID p_atlas,RID p_light_intance,float p_coverage,uint64_t p_light_version);

	/* ENVIRONMENT API */

	struct Environment : public RID_Data {

		VS::EnvironmentBG bg_mode;

		RID skybox_color;
		RID skybox_radiance;
		float skybox_scale;

		Color bg_color;
		float bg_energy;
		float skybox_ambient;

		Color ambient_color;
		float ambient_energy;
		float ambient_skybox_contribution;

		int canvas_max_layer;


		Environment() {
			bg_mode=VS::ENV_BG_CLEAR_COLOR;
			skybox_scale=1.0;
			bg_energy=1.0;
			skybox_ambient=0;
			ambient_energy=1.0;
			ambient_skybox_contribution=0.0;
			canvas_max_layer=0;
		}
	};

	RID_Owner<Environment> environment_owner;

	virtual RID environment_create();

	virtual void environment_set_background(RID p_env,VS::EnvironmentBG p_bg);
	virtual void environment_set_skybox(RID p_env,RID p_skybox,int p_radiance_size);
	virtual void environment_set_skybox_scale(RID p_env,float p_scale);
	virtual void environment_set_bg_color(RID p_env,const Color& p_color);
	virtual void environment_set_bg_energy(RID p_env,float p_energy);
	virtual void environment_set_canvas_max_layer(RID p_env,int p_max_layer);
	virtual void environment_set_ambient_light(RID p_env,const Color& p_color,float p_energy=1.0,float p_skybox_contribution=0.0);

	virtual void environment_set_glow(RID p_env,bool p_enable,int p_radius,float p_intensity,float p_strength,float p_bloom_treshold,VS::EnvironmentGlowBlendMode p_blend_mode);
	virtual void environment_set_fog(RID p_env,bool p_enable,float p_begin,float p_end,RID p_gradient_texture);

	virtual void environment_set_tonemap(RID p_env,bool p_enable,float p_exposure,float p_white,float p_min_luminance,float p_max_luminance,float p_auto_exp_speed,float p_auto_exp_scale,VS::EnvironmentToneMapper p_tone_mapper);
	virtual void environment_set_adjustment(RID p_env,bool p_enable,float p_brightness,float p_contrast,float p_saturation,RID p_ramp);


	/* LIGHT INSTANCE */

	struct LightInstance : public RID_Data {

		struct SplitInfo {

			CameraMatrix camera;
			Transform transform;
			float near;
			float far;
		};

		struct LightDataUBO {

			float light_pos_inv_radius[4];
			float light_direction_attenuation[4];
			float light_color_energy[4];
			float light_params[4]; //cone attenuation, specular, shadow darkening,
			float shadow_split_offsets[4];
			float shadow_matrix1[16];
			float shadow_matrix2[16];
			float shadow_matrix3[16];
			float shadow_matrix4[16];

		} light_ubo_data;


		SplitInfo split_info[4];

		RID light;
		RasterizerStorageGLES3::Light *light_ptr;

		CameraMatrix shadow_matrix[4];

		Transform transform;

		Vector3 light_vector;
		Vector3 spot_vector;
		float linear_att;

		GLuint light_ubo;

		uint64_t shadow_pass;
		uint64_t last_scene_pass;
		uint64_t last_pass;
		uint16_t light_index;

		Vector2 dp;

		CameraMatrix shadow_projection[4];

		Set<RID> shadow_atlases; //shadow atlases where this light is registered

		LightInstance() { }

	};

	mutable RID_Owner<LightInstance> light_instance_owner;

	virtual RID light_instance_create(RID p_light);
	virtual void light_instance_set_transform(RID p_light_instance,const Transform& p_transform);
	virtual void light_instance_mark_visible(RID p_light_instance);

	/* RENDER LIST */

	struct RenderList {

		enum {
			DEFAULT_MAX_ELEMENTS=65536,
			SORT_FLAG_SKELETON=1,
			SORT_FLAG_INSTANCING=2,
			MAX_DIRECTIONAL_LIGHTS=16,
			MAX_LIGHTS=4096,
			SORT_KEY_DEPTH_LAYER_SHIFT=58,
			SORT_KEY_LIGHT_TYPE_SHIFT=54, //type is most important
			SORT_KEY_LIGHT_INDEX_SHIFT=38, //type is most important
			SORT_KEY_LIGHT_INDEX_UNSHADED=uint64_t(0xF) << SORT_KEY_LIGHT_TYPE_SHIFT, //type is most important
			SORT_KEY_LIGHT_MASK=(uint64_t(0xFFFFF) << SORT_KEY_LIGHT_INDEX_SHIFT), //type is most important
			SORT_KEY_MATERIAL_INDEX_SHIFT=22,
			SORT_KEY_GEOMETRY_INDEX_SHIFT=6,
			SORT_KEY_GEOMETRY_TYPE_SHIFT=2,
			SORT_KEY_SKELETON_FLAG=2,
			SORT_KEY_MIRROR_FLAG=1

		};

		int max_elements;

		struct Element {

			RasterizerScene::InstanceBase *instance;
			RasterizerStorageGLES3::Geometry *geometry;
			RasterizerStorageGLES3::Material *material;
			RasterizerStorageGLES3::GeometryOwner *owner;
			uint64_t sort_key;
			bool *additive_ptr;
			bool additive;

		};


		Element *_elements;
		Element **elements;

		int element_count;
		int alpha_element_count;

		void clear() {

			element_count=0;
			alpha_element_count=0;
		}

		//should eventually be replaced by radix

		struct SortByKey {

			_FORCE_INLINE_ bool operator()(const Element* A,  const Element* B ) const {
				return A->sort_key < B->sort_key;
			}
		};

		void sort_by_key(bool p_alpha) {

			SortArray<Element*,SortByKey> sorter;
			if (p_alpha) {
				sorter.sort(&elements[max_elements-alpha_element_count],alpha_element_count);
			} else {
				sorter.sort(elements,element_count);
			}
		}

		struct SortByDepth {

			_FORCE_INLINE_ bool operator()(const Element* A,  const Element* B ) const {
				return A->instance->depth > B->instance->depth;
			}
		};

		void sort_by_depth(bool p_alpha) {

			SortArray<Element*,SortByDepth> sorter;
			if (p_alpha) {
				sorter.sort(&elements[max_elements-alpha_element_count],alpha_element_count);
			} else {
				sorter.sort(elements,element_count);
			}
		}


		_FORCE_INLINE_ Element* add_element() {

			if (element_count+alpha_element_count>=max_elements)
				return NULL;
			elements[element_count]=&_elements[element_count];
			return elements[element_count++];
		}

		_FORCE_INLINE_ Element* add_alpha_element() {

			if (element_count+alpha_element_count>=max_elements)
				return NULL;
			int idx = max_elements-alpha_element_count-1;
			elements[idx]=&_elements[idx];
			alpha_element_count++;
			return elements[idx];
		}

		void init() {

			element_count = 0;
			alpha_element_count =0;
			elements=memnew_arr(Element*,max_elements);
			_elements=memnew_arr(Element,max_elements);
			for (int i=0;i<max_elements;i++)
				elements[i]=&_elements[i]; // assign elements

		}


		RenderList() {

			max_elements=DEFAULT_MAX_ELEMENTS;
		}

		~RenderList() {
			memdelete_arr(elements);
			memdelete_arr(_elements);
		}
	};



	LightInstance *directional_light_instances[RenderList::MAX_DIRECTIONAL_LIGHTS];
	int directional_light_instance_count;

	LightInstance *light_instances[RenderList::MAX_LIGHTS];
	int light_instance_count;

	RenderList render_list;

	_FORCE_INLINE_ bool _setup_material(RasterizerStorageGLES3::Material* p_material,bool p_alpha_pass);
	_FORCE_INLINE_ void _setup_transform(InstanceBase *p_instance,const Transform& p_view_transform,const CameraMatrix& p_projection);
	_FORCE_INLINE_ void _setup_geometry(RenderList::Element *e);
	_FORCE_INLINE_ void _render_geometry(RenderList::Element *e);
	_FORCE_INLINE_ void _setup_light(LightInstance *p_light);

	void _render_list(RenderList::Element **p_elements, int p_element_count, const Transform& p_view_transform, const CameraMatrix& p_projection, RasterizerStorageGLES3::Texture *p_base_env, bool p_reverse_cull, bool p_alpha_pass);


	_FORCE_INLINE_ void _add_geometry(  RasterizerStorageGLES3::Geometry* p_geometry,  InstanceBase *p_instance, RasterizerStorageGLES3::GeometryOwner *p_owner,int p_material);

	void _draw_skybox(RID p_skybox, CameraMatrix& p_projection, const Transform& p_transform, bool p_vflip, float p_scale);

	void _setup_environment(Environment *env,CameraMatrix& p_cam_projection, const Transform& p_cam_transform);
	void _setup_lights(RID *p_light_cull_result, int p_light_cull_count, const Transform &p_camera_inverse_transform,const CameraMatrix& p_camera_projection);
	void _copy_screen();
	void _copy_to_front_buffer(Environment *env);

	virtual void render_scene(const Transform& p_cam_transform,CameraMatrix& p_cam_projection,bool p_cam_ortogonal,InstanceBase** p_cull_result,int p_cull_count,RID* p_light_cull_result,int p_light_cull_count,RID* p_directional_lights,int p_directional_light_count,RID p_environment);

	virtual bool free(RID p_rid);

	void _generate_brdf();

	virtual void set_scene_pass(uint64_t p_pass);

	void initialize();
	void finalize();
	RasterizerSceneGLES3();
};

#endif // RASTERIZERSCENEGLES3_H
