#ifndef RASTERIZERSCENEGLES3_H
#define RASTERIZERSCENEGLES3_H

#include "rasterizer_storage_gles3.h"
#include "drivers/gles3/shaders/scene.glsl.h"

class RasterizerSceneGLES3 : public RasterizerScene {
public:

	uint64_t render_pass;
	uint32_t current_material_index;
	uint32_t current_geometry_index;

	RID default_material;
	RID default_shader;

	RasterizerStorageGLES3 *storage;



	struct State {

		bool current_depth_test;
		bool current_depth_mask;
		bool texscreen_copied;
		int current_blend_mode;

		SceneShaderGLES3 scene_shader;


		struct SceneDataUBO {

			float projection_matrix[16];
			float camera_inverse_matrix[16];
			float camera_matrix[16];
			float time[4];
			float ambient_light[4];

		} ubo_data;

		GLuint scene_ubo;



	} state;

	struct RenderList {

		enum {
			DEFAULT_MAX_ELEMENTS=65536,
			MAX_LIGHTS=4,
			SORT_FLAG_SKELETON=1,
			SORT_FLAG_INSTANCING=2,

			SORT_KEY_DEPTH_LAYER_SHIFT=58,
			SORT_KEY_LIGHT_TYPE_SHIFT=54, //type is most important
			SORT_KEY_LIGHT_INDEX_SHIFT=38, //type is most important
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
				sorter.sort(&elements[max_elements-alpha_element_count-1],alpha_element_count);
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



	RenderList render_list;

	_FORCE_INLINE_ bool _setup_material(RasterizerStorageGLES3::Material* p_material,bool p_alpha_pass);
	_FORCE_INLINE_ void _setup_geometry(RenderList::Element *e);
	_FORCE_INLINE_ void _render_geometry(RenderList::Element *e);


	void _render_list(RenderList::Element **p_elements, int p_element_count, const Transform& p_view_transform, const CameraMatrix& p_projection, bool p_reverse_cull, bool p_alpha_pass);

	virtual RID light_instance_create(RID p_light);
	virtual void light_instance_set_transform(RID p_light_instance,const Transform& p_transform);

	_FORCE_INLINE_ void _add_geometry(  RasterizerStorageGLES3::Geometry* p_geometry,  InstanceBase *p_instance, RasterizerStorageGLES3::GeometryOwner *p_owner,int p_material);

	virtual void render_scene(const Transform& p_cam_transform,CameraMatrix& p_cam_projection,bool p_cam_ortogonal,InstanceBase** p_cull_result,int p_cull_count,RID* p_light_cull_result,int p_light_cull_count,RID* p_directional_lights,int p_directional_light_count,RID p_environment);

	virtual bool free(RID p_rid);

	void initialize();
	void finalize();
	RasterizerSceneGLES3();
};

#endif // RASTERIZERSCENEGLES3_H
