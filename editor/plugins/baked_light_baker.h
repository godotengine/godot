/*************************************************************************/
/*  baked_light_baker.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef BAKED_LIGHT_BAKER_H
#define BAKED_LIGHT_BAKER_H

#include "os/thread.h"
#include "scene/3d/baked_light_instance.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"

#if 0

class BakedLightBaker {
public:

	enum {

		ATTENUATION_CURVE_LEN=256,
		OCTANT_POOL_CHUNK=1000000
	};

	/*
	struct OctantLight {
		double accum[8][3];
	};
	*/

	struct Octant {
		bool leaf;
		AABB aabb;
		uint16_t texture_x;
		uint16_t texture_y;
		int sampler_ofs;
		float normal_accum[8][3];
		double full_accum[3];
		int parent;
		union {
			struct {
				int next_leaf;
				float offset[3];
				int bake_neighbour;
				bool first_neighbour;
				double light_accum[8][3];
			};
			int children[8];
		};
	};

	struct OctantHash {

		int next;
		uint32_t hash;
		uint64_t value;

	};

	struct MeshTexture {

		Vector<uint8_t> tex;
		int tex_w,tex_h;

		_FORCE_INLINE_ void get_color(const Vector2& p_uv,Color& ret) {

			if (tex_w && tex_h) {

				int x = Math::fast_ftoi(Math::fposmod(p_uv.x,1.0)*tex_w);
				int y = Math::fast_ftoi(Math::fposmod(p_uv.y,1.0)*tex_w);
				x=CLAMP(x,0,tex_w-1);
				y=CLAMP(y,0,tex_h-1);
				const uint8_t*ptr = &tex[(y*tex_w+x)*4];
				ret.r*=ptr[0]/255.0;
				ret.g*=ptr[1]/255.0;
				ret.b*=ptr[2]/255.0;
				ret.a*=ptr[3]/255.0;
			}
		}

	};

	struct Param {

		Color color;
		MeshTexture*tex;
		_FORCE_INLINE_ Color get_color(const Vector2& p_uv) {

			Color ret=color;
			if (tex)
				tex->get_color(p_uv,ret);
			return ret;

		}

	};

	struct MeshMaterial {

		Param diffuse;
		Param specular;
		Param emission;
	};

	struct Triangle {

		AABB aabb;
		Vector3 vertices[3];
		Vector2 uvs[3];
		Vector2 bake_uvs[3];
		Vector3 normals[3];
		MeshMaterial *material;
		int baked_texture;

		_FORCE_INLINE_ Vector2 get_uv(const Vector3& p_pos) {

			Vector3 v0 = vertices[1] - vertices[0];
			Vector3 v1 = vertices[2] - vertices[0];
			Vector3 v2 = p_pos - vertices[0];

			float d00 = v0.dot( v0);
			float d01 = v0.dot( v1);
			float d11 = v1.dot( v1);
			float d20 = v2.dot( v0);
			float d21 = v2.dot( v1);
			float denom = (d00 * d11 - d01 * d01);
			if (denom==0)
				return uvs[0];
			float v = (d11 * d20 - d01 * d21) / denom;
			float w = (d00 * d21 - d01 * d20) / denom;
			float u = 1.0f - v - w;

			return uvs[0]*u + uvs[1]*v  + uvs[2]*w;
		}

		_FORCE_INLINE_ void get_uv_and_normal(const Vector3& p_pos,Vector2& r_uv,Vector3& r_normal) {

			Vector3 v0 = vertices[1] - vertices[0];
			Vector3 v1 = vertices[2] - vertices[0];
			Vector3 v2 = p_pos - vertices[0];

			float d00 = v0.dot( v0);
			float d01 = v0.dot( v1);
			float d11 = v1.dot( v1);
			float d20 = v2.dot( v0);
			float d21 = v2.dot( v1);
			float denom = (d00 * d11 - d01 * d01);
			if (denom==0) {
				r_normal=normals[0];
				r_uv=uvs[0];
				return;
			}
			float v = (d11 * d20 - d01 * d21) / denom;
			float w = (d00 * d21 - d01 * d20) / denom;
			float u = 1.0f - v - w;

			r_uv=uvs[0]*u + uvs[1]*v  + uvs[2]*w;
			r_normal=(normals[0]*u+normals[1]*v+normals[2]*w).normalized();
		}
	};


	struct BVH {

		AABB aabb;
		Vector3 center;
		Triangle *leaf;
		BVH*children[2];
	};


	struct BVHCmpX {

		bool operator()(const BVH* p_left, const BVH* p_right) const {

			return p_left->center.x < p_right->center.x;
		}
	};

	struct BVHCmpY {

		bool operator()(const BVH* p_left, const BVH* p_right) const {

			return p_left->center.y < p_right->center.y;
		}
	};
	struct BVHCmpZ {

		bool operator()(const BVH* p_left, const BVH* p_right) const {

			return p_left->center.z < p_right->center.z;
		}
	};

	struct BakeTexture {

		Vector<uint8_t> data;
		int width,height;
	};


	struct LightData {

		VS::LightType type;

		Vector3 pos;
		Vector3 up;
		Vector3 left;
		Vector3 dir;
		Color diffuse;
		Color specular;
		float energy;
		float length;
		int rays_thrown;
		bool bake_shadow;

		float radius;
		float attenuation;
		float spot_angle;
		float darkening;
		float spot_attenuation;
		float area;

		float constant;

		bool bake_direct;

		Vector<float> attenuation_table;

	};


	Vector<LightData> lights;

	List<MeshMaterial> materials;
	List<MeshTexture> textures;

	AABB octree_aabb;
	Vector<Octant> octant_pool;
	int octant_pool_size;
	BVH*bvh;
	Vector<Triangle> triangles;
	Vector<BakeTexture> baked_textures;
	Transform base_inv;
	int leaf_list;
	int octree_depth;
	int bvh_depth;
	int cell_count;
	uint32_t *ray_stack;
	BVH **bvh_stack;
	uint32_t *octant_stack;
	uint32_t *octantptr_stack;

	struct ThreadStack {
		uint32_t *octant_stack;
		uint32_t *octantptr_stack;
		uint32_t *ray_stack;
		BVH **bvh_stack;
	};

	Map<Vector3,Vector3> endpoint_normal;
	Map<Vector3,uint64_t> endpoint_normal_bits;

	float cell_size;
	float plot_size; //multiplied by cell size
	float octree_extra_margin;

	int max_bounces;
	int64_t total_rays;
	bool use_diffuse;
	bool use_specular;
	bool use_translucency;
	bool linear_color;


	int baked_octree_texture_w;
	int baked_octree_texture_h;
	int baked_light_texture_w;
	int baked_light_texture_h;
	int lattice_size;
	float edge_damp;
	float normal_damp;
	float tint;
	float ao_radius;
	float ao_strength;

	bool paused;
	bool baking;
	bool first_bake_to_map;

	Map<Ref<Material>,MeshMaterial*> mat_map;
	Map<Ref<Texture>,MeshTexture*> tex_map;



	MeshTexture* _get_mat_tex(const Ref<Texture>& p_tex);
	void _add_mesh(const Ref<Mesh>& p_mesh,const Ref<Material>& p_mat_override,const Transform& p_xform,int p_baked_texture=-1);
	void _parse_geometry(Node* p_node);
	BVH* _parse_bvh(BVH** p_children,int p_size,int p_depth,int& max_depth);
	void _make_bvh();
	void _make_octree();
	void _make_octree_texture();
	void _octree_insert(int p_octant, Triangle* p_triangle, int p_depth);
	_FORCE_INLINE_ void _plot_pixel_to_lightmap(int x, int y, int width, int height, uint8_t *image, const Vector3& p_pos,const Vector3& p_normal,double *p_norm_ptr,float mult,float gamma);


	void _free_bvh(BVH* p_bvh);

	void _fix_lights();

	Ref<BakedLight> baked_light;


	//void _plot_light(const Vector3& p_plot_pos,const AABB& p_plot_aabb,const Color& p_light,int p_octant=0);
	void _plot_light(ThreadStack& thread_stack,const Vector3& p_plot_pos,const AABB& p_plot_aabb,const Color& p_light,const Color& p_tint_light,bool p_only_full,const Plane& p_plane);
	//void _plot_light_point(const Vector3& p_plot_pos, Octant *p_octant, const AABB& p_aabb,const Color& p_light);

	float _throw_ray(ThreadStack& thread_stack,bool p_bake_direct,const Vector3& p_begin, const Vector3& p_end,float p_rest,const Color& p_light,float *p_att_curve,float p_att_pos,int p_att_curve_len,int p_bounces,bool p_first_bounce=false,bool p_only_dist=false);


	float total_light_area;

	Vector<Thread*> threads;

	bool bake_thread_exit;
	static void _bake_thread_func(void *arg);

	void _start_thread();
	void _stop_thread();
public:


	void throw_rays(ThreadStack &thread_stack, int p_amount);
	double get_normalization(int p_light_idx) const;
	double get_modifier(int p_light_idx) const;

	void bake(const Ref<BakedLight>& p_light,Node *p_base);
	bool is_baking();
	void set_pause(bool p_pause);
	bool is_paused();
	uint64_t get_rays_thrown() { return total_rays; }

	Error transfer_to_lightmaps();

	void update_octree_sampler(PoolVector<int> &p_sampler);
	void update_octree_images(PoolVector<uint8_t> &p_octree,PoolVector<uint8_t> &p_light);

	Ref<BakedLight> get_baked_light() { return baked_light; }

	void clear();

	BakedLightBaker();
	~BakedLightBaker();

};

#endif // BAKED_LIGHT_BAKER_H
#endif
