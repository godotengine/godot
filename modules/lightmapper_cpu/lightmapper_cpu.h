/**************************************************************************/
/*  lightmapper_cpu.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef LIGHTMAPPER_CPU_H
#define LIGHTMAPPER_CPU_H

#include "core/local_vector.h"
#include "scene/3d/lightmapper.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"

#include <atomic>

class LightmapperCPU : public Lightmapper {
	GDCLASS(LightmapperCPU, Lightmapper)

	struct MeshInstance {
		MeshData data;
		int slice = 0;
		Vector2i offset;
		Vector2i size;
		bool cast_shadows;
		bool generate_lightmap;
		String node_name;
	};

	struct Light {
		Vector3 position;
		uint32_t type = LIGHT_TYPE_DIRECTIONAL;
		Vector3 direction;
		float energy;
		float indirect_multiplier;
		Color color;
		float range;
		float attenuation;
		float spot_angle;
		float spot_attenuation;
		float size;
		bool bake_direct;
	};

	struct LightmapTexel {
		Vector3 albedo;
		float alpha;
		Vector3 emission;
		Vector3 pos;
		Vector3 normal;

		Vector3 direct_light;
		Vector3 output_light;

		float area_coverage;
	};

	struct BakeParams {
		float bias;
		int bounces;
		float bounce_indirect_energy;
		int samples;
		bool use_denoiser = true;
		bool use_physical_light_attenuation = false;
		Ref<Image> environment_panorama;
		Basis environment_transform;
	};

	struct UVSeam {
		Vector2 edge0[2];
		Vector2 edge1[2];
	};

	struct SeamEdge {
		Vector3 pos[2];
		Vector3 normal[2];
		Vector2 uv[2];

		_FORCE_INLINE_ bool operator<(const SeamEdge &p_edge) const {
			return pos[0].x < p_edge.pos[0].x;
		}
	};

	struct AtlasOffset {
		int slice;
		int x;
		int y;
	};

	struct ThreadData;

	typedef void (LightmapperCPU::*BakeThreadFunc)(uint32_t, void *);

	struct ThreadData {
		LightmapperCPU *instance;
		uint32_t count;
		BakeThreadFunc thread_func;
		void *userdata;
	};

	BakeParams parameters;

	LocalVector<Ref<Image>> bake_textures;
	Map<RID, Ref<Image>> albedo_textures;
	Map<RID, Ref<Image>> emission_textures;

	LocalVector<MeshInstance> mesh_instances;
	LocalVector<Light> lights;

	LocalVector<LocalVector<LightmapTexel>> scene_lightmaps;
	LocalVector<LocalVector<int>> scene_lightmap_indices;
	Set<int> no_shadow_meshes;

	std::atomic<uint32_t> thread_progress;
	std::atomic<bool> thread_cancelled;

	Ref<LightmapRaycaster> raycaster;

	Error _layout_atlas(int p_max_size, Vector2i *r_atlas_size, int *r_atlas_slices);

	static void _thread_func_callback(void *p_thread_data);
	void _thread_func_wrapper(uint32_t p_idx, ThreadData *p_thread_data);
	bool _parallel_run(int p_count, const String &p_description, BakeThreadFunc p_thread_func, void *p_userdata, BakeStepFunc p_substep_func = nullptr);

	void _generate_buffer(uint32_t p_idx, void *p_unused);
	Ref<Image> _init_bake_texture(const MeshData::TextureDef &p_texture_def, const Map<RID, Ref<Image>> &p_tex_cache, Image::Format p_default_format);
	Color _bilinear_sample(const Ref<Image> &p_img, const Vector2 &p_uv, bool p_clamp_x = false, bool p_clamp_y = false);
	Vector3 _fix_sample_position(const Vector3 &p_position, const Vector3 &p_texel_center, const Vector3 &p_normal, const Vector3 &p_tangent, const Vector3 &p_bitangent, const Vector2 &p_texel_size);
	void _plot_triangle(const Vector2 *p_vertices, const Vector3 *p_positions, const Vector3 *p_normals, const Vector2 *p_uvs, const Ref<Image> &p_albedo_texture, const Ref<Image> &p_emission_texture, Vector2i p_size, LocalVector<LightmapTexel> &r_texels, LocalVector<int> &r_lightmap_indices);

	float _get_omni_attenuation(float distance, float inv_range, float decay) const;

	void _compute_direct_light(uint32_t p_idx, void *r_lightmap);

	void _compute_indirect_light(uint32_t p_idx, void *r_lightmap);

	void _post_process(uint32_t p_idx, void *r_output);
	void _compute_seams(const MeshInstance &p_mesh, LocalVector<UVSeam> &r_seams);
	void _fix_seams(const LocalVector<UVSeam> &p_seams, Vector3 *r_lightmap, Vector2i p_size);
	void _fix_seam(const Vector2 &p_pos0, const Vector2 &p_pos1, const Vector2 &p_uv0, const Vector2 &p_uv1, const Vector3 *p_read_buffer, Vector3 *r_write_buffer, const Vector2i &p_size);
	void _dilate_lightmap(Vector3 *r_lightmap, const LocalVector<int> p_indices, Vector2i p_size, int margin);

	void _blit_lightmap(const Vector<Vector3> &p_src, const Vector2i &p_size, Ref<Image> &p_dst, int p_x, int p_y, bool p_with_padding);

public:
	virtual void add_albedo_texture(Ref<Texture> p_texture);
	virtual void add_emission_texture(Ref<Texture> p_texture);
	virtual void add_mesh(const MeshData &p_mesh, Vector2i p_size);
	virtual void add_directional_light(bool p_bake_direct, const Vector3 &p_direction, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_size);
	virtual void add_omni_light(bool p_bake_direct, const Vector3 &p_position, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_range, float p_attenuation, float p_size);
	virtual void add_spot_light(bool p_bake_direct, const Vector3 &p_position, const Vector3 p_direction, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_range, float p_attenuation, float p_spot_angle, float p_spot_attenuation, float p_size);
	virtual BakeError bake(BakeQuality p_quality, bool p_use_denoiser, int p_bounces, float p_bounce_energy, float p_bias, bool p_generate_atlas, int p_max_texture_size, const Ref<Image> &p_environment_panorama, const Basis &p_environment_transform, BakeStepFunc p_step_function = nullptr, void *p_bake_userdata = nullptr, BakeStepFunc p_substep_function = nullptr);

	int get_bake_texture_count() const;
	Ref<Image> get_bake_texture(int p_index) const;
	int get_bake_mesh_count() const;
	Variant get_bake_mesh_userdata(int p_index) const;
	Rect2 get_bake_mesh_uv_scale(int p_index) const;
	int get_bake_mesh_texture_slice(int p_index) const;

	LightmapperCPU();
};

#endif // LIGHTMAPPER_CPU_H
