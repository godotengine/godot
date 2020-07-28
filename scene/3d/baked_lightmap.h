/*************************************************************************/
/*  baked_lightmap.h                                                     */
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

#ifndef BAKED_INDIRECT_LIGHT_H
#define BAKED_INDIRECT_LIGHT_H

#include "modules/raytrace/raytrace.h"
#include "multimesh_instance.h"
#include "scene/3d/light.h"
#include "scene/3d/visual_instance.h"

class BakedLightmapData : public Resource {
	GDCLASS(BakedLightmapData, Resource);

	RID baked_light;
	AABB bounds;
	float energy;
	int cell_subdiv;
	Transform cell_space_xform;

	struct User {

		NodePath path;
		struct {
			Ref<Texture> single;
			Ref<TextureLayered> layered;
		} lightmap;
		int lightmap_slice;
		Rect2 lightmap_uv_rect;
		int instance_index;
	};

	Vector<User> users;

	void _set_user_data(const Array &p_data);
	Array _get_user_data() const;

protected:
	static void _bind_methods();

public:
	void set_bounds(const AABB &p_bounds);
	AABB get_bounds() const;

	void set_octree(const PoolVector<uint8_t> &p_octree);
	PoolVector<uint8_t> get_octree() const;

	void set_cell_space_transform(const Transform &p_xform);
	Transform get_cell_space_transform() const;

	void set_cell_subdiv(int p_cell_subdiv);
	int get_cell_subdiv() const;

	void set_energy(float p_energy);
	float get_energy() const;

	void add_user(const NodePath &p_path, const Ref<Resource> &p_lightmap, int p_lightmap_slice, const Rect2 &p_lightmap_uv_rect, int p_instance);
	int get_user_count() const;
	NodePath get_user_path(int p_user) const;
	Ref<Resource> get_user_lightmap(int p_user) const;
	int get_user_lightmap_slice(int p_user) const;
	Rect2 get_user_lightmap_uv_rect(int p_user) const;
	int get_user_instance(int p_user) const;
	void clear_users();

	virtual RID get_rid() const;
	BakedLightmapData();
	~BakedLightmapData();
};

class BakedLightmap : public VisualInstance {
	GDCLASS(BakedLightmap, VisualInstance);

public:
	enum BakeQuality {
		BAKE_QUALITY_LOW,
		BAKE_QUALITY_MEDIUM,
		BAKE_QUALITY_HIGH,
		BAKE_QUALITY_ULTRA
	};

	enum BakeError {
		BAKE_ERROR_OK,
		BAKE_ERROR_NO_SAVE_PATH,
		BAKE_ERROR_NO_MESHES,
		BAKE_ERROR_CANT_CREATE_IMAGE,
		BAKE_ERROR_LIGHTMAP_SIZE,
		BAKE_ERROR_USER_ABORTED

	};

	enum EnvironmentMode {
		ENVIRONMENT_MODE_DISABLED,
		ENVIRONMENT_MODE_SCENE,
		ENVIRONMENT_MODE_CUSTOM_SKY,
		ENVIRONMENT_MODE_CUSTOM_COLOR,
	};

	typedef void (*BakeBeginFunc)(int);
	typedef bool (*BakeStepFunc)(int, const String &);
	typedef void (*BakeEndFunc)();

private:
	Vector3 extents;
	float default_texels_per_unit;
	float bias;
	BakeQuality bake_quality;
	bool generate_atlas;
	int max_atlas_size;
	bool capture_enabled;
	int bounces;
	bool use_denoiser;

	EnvironmentMode environment_mode;
	Ref<Sky> environment_custom_sky;
	Color environment_custom_color;
	float environment_custom_energy;

	BakeQuality capture_quality;
	float capture_propagation;
	float capture_cell_size;

	String image_path;

	Ref<BakedLightmapData> light_data;

	void _assign_lightmaps();
	void _clear_lightmaps();
	Vector<Color> _get_irradiance_map(Ref<Environment> p_env, Vector2i &r_size);

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;
	void _notification(int p_what);

public:
	static BakeBeginFunc bake_begin_function;
	static BakeStepFunc bake_step_function;
	static BakeEndFunc bake_end_function;
	static BakeStepFunc bake_substep_function;
	static BakeEndFunc bake_end_substep_function;

	void set_light_data(const Ref<BakedLightmapData> &p_data);
	Ref<BakedLightmapData> get_light_data() const;

	void set_capture_cell_size(float p_cell_size);
	float get_capture_cell_size() const;

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const;

	void set_default_texels_per_unit(const float &p_extents);
	float get_default_texels_per_unit() const;

	void set_capture_propagation(float p_propagation);
	float get_capture_propagation() const;

	void set_capture_quality(BakeQuality p_quality);
	BakeQuality get_capture_quality() const;

	void set_bake_quality(BakeQuality p_quality);
	BakeQuality get_bake_quality() const;

	void set_generate_atlas(bool p_enabled);
	bool is_generate_atlas_enabled() const;

	void set_max_atlas_size(int p_size);
	int get_max_atlas_size() const;

	void set_capture_enabled(bool p_enable);
	bool get_capture_enabled() const;

	void set_image_path(const String &p_path);
	String get_image_path() const;

	void set_environment_mode(EnvironmentMode p_mode);
	EnvironmentMode get_environment_mode() const;

	void set_environment_custom_sky(const Ref<Sky> &p_sky);
	Ref<Sky> get_environment_custom_sky() const;

	void set_environment_custom_color(const Color &p_color);
	Color get_environment_custom_color() const;

	void set_environment_custom_energy(float p_energy);
	float get_environment_custom_energy() const;

	void set_use_denoiser(bool p_enable);
	bool is_using_denoiser() const;

	void set_bounces(int p_bounces);
	int get_bounces() const;

	void set_bias(float p_bias);
	float get_bias() const;

	AABB get_aabb() const;
	PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	BakeError bake(Node *p_from_node, bool p_create_visual_debug = false);

#ifdef TOOLS_ENABLED
	virtual String get_configuration_warning() const;
#endif

	BakedLightmap();
};

#ifdef TOOLS_ENABLED
class RaytraceLightBaker {
	struct LightMapElement {

		Vector3 emission;
		Vector3 direct_light;
		Vector3 output;

		Vector3 albedo;
		float alpha;

		Vector3 pos;
		Vector3 normal;
	};

	struct PlotMesh {
		Ref<Material> override_material;
		Vector<Ref<Material> > instance_materials;
		Ref<Mesh> mesh;
		Vector2 size_hint;
		Transform local_xform;
		Node *node;
		int instance_idx;
		bool cast_shadows;
		bool save_lightmap;
	};

	struct PlotLight {
		Light *light;
		Transform global_xform;
	};

	List<PlotMesh> mesh_list;
	List<PlotLight> light_list;
	Set<String> used_mesh_names;
	Set<int> no_shadow_meshes;

	Vector<Vector<LightMapElement> > scene_lightmaps;
	Vector<Vector<int> > scene_lightmap_indices;
	Vector<Vector2i> scene_lightmap_sizes;

	void _find_meshes_and_lights(Node *p_at_node);

	void _init_sky(Ref<World> p_world);
	Vector2 _compute_lightmap_size(const PlotMesh &p_plot_mesh);
	Vector<Color> _get_bake_texture(Ref<Image> p_image, const Vector2 &p_bake_size, const Color &p_color_mul, const Color &p_color_add);
	Vector3 _fix_sample_position(const Vector3 &p_position, const Vector3 &p_normal, const Vector3 &p_tangent, const Vector3 &p_bitangent, const Vector2 &p_texel_size);
	void _plot_triangle(Vector2 *p_vertices, Vector3 *p_positions, Vector3 *p_normals, Vector2 *p_uvs, const Vector<Color> &p_albedo_texture, const Vector<Color> &p_emission_texture, int p_width, int p_height, Vector<LightMapElement> &r_texels, int *r_lightmap_indices);
	void _make_lightmap(const PlotMesh &p_plot_mesh, int p_idx);

	bool _cast_shadow_ray(RaytraceEngine::Ray &r_ray);
	void _compute_direct_light(const PlotLight &p_plot_light, LightMapElement *r_lightmap, int p_size);

	Error _compute_indirect_light(unsigned int mesh_id);
	void _compute_ray_trace(uint32_t p_idx, LightMapElement *r_texels);

	void _fix_seams(const PlotMesh &p_plot_mesh, Vector3 *r_lightmap, const Vector2i &p_size);
	void _fix_seam(const Vector2 &p_uv0, const Vector2 &p_uv1, const Vector2 &p_uv3, const Vector2 &p_uv4, Vector3 *r_lightmap, const Vector2i &p_size);

	static bool _bake_time(float p_secs, float p_progress);

public:
	float default_texels_per_unit;
	BakedLightmap::BakeQuality bake_quality;
	bool capture_enabled;
	int bounces;
	bool use_denoiser;
	Vector<Color> sky_data;
	Basis sky_orientation;
	Vector2i sky_size;
	float bias;

	BakedLightmap::BakeQuality capture_quality;
	float capture_propagation;

	AABB global_bounds;
	AABB local_bounds;
	AABB bake_bounds;
	int capture_subdiv;

	BakedLightmap::BakeError bake(Node *p_base_node, Node *p_from_node, bool p_generate_atlas, int p_max_atlas_size, String p_save_path, Ref<BakedLightmapData> r_lightmap_data);

	RaytraceLightBaker();
};
#endif

VARIANT_ENUM_CAST(BakedLightmap::BakeQuality);
VARIANT_ENUM_CAST(BakedLightmap::BakeError);
VARIANT_ENUM_CAST(BakedLightmap::EnvironmentMode);

#endif // BAKED_INDIRECT_LIGHT_H
