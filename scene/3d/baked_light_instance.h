/*************************************************************************/
/*  baked_light_instance.h                                               */
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
#ifndef BAKED_LIGHT_INSTANCE_H
#define BAKED_LIGHT_INSTANCE_H

#include "scene/3d/multimesh_instance.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/baked_light.h"

class BakedLightBaker;
class Light;

class BakedLight : public VisualInstance {
	GDCLASS(BakedLight, VisualInstance);

public:
	enum DebugMode {
		DEBUG_ALBEDO,
		DEBUG_LIGHT
	};

private:
	RID baked_light;
	int cell_subdiv;
	Rect3 bounds;
	int cells_per_axis;

	enum {
		CHILD_EMPTY = 0xFFFFFFFF,
	};

	/* BAKE DATA */

	struct BakeCell {

		uint32_t childs[8];
		float albedo[3]; //albedo in RGB24
		float light[3]; //accumulated light in 16:16 fixed point (needs to be integer for moving lights fast)
		float radiance[3]; //accumulated light in 16:16 fixed point (needs to be integer for moving lights fast)
		uint32_t used_sides;
		float alpha; //used for upsampling
		uint32_t light_pass; //used for baking light

		BakeCell() {
			for (int i = 0; i < 8; i++) {
				childs[i] = 0xFFFFFFFF;
			}

			for (int i = 0; i < 3; i++) {
				light[i] = 0;
				albedo[i] = 0;
				radiance[i] = 0;
			}
			alpha = 0;
			light_pass = 0;
			used_sides = 0;
		}
	};

	int bake_texture_size;
	int color_scan_cell_width;

	struct MaterialCache {
		//128x128 textures
		Vector<Color> albedo;
		Vector<Color> emission;
	};

	Vector<Color> _get_bake_texture(Image &p_image, const Color &p_color);

	Map<Ref<Material>, MaterialCache> material_cache;
	MaterialCache _get_material_cache(Ref<Material> p_material);

	int bake_cells_alloc;
	int bake_cells_used;
	int zero_alphas;
	Vector<int> bake_cells_level_used;
	PoolVector<BakeCell> bake_cells;
	PoolVector<BakeCell>::Write bake_cells_write;

	void _plot_face(int p_idx, int p_level, const Vector3 *p_vtx, const Vector2 *p_uv, const MaterialCache &p_material, const Rect3 &p_aabb);
	void _fixup_plot(int p_idx, int p_level, int p_x, int p_y, int p_z);
	void _bake_add_mesh(const Transform &p_xform, Ref<Mesh> &p_mesh);
	void _bake_add_to_aabb(const Transform &p_xform, Ref<Mesh> &p_mesh, bool &first);

	void _debug_mesh(int p_idx, int p_level, const Rect3 &p_aabb, DebugMode p_mode, Ref<MultiMesh> &p_multimesh, int &idx);
	void _debug_mesh_albedo();
	void _debug_mesh_light();

	_FORCE_INLINE_ int _find_cell(int x, int y, int z);
	int _plot_ray(const Vector3 &p_from, const Vector3 &p_to);

	uint32_t light_pass;

	void _bake_directional(int p_idx, int p_level, int p_x, int p_y, int p_z, const Vector3 &p_dir, const Color &p_color, int p_sign);
	void _upscale_light(int p_idx, int p_level);
	void _bake_light(Light *p_light);

	Color _cone_trace(const Vector3 &p_from, const Vector3 &p_dir, float p_half_angle);
	void _bake_radiance(int p_idx, int p_level, int p_x, int p_y, int p_z);

	friend class GeometryInstance;

	Set<GeometryInstance *> geometries;
	friend class Light;

	Set<Light *> lights;

protected:
	static void _bind_methods();

public:
	void set_cell_subdiv(int p_subdiv);
	int get_cell_subdiv() const;

	void bake();
	void bake_lights();
	void bake_radiance();

	void create_debug_mesh(DebugMode p_mode);

	virtual Rect3 get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	String get_configuration_warning() const;

	BakedLight();
	~BakedLight();
};

#if 0
class BakedLightSampler : public VisualInstance {
	GDCLASS(BakedLightSampler,VisualInstance);


public:

	enum Param {
		PARAM_RADIUS=VS::BAKED_LIGHT_SAMPLER_RADIUS,
		PARAM_STRENGTH=VS::BAKED_LIGHT_SAMPLER_STRENGTH,
		PARAM_ATTENUATION=VS::BAKED_LIGHT_SAMPLER_ATTENUATION,
		PARAM_DETAIL_RATIO=VS::BAKED_LIGHT_SAMPLER_DETAIL_RATIO,
		PARAM_MAX=VS::BAKED_LIGHT_SAMPLER_MAX
	};



protected:

	RID base;
	float params[PARAM_MAX];
	int resolution;
	static void _bind_methods();
public:

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_param(Param p_param,float p_value);
	float get_param(Param p_param) const;

	void set_resolution(int p_resolution);
	int get_resolution() const;

	BakedLightSampler();
	~BakedLightSampler();
};

VARIANT_ENUM_CAST( BakedLightSampler::Param );

#endif
#endif // BAKED_LIGHT_H
