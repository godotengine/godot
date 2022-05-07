/*************************************************************************/
/*  lightmapper.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef LIGHTMAPPER_H
#define LIGHTMAPPER_H

#include "scene/resources/mesh.h"

#if !defined(__aligned)

#if defined(_WIN32) && defined(_MSC_VER)
#define __aligned(...) __declspec(align(__VA_ARGS__))
#else
#define __aligned(...) __attribute__((aligned(__VA_ARGS__)))
#endif

#endif

class LightmapDenoiser : public Reference {
	GDCLASS(LightmapDenoiser, Reference)
protected:
	static LightmapDenoiser *(*create_function)();

public:
	virtual Ref<Image> denoise_image(const Ref<Image> &p_image) = 0;
	static Ref<LightmapDenoiser> create();
};

class LightmapRaycaster : public Reference {
	GDCLASS(LightmapRaycaster, Reference)
protected:
	static LightmapRaycaster *(*create_function)();

public:
	// compatible with embree3 rays
	struct __aligned(16) Ray {
		const static unsigned int INVALID_GEOMETRY_ID = ((unsigned int)-1); // from rtcore_common.h

		/*! Default construction does nothing. */
		_FORCE_INLINE_ Ray() :
				geomID(INVALID_GEOMETRY_ID) {}

		/*! Constructs a ray from origin, direction, and ray segment. Near
		 *  has to be smaller than far. */
		_FORCE_INLINE_ Ray(const Vector3 &org,
				const Vector3 &dir,
				float tnear = 0.0f,
				float tfar = INFINITY) :
				org(org),
				tnear(tnear),
				dir(dir),
				time(0.0f),
				tfar(tfar),
				mask(-1),
				u(0.0),
				v(0.0),
				primID(INVALID_GEOMETRY_ID),
				geomID(INVALID_GEOMETRY_ID),
				instID(INVALID_GEOMETRY_ID) {}

		/*! Tests if we hit something. */
		_FORCE_INLINE_ explicit operator bool() const { return geomID != INVALID_GEOMETRY_ID; }

	public:
		Vector3 org; //!< Ray origin + tnear
		float tnear; //!< Start of ray segment
		Vector3 dir; //!< Ray direction + tfar
		float time; //!< Time of this ray for motion blur.
		float tfar; //!< End of ray segment
		unsigned int mask; //!< used to mask out objects during traversal
		unsigned int id; //!< ray ID
		unsigned int flags; //!< ray flags

		Vector3 normal; //!< Not normalized geometry normal
		float u; //!< Barycentric u coordinate of hit
		float v; //!< Barycentric v coordinate of hit
		unsigned int primID; //!< primitive ID
		unsigned int geomID; //!< geometry ID
		unsigned int instID; //!< instance ID
	};

	virtual bool intersect(Ray &p_ray) = 0;

	virtual void intersect(Vector<Ray> &r_rays) = 0;

	virtual void add_mesh(const Vector<Vector3> &p_vertices, const Vector<Vector3> &p_normals, const Vector<Vector2> &p_uv2s, unsigned int p_id) = 0;
	virtual void set_mesh_alpha_texture(Ref<Image> p_alpha_texture, unsigned int p_id) = 0;
	virtual void commit() = 0;

	virtual void set_mesh_filter(const Set<int> &p_mesh_ids) = 0;
	virtual void clear_mesh_filter() = 0;

	static Ref<LightmapRaycaster> create();
};

class Lightmapper : public Reference {
	GDCLASS(Lightmapper, Reference)
public:
	enum LightType {
		LIGHT_TYPE_DIRECTIONAL,
		LIGHT_TYPE_OMNI,
		LIGHT_TYPE_SPOT
	};

	enum BakeError {
		BAKE_ERROR_LIGHTMAP_TOO_SMALL,
		BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES,
		BAKE_ERROR_NO_MESHES,
		BAKE_ERROR_USER_ABORTED,
		BAKE_ERROR_NO_RAYCASTER,
		BAKE_OK
	};

	enum BakeQuality {
		BAKE_QUALITY_LOW,
		BAKE_QUALITY_MEDIUM,
		BAKE_QUALITY_HIGH,
		BAKE_QUALITY_ULTRA,
	};

	typedef Lightmapper *(*CreateFunc)();

	static CreateFunc create_custom;
	static CreateFunc create_gpu;
	static CreateFunc create_cpu;

protected:
public:
	typedef bool (*BakeStepFunc)(float, const String &, void *, bool); //progress, step description, userdata, force refresh
	typedef void (*BakeEndFunc)(uint32_t); // time_started

	struct MeshData {
		struct TextureDef {
			RID tex_rid;
			Color mul;
			Color add;
		};

		//triangle data
		Vector<Vector3> points;
		Vector<Vector2> uv;
		Vector<Vector2> uv2;
		Vector<Vector3> normal;
		Vector<TextureDef> albedo;
		Vector<TextureDef> emission;
		Vector<int> surface_facecounts;
		Variant userdata;
	};

	virtual void add_albedo_texture(Ref<Texture> p_texture) = 0;
	virtual void add_emission_texture(Ref<Texture> p_texture) = 0;
	virtual void add_mesh(const MeshData &p_mesh, Vector2i p_size) = 0;
	virtual void add_directional_light(bool p_bake_direct, const Vector3 &p_direction, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_size) = 0;
	virtual void add_omni_light(bool p_bake_direct, const Vector3 &p_position, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_range, float p_attenuation, float p_size) = 0;
	virtual void add_spot_light(bool p_bake_direct, const Vector3 &p_position, const Vector3 p_direction, const Color &p_color, float p_energy, float p_indirect_multiplier, float p_range, float p_attenuation, float p_spot_angle, float p_spot_attenuation, float p_size) = 0;
	virtual BakeError bake(BakeQuality p_quality, bool p_use_denoiser, int p_bounces, float p_bounce_indirect_energy, float p_bias, bool p_generate_atlas, int p_max_texture_size, const Ref<Image> &p_environment_panorama, const Basis &p_environment_transform, BakeStepFunc p_step_function = nullptr, void *p_step_userdata = nullptr, BakeStepFunc p_substep_function = nullptr) = 0;

	virtual int get_bake_texture_count() const = 0;
	virtual Ref<Image> get_bake_texture(int p_index) const = 0;
	virtual int get_bake_mesh_count() const = 0;
	virtual Variant get_bake_mesh_userdata(int p_index) const = 0;
	virtual Rect2 get_bake_mesh_uv_scale(int p_index) const = 0;
	virtual int get_bake_mesh_texture_slice(int p_index) const = 0;

	static Ref<Lightmapper> create();

	Lightmapper();
};

#endif // LIGHTMAPPER_H
