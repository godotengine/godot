/*************************************************************************/
/*  lightmapper.h                                                        */
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

#ifndef LIGHTMAPPER_H
#define LIGHTMAPPER_H

#include "scene/resources/mesh.h"
#include "servers/rendering/rendering_device.h"

class LightmapDenoiser : public Reference {
	GDCLASS(LightmapDenoiser, Reference)
protected:
	static LightmapDenoiser *(*create_function)();

public:
	virtual Ref<Image> denoise_image(const Ref<Image> &p_image) = 0;
	static Ref<LightmapDenoiser> create();
};

class Lightmapper : public Reference {
	GDCLASS(Lightmapper, Reference)
public:
	enum GenerateProbes {
		GENERATE_PROBES_DISABLED,
		GENERATE_PROBES_SUBDIV_4,
		GENERATE_PROBES_SUBDIV_8,
		GENERATE_PROBES_SUBDIV_16,
		GENERATE_PROBES_SUBDIV_32,

	};

	enum LightType {
		LIGHT_TYPE_DIRECTIONAL,
		LIGHT_TYPE_OMNI,
		LIGHT_TYPE_SPOT
	};

	enum BakeError {
		BAKE_ERROR_LIGHTMAP_TOO_SMALL,
		BAKE_ERROR_LIGHTMAP_CANT_PRE_BAKE_MESHES,
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
	typedef bool (*BakeStepFunc)(float, const String &, void *, bool); //step index, step total, step description, userdata

	struct MeshData {
		//triangle data
		Vector<Vector3> points;
		Vector<Vector2> uv2;
		Vector<Vector3> normal;
		Ref<Image> albedo_on_uv2;
		Ref<Image> emission_on_uv2;
		Variant userdata;
	};

	virtual void add_mesh(const MeshData &p_mesh) = 0;
	virtual void add_directional_light(bool p_static, const Vector3 &p_direction, const Color &p_color, float p_energy, float p_angular_distance) = 0;
	virtual void add_omni_light(bool p_static, const Vector3 &p_position, const Color &p_color, float p_energy, float p_range, float p_attenuation, float p_size) = 0;
	virtual void add_spot_light(bool p_static, const Vector3 &p_position, const Vector3 p_direction, const Color &p_color, float p_energy, float p_range, float p_attenuation, float p_spot_angle, float p_spot_attenuation, float p_size) = 0;
	virtual void add_probe(const Vector3 &p_position) = 0;
	virtual BakeError bake(BakeQuality p_quality, bool p_use_denoiser, int p_bounces, float p_bias, int p_max_texture_size, bool p_bake_sh, GenerateProbes p_generate_probes, const Ref<Image> &p_environment_panorama, const Basis &p_environment_transform, BakeStepFunc p_step_function = nullptr, void *p_step_userdata = nullptr) = 0;

	virtual int get_bake_texture_count() const = 0;
	virtual Ref<Image> get_bake_texture(int p_index) const = 0;
	virtual int get_bake_mesh_count() const = 0;
	virtual Variant get_bake_mesh_userdata(int p_index) const = 0;
	virtual Rect2 get_bake_mesh_uv_scale(int p_index) const = 0;
	virtual int get_bake_mesh_texture_slice(int p_index) const = 0;
	virtual int get_bake_probe_count() const = 0;
	virtual Vector3 get_bake_probe_point(int p_probe) const = 0;
	virtual Vector<Color> get_bake_probe_sh(int p_probe) const = 0;

	static Ref<Lightmapper> create();

	Lightmapper();
};

#endif // LIGHTMAPPER_H
