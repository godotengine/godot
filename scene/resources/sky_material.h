/*************************************************************************/
/*  sky_material.h                                                       */
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

#include "core/templates/rid.h"
#include "scene/resources/material.h"

#ifndef SKY_MATERIAL_H
#define SKY_MATERIAL_H

class ProceduralSkyMaterial : public Material {
	GDCLASS(ProceduralSkyMaterial, Material);

private:
	Color sky_top_color;
	Color sky_horizon_color;
	float sky_curve;
	float sky_energy;

	Color ground_bottom_color;
	Color ground_horizon_color;
	float ground_curve;
	float ground_energy;

	float sun_angle_max;
	float sun_curve;

	static Mutex shader_mutex;
	static RID shader;
	static void _update_shader();
	mutable bool shader_set = false;

protected:
	static void _bind_methods();

public:
	void set_sky_top_color(const Color &p_sky_top);
	Color get_sky_top_color() const;

	void set_sky_horizon_color(const Color &p_sky_horizon);
	Color get_sky_horizon_color() const;

	void set_sky_curve(float p_curve);
	float get_sky_curve() const;

	void set_sky_energy(float p_energy);
	float get_sky_energy() const;

	void set_ground_bottom_color(const Color &p_ground_bottom);
	Color get_ground_bottom_color() const;

	void set_ground_horizon_color(const Color &p_ground_horizon);
	Color get_ground_horizon_color() const;

	void set_ground_curve(float p_curve);
	float get_ground_curve() const;

	void set_ground_energy(float p_energy);
	float get_ground_energy() const;

	void set_sun_angle_max(float p_angle);
	float get_sun_angle_max() const;

	void set_sun_curve(float p_curve);
	float get_sun_curve() const;

	virtual Shader::Mode get_shader_mode() const override;
	virtual RID get_shader_rid() const override;
	virtual RID get_rid() const override;

	static void cleanup_shader();

	ProceduralSkyMaterial();
	~ProceduralSkyMaterial();
};

//////////////////////////////////////////////////////
/* PanoramaSkyMaterial */

class PanoramaSkyMaterial : public Material {
	GDCLASS(PanoramaSkyMaterial, Material);

private:
	Ref<Texture2D> panorama;

	static Mutex shader_mutex;
	static RID shader_cache[2];
	static void _update_shader();
	mutable bool shader_set = false;

	bool filter = true;

protected:
	static void _bind_methods();

public:
	void set_panorama(const Ref<Texture2D> &p_panorama);
	Ref<Texture2D> get_panorama() const;

	void set_filtering_enabled(bool p_enabled);
	bool is_filtering_enabled() const;

	virtual Shader::Mode get_shader_mode() const override;
	virtual RID get_shader_rid() const override;
	virtual RID get_rid() const override;

	static void cleanup_shader();

	PanoramaSkyMaterial();
	~PanoramaSkyMaterial();
};

//////////////////////////////////////////////////////
/* PanoramaSkyMaterial */

class PhysicalSkyMaterial : public Material {
	GDCLASS(PhysicalSkyMaterial, Material);

private:
	static Mutex shader_mutex;
	static RID shader;

	float rayleigh;
	Color rayleigh_color;
	float mie;
	float mie_eccentricity;
	Color mie_color;
	float turbidity;
	float sun_disk_scale;
	Color ground_color;
	float exposure;
	float dither_strength;
	Ref<Texture2D> night_sky;
	static void _update_shader();
	mutable bool shader_set = false;

protected:
	static void _bind_methods();

public:
	void set_rayleigh_coefficient(float p_rayleigh);
	float get_rayleigh_coefficient() const;

	void set_rayleigh_color(Color p_rayleigh_color);
	Color get_rayleigh_color() const;

	void set_turbidity(float p_turbidity);
	float get_turbidity() const;

	void set_mie_coefficient(float p_mie);
	float get_mie_coefficient() const;

	void set_mie_eccentricity(float p_eccentricity);
	float get_mie_eccentricity() const;

	void set_mie_color(Color p_mie_color);
	Color get_mie_color() const;

	void set_sun_disk_scale(float p_sun_disk_scale);
	float get_sun_disk_scale() const;

	void set_ground_color(Color p_ground_color);
	Color get_ground_color() const;

	void set_exposure(float p_exposure);
	float get_exposure() const;

	void set_dither_strength(float p_dither_strength);
	float get_dither_strength() const;

	void set_night_sky(const Ref<Texture2D> &p_night_sky);
	Ref<Texture2D> get_night_sky() const;

	virtual Shader::Mode get_shader_mode() const override;
	virtual RID get_shader_rid() const override;

	static void cleanup_shader();
	virtual RID get_rid() const override;

	PhysicalSkyMaterial();
	~PhysicalSkyMaterial();
};

#endif /* !SKY_MATERIAL_H */
