/*************************************************************************/
/*  decal.h                                                              */
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

#ifndef DECAL_H
#define DECAL_H

#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/texture.h"
#include "servers/rendering_server.h"

class Decal : public VisualInstance3D {
	GDCLASS(Decal, VisualInstance3D);

public:
	enum DecalTexture {
		TEXTURE_ALBEDO,
		TEXTURE_NORMAL,
		TEXTURE_ORM,
		TEXTURE_EMISSION,
		TEXTURE_MAX
	};

private:
	RID decal;
	Vector3 extents;
	Ref<Texture2D> textures[TEXTURE_MAX];
	float emission_energy;
	float albedo_mix;
	Color modulate;
	uint32_t cull_mask;
	float normal_fade;
	float upper_fade;
	float lower_fade;
	bool distance_fade_enabled;
	float distance_fade_begin;
	float distance_fade_length;

protected:
	static void _bind_methods();

public:
	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const;

	void set_texture(DecalTexture p_type, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture(DecalTexture p_type) const;

	void set_emission_energy(float p_energy);
	float get_emission_energy() const;

	void set_albedo_mix(float p_mix);
	float get_albedo_mix() const;

	void set_modulate(Color p_modulate);
	Color get_modulate() const;

	void set_upper_fade(float p_energy);
	float get_upper_fade() const;

	void set_lower_fade(float p_fade);
	float get_lower_fade() const;

	void set_normal_fade(float p_fade);
	float get_normal_fade() const;

	void set_enable_distance_fade(bool p_enable);
	bool is_distance_fade_enabled() const;

	void set_distance_fade_begin(float p_distance);
	float get_distance_fade_begin() const;

	void set_distance_fade_length(float p_length);
	float get_distance_fade_length() const;

	void set_cull_mask(uint32_t p_layers);
	uint32_t get_cull_mask() const;

	virtual AABB get_aabb() const override;
	virtual Vector<Face3> get_faces(uint32_t p_usage_flags) const override;

	Decal();
	~Decal();
};

VARIANT_ENUM_CAST(Decal::DecalTexture);

#endif // DECAL_H
