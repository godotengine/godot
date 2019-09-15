/*************************************************************************/
/*  sky_material.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/rid.h"
#include "scene/resources/material.h"

#ifndef SKY_MATERIAL_H
#define SKY_MATERIAL_H

class SkyMaterial : public Material {

	GDCLASS(SkyMaterial, Material);

public:

private:
	union MaterialKey {

		struct {
			uint32_t texture_mask : 16;
			uint32_t texture_color : 1;
			uint32_t flags : 4;
			uint32_t emission_shape : 2;
			uint32_t trail_size_texture : 1;
			uint32_t trail_color_texture : 1;
			uint32_t invalid_key : 1;
			uint32_t has_emission_color : 1;
		};

		uint32_t key;

		bool operator<(const MaterialKey &p_key) const {
			return key < p_key.key;
		}
	};

	struct ShaderData {
		RID shader;
		int users;
	};

	static Map<MaterialKey, ShaderData> shader_map;

	MaterialKey current_key;

	_FORCE_INLINE_ MaterialKey _compute_key() const {

		MaterialKey mk;
		mk.key = 0;
		/*
		for (int i = 0; i < PARAM_MAX; i++) {
			if (tex_parameters[i].is_valid()) {
				mk.texture_mask |= (1 << i);
			}
		}
		for (int i = 0; i < FLAG_MAX; i++) {
			if (flags[i]) {
				mk.flags |= (1 << i);
			}
		}

		mk.texture_color = color_ramp.is_valid() ? 1 : 0;
		mk.emission_shape = emission_shape;
		mk.trail_color_texture = trail_color_modifier.is_valid() ? 1 : 0;
		mk.trail_size_texture = trail_size_modifier.is_valid() ? 1 : 0;
		mk.has_emission_color = emission_shape >= EMISSION_SHAPE_POINTS && emission_color_texture.is_valid();
		*/

		return mk;
	}

	static Mutex *material_mutex;
	static SelfList<SkyMaterial>::List *dirty_materials;

	struct ShaderNames {
		StringName placeholder;
	};

	static ShaderNames *shader_names;

	SelfList<SkyMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const;

public:
	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	RID get_shader_rid() const;

	virtual Shader::Mode get_shader_mode() const;

	SkyMaterial();
	~SkyMaterial();
};

#endif /* !SKY_MATERIAL_H */
