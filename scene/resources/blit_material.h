/**************************************************************************/
/*  blit_material.h                                                       */
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

#pragma once

#include "scene/resources/material.h"

class BlitMaterial : public Material {
	GDCLASS(BlitMaterial, Material);

public:
	enum BlendMode {
		BLEND_MODE_MIX,
		BLEND_MODE_ADD,
		BLEND_MODE_SUB,
		BLEND_MODE_MUL,
		BLEND_MODE_DISABLED
	};

private:
	union MaterialKey {
		struct {
			uint32_t blend_mode : 4;
			uint32_t invalid_key : 1;
		};

		uint32_t key = 0;

		static uint32_t hash(const MaterialKey &p_key) {
			return hash_murmur3_one_32(p_key.key);
		}
		bool operator==(const MaterialKey &p_key) const {
			return key == p_key.key;
		}
	};

	struct ShaderData {
		RID shader;
		int users = 0;
	};

	static inline HashMap<MaterialKey, ShaderData, MaterialKey> shader_map;

	MaterialKey current_key;

	_FORCE_INLINE_ MaterialKey _compute_key() const {
		MaterialKey mk;
		mk.key = 0;
		mk.blend_mode = blend_mode;
		return mk;
	}

	static inline Mutex material_mutex;
	static inline SelfList<BlitMaterial>::List dirty_materials;
	SelfList<BlitMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();

	BlendMode blend_mode = BLEND_MODE_MIX;

protected:
	static void _bind_methods();

public:
	void set_blend_mode(BlendMode p_blend_mode);
	BlendMode get_blend_mode() const;

	virtual RID get_shader_rid() const override;

	virtual Shader::Mode get_shader_mode() const override;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	BlitMaterial();
	virtual ~BlitMaterial();
};

VARIANT_ENUM_CAST(BlitMaterial::BlendMode);
