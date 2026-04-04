/**************************************************************************/
/*  canvas_item_material.hpp                                              */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CanvasItemMaterial : public Material {
	GDEXTENSION_CLASS(CanvasItemMaterial, Material)

public:
	enum BlendMode {
		BLEND_MODE_MIX = 0,
		BLEND_MODE_ADD = 1,
		BLEND_MODE_SUB = 2,
		BLEND_MODE_MUL = 3,
		BLEND_MODE_PREMULT_ALPHA = 4,
	};

	enum LightMode {
		LIGHT_MODE_NORMAL = 0,
		LIGHT_MODE_UNSHADED = 1,
		LIGHT_MODE_LIGHT_ONLY = 2,
	};

	void set_blend_mode(CanvasItemMaterial::BlendMode p_blend_mode);
	CanvasItemMaterial::BlendMode get_blend_mode() const;
	void set_light_mode(CanvasItemMaterial::LightMode p_light_mode);
	CanvasItemMaterial::LightMode get_light_mode() const;
	void set_particles_animation(bool p_particles_anim);
	bool get_particles_animation() const;
	void set_particles_anim_h_frames(int32_t p_frames);
	int32_t get_particles_anim_h_frames() const;
	void set_particles_anim_v_frames(int32_t p_frames);
	int32_t get_particles_anim_v_frames() const;
	void set_particles_anim_loop(bool p_loop);
	bool get_particles_anim_loop() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Material::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CanvasItemMaterial::BlendMode);
VARIANT_ENUM_CAST(CanvasItemMaterial::LightMode);

