/**************************************************************************/
/*  visual_shader_node_texture_parameter.hpp                              */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/visual_shader_node_parameter.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VisualShaderNodeTextureParameter : public VisualShaderNodeParameter {
	GDEXTENSION_CLASS(VisualShaderNodeTextureParameter, VisualShaderNodeParameter)

public:
	enum TextureType {
		TYPE_DATA = 0,
		TYPE_COLOR = 1,
		TYPE_NORMAL_MAP = 2,
		TYPE_ANISOTROPY = 3,
		TYPE_MAX = 4,
	};

	enum ColorDefault {
		COLOR_DEFAULT_WHITE = 0,
		COLOR_DEFAULT_BLACK = 1,
		COLOR_DEFAULT_TRANSPARENT = 2,
		COLOR_DEFAULT_MAX = 3,
	};

	enum TextureFilter {
		FILTER_DEFAULT = 0,
		FILTER_NEAREST = 1,
		FILTER_LINEAR = 2,
		FILTER_NEAREST_MIPMAP = 3,
		FILTER_LINEAR_MIPMAP = 4,
		FILTER_NEAREST_MIPMAP_ANISOTROPIC = 5,
		FILTER_LINEAR_MIPMAP_ANISOTROPIC = 6,
		FILTER_MAX = 7,
	};

	enum TextureRepeat {
		REPEAT_DEFAULT = 0,
		REPEAT_ENABLED = 1,
		REPEAT_DISABLED = 2,
		REPEAT_MAX = 3,
	};

	enum TextureSource {
		SOURCE_NONE = 0,
		SOURCE_SCREEN = 1,
		SOURCE_DEPTH = 2,
		SOURCE_NORMAL_ROUGHNESS = 3,
		SOURCE_MAX = 4,
	};

	void set_texture_type(VisualShaderNodeTextureParameter::TextureType p_type);
	VisualShaderNodeTextureParameter::TextureType get_texture_type() const;
	void set_color_default(VisualShaderNodeTextureParameter::ColorDefault p_color);
	VisualShaderNodeTextureParameter::ColorDefault get_color_default() const;
	void set_texture_filter(VisualShaderNodeTextureParameter::TextureFilter p_filter);
	VisualShaderNodeTextureParameter::TextureFilter get_texture_filter() const;
	void set_texture_repeat(VisualShaderNodeTextureParameter::TextureRepeat p_repeat);
	VisualShaderNodeTextureParameter::TextureRepeat get_texture_repeat() const;
	void set_texture_source(VisualShaderNodeTextureParameter::TextureSource p_source);
	VisualShaderNodeTextureParameter::TextureSource get_texture_source() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNodeParameter::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNodeTextureParameter::TextureType);
VARIANT_ENUM_CAST(VisualShaderNodeTextureParameter::ColorDefault);
VARIANT_ENUM_CAST(VisualShaderNodeTextureParameter::TextureFilter);
VARIANT_ENUM_CAST(VisualShaderNodeTextureParameter::TextureRepeat);
VARIANT_ENUM_CAST(VisualShaderNodeTextureParameter::TextureSource);

