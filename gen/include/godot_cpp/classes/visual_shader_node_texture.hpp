/**************************************************************************/
/*  visual_shader_node_texture.hpp                                        */
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
#include <godot_cpp/classes/visual_shader_node.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class VisualShaderNodeTexture : public VisualShaderNode {
	GDEXTENSION_CLASS(VisualShaderNodeTexture, VisualShaderNode)

public:
	enum Source {
		SOURCE_TEXTURE = 0,
		SOURCE_SCREEN = 1,
		SOURCE_2D_TEXTURE = 2,
		SOURCE_2D_NORMAL = 3,
		SOURCE_DEPTH = 4,
		SOURCE_PORT = 5,
		SOURCE_3D_NORMAL = 6,
		SOURCE_ROUGHNESS = 7,
		SOURCE_MAX = 8,
	};

	enum TextureType {
		TYPE_DATA = 0,
		TYPE_COLOR = 1,
		TYPE_NORMAL_MAP = 2,
		TYPE_MAX = 3,
	};

	void set_source(VisualShaderNodeTexture::Source p_value);
	VisualShaderNodeTexture::Source get_source() const;
	void set_texture(const Ref<Texture2D> &p_value);
	Ref<Texture2D> get_texture() const;
	void set_texture_type(VisualShaderNodeTexture::TextureType p_value);
	VisualShaderNodeTexture::TextureType get_texture_type() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNodeTexture::Source);
VARIANT_ENUM_CAST(VisualShaderNodeTexture::TextureType);

