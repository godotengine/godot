/**************************************************************************/
/*  portable_compressed_texture2d.hpp                                     */
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

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PortableCompressedTexture2D : public Texture2D {
	GDEXTENSION_CLASS(PortableCompressedTexture2D, Texture2D)

public:
	enum CompressionMode {
		COMPRESSION_MODE_LOSSLESS = 0,
		COMPRESSION_MODE_LOSSY = 1,
		COMPRESSION_MODE_BASIS_UNIVERSAL = 2,
		COMPRESSION_MODE_S3TC = 3,
		COMPRESSION_MODE_ETC2 = 4,
		COMPRESSION_MODE_BPTC = 5,
		COMPRESSION_MODE_ASTC = 6,
	};

	void create_from_image(const Ref<Image> &p_image, PortableCompressedTexture2D::CompressionMode p_compression_mode, bool p_normal_map = false, float p_lossy_quality = 0.8);
	Image::Format get_format() const;
	PortableCompressedTexture2D::CompressionMode get_compression_mode() const;
	void set_size_override(const Vector2 &p_size);
	Vector2 get_size_override() const;
	void set_keep_compressed_buffer(bool p_keep);
	bool is_keeping_compressed_buffer() const;
	void set_basisu_compressor_params(int32_t p_uastc_level, float p_rdo_quality_loss);
	static void set_keep_all_compressed_buffers(bool p_keep);
	static bool is_keeping_all_compressed_buffers();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Texture2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(PortableCompressedTexture2D::CompressionMode);

