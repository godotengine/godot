/**************************************************************************/
/*  rd_texture_format.hpp                                                 */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RDTextureFormat : public RefCounted {
	GDEXTENSION_CLASS(RDTextureFormat, RefCounted)

public:
	void set_format(RenderingDevice::DataFormat p_member);
	RenderingDevice::DataFormat get_format() const;
	void set_width(uint32_t p_member);
	uint32_t get_width() const;
	void set_height(uint32_t p_member);
	uint32_t get_height() const;
	void set_depth(uint32_t p_member);
	uint32_t get_depth() const;
	void set_array_layers(uint32_t p_member);
	uint32_t get_array_layers() const;
	void set_mipmaps(uint32_t p_member);
	uint32_t get_mipmaps() const;
	void set_texture_type(RenderingDevice::TextureType p_member);
	RenderingDevice::TextureType get_texture_type() const;
	void set_samples(RenderingDevice::TextureSamples p_member);
	RenderingDevice::TextureSamples get_samples() const;
	void set_usage_bits(BitField<RenderingDevice::TextureUsageBits> p_member);
	BitField<RenderingDevice::TextureUsageBits> get_usage_bits() const;
	void set_is_resolve_buffer(bool p_member);
	bool get_is_resolve_buffer() const;
	void set_is_discardable(bool p_member);
	bool get_is_discardable() const;
	void add_shareable_format(RenderingDevice::DataFormat p_format);
	void remove_shareable_format(RenderingDevice::DataFormat p_format);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

