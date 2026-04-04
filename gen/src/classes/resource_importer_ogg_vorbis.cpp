/**************************************************************************/
/*  resource_importer_ogg_vorbis.cpp                                      */
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

#include <godot_cpp/classes/resource_importer_ogg_vorbis.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/audio_stream_ogg_vorbis.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

Ref<AudioStreamOggVorbis> ResourceImporterOggVorbis::load_from_buffer(const PackedByteArray &p_stream_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceImporterOggVorbis::get_class_static()._native_ptr(), StringName("load_from_buffer")._native_ptr(), 354904730);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamOggVorbis>()));
	return Ref<AudioStreamOggVorbis>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamOggVorbis>(_gde_method_bind, nullptr, &p_stream_data));
}

Ref<AudioStreamOggVorbis> ResourceImporterOggVorbis::load_from_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceImporterOggVorbis::get_class_static()._native_ptr(), StringName("load_from_file")._native_ptr(), 797568536);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamOggVorbis>()));
	return Ref<AudioStreamOggVorbis>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamOggVorbis>(_gde_method_bind, nullptr, &p_path));
}

} // namespace godot
