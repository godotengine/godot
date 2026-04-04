/**************************************************************************/
/*  editor_export_platform.hpp                                            */
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

#include <godot_cpp/classes/editor_export_preset.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class EditorExportPlatform : public RefCounted {
	GDEXTENSION_CLASS(EditorExportPlatform, RefCounted)

public:
	enum ExportMessageType {
		EXPORT_MESSAGE_NONE = 0,
		EXPORT_MESSAGE_INFO = 1,
		EXPORT_MESSAGE_WARNING = 2,
		EXPORT_MESSAGE_ERROR = 3,
	};

	enum DebugFlags : uint64_t {
		DEBUG_FLAG_DUMB_CLIENT = 1,
		DEBUG_FLAG_REMOTE_DEBUG = 2,
		DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST = 4,
		DEBUG_FLAG_VIEW_COLLISIONS = 8,
		DEBUG_FLAG_VIEW_NAVIGATION = 16,
	};

	String get_os_name() const;
	Ref<EditorExportPreset> create_preset();
	Dictionary find_export_template(const String &p_template_file_name) const;
	Array get_current_presets() const;
	Dictionary save_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, bool p_embed = false);
	Dictionary save_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path);
	Dictionary save_pack_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path);
	Dictionary save_zip_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path);
	PackedStringArray gen_export_flags(BitField<EditorExportPlatform::DebugFlags> p_flags);
	Error export_project_files(const Ref<EditorExportPreset> &p_preset, bool p_debug, const Callable &p_save_cb, const Callable &p_shared_cb = Callable());
	Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags = (BitField<EditorExportPlatform::DebugFlags>)0);
	Error export_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags = (BitField<EditorExportPlatform::DebugFlags>)0);
	Error export_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags = (BitField<EditorExportPlatform::DebugFlags>)0);
	Error export_pack_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const PackedStringArray &p_patches = PackedStringArray(), BitField<EditorExportPlatform::DebugFlags> p_flags = (BitField<EditorExportPlatform::DebugFlags>)0);
	Error export_zip_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const PackedStringArray &p_patches = PackedStringArray(), BitField<EditorExportPlatform::DebugFlags> p_flags = (BitField<EditorExportPlatform::DebugFlags>)0);
	void clear_messages();
	void add_message(EditorExportPlatform::ExportMessageType p_type, const String &p_category, const String &p_message);
	int32_t get_message_count() const;
	EditorExportPlatform::ExportMessageType get_message_type(int32_t p_index) const;
	String get_message_category(int32_t p_index) const;
	String get_message_text(int32_t p_index) const;
	EditorExportPlatform::ExportMessageType get_worst_message_type() const;
	Error ssh_run_on_remote(const String &p_host, const String &p_port, const PackedStringArray &p_ssh_arg, const String &p_cmd_args, const Array &p_output = Array(), int32_t p_port_fwd = -1) const;
	int64_t ssh_run_on_remote_no_wait(const String &p_host, const String &p_port, const PackedStringArray &p_ssh_args, const String &p_cmd_args, int32_t p_port_fwd = -1) const;
	Error ssh_push_to_remote(const String &p_host, const String &p_port, const PackedStringArray &p_scp_args, const String &p_src_file, const String &p_dst_file) const;
	Dictionary get_internal_export_files(const Ref<EditorExportPreset> &p_preset, bool p_debug);
	static PackedStringArray get_forced_export_files(const Ref<EditorExportPreset> &p_preset = nullptr);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorExportPlatform::ExportMessageType);
VARIANT_BITFIELD_CAST(EditorExportPlatform::DebugFlags);

