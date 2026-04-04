/**************************************************************************/
/*  editor_export_plugin.hpp                                              */
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
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class EditorExportPlatform;
class EditorExportPreset;
class Node;
class Resource;
class StringName;

class EditorExportPlugin : public RefCounted {
	GDEXTENSION_CLASS(EditorExportPlugin, RefCounted)

public:
	void add_shared_object(const String &p_path, const PackedStringArray &p_tags, const String &p_target);
	void add_file(const String &p_path, const PackedByteArray &p_file, bool p_remap);
	void add_apple_embedded_platform_project_static_lib(const String &p_path);
	void add_apple_embedded_platform_framework(const String &p_path);
	void add_apple_embedded_platform_embedded_framework(const String &p_path);
	void add_apple_embedded_platform_plist_content(const String &p_plist_content);
	void add_apple_embedded_platform_linker_flags(const String &p_flags);
	void add_apple_embedded_platform_bundle_file(const String &p_path);
	void add_apple_embedded_platform_cpp_code(const String &p_code);
	void add_ios_project_static_lib(const String &p_path);
	void add_ios_framework(const String &p_path);
	void add_ios_embedded_framework(const String &p_path);
	void add_ios_plist_content(const String &p_plist_content);
	void add_ios_linker_flags(const String &p_flags);
	void add_ios_bundle_file(const String &p_path);
	void add_ios_cpp_code(const String &p_code);
	void add_macos_plugin_file(const String &p_path);
	void skip();
	Variant get_option(const StringName &p_name) const;
	Ref<EditorExportPreset> get_export_preset() const;
	Ref<EditorExportPlatform> get_export_platform() const;
	virtual void _export_file(const String &p_path, const String &p_type, const PackedStringArray &p_features);
	virtual void _export_begin(const PackedStringArray &p_features, bool p_is_debug, const String &p_path, uint32_t p_flags);
	virtual void _export_end();
	virtual bool _begin_customize_resources(const Ref<EditorExportPlatform> &p_platform, const PackedStringArray &p_features) const;
	virtual Ref<Resource> _customize_resource(const Ref<Resource> &p_resource, const String &p_path);
	virtual bool _begin_customize_scenes(const Ref<EditorExportPlatform> &p_platform, const PackedStringArray &p_features) const;
	virtual Node *_customize_scene(Node *p_scene, const String &p_path);
	virtual uint64_t _get_customization_configuration_hash() const;
	virtual void _end_customize_scenes();
	virtual void _end_customize_resources();
	virtual TypedArray<Dictionary> _get_export_options(const Ref<EditorExportPlatform> &p_platform) const;
	virtual Dictionary _get_export_options_overrides(const Ref<EditorExportPlatform> &p_platform) const;
	virtual bool _should_update_export_options(const Ref<EditorExportPlatform> &p_platform) const;
	virtual bool _get_export_option_visibility(const Ref<EditorExportPlatform> &p_platform, const String &p_option) const;
	virtual String _get_export_option_warning(const Ref<EditorExportPlatform> &p_platform, const String &p_option) const;
	virtual PackedStringArray _get_export_features(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual String _get_name() const;
	virtual bool _supports_platform(const Ref<EditorExportPlatform> &p_platform) const;
	virtual PackedStringArray _get_android_dependencies(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual PackedStringArray _get_android_dependencies_maven_repos(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual PackedStringArray _get_android_libraries(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual String _get_android_manifest_activity_element_contents(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual String _get_android_manifest_application_element_contents(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual String _get_android_manifest_element_contents(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const;
	virtual PackedByteArray _update_android_prebuilt_manifest(const Ref<EditorExportPlatform> &p_platform, const PackedByteArray &p_manifest_data) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_export_file), decltype(&T::_export_file)>) {
			BIND_VIRTUAL_METHOD(T, _export_file, 3533781844);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_begin), decltype(&T::_export_begin)>) {
			BIND_VIRTUAL_METHOD(T, _export_begin, 2765511433);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_end), decltype(&T::_export_end)>) {
			BIND_VIRTUAL_METHOD(T, _export_end, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_begin_customize_resources), decltype(&T::_begin_customize_resources)>) {
			BIND_VIRTUAL_METHOD(T, _begin_customize_resources, 1312023292);
		}
		if constexpr (!std::is_same_v<decltype(&B::_customize_resource), decltype(&T::_customize_resource)>) {
			BIND_VIRTUAL_METHOD(T, _customize_resource, 307917495);
		}
		if constexpr (!std::is_same_v<decltype(&B::_begin_customize_scenes), decltype(&T::_begin_customize_scenes)>) {
			BIND_VIRTUAL_METHOD(T, _begin_customize_scenes, 1312023292);
		}
		if constexpr (!std::is_same_v<decltype(&B::_customize_scene), decltype(&T::_customize_scene)>) {
			BIND_VIRTUAL_METHOD(T, _customize_scene, 498701822);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_customization_configuration_hash), decltype(&T::_get_customization_configuration_hash)>) {
			BIND_VIRTUAL_METHOD(T, _get_customization_configuration_hash, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_end_customize_scenes), decltype(&T::_end_customize_scenes)>) {
			BIND_VIRTUAL_METHOD(T, _end_customize_scenes, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_end_customize_resources), decltype(&T::_end_customize_resources)>) {
			BIND_VIRTUAL_METHOD(T, _end_customize_resources, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_export_options), decltype(&T::_get_export_options)>) {
			BIND_VIRTUAL_METHOD(T, _get_export_options, 488349689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_export_options_overrides), decltype(&T::_get_export_options_overrides)>) {
			BIND_VIRTUAL_METHOD(T, _get_export_options_overrides, 2837326714);
		}
		if constexpr (!std::is_same_v<decltype(&B::_should_update_export_options), decltype(&T::_should_update_export_options)>) {
			BIND_VIRTUAL_METHOD(T, _should_update_export_options, 1866233299);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_export_option_visibility), decltype(&T::_get_export_option_visibility)>) {
			BIND_VIRTUAL_METHOD(T, _get_export_option_visibility, 3537301980);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_export_option_warning), decltype(&T::_get_export_option_warning)>) {
			BIND_VIRTUAL_METHOD(T, _get_export_option_warning, 3340251247);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_export_features), decltype(&T::_get_export_features)>) {
			BIND_VIRTUAL_METHOD(T, _get_export_features, 1057664154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_name), decltype(&T::_get_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_name, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_supports_platform), decltype(&T::_supports_platform)>) {
			BIND_VIRTUAL_METHOD(T, _supports_platform, 1866233299);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_android_dependencies), decltype(&T::_get_android_dependencies)>) {
			BIND_VIRTUAL_METHOD(T, _get_android_dependencies, 1057664154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_android_dependencies_maven_repos), decltype(&T::_get_android_dependencies_maven_repos)>) {
			BIND_VIRTUAL_METHOD(T, _get_android_dependencies_maven_repos, 1057664154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_android_libraries), decltype(&T::_get_android_libraries)>) {
			BIND_VIRTUAL_METHOD(T, _get_android_libraries, 1057664154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_android_manifest_activity_element_contents), decltype(&T::_get_android_manifest_activity_element_contents)>) {
			BIND_VIRTUAL_METHOD(T, _get_android_manifest_activity_element_contents, 4013372917);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_android_manifest_application_element_contents), decltype(&T::_get_android_manifest_application_element_contents)>) {
			BIND_VIRTUAL_METHOD(T, _get_android_manifest_application_element_contents, 4013372917);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_android_manifest_element_contents), decltype(&T::_get_android_manifest_element_contents)>) {
			BIND_VIRTUAL_METHOD(T, _get_android_manifest_element_contents, 4013372917);
		}
		if constexpr (!std::is_same_v<decltype(&B::_update_android_prebuilt_manifest), decltype(&T::_update_android_prebuilt_manifest)>) {
			BIND_VIRTUAL_METHOD(T, _update_android_prebuilt_manifest, 3304965187);
		}
	}

public:
};

} // namespace godot

