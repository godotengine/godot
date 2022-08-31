/*************************************************************************/
/*  editor_export_plugin.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_EXPORT_PLUGIN_H
#define EDITOR_EXPORT_PLUGIN_H

#include "core/extension/native_extension.h"
#include "editor_export_preset.h"
#include "editor_export_shared_object.h"
#include "scene/main/node.h"

class EditorExportPlugin : public RefCounted {
	GDCLASS(EditorExportPlugin, RefCounted);

	friend class EditorExportPlatform;

	Ref<EditorExportPreset> export_preset;

	Vector<SharedObject> shared_objects;
	struct ExtraFile {
		String path;
		Vector<uint8_t> data;
		bool remap = false;
	};
	Vector<ExtraFile> extra_files;
	bool skipped = false;

	Vector<String> ios_frameworks;
	Vector<String> ios_embedded_frameworks;
	Vector<String> ios_project_static_libs;
	String ios_plist_content;
	String ios_linker_flags;
	Vector<String> ios_bundle_files;
	String ios_cpp_code;

	Vector<String> macos_plugin_files;

	_FORCE_INLINE_ void _clear() {
		shared_objects.clear();
		extra_files.clear();
		skipped = false;
	}

	_FORCE_INLINE_ void _export_end() {
		ios_frameworks.clear();
		ios_embedded_frameworks.clear();
		ios_bundle_files.clear();
		ios_plist_content = "";
		ios_linker_flags = "";
		ios_cpp_code = "";
		macos_plugin_files.clear();
	}

	// Export
	void _export_file_script(const String &p_path, const String &p_type, const Vector<String> &p_features);
	void _export_begin_script(const Vector<String> &p_features, bool p_debug, const String &p_path, int p_flags);
	void _export_end_script();

protected:
	void set_export_preset(const Ref<EditorExportPreset> &p_preset);
	Ref<EditorExportPreset> get_export_preset() const;

	void add_file(const String &p_path, const Vector<uint8_t> &p_file, bool p_remap);
	void add_shared_object(const String &p_path, const Vector<String> &tags, const String &p_target = String());

	void add_ios_framework(const String &p_path);
	void add_ios_embedded_framework(const String &p_path);
	void add_ios_project_static_lib(const String &p_path);
	void add_ios_plist_content(const String &p_plist_content);
	void add_ios_linker_flags(const String &p_flags);
	void add_ios_bundle_file(const String &p_path);
	void add_ios_cpp_code(const String &p_code);
	void add_macos_plugin_file(const String &p_path);

	void skip();

	virtual void _export_file(const String &p_path, const String &p_type, const HashSet<String> &p_features);
	virtual void _export_begin(const HashSet<String> &p_features, bool p_debug, const String &p_path, int p_flags);

	static void _bind_methods();

	GDVIRTUAL3(_export_file, String, String, Vector<String>)
	GDVIRTUAL4(_export_begin, Vector<String>, bool, String, uint32_t)
	GDVIRTUAL0(_export_end)

	GDVIRTUAL2RC(bool, _begin_customize_resources, const Ref<EditorExportPlatform> &, const Vector<String> &)
	GDVIRTUAL2R(Ref<Resource>, _customize_resource, const Ref<Resource> &, String)

	GDVIRTUAL2RC(bool, _begin_customize_scenes, const Ref<EditorExportPlatform> &, const Vector<String> &)
	GDVIRTUAL2R(Node *, _customize_scene, Node *, String)
	GDVIRTUAL0RC(uint64_t, _get_customization_configuration_hash)

	GDVIRTUAL0(_end_customize_scenes)
	GDVIRTUAL0(_end_customize_resources)

	GDVIRTUAL0RC(String, _get_name)

	bool _begin_customize_resources(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) const; // Return true if this plugin does property export customization
	Ref<Resource> _customize_resource(const Ref<Resource> &p_resource, const String &p_path); // If nothing is returned, it means do not touch (nothing changed). If something is returned (either the same or a different resource) it means changes are made.

	bool _begin_customize_scenes(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) const; // Return true if this plugin does property export customization
	Node *_customize_scene(Node *p_root, const String &p_path); // Return true if a change was made

	uint64_t _get_customization_configuration_hash() const; // Hash used for caching customized resources and scenes.

	void _end_customize_scenes();
	void _end_customize_resources();

	virtual String _get_name() const;

public:
	Vector<String> get_ios_frameworks() const;
	Vector<String> get_ios_embedded_frameworks() const;
	Vector<String> get_ios_project_static_libs() const;
	String get_ios_plist_content() const;
	String get_ios_linker_flags() const;
	Vector<String> get_ios_bundle_files() const;
	String get_ios_cpp_code() const;
	const Vector<String> &get_macos_plugin_files() const;

	EditorExportPlugin();
};

#endif // EDITOR_EXPORT_PLUGIN_H
