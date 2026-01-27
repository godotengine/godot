/**************************************************************************/
/*  export_plugin.cpp                                                     */
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

#include "export_plugin.h"

#include "core/crypto/hashing_context.h"
#include "core/extension/gdextension.h"
#include "core/io/config_file.h"
#include "core/io/json.h"
#include "core/io/resource_loader.h"
#include "core/os/memory.h"
#include "core/string/string_builder.h"
#include "editor/export/editor_export_platform.h"
#include "logo_svg.gen.h"
#include "run_icon_svg.gen.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "editor/editor_string_names.h"
#include "editor/export/editor_export.h"
#include "editor/export/editor_export_platform_utils.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/import/resource_importer_texture_settings.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/image_texture.h"

#include "modules/modules_enabled.gen.h" // For mono.
#include "modules/svg/image_loader_svg.h"

#include <functional>

/**
 * EditorExportPlatformWeb::ExportData::ResourceData
 */
uint32_t EditorExportPlatformWeb::ExportData::ResourceData::get_size() const {
	uint32_t size = 0;
	if (native_file.exists) {
		size += native_file.size;
	}
	if (remap_file.exists) {
		size += remap_file.size;
	}
	if (remapped_file.exists) {
		size += remapped_file.size;
	}
	return size;
}

Dictionary EditorExportPlatformWeb::ExportData::ResourceData::get_as_resource_dictionary() const {
	Dictionary data;
	Dictionary resources;

	if (native_file.exists) {
		resources[native_file.resource_path] = native_file.get_as_dictionary();
	}
	if (remap_file.exists) {
		resources[remap_file.resource_path] = remap_file.get_as_dictionary();
	}
	if (remapped_file.exists) {
		resources[remapped_file.resource_path] = remapped_file.get_as_dictionary();
	}
	data["files"] = resources;
	data["totalSize"] = get_size();
	return data;
}

String EditorExportPlatformWeb::ExportData::ResourceData::get_resource_path() const {
	if (remap_file.exists) {
		return remap_file.resource_path;
	}
	return native_file.resource_path;
}

void EditorExportPlatformWeb::ExportData::ResourceData::flatten_dependencies(LocalVector<const ResourceData *> *p_deps) const {
	ERR_FAIL_NULL(p_deps);

	for (const ResourceData *dependency : dependencies) {
		if (p_deps->has(dependency)) {
			continue;
		}
		p_deps->push_back(dependency);
		dependency->flatten_dependencies(p_deps);
	}
}

HashSet<String> EditorExportPlatformWeb::ExportData::get_features_set() const {
	List<String> features_list;

	preset->get_platform()->get_platform_features(&features_list);
	preset->get_platform()->get_preset_features(preset, &features_list);

	String custom = preset->get_custom_features();
	Vector<String> custom_list = custom.split(",");
	for (int i = 0; i < custom_list.size(); i++) {
		String f = custom_list[i].strip_edges();
		if (!f.is_empty()) {
			features_list.push_back(f);
		}
	}

	HashSet<String> features_set;
	for (const String &feature : features_list) {
		features_set.insert(feature);
	}
	return features_set;
}

EditorExportPlatformWeb::ExportData::ResourceData *EditorExportPlatformWeb::ExportData::add_dependency(const String &p_path, const HashSet<String> &p_features_set, Ref<FileAccess> p_uid_cache, /* bool p_encrypt, */ Error *r_error) {
	ResourceData *data = nullptr;
	List<ResourceData>::Element *data_iterator = nullptr;

	String remap_path;
	bool has_suffix_import = false;

#define SET_ERR(m_err)        \
	if (r_error != nullptr) { \
		*r_error = (m_err);   \
	}                         \
	((void)0)

#define _HANDLE_ERR_COND_V_MSG(m_cond, m_err, m_msg)       \
	if (unlikely((m_cond))) {                              \
		SET_ERR(m_err);                                    \
		if (data != nullptr && data_iterator != nullptr) { \
			dependencies.erase(data_iterator);             \
		}                                                  \
		ERR_FAIL_V_MSG(nullptr, (m_msg));                  \
		return nullptr;                                    \
	}                                                      \
	((void)0)

	_HANDLE_ERR_COND_V_MSG(p_path.is_empty(), ERR_INVALID_PARAMETER, "p_path.is_empty()");

	if (dependencies_map.has(p_path)) {
		SET_ERR(OK);
		return dependencies_map[p_path];
	}

	data_iterator = dependencies.push_back({});
	data = &data_iterator->get();
	data->path = p_path;
	update_file(&data->native_file, p_path);

	if (FileAccess::exists(data->native_file.absolute_path + SUFFIX_IMPORT)) {
		has_suffix_import = true;
		remap_path = data->native_file.resource_path + SUFFIX_IMPORT;
	} else if (FileAccess::exists(data->native_file.absolute_path + SUFFIX_REMAP)) {
		remap_path = data->native_file.resource_path + SUFFIX_REMAP;
	}

	_HANDLE_ERR_COND_V_MSG(!data->native_file.exists && remap_path.is_empty(), ERR_FILE_NOT_FOUND, vformat(R"*("%s" doesn't exist, and there is no remap/import file.)*", data->native_file.absolute_path));

	if (!remap_path.is_empty()) {
		update_file(&data->remap_file, remap_path);
		_HANDLE_ERR_COND_V_MSG(!data->remap_file.exists, ERR_FILE_NOT_FOUND, vformat(R"*("%s" doesn't exist)*", data->remap_file.absolute_path));

		Error err;
		Ref<FileAccess> remap_file_access = FileAccess::open(data->remap_file.absolute_path, FileAccess::READ, &err);
		_HANDLE_ERR_COND_V_MSG(err != OK, err, vformat(R"*(Error while opening "%s": %s)*", data->remap_file.absolute_path, error_names[err]));

		Ref<ConfigFile> remap_file;
		remap_file.instantiate();
		// if (p_is_encrypted) {
		// 	Ref<FileAccessEncrypted> remap_file_access_encrypted;
		// 	remap_file_access_encrypted.instantiate();
		// 	Error err = remap_file_access_encrypted->open_and_parse(remap_file_access, p_key, FileAccessEncrypted::MODE_READ, false);
		// 	ERR_FAIL_COND_V(err != OK, err);
		// 	remap_file_access = remap_file_access_encrypted;
		// }
		remap_file->parse(remap_file_access->get_as_text());

		const String PREFIX_PATH = "path.";
		const String PATH_UID = "uid";

		String remapped_path;
		String uid_path;
		Vector<String> remap_section_keys = remap_file->get_section_keys("remap");
		for (const String &remap_section_key : remap_section_keys) {
			bool found = false;
			if (remap_section_key == PATH_UID) {
				p_uid_cache->seek(0);
				uid_path = ResourceUID::get_path_from_cache(p_uid_cache, remap_file->get_value("remap", remap_section_key));
				continue;
			}
			if (remap_section_key.begins_with(PREFIX_PATH)) {
				String type = remap_section_key.trim_prefix(PREFIX_PATH);
				if (p_features_set.has(type)) {
					found = true;
				}
			}
			if (remap_section_key == "path") {
				found = true;
			}
			if (!found) {
				continue;
			}
			remapped_path = remap_file->get_value("remap", remap_section_key);
			break;
		}
		if (remapped_path.is_empty() && !uid_path.is_empty()) {
			remapped_path = uid_path;
		}
		_HANDLE_ERR_COND_V_MSG(remapped_path.is_empty(), ERR_PARSE_ERROR, vformat(R"*(Could not find any remap path in %s file "%s")*", has_suffix_import ? "import" : "remap", data->remap_file.absolute_path));

		update_file(&data->remapped_file, remapped_path);
	}

	dependencies_map.insert(p_path, data);

	File *resource_file = nullptr;
	if (data->native_file.exists && !data->remap_file.exists) {
		resource_file = &data->native_file;
	} else if (data->remapped_file.exists) {
		resource_file = &data->remapped_file;
	}

	if (resource_file != nullptr) {
		List<String> remapped_dependencies;
		ResourceLoader::get_dependencies(resource_file->absolute_path, &remapped_dependencies);
		for (const String &remapped_dependency : remapped_dependencies) {
			Error error;
			String remapped_dependency_path = EditorExportPlatformUtils::get_path_from_dependency(remapped_dependency);
			ResourceData *dependency = add_dependency(remapped_dependency_path, p_features_set, p_uid_cache, &error);
			_HANDLE_ERR_COND_V_MSG(error != OK, error, vformat(R"*(Error while processing remapped dependencies of "%s": couldn't add dependency of "%s")*", resource_file->absolute_path, remapped_dependency_path));
			data->dependencies.push_back(dependency);
		}
	}

	SET_ERR(OK);
	return data;

#undef _HANDLE_ERR_COND_V_MSG
#undef SET_ERR
}

void EditorExportPlatformWeb::ExportData::update_file(File *p_file, const String &p_resource_path) {
	ERR_FAIL_NULL(p_file);
	ERR_FAIL_COND(p_resource_path.is_empty());

	p_file->resource_path = p_resource_path;
	p_file->absolute_path = res_to_global(p_resource_path);
	p_file->exists = FileAccess::exists(p_file->absolute_path);
	if (!p_file->exists) {
		return;
	}

	p_file->size = FileAccess::get_size(p_file->absolute_path);
	if (p_file->size == 0) {
		return;
	}

	Ref<HashingContext> context_md5;
	context_md5.instantiate();
	context_md5->start(HashingContext::HASH_MD5);

	Ref<HashingContext> context_sha256;
	context_sha256.instantiate();
	context_sha256->start(HashingContext::HASH_SHA256);

	const uint64_t CHUNK_SIZE = 1024;
	Error error;
	Ref<FileAccess> file = FileAccess::open(p_file->absolute_path, FileAccess::READ, &error);
	if (error != OK) {
		ERR_FAIL_MSG(vformat(R"*(Error while opening "%s": %s)*", p_file->absolute_path, error_names[error]));
	}

	while (file->get_position() < file->get_length()) {
		uint64_t remaining = file->get_length() - file->get_position();
		PackedByteArray chunk = file->get_buffer(MIN(remaining, CHUNK_SIZE));
		context_md5->update(chunk);
		context_sha256->update(chunk);
	}

	PackedByteArray hash_md5 = context_md5->finish();
	PackedByteArray hash_sha256 = context_sha256->finish();

	p_file->md5 = String::hex_encode_buffer(hash_md5.ptr(), hash_md5.size());
	p_file->sha256 = String::hex_encode_buffer(hash_sha256.ptr(), hash_sha256.size());
}

Dictionary EditorExportPlatformWeb::ExportData::get_deps_json_dictionary(const ResourceData *p_dependency) {
	Dictionary deps;
	Dictionary resources;

	// Resources.
	deps["resources"] = resources;
	resources[p_dependency->path] = p_dependency->get_as_resource_dictionary();

	// Dependencies.
	Dictionary deps_dependencies;
	deps["dependencies"] = deps_dependencies;

	// Recursive dependency lambda.
	std::function<void(const ExportData::ResourceData *)> _l_add_deps_dependencies;
	_l_add_deps_dependencies = [&](const ExportData::ResourceData *l_dependency) -> void {
		resources[l_dependency->path] = l_dependency->get_as_resource_dictionary();
		LocalVector<const ExportData::ResourceData *> local_dependencies;
		l_dependency->flatten_dependencies(&local_dependencies);

		PackedStringArray paths_array;
		for (const ExportData::ResourceData *local_dependency : local_dependencies) {
			if (local_dependency->path != l_dependency->path) {
				paths_array.push_back(local_dependency->path);
			}
			if (!deps_dependencies.has(local_dependency->path)) {
				// Prevent infinite recursion.
				deps_dependencies[local_dependency->path] = {};
				_l_add_deps_dependencies(local_dependency);
			}
		}
		paths_array.sort_custom<FileNoCaseComparator>();
		deps_dependencies[l_dependency->path] = paths_array;
	};

	// Loop through each dependencies to find their dependencies.
	for (const ExportData::ResourceData *dependency : p_dependency->dependencies) {
		_l_add_deps_dependencies(dependency);
	}

	// Register the asked dependency itself.
	_l_add_deps_dependencies(p_dependency);

	return deps;
}

Error EditorExportPlatformWeb::ExportData::save_deps_json(const ResourceData *p_dependency) {
	Dictionary deps = get_deps_json_dictionary(p_dependency);
	String resource_path = p_dependency->get_resource_path();
	if (resource_path == p_dependency->remap_file.resource_path) {
		if (resource_path.ends_with(SUFFIX_REMAP)) {
			resource_path = resource_path.trim_suffix(SUFFIX_REMAP);
		} else {
			resource_path = resource_path.trim_suffix(SUFFIX_IMPORT);
		}
	}

	String deps_json_file_path = res_to_global(resource_path) + ".deps.json";
	Error error;
	Ref<FileAccess> deps_json_file = FileAccess::open(deps_json_file_path, FileAccess::WRITE, &error);
	if (error != OK) {
		ERR_PRINT(vformat(R"*(Could not write to "%s".)*", deps_json_file_path));
		return error;
	}
	deps_json_file->store_string(JSON::stringify(deps, String(" ").repeat(2)));

	return OK;
}

/**
 * EditorExportPlatformWeb
 */

Error EditorExportPlatformWeb::_extract_template(const String &p_template, const String &p_dir, const String &p_name, bool pwa) {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);
	unzFile pkg = unzOpen2(p_template.utf8().get_data(), &io);

	if (!pkg) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Prepare Templates"), vformat(TTR("Could not open template for export: \"%s\"."), p_template));
		return ERR_FILE_NOT_FOUND;
	}

	if (unzGoToFirstFile(pkg) != UNZ_OK) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Prepare Templates"), vformat(TTR("Invalid export template: \"%s\"."), p_template));
		unzClose(pkg);
		return ERR_FILE_CORRUPT;
	}

	do {
		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String file = String::utf8(fname);

		// Skip folders.
		if (file.ends_with("/")) {
			continue;
		}

		// Skip service worker and offline page if not exporting pwa.
		if (!pwa && (file == "godot.service.worker.js" || file == "godot.offline.html")) {
			continue;
		}
		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptrw(), data.size());
		unzCloseCurrentFile(pkg);

		//write
		String dst = p_dir.path_join(file.replace("godot", p_name));
		Ref<FileAccess> f = FileAccess::open(dst, FileAccess::WRITE);
		if (f.is_null()) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Prepare Templates"), vformat(TTR("Could not write file: \"%s\"."), dst));
			unzClose(pkg);
			return ERR_FILE_CANT_WRITE;
		}
		f->store_buffer(data.ptr(), data.size());

	} while (unzGoToNextFile(pkg) == UNZ_OK);
	unzClose(pkg);
	return OK;
}

Error EditorExportPlatformWeb::_write_or_error(const uint8_t *p_content, int p_size, String p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	if (f.is_null()) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), p_path));
		return ERR_FILE_CANT_WRITE;
	}
	f->store_buffer(p_content, p_size);
	return OK;
}

void EditorExportPlatformWeb::_replace_strings(const HashMap<String, String> &p_replaces, Vector<uint8_t> &r_template) {
	String str_template = String::utf8(reinterpret_cast<const char *>(r_template.ptr()), r_template.size());
	String out;
	Vector<String> lines = str_template.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		String current_line = lines[i];
		for (const KeyValue<String, String> &E : p_replaces) {
			current_line = current_line.replace(E.key, E.value);
		}
		out += current_line + "\n";
	}
	CharString cs = out.utf8();
	r_template.resize(cs.length());
	for (int i = 0; i < cs.length(); i++) {
		r_template.write[i] = cs[i];
	}
}

void EditorExportPlatformWeb::_fix_html(Vector<uint8_t> &p_html, const Ref<EditorExportPreset> &p_preset, const String &p_name, bool p_debug, BitField<EditorExportPlatform::DebugFlags> p_flags, const Vector<SharedObject> p_shared_objects, const Dictionary &p_file_sizes, const Dictionary &p_async_pck_data) {
	// Engine.js config
	Dictionary config;
	Array libs;
	for (int i = 0; i < p_shared_objects.size(); i++) {
		libs.push_back(p_shared_objects[i].path.get_file());
	}
	Vector<String> flags = gen_export_flags(p_flags & (~EditorExportPlatformData::DEBUG_FLAG_DUMB_CLIENT));
	Array args;
	for (int i = 0; i < flags.size(); i++) {
		args.push_back(flags[i]);
	}
	config["canvasResizePolicy"] = p_preset->get("html/canvas_resize_policy");
	config["experimentalVK"] = p_preset->get("html/experimental_virtual_keyboard");
	config["focusCanvas"] = p_preset->get("html/focus_canvas_on_start");
	config["gdextensionLibs"] = libs;
	config["executable"] = p_name;
	config["args"] = args;
	config["fileSizes"] = p_file_sizes;
	config["ensureCrossOriginIsolationHeaders"] = (bool)p_preset->get("progressive_web_app/ensure_cross_origin_isolation_headers");

	config["godotPoolSize"] = p_preset->get("threads/godot_pool_size");
	config["emscriptenPoolSize"] = p_preset->get("threads/emscripten_pool_size");

	AsyncLoadSetting async_initial_load_mode = (AsyncLoadSetting)(int)p_preset->get("async/initial_load_mode");
	switch (async_initial_load_mode) {
		case ASYNC_LOAD_SETTING_LOAD_EVERYTHING: {
			config["mainPack"] = p_name + ".pck";
		} break;
		case ASYNC_LOAD_SETTING_MINIMUM_INITIAL_RESOURCES: {
			config["mainPack"] = p_name + ".asyncpck";
			config["asyncPckData"] = p_async_pck_data;
		} break;
	}

	String head_include;
	if (p_preset->get("html/export_icon")) {
		head_include += "<link id=\"-gd-engine-icon\" rel=\"icon\" type=\"image/png\" href=\"" + p_name + ".icon.png\" />\n";
		head_include += "<link rel=\"apple-touch-icon\" href=\"" + p_name + ".apple-touch-icon.png\"/>\n";
	}
	if (p_preset->get("progressive_web_app/enabled")) {
		head_include += "<link rel=\"manifest\" href=\"" + p_name + ".manifest.json\">\n";
		config["serviceWorker"] = p_name + ".service.worker.js";
	}

	// Replaces HTML string
	const String str_config = Variant(config).to_json_string();
	const String custom_head_include = p_preset->get("html/head_include");
	HashMap<String, String> replaces;
	replaces["$GODOT_URL"] = p_name + ".js";
	replaces["$GODOT_PROJECT_NAME"] = get_project_setting(p_preset, "application/config/name");
	replaces["$GODOT_HEAD_INCLUDE"] = head_include + custom_head_include;
	replaces["$GODOT_CONFIG"] = str_config;
	replaces["$GODOT_SPLASH_COLOR"] = "#" + Color(get_project_setting(p_preset, "application/boot_splash/bg_color")).to_html(false);

	Vector<String> godot_splash_classes;
	godot_splash_classes.push_back("show-image--" + String(get_project_setting(p_preset, "application/boot_splash/show_image")));
	RenderingServer::SplashStretchMode boot_splash_stretch_mode = get_project_setting(p_preset, "application/boot_splash/stretch_mode");
	godot_splash_classes.push_back("fullsize--" + String(((boot_splash_stretch_mode != RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_DISABLED) ? "true" : "false")));
	godot_splash_classes.push_back("use-filter--" + String(get_project_setting(p_preset, "application/boot_splash/use_filter")));
	replaces["$GODOT_SPLASH_CLASSES"] = String(" ").join(godot_splash_classes);
	replaces["$GODOT_SPLASH"] = p_name + ".png";

	if (p_preset->get("variant/thread_support")) {
		replaces["$GODOT_THREADS_ENABLED"] = "true";
	} else {
		replaces["$GODOT_THREADS_ENABLED"] = "false";
	}

	_replace_strings(replaces, p_html);
}

Error EditorExportPlatformWeb::_add_manifest_icon(const Ref<EditorExportPreset> &p_preset, const String &p_path, const String &p_icon, int p_size, Array &r_arr) {
	const String name = p_path.get_file().get_basename();
	const String icon_name = vformat("%s.%dx%d.png", name, p_size, p_size);
	const String icon_dest = p_path.get_base_dir().path_join(icon_name);

	Ref<Image> icon;
	if (!p_icon.is_empty()) {
		Error err = OK;
		icon = _load_icon_or_splash_image(p_icon, &err);
		if (err != OK || icon.is_null() || icon->is_empty()) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Icon Creation"), vformat(TTR("Could not read file: \"%s\"."), p_icon));
			return err;
		}
		if (icon->get_width() != p_size || icon->get_height() != p_size) {
			icon->resize(p_size, p_size);
		}
	} else {
		icon = _get_project_icon(p_preset);
		icon->resize(p_size, p_size);
	}
	const Error err = icon->save_png(icon_dest);
	if (err != OK) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Icon Creation"), vformat(TTR("Could not write file: \"%s\"."), icon_dest));
		return err;
	}
	Dictionary icon_dict;
	icon_dict["sizes"] = vformat("%dx%d", p_size, p_size);
	icon_dict["type"] = "image/png";
	icon_dict["src"] = icon_name;
	r_arr.push_back(icon_dict);
	return err;
}

Error EditorExportPlatformWeb::_build_pwa(const Ref<EditorExportPreset> &p_preset, const String p_path, const Vector<SharedObject> &p_shared_objects) {
	String proj_name = get_project_setting(p_preset, "application/config/name");
	if (proj_name.is_empty()) {
		proj_name = "Godot Game";
	}

	// Service worker
	const String dir = p_path.get_base_dir();
	const String name = p_path.get_file().get_basename();
	bool extensions = (bool)p_preset->get("variant/extensions_support");
	bool ensure_crossorigin_isolation_headers = (bool)p_preset->get("progressive_web_app/ensure_cross_origin_isolation_headers");
	HashMap<String, String> replaces;
	replaces["___GODOT_VERSION___"] = String::num_int64(OS::get_singleton()->get_unix_time()) + "|" + String::num_int64(OS::get_singleton()->get_ticks_usec());
	replaces["___GODOT_NAME___"] = proj_name.substr(0, 16);
	replaces["___GODOT_OFFLINE_PAGE___"] = name + ".offline.html";
	replaces["___GODOT_ENSURE_CROSSORIGIN_ISOLATION_HEADERS___"] = ensure_crossorigin_isolation_headers ? "true" : "false";

	// Files cached during worker install.
	Array cache_files = {
		name + ".html",
		name + ".js",
		name + ".offline.html"
	};
	if (p_preset->get("html/export_icon")) {
		cache_files.push_back(name + ".icon.png");
		cache_files.push_back(name + ".apple-touch-icon.png");
	}

	cache_files.push_back(name + ".audio.worklet.js");
	cache_files.push_back(name + ".audio.position.worklet.js");
	replaces["___GODOT_CACHE___"] = Variant(cache_files).to_json_string();

	// Heavy files that are cached on demand.
	Array opt_cache_files = {
		name + ".wasm",
	};

	AsyncLoadSetting async_initial_load_mode = (AsyncLoadSetting)(int)p_preset->get("async/initial_load_mode");
	switch (async_initial_load_mode) {
		case ASYNC_LOAD_SETTING_LOAD_EVERYTHING: {
			opt_cache_files.push_back(name + ".pck");
		} break;

		case ASYNC_LOAD_SETTING_MINIMUM_INITIAL_RESOURCES: {
			// TODO: Add AsyncPCK contents to the cache.
		} break;
	}

	if (extensions) {
		opt_cache_files.push_back(name + ".side.wasm");
		for (int i = 0; i < p_shared_objects.size(); i++) {
			opt_cache_files.push_back(p_shared_objects[i].path.get_file());
		}
	}
	replaces["___GODOT_OPT_CACHE___"] = Variant(opt_cache_files).to_json_string();

	const String sw_path = dir.path_join(name + ".service.worker.js");
	Vector<uint8_t> sw;
	{
		Ref<FileAccess> f = FileAccess::open(sw_path, FileAccess::READ);
		if (f.is_null()) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("PWA"), vformat(TTR("Could not read file: \"%s\"."), sw_path));
			return ERR_FILE_CANT_READ;
		}
		sw.resize(f->get_length());
		f->get_buffer(sw.ptrw(), sw.size());
	}
	_replace_strings(replaces, sw);
	Error err = _write_or_error(sw.ptr(), sw.size(), dir.path_join(name + ".service.worker.js"));
	if (err != OK) {
		// Message is supplied by the subroutine method.
		return err;
	}

	// Custom offline page
	const String offline_page = p_preset->get("progressive_web_app/offline_page");
	if (!offline_page.is_empty()) {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		const String offline_dest = dir.path_join(name + ".offline.html");
		err = da->copy(ProjectSettings::get_singleton()->globalize_path(offline_page), offline_dest);
		if (err != OK) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("PWA"), vformat(TTR("Could not read file: \"%s\"."), offline_dest));
			return err;
		}
	}

	// Manifest
	const char *modes[4] = { "fullscreen", "standalone", "minimal-ui", "browser" };
	const char *orientations[3] = { "any", "landscape", "portrait" };
	const int display = CLAMP(int(p_preset->get("progressive_web_app/display")), 0, 4);
	const int orientation = CLAMP(int(p_preset->get("progressive_web_app/orientation")), 0, 3);

	Dictionary manifest;
	manifest["name"] = proj_name;
	manifest["start_url"] = "./" + name + ".html";
	manifest["display"] = String::utf8(modes[display]);
	manifest["orientation"] = String::utf8(orientations[orientation]);
	manifest["background_color"] = "#" + p_preset->get("progressive_web_app/background_color").operator Color().to_html(false);

	Array icons_arr;
	const String icon144_path = p_preset->get("progressive_web_app/icon_144x144");
	err = _add_manifest_icon(p_preset, p_path, icon144_path, 144, icons_arr);
	if (err != OK) {
		// Message is supplied by the subroutine method.
		return err;
	}
	const String icon180_path = p_preset->get("progressive_web_app/icon_180x180");
	err = _add_manifest_icon(p_preset, p_path, icon180_path, 180, icons_arr);
	if (err != OK) {
		// Message is supplied by the subroutine method.
		return err;
	}
	const String icon512_path = p_preset->get("progressive_web_app/icon_512x512");
	err = _add_manifest_icon(p_preset, p_path, icon512_path, 512, icons_arr);
	if (err != OK) {
		// Message is supplied by the subroutine method.
		return err;
	}
	manifest["icons"] = icons_arr;

	CharString cs = Variant(manifest).to_json_string().utf8();
	err = _write_or_error((const uint8_t *)cs.get_data(), cs.length(), dir.path_join(name + ".manifest.json"));
	if (err != OK) {
		// Message is supplied by the subroutine method.
		return err;
	}

	return OK;
}

void EditorExportPlatformWeb::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const {
	if (p_preset->get("vram_texture_compression/for_desktop")) {
		r_features->push_back("s3tc");
		r_features->push_back("bptc");
	}
	if (p_preset->get("vram_texture_compression/for_mobile")) {
		r_features->push_back("etc2");
		r_features->push_back("astc");
	}
	if (p_preset->get("variant/thread_support").operator bool()) {
		r_features->push_back("threads");
	} else {
		r_features->push_back("nothreads");
	}
	if (p_preset->get("variant/extensions_support").operator bool()) {
		r_features->push_back("web_extensions");
	} else {
		r_features->push_back("web_noextensions");
	}
	r_features->push_back("wasm32");
}

void EditorExportPlatformWeb::get_export_options(List<ExportOption> *r_options) const {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "async/initial_load_mode", PROPERTY_HINT_ENUM, "Load Everything,Load Minimum Initial Resources"), 0, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "async/initial_load_forced_files_filters_to_include"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "async/initial_load_forced_files_filters_to_exclude"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "async/initial_load_forced_files", PROPERTY_HINT_ARRAY_TYPE, MAKE_FILE_ARRAY_TYPE_HINT("*")), PackedStringArray()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "variant/extensions_support"), false)); // GDExtension support.
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "variant/thread_support"), false, true)); // Thread support (i.e. run with or without COEP/COOP headers).
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "vram_texture_compression/for_desktop"), true)); // S3TC
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "vram_texture_compression/for_mobile"), false)); // ETC or ETC2, depending on renderer

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "html/export_icon"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "html/custom_html_shell", PROPERTY_HINT_FILE, "*.html"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "html/head_include", PROPERTY_HINT_MULTILINE_TEXT, "monospace,no_wrap"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "html/canvas_resize_policy", PROPERTY_HINT_ENUM, "None,Project,Adaptive"), 2));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "html/focus_canvas_on_start"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "html/experimental_virtual_keyboard"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "progressive_web_app/enabled"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "progressive_web_app/ensure_cross_origin_isolation_headers"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "progressive_web_app/offline_page", PROPERTY_HINT_FILE, "*.html"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "progressive_web_app/display", PROPERTY_HINT_ENUM, "Fullscreen,Standalone,Minimal UI,Browser"), 1));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "progressive_web_app/orientation", PROPERTY_HINT_ENUM, "Any,Landscape,Portrait"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "progressive_web_app/icon_144x144", PROPERTY_HINT_FILE, "*.png,*.webp,*.svg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "progressive_web_app/icon_180x180", PROPERTY_HINT_FILE, "*.png,*.webp,*.svg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "progressive_web_app/icon_512x512", PROPERTY_HINT_FILE, "*.png,*.webp,*.svg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::COLOR, "progressive_web_app/background_color", PROPERTY_HINT_COLOR_NO_ALPHA), Color()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "threads/emscripten_pool_size"), 8));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "threads/godot_pool_size"), 4));
}

bool EditorExportPlatformWeb::get_export_option_visibility(const EditorExportPreset *p_preset, const String &p_option) const {
	if (p_option == "async/initial_load_forced_files" || p_option == "async/initial_load_forced_files_filters_to_include" || p_option == "async/initial_load_forced_files_filters_to_exclude") {
		return (int)p_preset->get("async/initial_load_mode") != ASYNC_LOAD_SETTING_LOAD_EVERYTHING;
	}

	bool advanced_options_enabled = p_preset->are_advanced_options_enabled();
	if (p_option == "custom_template/debug" || p_option == "custom_template/release") {
		return advanced_options_enabled;
	}

	if (p_option == "threads/godot_pool_size" || p_option == "threads/emscripten_pool_size") {
		return p_preset->get("variant/thread_support").operator bool();
	}

	return true;
}

String EditorExportPlatformWeb::get_name() const {
	return "Web";
}

String EditorExportPlatformWeb::get_os_name() const {
	return "Web";
}

Ref<Texture2D> EditorExportPlatformWeb::get_logo() const {
	return logo;
}

bool EditorExportPlatformWeb::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
#ifdef MODULE_MONO_ENABLED
	// Don't check for additional errors, as this particular error cannot be resolved.
	r_error += TTR("Exporting to Web is currently not supported in Godot 4 when using C#/.NET. Use Godot 3 to target Web with C#/Mono instead.") + "\n";
	r_error += TTR("If this project does not use C#, use a non-C# editor build to export the project.") + "\n";
	return false;
#else

	String err;

	if ((int)p_preset->get("async/initial_load_mode") != AsyncLoadSetting::ASYNC_LOAD_SETTING_LOAD_EVERYTHING) {
		if (String(EditorExportPlatformUtils::get_project_setting(p_preset, "application/run/main_scene")).is_empty()) {
			err += TTR("No main scene has been set. The main scene must be set for the web platform in order to preload the minimal files.") + "\n";
		}
	}

	bool valid = false;
	bool extensions = (bool)p_preset->get("variant/extensions_support");
	bool thread_support = (bool)p_preset->get("variant/thread_support");

	// Look for export templates (first official, and if defined custom templates).
	bool dvalid = exists_export_template(_get_template_name(extensions, thread_support, true), &err);
	bool rvalid = exists_export_template(_get_template_name(extensions, thread_support, false), &err);

	if (p_preset->get("custom_template/debug") != "") {
		dvalid = FileAccess::exists(p_preset->get("custom_template/debug"));
		if (!dvalid) {
			err += TTR("Custom debug template not found.") + "\n";
		}
	}
	if (p_preset->get("custom_template/release") != "") {
		rvalid = FileAccess::exists(p_preset->get("custom_template/release"));
		if (!rvalid) {
			err += TTR("Custom release template not found.") + "\n";
		}
	}

	valid = dvalid || rvalid;
	r_missing_templates = !valid;

	if (!err.is_empty()) {
		r_error = err;
	}

	return valid;
#endif // !MODULE_MONO_ENABLED
}

bool EditorExportPlatformWeb::has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const {
	String err;
	bool valid = true;

	// Validate the project configuration.

	if (p_preset->get("vram_texture_compression/for_mobile")) {
		if (!ResourceImporterTextureSettings::should_import_etc2_astc()) {
			valid = false;
		}
	}

	if (!err.is_empty()) {
		r_error = err;
	}

	return valid;
}

List<String> EditorExportPlatformWeb::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	List<String> list;
	list.push_back("html");
	return list;
}

Error EditorExportPlatformWeb::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	const String custom_debug = p_preset->get("custom_template/debug");
	const String custom_release = p_preset->get("custom_template/release");
	const String custom_html = p_preset->get("html/custom_html_shell");
	const bool export_icon = p_preset->get("html/export_icon");
	const bool pwa = p_preset->get("progressive_web_app/enabled");

	String path = p_path;
	if (!path.is_absolute_path()) {
		if (!path.begins_with("res://")) {
			path = "res://" + path;
		}
		path = ProjectSettings::get_singleton()->globalize_path(path);
	}

	const String base_dir = path.get_base_dir() + "/";
	const String base_path = path.get_basename();
	const String base_name = path.get_file().get_basename();

	if (!DirAccess::exists(base_dir)) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Target folder does not exist or is inaccessible: \"%s\""), base_dir));
		return ERR_FILE_BAD_PATH;
	}

	// Find the correct template
	String template_path = p_debug ? custom_debug : custom_release;
	template_path = template_path.strip_edges();
	if (template_path.is_empty()) {
		bool extensions = (bool)p_preset->get("variant/extensions_support");
		bool thread_support = (bool)p_preset->get("variant/thread_support");
		template_path = find_export_template(_get_template_name(extensions, thread_support, p_debug));
	}

	if (!template_path.is_empty() && !FileAccess::exists(template_path)) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Prepare Templates"), vformat(TTR("Template file not found: \"%s\"."), template_path));
		return ERR_FILE_NOT_FOUND;
	}

	Error error;

	// Export pck and shared objects
	Vector<SharedObject> shared_objects;
	String pck_path;

	// Async PCK related.
	Dictionary async_pck_data;
	// Parse generated file sizes (pck and wasm, to help show a meaningful loading bar).
	Dictionary file_sizes;

	AsyncLoadSetting async_initial_load_mode = (AsyncLoadSetting)(int)p_preset->get("async/initial_load_mode");
	switch (async_initial_load_mode) {
		case ASYNC_LOAD_SETTING_LOAD_EVERYTHING: {
			pck_path = base_path + ".pck";

			error = save_pack(p_preset, p_debug, pck_path, &shared_objects);
			if (error != OK) {
				add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), pck_path));
				return error;
			}

			{
				Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
				for (int i = 0; i < shared_objects.size(); i++) {
					String dst = base_dir.path_join(shared_objects[i].path.get_file());
					error = da->copy(shared_objects[i].path, dst);
					if (error != OK) {
						add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), shared_objects[i].path.get_file()));
						return error;
					}
				}
			}

			// Updating file sizes.
			Ref<FileAccess> f = FileAccess::open(pck_path, FileAccess::READ);
			if (f.is_valid()) {
				file_sizes[pck_path.get_file()] = (uint64_t)f->get_length();
			}

		} break;

		case ASYNC_LOAD_SETTING_MINIMUM_INITIAL_RESOURCES: {
			pck_path = base_path + ".asyncpck";

			if (DirAccess::dir_exists_absolute(pck_path)) {
				Ref<DirAccess> pck_path_access = DirAccess::create_for_path(pck_path);
				pck_path_access->change_dir(pck_path);
				pck_path_access->erase_contents_recursive();
				pck_path_access->change_dir("..");
				pck_path_access->remove_absolute(pck_path);
			}

			ExportData export_data;
			export_data.assets_directory = pck_path.path_join("assets");
			export_data.libraries_directory = pck_path.path_join("libraries");
			export_data.pack_data.path = "assets.sparsepck";
			export_data.pack_data.use_sparse_pck = true;
			export_data.preset = p_preset;

			HashSet<String> features_set = export_data.get_features_set();

			error = export_project_files(p_preset, p_debug, &EditorExportPlatformWeb::_rename_and_store_file_in_async_pck, nullptr, &export_data);
			if (error != OK) {
				add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write async pck: \"%s\"."), pck_path));
				return error;
			}

			PackedByteArray encoded_data;
			error = _generate_sparse_pck_metadata(p_preset, export_data.pack_data, encoded_data, true);
			if (error != OK) {
				add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not encode contents of async pck: \"%s\"."), pck_path));
				return error;
			}

			error = EditorExportPlatformUtils::store_file_at_path(export_data.assets_directory.path_join("assets.sparsepck"), encoded_data);
			if (error != OK) {
				add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not store contents of async pck: \"%s\"."), pck_path));
				return error;
			}

			// bool is_encrypted = p_preset->get_enc_pck() && p_preset->get_enc_directory();

			{
				Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
				for (int i = 0; i < shared_objects.size(); i++) {
					String dst = export_data.libraries_directory.path_join(shared_objects[i].path.get_file());
					error = da->copy(shared_objects[i].path, dst);
					if (error != OK) {
						add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), shared_objects[i].path.get_file()));
						return error;
					}
				}
			}

			{
				Dictionary async_pck_data_directories;
				Dictionary async_pck_data_initial_load;

				async_pck_data["directories"] = async_pck_data_directories;
				async_pck_data["initialLoad"] = async_pck_data_initial_load;

				const String PREFIX_ASSETS_DIRECTORY = export_data.assets_directory + "/";

				const String SECTION_REMAP = "remap";
				const String SECTION_KEY_PATH = "path";
				const String SECTION_KEY_PATH_PREFIX = "path.";

				const String PATH_GODOT_DIR = ".godot/";

				HashSet<String> exported_files;
				HashSet<String> internal_files;
				HashSet<String> standalone_files;
				HashSet<String> remap_files;
				HashSet<String> import_files;

				uint64_t total_size = 0;

				Ref<FileAccess> uid_cache = FileAccess::open(export_data.res_to_global(PATH_GODOT_UID_CACHE), FileAccess::READ, &error);
				if (error != OK) {
					return error;
				}

				for (const String &exported_file : export_data.exported_files) {
					String local_exported_file = PREFIX_RES + exported_file.trim_prefix(PREFIX_ASSETS_DIRECTORY).simplify_path();
					exported_files.insert(local_exported_file);
				}

				for (const String &exported_file : exported_files) {
					if (exported_file.begins_with(PATH_GODOT_DIR) || exported_file == PATH_PROJECT_BINARY || exported_file == PATH_ASSETS_SPARSEPCK) {
						internal_files.insert(exported_file);
						continue;
					}

					if (exported_file.ends_with(SUFFIX_REMAP)) {
						remap_files.insert(exported_file);
						continue;
					} else if (exported_file.ends_with(SUFFIX_IMPORT)) {
						import_files.insert(exported_file);
						continue;
					}

					standalone_files.insert(exported_file);
				}

				for (const String &internal_file : internal_files) {
					export_data.add_dependency(internal_file, features_set, uid_cache, &error);
					if (error != OK) {
						return error;
					}
				}

				for (const String &remap_file : remap_files) {
					export_data.add_dependency(remap_file.trim_suffix(SUFFIX_REMAP), features_set, uid_cache, &error);
					if (error != OK) {
						return error;
					}
				}

				for (const String &import_file : import_files) {
					export_data.add_dependency(import_file.trim_suffix(SUFFIX_IMPORT), features_set, uid_cache, &error);
					if (error != OK) {
						return error;
					}
				}

				for (const String &standalone_file : standalone_files) {
					export_data.add_dependency(standalone_file, features_set, uid_cache, &error);
					if (error != OK) {
						return error;
					}
				}

				for (ExportData::ResourceData &dependency : export_data.dependencies) {
					if (dependency.path.begins_with(PREFIX_RES + PATH_GODOT_DIR)) {
						continue;
					}
					error = export_data.save_deps_json(&dependency);
					if (error != OK) {
						return error;
					}
				}

				HashSet<const ExportData::ResourceData *> initial_load_dependencies;
				{
					Vector<String> initial_load_in_filters;
					Vector<String> initial_load_ex_filters;

					Vector<String> initial_load_in_split = String(p_preset->get("async/initial_load_forced_files_filters_to_include")).split(",");
					for (int i = 0; i < initial_load_in_split.size(); i++) {
						String initial_load_in_filter = initial_load_in_split[i].strip_edges();
						if (initial_load_in_filter.is_empty()) {
							continue;
						}
						initial_load_in_filters.push_back(initial_load_in_filter);
					}

					Vector<String> initial_load_ex_split = String(p_preset->get("async/initial_load_forced_files_filters_to_exclude")).split(",");
					for (int i = 0; i < initial_load_ex_split.size(); i++) {
						String initial_load_ex_filter = initial_load_ex_split[i].strip_edges();
						if (initial_load_ex_filter.is_empty()) {
							continue;
						}
						initial_load_ex_filters.push_back(initial_load_ex_filter);
					}

					if (initial_load_in_filters.size() > 0) {
						for (const ExportData::ResourceData &dependency : export_data.dependencies) {
							const String &dependency_path = dependency.path;
							bool add_as_initial_load = false;
							for (const String &in_filter : initial_load_in_filters) {
								if (dependency_path.matchn(in_filter) || dependency_path.trim_prefix(PREFIX_RES).matchn(in_filter)) {
									add_as_initial_load = true;
									break;
								}
							}

							for (const String &ex_filter : initial_load_ex_filters) {
								if (dependency_path.matchn(ex_filter) || dependency_path.trim_prefix(PREFIX_RES).matchn(ex_filter)) {
									add_as_initial_load = false;
									break;
								}
							}

							if (add_as_initial_load) {
								initial_load_dependencies.insert(&dependency);
							}
						}
					}
				}

				HashSet<String> mandatory_initial_load_files = _get_mandatory_initial_load_files(p_preset);
				for (const String &mandatory_initial_load_file : mandatory_initial_load_files) {
					export_data.add_dependency(mandatory_initial_load_file, features_set, uid_cache);
				}
				for (const String &mandatory_initial_load_file : mandatory_initial_load_files) {
					ExportData::ResourceData *mandatory_resource_data = export_data.dependencies_map[mandatory_initial_load_file];
					initial_load_dependencies.insert(mandatory_resource_data);
					LocalVector<const ExportData::ResourceData *> mandatory_resource_data_dependencies;
					mandatory_resource_data->flatten_dependencies(&mandatory_resource_data_dependencies);
					for (const ExportData::ResourceData *mandatory_resource_data_dependency : mandatory_resource_data_dependencies) {
						initial_load_dependencies.insert(mandatory_resource_data_dependency);
					}
				}

				{
					PackedStringArray initial_load_paths;
					HashSet<const ExportData::ResourceData *> initial_load_assets;
					for (const ExportData::ResourceData *dependency : initial_load_dependencies) {
						if (dependency->remap_file.exists || dependency->native_file.exists) {
							initial_load_assets.insert(dependency);
						}
					}

					LocalVector<const ExportData::ResourceData *> initial_load_assets_data;
					for (const ExportData::ResourceData *initial_load_asset : initial_load_assets) {
						initial_load_assets_data.push_back(initial_load_asset);
					}

					_add_resource_data_tree_message(initial_load_assets_data, "Files that will be initially loaded (sorted in alphabetical order):", true, false);
					_add_resource_data_tree_message(initial_load_assets_data, "Files that will be initially loaded (sorted by size):", false, true);

					uint64_t initial_load_assets_size = initial_load_assets_data.size();
					for (uint64_t i = 0; i < initial_load_assets_size; i++) {
						const ExportData::ResourceData *initial_load_asset = initial_load_assets_data[i];

						uint64_t asset_size = 0;
						if (initial_load_asset->remap_file.exists) {
							asset_size += initial_load_asset->remap_file.size;
							asset_size += initial_load_asset->remapped_file.size;
						} else if (initial_load_asset->native_file.exists) {
							asset_size += initial_load_asset->native_file.size;
						} else {
							ERR_FAIL_V(ERR_BUG);
						}

						total_size += asset_size;
					}

					StringBuilder log_entry_builder;
					log_entry_builder.append("If some files seem to be missing from this list, be sure to edit \"async/initial_load_forced_files*\" in the preset settings.\n");
					log_entry_builder.append("For files not in this list, you will need to call `OS.async_pck_install_file()` beforehand.\n");
					log_entry_builder.append("\n");
					log_entry_builder.append(vformat("Total initial load size: %s", String::humanize_size(total_size)));

					add_message(EditorExportPlatformData::EXPORT_MESSAGE_INFO, TTR("Initial load"), log_entry_builder.as_string());
				}

				{
					async_pck_data_directories["assets"] = export_data.assets_directory.trim_prefix(base_dir);
					async_pck_data_directories["libraries"] = export_data.libraries_directory.trim_prefix(base_dir);
				}

				for (const ExportData::ResourceData *dependency : initial_load_dependencies) {
					Dictionary dependency_dict;
					Array dependency_dependencies;
					Array dependency_files;

					dependency_dict["files"] = dependency_files;

					for (const ExportData::ResourceData *dependency_dependency : dependency->dependencies) {
						dependency_dependencies.push_back(dependency_dependency->path);
					}

					if (dependency_dependencies.size() > 0) {
						dependency_dict["dependencies"] = dependency_dependencies;
					}

					if (dependency->native_file.exists) {
						dependency_files.push_back(dependency->native_file.resource_path);
					}
					if (dependency->remap_file.exists) {
						dependency_files.push_back(dependency->remap_file.resource_path);
					}
					if (dependency->remapped_file.exists) {
						dependency_files.push_back(dependency->remapped_file.resource_path);
					}

					async_pck_data_initial_load[dependency->path] = dependency_dict;
				}

				for (ExportData::ResourceData &dependency : export_data.dependencies) {
					if (dependency.native_file.exists) {
						file_sizes[dependency.native_file.absolute_path.trim_prefix(base_dir)] = dependency.native_file.size;
					}
					if (dependency.remap_file.exists) {
						file_sizes[dependency.remap_file.absolute_path.trim_prefix(base_dir)] = dependency.remap_file.size;
					}
					if (dependency.remapped_file.exists) {
						file_sizes[dependency.remapped_file.absolute_path.trim_prefix(base_dir)] = dependency.remapped_file.size;
					}
				}
			}
		} break;

		default: {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR(R"*(Invalid `async/initial_load_mode` value: %s)*"), async_initial_load_mode));
			return ERR_INVALID_PARAMETER;
		} break;
	}

	// Extract templates.
	error = _extract_template(template_path, base_dir, base_name, pwa);
	if (error) {
		// Message is supplied by the subroutine method.
		return error;
	}

	Ref<FileAccess> f = FileAccess::open(base_path + ".wasm", FileAccess::READ);
	if (f.is_valid()) {
		file_sizes[base_name + ".wasm"] = (uint64_t)f->get_length();
	}

	// Read the HTML shell file (custom or from template).
	const String html_path = custom_html.is_empty() ? base_path + ".html" : custom_html;
	Vector<uint8_t> html;
	f = FileAccess::open(html_path, FileAccess::READ);
	if (f.is_null()) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read HTML shell: \"%s\"."), html_path));
		return ERR_FILE_CANT_READ;
	}
	html.resize(f->get_length());
	f->get_buffer(html.ptrw(), html.size());
	f.unref(); // close file.

	// Generate HTML file with replaced strings.
	_fix_html(html, p_preset, base_name, p_debug, p_flags, shared_objects, file_sizes, async_pck_data);
	Error err = _write_or_error(html.ptr(), html.size(), path);
	if (err != OK) {
		// Message is supplied by the subroutine method.
		return err;
	}
	html.resize(0);

	// Export splash (why?)
	Ref<Image> splash = _get_project_splash(p_preset);
	const String splash_png_path = base_path + ".png";
	if (splash->save_png(splash_png_path) != OK) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), splash_png_path));
		return ERR_FILE_CANT_WRITE;
	}

	// Save a favicon that can be accessed without waiting for the project to finish loading.
	// This way, the favicon can be displayed immediately when loading the page.
	if (export_icon) {
		Ref<Image> favicon = _get_project_icon(p_preset);
		const String favicon_png_path = base_path + ".icon.png";
		if (favicon->save_png(favicon_png_path) != OK) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), favicon_png_path));
			return ERR_FILE_CANT_WRITE;
		}
		favicon->resize(180, 180);
		const String apple_icon_png_path = base_path + ".apple-touch-icon.png";
		if (favicon->save_png(apple_icon_png_path) != OK) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), apple_icon_png_path));
			return ERR_FILE_CANT_WRITE;
		}
	}

	// Generate the PWA worker and manifest
	if (pwa) {
		err = _build_pwa(p_preset, path, shared_objects);
		if (err != OK) {
			// Message is supplied by the subroutine method.
			return err;
		}
	}

	return OK;
}

void EditorExportPlatformWeb::_add_resource_data_tree_message(LocalVector<const ExportData::ResourceData *> &p_resource_data_entries, const String &p_context, bool p_sort_with_file_no_case_comparator, bool p_sort_with_size_comparator) {
	if (p_sort_with_file_no_case_comparator) {
		p_resource_data_entries.sort_custom<ExportData::ResourceData::FileNoCaseComparator>();
	}
	if (p_sort_with_size_comparator) {
		p_resource_data_entries.sort_custom<ExportData::ResourceData::SizeComparator>();
	}

	StringBuilder log_entry_builder;
	const String new_line_char = "\n";
	log_entry_builder.append(vformat("%s\n", p_context));

	uint64_t initial_load_assets_size = p_resource_data_entries.size();
	for (uint64_t i = 0; i < initial_load_assets_size; i++) {
		const ExportData::ResourceData *initial_load_asset = p_resource_data_entries[i];

		uint64_t asset_size = 0;
		if (initial_load_asset->remap_file.exists) {
			asset_size += initial_load_asset->remap_file.size;
			asset_size += initial_load_asset->remapped_file.size;
		} else if (initial_load_asset->native_file.exists) {
			asset_size += initial_load_asset->native_file.size;
		} else {
			ERR_FAIL();
		}

		String fork_char = i < initial_load_assets_size - 1
				? U"├"
				: U"└";
		String parent_tree_line = i < initial_load_assets_size - 1
				? U"|"
				: U" ";

		log_entry_builder.append(vformat(UR"*(%s── 📦 "%s" [%s]%s)*", fork_char, initial_load_asset->path, String::humanize_size(asset_size), new_line_char));

		if (initial_load_asset->remap_file.exists) {
			log_entry_builder.append(vformat(UR"*(%s    ├ 📤 "%s" [%s]%s)*", parent_tree_line, initial_load_asset->remap_file.resource_path, String::humanize_size(initial_load_asset->remap_file.size), new_line_char));
			log_entry_builder.append(vformat(UR"*(%s    └ 📤 "%s" [%s]%s)*", parent_tree_line, initial_load_asset->remapped_file.resource_path, String::humanize_size(initial_load_asset->remapped_file.size), new_line_char));
		} else if (initial_load_asset->native_file.exists) {
			log_entry_builder.append(vformat(UR"*(%s    └ 📤 "%s" [%s]%s)*", parent_tree_line, initial_load_asset->native_file.resource_path, String::humanize_size(initial_load_asset->native_file.size), new_line_char));
		}
	}

	log_entry_builder.append("\n====================\n");

	add_message(EditorExportPlatformData::EXPORT_MESSAGE_INFO, TTR("Initial load"), log_entry_builder.as_string());
}

bool EditorExportPlatformWeb::poll_export() {
	Ref<EditorExportPreset> preset;

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		Ref<EditorExportPreset> ep = EditorExport::get_singleton()->get_export_preset(i);
		if (ep->is_runnable() && ep->get_platform() == this) {
			preset = ep;
			break;
		}
	}

	RemoteDebugState prev_remote_debug_state = remote_debug_state;
	remote_debug_state = REMOTE_DEBUG_STATE_UNAVAILABLE;

	if (preset.is_valid()) {
		const bool debug = true;
		// Throwaway variables to pass to `can_export`.
		String err;
		bool missing_templates;

		if (can_export(preset, err, missing_templates, debug)) {
			if (server->is_listening()) {
				remote_debug_state = REMOTE_DEBUG_STATE_SERVING;
			} else {
				remote_debug_state = REMOTE_DEBUG_STATE_AVAILABLE;
			}
		}
	}

	if (remote_debug_state != REMOTE_DEBUG_STATE_SERVING && server->is_listening()) {
		server->stop();
	}

	return remote_debug_state != prev_remote_debug_state;
}

Ref<Texture2D> EditorExportPlatformWeb::get_option_icon(int p_index) const {
	Ref<Texture2D> play_icon = EditorExportPlatform::get_option_icon(p_index);

	switch (remote_debug_state) {
		case REMOTE_DEBUG_STATE_UNAVAILABLE: {
			return nullptr;
		} break;

		case REMOTE_DEBUG_STATE_AVAILABLE: {
			switch (p_index) {
				case 0:
				case 1:
					return play_icon;
				default:
					ERR_FAIL_V(nullptr);
			}
		} break;

		case REMOTE_DEBUG_STATE_SERVING: {
			switch (p_index) {
				case 0:
					return play_icon;
				case 1:
					return restart_icon;
				case 2:
					return stop_icon;
				default:
					ERR_FAIL_V(nullptr);
			}
		} break;
	}

	return nullptr;
}

int EditorExportPlatformWeb::get_options_count() const {
	switch (remote_debug_state) {
		case REMOTE_DEBUG_STATE_UNAVAILABLE: {
			return 0;
		} break;

		case REMOTE_DEBUG_STATE_AVAILABLE: {
			return 2;
		} break;

		case REMOTE_DEBUG_STATE_SERVING: {
			return 3;
		} break;
	}

	return 0;
}

String EditorExportPlatformWeb::get_option_label(int p_index) const {
	String run_in_browser = TTR("Run in Browser");
	String start_http_server = TTR("Start HTTP Server");
	String reexport_project = TTR("Re-export Project");
	String stop_http_server = TTR("Stop HTTP Server");

	switch (remote_debug_state) {
		case REMOTE_DEBUG_STATE_UNAVAILABLE:
			return "";

		case REMOTE_DEBUG_STATE_AVAILABLE: {
			switch (p_index) {
				case 0:
					return run_in_browser;
				case 1:
					return start_http_server;
				default:
					ERR_FAIL_V("");
			}
		} break;

		case REMOTE_DEBUG_STATE_SERVING: {
			switch (p_index) {
				case 0:
					return run_in_browser;
				case 1:
					return reexport_project;
				case 2:
					return stop_http_server;
				default:
					ERR_FAIL_V("");
			}
		} break;
	}

	return "";
}

String EditorExportPlatformWeb::get_option_tooltip(int p_index) const {
	String run_in_browser = TTR("Run exported HTML in the system's default browser.");
	String start_http_server = TTR("Start the HTTP server.");
	String reexport_project = TTR("Export project again to account for updates.");
	String stop_http_server = TTR("Stop the HTTP server.");

	switch (remote_debug_state) {
		case REMOTE_DEBUG_STATE_UNAVAILABLE:
			return "";

		case REMOTE_DEBUG_STATE_AVAILABLE: {
			switch (p_index) {
				case 0:
					return run_in_browser;
				case 1:
					return start_http_server;
				default:
					ERR_FAIL_V("");
			}
		} break;

		case REMOTE_DEBUG_STATE_SERVING: {
			switch (p_index) {
				case 0:
					return run_in_browser;
				case 1:
					return reexport_project;
				case 2:
					return stop_http_server;
				default:
					ERR_FAIL_V("");
			}
		} break;
	}

	return "";
}

Error EditorExportPlatformWeb::run(const Ref<EditorExportPreset> &p_preset, int p_option, BitField<EditorExportPlatform::DebugFlags> p_debug_flags) {
	const uint16_t bind_port = EDITOR_GET("export/web/http_port");
	// Resolve host if needed.
	const String bind_host = EDITOR_GET("export/web/http_host");
	const bool use_tls = EDITOR_GET("export/web/use_tls");

	switch (remote_debug_state) {
		case REMOTE_DEBUG_STATE_UNAVAILABLE: {
			return FAILED;
		} break;

		case REMOTE_DEBUG_STATE_AVAILABLE: {
			switch (p_option) {
				// Run in Browser.
				case 0: {
					Error err = _run_export_project(p_preset, p_debug_flags);
					if (err != OK) {
						return err;
					}
					err = _start_server(bind_host, bind_port, use_tls);
					if (err != OK) {
						return err;
					}
					return _launch_browser(bind_host, bind_port, use_tls);
				} break;

				// Start HTTP Server.
				case 1: {
					Error err = _run_export_project(p_preset, p_debug_flags);
					if (err != OK) {
						return err;
					}
					return _start_server(bind_host, bind_port, use_tls);
				} break;

				default: {
					ERR_FAIL_V_MSG(FAILED, vformat(R"(Invalid option "%s" for the current state.)", p_option));
				}
			}
		} break;

		case REMOTE_DEBUG_STATE_SERVING: {
			switch (p_option) {
				// Run in Browser.
				case 0: {
					Error err = _run_export_project(p_preset, p_debug_flags);
					if (err != OK) {
						return err;
					}
					return _launch_browser(bind_host, bind_port, use_tls);
				} break;

				// Re-export Project.
				case 1: {
					return _run_export_project(p_preset, p_debug_flags);
				} break;

				// Stop HTTP Server.
				case 2: {
					return _stop_server();
				} break;

				default: {
					ERR_FAIL_V_MSG(FAILED, vformat(R"(Invalid option "%s" for the current state.)", p_option));
				}
			}
		} break;
	}

	return FAILED;
}

Error EditorExportPlatformWeb::_run_export_project(const Ref<EditorExportPreset> &p_preset, int p_debug_flags) {
	const String dest = EditorPaths::get_singleton()->get_temp_dir().path_join("web");
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!da->dir_exists(dest)) {
		Error err = da->make_dir_recursive(dest);
		if (err != OK) {
			add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Run"), vformat(TTR("Could not create HTTP server directory: %s."), dest));
			return err;
		}
	}

	const String basepath = dest.path_join("tmp_js_export");
	Error err = export_project(p_preset, true, basepath + ".html", p_debug_flags);
	if (err != OK) {
		// Export generates several files, clean them up on failure.
		DirAccess::remove_file_or_error(basepath + ".html");
		DirAccess::remove_file_or_error(basepath + ".offline.html");
		DirAccess::remove_file_or_error(basepath + ".js");
		DirAccess::remove_file_or_error(basepath + ".audio.worklet.js");
		DirAccess::remove_file_or_error(basepath + ".audio.position.worklet.js");
		DirAccess::remove_file_or_error(basepath + ".service.worker.js");
		DirAccess::remove_file_or_error(basepath + ".asyncpck");
		DirAccess::remove_file_or_error(basepath + ".png");
		DirAccess::remove_file_or_error(basepath + ".side.wasm");
		DirAccess::remove_file_or_error(basepath + ".wasm");
		DirAccess::remove_file_or_error(basepath + ".icon.png");
		DirAccess::remove_file_or_error(basepath + ".apple-touch-icon.png");
	}
	return err;
}

Error EditorExportPlatformWeb::_launch_browser(const String &p_bind_host, const uint16_t p_bind_port, const bool p_use_tls) {
	OS::get_singleton()->shell_open(String((p_use_tls ? "https://" : "http://") + p_bind_host + ":" + itos(p_bind_port) + "/tmp_js_export.html"));
	// FIXME: Find out how to clean up export files after running the successfully
	// exported game. Might not be trivial.
	return OK;
}

Error EditorExportPlatformWeb::_start_server(const String &p_bind_host, const uint16_t p_bind_port, const bool p_use_tls) {
	IPAddress bind_ip;
	if (p_bind_host.is_valid_ip_address()) {
		bind_ip = p_bind_host;
	} else {
		bind_ip = IP::get_singleton()->resolve_hostname(p_bind_host);
	}
	ERR_FAIL_COND_V_MSG(!bind_ip.is_valid(), ERR_INVALID_PARAMETER, "Invalid editor setting 'export/web/http_host': '" + p_bind_host + "'. Try using '127.0.0.1'.");

	const String tls_key = EDITOR_GET("export/web/tls_key");
	const String tls_cert = EDITOR_GET("export/web/tls_certificate");

	// Restart server.
	server->stop();
	Error err = server->listen(p_bind_port, bind_ip, p_use_tls, tls_key, tls_cert);
	if (err != OK) {
		add_message(EditorExportPlatformData::EXPORT_MESSAGE_ERROR, TTR("Run"), vformat(TTR("Error starting HTTP server: %d."), err));
	}
	return err;
}

Error EditorExportPlatformWeb::_stop_server() {
	server->stop();
	return OK;
}

Error EditorExportPlatformWeb::_rename_and_store_file_in_async_pck(const Ref<EditorExportPreset> &p_preset, void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key, uint64_t p_seed, bool p_delta) {
	ExportData *export_data = static_cast<ExportData *>(p_userdata);
	const String simplified_path = EditorExportPlatform::simplify_path(p_path);

	Vector<uint8_t> encoded_data;
	EditorExportPlatformData::SavedData saved_data;
	Error err = EditorExportPlatformUtils::store_temp_file(simplified_path, p_data, p_enc_in_filters, p_enc_ex_filters, p_key, p_seed, p_delta, encoded_data, saved_data);
	if (err != OK) {
		return err;
	}

	const String target_path = export_data->assets_directory.path_join(simplified_path.trim_prefix("res://"));
	export_data->exported_files.insert(target_path);
	err = EditorExportPlatformUtils::store_file_at_path(target_path, encoded_data);

	export_data->pack_data.file_ofs.push_back(saved_data);

	return OK;
}

HashSet<String> EditorExportPlatformWeb::_get_mandatory_initial_load_files(const Ref<EditorExportPreset> &p_preset) {
	HashSet<String> mandatory_initial_load_files;

	{
		// Main scene.
		mandatory_initial_load_files.insert(
				EditorExportPlatformUtils::get_path_from_dependency(
						EditorExportPlatformUtils::get_project_setting(p_preset, "application/run/main_scene")));
	}

	{
		// Translation files.
		PackedStringArray translations = EditorExportPlatformUtils::get_project_setting(p_preset, "internationalization/locale/translations");
		for (const String &translation : translations) {
			mandatory_initial_load_files.insert(EditorExportPlatformUtils::get_path_from_dependency(translation));
		}
	}

	{
		// Autoload files.
		HashMap<StringName, ProjectSettings::AutoloadInfo> autoload_list = ProjectSettings::get_singleton()->get_autoload_list();
		for (const KeyValue<StringName, ProjectSettings::AutoloadInfo> &key_value : autoload_list) {
			mandatory_initial_load_files.insert(
					EditorExportPlatformUtils::get_path_from_dependency(key_value.value.path));
		}
	}

	{
		// Global class files.
		LocalVector<StringName> global_classes;
		ScriptServer::get_global_class_list(global_classes);

		for (const StringName &global_class : global_classes) {
			String global_class_path = ScriptServer::get_global_class_path(global_class);
			mandatory_initial_load_files.insert(
					EditorExportPlatformUtils::get_path_from_dependency(global_class_path));
		}
	}

	{
		// Single files.
		auto _l_add_project_setting_if_file_exists = [&](const String &l_project_setting) -> void {
			const String project_setting_file = ResourceUID::ensure_path(EditorExportPlatformUtils::get_project_setting(p_preset, l_project_setting));
			String path = EditorExportPlatformUtils::get_path_from_dependency(project_setting_file);
			if (FileAccess::exists(path)) {
				mandatory_initial_load_files.insert(path);
			}
		};

		// Icon path.
		_l_add_project_setting_if_file_exists("application/config/icon");
		// Default bus layout path.
		_l_add_project_setting_if_file_exists("audio/buses/default_bus_layout");
		// Certificate bundle override.
		_l_add_project_setting_if_file_exists("network/tls/certificate_bundle_override");
		// Default environment.
		_l_add_project_setting_if_file_exists("rendering/environment/defaults/default_environment");
		// Default XR action map.
		_l_add_project_setting_if_file_exists("xr/openxr/default_action_map");
	}

	{
		// Export-related files.
		mandatory_initial_load_files.insert(PATH_PROJECT_BINARY);
		mandatory_initial_load_files.insert(PATH_ASSETS_SPARSEPCK);
		mandatory_initial_load_files.insert(PATH_GODOT_UID_CACHE);
		mandatory_initial_load_files.insert(PATH_GODOT_GLOBAL_SCRIPT_CLASS_CACHE);
	}

	return mandatory_initial_load_files;
}

Ref<Texture2D> EditorExportPlatformWeb::get_run_icon() const {
	return run_icon;
}

void EditorExportPlatformWeb::initialize() {
	if (!EditorNode::get_singleton()) {
		return;
	}

	server.instantiate();

	Ref<Image> img = memnew(Image);
	const bool upsample = !Math::is_equal_approx(Math::round(EDSCALE), EDSCALE);

	ImageLoaderSVG::create_image_from_string(img, _web_logo_svg, EDSCALE, upsample, false);
	logo = ImageTexture::create_from_image(img);

	ImageLoaderSVG::create_image_from_string(img, _web_run_icon_svg, EDSCALE, upsample, false);
	run_icon = ImageTexture::create_from_image(img);

	Ref<Theme> theme = EditorNode::get_singleton()->get_editor_theme();
	if (theme.is_valid()) {
		stop_icon = theme->get_icon(SNAME("Stop"), EditorStringName(EditorIcons));
		restart_icon = theme->get_icon(SNAME("Reload"), EditorStringName(EditorIcons));
	} else {
		stop_icon.instantiate();
		restart_icon.instantiate();
	}
}
