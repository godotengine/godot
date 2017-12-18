/*************************************************************************/
/*  editor_export.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_export.h"

#include "editor/editor_file_system.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "io/config_file.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "io/zip_io.h"
#include "os/file_access.h"
#include "project_settings.h"
#include "scene/resources/scene_format_text.h"
#include "script_language.h"
#include "thirdparty/misc/md5.h"
#include "version.h"

static int _get_pad(int p_alignment, int p_n) {

	int rest = p_n % p_alignment;
	int pad = 0;
	if (rest > 0) {
		pad = p_alignment - rest;
	};

	return pad;
}

#define PCK_PADDING 16

bool EditorExportPreset::_set(const StringName &p_name, const Variant &p_value) {

	if (values.has(p_name)) {
		values[p_name] = p_value;
		EditorExport::singleton->save_presets();
		return true;
	}

	return false;
}

bool EditorExportPreset::_get(const StringName &p_name, Variant &r_ret) const {

	if (values.has(p_name)) {
		r_ret = values[p_name];
		return true;
	}

	return false;
}

void EditorExportPreset::_get_property_list(List<PropertyInfo> *p_list) const {

	for (const List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {

		if (platform->get_option_visibility(E->get().name, values)) {
			p_list->push_back(E->get());
		}
	}
}

Ref<EditorExportPlatform> EditorExportPreset::get_platform() const {

	return platform;
}

Vector<String> EditorExportPreset::get_files_to_export() const {

	Vector<String> files;
	for (Set<String>::Element *E = selected_files.front(); E; E = E->next()) {
		files.push_back(E->get());
	}
	return files;
}

void EditorExportPreset::set_name(const String &p_name) {
	name = p_name;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_name() const {
	return name;
}

void EditorExportPreset::set_runnable(bool p_enable) {

	runnable = p_enable;
	EditorExport::singleton->save_presets();
}

bool EditorExportPreset::is_runnable() const {

	return runnable;
}

void EditorExportPreset::set_export_filter(ExportFilter p_filter) {

	export_filter = p_filter;
	EditorExport::singleton->save_presets();
}

EditorExportPreset::ExportFilter EditorExportPreset::get_export_filter() const {
	return export_filter;
}

void EditorExportPreset::set_include_filter(const String &p_include) {

	include_filter = p_include;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_include_filter() const {

	return include_filter;
}

void EditorExportPreset::set_exclude_filter(const String &p_exclude) {

	exclude_filter = p_exclude;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_exclude_filter() const {

	return exclude_filter;
}

void EditorExportPreset::add_export_file(const String &p_path) {

	selected_files.insert(p_path);
	EditorExport::singleton->save_presets();
}

void EditorExportPreset::remove_export_file(const String &p_path) {
	selected_files.erase(p_path);
	EditorExport::singleton->save_presets();
}

bool EditorExportPreset::has_export_file(const String &p_path) {

	return selected_files.has(p_path);
}

void EditorExportPreset::add_patch(const String &p_path, int p_at_pos) {

	if (p_at_pos < 0)
		patches.push_back(p_path);
	else
		patches.insert(p_at_pos, p_path);
	EditorExport::singleton->save_presets();
}

void EditorExportPreset::remove_patch(int p_idx) {
	patches.remove(p_idx);
	EditorExport::singleton->save_presets();
}

void EditorExportPreset::set_patch(int p_index, const String &p_path) {
	ERR_FAIL_INDEX(p_index, patches.size());
	patches[p_index] = p_path;
	EditorExport::singleton->save_presets();
}
String EditorExportPreset::get_patch(int p_index) {

	ERR_FAIL_INDEX_V(p_index, patches.size(), String());
	return patches[p_index];
}

Vector<String> EditorExportPreset::get_patches() const {
	return patches;
}

void EditorExportPreset::set_custom_features(const String &p_custom_features) {

	custom_features = p_custom_features;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_custom_features() const {

	return custom_features;
}

EditorExportPreset::EditorExportPreset() {

	export_filter = EXPORT_ALL_RESOURCES;
	runnable = false;
}

///////////////////////////////////

void EditorExportPlatform::gen_debug_flags(Vector<String> &r_flags, int p_flags) {

	String host = EditorSettings::get_singleton()->get("network/debug/remote_host");
	int remote_port = (int)EditorSettings::get_singleton()->get("network/debug/remote_port");

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST)
		host = "localhost";

	if (p_flags & DEBUG_FLAG_DUMB_CLIENT) {
		int port = EditorSettings::get_singleton()->get("filesystem/file_server/port");
		String passwd = EditorSettings::get_singleton()->get("filesystem/file_server/password");
		r_flags.push_back("--remote-fs");
		r_flags.push_back(host + ":" + itos(port));
		if (passwd != "") {
			r_flags.push_back("--remote-fs-password");
			r_flags.push_back(passwd);
		}
	}

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG) {

		r_flags.push_back("--remote-debug");

		r_flags.push_back(host + ":" + String::num(remote_port));

		List<String> breakpoints;
		ScriptEditor::get_singleton()->get_breakpoints(&breakpoints);

		if (breakpoints.size()) {

			r_flags.push_back("--breakpoints");
			String bpoints;
			for (const List<String>::Element *E = breakpoints.front(); E; E = E->next()) {

				bpoints += E->get().replace(" ", "%20");
				if (E->next())
					bpoints += ",";
			}

			r_flags.push_back(bpoints);
		}
	}

	if (p_flags & DEBUG_FLAG_VIEW_COLLISONS) {

		r_flags.push_back("--debug-collisions");
	}

	if (p_flags & DEBUG_FLAG_VIEW_NAVIGATION) {

		r_flags.push_back("--debug-navigation");
	}
}

Error EditorExportPlatform::_save_pack_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {
	if (p_path.ends_with(".so") || p_path.ends_with(".dylib") || p_path.ends_with(".dll"))
		return OK;

	PackData *pd = (PackData *)p_userdata;

	SavedData sd;
	sd.path_utf8 = p_path.utf8();
	sd.ofs = pd->f->get_position();
	sd.size = p_data.size();

	pd->f->store_buffer(p_data.ptr(), p_data.size());
	int pad = _get_pad(PCK_PADDING, sd.size);
	for (int i = 0; i < pad; i++) {
		pd->f->store_8(0);
	}

	{
		MD5_CTX ctx;
		MD5Init(&ctx);
		MD5Update(&ctx, (unsigned char *)p_data.ptr(), p_data.size());
		MD5Final(&ctx);
		sd.md5.resize(16);
		for (int i = 0; i < 16; i++) {
			sd.md5[i] = ctx.digest[i];
		}
	}

	pd->file_ofs.push_back(sd);

	pd->ep->step(TTR("Storing File:") + " " + p_path, 2 + p_file * 100 / p_total, false);

	return OK;
}

Error EditorExportPlatform::_save_zip_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {

	String path = p_path.replace_first("res://", "");

	ZipData *zd = (ZipData *)p_userdata;

	zipFile zip = (zipFile)zd->zip;

	zipOpenNewFileInZip(zip,
			path.utf8().get_data(),
			NULL,
			NULL,
			0,
			NULL,
			0,
			NULL,
			Z_DEFLATED,
			Z_DEFAULT_COMPRESSION);

	zipWriteInFileInZip(zip, p_data.ptr(), p_data.size());
	zipCloseFileInZip(zip);

	zd->ep->step(TTR("Storing File:") + " " + p_path, 2 + p_file * 100 / p_total, false);

	return OK;
}

String EditorExportPlatform::find_export_template(String template_file_name, String *err) const {

	String current_version = itos(VERSION_MAJOR) + "." + itos(VERSION_MINOR) + "-" + VERSION_STATUS + VERSION_MODULE_CONFIG;
	String template_path = EditorSettings::get_singleton()->get_templates_dir().plus_file(current_version).plus_file(template_file_name);

	if (FileAccess::exists(template_path)) {
		return template_path;
	}

	// Not found
	if (err) {
		*err += "No export template found at \"" + template_path + "\".";
	}
	return String();
}

bool EditorExportPlatform::exists_export_template(String template_file_name, String *err) const {
	return find_export_template(template_file_name, err) != "";
}

Ref<EditorExportPreset> EditorExportPlatform::create_preset() {

	Ref<EditorExportPreset> preset;
	preset.instance();
	preset->platform = Ref<EditorExportPlatform>(this);

	List<ExportOption> options;
	get_export_options(&options);

	for (List<ExportOption>::Element *E = options.front(); E; E = E->next()) {

		preset->properties.push_back(E->get().option);
		preset->values[E->get().option.name] = E->get().default_value;
	}

	return preset;
}

void EditorExportPlatform::_export_find_resources(EditorFileSystemDirectory *p_dir, Set<String> &p_paths) {

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_export_find_resources(p_dir->get_subdir(i), p_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		p_paths.insert(p_dir->get_file_path(i));
	}
}

void EditorExportPlatform::_export_find_dependencies(const String &p_path, Set<String> &p_paths) {

	if (p_paths.has(p_path))
		return;

	p_paths.insert(p_path);

	EditorFileSystemDirectory *dir;
	int file_idx;
	dir = EditorFileSystem::get_singleton()->find_file(p_path, &file_idx);
	if (!dir)
		return;

	Vector<String> deps = dir->get_file_deps(file_idx);

	for (int i = 0; i < deps.size(); i++) {

		_export_find_dependencies(deps[i], p_paths);
	}
}

void EditorExportPlatform::_edit_files_with_filter(DirAccess *da, const Vector<String> &p_filters, Set<String> &r_list, bool exclude) {

	da->list_dir_begin();
	String cur_dir = da->get_current_dir().replace("\\", "/");
	if (!cur_dir.ends_with("/"))
		cur_dir += "/";

	Vector<String> dirs;
	String f;
	while ((f = da->get_next()) != "") {
		if (da->current_is_dir())
			dirs.push_back(f);
		else {
			String fullpath = cur_dir + f;
			for (int i = 0; i < p_filters.size(); ++i) {
				if (fullpath.matchn(p_filters[i])) {
					if (!exclude) {
						r_list.insert(fullpath);
					} else {
						r_list.erase(fullpath);
					}
				}
			}
		}
	}

	da->list_dir_end();

	for (int i = 0; i < dirs.size(); ++i) {
		String dir = dirs[i];
		if (dir.begins_with("."))
			continue;
		da->change_dir(dir);
		_edit_files_with_filter(da, p_filters, r_list, exclude);
		da->change_dir("..");
	}
}

void EditorExportPlatform::_edit_filter_list(Set<String> &r_list, const String &p_filter, bool exclude) {

	if (p_filter == "")
		return;
	Vector<String> split = p_filter.split(",");
	Vector<String> filters;
	for (int i = 0; i < split.size(); i++) {
		String f = split[i].strip_edges();
		if (f.empty())
			continue;
		filters.push_back(f);
	}

	DirAccess *da = DirAccess::open("res://");
	ERR_FAIL_NULL(da);
	_edit_files_with_filter(da, filters, r_list, exclude);
	memdelete(da);
}

void EditorExportPlugin::add_file(const String &p_path, const Vector<uint8_t> &p_file, bool p_remap) {

	ExtraFile ef;
	ef.data = p_file;
	ef.path = p_path;
	ef.remap = p_remap;
	extra_files.push_back(ef);
}

void EditorExportPlugin::add_shared_object(const String &p_path, const Vector<String> &tags) {

	shared_objects.push_back(SharedObject(p_path, tags));
}

void EditorExportPlugin::add_ios_framework(const String &p_path) {
	ios_frameworks.push_back(p_path);
}

Vector<String> EditorExportPlugin::get_ios_frameworks() const {
	return ios_frameworks;
}

void EditorExportPlugin::add_ios_plist_content(const String &p_plist_content) {
	ios_plist_content += p_plist_content + "\n";
}

String EditorExportPlugin::get_ios_plist_content() const {
	return ios_plist_content;
}

void EditorExportPlugin::add_ios_linker_flags(const String &p_flags) {
	if (ios_linker_flags.length() > 0) {
		ios_linker_flags += ' ';
	}
	ios_linker_flags += p_flags;
}

String EditorExportPlugin::get_ios_linker_flags() const {
	return ios_linker_flags;
}

void EditorExportPlugin::add_ios_bundle_file(const String &p_path) {
	ios_bundle_files.push_back(p_path);
}

Vector<String> EditorExportPlugin::get_ios_bundle_files() const {
	return ios_bundle_files;
}

void EditorExportPlugin::add_ios_cpp_code(const String &p_code) {
	ios_cpp_code += p_code;
}

String EditorExportPlugin::get_ios_cpp_code() const {
	return ios_cpp_code;
}

void EditorExportPlugin::_export_file_script(const String &p_path, const String &p_type, const PoolVector<String> &p_features) {

	if (get_script_instance()) {
		get_script_instance()->call("_export_file", p_path, p_type, p_features);
	}
}

void EditorExportPlugin::_export_begin_script(const PoolVector<String> &p_features, bool p_debug, const String &p_path, int p_flags) {

	if (get_script_instance()) {
		get_script_instance()->call("_export_begin", p_features, p_debug, p_path, p_flags);
	}
}

void EditorExportPlugin::_export_file(const String &p_path, const String &p_type, const Set<String> &p_features) {
}

void EditorExportPlugin::_export_begin(const Set<String> &p_features, bool p_debug, const String &p_path, int p_flags) {
}

void EditorExportPlugin::skip() {

	skipped = true;
}

void EditorExportPlugin::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_shared_object", "path", "tags"), &EditorExportPlugin::add_shared_object);
	ClassDB::bind_method(D_METHOD("add_file", "path", "file", "remap"), &EditorExportPlugin::add_file);
	ClassDB::bind_method(D_METHOD("add_ios_framework", "path"), &EditorExportPlugin::add_ios_framework);
	ClassDB::bind_method(D_METHOD("add_ios_plist_content", "plist_content"), &EditorExportPlugin::add_ios_plist_content);
	ClassDB::bind_method(D_METHOD("add_ios_linker_flags", "flags"), &EditorExportPlugin::add_ios_linker_flags);
	ClassDB::bind_method(D_METHOD("add_ios_bundle_file", "path"), &EditorExportPlugin::add_ios_bundle_file);
	ClassDB::bind_method(D_METHOD("add_ios_cpp_code", "code"), &EditorExportPlugin::add_ios_cpp_code);
	ClassDB::bind_method(D_METHOD("skip"), &EditorExportPlugin::skip);

	BIND_VMETHOD(MethodInfo("_export_file", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::STRING, "type"), PropertyInfo(Variant::POOL_STRING_ARRAY, "features")));
	BIND_VMETHOD(MethodInfo("_export_begin", PropertyInfo(Variant::POOL_STRING_ARRAY, "features"), PropertyInfo(Variant::BOOL, "is_debug"), PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "flags")));
}

EditorExportPlugin::EditorExportPlugin() {
	skipped = false;
}

EditorExportPlatform::FeatureContainers EditorExportPlatform::get_feature_containers(const Ref<EditorExportPreset> &p_preset) {
	Ref<EditorExportPlatform> platform = p_preset->get_platform();
	List<String> feature_list;
	platform->get_platform_features(&feature_list);
	platform->get_preset_features(p_preset, &feature_list);

	FeatureContainers result;
	for (List<String>::Element *E = feature_list.front(); E; E = E->next()) {
		result.features.insert(E->get());
		result.features_pv.push_back(E->get());
	}
	return result;
}

EditorExportPlatform::ExportNotifier::ExportNotifier(EditorExportPlatform &p_platform, const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	FeatureContainers features = p_platform.get_feature_containers(p_preset);
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	//initial export plugin callback
	for (int i = 0; i < export_plugins.size(); i++) {
		if (export_plugins[i]->get_script_instance()) { //script based
			export_plugins[i]->_export_begin_script(features.features_pv, p_debug, p_path, p_flags);
		} else {
			export_plugins[i]->_export_begin(features.features, p_debug, p_path, p_flags);
		}
	}
}

EditorExportPlatform::ExportNotifier::~ExportNotifier() {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		export_plugins[i]->_export_end();
	}
}

Error EditorExportPlatform::export_project_files(const Ref<EditorExportPreset> &p_preset, EditorExportSaveFunction p_func, void *p_udata, EditorExportSaveSharedObject p_so_func) {
	//figure out paths of files that will be exported
	Set<String> paths;
	Vector<String> path_remaps;

	if (p_preset->get_export_filter() == EditorExportPreset::EXPORT_ALL_RESOURCES) {
		//find stuff
		_export_find_resources(EditorFileSystem::get_singleton()->get_filesystem(), paths);
	} else {
		bool scenes_only = p_preset->get_export_filter() == EditorExportPreset::EXPORT_SELECTED_SCENES;

		Vector<String> files = p_preset->get_files_to_export();
		for (int i = 0; i < files.size(); i++) {
			if (scenes_only && ResourceLoader::get_resource_type(files[i]) != "PackedScene")
				continue;

			_export_find_dependencies(files[i], paths);
		}
	}

	_edit_filter_list(paths, p_preset->get_include_filter(), false);
	_edit_filter_list(paths, p_preset->get_exclude_filter(), true);

	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		if (p_so_func) {
			for (int j = 0; j < export_plugins[i]->shared_objects.size(); j++) {
				p_so_func(p_udata, export_plugins[i]->shared_objects[j]);
			}
		}
		for (int j = 0; j < export_plugins[i]->extra_files.size(); j++) {
			p_func(p_udata, export_plugins[i]->extra_files[j].path, export_plugins[i]->extra_files[j].data, 0, paths.size());
		}

		export_plugins[i]->_clear();
	}

	FeatureContainers feature_containers = get_feature_containers(p_preset);
	Set<String> &features = feature_containers.features;
	PoolVector<String> &features_pv = feature_containers.features_pv;

	//store everything in the export medium
	int idx = 0;
	int total = paths.size();

	for (Set<String>::Element *E = paths.front(); E; E = E->next()) {

		String path = E->get();
		String type = ResourceLoader::get_resource_type(path);

		if (FileAccess::exists(path + ".import")) {
			//file is imported, replace by what it imports
			Ref<ConfigFile> config;
			config.instance();
			Error err = config->load(path + ".import");
			if (err != OK) {
				ERR_PRINTS("Could not parse: '" + path + "', not exported.");
				continue;
			}

			List<String> remaps;
			config->get_section_keys("remap", &remaps);

			for (List<String>::Element *F = remaps.front(); F; F = F->next()) {

				String remap = F->get();
				if (remap == "path") {
					String remapped_path = config->get_value("remap", remap);
					Vector<uint8_t> array = FileAccess::get_file_as_array(remapped_path);
					p_func(p_udata, remapped_path, array, idx, total);
				} else if (remap.begins_with("path.")) {
					String feature = remap.get_slice(".", 1);
					if (features.has(feature)) {
						String remapped_path = config->get_value("remap", remap);
						Vector<uint8_t> array = FileAccess::get_file_as_array(remapped_path);
						p_func(p_udata, remapped_path, array, idx, total);
					}
				}
			}

			//also save the .import file
			Vector<uint8_t> array = FileAccess::get_file_as_array(path + ".import");
			p_func(p_udata, path + ".import", array, idx, total);

		} else {

			bool do_export = true;
			for (int i = 0; i < export_plugins.size(); i++) {
				if (export_plugins[i]->get_script_instance()) { //script based
					export_plugins[i]->_export_file_script(path, type, features_pv);
				} else {
					export_plugins[i]->_export_file(path, type, features);
				}
				if (p_so_func) {
					for (int j = 0; j < export_plugins[i]->shared_objects.size(); j++) {
						p_so_func(p_udata, export_plugins[i]->shared_objects[j]);
					}
				}

				for (int j = 0; j < export_plugins[i]->extra_files.size(); j++) {
					p_func(p_udata, export_plugins[i]->extra_files[j].path, export_plugins[i]->extra_files[j].data, idx, total);
					if (export_plugins[i]->extra_files[j].remap) {
						do_export = false; //if remap, do not
						path_remaps.push_back(path);
						path_remaps.push_back(export_plugins[i]->extra_files[j].path);
					}
				}

				if (export_plugins[i]->skipped) {
					do_export = false;
				}
				export_plugins[i]->_clear();

				if (!do_export)
					break; //apologies, not exporting
			}
			//just store it as it comes
			if (do_export) {
				Vector<uint8_t> array = FileAccess::get_file_as_array(path);
				p_func(p_udata, path, array, idx, total);
			}
		}

		idx++;
	}

	//save config!

	Vector<String> custom_list;

	if (p_preset->get_custom_features() != String()) {

		Vector<String> tmp_custom_list = p_preset->get_custom_features().split(",");

		for (int i = 0; i < tmp_custom_list.size(); i++) {
			String f = tmp_custom_list[i].strip_edges();
			if (f != String()) {
				custom_list.push_back(f);
			}
		}
	}

	ProjectSettings::CustomMap custom_map;
	if (path_remaps.size()) {
		if (1) { //new remap mode, use always as it's friendlier with multiple .pck exports
			for (int i = 0; i < path_remaps.size(); i += 2) {
				String from = path_remaps[i];
				String to = path_remaps[i + 1];
				String remap_file = "[remap]\n\npath=\"" + to.c_escape() + "\"\n";
				CharString utf8 = remap_file.utf8();
				Vector<uint8_t> new_file;
				new_file.resize(utf8.length());
				for (int j = 0; j < utf8.length(); j++) {
					new_file[j] = utf8[j];
				}

				p_func(p_udata, from + ".remap", new_file, idx, total);
			}
		} else {
			//old remap mode, will still work, but it's unused because it's not multiple pck export friendly
			custom_map["path_remap/remapped_paths"] = path_remaps;
		}
	}

	// Store icon and splash images directly, they need to bypass the import system and be loaded as images
	String icon = ProjectSettings::get_singleton()->get("application/config/icon");
	String splash = ProjectSettings::get_singleton()->get("application/boot_splash/image");
	if (icon != String() && FileAccess::exists(icon)) {
		Vector<uint8_t> array = FileAccess::get_file_as_array(icon);
		p_func(p_udata, icon, array, idx, total);
	}
	if (splash != String() && FileAccess::exists(splash)) {
		Vector<uint8_t> array = FileAccess::get_file_as_array(splash);
		p_func(p_udata, splash, array, idx, total);
	}

	String config_file = "project.binary";
	String engine_cfb = EditorSettings::get_singleton()->get_cache_dir().plus_file("tmp" + config_file);
	ProjectSettings::get_singleton()->save_custom(engine_cfb, custom_map, custom_list);
	Vector<uint8_t> data = FileAccess::get_file_as_array(engine_cfb);

	p_func(p_udata, "res://" + config_file, data, idx, total);

	return OK;
}

Error EditorExportPlatform::_add_shared_object(void *p_userdata, const SharedObject &p_so) {
	PackData *pack_data = (PackData *)p_userdata;
	if (pack_data->so_files) {
		pack_data->so_files->push_back(p_so);
	}

	return OK;
}

Error EditorExportPlatform::save_pack(const Ref<EditorExportPreset> &p_preset, const String &p_path, Vector<SharedObject> *p_so_files) {

	EditorProgress ep("savepack", TTR("Packing"), 102);

	String tmppath = EditorSettings::get_singleton()->get_cache_dir().plus_file("packtmp");
	FileAccess *ftmp = FileAccess::open(tmppath, FileAccess::WRITE);
	ERR_FAIL_COND_V(!ftmp, ERR_CANT_CREATE)

	PackData pd;
	pd.ep = &ep;
	pd.f = ftmp;
	pd.so_files = p_so_files;

	Error err = export_project_files(p_preset, _save_pack_file, &pd, _add_shared_object);

	memdelete(ftmp); //close tmp file

	if (err)
		return err;

	pd.file_ofs.sort(); //do sort, so we can do binary search later

	FileAccess *f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V(!f, ERR_CANT_CREATE)
	f->store_32(0x43504447); //GDPK
	f->store_32(1); //pack version
	f->store_32(VERSION_MAJOR);
	f->store_32(VERSION_MINOR);
	f->store_32(0); //hmph
	for (int i = 0; i < 16; i++) {
		//reserved
		f->store_32(0);
	}

	f->store_32(pd.file_ofs.size()); //amount of files

	size_t header_size = f->get_position();

	//precalculate header size

	for (int i = 0; i < pd.file_ofs.size(); i++) {
		header_size += 4; // size of path string (32 bits is enough)
		uint32_t string_len = pd.file_ofs[i].path_utf8.length();
		header_size += string_len + _get_pad(4, string_len); ///size of path string
		header_size += 8; // offset to file _with_ header size included
		header_size += 8; // size of file
		header_size += 16; // md5
	}

	size_t header_padding = _get_pad(PCK_PADDING, header_size);

	for (int i = 0; i < pd.file_ofs.size(); i++) {

		uint32_t string_len = pd.file_ofs[i].path_utf8.length();
		uint32_t pad = _get_pad(4, string_len);
		;
		f->store_32(string_len + pad);
		f->store_buffer((const uint8_t *)pd.file_ofs[i].path_utf8.get_data(), string_len);
		for (uint32_t j = 0; j < pad; j++) {
			f->store_8(0);
		}

		f->store_64(pd.file_ofs[i].ofs + header_padding + header_size);
		f->store_64(pd.file_ofs[i].size); // pay attention here, this is where file is
		f->store_buffer(pd.file_ofs[i].md5.ptr(), 16); //also save md5 for file
	}

	for (uint32_t j = 0; j < header_padding; j++) {
		f->store_8(0);
	}

	//save the rest of the data

	ftmp = FileAccess::open(tmppath, FileAccess::READ);
	if (!ftmp) {
		memdelete(f);
		ERR_FAIL_COND_V(!ftmp, ERR_CANT_CREATE)
	}

	const int bufsize = 16384;
	uint8_t buf[bufsize];

	while (true) {

		int got = ftmp->get_buffer(buf, bufsize);
		if (got <= 0)
			break;
		f->store_buffer(buf, got);
	}

	memdelete(ftmp);

	f->store_32(0x43504447); //GDPK
	memdelete(f);

	return OK;
}

Error EditorExportPlatform::save_zip(const Ref<EditorExportPreset> &p_preset, const String &p_path) {

	EditorProgress ep("savezip", TTR("Packing"), 102);

	//FileAccess *tmp = FileAccess::open(tmppath,FileAccess::WRITE);

	FileAccess *src_f;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);
	zipFile zip = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, NULL, &io);

	ZipData zd;
	zd.ep = &ep;
	zd.zip = zip;

	Error err = export_project_files(p_preset, _save_zip_file, &zd);
	if (err != OK)
		ERR_PRINT("Failed to export project files");

	zipClose(zip, NULL);

	return OK;
}

void EditorExportPlatform::gen_export_flags(Vector<String> &r_flags, int p_flags) {

	String host = EditorSettings::get_singleton()->get("network/debug/remote_host");
	int remote_port = (int)EditorSettings::get_singleton()->get("network/debug/remote_port");

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST)
		host = "localhost";

	if (p_flags & DEBUG_FLAG_DUMB_CLIENT) {
		int port = EditorSettings::get_singleton()->get("filesystem/file_server/port");
		String passwd = EditorSettings::get_singleton()->get("filesystem/file_server/password");
		r_flags.push_back("--remote-fs");
		r_flags.push_back(host + ":" + itos(port));
		if (passwd != "") {
			r_flags.push_back("--remote-fs-password");
			r_flags.push_back(passwd);
		}
	}

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG) {

		r_flags.push_back("--remote-debug");

		r_flags.push_back(host + ":" + String::num(remote_port));

		List<String> breakpoints;
		ScriptEditor::get_singleton()->get_breakpoints(&breakpoints);

		if (breakpoints.size()) {

			r_flags.push_back("--breakpoints");
			String bpoints;
			for (const List<String>::Element *E = breakpoints.front(); E; E = E->next()) {

				bpoints += E->get().replace(" ", "%20");
				if (E->next())
					bpoints += ",";
			}

			r_flags.push_back(bpoints);
		}
	}

	if (p_flags & DEBUG_FLAG_VIEW_COLLISONS) {

		r_flags.push_back("--debug-collisions");
	}

	if (p_flags & DEBUG_FLAG_VIEW_NAVIGATION) {

		r_flags.push_back("--debug-navigation");
	}
}
EditorExportPlatform::EditorExportPlatform() {
}

////

EditorExport *EditorExport::singleton = NULL;

void EditorExport::_save() {

	Ref<ConfigFile> config;
	config.instance();
	for (int i = 0; i < export_presets.size(); i++) {

		Ref<EditorExportPreset> preset = export_presets[i];
		String section = "preset." + itos(i);

		config->set_value(section, "name", preset->get_name());
		config->set_value(section, "platform", preset->get_platform()->get_name());
		config->set_value(section, "runnable", preset->is_runnable());
		config->set_value(section, "custom_features", preset->get_custom_features());
		bool save_files = false;
		switch (preset->get_export_filter()) {
			case EditorExportPreset::EXPORT_ALL_RESOURCES: {
				config->set_value(section, "export_filter", "all_resources");
			} break;
			case EditorExportPreset::EXPORT_SELECTED_SCENES: {
				config->set_value(section, "export_filter", "scenes");
				save_files = true;
			} break;
			case EditorExportPreset::EXPORT_SELECTED_RESOURCES: {
				config->set_value(section, "export_filter", "resources");
				save_files = true;
			} break;
		}

		if (save_files) {
			Vector<String> export_files = preset->get_files_to_export();
			config->set_value(section, "export_files", export_files);
		}
		config->set_value(section, "include_filter", preset->get_include_filter());
		config->set_value(section, "exclude_filter", preset->get_exclude_filter());
		config->set_value(section, "patch_list", preset->get_patches());

		String option_section = "preset." + itos(i) + ".options";

		for (const List<PropertyInfo>::Element *E = preset->get_properties().front(); E; E = E->next()) {
			config->set_value(option_section, E->get().name, preset->get(E->get().name));
		}
	}

	config->save("res://export_presets.cfg");
}

void EditorExport::save_presets() {

	if (block_save)
		return;
	save_timer->start();
}

void EditorExport::_bind_methods() {

	ClassDB::bind_method("_save", &EditorExport::_save);
}

void EditorExport::add_export_platform(const Ref<EditorExportPlatform> &p_platform) {

	export_platforms.push_back(p_platform);
}

int EditorExport::get_export_platform_count() {

	return export_platforms.size();
}

Ref<EditorExportPlatform> EditorExport::get_export_platform(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, export_platforms.size(), Ref<EditorExportPlatform>());

	return export_platforms[p_idx];
}

void EditorExport::add_export_preset(const Ref<EditorExportPreset> &p_preset, int p_at_pos) {

	if (p_at_pos < 0)
		export_presets.push_back(p_preset);
	else
		export_presets.insert(p_at_pos, p_preset);
}

int EditorExport::get_export_preset_count() const {

	return export_presets.size();
}

Ref<EditorExportPreset> EditorExport::get_export_preset(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, export_presets.size(), Ref<EditorExportPreset>());
	return export_presets[p_idx];
}

void EditorExport::remove_export_preset(int p_idx) {

	export_presets.remove(p_idx);
}

void EditorExport::add_export_plugin(const Ref<EditorExportPlugin> &p_plugin) {

	if (export_plugins.find(p_plugin) == -1) {
		export_plugins.push_back(p_plugin);
	}
}

void EditorExport::remove_export_plugin(const Ref<EditorExportPlugin> &p_plugin) {

	export_plugins.erase(p_plugin);
}

Vector<Ref<EditorExportPlugin> > EditorExport::get_export_plugins() {

	return export_plugins;
}

void EditorExport::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		load_config();
	}
}

void EditorExport::load_config() {

	Ref<ConfigFile> config;
	config.instance();
	Error err = config->load("res://export_presets.cfg");
	if (err != OK)
		return;

	block_save = true;

	int index = 0;
	while (true) {

		String section = "preset." + itos(index);
		if (!config->has_section(section))
			break;

		String platform = config->get_value(section, "platform");

		Ref<EditorExportPreset> preset;

		for (int i = 0; i < export_platforms.size(); i++) {
			if (export_platforms[i]->get_name() == platform) {
				preset = export_platforms[i]->create_preset();
				break;
			}
		}

		if (!preset.is_valid()) {
			index++;
			ERR_CONTINUE(!preset.is_valid());
		}

		preset->set_name(config->get_value(section, "name"));
		preset->set_runnable(config->get_value(section, "runnable"));

		if (config->has_section_key(section, "custom_features")) {
			preset->set_custom_features(config->get_value(section, "custom_features"));
		}

		String export_filter = config->get_value(section, "export_filter");

		bool get_files = false;

		if (export_filter == "all_resources") {
			preset->set_export_filter(EditorExportPreset::EXPORT_ALL_RESOURCES);
		} else if (export_filter == "scenes") {
			preset->set_export_filter(EditorExportPreset::EXPORT_SELECTED_SCENES);
			get_files = true;
		} else if (export_filter == "resources") {
			preset->set_export_filter(EditorExportPreset::EXPORT_SELECTED_RESOURCES);
			get_files = true;
		}

		if (get_files) {

			Vector<String> files = config->get_value(section, "export_files");

			for (int i = 0; i < files.size(); i++) {
				preset->add_export_file(files[i]);
			}
		}

		preset->set_include_filter(config->get_value(section, "include_filter"));
		preset->set_exclude_filter(config->get_value(section, "exclude_filter"));

		Vector<String> patch_list = config->get_value(section, "patch_list");

		for (int i = 0; i < patch_list.size(); i++) {
			preset->add_patch(patch_list[i]);
		}

		String option_section = "preset." + itos(index) + ".options";

		List<String> options;

		config->get_section_keys(option_section, &options);

		for (List<String>::Element *E = options.front(); E; E = E->next()) {

			Variant value = config->get_value(option_section, E->get());

			preset->set(E->get(), value);
		}

		add_export_preset(preset);
		index++;
	}

	block_save = false;
}

bool EditorExport::poll_export_platforms() {

	bool changed = false;
	for (int i = 0; i < export_platforms.size(); i++) {
		if (export_platforms[i]->poll_devices()) {
			changed = true;
		}
	}

	return changed;
}

EditorExport::EditorExport() {

	save_timer = memnew(Timer);
	add_child(save_timer);
	save_timer->set_wait_time(0.8);
	save_timer->set_one_shot(true);
	save_timer->connect("timeout", this, "_save");
	block_save = false;

	singleton = this;
}

EditorExport::~EditorExport() {
}

//////////

void EditorExportPlatformPC::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

	if (p_preset->get("texture_format/s3tc")) {
		r_features->push_back("s3tc");
	}
	if (p_preset->get("texture_format/etc")) {
		r_features->push_back("etc");
	}
	if (p_preset->get("texture_format/etc2")) {
		r_features->push_back("etc2");
	}

	if (p_preset->get("binary_format/64_bits")) {
		r_features->push_back("64");
	} else {
		r_features->push_back("32");
	}
}

void EditorExportPlatformPC::get_export_options(List<ExportOption> *r_options) {

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "binary_format/64_bits"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE), ""));
}

String EditorExportPlatformPC::get_name() const {

	return name;
}

String EditorExportPlatformPC::get_os_name() const {

	return os_name;
}
Ref<Texture> EditorExportPlatformPC::get_logo() const {

	return logo;
}

bool EditorExportPlatformPC::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

	String err;
	bool valid = true;

	if (use64 && (!exists_export_template(debug_file_64, &err) || !exists_export_template(release_file_64, &err))) {
		valid = false;
	}

	if (!use64 && (!exists_export_template(debug_file_32, &err) || !exists_export_template(release_file_32, &err))) {
		valid = false;
	}

	String custom_debug_binary = p_preset->get("custom_template/debug");
	String custom_release_binary = p_preset->get("custom_template/release");

	if (custom_debug_binary == "" && custom_release_binary == "") {
		if (!err.empty())
			r_error = err;
		r_missing_templates = !valid;
		return valid;
	}

	bool dvalid = true;
	bool rvalid = true;

	if (!FileAccess::exists(custom_debug_binary)) {
		dvalid = false;
		err = "Custom debug binary not found.\n";
	}

	if (!FileAccess::exists(custom_release_binary)) {
		rvalid = false;
		err += "Custom release binary not found.\n";
	}

	if (dvalid || rvalid)
		valid = true;
	else
		valid = false;

	if (!err.empty())
		r_error = err;
	return valid;
}

String EditorExportPlatformPC::get_binary_extension(const Ref<EditorExportPreset> &p_preset) const {
	for (Map<String, String>::Element *E = extensions.front(); E; E = E->next()) {
		if (p_preset->get(E->key())) {
			return extensions[E->key()];
		}
	}

	if (extensions.has("default")) {
		return extensions["default"];
	}

	return "";
}

Error EditorExportPlatformPC::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String custom_debug = p_preset->get("custom_template/debug");
	String custom_release = p_preset->get("custom_template/release");

	String template_path = p_debug ? custom_debug : custom_release;

	template_path = template_path.strip_edges();

	if (template_path == String()) {

		if (p_preset->get("binary_format/64_bits")) {
			if (p_debug) {
				template_path = find_export_template(debug_file_64);
			} else {
				template_path = find_export_template(release_file_64);
			}
		} else {
			if (p_debug) {
				template_path = find_export_template(debug_file_32);
			} else {
				template_path = find_export_template(release_file_32);
			}
		}
	}

	if (template_path != String() && !FileAccess::exists(template_path)) {
		EditorNode::get_singleton()->show_warning(TTR("Template file not found:\n") + template_path);
		return ERR_FILE_NOT_FOUND;
	}

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = da->copy(template_path, p_path, get_chmod_flags());
	memdelete(da);

	if (err != OK) {
		return err;
	}

	String pck_path = p_path.get_basename() + ".pck";

	return save_pack(p_preset, pck_path);
}

void EditorExportPlatformPC::set_extension(const String &p_extension, const String &p_feature_key) {
	extensions[p_feature_key] = p_extension;
}

void EditorExportPlatformPC::set_name(const String &p_name) {
	name = p_name;
}

void EditorExportPlatformPC::set_os_name(const String &p_name) {
	os_name = p_name;
}

void EditorExportPlatformPC::set_logo(const Ref<Texture> &p_logo) {
	logo = p_logo;
}

void EditorExportPlatformPC::set_release_64(const String &p_file) {

	release_file_64 = p_file;
}

void EditorExportPlatformPC::set_release_32(const String &p_file) {

	release_file_32 = p_file;
}
void EditorExportPlatformPC::set_debug_64(const String &p_file) {

	debug_file_64 = p_file;
}
void EditorExportPlatformPC::set_debug_32(const String &p_file) {

	debug_file_32 = p_file;
}

void EditorExportPlatformPC::add_platform_feature(const String &p_feature) {

	extra_features.insert(p_feature);
}

void EditorExportPlatformPC::get_platform_features(List<String> *r_features) {
	r_features->push_back("pc"); //all pcs support "pc"
	r_features->push_back("s3tc"); //all pcs support "s3tc" compression
	r_features->push_back(get_os_name()); //OS name is a feature
	for (Set<String>::Element *E = extra_features.front(); E; E = E->next()) {
		r_features->push_back(E->get());
	}
}

int EditorExportPlatformPC::get_chmod_flags() const {

	return chmod_flags;
}

void EditorExportPlatformPC::set_chmod_flags(int p_flags) {

	chmod_flags = p_flags;
}

EditorExportPlatformPC::EditorExportPlatformPC() {

	chmod_flags = -1;
}

///////////////////////

void EditorExportTextSceneToBinaryPlugin::_export_file(const String &p_path, const String &p_type, const Set<String> &p_features) {

	String extension = p_path.get_extension().to_lower();
	if (extension != "tres" && extension != "tscn") {
		return;
	}

	print_line("exporting " + p_path);

	bool convert = GLOBAL_GET("editor/convert_text_resources_to_binary_on_export");
	if (!convert)
		return;
	String tmp_path = EditorSettings::get_singleton()->get_cache_dir().plus_file("file.res");
	Error err = ResourceFormatLoaderText::convert_file_to_binary(p_path, tmp_path);
	ERR_FAIL_COND(err != OK);
	Vector<uint8_t> data = FileAccess::get_file_as_array(tmp_path);
	ERR_FAIL_COND(data.size() == 0);
	add_file(p_path + ".converted.res", data, true);
}

EditorExportTextSceneToBinaryPlugin::EditorExportTextSceneToBinaryPlugin() {

	GLOBAL_DEF("editor/convert_text_resources_to_binary_on_export", false);
}
