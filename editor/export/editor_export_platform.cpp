/**************************************************************************/
/*  editor_export_platform.cpp                                            */
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

#include "editor_export_platform.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/extension/gdextension.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/file_access_pack.h" // PACK_HEADER_MAGIC, PACK_FORMAT_VERSION
#include "core/io/zip_io.h"
#include "core/version.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/export/editor_export.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor_export_plugin.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/packed_scene.h"

static int _get_pad(int p_alignment, int p_n) {
	int rest = p_n % p_alignment;
	int pad = 0;
	if (rest > 0) {
		pad = p_alignment - rest;
	};

	return pad;
}

#define PCK_PADDING 16

bool EditorExportPlatform::fill_log_messages(RichTextLabel *p_log, Error p_err) {
	bool has_messages = false;

	int msg_count = get_message_count();

	p_log->add_text(TTR("Project export for platform:") + " ");
	p_log->add_image(get_logo(), 16 * EDSCALE, 16 * EDSCALE, Color(1.0, 1.0, 1.0), INLINE_ALIGNMENT_CENTER);
	p_log->add_text(" ");
	p_log->add_text(get_name());
	p_log->add_text(" - ");
	if (p_err == OK) {
		if (get_worst_message_type() >= EditorExportPlatform::EXPORT_MESSAGE_WARNING) {
			p_log->add_image(p_log->get_editor_theme_icon(SNAME("StatusWarning")), 16 * EDSCALE, 16 * EDSCALE, Color(1.0, 1.0, 1.0), INLINE_ALIGNMENT_CENTER);
			p_log->add_text(" ");
			p_log->add_text(TTR("Completed with warnings."));
			has_messages = true;
		} else {
			p_log->add_image(p_log->get_editor_theme_icon(SNAME("StatusSuccess")), 16 * EDSCALE, 16 * EDSCALE, Color(1.0, 1.0, 1.0), INLINE_ALIGNMENT_CENTER);
			p_log->add_text(" ");
			p_log->add_text(TTR("Completed successfully."));
			if (msg_count > 0) {
				has_messages = true;
			}
		}
	} else {
		p_log->add_image(p_log->get_editor_theme_icon(SNAME("StatusError")), 16 * EDSCALE, 16 * EDSCALE, Color(1.0, 1.0, 1.0), INLINE_ALIGNMENT_CENTER);
		p_log->add_text(" ");
		p_log->add_text(TTR("Failed."));
		has_messages = true;
	}
	p_log->add_newline();

	if (msg_count) {
		p_log->push_table(2);
		p_log->set_table_column_expand(0, false);
		p_log->set_table_column_expand(1, true);
		for (int m = 0; m < msg_count; m++) {
			EditorExportPlatform::ExportMessage msg = get_message(m);
			Color color = p_log->get_theme_color(SNAME("font_color"), SNAME("Label"));
			Ref<Texture> icon;

			switch (msg.msg_type) {
				case EditorExportPlatform::EXPORT_MESSAGE_INFO: {
					color = p_log->get_theme_color(SNAME("font_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.6);
				} break;
				case EditorExportPlatform::EXPORT_MESSAGE_WARNING: {
					icon = p_log->get_editor_theme_icon(SNAME("Warning"));
					color = p_log->get_theme_color(SNAME("warning_color"), EditorStringName(Editor));
				} break;
				case EditorExportPlatform::EXPORT_MESSAGE_ERROR: {
					icon = p_log->get_editor_theme_icon(SNAME("Error"));
					color = p_log->get_theme_color(SNAME("error_color"), EditorStringName(Editor));
				} break;
				default:
					break;
			}

			p_log->push_cell();
			p_log->add_text("\t");
			if (icon.is_valid()) {
				p_log->add_image(icon);
			}
			p_log->pop();

			p_log->push_cell();
			p_log->push_color(color);
			p_log->add_text(vformat("[%s]: %s", msg.category, msg.text));
			p_log->pop();
			p_log->pop();
		}
		p_log->pop();
		p_log->add_newline();
	}
	p_log->add_newline();
	return has_messages;
}

void EditorExportPlatform::gen_debug_flags(Vector<String> &r_flags, int p_flags) {
	String host = EDITOR_GET("network/debug/remote_host");
	int remote_port = (int)EDITOR_GET("network/debug/remote_port");

	if (EditorSettings::get_singleton()->has_setting("export/android/use_wifi_for_remote_debug") && EDITOR_GET("export/android/use_wifi_for_remote_debug")) {
		host = EDITOR_GET("export/android/wifi_remote_debug_host");
	} else if (p_flags & DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST) {
		host = "localhost";
	}

	if (p_flags & DEBUG_FLAG_DUMB_CLIENT) {
		int port = EDITOR_GET("filesystem/file_server/port");
		String passwd = EDITOR_GET("filesystem/file_server/password");
		r_flags.push_back("--remote-fs");
		r_flags.push_back(host + ":" + itos(port));
		if (!passwd.is_empty()) {
			r_flags.push_back("--remote-fs-password");
			r_flags.push_back(passwd);
		}
	}

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG) {
		r_flags.push_back("--remote-debug");

		r_flags.push_back(get_debug_protocol() + host + ":" + String::num(remote_port));

		List<String> breakpoints;
		ScriptEditor::get_singleton()->get_breakpoints(&breakpoints);

		if (breakpoints.size()) {
			r_flags.push_back("--breakpoints");
			String bpoints;
			for (const List<String>::Element *E = breakpoints.front(); E; E = E->next()) {
				bpoints += E->get().replace(" ", "%20");
				if (E->next()) {
					bpoints += ",";
				}
			}

			r_flags.push_back(bpoints);
		}
	}

	if (p_flags & DEBUG_FLAG_VIEW_COLLISIONS) {
		r_flags.push_back("--debug-collisions");
	}

	if (p_flags & DEBUG_FLAG_VIEW_NAVIGATION) {
		r_flags.push_back("--debug-navigation");
	}
}

Error EditorExportPlatform::_save_pack_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key) {
	ERR_FAIL_COND_V_MSG(p_total < 1, ERR_PARAMETER_RANGE_ERROR, "Must select at least one file to export.");

	PackData *pd = (PackData *)p_userdata;

	SavedData sd;
	sd.path_utf8 = p_path.utf8();
	sd.ofs = pd->f->get_position();
	sd.size = p_data.size();
	sd.encrypted = false;

	for (int i = 0; i < p_enc_in_filters.size(); ++i) {
		if (p_path.matchn(p_enc_in_filters[i]) || p_path.replace("res://", "").matchn(p_enc_in_filters[i])) {
			sd.encrypted = true;
			break;
		}
	}

	for (int i = 0; i < p_enc_ex_filters.size(); ++i) {
		if (p_path.matchn(p_enc_ex_filters[i]) || p_path.replace("res://", "").matchn(p_enc_ex_filters[i])) {
			sd.encrypted = false;
			break;
		}
	}

	Ref<FileAccessEncrypted> fae;
	Ref<FileAccess> ftmp = pd->f;

	if (sd.encrypted) {
		fae.instantiate();
		ERR_FAIL_COND_V(fae.is_null(), ERR_SKIP);

		Error err = fae->open_and_parse(ftmp, p_key, FileAccessEncrypted::MODE_WRITE_AES256, false);
		ERR_FAIL_COND_V(err != OK, ERR_SKIP);
		ftmp = fae;
	}

	// Store file content.
	ftmp->store_buffer(p_data.ptr(), p_data.size());

	if (fae.is_valid()) {
		ftmp.unref();
		fae.unref();
	}

	int pad = _get_pad(PCK_PADDING, pd->f->get_position());
	for (int i = 0; i < pad; i++) {
		pd->f->store_8(0);
	}

	// Store MD5 of original file.
	{
		unsigned char hash[16];
		CryptoCore::md5(p_data.ptr(), p_data.size(), hash);
		sd.md5.resize(16);
		for (int i = 0; i < 16; i++) {
			sd.md5.write[i] = hash[i];
		}
	}

	pd->file_ofs.push_back(sd);

	// TRANSLATORS: This is an editor progress label describing the storing of a file.
	if (pd->ep->step(vformat(TTR("Storing File: %s"), p_path), 2 + p_file * 100 / p_total, false)) {
		return ERR_SKIP;
	}

	return OK;
}

Error EditorExportPlatform::_save_zip_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key) {
	ERR_FAIL_COND_V_MSG(p_total < 1, ERR_PARAMETER_RANGE_ERROR, "Must select at least one file to export.");

	String path = p_path.replace_first("res://", "");

	ZipData *zd = (ZipData *)p_userdata;

	zipFile zip = (zipFile)zd->zip;

	zipOpenNewFileInZip(zip,
			path.utf8().get_data(),
			nullptr,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			Z_DEFLATED,
			Z_DEFAULT_COMPRESSION);

	zipWriteInFileInZip(zip, p_data.ptr(), p_data.size());
	zipCloseFileInZip(zip);

	if (zd->ep->step(TTR("Storing File:") + " " + p_path, 2 + p_file * 100 / p_total, false)) {
		return ERR_SKIP;
	}

	return OK;
}

Ref<ImageTexture> EditorExportPlatform::get_option_icon(int p_index) const {
	Ref<Theme> theme = EditorNode::get_singleton()->get_editor_theme();
	ERR_FAIL_COND_V(theme.is_null(), Ref<ImageTexture>());
	if (EditorNode::get_singleton()->get_main_screen_control()->is_layout_rtl()) {
		return theme->get_icon(SNAME("PlayBackwards"), EditorStringName(EditorIcons));
	} else {
		return theme->get_icon(SNAME("Play"), EditorStringName(EditorIcons));
	}
}

String EditorExportPlatform::find_export_template(String template_file_name, String *err) const {
	String current_version = VERSION_FULL_CONFIG;
	String template_path = EditorPaths::get_singleton()->get_export_templates_dir().path_join(current_version).path_join(template_file_name);

	if (FileAccess::exists(template_path)) {
		return template_path;
	}

	// Not found
	if (err) {
		*err += TTR("No export template found at the expected path:") + "\n" + template_path + "\n";
	}
	return String();
}

bool EditorExportPlatform::exists_export_template(String template_file_name, String *err) const {
	return find_export_template(template_file_name, err) != "";
}

Ref<EditorExportPreset> EditorExportPlatform::create_preset() {
	Ref<EditorExportPreset> preset;
	preset.instantiate();
	preset->platform = Ref<EditorExportPlatform>(this);

	List<ExportOption> options;
	get_export_options(&options);

	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		export_plugins.write[i]->_get_export_options(Ref<EditorExportPlatform>(this), &options);
	}

	for (const ExportOption &E : options) {
		StringName option_name = E.option.name;
		preset->properties[option_name] = E.option;
		preset->values[option_name] = E.default_value;
		preset->update_visibility[option_name] = E.update_visibility;
	}

	return preset;
}

void EditorExportPlatform::_export_find_resources(EditorFileSystemDirectory *p_dir, HashSet<String> &p_paths) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_export_find_resources(p_dir->get_subdir(i), p_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "TextFile") {
			continue;
		}
		p_paths.insert(p_dir->get_file_path(i));
	}
}

void EditorExportPlatform::_export_find_customized_resources(const Ref<EditorExportPreset> &p_preset, EditorFileSystemDirectory *p_dir, EditorExportPreset::FileExportMode p_mode, HashSet<String> &p_paths) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		EditorFileSystemDirectory *subdir = p_dir->get_subdir(i);
		_export_find_customized_resources(p_preset, subdir, p_preset->get_file_export_mode(subdir->get_path(), p_mode), p_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "TextFile") {
			continue;
		}
		String path = p_dir->get_file_path(i);
		EditorExportPreset::FileExportMode file_mode = p_preset->get_file_export_mode(path, p_mode);
		if (file_mode != EditorExportPreset::MODE_FILE_REMOVE) {
			p_paths.insert(path);
		}
	}
}

void EditorExportPlatform::_export_find_dependencies(const String &p_path, HashSet<String> &p_paths) {
	if (p_paths.has(p_path)) {
		return;
	}

	p_paths.insert(p_path);

	EditorFileSystemDirectory *dir;
	int file_idx;
	dir = EditorFileSystem::get_singleton()->find_file(p_path, &file_idx);
	if (!dir) {
		return;
	}

	Vector<String> deps = dir->get_file_deps(file_idx);

	for (int i = 0; i < deps.size(); i++) {
		_export_find_dependencies(deps[i], p_paths);
	}
}

void EditorExportPlatform::_edit_files_with_filter(Ref<DirAccess> &da, const Vector<String> &p_filters, HashSet<String> &r_list, bool exclude) {
	da->list_dir_begin();
	String cur_dir = da->get_current_dir().replace("\\", "/");
	if (!cur_dir.ends_with("/")) {
		cur_dir += "/";
	}
	String cur_dir_no_prefix = cur_dir.replace("res://", "");

	Vector<String> dirs;
	String f = da->get_next();
	while (!f.is_empty()) {
		if (da->current_is_dir()) {
			dirs.push_back(f);
		} else {
			String fullpath = cur_dir + f;
			// Test also against path without res:// so that filters like `file.txt` can work.
			String fullpath_no_prefix = cur_dir_no_prefix + f;
			for (int i = 0; i < p_filters.size(); ++i) {
				if (fullpath.matchn(p_filters[i]) || fullpath_no_prefix.matchn(p_filters[i])) {
					if (!exclude) {
						r_list.insert(fullpath);
					} else {
						r_list.erase(fullpath);
					}
				}
			}
		}
		f = da->get_next();
	}

	da->list_dir_end();

	for (int i = 0; i < dirs.size(); ++i) {
		String dir = dirs[i];
		if (dir.begins_with(".")) {
			continue;
		}

		if (EditorFileSystem::_should_skip_directory(cur_dir + dir)) {
			continue;
		}

		da->change_dir(dir);
		_edit_files_with_filter(da, p_filters, r_list, exclude);
		da->change_dir("..");
	}
}

void EditorExportPlatform::_edit_filter_list(HashSet<String> &r_list, const String &p_filter, bool exclude) {
	if (p_filter.is_empty()) {
		return;
	}
	Vector<String> split = p_filter.split(",");
	Vector<String> filters;
	for (int i = 0; i < split.size(); i++) {
		String f = split[i].strip_edges();
		if (f.is_empty()) {
			continue;
		}
		filters.push_back(f);
	}

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	ERR_FAIL_COND(da.is_null());
	_edit_files_with_filter(da, filters, r_list, exclude);
}

HashSet<String> EditorExportPlatform::get_features(const Ref<EditorExportPreset> &p_preset, bool p_debug) const {
	Ref<EditorExportPlatform> platform = p_preset->get_platform();
	List<String> feature_list;
	platform->get_platform_features(&feature_list);
	platform->get_preset_features(p_preset, &feature_list);

	HashSet<String> result;
	for (const String &E : feature_list) {
		result.insert(E);
	}

	result.insert("template");
	if (p_debug) {
		result.insert("debug");
		result.insert("template_debug");
	} else {
		result.insert("release");
		result.insert("template_release");
	}

	if (!p_preset->get_custom_features().is_empty()) {
		Vector<String> tmp_custom_list = p_preset->get_custom_features().split(",");

		for (int i = 0; i < tmp_custom_list.size(); i++) {
			String f = tmp_custom_list[i].strip_edges();
			if (!f.is_empty()) {
				result.insert(f);
			}
		}
	}

	return result;
}

EditorExportPlatform::ExportNotifier::ExportNotifier(EditorExportPlatform &p_platform, const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	HashSet<String> features = p_platform.get_features(p_preset, p_debug);
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	//initial export plugin callback
	for (int i = 0; i < export_plugins.size(); i++) {
		export_plugins.write[i]->set_export_preset(p_preset);
		if (GDVIRTUAL_IS_OVERRIDDEN_PTR(export_plugins[i], _export_begin)) {
			PackedStringArray features_psa;
			for (const String &feature : features) {
				features_psa.push_back(feature);
			}
			export_plugins.write[i]->_export_begin_script(features_psa, p_debug, p_path, p_flags);
		} else {
			export_plugins.write[i]->_export_begin(features, p_debug, p_path, p_flags);
		}
	}
}

EditorExportPlatform::ExportNotifier::~ExportNotifier() {
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		if (GDVIRTUAL_IS_OVERRIDDEN_PTR(export_plugins[i], _export_end)) {
			export_plugins.write[i]->_export_end_script();
		}
		export_plugins.write[i]->_export_end();
		export_plugins.write[i]->set_export_preset(Ref<EditorExportPlugin>());
	}
}

bool EditorExportPlatform::_export_customize_dictionary(Dictionary &dict, LocalVector<Ref<EditorExportPlugin>> &customize_resources_plugins) {
	bool changed = false;

	List<Variant> keys;
	dict.get_key_list(&keys);
	for (const Variant &K : keys) {
		Variant v = dict[K];
		switch (v.get_type()) {
			case Variant::OBJECT: {
				Ref<Resource> res = v;
				if (res.is_valid()) {
					for (Ref<EditorExportPlugin> &plugin : customize_resources_plugins) {
						Ref<Resource> new_res = plugin->_customize_resource(res, "");
						if (new_res.is_valid()) {
							changed = true;
							if (new_res != res) {
								dict[K] = new_res;
								res = new_res;
							}
							break;
						}
					}

					// If it was not replaced, go through and see if there is something to replace.
					if (res.is_valid() && !res->get_path().is_resource_file() && _export_customize_object(res.ptr(), customize_resources_plugins), true) {
						changed = true;
					}
				}

			} break;
			case Variant::DICTIONARY: {
				Dictionary d = v;
				if (_export_customize_dictionary(d, customize_resources_plugins)) {
					changed = true;
				}
			} break;
			case Variant::ARRAY: {
				Array a = v;
				if (_export_customize_array(a, customize_resources_plugins)) {
					changed = true;
				}
			} break;
			default: {
			}
		}
	}
	return changed;
}

bool EditorExportPlatform::_export_customize_array(Array &arr, LocalVector<Ref<EditorExportPlugin>> &customize_resources_plugins) {
	bool changed = false;

	for (int i = 0; i < arr.size(); i++) {
		Variant v = arr.get(i);
		switch (v.get_type()) {
			case Variant::OBJECT: {
				Ref<Resource> res = v;
				if (res.is_valid()) {
					for (Ref<EditorExportPlugin> &plugin : customize_resources_plugins) {
						Ref<Resource> new_res = plugin->_customize_resource(res, "");
						if (new_res.is_valid()) {
							changed = true;
							if (new_res != res) {
								arr.set(i, new_res);
								res = new_res;
							}
							break;
						}
					}

					// If it was not replaced, go through and see if there is something to replace.
					if (res.is_valid() && !res->get_path().is_resource_file() && _export_customize_object(res.ptr(), customize_resources_plugins), true) {
						changed = true;
					}
				}
			} break;
			case Variant::DICTIONARY: {
				Dictionary d = v;
				if (_export_customize_dictionary(d, customize_resources_plugins)) {
					changed = true;
				}
			} break;
			case Variant::ARRAY: {
				Array a = v;
				if (_export_customize_array(a, customize_resources_plugins)) {
					changed = true;
				}
			} break;
			default: {
			}
		}
	}
	return changed;
}

bool EditorExportPlatform::_export_customize_object(Object *p_object, LocalVector<Ref<EditorExportPlugin>> &customize_resources_plugins) {
	bool changed = false;

	List<PropertyInfo> props;
	p_object->get_property_list(&props);
	for (const PropertyInfo &E : props) {
		switch (E.type) {
			case Variant::OBJECT: {
				Ref<Resource> res = p_object->get(E.name);
				if (res.is_valid()) {
					for (Ref<EditorExportPlugin> &plugin : customize_resources_plugins) {
						Ref<Resource> new_res = plugin->_customize_resource(res, "");
						if (new_res.is_valid()) {
							changed = true;
							if (new_res != res) {
								p_object->set(E.name, new_res);
								res = new_res;
							}
							break;
						}
					}

					// If it was not replaced, go through and see if there is something to replace.
					if (res.is_valid() && !res->get_path().is_resource_file() && _export_customize_object(res.ptr(), customize_resources_plugins), true) {
						changed = true;
					}
				}

			} break;
			case Variant::DICTIONARY: {
				Dictionary d = p_object->get(E.name);
				if (_export_customize_dictionary(d, customize_resources_plugins)) {
					// May have been generated, so set back just in case
					p_object->set(E.name, d);
					changed = true;
				}
			} break;
			case Variant::ARRAY: {
				Array a = p_object->get(E.name);
				if (_export_customize_array(a, customize_resources_plugins)) {
					// May have been generated, so set back just in case
					p_object->set(E.name, a);
					changed = true;
				}
			} break;
			default: {
			}
		}
	}
	return changed;
}

bool EditorExportPlatform::_is_editable_ancestor(Node *p_root, Node *p_node) {
	while (p_node != nullptr && p_node != p_root) {
		if (p_root->is_editable_instance(p_node)) {
			return true;
		}
		p_node = p_node->get_owner();
	}
	return false;
}

bool EditorExportPlatform::_export_customize_scene_resources(Node *p_root, Node *p_node, LocalVector<Ref<EditorExportPlugin>> &customize_resources_plugins) {
	bool changed = false;

	if (p_root == p_node || p_node->get_owner() == p_root || _is_editable_ancestor(p_root, p_node)) {
		if (_export_customize_object(p_node, customize_resources_plugins)) {
			changed = true;
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		if (_export_customize_scene_resources(p_root, p_node->get_child(i), customize_resources_plugins)) {
			changed = true;
		}
	}

	return changed;
}

String EditorExportPlatform::_export_customize(const String &p_path, LocalVector<Ref<EditorExportPlugin>> &customize_resources_plugins, LocalVector<Ref<EditorExportPlugin>> &customize_scenes_plugins, HashMap<String, FileExportCache> &export_cache, const String &export_base_path, bool p_force_save) {
	if (!p_force_save && customize_resources_plugins.is_empty() && customize_scenes_plugins.is_empty()) {
		return p_path; // do none
	}

	// Check if a cache exists
	if (export_cache.has(p_path)) {
		FileExportCache &fec = export_cache[p_path];

		if (fec.saved_path.is_empty() || FileAccess::exists(fec.saved_path)) {
			// Destination file exists (was not erased) or not needed

			uint64_t mod_time = FileAccess::get_modified_time(p_path);
			if (fec.source_modified_time == mod_time) {
				// Cached (modified time matches).
				fec.used = true;
				return fec.saved_path.is_empty() ? p_path : fec.saved_path;
			}

			String md5 = FileAccess::get_md5(p_path);
			if (FileAccess::exists(p_path + ".import")) {
				// Also consider the import file in the string
				md5 += FileAccess::get_md5(p_path + ".import");
			}
			if (fec.source_md5 == md5) {
				// Cached (md5 matches).
				fec.source_modified_time = mod_time;
				fec.used = true;
				return fec.saved_path.is_empty() ? p_path : fec.saved_path;
			}
		}
	}

	FileExportCache fec;
	fec.used = true;
	fec.source_modified_time = FileAccess::get_modified_time(p_path);

	String md5 = FileAccess::get_md5(p_path);
	if (FileAccess::exists(p_path + ".import")) {
		// Also consider the import file in the string
		md5 += FileAccess::get_md5(p_path + ".import");
	}

	fec.source_md5 = md5;

	// Check if it should convert

	String type = ResourceLoader::get_resource_type(p_path);

	bool modified = false;

	String save_path;

	if (type == "PackedScene") { // Its a scene.
		Ref<PackedScene> ps = ResourceLoader::load(p_path, "PackedScene", ResourceFormatLoader::CACHE_MODE_IGNORE);
		ERR_FAIL_COND_V(ps.is_null(), p_path);
		Node *node = ps->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE); // Make sure the child scene root gets the correct inheritance chain.
		ERR_FAIL_NULL_V(node, p_path);
		if (!customize_scenes_plugins.is_empty()) {
			for (Ref<EditorExportPlugin> &plugin : customize_scenes_plugins) {
				Node *customized = plugin->_customize_scene(node, p_path);
				if (customized != nullptr) {
					node = customized;
					modified = true;
				}
			}
		}
		if (!customize_resources_plugins.is_empty()) {
			if (_export_customize_scene_resources(node, node, customize_resources_plugins)) {
				modified = true;
			}
		}

		if (modified || p_force_save) {
			// If modified, save it again. This is also used for TSCN -> SCN conversion on export.

			String base_file = p_path.get_file().get_basename() + ".scn"; // use SCN for saving (binary) and repack (If conversting, TSCN PackedScene representation is inefficient, so repacking is also desired).
			save_path = export_base_path.path_join("export-" + p_path.md5_text() + "-" + base_file);

			Ref<PackedScene> s;
			s.instantiate();
			s->pack(node);
			Error err = ResourceSaver::save(s, save_path);
			ERR_FAIL_COND_V_MSG(err != OK, p_path, "Unable to save export scene file to: " + save_path);
		}
	} else {
		Ref<Resource> res = ResourceLoader::load(p_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE);
		ERR_FAIL_COND_V(res.is_null(), p_path);

		if (!customize_resources_plugins.is_empty()) {
			for (Ref<EditorExportPlugin> &plugin : customize_resources_plugins) {
				Ref<Resource> new_res = plugin->_customize_resource(res, p_path);
				if (new_res.is_valid()) {
					modified = true;
					if (new_res != res) {
						res = new_res;
					}
					break;
				}
			}

			if (_export_customize_object(res.ptr(), customize_resources_plugins)) {
				modified = true;
			}
		}

		if (modified || p_force_save) {
			// If modified, save it again. This is also used for TRES -> RES conversion on export.

			String base_file = p_path.get_file().get_basename() + ".res"; // use RES for saving (binary)
			save_path = export_base_path.path_join("export-" + p_path.md5_text() + "-" + base_file);

			Error err = ResourceSaver::save(res, save_path);
			ERR_FAIL_COND_V_MSG(err != OK, p_path, "Unable to save export resource file to: " + save_path);
		}
	}

	fec.saved_path = save_path;

	export_cache[p_path] = fec;

	return save_path.is_empty() ? p_path : save_path;
}

String EditorExportPlatform::_get_script_encryption_key(const Ref<EditorExportPreset> &p_preset) const {
	const String from_env = OS::get_singleton()->get_environment(ENV_SCRIPT_ENCRYPTION_KEY);
	if (!from_env.is_empty()) {
		return from_env.to_lower();
	}
	return p_preset->get_script_encryption_key().to_lower();
}

Vector<String> EditorExportPlatform::get_forced_export_files() {
	Vector<String> files;

	files.push_back(ProjectSettings::get_singleton()->get_global_class_list_path());

	String icon = GLOBAL_GET("application/config/icon");
	String splash = GLOBAL_GET("application/boot_splash/image");
	if (!icon.is_empty() && FileAccess::exists(icon)) {
		files.push_back(icon);
	}
	if (!splash.is_empty() && FileAccess::exists(splash) && icon != splash) {
		files.push_back(splash);
	}
	String resource_cache_file = ResourceUID::get_cache_file();
	if (FileAccess::exists(resource_cache_file)) {
		files.push_back(resource_cache_file);
	}

	String extension_list_config_file = GDExtension::get_extension_list_config_file();
	if (FileAccess::exists(extension_list_config_file)) {
		files.push_back(extension_list_config_file);
	}

	// Store text server data if it is supported.
	if (TS->has_feature(TextServer::FEATURE_USE_SUPPORT_DATA)) {
		bool use_data = GLOBAL_GET("internationalization/locale/include_text_server_data");
		if (use_data) {
			// Try using user provided data file.
			if (!TS->get_support_data_filename().is_empty()) {
				String ts_data = "res://" + TS->get_support_data_filename();
				if (FileAccess::exists(ts_data)) {
					files.push_back(ts_data);
				} else {
					// Use default text server data.
					String abs_path = ProjectSettings::get_singleton()->globalize_path(ts_data);
					ERR_FAIL_COND_V(!TS->save_support_data(abs_path), files);
					if (FileAccess::exists(abs_path)) {
						files.push_back(ts_data);
						// Remove the file later.
						MessageQueue::get_singleton()->push_callable(callable_mp_static(DirAccess::remove_absolute), abs_path);
					}
				}
			}
		}
	}

	return files;
}

Error EditorExportPlatform::export_project_files(const Ref<EditorExportPreset> &p_preset, bool p_debug, EditorExportSaveFunction p_func, void *p_udata, EditorExportSaveSharedObject p_so_func) {
	//figure out paths of files that will be exported
	HashSet<String> paths;
	Vector<String> path_remaps;

	if (p_preset->get_export_filter() == EditorExportPreset::EXPORT_ALL_RESOURCES) {
		//find stuff
		_export_find_resources(EditorFileSystem::get_singleton()->get_filesystem(), paths);
	} else if (p_preset->get_export_filter() == EditorExportPreset::EXCLUDE_SELECTED_RESOURCES) {
		_export_find_resources(EditorFileSystem::get_singleton()->get_filesystem(), paths);
		Vector<String> files = p_preset->get_files_to_export();
		for (int i = 0; i < files.size(); i++) {
			paths.erase(files[i]);
		}
	} else if (p_preset->get_export_filter() == EditorExportPreset::EXPORT_CUSTOMIZED) {
		_export_find_customized_resources(p_preset, EditorFileSystem::get_singleton()->get_filesystem(), p_preset->get_file_export_mode("res://"), paths);
	} else {
		bool scenes_only = p_preset->get_export_filter() == EditorExportPreset::EXPORT_SELECTED_SCENES;

		Vector<String> files = p_preset->get_files_to_export();
		for (int i = 0; i < files.size(); i++) {
			if (scenes_only && ResourceLoader::get_resource_type(files[i]) != "PackedScene") {
				continue;
			}

			_export_find_dependencies(files[i], paths);
		}

		// Add autoload resources and their dependencies
		List<PropertyInfo> props;
		ProjectSettings::get_singleton()->get_property_list(&props);

		for (const PropertyInfo &pi : props) {
			if (!pi.name.begins_with("autoload/")) {
				continue;
			}

			String autoload_path = GLOBAL_GET(pi.name);

			if (autoload_path.begins_with("*")) {
				autoload_path = autoload_path.substr(1);
			}

			_export_find_dependencies(autoload_path, paths);
		}
	}

	//add native icons to non-resource include list
	_edit_filter_list(paths, String("*.icns"), false);
	_edit_filter_list(paths, String("*.ico"), false);

	_edit_filter_list(paths, p_preset->get_include_filter(), false);
	_edit_filter_list(paths, p_preset->get_exclude_filter(), true);

	// Ignore import files, since these are automatically added to the jar later with the resources
	_edit_filter_list(paths, String("*.import"), true);

	// Get encryption filters.
	bool enc_pck = p_preset->get_enc_pck();
	Vector<String> enc_in_filters;
	Vector<String> enc_ex_filters;
	Vector<uint8_t> key;

	if (enc_pck) {
		Vector<String> enc_in_split = p_preset->get_enc_in_filter().split(",");
		for (int i = 0; i < enc_in_split.size(); i++) {
			String f = enc_in_split[i].strip_edges();
			if (f.is_empty()) {
				continue;
			}
			enc_in_filters.push_back(f);
		}

		Vector<String> enc_ex_split = p_preset->get_enc_ex_filter().split(",");
		for (int i = 0; i < enc_ex_split.size(); i++) {
			String f = enc_ex_split[i].strip_edges();
			if (f.is_empty()) {
				continue;
			}
			enc_ex_filters.push_back(f);
		}

		// Get encryption key.
		String script_key = _get_script_encryption_key(p_preset);
		key.resize(32);
		if (script_key.length() == 64) {
			for (int i = 0; i < 32; i++) {
				int v = 0;
				if (i * 2 < script_key.length()) {
					char32_t ct = script_key[i * 2];
					if (is_digit(ct)) {
						ct = ct - '0';
					} else if (ct >= 'a' && ct <= 'f') {
						ct = 10 + ct - 'a';
					}
					v |= ct << 4;
				}

				if (i * 2 + 1 < script_key.length()) {
					char32_t ct = script_key[i * 2 + 1];
					if (is_digit(ct)) {
						ct = ct - '0';
					} else if (ct >= 'a' && ct <= 'f') {
						ct = 10 + ct - 'a';
					}
					v |= ct;
				}
				key.write[i] = v;
			}
		}
	}

	Error err = OK;
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();

	struct SortByName {
		bool operator()(const Ref<EditorExportPlugin> &left, const Ref<EditorExportPlugin> &right) const {
			return left->get_name() < right->get_name();
		}
	};

	// Always sort by name, to so if for some reason they are re-arranged, it still works.
	export_plugins.sort_custom<SortByName>();

	for (int i = 0; i < export_plugins.size(); i++) {
		if (p_so_func) {
			for (int j = 0; j < export_plugins[i]->shared_objects.size(); j++) {
				err = p_so_func(p_udata, export_plugins[i]->shared_objects[j]);
				if (err != OK) {
					return err;
				}
			}
		}
		for (int j = 0; j < export_plugins[i]->extra_files.size(); j++) {
			err = p_func(p_udata, export_plugins[i]->extra_files[j].path, export_plugins[i]->extra_files[j].data, 0, paths.size(), enc_in_filters, enc_ex_filters, key);
			if (err != OK) {
				return err;
			}
		}

		export_plugins.write[i]->_clear();
	}

	HashSet<String> features = get_features(p_preset, p_debug);
	PackedStringArray features_psa;
	for (const String &feature : features) {
		features_psa.push_back(feature);
	}

	// Check if custom processing is needed
	uint32_t custom_resources_hash = HASH_MURMUR3_SEED;
	uint32_t custom_scene_hash = HASH_MURMUR3_SEED;

	LocalVector<Ref<EditorExportPlugin>> customize_resources_plugins;
	LocalVector<Ref<EditorExportPlugin>> customize_scenes_plugins;

	for (int i = 0; i < export_plugins.size(); i++) {
		if (export_plugins.write[i]->_begin_customize_resources(Ref<EditorExportPlatform>(this), features_psa)) {
			customize_resources_plugins.push_back(export_plugins[i]);

			custom_resources_hash = hash_murmur3_one_64(export_plugins[i]->get_name().hash64(), custom_resources_hash);
			uint64_t hash = export_plugins[i]->_get_customization_configuration_hash();
			custom_resources_hash = hash_murmur3_one_64(hash, custom_resources_hash);
		}
		if (export_plugins.write[i]->_begin_customize_scenes(Ref<EditorExportPlatform>(this), features_psa)) {
			customize_scenes_plugins.push_back(export_plugins[i]);

			custom_resources_hash = hash_murmur3_one_64(export_plugins[i]->get_name().hash64(), custom_resources_hash);
			uint64_t hash = export_plugins[i]->_get_customization_configuration_hash();
			custom_scene_hash = hash_murmur3_one_64(hash, custom_scene_hash);
		}
	}

	HashMap<String, FileExportCache> export_cache;
	String export_base_path = ProjectSettings::get_singleton()->get_project_data_path().path_join("exported/") + itos(custom_resources_hash);

	bool convert_text_to_binary = GLOBAL_GET("editor/export/convert_text_resources_to_binary");

	if (convert_text_to_binary || !customize_resources_plugins.is_empty() || !customize_scenes_plugins.is_empty()) {
		// See if we have something to open
		Ref<FileAccess> f = FileAccess::open(export_base_path.path_join("file_cache"), FileAccess::READ);
		if (f.is_valid()) {
			String l = f->get_line();
			while (l != String()) {
				Vector<String> fields = l.split("::");
				if (fields.size() == 4) {
					FileExportCache fec;
					String path = fields[0];
					fec.source_md5 = fields[1].strip_edges();
					fec.source_modified_time = fields[2].strip_edges().to_int();
					fec.saved_path = fields[3];
					fec.used = false; // Assume unused until used.
					export_cache[path] = fec;
				}
				l = f->get_line();
			}
		} else {
			// create the path
			Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
			d->change_dir(ProjectSettings::get_singleton()->get_project_data_path());
			d->make_dir_recursive("exported/" + itos(custom_resources_hash));
		}
	}

	//store everything in the export medium
	int idx = 0;
	int total = paths.size();

	for (const String &E : paths) {
		String path = E;
		String type = ResourceLoader::get_resource_type(path);

		if (FileAccess::exists(path + ".import")) {
			// Before doing this, try to see if it can be customized.

			String export_path = _export_customize(path, customize_resources_plugins, customize_scenes_plugins, export_cache, export_base_path, false);

			if (export_path != path) {
				// It was actually customized.
				// Since the original file is likely not recognized, just use the import system.

				Ref<ConfigFile> config;
				config.instantiate();
				err = config->load(path + ".import");
				if (err != OK) {
					ERR_PRINT("Could not parse: '" + path + "', not exported.");
					continue;
				}
				config->set_value("remap", "type", ResourceLoader::get_resource_type(export_path));

				// Erase all Paths.
				List<String> keys;
				config->get_section_keys("remap", &keys);
				for (const String &K : keys) {
					if (K.begins_with("path")) {
						config->erase_section_key("remap", K);
					}
				}
				// Set actual converted path.
				config->set_value("remap", "path", export_path);

				// Erase useless sections.
				config->erase_section("deps");
				config->erase_section("params");

				String import_text = config->encode_to_text();
				CharString cs = import_text.utf8();
				Vector<uint8_t> sarr;
				sarr.resize(cs.size());
				memcpy(sarr.ptrw(), cs.ptr(), sarr.size());

				err = p_func(p_udata, path + ".import", sarr, idx, total, enc_in_filters, enc_ex_filters, key);
				if (err != OK) {
					return err;
				}
				// Now actual remapped file:
				sarr = FileAccess::get_file_as_bytes(export_path);
				err = p_func(p_udata, export_path, sarr, idx, total, enc_in_filters, enc_ex_filters, key);
				if (err != OK) {
					return err;
				}
			} else {
				// File is imported and not customized, replace by what it imports.
				Ref<ConfigFile> config;
				config.instantiate();
				err = config->load(path + ".import");
				if (err != OK) {
					ERR_PRINT("Could not parse: '" + path + "', not exported.");
					continue;
				}

				String importer_type = config->get_value("remap", "importer");

				if (importer_type == "keep") {
					// Just keep file as-is.
					Vector<uint8_t> array = FileAccess::get_file_as_bytes(path);
					err = p_func(p_udata, path, array, idx, total, enc_in_filters, enc_ex_filters, key);

					if (err != OK) {
						return err;
					}

					continue;
				}

				List<String> remaps;
				config->get_section_keys("remap", &remaps);

				HashSet<String> remap_features;

				for (const String &F : remaps) {
					String remap = F;
					String feature = remap.get_slice(".", 1);
					if (features.has(feature)) {
						remap_features.insert(feature);
					}
				}

				if (remap_features.size() > 1) {
					this->resolve_platform_feature_priorities(p_preset, remap_features);
				}

				err = OK;

				for (const String &F : remaps) {
					String remap = F;
					if (remap == "path") {
						String remapped_path = config->get_value("remap", remap);
						Vector<uint8_t> array = FileAccess::get_file_as_bytes(remapped_path);
						err = p_func(p_udata, remapped_path, array, idx, total, enc_in_filters, enc_ex_filters, key);
					} else if (remap.begins_with("path.")) {
						String feature = remap.get_slice(".", 1);

						if (remap_features.has(feature)) {
							String remapped_path = config->get_value("remap", remap);
							Vector<uint8_t> array = FileAccess::get_file_as_bytes(remapped_path);
							err = p_func(p_udata, remapped_path, array, idx, total, enc_in_filters, enc_ex_filters, key);
						} else {
							// Remove paths if feature not enabled.
							config->erase_section_key("remap", remap);
						}
					}
				}

				if (err != OK) {
					return err;
				}

				// Erase useless sections.
				config->erase_section("deps");
				config->erase_section("params");

				String import_text = config->encode_to_text();
				CharString cs = import_text.utf8();
				Vector<uint8_t> sarr;
				sarr.resize(cs.size());
				memcpy(sarr.ptrw(), cs.ptr(), sarr.size());

				err = p_func(p_udata, path + ".import", sarr, idx, total, enc_in_filters, enc_ex_filters, key);

				if (err != OK) {
					return err;
				}
			}

		} else {
			// Customize.

			bool do_export = true;
			for (int i = 0; i < export_plugins.size(); i++) {
				if (GDVIRTUAL_IS_OVERRIDDEN_PTR(export_plugins[i], _export_file)) {
					export_plugins.write[i]->_export_file_script(path, type, features_psa);
				} else {
					export_plugins.write[i]->_export_file(path, type, features);
				}
				if (p_so_func) {
					for (int j = 0; j < export_plugins[i]->shared_objects.size(); j++) {
						err = p_so_func(p_udata, export_plugins[i]->shared_objects[j]);
						if (err != OK) {
							return err;
						}
					}
				}

				for (int j = 0; j < export_plugins[i]->extra_files.size(); j++) {
					err = p_func(p_udata, export_plugins[i]->extra_files[j].path, export_plugins[i]->extra_files[j].data, idx, total, enc_in_filters, enc_ex_filters, key);
					if (err != OK) {
						return err;
					}
					if (export_plugins[i]->extra_files[j].remap) {
						do_export = false; //if remap, do not
						path_remaps.push_back(path);
						path_remaps.push_back(export_plugins[i]->extra_files[j].path);
					}
				}

				if (export_plugins[i]->skipped) {
					do_export = false;
				}
				export_plugins.write[i]->_clear();

				if (!do_export) {
					break; //apologies, not exporting
				}
			}
			//just store it as it comes
			if (do_export) {
				// Customization only happens if plugins did not take care of it before
				bool force_binary = convert_text_to_binary && (path.get_extension().to_lower() == "tres" || path.get_extension().to_lower() == "tscn");
				String export_path = _export_customize(path, customize_resources_plugins, customize_scenes_plugins, export_cache, export_base_path, force_binary);

				if (export_path != path) {
					// Add a remap entry
					path_remaps.push_back(path);
					path_remaps.push_back(export_path);
				}

				Vector<uint8_t> array = FileAccess::get_file_as_bytes(export_path);
				err = p_func(p_udata, export_path, array, idx, total, enc_in_filters, enc_ex_filters, key);
				if (err != OK) {
					return err;
				}
			}
		}

		idx++;
	}

	if (convert_text_to_binary || !customize_resources_plugins.is_empty() || !customize_scenes_plugins.is_empty()) {
		// End scene customization

		String fcache = export_base_path.path_join("file_cache");
		Ref<FileAccess> f = FileAccess::open(fcache, FileAccess::WRITE);

		if (f.is_valid()) {
			for (const KeyValue<String, FileExportCache> &E : export_cache) {
				if (E.value.used) { // May be old, unused
					String l = E.key + "::" + E.value.source_md5 + "::" + itos(E.value.source_modified_time) + "::" + E.value.saved_path;
					f->store_line(l);
				}
			}
		} else {
			ERR_PRINT("Error opening export file cache: " + fcache);
		}

		for (Ref<EditorExportPlugin> &plugin : customize_resources_plugins) {
			plugin->_end_customize_resources();
		}

		for (Ref<EditorExportPlugin> &plugin : customize_scenes_plugins) {
			plugin->_end_customize_scenes();
		}
	}
	//save config!

	Vector<String> custom_list;

	if (!p_preset->get_custom_features().is_empty()) {
		Vector<String> tmp_custom_list = p_preset->get_custom_features().split(",");

		for (int i = 0; i < tmp_custom_list.size(); i++) {
			String f = tmp_custom_list[i].strip_edges();
			if (!f.is_empty()) {
				custom_list.push_back(f);
			}
		}
	}
	for (int i = 0; i < export_plugins.size(); i++) {
		custom_list.append_array(export_plugins[i]->_get_export_features(Ref<EditorExportPlatform>(this), p_debug));
	}

	ProjectSettings::CustomMap custom_map;
	if (path_remaps.size()) {
		if (true) { //new remap mode, use always as it's friendlier with multiple .pck exports
			for (int i = 0; i < path_remaps.size(); i += 2) {
				String from = path_remaps[i];
				String to = path_remaps[i + 1];
				String remap_file = "[remap]\n\npath=\"" + to.c_escape() + "\"\n";
				CharString utf8 = remap_file.utf8();
				Vector<uint8_t> new_file;
				new_file.resize(utf8.length());
				for (int j = 0; j < utf8.length(); j++) {
					new_file.write[j] = utf8[j];
				}

				err = p_func(p_udata, from + ".remap", new_file, idx, total, enc_in_filters, enc_ex_filters, key);
				if (err != OK) {
					return err;
				}
			}
		} else {
			//old remap mode, will still work, but it's unused because it's not multiple pck export friendly
			custom_map["path_remap/remapped_paths"] = path_remaps;
		}
	}

	Vector<String> forced_export = get_forced_export_files();
	for (int i = 0; i < forced_export.size(); i++) {
		Vector<uint8_t> array = FileAccess::get_file_as_bytes(forced_export[i]);
		err = p_func(p_udata, forced_export[i], array, idx, total, enc_in_filters, enc_ex_filters, key);
		if (err != OK) {
			return err;
		}
	}

	String config_file = "project.binary";
	String engine_cfb = EditorPaths::get_singleton()->get_cache_dir().path_join("tmp" + config_file);
	ProjectSettings::get_singleton()->save_custom(engine_cfb, custom_map, custom_list);
	Vector<uint8_t> data = FileAccess::get_file_as_bytes(engine_cfb);
	DirAccess::remove_file_or_error(engine_cfb);

	return p_func(p_udata, "res://" + config_file, data, idx, total, enc_in_filters, enc_ex_filters, key);
}

Error EditorExportPlatform::_add_shared_object(void *p_userdata, const SharedObject &p_so) {
	PackData *pack_data = (PackData *)p_userdata;
	if (pack_data->so_files) {
		pack_data->so_files->push_back(p_so);
	}

	return OK;
}

void EditorExportPlatform::zip_folder_recursive(zipFile &p_zip, const String &p_root_path, const String &p_folder, const String &p_pkg_name) {
	String dir = p_folder.is_empty() ? p_root_path : p_root_path.path_join(p_folder);

	Ref<DirAccess> da = DirAccess::open(dir);
	ERR_FAIL_COND(da.is_null());

	da->list_dir_begin();
	String f = da->get_next();
	while (!f.is_empty()) {
		if (f == "." || f == "..") {
			f = da->get_next();
			continue;
		}
		if (da->is_link(f)) {
			OS::DateTime dt = OS::get_singleton()->get_datetime();

			zip_fileinfo zipfi;
			zipfi.tmz_date.tm_year = dt.year;
			zipfi.tmz_date.tm_mon = dt.month - 1; // Note: "tm" month range - 0..11, Godot month range - 1..12, https://www.cplusplus.com/reference/ctime/tm/
			zipfi.tmz_date.tm_mday = dt.day;
			zipfi.tmz_date.tm_hour = dt.hour;
			zipfi.tmz_date.tm_min = dt.minute;
			zipfi.tmz_date.tm_sec = dt.second;
			zipfi.dosDate = 0;
			// 0120000: symbolic link type
			// 0000755: permissions rwxr-xr-x
			// 0000644: permissions rw-r--r--
			uint32_t _mode = 0120644;
			zipfi.external_fa = (_mode << 16L) | !(_mode & 0200);
			zipfi.internal_fa = 0;

			zipOpenNewFileInZip4(p_zip,
					p_folder.path_join(f).utf8().get_data(),
					&zipfi,
					nullptr,
					0,
					nullptr,
					0,
					nullptr,
					Z_DEFLATED,
					Z_DEFAULT_COMPRESSION,
					0,
					-MAX_WBITS,
					DEF_MEM_LEVEL,
					Z_DEFAULT_STRATEGY,
					nullptr,
					0,
					0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
					0);

			String target = da->read_link(f);
			zipWriteInFileInZip(p_zip, target.utf8().get_data(), target.utf8().size());
			zipCloseFileInZip(p_zip);
		} else if (da->current_is_dir()) {
			zip_folder_recursive(p_zip, p_root_path, p_folder.path_join(f), p_pkg_name);
		} else {
			bool _is_executable = is_executable(dir.path_join(f));

			OS::DateTime dt = OS::get_singleton()->get_datetime();

			zip_fileinfo zipfi;
			zipfi.tmz_date.tm_year = dt.year;
			zipfi.tmz_date.tm_mon = dt.month - 1; // Note: "tm" month range - 0..11, Godot month range - 1..12, https://www.cplusplus.com/reference/ctime/tm/
			zipfi.tmz_date.tm_mday = dt.day;
			zipfi.tmz_date.tm_hour = dt.hour;
			zipfi.tmz_date.tm_min = dt.minute;
			zipfi.tmz_date.tm_sec = dt.second;
			zipfi.dosDate = 0;
			// 0100000: regular file type
			// 0000755: permissions rwxr-xr-x
			// 0000644: permissions rw-r--r--
			uint32_t _mode = (_is_executable ? 0100755 : 0100644);
			zipfi.external_fa = (_mode << 16L) | !(_mode & 0200);
			zipfi.internal_fa = 0;

			zipOpenNewFileInZip4(p_zip,
					p_folder.path_join(f).utf8().get_data(),
					&zipfi,
					nullptr,
					0,
					nullptr,
					0,
					nullptr,
					Z_DEFLATED,
					Z_DEFAULT_COMPRESSION,
					0,
					-MAX_WBITS,
					DEF_MEM_LEVEL,
					Z_DEFAULT_STRATEGY,
					nullptr,
					0,
					0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
					0);

			Ref<FileAccess> fa = FileAccess::open(dir.path_join(f), FileAccess::READ);
			if (fa.is_null()) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("ZIP Creation"), vformat(TTR("Could not open file to read from path \"%s\"."), dir.path_join(f)));
				return;
			}
			const int bufsize = 16384;
			uint8_t buf[bufsize];

			while (true) {
				uint64_t got = fa->get_buffer(buf, bufsize);
				if (got == 0) {
					break;
				}
				zipWriteInFileInZip(p_zip, buf, got);
			}

			zipCloseFileInZip(p_zip);
		}
		f = da->get_next();
	}
	da->list_dir_end();
}

Error EditorExportPlatform::save_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, Vector<SharedObject> *p_so_files, bool p_embed, int64_t *r_embedded_start, int64_t *r_embedded_size) {
	EditorProgress ep("savepack", TTR("Packing"), 102, true);

	// Create the temporary export directory if it doesn't exist.
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->make_dir_recursive(EditorPaths::get_singleton()->get_cache_dir());

	String tmppath = EditorPaths::get_singleton()->get_cache_dir().path_join("packtmp");
	Ref<FileAccess> ftmp = FileAccess::open(tmppath, FileAccess::WRITE);
	if (ftmp.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), vformat(TTR("Cannot create file \"%s\"."), tmppath));
		return ERR_CANT_CREATE;
	}

	PackData pd;
	pd.ep = &ep;
	pd.f = ftmp;
	pd.so_files = p_so_files;

	Error err = export_project_files(p_preset, p_debug, _save_pack_file, &pd, _add_shared_object);

	// Close temp file.
	pd.f.unref();
	ftmp.unref();

	if (err != OK) {
		DirAccess::remove_file_or_error(tmppath);
		add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), TTR("Failed to export project files."));
		return err;
	}

	pd.file_ofs.sort(); //do sort, so we can do binary search later

	Ref<FileAccess> f;
	int64_t embed_pos = 0;
	if (!p_embed) {
		// Regular output to separate PCK file
		f = FileAccess::open(p_path, FileAccess::WRITE);
		if (f.is_null()) {
			DirAccess::remove_file_or_error(tmppath);
			add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), vformat(TTR("Can't open file for writing at path \"%s\"."), p_path));
			return ERR_CANT_CREATE;
		}
	} else {
		// Append to executable
		f = FileAccess::open(p_path, FileAccess::READ_WRITE);
		if (f.is_null()) {
			DirAccess::remove_file_or_error(tmppath);
			add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), vformat(TTR("Can't open file for reading-writing at path \"%s\"."), p_path));
			return ERR_FILE_CANT_OPEN;
		}

		f->seek_end();
		embed_pos = f->get_position();

		if (r_embedded_start) {
			*r_embedded_start = embed_pos;
		}

		// Ensure embedded PCK starts at a 64-bit multiple
		int pad = f->get_position() % 8;
		for (int i = 0; i < pad; i++) {
			f->store_8(0);
		}
	}

	int64_t pck_start_pos = f->get_position();

	f->store_32(PACK_HEADER_MAGIC);
	f->store_32(PACK_FORMAT_VERSION);
	f->store_32(VERSION_MAJOR);
	f->store_32(VERSION_MINOR);
	f->store_32(VERSION_PATCH);

	uint32_t pack_flags = 0;
	bool enc_pck = p_preset->get_enc_pck();
	bool enc_directory = p_preset->get_enc_directory();
	if (enc_pck && enc_directory) {
		pack_flags |= PACK_DIR_ENCRYPTED;
	}
	f->store_32(pack_flags); // flags

	uint64_t file_base_ofs = f->get_position();
	f->store_64(0); // files base

	for (int i = 0; i < 16; i++) {
		//reserved
		f->store_32(0);
	}

	f->store_32(pd.file_ofs.size()); //amount of files

	Ref<FileAccessEncrypted> fae;
	Ref<FileAccess> fhead = f;

	if (enc_pck && enc_directory) {
		String script_key = _get_script_encryption_key(p_preset);
		Vector<uint8_t> key;
		key.resize(32);
		if (script_key.length() == 64) {
			for (int i = 0; i < 32; i++) {
				int v = 0;
				if (i * 2 < script_key.length()) {
					char32_t ct = script_key[i * 2];
					if (is_digit(ct)) {
						ct = ct - '0';
					} else if (ct >= 'a' && ct <= 'f') {
						ct = 10 + ct - 'a';
					}
					v |= ct << 4;
				}

				if (i * 2 + 1 < script_key.length()) {
					char32_t ct = script_key[i * 2 + 1];
					if (is_digit(ct)) {
						ct = ct - '0';
					} else if (ct >= 'a' && ct <= 'f') {
						ct = 10 + ct - 'a';
					}
					v |= ct;
				}
				key.write[i] = v;
			}
		}
		fae.instantiate();
		if (fae.is_null()) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), TTR("Can't create encrypted file."));
			return ERR_CANT_CREATE;
		}

		err = fae->open_and_parse(f, key, FileAccessEncrypted::MODE_WRITE_AES256, false);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), TTR("Can't open encrypted file to write."));
			return ERR_CANT_CREATE;
		}

		fhead = fae;
	}

	for (int i = 0; i < pd.file_ofs.size(); i++) {
		uint32_t string_len = pd.file_ofs[i].path_utf8.length();
		uint32_t pad = _get_pad(4, string_len);

		fhead->store_32(string_len + pad);
		fhead->store_buffer((const uint8_t *)pd.file_ofs[i].path_utf8.get_data(), string_len);
		for (uint32_t j = 0; j < pad; j++) {
			fhead->store_8(0);
		}

		fhead->store_64(pd.file_ofs[i].ofs);
		fhead->store_64(pd.file_ofs[i].size); // pay attention here, this is where file is
		fhead->store_buffer(pd.file_ofs[i].md5.ptr(), 16); //also save md5 for file
		uint32_t flags = 0;
		if (pd.file_ofs[i].encrypted) {
			flags |= PACK_FILE_ENCRYPTED;
		}
		fhead->store_32(flags);
	}

	if (fae.is_valid()) {
		fhead.unref();
		fae.unref();
	}

	int header_padding = _get_pad(PCK_PADDING, f->get_position());
	for (int i = 0; i < header_padding; i++) {
		f->store_8(0);
	}

	uint64_t file_base = f->get_position();
	f->seek(file_base_ofs);
	f->store_64(file_base); // update files base
	f->seek(file_base);

	// Save the rest of the data.

	ftmp = FileAccess::open(tmppath, FileAccess::READ);
	if (ftmp.is_null()) {
		DirAccess::remove_file_or_error(tmppath);
		add_message(EXPORT_MESSAGE_ERROR, TTR("Save PCK"), vformat(TTR("Can't open file to read from path \"%s\"."), tmppath));
		return ERR_CANT_CREATE;
	}

	const int bufsize = 16384;
	uint8_t buf[bufsize];

	while (true) {
		uint64_t got = ftmp->get_buffer(buf, bufsize);
		if (got == 0) {
			break;
		}
		f->store_buffer(buf, got);
	}

	ftmp.unref(); // Close temp file.

	if (p_embed) {
		// Ensure embedded data ends at a 64-bit multiple
		uint64_t embed_end = f->get_position() - embed_pos + 12;
		uint64_t pad = embed_end % 8;
		for (uint64_t i = 0; i < pad; i++) {
			f->store_8(0);
		}

		uint64_t pck_size = f->get_position() - pck_start_pos;
		f->store_64(pck_size);
		f->store_32(PACK_HEADER_MAGIC);

		if (r_embedded_size) {
			*r_embedded_size = f->get_position() - embed_pos;
		}
	}

	DirAccess::remove_file_or_error(tmppath);

	return OK;
}

Error EditorExportPlatform::save_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	EditorProgress ep("savezip", TTR("Packing"), 102, true);

	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);
	zipFile zip = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, nullptr, &io);

	ZipData zd;
	zd.ep = &ep;
	zd.zip = zip;

	Error err = export_project_files(p_preset, p_debug, _save_zip_file, &zd);
	if (err != OK && err != ERR_SKIP) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Save ZIP"), TTR("Failed to export project files."));
	}

	zipClose(zip, nullptr);

	return OK;
}

Error EditorExportPlatform::export_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);
	return save_pack(p_preset, p_debug, p_path);
}

Error EditorExportPlatform::export_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);
	return save_zip(p_preset, p_debug, p_path);
}

void EditorExportPlatform::gen_export_flags(Vector<String> &r_flags, int p_flags) {
	String host = EDITOR_GET("network/debug/remote_host");
	int remote_port = (int)EDITOR_GET("network/debug/remote_port");

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST) {
		host = "localhost";
	}

	if (p_flags & DEBUG_FLAG_DUMB_CLIENT) {
		int port = EDITOR_GET("filesystem/file_server/port");
		String passwd = EDITOR_GET("filesystem/file_server/password");
		r_flags.push_back("--remote-fs");
		r_flags.push_back(host + ":" + itos(port));
		if (!passwd.is_empty()) {
			r_flags.push_back("--remote-fs-password");
			r_flags.push_back(passwd);
		}
	}

	if (p_flags & DEBUG_FLAG_REMOTE_DEBUG) {
		r_flags.push_back("--remote-debug");

		r_flags.push_back(get_debug_protocol() + host + ":" + String::num(remote_port));

		List<String> breakpoints;
		ScriptEditor::get_singleton()->get_breakpoints(&breakpoints);

		if (breakpoints.size()) {
			r_flags.push_back("--breakpoints");
			String bpoints;
			for (List<String>::Element *E = breakpoints.front(); E; E = E->next()) {
				bpoints += E->get().replace(" ", "%20");
				if (E->next()) {
					bpoints += ",";
				}
			}

			r_flags.push_back(bpoints);
		}
	}

	if (p_flags & DEBUG_FLAG_VIEW_COLLISIONS) {
		r_flags.push_back("--debug-collisions");
	}

	if (p_flags & DEBUG_FLAG_VIEW_NAVIGATION) {
		r_flags.push_back("--debug-navigation");
	}
}

bool EditorExportPlatform::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
	bool valid = true;

#ifndef ANDROID_ENABLED
	String templates_error;
	valid = valid && has_valid_export_configuration(p_preset, templates_error, r_missing_templates, p_debug);

	if (!templates_error.is_empty()) {
		r_error += templates_error;
	}

	String export_plugins_warning;
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		Ref<EditorExportPlatform> export_platform = Ref<EditorExportPlatform>(this);
		if (!export_plugins[i]->supports_platform(export_platform)) {
			continue;
		}

		String plugin_warning = export_plugins.write[i]->_has_valid_export_configuration(export_platform, p_preset);
		if (!plugin_warning.is_empty()) {
			export_plugins_warning += plugin_warning;
		}
	}

	if (!export_plugins_warning.is_empty()) {
		r_error += export_plugins_warning;
	}
#endif

	String project_configuration_error;
	valid = valid && has_valid_project_configuration(p_preset, project_configuration_error);

	if (!project_configuration_error.is_empty()) {
		r_error += project_configuration_error;
	}

	return valid;
}

Error EditorExportPlatform::ssh_run_on_remote(const String &p_host, const String &p_port, const Vector<String> &p_ssh_args, const String &p_cmd_args, String *r_out, int p_port_fwd) const {
	String ssh_path = EditorSettings::get_singleton()->get("export/ssh/ssh");
	if (ssh_path.is_empty()) {
		ssh_path = "ssh";
	}

	List<String> args;
	args.push_back("-p");
	args.push_back(p_port);
	args.push_back("-q");
	args.push_back("-o");
	args.push_back("LogLevel=error");
	args.push_back("-o");
	args.push_back("BatchMode=yes");
	args.push_back("-o");
	args.push_back("StrictHostKeyChecking=no");
	for (const String &E : p_ssh_args) {
		args.push_back(E);
	}
	if (p_port_fwd > 0) {
		args.push_back("-R");
		args.push_back(vformat("%d:localhost:%d", p_port_fwd, p_port_fwd));
	}
	args.push_back(p_host);
	args.push_back(p_cmd_args);

	String out;
	int exit_code = -1;

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print("Executing: %s", ssh_path.utf8().get_data());
		for (const String &arg : args) {
			OS::get_singleton()->print(" %s", arg.utf8().get_data());
		}
		OS::get_singleton()->print("\n");
	}

	Error err = OS::get_singleton()->execute(ssh_path, args, &out, &exit_code, true);
	if (out.is_empty()) {
		print_verbose(vformat("Exit code: %d", exit_code));
	} else {
		print_verbose(vformat("Exit code: %d, Output: %s", exit_code, out.replace("\r\n", "\n")));
	}
	if (r_out) {
		*r_out = out.replace("\r\n", "\n").get_slice("\n", 0);
	}
	if (err != OK) {
		return err;
	} else if (exit_code != 0) {
		if (!out.is_empty()) {
			print_line(out);
		}
		return FAILED;
	}
	return OK;
}

Error EditorExportPlatform::ssh_run_on_remote_no_wait(const String &p_host, const String &p_port, const Vector<String> &p_ssh_args, const String &p_cmd_args, OS::ProcessID *r_pid, int p_port_fwd) const {
	String ssh_path = EditorSettings::get_singleton()->get("export/ssh/ssh");
	if (ssh_path.is_empty()) {
		ssh_path = "ssh";
	}

	List<String> args;
	args.push_back("-p");
	args.push_back(p_port);
	args.push_back("-q");
	args.push_back("-o");
	args.push_back("LogLevel=error");
	args.push_back("-o");
	args.push_back("BatchMode=yes");
	args.push_back("-o");
	args.push_back("StrictHostKeyChecking=no");
	for (const String &E : p_ssh_args) {
		args.push_back(E);
	}
	if (p_port_fwd > 0) {
		args.push_back("-R");
		args.push_back(vformat("%d:localhost:%d", p_port_fwd, p_port_fwd));
	}
	args.push_back(p_host);
	args.push_back(p_cmd_args);

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print("Executing: %s", ssh_path.utf8().get_data());
		for (const String &arg : args) {
			OS::get_singleton()->print(" %s", arg.utf8().get_data());
		}
		OS::get_singleton()->print("\n");
	}

	return OS::get_singleton()->create_process(ssh_path, args, r_pid);
}

Error EditorExportPlatform::ssh_push_to_remote(const String &p_host, const String &p_port, const Vector<String> &p_scp_args, const String &p_src_file, const String &p_dst_file) const {
	String scp_path = EditorSettings::get_singleton()->get("export/ssh/scp");
	if (scp_path.is_empty()) {
		scp_path = "scp";
	}

	List<String> args;
	args.push_back("-P");
	args.push_back(p_port);
	args.push_back("-q");
	args.push_back("-o");
	args.push_back("LogLevel=error");
	args.push_back("-o");
	args.push_back("BatchMode=yes");
	args.push_back("-o");
	args.push_back("StrictHostKeyChecking=no");
	for (const String &E : p_scp_args) {
		args.push_back(E);
	}
	args.push_back(p_src_file);
	args.push_back(vformat("%s:%s", p_host, p_dst_file));

	String out;
	int exit_code = -1;

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print("Executing: %s", scp_path.utf8().get_data());
		for (const String &arg : args) {
			OS::get_singleton()->print(" %s", arg.utf8().get_data());
		}
		OS::get_singleton()->print("\n");
	}

	Error err = OS::get_singleton()->execute(scp_path, args, &out, &exit_code, true);
	if (err != OK) {
		return err;
	} else if (exit_code != 0) {
		if (!out.is_empty()) {
			print_line(out);
		}
		return FAILED;
	}
	return OK;
}

void EditorExportPlatform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_os_name"), &EditorExportPlatform::get_os_name);
}

EditorExportPlatform::EditorExportPlatform() {
}
