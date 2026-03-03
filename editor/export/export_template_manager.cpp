/**************************************************************************/
/*  export_template_manager.cpp                                           */
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

#include "export_template_manager.h"

#include "core/config/engine.h"
#include "core/error/error_list.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/io/zip_io.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/export/editor_export.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/file_system/editor_paths.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/gui/progress_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/link_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"
#include "scene/main/http_request.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"
#include "servers/display/display_server.h"

void ExportTemplateManager::_request_mirrors() {
	mirrors_list->clear();
	mirrors_empty = true;
	_update_install_button();

	// Downloadable export templates are only available for stable and official alpha/beta/RC builds
	// (which always have a number following their status, e.g. "alpha1").
	// Therefore, don't display download-related features when using a development version
	// (whose builds aren't numbered).
	if (!strcmp(GODOT_VERSION_STATUS, "dev") || !strcmp(GODOT_VERSION_STATUS, "beta") || !strcmp(GODOT_VERSION_STATUS, "rc")) {
		_set_empty_mirror_list();
		mirrors_list->set_tooltip_text(TTRC("Official export templates aren't available for development builds."));
#ifdef REAL_T_IS_DOUBLE
	} else if (true) {
		_set_empty_mirror_list();
		mirrors_list->set_tooltip_text(TTRC("Official export templates aren't available for double-precision builds."));
#endif
	} else if (!_is_online()) {
		mirrors_list->set_tooltip_text(TTRC("Template downloading is disabled in offline mode."));
	} else {
		mirrors_list->set_tooltip_text(String());
	}
	const String mirrors_metadata_url = vformat("https://godotengine.org/mirrorlist/%s.json", "4.6.stable" /*GODOT_VERSION_FULL_CONFIG*/); // TODO: debug-adjusted, uncomment before merging xd
	mirrors_requester->request(mirrors_metadata_url);
}

void ExportTemplateManager::_mirrors_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	mirrors_list->clear();

	if (p_result != HTTPRequest::RESULT_SUCCESS || p_response_code != HTTPClient::RESPONSE_OK) {
		String error = TTR("Error getting the list of mirrors.") + "\n";
		if (p_result == HTTPRequest::RESULT_SUCCESS && p_response_code == HTTPClient::RESPONSE_NOT_FOUND) {
			// Response successful, but wrong address.
			error += TTR("No mirrors found for this version. Template download is only available for official releases.");
		} else {
			error += vformat(TTR("Result: %d\nResponse code: %d"), p_result, p_response_code);
		}
		EditorNode::get_singleton()->show_warning(error);
		_set_empty_mirror_list();
		return;
	}

	String response_json = String::utf8((const char *)p_body.ptr(), p_body.size());

	JSON json;
	Error err = json.parse(response_json);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error parsing JSON with the list of mirrors. Please report this issue!"));
		_set_empty_mirror_list();
		return;
	}

	bool mirrors_available = false;

	Dictionary mirror_data = json.get_data();
	if (mirror_data.has("mirrors")) {
		Array mirrors = mirror_data["mirrors"];
		mirrors.push_front(Dictionary({ { "name", "localhost8k" }, { "url", "http://127.0.0.1:8000" } })); // TODO: debug-only, remove before merging xd
		for (const Variant &mirror : mirrors) {
			Dictionary m = mirror;
			ERR_CONTINUE(!m.has("url") || !m.has("name"));

			mirrors_list->add_item(m["name"]);
			mirrors_list->set_item_metadata(-1, m["url"]);

			mirrors_available = true;
		}
		// Hard-coded for translation. Should match the up-to-date list of mirrors.
		// TTR("Official Releases mirror")
	}
	if (!mirrors_available) {
		_set_empty_mirror_list();
	} else {
		mirrors_list->set_disabled(false);
		open_mirror->set_disabled(false);
		mirrors_empty = false;

		_update_install_button();
		if (!is_downloading()) {
			// Some tree buttons won't show until mirrors are loaded.
			_update_template_tree();
		}
	}
}

void ExportTemplateManager::_set_empty_mirror_list() {
	mirrors_list->add_item(TTRC("No mirrors"));
	mirrors_list->set_disabled(true);
	open_mirror->set_disabled(true);
	mirrors_empty = true;
	_update_install_button();
}

String ExportTemplateManager::_get_current_mirror_url() const {
	return mirrors_list->get_item_metadata(mirrors_list->get_selected());
}

void ExportTemplateManager::_update_online_mode() {
	offline_container->set_visible((int)EDITOR_GET("network/connection/network_mode") == EditorSettings::NETWORK_OFFLINE);

	if (_is_online()) {
		_update_install_button();
	} else {
		mirrors_list->clear();
		_set_empty_mirror_list();
	}
}

bool ExportTemplateManager::_is_online() const {
	return !offline_container->is_visible();
}

void ExportTemplateManager::_force_online_mode() {
	EditorSettings::get_singleton()->set_setting("network/connection/network_mode", EditorSettings::NETWORK_ONLINE);
	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();

	_update_online_mode();
	_request_mirrors();
}

void ExportTemplateManager::_open_mirror() {
	OS::get_singleton()->shell_open(_get_current_mirror_url());
}

void ExportTemplateManager::_initialize_template_data() {
	// Base templates.
	{
		TemplateInfo info;
		info.name = "Windows x86-32";
		info.description = TTRC("32-bit build for Microsoft Windows, including console wrapper.");
		info.file_list = { "windows_debug_x86_32.exe", "windows_debug_x86_32_console.exe", "windows_release_x86_32.exe", "windows_release_x86_32_console.exe" };
		template_data[TemplateID::WINDOWS_X86_32] = info;
	}
	{
		TemplateInfo info;
		info.name = "Windows x86-64";
		info.description = TTRC("64-bit build for Microsoft Windows, including console wrapper.");
		info.file_list = { "windows_debug_x86_64.exe", "windows_debug_x86_64_console.exe", "windows_release_x86_64.exe", "windows_release_x86_64_console.exe" };
		template_data[TemplateID::WINDOWS_X86_64] = info;
	}
	{
		TemplateInfo info;
		info.name = "Windows ARM-64";
		info.description = TTRC("32-bit build for Microsoft Windows on ARM architecture, including console wrapper.");
		info.file_list = { "windows_debug_arm64.exe", "windows_debug_arm64_console.exe", "windows_release_arm64.exe", "windows_release_arm64_console.exe" };
		template_data[TemplateID::WINDOWS_ARM64] = info;
	}

	{
		TemplateInfo info;
		info.name = "Linux x86-32";
		info.description = TTRC("32-bit build for Linux systems.");
		info.file_list = { "linux_debug.x86_32", "linux_release.x86_32" };
		template_data[TemplateID::LINUX_X86_32] = info;
	}
	{
		TemplateInfo info;
		info.name = "Linux x86-64";
		info.description = TTRC("64-bit build for Linux systems.");
		info.file_list = { "linux_debug.x86_64", "linux_release.x86_64" };
		template_data[TemplateID::LINUX_X86_64] = info;
	}
	{
		TemplateInfo info;
		info.name = "Linux ARM-32";
		info.description = TTRC("32-bit build for Linux systems on ARM architecture.");
		info.file_list = { "linux_debug.arm32", "linux_release.arm32" };
		template_data[TemplateID::LINUX_ARM32] = info;
	}
	{
		TemplateInfo info;
		info.name = "Linux ARM-64";
		info.description = TTRC("64-bit build for Linux systems on ARM architecture.");
		info.file_list = { "linux_debug.arm64", "linux_release.arm64" };
		template_data[TemplateID::LINUX_ARM64] = info;
	}

	{
		TemplateInfo info;
		info.name = "macOS";
		info.description = TTRC("Universal build for macOS.");
		info.file_list = { "macos.zip" };
		template_data[TemplateID::MACOS] = info;
	}

	{
		TemplateInfo info;
		info.name = "Web";
		info.description = TTRC("Regular web build with threading support. Threads improve performance, but require \"cross-origin isolated\" website to run.");
		info.file_list = { "web_debug.zip", "web_release.zip" };
		template_data[TemplateID::WEB] = info;
	}
	{
		TemplateInfo info;
		info.name = TTR("Web with Extensions");
		info.description = TTRC("Web build with support for GDExtextensions. Only useful if you use GDExtensions, otherwise it only increases build size.");
		info.file_list = { "web_dlink_debug.zip", "web_dlink_release.zip" };
		template_data[TemplateID::WEB_EXTENSIONS] = info;
	}
	{
		TemplateInfo info;
		info.name = TTR("Web Single-Threaded");
		info.description = TTRC("Web build without threading support.");
		info.file_list = { "web_nothreads_debug.zip", "web_nothreads_release.zip" };
		template_data[TemplateID::WEB_NOTHREADS] = info;
	}
	{
		TemplateInfo info;
		info.name = TTR("Web with Extensions Single-Threaded");
		info.description = TTRC("Web build with GDExtension support and no threading support.");
		info.file_list = { "web_dlink_nothreads_debug.zip", "web_dlink_nothreads_release.zip" };
		template_data[TemplateID::WEB_EXTENSIONS_NOTHREADS] = info;
	}

	{
		TemplateInfo info;
		info.name = "Android";
		info.description = TTRC("Basic Android APK template.");
		info.file_list = { "android_debug.apk", "android_release.apk" };
		template_data[TemplateID::ANDROID] = info;
	}
	{
		TemplateInfo info;
		info.name = TTR("Android Source");
		info.description = TTRC("Template for Gradle builds for Android.");
		info.file_list = { "android_source.zip" };
		template_data[TemplateID::ANDROID_SOURCE] = info;
	}

	{
		TemplateInfo info;
		info.name = "iOS";
		info.description = TTRC("Build for Apple's iOS.");
		info.file_list = { "ios.zip" };
		template_data[TemplateID::IOS] = info;
	}

	{
		TemplateInfo info;
		info.name = TTR("ICU Data");
		info.description = TTRC("Line breaking dictionaries for TextServer, used by certain languages.");
		info.file_list = { "icudt_godot.dat" };
		template_data[TemplateID::ICU_DATA] = info;
	}

	// Platforms.
	{
		PlatformInfo info;
		info.name = "Windows";
		info.icon = _get_platform_icon("Windows Desktop");
		info.templates = { TemplateID::WINDOWS_X86_32, TemplateID::WINDOWS_X86_64, TemplateID::WINDOWS_ARM64 };
		info.group = TTR("Desktop", "Platform Group");
		platform_map[PlatformID::WINDOWS] = info;
	}
	{
		PlatformInfo info;
		info.name = "Linux";
		info.icon = _get_platform_icon("Linux");
		info.templates = { TemplateID::LINUX_X86_32, TemplateID::LINUX_X86_64, TemplateID::LINUX_ARM32, TemplateID::LINUX_ARM64 };
		info.group = TTR("Desktop", "Platform Group");
		platform_map[PlatformID::LINUX] = info;
	}
	{
		PlatformInfo info;
		info.name = "macOS";
		info.icon = _get_platform_icon("macOS");
		info.templates = { TemplateID::MACOS };
		info.group = TTR("Desktop", "Platform Group");
		platform_map[PlatformID::MACOS] = info;
	}
	{
		PlatformInfo info;
		info.name = "Android";
		info.icon = _get_platform_icon("Android");
		info.templates = { TemplateID::ANDROID, TemplateID::ANDROID_SOURCE };
		info.group = TTR("Mobile", "Platform Group");
		platform_map[PlatformID::ANDROID] = info;
	}
	{
		PlatformInfo info;
		info.name = "iOS";
		info.icon = _get_platform_icon("iOS");
		info.templates = { TemplateID::IOS };
		info.group = TTR("Mobile", "Platform Group");
		platform_map[PlatformID::IOS] = info;
	}
	{
		PlatformInfo info;
		info.name = "Web";
		info.icon = _get_platform_icon("Web");
		info.templates = { TemplateID::WEB, TemplateID::WEB_EXTENSIONS, TemplateID::WEB_NOTHREADS, TemplateID::WEB_EXTENSIONS_NOTHREADS };
		info.group = TTR("Web", "Platform Group");
		platform_map[PlatformID::WEB] = info;
	}
	{
		PlatformInfo info;
		info.name = TTR("Common");
		info.templates = { TemplateID::ICU_DATA };
		platform_map[PlatformID::COMMON] = info;
	}

	// Template directory status.
	DirAccess::make_dir_recursive_absolute(_get_template_folder_path(VERSION_FULL_CONFIG));
	Ref<DirAccess> templates_dir = DirAccess::open(EditorPaths::get_singleton()->get_export_templates_dir());
	ERR_FAIL_COND(templates_dir.is_null());

	for (const String &dir : templates_dir->get_directories()) {
		if (dir == GODOT_VERSION_FULL_CONFIG) {
			version_list->add_item(dir);
			version_list->set_item_custom_fg_color(-1, theme_cache.current_version_color);
			version_list->select(version_list->get_item_count() - 1);
		} else {
			version_list->add_item(dir);
		}
		version_list->set_item_metadata(-1, dir);
	}
}

void ExportTemplateManager::_update_template_tree() {
	downloading_items.clear();

	const String selected_version = version_list->get_item_text(version_list->get_current());
	Ref<DirAccess> template_directory = DirAccess::open(_get_template_folder_path(selected_version));
	ERR_FAIL_COND(template_directory.is_null());

	bool is_current_version = (selected_version == GODOT_VERSION_FULL_CONFIG);
	HashMap<TemplateID, LocalVector<String>> installed_template_files;

	for (const KeyValue<PlatformID, PlatformInfo> &KV : platform_map) {
		for (TemplateID id : KV.value.templates) {
			for (const String &file : template_data[id].file_list) {
				if (template_directory->file_exists(file)) {
					installed_template_files[id].push_back(file);
				}
			}
		}
	}

	_fill_template_tree(available_templates_tree, installed_template_files, is_current_version);
	_fill_template_tree(installed_templates_tree, installed_template_files, is_current_version);
}

void ExportTemplateManager::_update_template_tree_theme(Tree *p_tree) {
	if (is_downloading()) {
		// Prevents hiding progress bar.
		Ref<StyleBoxEmpty> empty_style;
		empty_style.instantiate();

		p_tree->add_theme_style_override(SNAME("hovered"), empty_style);
		p_tree->add_theme_style_override(SNAME("hovered_dimmed"), empty_style);
		p_tree->add_theme_style_override(SNAME("selected"), empty_style);
		p_tree->add_theme_style_override(SNAME("selected_focus"), empty_style);
		p_tree->add_theme_style_override(SNAME("hovered_selected"), empty_style);
		p_tree->add_theme_style_override(SNAME("hovered_selected_focus"), empty_style);
	} else {
		p_tree->remove_theme_style_override(SNAME("hovered"));
		p_tree->remove_theme_style_override(SNAME("hovered_dimmed"));
		p_tree->remove_theme_style_override(SNAME("selected"));
		p_tree->remove_theme_style_override(SNAME("selected_focus"));
		p_tree->remove_theme_style_override(SNAME("hovered_selected"));
		p_tree->remove_theme_style_override(SNAME("hovered_selected_focus"));
	}
}

void ExportTemplateManager::_fill_template_tree(Tree *p_tree, const HashMap<TemplateID, LocalVector<String>> &p_installed_template_files, bool p_is_current_version) {
	bool is_installed_tree = (p_tree == installed_templates_tree);
	bool is_available_tree = !is_installed_tree; // For readability.
	const LocalVector<String> empty_vector;

	if (p_tree->get_root()) {
		_update_folding_cache(p_tree->get_root());
		p_tree->clear();
	}

	TreeItem *platform_parent = p_tree->create_item();
	_setup_item_text(platform_parent, String());

	if (is_available_tree && !p_is_current_version) {
		TreeItem *nodownloadsforyou = platform_parent->create_child();
		nodownloadsforyou->set_text(0, TTR("Downloads are only available for the current Godot version."));
		nodownloadsforyou->set_custom_color(0, get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
		return;
	}

	String current_group;
	for (const KeyValue<PlatformID, PlatformInfo> &KV : platform_map) {
		const PlatformInfo &template_platform = KV.value;

		bool all_installed = true;
		bool any_installed = false;
		for (TemplateID id : template_platform.templates) {
			if (p_installed_template_files.has(id) && !queued_templates.has(template_data[id].name)) {
				any_installed = true;
			} else {
				all_installed = false;
			}

			if (any_installed && !all_installed) {
				// Not going to change anymore.
				break;
			}
		}

		if ((is_available_tree && all_installed) || (is_installed_tree && !any_installed)) {
			continue;
		}

		if (is_available_tree && template_platform.group != current_group) {
			// Use platform groups only for available templates.
			_apply_item_folding(platform_parent);
			current_group = template_platform.group;

			if (current_group.is_empty()) {
				platform_parent = p_tree->get_root();
			} else {
				platform_parent = p_tree->create_item();
				if (!is_downloading()) {
					_set_item_type(platform_parent, TreeItem::CELL_MODE_CHECK);
				}
				_setup_item_text(platform_parent, current_group);
			}
		}

		TreeItem *platform_item = platform_parent->create_child();
		if (is_available_tree && !is_downloading()) {
			_set_item_type(platform_item, TreeItem::CELL_MODE_CHECK);
		}
		_setup_item_text(platform_item, template_platform.name);
		platform_item->set_icon(0, template_platform.icon);
		platform_item->set_icon_max_width(0, theme_cache.icon_width);

		for (TemplateID id : template_platform.templates) {
			TemplateInfo &template_info = template_data[id];

			bool is_template_installed = p_installed_template_files.has(id);
			if (!queued_templates.has(template_info.name)) {
				if (is_template_installed == is_available_tree) {
					continue;
				}
			} else if (is_installed_tree) {
				continue;
			}

			const LocalVector<String> &installed_files = is_template_installed ? p_installed_template_files[id] : empty_vector;

			TreeItem *template_item;
			if (template_platform.templates.size() == 1 && template_info.name == template_platform.name) {
				// Single template with the same name as platform, so it can be skipped.
				template_item = platform_item;
			} else {
				template_item = platform_item->create_child();
			}

			if (is_available_tree) {
				if (queued_templates.has(template_info.name)) {
					_set_item_type(template_item, TreeItem::CELL_MODE_CUSTOM);
					template_item->add_button(0, theme_cache.cancel_icon, (int)ButtonID::CANCEL);
					template_item->set_button_tooltip_text(0, -1, TTR("Cancel downloading this template."));
				} else if (!is_downloading()) {
					_set_item_type(template_item, TreeItem::CELL_MODE_CHECK);
				}
			}
			_setup_item_text(template_item, template_info.name);
			template_item->set_tooltip_text(0, TTR(template_info.description));

			bool any_missing = false;
			bool any_failed = false;
			for (const String &file : template_info.file_list) {
				FileMetadata *meta = _get_file_metadata(file);

				TreeItem *file_item = template_item->create_child();
				file_item->set_meta(FILE_META, true);

				if (meta->download_status == DownloadStatus::FAILED) {
					_add_fail_reason_button(file_item, file);
					any_failed = true;
				}

				if (is_available_tree && !is_downloading()) {
					_set_item_type(file_item, TreeItem::CELL_MODE_CHECK);
				} else if (meta->download_status != DownloadStatus::NONE || queued_files.has(file)) {
					if (!_status_is_finished(meta->download_status)) {
						_set_item_type(file_item, TreeItem::CELL_MODE_CUSTOM);

						file_item->add_button(0, theme_cache.cancel_icon, (int)ButtonID::CANCEL);
						file_item->set_button_tooltip_text(0, -1, TTRC("Cancel downloading this file."));
						downloading_items.push_back(file_item);

						if (meta->download_status == DownloadStatus::NONE) {
							meta->download_status = DownloadStatus::PENDING;
						}
					}
				}
				_setup_item_text(file_item, file);

				if (is_installed_tree) {
					if (installed_files.has(file)) {
						file_item->add_button(0, theme_cache.remove_icon, (int)ButtonID::REMOVE);
						file_item->set_button_tooltip_text(0, -1, TTR("Remove this file."));
					} else {
						file_item->set_custom_color(0, theme_cache.missing_file_color);
						if (p_is_current_version && !is_downloading() && _can_download_templates()) {
							file_item->add_button(0, theme_cache.install_icon, (int)ButtonID::DOWNLOAD);
							file_item->set_button_tooltip_text(0, -1, TTR("Download this missing file."));
						}
						meta->is_missing = true;
						any_missing = true;
					}
				}
			}
			if (any_failed || any_missing) {
				template_item->set_custom_color(0, theme_cache.incomplete_template_color);
				if (any_failed) {
					template_item->add_button(0, theme_cache.failure_icon, (int)ButtonID::NONE);
					template_item->set_button_tooltip_text(0, -1, TTR("Some files have failed to download."));
				}

				if (any_missing && p_is_current_version && !is_downloading() && _can_download_templates()) {
					template_item->add_button(0, theme_cache.repair_icon, (int)ButtonID::REPAIR);
					template_item->set_button_tooltip_text(0, -1, TTR("Download missing template files."));
				}
			}
			if (is_installed_tree) {
				template_item->add_button(0, theme_cache.remove_icon, (int)ButtonID::REMOVE);
				template_item->set_button_tooltip_text(0, -1, TTR("Remove this template."));
			}
			_apply_item_folding(template_item, true);
		}
		_apply_item_folding(platform_item);
	}

	if (p_tree->get_root()->get_child_count() == 0) {
		TreeItem *empty = p_tree->create_item();
		empty->set_text(0, is_available_tree ? TTR("All templates installed.") : TTR("No templates installed."));
		empty->set_custom_color(0, get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
	}
}

void ExportTemplateManager::_update_install_button() {
	if (is_downloading()) {
		install_button->set_text(TTRC("Downloading templates..."));
		install_button->set_disabled(true);
		install_button->set_tooltip_text(String());
		return;
	}

	download_all_enabled = true;
	for (TreeItem *item = available_templates_tree->get_root(); item; item = item->get_next_in_tree()) {
		if (item->is_checked(0)) {
			download_all_enabled = false;
			break;
		}
	}
	if (download_all_enabled) {
		install_button->set_text(TTRC("Install All Templates"));
	} else {
		install_button->set_text(TTRC("Install Selected Templates"));
	}

	install_button->set_disabled(!_can_download_templates());
	if (install_button->is_disabled()) {
		if (mirrors_empty) {
			install_button->set_tooltip_text(TTRC("No mirrors available for download."));
		} else if (!_is_online()) {
			install_button->set_tooltip_text(TTRC("Download not available in offline mode."));
		} else {
			install_button->set_tooltip_text(TTRC("Downloads are only available for the current Godot version."));
		}
	} else {
		install_button->set_tooltip_text(String());
	}
}

bool ExportTemplateManager::_can_download_templates() {
	const String selected_version = version_list->get_item_text(version_list->get_current());
	return !mirrors_empty && _is_online() && selected_version == GODOT_VERSION_FULL_CONFIG;
}

void ExportTemplateManager::_update_folding_cache(TreeItem *p_item) {
	folding_cache[_get_item_path(p_item)] = p_item->is_collapsed();
	if (p_item->get_cell_mode(0) == TreeItem::CELL_MODE_CHECK) {
		if (p_item->is_indeterminate(0)) {
			checked_cache[_get_item_path(p_item)] = 1;
		} else {
			checked_cache[_get_item_path(p_item)] = p_item->is_checked(0) ? 2 : 0;
		}
	}
	for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
		_update_folding_cache(child);
	}
}

String ExportTemplateManager::_get_template_folder_path(const String &p_version) const {
	return EditorPaths::get_singleton()->get_export_templates_dir().path_join(p_version);
}

Ref<Texture2D> ExportTemplateManager::_get_platform_icon(const String &p_platform_name) {
	for (int i = 0; i < EditorExport::get_singleton()->get_export_platform_count(); i++) {
		Ref<EditorExportPlatform> platform = EditorExport::get_singleton()->get_export_platform(i);
		if (platform->get_name() == p_platform_name) {
			return platform->get_logo();
		}
	}
	return Ref<Texture2D>();
}

void ExportTemplateManager::_version_selected() {
	if (!is_downloading()) {
		file_metadata.clear();
		_update_template_tree();
	}
	_update_install_button();
}

void ExportTemplateManager::_tree_button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_button) {
	switch ((ButtonID)p_id) {
		case ButtonID::DOWNLOAD: {
			_install_templates(p_item);
		} break;

		case ButtonID::REPAIR: {
			p_item->set_collapsed(false);
			_install_templates(p_item);
		} break;

		case ButtonID::REMOVE: {
			const String selected_version = version_list->get_item_text(version_list->get_current());
			const String template_directory = _get_template_folder_path(selected_version);

			if (_item_is_file(p_item)) {
				OS::get_singleton()->move_to_trash(template_directory.path_join(p_item->get_text(0)));
				file_metadata.erase(p_item->get_text(0));
			} else {
				for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
					if (!_get_file_metadata(child)->is_missing) {
						OS::get_singleton()->move_to_trash(template_directory.path_join(child->get_text(0)));
					}
					file_metadata.erase(child->get_text(0));
				}
			}
			_update_template_tree();
		} break;

		case ButtonID::CANCEL: {
			if (_item_is_file(p_item)) {
				_cancel_item_download(p_item);
				if (_is_template_download_finished(p_item->get_parent())) {
					queued_templates.erase(p_item->get_parent()->get_text(0));
				}
			} else {
				queued_templates.erase(p_item->get_text(0));
				for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
					if (_get_file_metadata(child)->download_status != DownloadStatus::NONE) {
						_cancel_item_download(child);
					}
				}
			}
			_process_download_queue();
			_update_template_tree();
		} break;

		case ButtonID::FAIL: {
			FileMetadata *meta = _get_file_metadata(p_item);
			EditorNode::get_singleton()->show_warning(meta->fail_reason + ".", TTR("Download Failed"));
		} break;

		case ButtonID::NONE: {
		} break;
	}
}

void ExportTemplateManager::_tree_item_edited() {
	TreeItem *edited = available_templates_tree->get_edited();
	ERR_FAIL_NULL(edited);

	edited->propagate_check(0, false);
	_update_install_button();
}

void ExportTemplateManager::_install_templates(TreeItem *p_files) {
	_queue_download_tree_item(p_files ? p_files : available_templates_tree->get_root());
	download_count = queued_files.size();

	file_metadata.clear();
	_update_template_tree();
	_process_download_queue();
	_update_install_button();
	_update_template_tree_theme(installed_templates_tree);
	_update_template_tree_theme(available_templates_tree);

	ProgressIndicator *indicator = EditorNode::get_bottom_panel()->get_progress_indicator();
	indicator->set_tooltip_text(TTRC("Downloading export templates..."));
	indicator->set_value(0);
	indicator->show();
}

void ExportTemplateManager::_open_template_directory() {
	const String selected_version = version_list->get_item_text(version_list->get_current());
	OS::get_singleton()->shell_show_in_file_manager(_get_template_folder_path(selected_version), true);
}

void ExportTemplateManager::_queue_download_tree_item(TreeItem *p_item) {
	if (_item_is_file(p_item)) {
		bool valid;
		bool is_installed_tree = p_item->get_tree() == installed_templates_tree;
		if (is_installed_tree) {
			FileMetadata *meta = _get_file_metadata(p_item);
			valid = meta->is_missing;
		} else {
			valid = download_all_enabled || p_item->is_checked(0);
		}

		if (valid) {
			queued_files.insert(p_item->get_text(0));
			if (!is_installed_tree) {
				queued_templates.insert(p_item->get_parent()->get_text(0));
			}
		}
	} else {
		for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
			_queue_download_tree_item(child);
		}
	}
}

void ExportTemplateManager::_process_download_queue() {
	queue_update_pending = false;

	int downloader_index = 0;
	bool is_finished = true;
	for (TreeItem *item : downloading_items) {
		FileMetadata *meta = _get_file_metadata(item);

		is_finished = is_finished && _status_is_finished(meta->download_status);
		if (meta->download_status != DownloadStatus::PENDING) {
			continue;
		}

		HTTPRequest *downloader = _get_available_downloader(&downloader_index);
		if (!downloader) {
			break;
		}
		downloader_index++;

		const String filename = item->get_text(0);
		downloader->set_download_file(EditorPaths::get_singleton()->get_cache_dir().path_join(filename));

		Error err = downloader->request(_get_current_mirror_url() + "/" + filename);
		if (err == OK) {
			meta->download_status = DownloadStatus::IN_PROGRESS;
			meta->downloader = downloader;
		} else {
			_item_download_failed(item, TTR(error_names[err]));
		}
	}

	if (is_finished) {
		// Exit "downloading mode".
		queued_templates.clear();
		downloading_items.clear();
		set_process_internal(false);
		_update_template_tree_theme(installed_templates_tree);
		_update_template_tree_theme(available_templates_tree);
		_update_install_button();
		EditorNode::get_bottom_panel()->get_progress_indicator()->hide();
	} else {
		set_process_internal(true);
	}
}

void ExportTemplateManager::_queue_process_download_queue() {
	if (queue_update_pending) {
		return;
	}
	callable_mp(this, &ExportTemplateManager::_process_download_queue).call_deferred();
	queue_update_pending = true;
}

HTTPRequest *ExportTemplateManager::_get_available_downloader(int *r_from_index) {
	int counter = -1;
	for (HTTPRequest *downloader : downloaders) {
		counter++;
		if (counter < *r_from_index) {
			continue;
		}
		if (downloader->get_http_client_status() == HTTPClient::STATUS_DISCONNECTED) {
			*r_from_index = counter;
			return downloader;
		}
	}
	return nullptr;
}

void ExportTemplateManager::_download_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, HTTPRequest *p_downloader) {
	const String filename = p_downloader->get_download_file().get_file();
	bool found = false;
	bool template_finished = false;

	queued_files.erase(filename);
	for (TreeItem *item : downloading_items) {
		if (item->get_text(0) != filename) {
			continue;
		}

		FileMetadata *meta = _get_file_metadata(filename);
		meta->downloader = nullptr;

		if (p_result == HTTPRequest::RESULT_SUCCESS && p_response_code == HTTPClient::RESPONSE_OK) {
			DirAccess::rename_absolute(p_downloader->get_download_file(), _get_template_folder_path(VERSION_FULL_CONFIG).path_join(filename));

			item->clear_buttons();
			meta->download_status = DownloadStatus::COMPLETED;
			meta->is_missing = false;
		} else {
			_item_download_failed(item, _get_download_error(p_result, p_response_code));
		}

		found = true;
		template_finished = _is_template_download_finished(item->get_parent());
		if (template_finished) {
			queued_templates.erase(item->get_parent()->get_text(0));
		}

		break;
	}
	if (!found) {
		ERR_FAIL_COND(!found);
	}
	_queue_process_download_queue();

	if (template_finished) {
		_update_template_tree();
	}
}

bool ExportTemplateManager::_is_template_download_finished(TreeItem *p_template) {
	for (TreeItem *child = p_template->get_first_child(); child; child = child->get_next()) {
		if (!downloading_items.has(child)) {
			continue;
		}
		FileMetadata *meta = _get_file_metadata(child);
		if (!_status_is_finished(meta->download_status)) {
			return false;
		}
	}
	return true;
}

String ExportTemplateManager::_get_download_error(int p_result, int p_response_code) const {
	switch (p_result) {
		case HTTPRequest::RESULT_CANT_RESOLVE:
			return TTR("Can't resolve the requested address");
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH:
		case HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT:
			return TTR("Can't connect to the mirror");
		case HTTPRequest::RESULT_NO_RESPONSE:
			return TTR("No response from the mirror");
		case HTTPRequest::RESULT_REQUEST_FAILED:
			return TTR("Request failed");
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED:
			return TTR("Request ended up in a redirect loop");
	}

	switch (p_response_code) {
		case HTTPClient::RESPONSE_FORBIDDEN:
			return TTR("Forbidden");
		case HTTPClient::RESPONSE_NOT_FOUND:
			return TTR("Not found");
		default: // Handle only common errors.
			return vformat(TTR("Response code: %d"), p_response_code);
	}
}

void ExportTemplateManager::_apply_item_folding(TreeItem *p_item, bool p_default) {
	if (folding_cache.is_empty()) {
		if (p_default) {
			p_item->set_collapsed(true);
		}
	} else {
		bool *cached = folding_cache.getptr(_get_item_path(p_item));
		if (cached) {
			p_item->set_collapsed(*cached);
		} else if (p_default) {
			p_item->set_collapsed(true);
		}
	}
}

void ExportTemplateManager::_cancel_item_download(TreeItem *p_item) {
	_item_download_failed(p_item, TTR("Canceled by the user"));
	queued_files.erase(p_item->get_text(0));

	FileMetadata *meta = _get_file_metadata(p_item);
	if (meta->downloader) {
		meta->downloader->cancel_request();
		meta->downloader = nullptr;
	}
}

void ExportTemplateManager::_item_download_failed(TreeItem *p_item, const String &p_reason) {
	FileMetadata *meta = _get_file_metadata(p_item);
	meta->fail_reason = p_reason;
	meta->download_status = DownloadStatus::FAILED;

	p_item->clear_buttons();
	_add_fail_reason_button(p_item);
}

void ExportTemplateManager::_add_fail_reason_button(TreeItem *p_item, const String &p_filename) {
	FileMetadata *meta = _get_file_metadata(p_filename.is_empty() ? p_item->get_text(0) : p_filename);
	p_item->add_button(0, theme_cache.failure_icon, (int)ButtonID::FAIL);
	p_item->set_button_tooltip_text(0, -1, vformat(TTR("Download failed.\nReason: %s."), meta->fail_reason));
}

void ExportTemplateManager::_set_item_type(TreeItem *p_item, int p_type) {
	switch (p_type) {
		case TreeItem::CELL_MODE_CHECK: {
			p_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
			p_item->set_editable(0, true);
		} break;

		case TreeItem::CELL_MODE_CUSTOM: {
			p_item->set_cell_mode(0, TreeItem::CELL_MODE_CUSTOM);
			p_item->set_custom_draw_callback(0, callable_mp(this, &ExportTemplateManager::_draw_item_progress));
		} break;
	}
}

void ExportTemplateManager::_setup_item_text(TreeItem *p_item, const String &p_text) {
	if (p_item == p_item->get_tree()->get_root()) {
		if (p_item->get_tree() == installed_templates_tree) {
			p_item->set_meta(PATH_META, "installed/");
		} else {
			p_item->set_meta(PATH_META, "available/");
		}
	} else {
		p_item->set_text(0, p_text);
		const String path = p_item->get_parent()->get_meta(PATH_META).operator String() + p_text;
		p_item->set_meta(PATH_META, path);

		if (p_item->get_cell_mode(0) == TreeItem::CELL_MODE_CHECK) {
			int *checked = checked_cache.getptr(path);
			if (checked) {
				if (*checked == 1) {
					p_item->set_indeterminate(0, true);
				} else {
					p_item->set_checked(0, *checked == 2);
				}
			}
		}
	}
}

ExportTemplateManager::FileMetadata *ExportTemplateManager::_get_file_metadata(const String &p_text) const {
	FileMetadata *meta = file_metadata.getptr(p_text);
	if (likely(meta)) {
		return meta;
	}
	HashMap<String, FileMetadata>::Iterator it = file_metadata.insert(p_text, FileMetadata());
	return &it->value;
}

ExportTemplateManager::FileMetadata *ExportTemplateManager::_get_file_metadata(const TreeItem *p_item) const {
	return _get_file_metadata(p_item->get_text(0));
}

String ExportTemplateManager::_get_item_path(TreeItem *p_item) const {
	return p_item->get_meta(PATH_META, String());
}

bool ExportTemplateManager::_item_is_file(TreeItem *p_item) const {
	return p_item->get_meta(FILE_META, false).operator bool();
}

float ExportTemplateManager::_get_download_progress(const TreeItem *p_item) const {
	FileMetadata *meta = _get_file_metadata(p_item);
	switch (meta->download_status) {
		case DownloadStatus::NONE:
		case DownloadStatus::PENDING: {
			return 0.0;
		}

		case DownloadStatus::IN_PROGRESS: {
			if (!meta->downloader) {
				return 0.0;
			}
			return (float)meta->downloader->get_downloaded_bytes() / (float)meta->downloader->get_body_size();
		}

		case DownloadStatus::COMPLETED: {
			return 1.0;
		}

		case DownloadStatus::FAILED: {
			return meta->progress_cache;
		}
	}
	return 0.0;
}

void ExportTemplateManager::_draw_item_progress(TreeItem *p_item, const Rect2 &p_rect) {
	Tree *owning_tree = p_item->get_tree();
	owning_tree->draw_rect(p_rect, Color(0, 0, 0, 0.5));

	if (!_item_is_file(p_item)) {
		float progress = 0.0;
		int item_count = 0;

		bool has_fail = false;
		for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
			if (!downloading_items.has(child)) {
				continue;
			}
			item_count++;
			progress += _get_download_progress(child);

			FileMetadata *meta = _get_file_metadata(child);
			has_fail = has_fail || meta->download_status == DownloadStatus::FAILED;
		}
		progress /= item_count;
		owning_tree->draw_rect(Rect2(p_rect.position, Vector2(p_rect.size.x * progress, p_rect.size.y)), has_fail ? theme_cache.download_failed_color : theme_cache.download_progress_color);
		return;
	}

	FileMetadata *meta = _get_file_metadata(p_item);
	switch (meta->download_status) {
		case DownloadStatus::NONE: {
		} break;

		case DownloadStatus::PENDING: {
			uint64_t frame = Engine::get_singleton()->get_frames_drawn();
			const Ref<Texture2D> progress_texture = theme_cache.progress_icons[frame / 4 % 8];
			owning_tree->draw_texture(progress_texture, Vector2(p_rect.get_end().x - progress_texture->get_width(), p_rect.position.y + p_rect.size.y * 0.5 - progress_texture->get_height() * 0.5));
		} break;

		case DownloadStatus::IN_PROGRESS: {
			float progress = _get_download_progress(p_item);
			meta->progress_cache = progress;
			owning_tree->draw_rect(Rect2(p_rect.position, Vector2(p_rect.size.x * progress, p_rect.size.y)), theme_cache.download_progress_color);
		} break;

		case DownloadStatus::COMPLETED: {
			owning_tree->draw_rect(p_rect, theme_cache.download_progress_color);
		} break;

		case DownloadStatus::FAILED: {
			owning_tree->draw_rect(Rect2(p_rect.position, Vector2(p_rect.size.x * _get_download_progress(p_item), p_rect.size.y)), theme_cache.download_failed_color);
		} break;
	}
}

void ExportTemplateManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			EditorNode::get_bottom_panel()->get_progress_indicator()->connect("clicked", callable_mp(this, &ExportTemplateManager::popup_manager));
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (template_data.is_empty()) {
				break;
			}
			platform_map[PlatformID::WINDOWS].group = TTR("Desktop", "Platform Group");
			platform_map[PlatformID::LINUX].group = TTR("Desktop", "Platform Group");
			platform_map[PlatformID::MACOS].group = TTR("Desktop", "Platform Group");
			platform_map[PlatformID::WEB].group = TTR("Web", "Platform Group");
			platform_map[PlatformID::ANDROID].group = TTR("Mobile", "Platform Group");
			platform_map[PlatformID::IOS].group = TTR("Mobile", "Platform Group");
			platform_map[PlatformID::COMMON].name = TTR("Common");
			template_data[TemplateID::WEB_EXTENSIONS].name = TTR("Web with Extensions");
			template_data[TemplateID::WEB_NOTHREADS].name = TTR("Web Single-Threaded");
			template_data[TemplateID::WEB_EXTENSIONS_NOTHREADS].name = TTR("Web with Extensions Single-Threaded");
			template_data[TemplateID::ANDROID_SOURCE].name = TTR("Android Source");
			template_data[TemplateID::ANDROID_SOURCE].name = TTR("ICU Data");
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			open_folder_button->set_button_icon(get_editor_theme_icon("Folder"));
			install_button->set_button_icon(get_editor_theme_icon("AssetLib"));
			open_mirror->set_button_icon(get_editor_theme_icon("ExternalLink"));

			theme_cache.install_icon = get_editor_theme_icon("AssetLib");
			theme_cache.remove_icon = get_editor_theme_icon("Remove");
			theme_cache.repair_icon = get_editor_theme_icon("Tools");
			theme_cache.failure_icon = get_editor_theme_icon("NodeWarning");
			theme_cache.cancel_icon = get_editor_theme_icon("Close");
			for (int i = 0; i < 8; i++) {
				theme_cache.progress_icons[i] = get_editor_theme_icon("Progress" + itos(i + 1));
			}

			theme_cache.current_version_color = get_theme_color("accent_color", EditorStringName(Editor));
			theme_cache.incomplete_template_color = get_theme_color("warning_color", EditorStringName(Editor));
			theme_cache.missing_file_color = get_theme_color("error_color", EditorStringName(Editor));
			theme_cache.download_progress_color = Color(get_theme_color("success_color", EditorStringName(Editor)), 0.5);
			theme_cache.download_failed_color = Color(theme_cache.missing_file_color, 0.5);

			theme_cache.icon_width = get_theme_constant("class_icon_size", EditorStringName(Editor));
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			available_templates_tree->queue_redraw();
			installed_templates_tree->queue_redraw();

			float progress = 0.0;
			int indeterminate_count = download_count;
			for (const TreeItem *item : downloading_items) {
				progress += _get_download_progress(item);
				indeterminate_count--;
			}
			progress += indeterminate_count;
			EditorNode::get_bottom_panel()->get_progress_indicator()->set_value(progress / download_count);
		}
	}
}

String ExportTemplateManager::get_android_build_directory(const Ref<EditorExportPreset> &p_preset) {
	if (p_preset.is_valid()) {
		String gradle_build_dir = p_preset->get("gradle_build/gradle_build_directory");
		if (!gradle_build_dir.is_empty()) {
			return gradle_build_dir.path_join("build");
		}
	}
	return "res://android/build";
}

String ExportTemplateManager::get_android_source_zip(const Ref<EditorExportPreset> &p_preset) {
	if (p_preset.is_valid()) {
		String android_source_zip = p_preset->get("gradle_build/android_source_template");
		if (!android_source_zip.is_empty()) {
			return android_source_zip;
		}
	}

	const String templates_dir = EditorPaths::get_singleton()->get_export_templates_dir().path_join(GODOT_VERSION_FULL_CONFIG);
	return templates_dir.path_join("android_source.zip");
}

String ExportTemplateManager::get_android_template_identifier(const Ref<EditorExportPreset> &p_preset) {
	// The template identifier is the Godot version for the default template, and the full path plus md5 hash for custom templates.
	if (p_preset.is_valid()) {
		String android_source_zip = p_preset->get("gradle_build/android_source_template");
		if (!android_source_zip.is_empty()) {
			return android_source_zip + String(" [") + FileAccess::get_md5(android_source_zip) + String("]");
		}
	}
	return GODOT_VERSION_FULL_CONFIG;
}

bool ExportTemplateManager::is_android_template_installed(const Ref<EditorExportPreset> &p_preset) {
	return DirAccess::exists(get_android_build_directory(p_preset));
}

bool ExportTemplateManager::can_install_android_template(const Ref<EditorExportPreset> &p_preset) {
	return FileAccess::exists(get_android_source_zip(p_preset));
}

Error ExportTemplateManager::install_android_template(const Ref<EditorExportPreset> &p_preset) {
	const String source_zip = get_android_source_zip(p_preset);
	ERR_FAIL_COND_V(!FileAccess::exists(source_zip), ERR_CANT_OPEN);
	return install_android_template_from_file(source_zip, p_preset);
}

Error ExportTemplateManager::install_android_template_from_file(const String &p_file, const Ref<EditorExportPreset> &p_preset) {
	// To support custom Android builds, we install the Java source code and buildsystem
	// from android_source.zip to the project's res://android folder.

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	ERR_FAIL_COND_V(da.is_null(), ERR_CANT_CREATE);

	String build_dir = get_android_build_directory(p_preset);
	String parent_dir = build_dir.get_base_dir();

	// Make parent of the build dir (if it does not exist).
	da->make_dir_recursive(parent_dir);
	{
		// Add identifier, to ensure building won't work if the current template doesn't match.
		Ref<FileAccess> f = FileAccess::open(parent_dir.path_join(".build_version"), FileAccess::WRITE);
		ERR_FAIL_COND_V(f.is_null(), ERR_CANT_CREATE);
		f->store_line(get_android_template_identifier(p_preset));
	}

	// Create the android build directory.
	Error err = da->make_dir_recursive(build_dir);
	ERR_FAIL_COND_V(err != OK, err);
	{
		// Add an empty .gdignore file to avoid scan.
		Ref<FileAccess> f = FileAccess::open(build_dir.path_join(".gdignore"), FileAccess::WRITE);
		ERR_FAIL_COND_V(f.is_null(), ERR_CANT_CREATE);
		f->store_line("");
	}

	// Uncompress source template.

	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
	ERR_FAIL_NULL_V_MSG(pkg, ERR_CANT_OPEN, "Android sources not in ZIP format.");

	int ret = unzGoToFirstFile(pkg);
	int total_files = 0;
	// Count files to unzip.
	while (ret == UNZ_OK) {
		total_files++;
		ret = unzGoToNextFile(pkg);
	}
	ret = unzGoToFirstFile(pkg);

	ProgressDialog::get_singleton()->add_task("uncompress_src", TTR("Uncompressing Android Build Sources"), total_files);

	HashSet<String> dirs_tested;
	int idx = 0;
	while (ret == UNZ_OK) {
		// Get file path.
		unz_file_info info;
		char fpath[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fpath, 16384, nullptr, 0, nullptr, 0);
		if (ret != UNZ_OK) {
			break;
		}

		String path = String::utf8(fpath);
		String base_dir = path.get_base_dir();

		if (!path.ends_with("/")) {
			Vector<uint8_t> uncomp_data;
			uncomp_data.resize(info.uncompressed_size);

			// Read.
			unzOpenCurrentFile(pkg);
			unzReadCurrentFile(pkg, uncomp_data.ptrw(), uncomp_data.size());
			unzCloseCurrentFile(pkg);

			if (!dirs_tested.has(base_dir)) {
				da->make_dir_recursive(build_dir.path_join(base_dir));
				dirs_tested.insert(base_dir);
			}

			String to_write = build_dir.path_join(path);
			Ref<FileAccess> f = FileAccess::open(to_write, FileAccess::WRITE);
			if (f.is_valid()) {
				f->store_buffer(uncomp_data.ptr(), uncomp_data.size());
				f.unref(); // close file.
#ifndef WINDOWS_ENABLED
				FileAccess::set_unix_permissions(to_write, (info.external_fa >> 16) & 0x01FF);
#endif
			} else {
				ERR_PRINT("Can't uncompress file: " + to_write);
			}
		}

		ProgressDialog::get_singleton()->task_step("uncompress_src", path, idx);

		idx++;
		ret = unzGoToNextFile(pkg);
	}

	ProgressDialog::get_singleton()->end_task("uncompress_src");
	unzClose(pkg);
	EditorFileSystem::get_singleton()->scan_changes();
	return OK;
}

void ExportTemplateManager::popup_manager() {
	if (template_data.is_empty()) {
		_initialize_template_data();
	}
	_update_online_mode();

	if (!is_downloading()) {
		_update_template_tree();
		_request_mirrors();
	}
	popup_centered_clamped(Vector2i(640, 700) * EDSCALE);
}

bool ExportTemplateManager::is_downloading() const {
	return !queued_files.is_empty();
}

void ExportTemplateManager::stop_download() {
	for (TreeItem *item : downloading_items) {
		FileMetadata *meta = _get_file_metadata(item);
		if (meta && !_status_is_finished(meta->download_status)) {
			_cancel_item_download(item);
		}
	}
}

ExportTemplateManager::ExportTemplateManager() {
	set_title(TTRC("Export Template Manager"));
	set_ok_button_text(TTRC("Close"));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	HBoxContainer *download_header = memnew(HBoxContainer);
	download_header->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
	main_vb->add_child(download_header);

	download_header->add_child(memnew(Label(TTRC("Download from:"))));

	mirrors_list = memnew(OptionButton);
	mirrors_list->set_accessibility_name(TTRC("Mirror"));
	download_header->add_child(mirrors_list);

	open_mirror = memnew(Button);
	open_mirror->set_tooltip_text(TTRC("Open in Web Browser"));
	download_header->add_child(open_mirror);
	open_mirror->connect(SceneStringName(pressed), callable_mp(this, &ExportTemplateManager::_open_mirror));

	install_button = memnew(Button);
	install_button->set_h_size_flags(Control::SIZE_SHRINK_END | Control::SIZE_EXPAND);
	download_header->add_child(install_button);
	install_button->connect(SceneStringName(pressed), callable_mp(this, &ExportTemplateManager::_install_templates).bind((TreeItem *)nullptr));

	HSplitContainer *main_split = memnew(HSplitContainer);
	main_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(main_split);

	VBoxContainer *side_vb = memnew(VBoxContainer);
	main_split->add_child(side_vb);

	Label *version_header = memnew(Label(TTRC("Godot Version")));
	version_header->set_theme_type_variation("HeaderSmall");
	side_vb->add_child(version_header);

	version_list = memnew(ItemList);
	version_list->set_accessibility_name(TTRC("Godot Version List"));
	version_list->set_theme_type_variation("ItemListSecondary");
	version_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	side_vb->add_child(version_list);
	version_list->connect(SceneStringName(item_selected), callable_mp(this, &ExportTemplateManager::_version_selected).unbind(1));

	open_folder_button = memnew(Button);
	open_folder_button->set_tooltip_text(TTRC("Open templates directory."));
	open_folder_button->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
	side_vb->add_child(open_folder_button);
	open_folder_button->connect(SceneStringName(pressed), callable_mp(this, &ExportTemplateManager::_open_template_directory));

	VSplitContainer *center_split = memnew(VSplitContainer);
	center_split->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_split->add_child(center_split);

	VBoxContainer *available_templates_container = memnew(VBoxContainer);
	available_templates_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	center_split->add_child(available_templates_container);

	Label *template_header2 = memnew(Label(TTRC("Available Templates")));
	template_header2->set_theme_type_variation("HeaderSmall");
	template_header2->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	available_templates_container->add_child(template_header2);

	available_templates_tree = memnew(Tree);
	available_templates_tree->set_accessibility_name(TTRC("Available Templates"));
	available_templates_tree->set_hide_root(true);
	available_templates_tree->set_theme_type_variation("TreeSecondary");
	available_templates_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	available_templates_tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	available_templates_container->add_child(available_templates_tree);
	available_templates_tree->connect("button_clicked", callable_mp(this, &ExportTemplateManager::_tree_button_clicked));
	available_templates_tree->connect("item_edited", callable_mp(this, &ExportTemplateManager::_tree_item_edited));

	VBoxContainer *installed_templates_container = memnew(VBoxContainer);
	installed_templates_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	center_split->add_child(installed_templates_container);

	Label *template_header = memnew(Label(TTRC("Installed Templates")));
	template_header->set_theme_type_variation("HeaderSmall");
	installed_templates_container->add_child(template_header);

	installed_templates_tree = memnew(Tree);
	installed_templates_tree->set_accessibility_name(TTRC("Installed Templates"));
	installed_templates_tree->set_hide_root(true);
	installed_templates_tree->set_theme_type_variation("TreeSecondary");
	installed_templates_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	installed_templates_tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	installed_templates_container->add_child(installed_templates_tree);
	installed_templates_tree->connect("button_clicked", callable_mp(this, &ExportTemplateManager::_tree_button_clicked));

	offline_container = memnew(HBoxContainer);
	offline_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	offline_container->hide();
	main_vb->add_child(offline_container);

	Label *offline_mode_label = memnew(Label(TTRC("Offline mode, some functionality is not available.")));
	offline_container->add_child(offline_mode_label);

	LinkButton *enable_online_button = memnew(LinkButton);
	enable_online_button->set_text(TTRC("Go Online"));
	offline_container->add_child(enable_online_button);
	enable_online_button->connect(SceneStringName(pressed), callable_mp(this, &ExportTemplateManager::_force_online_mode));

	mirrors_requester = memnew(HTTPRequest);
	mirrors_requester->connect("request_completed", callable_mp(this, &ExportTemplateManager::_mirrors_request_completed));
	add_child(mirrors_requester);

	for (int i = 0; i < 5; i++) {
		HTTPRequest *downloader = memnew(HTTPRequest);
		downloader->set_use_threads(true);
		add_child(downloader);
		downloaders.push_back(downloader);
		downloader->connect("request_completed", callable_mp(this, &ExportTemplateManager::_download_request_completed).bind(downloader), CONNECT_DEFERRED);
	}
}
