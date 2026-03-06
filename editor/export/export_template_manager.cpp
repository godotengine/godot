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
#include "core/io/dir_access.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "core/version.h"
#include "editor/editor_string_names.h"
#include "editor/export/editor_export.h"
#include "editor/file_system/editor_paths.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"
#include "scene/main/http_request.h"
#include "scene/resources/texture.h"

void ExportTemplateManager::_initialize_template_data() {
	// Base templates.
	{
		TemplateInfo info;
		info.name = "Windows x86-32";
		info.file_list = { "windows_debug_x86_32.exe", "windows_debug_x86_32_console.exe", "windows_release_x86_32.exe", "windows_release_x86_32_console.exe" };
		template_data[TemplateID::WINDOWS_X86_32] = info;
	}
	{
		TemplateInfo info;
		info.name = "Windows x86-64";
		info.file_list = { "windows_debug_x86_64.exe", "windows_debug_x86_64_console.exe", "windows_release_x86_64.exe", "windows_release_x86_64_console.exe" };
		template_data[TemplateID::WINDOWS_X86_64] = info;
	}
	{
		TemplateInfo info;
		info.name = "Windows ARM-64";
		info.file_list = { "windows_debug_arm64.exe", "windows_debug_arm64_console.exe", "windows_release_arm64.exe", "windows_release_arm64_console.exe" };
		template_data[TemplateID::WINDOWS_ARM64] = info;
	}

	{
		TemplateInfo info;
		info.name = "Linux x86-32";
		info.file_list = { "linux_debug.x86_32", "linux_release.x86_32" };
		template_data[TemplateID::LINUX_X86_32] = info;
	}
	{
		TemplateInfo info;
		info.name = "Linux x86-64";
		info.file_list = { "linux_debug.x86_64", "linux_release.x86_64" };
		template_data[TemplateID::LINUX_X86_64] = info;
	}
	{
		TemplateInfo info;
		info.name = "Linux ARM-32";
		info.file_list = { "linux_debug.arm32", "linux_release.arm32" };
		template_data[TemplateID::LINUX_ARM32] = info;
	}
	{
		TemplateInfo info;
		info.name = "Linux ARM-64";
		info.file_list = { "linux_debug.arm64", "linux_release.arm64" };
		template_data[TemplateID::LINUX_ARM64] = info;
	}

	{
		TemplateInfo info;
		info.name = "macOS";
		info.file_list = { "macos.zip" };
		template_data[TemplateID::MACOS] = info;
	}

	{
		TemplateInfo info;
		info.name = "Web";
		info.file_list = { "web_debug.zip", "web_release.zip" };
		template_data[TemplateID::WEB] = info;
	}
	{
		TemplateInfo info;
		info.name = "Web with Extensions";
		info.file_list = { "web_dlink_debug.zip", "web_dlink_release.zip" };
		template_data[TemplateID::WEB_EXTENSIONS] = info;
	}
	{
		TemplateInfo info;
		info.name = "Web Single-Threaded";
		info.file_list = { "web_nothreads_debug.zip", "web_nothreads_release.zip" };
		template_data[TemplateID::WEB_NOTHREADS] = info;
	}
	{
		TemplateInfo info;
		info.name = "Web with Extensions Single-Threaded";
		info.file_list = { "web_dlink_nothreads_debug.zip", "web_dlink_nothreads_release.zip" };
		template_data[TemplateID::WEB_EXTENSIONS_NOTHREADS] = info;
	}

	{
		TemplateInfo info;
		info.name = "Android";
		info.file_list = { "android_debug.apk", "android_release.apk" };
		template_data[TemplateID::ANDROID] = info;
	}
	{
		TemplateInfo info;
		info.name = "Android Source";
		info.file_list = { "android_source.zip" };
		template_data[TemplateID::ANDROID_SOURCE] = info;
	}

	{
		TemplateInfo info;
		info.name = "iOS";
		info.file_list = { "ios.zip" };
		template_data[TemplateID::IOS] = info;
	}

	{
		TemplateInfo info;
		info.name = "ICU Data";
		info.file_list = { "icudt_godot.dat" };
		template_data[TemplateID::ICU_DATA] = info;
	}

	// Platforms.
	{
		PlatformInfo info;
		info.name = "Windows";
		info.icon = _get_platform_icon("Windows Desktop");
		info.templates = { TemplateID::WINDOWS_X86_32, TemplateID::WINDOWS_X86_64, TemplateID::WINDOWS_ARM64 };
		info.group = "Desktop";
		platform_map[PlatformID::WINDOWS] = info;
	}
	{
		PlatformInfo info;
		info.name = "Linux";
		info.icon = _get_platform_icon("Linux");
		info.templates = { TemplateID::LINUX_X86_32, TemplateID::LINUX_X86_64, TemplateID::LINUX_ARM32, TemplateID::LINUX_ARM64 };
		info.group = "Desktop";
		platform_map[PlatformID::LINUX] = info;
	}
	{
		PlatformInfo info;
		info.name = "macOS";
		info.icon = _get_platform_icon("macOS");
		info.templates = { TemplateID::MACOS };
		info.group = "Desktop";
		platform_map[PlatformID::MACOS] = info;
	}
	{
		PlatformInfo info;
		info.name = "Android";
		info.icon = _get_platform_icon("Android");
		info.templates = { TemplateID::ANDROID, TemplateID::ANDROID_SOURCE };
		info.group = "Mobile";
		platform_map[PlatformID::ANDROID] = info;
	}
	{
		PlatformInfo info;
		info.name = "iOS";
		info.icon = _get_platform_icon("iOS");
		info.templates = { TemplateID::IOS };
		info.group = "Mobile";
		platform_map[PlatformID::IOS] = info;
	}
	{
		PlatformInfo info;
		info.name = "Web";
		info.icon = _get_platform_icon("Web");
		info.templates = { TemplateID::WEB, TemplateID::WEB_EXTENSIONS, TemplateID::WEB_NOTHREADS, TemplateID::WEB_EXTENSIONS_NOTHREADS };
		info.group = "Web";
		platform_map[PlatformID::ANDROID] = info;
	}
	{
		PlatformInfo info;
		info.name = "Common";
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
			version_list->select(version_list->get_item_count() - 1);
			version_list->set_item_custom_fg_color(-1, get_theme_color("accent_color", EditorStringName(Editor)));
		} else {
			version_list->add_item(dir);
		}
		version_list->set_item_metadata(-1, dir);
	}
}

void ExportTemplateManager::_update_template_tree() {
	const int icon_width = get_theme_constant("class_icon_size", EditorStringName(Editor));
	const Color incomplete_template_color = get_theme_color("warning_color", EditorStringName(Editor));
	const Color missing_file_color = get_theme_color("error_color", EditorStringName(Editor));

	Ref<Texture2D> install_icon = get_editor_theme_icon("AssetLib");
	Ref<Texture2D> remove_icon = get_editor_theme_icon("Remove");
	Ref<Texture2D> repair_icon = get_editor_theme_icon("Tools");

	const String selected_version = version_list->get_item_text(version_list->get_current());
	Ref<DirAccess> template_directory = DirAccess::open(_get_template_folder_path(selected_version));
	ERR_FAIL_COND(template_directory.is_null());

	bool is_current_version = (selected_version == GODOT_VERSION_FULL_CONFIG);
	HashMap<TemplateID, LocalVector<String>> installed_template_files;

	// List installed templates.
	installed_templates_tree->clear();
	TreeItem *platform_parent = installed_templates_tree->create_item();

	for (const KeyValue<PlatformID, PlatformInfo> &KV : platform_map) {
		LocalVector<String> installed_files;

		for (TemplateID id : KV.value.templates) {
			for (const String &file : template_data[id].file_list) {
				if (template_directory->file_exists(file)) {
					installed_files.push_back(file);
					installed_template_files[id].push_back(file);
				}
			}
		}

		if (installed_files.is_empty()) {
			// Nothing installed, skip platform.
			continue;
		}

		TreeItem *platform_item = installed_templates_tree->create_item();
		platform_item->set_text(0, KV.value.name);
		platform_item->set_icon(0, KV.value.icon);
		platform_item->set_icon_max_width(0, icon_width);

		for (TemplateID id : KV.value.templates) {
			if (!installed_template_files.has(id)) {
				continue;
			}

			TreeItem *template_item = platform_item->create_child();
			template_item->set_text(0, template_data[id].name);

			bool any_missing = false;
			for (const String &file : template_data[id].file_list) {
				Dictionary item_metadata;
				item_metadata["is_file"] = true;

				TreeItem *file_item = template_item->create_child();
				file_item->set_metadata(0, item_metadata);

				if (queued_files.has(file)) {
					_setup_downloading_item(file_item, file);
				} else {
					file_item->set_text(0, file);

					if (installed_files.has(file)) {
						file_item->add_button(0, remove_icon, (int)ButtonID::REMOVE);
						file_item->set_button_tooltip_text(0, -1, TTRC("Remove this file."));
					} else {
						file_item->set_custom_color(0, missing_file_color);
						if (is_current_version) {
							file_item->add_button(0, install_icon, (int)ButtonID::DOWNLOAD);
							file_item->set_button_tooltip_text(0, -1, TTRC("Download this missing file."));
						}
						item_metadata["is_missing"] = true;
						any_missing = true;
					}
				}
			}
			_apply_item_folding(template_item, true);

			if (any_missing) {
				template_item->set_custom_color(0, incomplete_template_color);

				if (is_current_version) {
					template_item->add_button(0, repair_icon, (int)ButtonID::REPAIR);
					template_item->set_button_tooltip_text(0, -1, TTRC("Download missing template files."));
				}
			}
			template_item->add_button(0, remove_icon, (int)ButtonID::REMOVE);
			template_item->set_button_tooltip_text(0, -1, TTRC("Remove this template."));
		}
		_apply_item_folding(platform_item);
	}

	if (!is_current_version) {
		available_templates_container->hide();
		return;
	}
	available_templates_container->show();

	// List non-installed templates, available for download.
	available_templates_tree->clear();
	platform_parent = available_templates_tree->create_item();
	String current_group;

	for (const KeyValue<PlatformID, PlatformInfo> &KV : platform_map) {
		const PlatformInfo &template_platform = KV.value;

		bool all_installed = true;
		for (TemplateID id : template_platform.templates) {
			if (!installed_template_files.has(id)) {
				all_installed = false;
				break;
			}
		}

		if (all_installed) {
			// All is installed, skip platform.
			continue;
		}

		if (template_platform.group != current_group) {
			_apply_item_folding(platform_parent);
			current_group = template_platform.group;

			if (current_group.is_empty()) {
				platform_parent = available_templates_tree->get_root();
			} else {
				platform_parent = available_templates_tree->create_item();
				if (!_is_downloading()) {
					_setup_check_item(platform_parent, current_group);
				} else {
					platform_parent->set_text(0, current_group);
				}
			}
		}

		TreeItem *platform_item = platform_parent->create_child();
		if (!_is_downloading()) {
			_setup_check_item(platform_item, template_platform.name);
		} else {
			platform_item->set_text(0, template_platform.name);
		}
		platform_item->set_icon(0, template_platform.icon);
		platform_item->set_icon_max_width(0, icon_width);

		for (TemplateID id : template_platform.templates) {
			if (installed_template_files.has(id)) {
				continue;
			}
			TemplateInfo &template_info = template_data[id];

			TreeItem *template_item = platform_item->create_child();
			if (queued_templates.has(template_info.name)) {
				_setup_custom_item(template_item, template_info.name);
				template_item->add_button(0, get_editor_theme_icon(SNAME("Close")), (int)ButtonID::CANCEL);
				template_item->set_button_tooltip_text(0, -1, TTRC("Cancel downloading this template."));
			} else {
				_setup_check_item(template_item, template_info.name);
			}

			for (const String &file : template_info.file_list) {
				Dictionary item_metadata;
				item_metadata["is_file"] = true;

				TreeItem *file_item = template_item->create_child();
				file_item->set_metadata(0, item_metadata);

				if (queued_files.has(file)) {
					_setup_downloading_item(file_item, file);
				} else {
					_setup_check_item(file_item, file);
				}
			}
			_apply_item_folding(template_item, true);
		}
		_apply_item_folding(platform_item);
	}

	if (installed_templates_tree->get_root()->get_child_count() == 0) {
		TreeItem *empty = installed_templates_tree->create_item();
		empty->set_text(0, TTRC("No templates installed."));
		empty->set_custom_color(0, get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
	}
	if (available_templates_tree->get_root()->get_child_count() == 0) {
		TreeItem *empty = available_templates_tree->create_item();
		empty->set_text(0, TTRC("All templates installed."));
		empty->set_custom_color(0, get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
		install_button->hide();
	} else {
		install_button->set_visible(!_is_downloading());
		if (install_button->is_visible()) {
			_update_install_button_text();
		}
	}
}

void ExportTemplateManager::_update_template_tree_with_folding() {
	_update_folding_cache(folding_cache_installed, installed_templates_tree->get_root());
	_update_folding_cache(folding_cache_available, available_templates_tree->get_root());

	_update_template_tree();

	folding_cache_installed.clear();
	folding_cache_available.clear();
}

void ExportTemplateManager::_update_install_button_text() {
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
}

void ExportTemplateManager::_update_folding_cache(HashMap<String, bool> &p_cache, TreeItem *p_item) {
	p_cache[p_item->get_text(0)] = p_item->is_collapsed();
	for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
		_update_folding_cache(p_cache, child);
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
	if (!_is_downloading()) {
		_update_template_tree();
	}
}

void ExportTemplateManager::_tree_button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_button) {
	switch ((ButtonID)p_id) {
		case ButtonID::DOWNLOAD: {
			_queue_download_tree_item(p_item);
			_update_template_tree_with_folding();
			_process_download_queue();
		} break;

		case ButtonID::REPAIR: {
			for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
				if (child->get_metadata(0).operator Dictionary().has("is_missing")) {
					_queue_download_tree_item(child);
				}
			}
			_update_template_tree_with_folding();
			_process_download_queue();
		} break;

		case ButtonID::REMOVE: {
			const String selected_version = version_list->get_item_text(version_list->get_current());
			const String template_directory = _get_template_folder_path(selected_version);

			if (_item_is_file(p_item)) {
				OS::get_singleton()->move_to_trash(template_directory.path_join(p_item->get_text(0)));
			} else {
				for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
					if (!child->get_metadata(0).operator Dictionary().has("is_missing")) {
						OS::get_singleton()->move_to_trash(template_directory.path_join(child->get_text(0)));
					}
				}
			}
			_update_template_tree_with_folding();
		} break;

		case ButtonID::CANCEL: {
			if (_item_is_file(p_item)) {
				_fail_item_download(p_item, TTRC("Canceled by the user."));
			} else {
				for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
					if (child->get_metadata(0).operator Dictionary().has("download_status")) {
						_fail_item_download(p_item, TTRC("Canceled by the user."));
					}
				}
			}
			_process_download_queue();
		} break;
	}
}

void ExportTemplateManager::_tree_item_edited() {
	TreeItem *edited = available_templates_tree->get_edited();
	ERR_FAIL_NULL(edited);

	edited->propagate_check(0, false);
	_update_install_button_text();
}

void ExportTemplateManager::_install_templates() {
	_queue_download_tree_item(available_templates_tree->get_root());
	_update_template_tree_with_folding();
	_process_download_queue();
}

void ExportTemplateManager::_open_template_directory() {
	const String selected_version = version_list->get_item_text(version_list->get_current());
	OS::get_singleton()->shell_show_in_file_manager(_get_template_folder_path(selected_version), true);
}

void ExportTemplateManager::_queue_download_tree_item(TreeItem *p_item) {
	if (_item_is_file(p_item)) {
		if (download_all_enabled || p_item->is_checked(0) || p_item->get_tree() == installed_templates_tree) {
			queued_files.insert(p_item->get_text(0));
			queued_templates.insert(p_item->get_parent()->get_text(0));
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
		Dictionary item_metadata = item->get_metadata(0);
		DownloadStatus status = item_metadata["download_status"];
		is_finished = is_finished && status == DownloadStatus::COMPLETED;

		if (status != DownloadStatus::PENDING) {
			continue;
		}

		HTTPRequest *downloader = _get_available_downloader(&downloader_index);
		if (!downloader) {
			break;
		}
		downloader_index++;

		const String filename = item->get_text(0);
		downloader->set_download_file(_get_template_folder_path(GODOT_VERSION_FULL_CONFIG).path_join(filename));

		Error err = downloader->request(vformat("http://127.0.0.1:8000/%s", filename));
		if (err == OK) {
			item_metadata["download_status"] = DownloadStatus::IN_PROGRESS;
			item_metadata["downloader"] = downloader;
		}
	}

	if (is_finished) {
		queued_files.clear();
		queued_templates.clear();
		downloading_items.clear();
		set_process_internal(false);
		_update_template_tree();
	} else {
		set_process_internal(true);
		available_templates_tree->queue_redraw();
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
	for (TreeItem *item : downloading_items) {
		if (item->get_text(0) == filename) {
			item->clear_buttons();

			Dictionary item_metadata = item->get_metadata(0);
			item_metadata.erase("downloader");
			item_metadata["download_status"] = DownloadStatus::COMPLETED;
			found = true;
			break;
		}
	}
	ERR_FAIL_COND(!found);
	_queue_process_download_queue();
}

bool ExportTemplateManager::_is_downloading() const {
	return !queued_files.is_empty();
}

void ExportTemplateManager::_setup_custom_item(TreeItem *p_item, const String &p_text) {
	p_item->set_cell_mode(0, TreeItem::CELL_MODE_CUSTOM);
	p_item->set_custom_draw_callback(0, callable_mp(this, &ExportTemplateManager::_draw_item_progress));
	p_item->set_text(0, p_text);
}

void ExportTemplateManager::_setup_downloading_item(TreeItem *p_item, const String &p_text) {
	_setup_custom_item(p_item, p_text);
	p_item->add_button(0, get_editor_theme_icon(SNAME("Close")), (int)ButtonID::CANCEL);
	p_item->set_button_tooltip_text(0, -1, TTRC("Cancel downloading this file."));
	downloading_items.push_back(p_item);

	Dictionary item_metadata = p_item->get_metadata(0);
	item_metadata["download_status"] = DownloadStatus::PENDING;
}

void ExportTemplateManager::_setup_check_item(TreeItem *p_item, const String &p_text) {
	p_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	p_item->set_editable(0, true);
	p_item->set_text(0, p_text);
}

void ExportTemplateManager::_apply_item_folding(TreeItem *p_item, bool p_default) {
	HashMap<String, bool> &cache = (p_item->get_tree() == available_templates_tree ? folding_cache_available : folding_cache_installed);
	if (cache.is_empty()) {
		if (p_default) {
			p_item->set_collapsed(true);
		}
	} else {
		p_item->set_collapsed(cache[p_item->get_text(0)]);
	}
}

void ExportTemplateManager::_fail_item_download(TreeItem *p_item, const String &p_reason) {
	Dictionary item_metadata = p_item->get_metadata(0);
	HTTPRequest *downloader = Object::cast_to<HTTPRequest>(item_metadata["downloader"].get_validated_object());
	if (downloader) {
		// TODO: See if it sends finished signal etc.
		downloader->cancel_request();
	}

	item_metadata["download_status"] = DownloadStatus::FAILED;
	item_metadata["fail_reason"] = p_reason;
	item_metadata["fail_progress"] = _get_download_progress(p_item);

	p_item->clear_buttons();
	p_item->add_button(0, get_editor_theme_icon(SNAME("NodeWarning")));
	p_item->set_button_tooltip_text(0, -1, vformat(TTR("Download failed.\nReason: %s."), TTR(p_reason)));
}

bool ExportTemplateManager::_item_is_file(TreeItem *p_item) const {
	// Only file items have metadata.
	return p_item->get_metadata(0).operator bool();
}

float ExportTemplateManager::_get_download_progress(TreeItem *p_item) const {
	Dictionary item_metadata = p_item->get_metadata(0);
	DownloadStatus status = (DownloadStatus)item_metadata["download_status"];

	switch (status) {
		case DownloadStatus::PENDING: {
			return 0.0;
		}

		case DownloadStatus::IN_PROGRESS: {
			HTTPRequest *downloader = Object::cast_to<HTTPRequest>(item_metadata["downloader"].get_validated_object());
			if (!downloader) {
				return 0.0;
			}
			return (float)downloader->get_downloaded_bytes() / (float)downloader->get_body_size();
		}

		case DownloadStatus::COMPLETED: {
			return 1.0;
		}

		case DownloadStatus::FAILED: {
			return item_metadata["fail_progress"];
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

		for (TreeItem *child = p_item->get_first_child(); child; child = child->get_next()) {
			if (!downloading_items.has(child)) {
				continue;
			}
			item_count++;
			progress += _get_download_progress(child);
		}
		progress /= item_count;
		owning_tree->draw_rect(Rect2(p_rect.position, Vector2(p_rect.size.x * progress, p_rect.size.y)), Color(0, 1, 0, 0.5));
		return;
	}

	Dictionary item_metadata = p_item->get_metadata(0);
	DownloadStatus status = (DownloadStatus)item_metadata["download_status"];
	switch (status) {
		case DownloadStatus::PENDING: {
			uint64_t frame = Engine::get_singleton()->get_frames_drawn();
			const Ref<Texture2D> progress_texture = get_editor_theme_icon("Progress" + itos((frame / 4) % 8 + 1));
			owning_tree->draw_texture(progress_texture, Vector2(p_rect.get_end().x - progress_texture->get_width(), p_rect.position.y + p_rect.size.y * 0.5 - progress_texture->get_height() * 0.5));
		} break;

		case DownloadStatus::IN_PROGRESS: {
			owning_tree->draw_rect(Rect2(p_rect.position, Vector2(p_rect.size.x * _get_download_progress(p_item), p_rect.size.y)), Color(0, 1, 0, 0.5));
		} break;

		case DownloadStatus::COMPLETED: {
			owning_tree->draw_rect(p_rect, Color(0, 1, 0, 0.5));
		} break;

		case DownloadStatus::FAILED: {
			owning_tree->draw_rect(Rect2(p_rect.position, Vector2(p_rect.size.x * _get_download_progress(p_item), p_rect.size.y)), Color(1, 0, 0, 0.5));
		} break;
	}
}

void ExportTemplateManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			open_folder_button->set_button_icon(get_editor_theme_icon("Folder"));
			install_button->set_button_icon(get_editor_theme_icon("AssetLib"));
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			available_templates_tree->queue_redraw();
		}
	}
}

void ExportTemplateManager::popup_manager() {
	if (template_data.is_empty()) {
		_update_install_button_text();
		_initialize_template_data();
	}

	if (!_is_downloading()) {
		_update_template_tree();
	}
	popup_centered_clamped(Vector2i(600, 640) * EDSCALE);
}

ExportTemplateManager::ExportTemplateManager() {
	set_title(TTRC("Export Template Manager"));
	set_ok_button_text(TTRC("Close"));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	main_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(main_hb);

	VBoxContainer *side_vb = memnew(VBoxContainer);
	main_hb->add_child(side_vb);

	Label *version_header = memnew(Label(TTRC("Engine Version")));
	version_header->set_theme_type_variation("HeaderSmall");
	side_vb->add_child(version_header);

	version_list = memnew(ItemList);
	version_list->set_theme_type_variation("ItemListSecondary");
	version_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	side_vb->add_child(version_list);
	version_list->connect("item_selected", callable_mp(this, &ExportTemplateManager::_version_selected).unbind(1));

	open_folder_button = memnew(Button);
	open_folder_button->set_tooltip_text(TTRC("Open templates directory."));
	open_folder_button->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
	side_vb->add_child(open_folder_button);
	open_folder_button->connect(SceneStringName(pressed), callable_mp(this, &ExportTemplateManager::_open_template_directory));

	VSplitContainer *center_split = memnew(VSplitContainer);
	center_split->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_hb->add_child(center_split);

	VBoxContainer *installed_templates_container = memnew(VBoxContainer);
	installed_templates_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	center_split->add_child(installed_templates_container);

	Label *template_header = memnew(Label(TTRC("Installed Templates")));
	template_header->set_theme_type_variation("HeaderSmall");
	installed_templates_container->add_child(template_header);

	installed_templates_tree = memnew(Tree);
	installed_templates_tree->set_hide_root(true);
	installed_templates_tree->set_theme_type_variation("TreeSecondary");
	installed_templates_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	installed_templates_container->add_child(installed_templates_tree);
	installed_templates_tree->connect("button_clicked", callable_mp(this, &ExportTemplateManager::_tree_button_clicked));

	available_templates_container = memnew(VBoxContainer);
	available_templates_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	center_split->add_child(available_templates_container);

	HBoxContainer *available_header_hb = memnew(HBoxContainer);
	available_templates_container->add_child(available_header_hb);

	Label *template_header2 = memnew(Label(TTRC("Available Templates")));
	template_header2->set_theme_type_variation("HeaderSmall");
	template_header2->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	available_header_hb->add_child(template_header2);

	install_button = memnew(Button);
	available_header_hb->add_child(install_button);
	install_button->connect(SceneStringName(pressed), callable_mp(this, &ExportTemplateManager::_install_templates));

	available_templates_tree = memnew(Tree);
	available_templates_tree->set_hide_root(true);
	available_templates_tree->set_theme_type_variation("TreeSecondary");
	available_templates_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	available_templates_container->add_child(available_templates_tree);
	available_templates_tree->connect("button_clicked", callable_mp(this, &ExportTemplateManager::_tree_button_clicked));
	available_templates_tree->connect("item_edited", callable_mp(this, &ExportTemplateManager::_tree_item_edited));

	Label *offline_mode_label = memnew(Label(TTRC("Offline mode, some functionality is not available.")));
	offline_mode_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	offline_mode_label->hide();
	main_vb->add_child(offline_mode_label);

	for (int i = 0; i < 5; i++) {
		HTTPRequest *downloader = memnew(HTTPRequest);
		downloader->set_use_threads(true);
		add_child(downloader);
		downloaders.push_back(downloader);
		downloader->connect("request_completed", callable_mp(this, &ExportTemplateManager::_download_request_completed).bind(downloader));
	}
}
