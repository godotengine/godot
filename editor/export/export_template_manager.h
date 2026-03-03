/**************************************************************************/
/*  export_template_manager.h                                             */
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

#pragma once

#include "scene/gui/dialogs.h"

class Button;
class EditorExportPreset;
class HTTPRequest;
class ItemList;
class HBoxContainer;
class OptionButton;
class Texture2D;
class Tree;
class TreeItem;

class ExportTemplateManager : public AcceptDialog {
	GDCLASS(ExportTemplateManager, AcceptDialog);

	const StringName PATH_META = "path";
	const StringName FILE_META = "file";

	enum class TemplateID {
		WINDOWS_X86_32,
		WINDOWS_X86_64,
		WINDOWS_ARM64,

		LINUX_X86_32,
		LINUX_X86_64,
		LINUX_ARM32,
		LINUX_ARM64,

		MACOS,

		WEB,
		WEB_EXTENSIONS,
		WEB_NOTHREADS,
		WEB_EXTENSIONS_NOTHREADS,

		ANDROID,
		ANDROID_SOURCE,

		IOS,

		ICU_DATA,
	};

	enum class PlatformID {
		WINDOWS,
		LINUX,
		MACOS,
		WEB,
		ANDROID,
		IOS,
		COMMON,
	};

	enum class DownloadStatus {
		NONE,
		PENDING,
		IN_PROGRESS,
		COMPLETED,
		FAILED,
	};

	enum class ButtonID {
		DOWNLOAD,
		REPAIR,
		REMOVE,
		CANCEL,
		FAIL,
		NONE,
	};

	struct PlatformInfo {
		String name;
		Ref<Texture2D> icon;
		HashSet<TemplateID> templates;
		String group;
	};

	struct TemplateInfo {
		String name;
		String description;
		PackedStringArray file_list;
	};

	struct FileMetadata {
		DownloadStatus download_status = DownloadStatus::NONE;
		HTTPRequest *downloader = nullptr;
		String fail_reason;
		float progress_cache = 0.0;
		bool is_missing = false;
	};

	bool mirrors_empty = true;

	HashMap<PlatformID, PlatformInfo> platform_map;
	HashMap<TemplateID, TemplateInfo> template_data;

	HTTPRequest *mirrors_requester = nullptr;
	LocalVector<HTTPRequest *> downloaders;

	bool download_all_enabled = true;
	HashSet<String> queued_templates;
	HashSet<String> queued_files;
	int download_count = 0;
	mutable HashMap<String, FileMetadata> file_metadata;
	LocalVector<TreeItem *> downloading_items;
	bool queue_update_pending = false;

	HashMap<String, int> checked_cache;
	HashMap<String, bool> folding_cache;

	OptionButton *mirrors_list = nullptr;
	Button *open_mirror = nullptr;
	ItemList *version_list = nullptr;
	Tree *installed_templates_tree = nullptr;
	Tree *available_templates_tree = nullptr;
	Button *open_folder_button = nullptr;
	Button *install_button = nullptr;
	HBoxContainer *offline_container = nullptr;

	void _request_mirrors();
	void _mirrors_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _set_empty_mirror_list();
	String _get_current_mirror_url() const;
	void _update_online_mode();
	bool _is_online() const;
	void _force_online_mode();
	void _open_mirror();

	void _initialize_template_data();
	void _update_template_tree();
	void _update_template_tree_theme(Tree *p_tree);
	void _fill_template_tree(Tree *p_tree, const HashMap<TemplateID, LocalVector<String>> &p_installed_template_files, bool p_is_current_version);
	void _update_template_tree_with_folding();
	void _update_install_button();
	bool _can_download_templates();

	void _update_folding_cache(TreeItem *p_item);

	String _get_template_folder_path(const String &p_version) const;
	Ref<Texture2D> _get_platform_icon(const String &p_platform_name);

	void _version_selected();
	void _tree_button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_button);
	void _tree_item_edited();
	void _install_templates(TreeItem *p_files = nullptr);
	void _open_template_directory();

	void _queue_download_tree_item(TreeItem *p_item);
	void _process_download_queue();
	void _queue_process_download_queue();
	HTTPRequest *_get_available_downloader(int *r_from_index);
	void _download_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, HTTPRequest *p_downloader);
	bool _is_template_download_finished(TreeItem *p_template);
	String _get_download_error(int p_result, int p_response_code) const;

	void _set_item_type(TreeItem *p_item, int p_type);
	void _setup_item_text(TreeItem *p_item, const String &p_text);
	FileMetadata *_get_file_metadata(const String &p_text) const;
	FileMetadata *_get_file_metadata(const TreeItem *p_item) const;
	void _apply_item_folding(TreeItem *p_item, bool p_default = false);
	void _cancel_item_download(TreeItem *p_item);
	void _item_download_failed(TreeItem *p_item, const String &p_reason);
	void _add_fail_reason_button(TreeItem *p_item, const String &p_filename = String());

	String _get_item_path(TreeItem *p_item) const;
	bool _item_is_file(TreeItem *p_item) const;
	bool _status_is_finished(DownloadStatus p_status) { return p_status == DownloadStatus::COMPLETED || p_status == DownloadStatus::FAILED; }
	float _get_download_progress(const TreeItem *p_item) const;
	void _draw_item_progress(TreeItem *p_item, const Rect2 &p_rect);

	struct ThemeCache {
		Ref<Texture2D> install_icon;
		Ref<Texture2D> remove_icon;
		Ref<Texture2D> repair_icon;
		Ref<Texture2D> failure_icon;
		Ref<Texture2D> cancel_icon;
		Ref<Texture2D> progress_icons[8];

		Color current_version_color;
		Color incomplete_template_color;
		Color missing_file_color;
		Color download_progress_color;
		Color download_failed_color;

		int icon_width = 0;
	} theme_cache;

protected:
	void _notification(int p_what);

public:
	static String get_android_build_directory(const Ref<EditorExportPreset> &p_preset);
	static String get_android_source_zip(const Ref<EditorExportPreset> &p_preset);
	static String get_android_template_identifier(const Ref<EditorExportPreset> &p_preset);

	bool is_android_template_installed(const Ref<EditorExportPreset> &p_preset);
	bool can_install_android_template(const Ref<EditorExportPreset> &p_preset);
	Error install_android_template(const Ref<EditorExportPreset> &p_preset);
	Error install_android_template_from_file(const String &p_file, const Ref<EditorExportPreset> &p_preset);

	void popup_manager();
	bool is_downloading() const;
	void stop_download();

	ExportTemplateManager();
};
