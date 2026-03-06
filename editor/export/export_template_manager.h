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
class Texture2D;
class Tree;
class TreeItem;

class ExportTemplateManager : public AcceptDialog {
	GDCLASS(ExportTemplateManager, AcceptDialog);

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
	};

	struct PlatformInfo {
		String name;
		Ref<Texture2D> icon;
		HashSet<TemplateID> templates;
		String group;
	};

	struct TemplateInfo {
		String name;
		PackedStringArray file_list;
	};

	HashMap<PlatformID, PlatformInfo> platform_map;
	HashMap<TemplateID, TemplateInfo> template_data;

	LocalVector<HTTPRequest *> downloaders;

	bool download_all_enabled = true;
	HashSet<String> queued_templates;
	HashSet<String> queued_files;
	LocalVector<TreeItem *> downloading_items;
	bool queue_update_pending = false;

	HashMap<String, bool> folding_cache_installed;
	HashMap<String, bool> folding_cache_available;

	ItemList *version_list = nullptr;
	Tree *installed_templates_tree = nullptr;
	VBoxContainer *available_templates_container = nullptr;
	Tree *available_templates_tree = nullptr;
	Button *open_folder_button = nullptr;
	Button *install_button = nullptr;

	void _initialize_template_data();
	void _update_template_tree();
	void _update_template_tree_with_folding();
	void _update_install_button_text();

	void _update_folding_cache(HashMap<String, bool> &p_cache, TreeItem *p_item);

	String _get_template_folder_path(const String &p_version) const;
	Ref<Texture2D> _get_platform_icon(const String &p_platform_name);

	void _version_selected();
	void _tree_button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_button);
	void _tree_item_edited();
	void _install_templates();
	void _open_template_directory();

	void _queue_download_tree_item(TreeItem *p_item);
	void _process_download_queue();
	void _queue_process_download_queue();
	HTTPRequest *_get_available_downloader(int *r_from_index);
	void _download_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, HTTPRequest *p_downloader);
	bool _is_downloading() const;

	void _setup_custom_item(TreeItem *p_item, const String &p_text);
	void _setup_downloading_item(TreeItem *p_item, const String &p_text);
	void _setup_check_item(TreeItem *p_item, const String &p_text);
	void _apply_item_folding(TreeItem *p_item, bool p_default = false);
	void _fail_item_download(TreeItem *p_item, const String &p_reason);

	bool _item_is_file(TreeItem *p_item) const;
	float _get_download_progress(TreeItem *p_item) const;
	void _draw_item_progress(TreeItem *p_item, const Rect2 &p_rect);

protected:
	void _notification(int p_what);

public:
	// TODO :D
	static String get_android_build_directory(const Ref<EditorExportPreset> &p_preset) { return ""; }
	static String get_android_template_identifier(const Ref<EditorExportPreset> &p_preset) { return ""; }
	static void install_android_template(const Ref<EditorExportPreset> &p_preset) {}
	static void install_android_template_from_file(const String &p_file, const Ref<EditorExportPreset> &p_preset) {}
	bool is_android_template_installed(const Ref<EditorExportPreset> &p_preset) { return false; }
	bool can_install_android_template(const Ref<EditorExportPreset> &p_preset) { return false; }

	void popup_manager();

	ExportTemplateManager();
};
