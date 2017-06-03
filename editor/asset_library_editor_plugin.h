/*************************************************************************/
/*  asset_library_editor_plugin.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef ASSET_LIBRARY_EDITOR_PLUGIN_H
#define ASSET_LIBRARY_EDITOR_PLUGIN_H

#include "editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/link_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/separator.h"
#include "scene/gui/tab_container.h"

#include "editor_plugin_settings.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_button.h"

#include "editor_asset_installer.h"
#include "scene/main/http_request.h"

class EditorAssetLibraryItem : public PanelContainer {

	GDCLASS(EditorAssetLibraryItem, PanelContainer);

	TextureButton *icon;
	LinkButton *title;
	LinkButton *category;
	LinkButton *author;
	TextureRect *stars[5];
	Label *price;

	int asset_id;
	int category_id;
	int author_id;

	void _asset_clicked();
	void _category_clicked();
	void _author_clicked();

	void set_image(int p_type, int p_index, const Ref<Texture> &p_image);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void configure(const String &p_title, int p_asset_id, const String &p_category, int p_category_id, const String &p_author, int p_author_id, int p_rating, const String &p_cost);

	EditorAssetLibraryItem();
};

class EditorAssetLibraryItemDescription : public ConfirmationDialog {

	GDCLASS(EditorAssetLibraryItemDescription, ConfirmationDialog);

	EditorAssetLibraryItem *item;
	RichTextLabel *description;
	ScrollContainer *previews;
	HBoxContainer *preview_hb;

	struct Preview {
		int id;
		bool is_video;
		String video_link;
		Button *button;
		Ref<Texture> image;
	};

	Vector<Preview> preview_images;
	TextureRect *preview;

	void set_image(int p_type, int p_index, const Ref<Texture> &p_image);

	int asset_id;
	String download_url;
	String title;
	String sha256;
	Ref<Texture> icon;

	void _link_click(const String &p_url);
	void _preview_click(int p_index);

protected:
	static void _bind_methods();

public:
	void configure(const String &p_title, int p_asset_id, const String &p_category, int p_category_id, const String &p_author, int p_author_id, int p_rating, const String &p_cost, int p_version, const String &p_version_string, const String &p_description, const String &p_download_url, const String &p_browse_url, const String &p_sha256_hash);
	void add_preview(int p_id, bool p_video, const String &p_url);

	String get_title() { return title; }
	Ref<Texture> get_preview_icon() { return icon; }
	String get_download_url() { return download_url; }
	int get_asset_id() { return asset_id; }
	String get_sha256() { return sha256; }
	EditorAssetLibraryItemDescription();
};

class EditorAssetLibraryItemDownload : public PanelContainer {

	GDCLASS(EditorAssetLibraryItemDownload, PanelContainer);

	TextureRect *icon;
	Label *title;
	ProgressBar *progress;
	Button *install;
	Button *retry;
	TextureButton *dismiss;

	AcceptDialog *download_error;
	HTTPRequest *download;
	String host;
	String sha256;
	Label *status;

	int prev_status;

	int asset_id;

	bool external_install;

	EditorAssetInstaller *asset_installer;

	void _close();
	void _install();
	void _make_request();
	void _http_download_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_external_install(bool p_enable) { external_install = p_enable; }
	int get_asset_id() { return asset_id; }
	void configure(const String &p_title, int p_asset_id, const Ref<Texture> &p_preview, const String &p_download_url, const String &p_sha256_hash);
	EditorAssetLibraryItemDownload();
};

class EditorAssetLibrary : public PanelContainer {
	GDCLASS(EditorAssetLibrary, PanelContainer);

	String host;

	EditorFileDialog *asset_open;
	EditorAssetInstaller *asset_installer;

	void _asset_open();
	void _asset_file_selected(const String &p_file);

	ScrollContainer *library_scroll;
	VBoxContainer *library_vb;
	LineEdit *filter;
	OptionButton *categories;
	OptionButton *repository;
	OptionButton *sort;
	ToolButton *reverse;
	Button *search;
	ProgressBar *load_status;
	HBoxContainer *error_hb;
	Label *error_label;
	MenuButton *support;

	HBoxContainer *contents;

	HBoxContainer *asset_top_page;
	GridContainer *asset_items;
	HBoxContainer *asset_bottom_page;

	HTTPRequest *request;

	bool templates_only;

	enum Support {
		SUPPORT_OFFICIAL,
		SUPPORT_COMMUNITY,
		SUPPORT_TESTING,
		SUPPORT_MAX
	};

	enum SortOrder {
		SORT_RATING,
		SORT_DOWNLOADS,
		SORT_NAME,
		SORT_COST,
		SORT_UPDATED,
		SORT_MAX
	};

	static const char *sort_key[SORT_MAX];
	static const char *sort_text[SORT_MAX];
	static const char *support_key[SUPPORT_MAX];

	///MainListing

	enum ImageType {
		IMAGE_QUEUE_ICON,
		IMAGE_QUEUE_THUMBNAIL,
		IMAGE_QUEUE_SCREENSHOT,

	};

	struct ImageQueue {

		bool active;
		int queue_id;
		int asset_id;
		ImageType image_type;
		int image_index;
		String image_url;
		HTTPRequest *request;
		ObjectID target;
	};

	int last_queue_id;
	Map<int, ImageQueue> image_queue;

	void _image_update(bool use_cache, bool final, const PoolByteArray &p_data, int p_queue_id);
	void _image_request_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data, int p_queue_id);
	void _request_image(ObjectID p_for, String p_image_url, ImageType p_type, int p_image_index);
	void _update_image_queue();

	HBoxContainer *_make_pages(int p_page, int p_page_count, int p_page_len, int p_total_items, int p_current_items);

	//
	EditorAssetLibraryItemDescription *description;
	//

	enum RequestType {
		REQUESTING_NONE,
		REQUESTING_CONFIG,
		REQUESTING_SEARCH,
		REQUESTING_ASSET,
	};

	RequestType requesting;
	Dictionary category_map;

	ScrollContainer *downloads_scroll;
	HBoxContainer *downloads_hb;

	void _install_asset();

	void _select_author(int p_id);
	void _select_category(int p_id);
	void _select_asset(int p_id);

	void _manage_plugins();

	void _search(int p_page = 0);
	void _rerun_search(int p_ignore);
	void _api_request(const String &p_request, RequestType p_request_type, const String &p_arguments = "");
	void _http_request_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data);
	void _http_download_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data);

	void _repository_changed(int p_repository_id);
	void _support_toggled(int p_support);

	void _install_external_asset(String p_zip_path, String p_title);

	friend class EditorAssetLibraryItemDescription;
	friend class EditorAssetLibraryItem;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	EditorAssetLibrary(bool p_templates_only = false);
};

class AssetLibraryEditorPlugin : public EditorPlugin {

	GDCLASS(AssetLibraryEditorPlugin, EditorPlugin);

	EditorAssetLibrary *addon_library;
	EditorNode *editor;

public:
	virtual String get_name() const { return "AssetLib"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_object) {}
	virtual bool handles(Object *p_object) const { return false; }
	virtual void make_visible(bool p_visible);
	//virtual bool get_remove_list(List<Node*> *p_list) { return canvas_item_editor->get_remove_list(p_list); }
	//virtual Dictionary get_state() const;
	//virtual void set_state(const Dictionary& p_state);

	AssetLibraryEditorPlugin(EditorNode *p_node);
	~AssetLibraryEditorPlugin();
};

#endif // EDITORASSETLIBRARY_H
