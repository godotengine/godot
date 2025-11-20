/**************************************************************************/
/*  asset_library_editor_plugin.h                                         */
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

#include "editor/asset_library/editor_asset_installer.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/link_button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_button.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/http_request.h"

class EditorFileDialog;
class HSeparator;
class MenuButton;
class VSeparator;

class EditorAssetLibraryItem : public MarginContainer {
	GDCLASS(EditorAssetLibraryItem, MarginContainer);

	MarginContainer *margin = nullptr;
	Button *button = nullptr;
	TextureRect *icon = nullptr;
	Label *title = nullptr;
	LinkButton *author = nullptr;
	LinkButton *license = nullptr;
	HSeparator *separator = nullptr;
	Control *spacer = nullptr;
	HBoxContainer *author_license_hbox = nullptr;
	TextureRect *rating_icon = nullptr;
	Label *rating_count = nullptr;

	String title_text;
	String asset_id;
	String author_id;
	String license_url;

	bool is_hovering = false;
	bool is_clickable = false;

	int author_width = 0;
	int price_width = 0;

	void _asset_clicked();
	void _author_clicked();
	void _license_clicked();

	void _calculate_misc_links_size();

	void set_image(int p_type, int p_index, const Ref<Texture2D> &p_image);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void configure(const String &p_title, const String &p_asset_id, const String &p_author, const String &p_author_id, const String &p_license_type, const String &p_license_url, int p_rating);

	void calculate_misc_links_ratio();

	EditorAssetLibraryItem(bool p_clickable = false);
};

class EditorAssetLibraryZoomMode : public CanvasLayer {
	GDCLASS(EditorAssetLibraryZoomMode, CanvasLayer);

	Control *previews = nullptr;

	virtual void input(const Ref<InputEvent> &p_event) override;

public:
	Control *remove_previews();

	EditorAssetLibraryZoomMode(Control *p_previews);
};

class EditorAssetLibraryItemDescription : public ConfirmationDialog {
	GDCLASS(EditorAssetLibraryItemDescription, ConfirmationDialog);

	EditorAssetLibraryItem *item = nullptr;
	HBoxContainer *root = nullptr;
	RichTextLabel *description = nullptr;
	Label *version_label = nullptr;
	Label *version = nullptr;
	OptionButton *version_list = nullptr;
	Button *store = nullptr;
	Button *source = nullptr;
	VBoxContainer *desc_vbox = nullptr;

	VBoxContainer *previews_vbox = nullptr;
	Button *previous_preview = nullptr;
	Button *next_preview = nullptr;
	ScrollContainer *previews = nullptr;
	HBoxContainer *preview_hb = nullptr;
	PanelContainer *previews_bg = nullptr;

	Button *zoom_button = nullptr;
	EditorAssetLibraryZoomMode *zoom_mode = nullptr;

	struct Preview {
		int id = 0;
		bool is_video = false;
		String video_link;
		String thumbnail;
		Button *button = nullptr;
		Ref<Texture2D> image;
	};

	Vector<Preview> preview_images;
	Ref<ButtonGroup> preview_group;
	TextureRect *preview = nullptr;

	void set_image(int p_type, int p_index, const Ref<Texture2D> &p_image);

	struct Release {
		String url;
		String version;
		String sha256;
	};

	String asset_id;
	Vector<Release> releases;
	String store_url;
	String source_url;
	String title;
	Ref<Texture2D> icon;

public:
	enum InstallMode {
		MODE_DOWNLOAD,
		MODE_DOWNLOADING,
		MODE_INSTALL,
	};

private:
	InstallMode install_mode = MODE_DOWNLOAD;

	void _confirmed();
	void _store_pressed();
	void _source_pressed();
	void _link_click(const String &p_url);

	void _previous_preview_pressed();
	void _next_preview_pressed();

	void _zoom_toggled(bool p_pressed);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void configure(const String &p_title, const String &p_asset_id, const String &p_author, const String &p_author_id, const String &p_license_type, const String &p_license_url, int p_rating, const String &p_description, const HashMap<String, String> &p_tags, const String &p_store_url, const String &p_source_url);
	void set_install_mode(InstallMode p_mode);
	void add_release(const String &p_url, const String &p_version, const String &p_sha256);
	void add_preview(int p_id, bool p_video = false, const String &p_url = "", const String &p_thumbnail = "");
	void preview_click(int p_id);

	String get_title() { return title; }
	Ref<Texture2D> get_preview_icon() { return icon; }
	EditorAssetLibraryItemDescription();
};

class EditorAssetLibraryItemDownload : public MarginContainer {
	GDCLASS(EditorAssetLibraryItemDownload, MarginContainer);

	PanelContainer *panel = nullptr;
	TextureRect *icon = nullptr;
	Label *title = nullptr;
	Label *version = nullptr;
	ProgressBar *progress = nullptr;
	Button *install_button = nullptr;
	Button *retry_button = nullptr;
	TextureButton *dismiss_button = nullptr;
	HBoxContainer *progress_hbox = nullptr;
	Control *spacer = nullptr;

	AcceptDialog *download_error = nullptr;
	HTTPRequest *download = nullptr;
	String host;
	String sha256;
	Label *status = nullptr;

	int prev_status;

	String asset_id;

	bool external_install;

	EditorAssetInstaller *asset_installer = nullptr;

	void _close();
	void _make_request();
	void _http_download_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_external_install(bool p_enable) { external_install = p_enable; }
	String get_asset_id() { return asset_id; }
	void configure(const String &p_title, const String &p_asset_id, const String &p_version, const Ref<Texture2D> &p_preview, const String &p_download_url, const String &p_sha256);

	bool can_install() const;
	void install();

	EditorAssetLibraryItemDownload();
};

class EditorAssetLibrary : public PanelContainer {
	GDCLASS(EditorAssetLibrary, PanelContainer);

	String host;

	EditorFileDialog *asset_open = nullptr;
	EditorAssetInstaller *asset_installer = nullptr;

	void _asset_open();
	void _asset_file_selected(const String &p_file);
	void _update_repository_options();

	MarginContainer *library_mc = nullptr;
	ScrollContainer *library_scroll = nullptr;
	VBoxContainer *library_vb = nullptr;
	VBoxContainer *library_message_box = nullptr;
	Label *library_message = nullptr;
	Button *library_message_button = nullptr;
	Callable library_message_action;

	void _set_library_message(const String &p_message);
	void _set_library_message_with_action(const String &p_message, const String &p_action_text, const Callable &p_action);

	LineEdit *filter = nullptr;
	Timer *filter_debounce_timer = nullptr;
	OptionButton *sort = nullptr;
	OptionButton *categories = nullptr;
	OptionButton *repository = nullptr;
	MenuButton *licenses = nullptr;
	HBoxContainer *error_hb = nullptr;
	TextureRect *error_tr = nullptr;
	Label *error_label = nullptr;

	HBoxContainer *contents = nullptr;

	HBoxContainer *asset_top_page = nullptr;
	GridContainer *asset_items = nullptr;
	HBoxContainer *asset_bottom_page = nullptr;

	HTTPRequest *request = nullptr;

	bool templates_only = false;
	bool initial_loading = true;
	bool loading_blocked = false;

	void _force_online_mode();

	bool licenses_changed = false;

	void _licenses_id_pressed(int p_id);
	void _licenses_popup_hide();

	enum SortOrder {
		SORT_RELEVANCE,
		SORT_UPDATED,
		SORT_UPDATED_REVERSE,
		SORT_REVIEWS,
		SORT_REVIEWS_REVERSE,
		SORT_CREATED,
		SORT_CREATED_REVERSE,
		SORT_MAX
	};

	static const char *sort_key[SORT_MAX];
	static const char *sort_text[SORT_MAX];

	constexpr static Size2 THUMBNAIL_SIZE = Size2(114, 64);

	enum ImageType {
		IMAGE_QUEUE_THUMBNAIL,
		IMAGE_QUEUE_VIDEO_THUMBNAIL,
		IMAGE_QUEUE_SCREENSHOT,

	};

	struct ImageQueue {
		bool active = false;
		int queue_id = 0;
		ImageType image_type = ImageType::IMAGE_QUEUE_THUMBNAIL;
		int image_index = 0;
		String image_url;
		HTTPRequest *request = nullptr;
		ObjectID target;
		int asset_id = -1;

		Thread *thread = nullptr;
		bool use_cache = false;
		PackedByteArray data;
		Ref<ImageTexture> texture;
		bool update_finished = false;
	};

	int last_queue_id;
	HashMap<int, ImageQueue> image_queue;

	static void _image_update(void *p_image_queue);
	void _image_request_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data, int p_queue_id);
	void _request_image(ObjectID p_for, int p_asset_id, const String &p_image_url, ImageType p_type, int p_image_index);
	void _update_image_queue();

	int current_page = 0;

	HBoxContainer *_make_pages(int p_page, int p_page_count, int p_page_len, int p_total_items, int p_current_items);

	enum RequestType {
		REQUESTING_NONE,
		REQUESTING_CHECK,
		REQUESTING_TAGS,
		REQUESTING_LICENSES,
		REQUESTING_SEARCH,
		REQUESTING_ASSET,
		REQUESTING_RELEASES,
	};

	Dictionary category_map;

	ScrollContainer *downloads_scroll = nullptr;
	HBoxContainer *downloads_hb = nullptr;

	EditorAssetLibraryItemDescription *description = nullptr;

	void _install_asset(const String &p_asset_id, const String &p_version, const String &p_download_url, const String &p_sha256);
	void _tag_clicked(const String &p_tag);

	void _select_author(const String &p_author);
	void _select_asset(const String &p_id);

	void _manage_plugins();

	void _search(int p_page = 1);
	void _api_request(const String &p_request, RequestType p_request_type, bool p_is_parallel = false);
	void _http_request_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data, HTTPRequest *p_requester);
	void _request_current_config();
	EditorAssetLibraryItemDownload *_get_asset_in_progress(const String &p_asset_id) const;

	void _repository_changed(int p_repository_id);

	void _install_external_asset(const String &p_zip_path, const String &p_title);

	void _update_asset_items_columns();
	void _update_downloads_section();

	friend class EditorAssetLibraryItemDescription;
	friend class EditorAssetLibraryItem;

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

public:
	EditorAssetLibrary(bool p_templates_only = false);
};

class AssetLibraryEditorPlugin : public EditorPlugin {
	GDCLASS(AssetLibraryEditorPlugin, EditorPlugin);

	EditorAssetLibrary *addon_library = nullptr;

public:
	static bool is_available();

	virtual String get_plugin_name() const override { return TTRC("Asset Store"); }
	virtual const Ref<Texture2D> get_plugin_icon() const override;
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override {}
	virtual bool handles(Object *p_object) const override { return false; }
	virtual void make_visible(bool p_visible) override;

	AssetLibraryEditorPlugin();
};
