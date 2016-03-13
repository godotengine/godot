#ifndef ADDON_EDITOR_PLUGIN_H
#define ADDON_EDITOR_PLUGIN_H


#include "editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/link_button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/separator.h"

#include "scene/gui/grid_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_button.h"
#include "scene/gui/rich_text_label.h"
#include "editor_plugin_settings.h"

#include "scene/main/http_request.h"
#include "editor_asset_installer.h"

class EditorAssetLibraryItem : public PanelContainer {

	OBJ_TYPE( EditorAssetLibraryItem, PanelContainer );

	TextureButton *icon;
	LinkButton* title;
	LinkButton* category;
	LinkButton* author;
	TextureFrame *stars[5];
	Label* price;

	int asset_id;
	int category_id;
	int author_id;


	void _asset_clicked();
	void _category_clicked();
	void _author_clicked();


	void set_image(int p_type,int p_index,const Ref<Texture>& p_image);

protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void configure(const String& p_title,int p_asset_id,const String& p_category,int p_category_id,const String& p_author,int p_author_id,int p_rating,const String& p_cost);


	EditorAssetLibraryItem();
};


class EditorAddonLibraryItemDescription : public ConfirmationDialog {

	OBJ_TYPE(EditorAddonLibraryItemDescription, ConfirmationDialog);

	EditorAssetLibraryItem *item;
	RichTextLabel *description;
	ScrollContainer *previews;
	HBoxContainer *preview_hb;

	struct Preview {
		int id;
		String video_link;
		Button *button;
	};

	Vector<Preview> preview_images;
	TextureFrame *preview;

	void set_image(int p_type,int p_index,const Ref<Texture>& p_image);

	int asset_id;
	String download_url;
	String title;
	Ref<Texture> icon;

	void _link_click(const String& p_url);
protected:

	static void _bind_methods();
public:

	void configure(const String& p_title,int p_asset_id,const String& p_category,int p_category_id,const String& p_author,int p_author_id,int p_rating,const String& p_cost,const String& p_version,const String& p_description,const String& p_download_url,const String& p_browse_url);
	void add_preview(int p_id, bool p_video,const String& p_url);

	String get_title() { return title; }
	Ref<Texture> get_preview_icon() { return icon; }
	String get_download_url() { return download_url; }
	int get_asset_id() { return asset_id; }
	EditorAddonLibraryItemDescription();

};

class EditorAddonLibraryItemDownload : public PanelContainer {

	OBJ_TYPE(EditorAddonLibraryItemDownload, PanelContainer);


	TextureFrame *icon;
	Label* title;
	ProgressBar *progress;
	Button *install;
	TextureButton *dismiss;

	AcceptDialog *download_error;
	HTTPRequest *download;
	String host;
	Label *status;

	int prev_status;

	int asset_id;

	EditorAssetInstaller *asset_installer;

	void _close();
	void _install();
	void _http_download_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data);

protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	int get_asset_id() { return asset_id; }
	void configure(const String& p_title,int p_asset_id,const Ref<Texture>& p_preview, const String& p_download_url);
	EditorAddonLibraryItemDownload();

};

class EditorAddonLibrary : public PanelContainer {
	OBJ_TYPE(EditorAddonLibrary,PanelContainer);

	String host;

	EditorFileDialog *asset_open;
	EditorAssetInstaller *asset_installer;


	void _asset_open();
	void _asset_file_selected(const String& p_file);


	ScrollContainer *library_scroll;
	VBoxContainer *library_vb;
	LineEdit *filter;
	OptionButton *categories;
	OptionButton *repository;
	OptionButton *sort;
	CheckBox *reverse;
	Button *search;
	ProgressBar *load_status;
	HBoxContainer *error_hb;
	Label *error_label;

	HBoxContainer *contents;

	HBoxContainer *asset_top_page;
	GridContainer *asset_items;
	HBoxContainer *asset_bottom_page;

	HTTPRequest *request;


	enum SortOrder {
		SORT_RATING,
		SORT_DOWNLOADS,
		SORT_NAME,
		SORT_COST,
		SORT_UPDATED,
		SORT_MAX
	};


	static const char* sort_key[SORT_MAX];
	static const char* sort_text[SORT_MAX];


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
		HTTPRequest *request;
		ObjectID target;
	};

	int last_queue_id;
	Map<int,ImageQueue> image_queue;


	void _image_request_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data, int p_queue_id);

	void _request_image(ObjectID p_for,int p_asset_id,ImageType p_type,int p_image_index);
	void _update_image_queue();

	HBoxContainer* _make_pages(int p_page, int p_max_page, int p_page_len, int p_total_items, int p_current_items);

	//
	EditorAddonLibraryItemDescription *description;

	String current_request;
	//

	enum RequestType {
		REQUESTING_NONE,
		REQUESTING_CONFIG,
	};


	RequestType requesting;


	ScrollContainer *downloads_scroll;
	HBoxContainer *downloads_hb;



	void _install_asset();

	void _select_author(int p_id);
	void _select_category(int p_id);
	void _select_asset(int p_id);

	void _manage_plugins();

	void _search(int p_page=0);
	void _api_request(const String& p_request, const String &p_arguments="");
	void _http_request_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data);
	void _http_download_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data);

friend class EditorAddonLibraryItemDescription;
friend class EditorAssetLibraryItem;
protected:

	static void _bind_methods();
	void _notification(int p_what);
public:
	EditorAddonLibrary();
};

class AddonEditorPlugin : public EditorPlugin {

	OBJ_TYPE( AddonEditorPlugin, EditorPlugin );

	EditorAddonLibrary *addon_library;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Addons"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_object) {}
	virtual bool handles(Object *p_object) const { return false; }
	virtual void make_visible(bool p_visible);
	//virtual bool get_remove_list(List<Node*> *p_list) { return canvas_item_editor->get_remove_list(p_list); }
	//virtual Dictionary get_state() const;
	//virtual void set_state(const Dictionary& p_state);

	AddonEditorPlugin(EditorNode *p_node);
	~AddonEditorPlugin();

};

#endif // EDITORASSETLIBRARY_H
