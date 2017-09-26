/*************************************************************************/
/*  asset_library_editor_plugin.cpp                                      */
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
#include "asset_library_editor_plugin.h"

#include "editor_node.h"
#include "editor_settings.h"
#include "io/json.h"

void EditorAssetLibraryItem::configure(const String &p_title, int p_asset_id, const String &p_category, int p_category_id, const String &p_author, int p_author_id, int p_rating, const String &p_cost) {

	title->set_text(p_title);
	asset_id = p_asset_id;
	category->set_text(p_category);
	category_id = p_category_id;
	author->set_text(p_author);
	author_id = p_author_id;
	price->set_text(p_cost);

	for (int i = 0; i < 5; i++) {
		if (i < p_rating)
			stars[i]->set_texture(get_icon("RatingStar", "EditorIcons"));
		else
			stars[i]->set_texture(get_icon("RatingNoStar", "EditorIcons"));
	}
}

void EditorAssetLibraryItem::set_image(int p_type, int p_index, const Ref<Texture> &p_image) {

	ERR_FAIL_COND(p_type != EditorAssetLibrary::IMAGE_QUEUE_ICON);
	ERR_FAIL_COND(p_index != 0);

	icon->set_normal_texture(p_image);
}

void EditorAssetLibraryItem::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		icon->set_normal_texture(get_icon("GodotAssetDefault", "EditorIcons"));
		category->add_color_override("font_color", Color(0.5, 0.5, 0.5));
		author->add_color_override("font_color", Color(0.5, 0.5, 0.5));
	}
}

void EditorAssetLibraryItem::_asset_clicked() {

	emit_signal("asset_selected", asset_id);
}

void EditorAssetLibraryItem::_category_clicked() {

	emit_signal("category_selected", category_id);
}
void EditorAssetLibraryItem::_author_clicked() {

	emit_signal("author_selected", author_id);
}

void EditorAssetLibraryItem::_bind_methods() {

	ClassDB::bind_method("set_image", &EditorAssetLibraryItem::set_image);
	ClassDB::bind_method("_asset_clicked", &EditorAssetLibraryItem::_asset_clicked);
	ClassDB::bind_method("_category_clicked", &EditorAssetLibraryItem::_category_clicked);
	ClassDB::bind_method("_author_clicked", &EditorAssetLibraryItem::_author_clicked);
	ADD_SIGNAL(MethodInfo("asset_selected"));
	ADD_SIGNAL(MethodInfo("category_selected"));
	ADD_SIGNAL(MethodInfo("author_selected"));
}

EditorAssetLibraryItem::EditorAssetLibraryItem() {

	Ref<StyleBoxEmpty> border;
	border.instance();
	border->set_default_margin(MARGIN_LEFT, 5);
	border->set_default_margin(MARGIN_RIGHT, 5);
	border->set_default_margin(MARGIN_BOTTOM, 5);
	border->set_default_margin(MARGIN_TOP, 5);
	add_style_override("panel", border);

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	icon = memnew(TextureButton);
	icon->set_default_cursor_shape(CURSOR_POINTING_HAND);
	icon->connect("pressed", this, "_asset_clicked");

	hb->add_child(icon);

	VBoxContainer *vb = memnew(VBoxContainer);

	hb->add_child(vb);
	vb->set_h_size_flags(SIZE_EXPAND_FILL);

	title = memnew(LinkButton);
	title->set_text("My Awesome Addon");
	title->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	title->connect("pressed", this, "_asset_clicked");
	vb->add_child(title);

	category = memnew(LinkButton);
	category->set_text("Editor Tools");
	category->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	category->connect("pressed", this, "_category_clicked");
	vb->add_child(category);

	author = memnew(LinkButton);
	author->set_text("Johny Tolengo");
	author->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	author->connect("pressed", this, "_author_clicked");
	vb->add_child(author);

	HBoxContainer *rating_hb = memnew(HBoxContainer);
	vb->add_child(rating_hb);

	for (int i = 0; i < 5; i++) {
		stars[i] = memnew(TextureRect);
		rating_hb->add_child(stars[i]);
	}
	price = memnew(Label);
	price->set_text(TTR("Free"));
	vb->add_child(price);

	set_custom_minimum_size(Size2(250, 100));
	set_h_size_flags(SIZE_EXPAND_FILL);

	set_mouse_filter(MOUSE_FILTER_PASS);
}

//////////////////////////////////////////////////////////////////////////////

void EditorAssetLibraryItemDescription::set_image(int p_type, int p_index, const Ref<Texture> &p_image) {

	switch (p_type) {

		case EditorAssetLibrary::IMAGE_QUEUE_ICON: {

			item->call("set_image", p_type, p_index, p_image);
			icon = p_image;
		} break;
		case EditorAssetLibrary::IMAGE_QUEUE_THUMBNAIL: {

			for (int i = 0; i < preview_images.size(); i++) {
				if (preview_images[i].id == p_index) {
					preview_images[i].button->set_icon(p_image);
					break;
				}
			}
			//item->call("set_image",p_type,p_index,p_image);
		} break;
		case EditorAssetLibrary::IMAGE_QUEUE_SCREENSHOT: {

			for (int i = 0; i < preview_images.size(); i++) {
				if (preview_images[i].id == p_index) {
					preview_images[i].image = p_image;
					if (preview_images[i].button->is_pressed()) {
						_preview_click(p_index);
					}
					break;
				}
			}
			//item->call("set_image",p_type,p_index,p_image);
		} break;
	}
}
void EditorAssetLibraryItemDescription::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			previews_bg->add_style_override("panel", get_stylebox("normal", "TextEdit"));
			desc_bg->add_style_override("panel", get_stylebox("normal", "TextEdit"));
		} break;
	}
}
void EditorAssetLibraryItemDescription::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_image"), &EditorAssetLibraryItemDescription::set_image);
	ClassDB::bind_method(D_METHOD("_link_click"), &EditorAssetLibraryItemDescription::_link_click);
	ClassDB::bind_method(D_METHOD("_preview_click"), &EditorAssetLibraryItemDescription::_preview_click);
}

void EditorAssetLibraryItemDescription::_link_click(const String &p_url) {
	ERR_FAIL_COND(!p_url.begins_with("http"));
	OS::get_singleton()->shell_open(p_url);
}

void EditorAssetLibraryItemDescription::_preview_click(int p_id) {
	for (int i = 0; i < preview_images.size(); i++) {
		if (preview_images[i].id == p_id) {
			preview_images[i].button->set_pressed(true);
			if (!preview_images[i].is_video) {
				if (preview_images[i].image.is_valid()) {
					preview->set_texture(preview_images[i].image);
				}
			} else {
				_link_click(preview_images[i].video_link);
			}
		} else {
			preview_images[i].button->set_pressed(false);
		}
	}
}

void EditorAssetLibraryItemDescription::configure(const String &p_title, int p_asset_id, const String &p_category, int p_category_id, const String &p_author, int p_author_id, int p_rating, const String &p_cost, int p_version, const String &p_version_string, const String &p_description, const String &p_download_url, const String &p_browse_url, const String &p_sha256_hash) {

	asset_id = p_asset_id;
	title = p_title;
	download_url = p_download_url;
	sha256 = p_sha256_hash;
	item->configure(p_title, p_asset_id, p_category, p_category_id, p_author, p_author_id, p_rating, p_cost);
	description->clear();
	description->add_text(TTR("Version:") + " " + p_version_string + "\n");
	description->add_text(TTR("Contents:") + " ");
	description->push_meta(p_browse_url);
	description->add_text(TTR("View Files"));
	description->pop();
	description->add_text("\n" + TTR("Description:") + "\n\n");
	description->append_bbcode(p_description);
	set_title(p_title);
}

void EditorAssetLibraryItemDescription::add_preview(int p_id, bool p_video, const String &p_url) {

	Preview preview;
	preview.id = p_id;
	preview.video_link = p_url;
	preview.is_video = p_video;
	preview.button = memnew(Button);
	preview.button->set_flat(true);
	preview.button->set_icon(get_icon("ThumbnailWait", "EditorIcons"));
	preview.button->set_toggle_mode(true);
	preview.button->connect("pressed", this, "_preview_click", varray(p_id));
	preview_hb->add_child(preview.button);
	if (!p_video) {
		preview.image = get_icon("ThumbnailWait", "EditorIcons");
	}
	if (preview_images.size() == 0 && !p_video) {
		_preview_click(p_id);
	}
	preview_images.push_back(preview);
}

EditorAssetLibraryItemDescription::EditorAssetLibraryItemDescription() {

	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	HBoxContainer *hbox = memnew(HBoxContainer);
	vbox->add_child(hbox);
	vbox->add_constant_override("separation", 15);
	VBoxContainer *desc_vbox = memnew(VBoxContainer);
	hbox->add_child(desc_vbox);
	hbox->add_constant_override("separation", 15);

	item = memnew(EditorAssetLibraryItem);

	desc_vbox->add_child(item);
	desc_vbox->set_custom_minimum_size(Size2(300, 0));

	desc_bg = memnew(PanelContainer);
	desc_vbox->add_child(desc_bg);
	desc_bg->set_v_size_flags(SIZE_EXPAND_FILL);

	description = memnew(RichTextLabel);
	description->connect("meta_clicked", this, "_link_click");
	desc_bg->add_child(description);

	preview = memnew(TextureRect);
	preview->set_custom_minimum_size(Size2(640, 345));
	hbox->add_child(preview);

	previews_bg = memnew(PanelContainer);
	vbox->add_child(previews_bg);
	previews_bg->set_custom_minimum_size(Size2(0, 85));

	previews = memnew(ScrollContainer);
	previews_bg->add_child(previews);
	previews->set_enable_v_scroll(false);
	previews->set_enable_h_scroll(true);
	preview_hb = memnew(HBoxContainer);
	preview_hb->set_v_size_flags(SIZE_EXPAND_FILL);

	previews->add_child(preview_hb);
	get_ok()->set_text(TTR("Install"));
	get_cancel()->set_text(TTR("Close"));
}
///////////////////////////////////////////////////////////////////////////////////

void EditorAssetLibraryItemDownload::_http_download_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data) {

	String error_text;
	print_line("COMPLETED: " + itos(p_status) + " code: " + itos(p_code) + " data size: " + itos(p_data.size()));

	switch (p_status) {

		case HTTPRequest::RESULT_CANT_RESOLVE: {
			error_text = TTR("Can't resolve hostname:") + " " + host;
			status->set_text(TTR("Can't resolve."));
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: {
			error_text = TTR("Connection error, please try again.");
			status->set_text(TTR("Can't connect."));
		} break;
		case HTTPRequest::RESULT_SSL_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT: {
			error_text = TTR("Can't connect to host:") + " " + host;
			status->set_text(TTR("Can't connect."));
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			error_text = TTR("No response from host:") + " " + host;
			status->set_text(TTR("No response."));
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			error_text = TTR("Request failed, return code:") + " " + itos(p_code);
			status->set_text(TTR("Req. Failed."));
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			error_text = TTR("Request failed, too many redirects");
			status->set_text(TTR("Redirect Loop."));
		} break;
		default: {
			if (p_code != 200) {
				error_text = TTR("Request failed, return code:") + " " + itos(p_code);
				status->set_text(TTR("Failed:") + " " + itos(p_code));
			} else if (sha256 != "") {
				String download_sha256 = FileAccess::get_sha256(download->get_download_file());
				if (sha256 != download_sha256) {
					error_text = TTR("Bad download hash, assuming file has been tampered with.") + "\n";
					error_text += TTR("Expected:") + " " + sha256 + "\n" + TTR("Got:") + " " + download_sha256;
					status->set_text(TTR("Failed sha256 hash check"));
				}
			}
		} break;
	}

	if (error_text != String()) {
		download_error->set_text(TTR("Asset Download Error:") + "\n" + error_text);
		download_error->popup_centered_minsize();
		return;
	}

	progress->set_max(download->get_body_size());
	progress->set_value(download->get_downloaded_bytes());

	print_line("max: " + itos(download->get_body_size()) + " bytes: " + itos(download->get_downloaded_bytes()));
	install->set_disabled(false);

	progress->set_value(download->get_downloaded_bytes());

	status->set_text(TTR("Success!") + " (" + String::humanize_size(download->get_downloaded_bytes()) + ")");
	set_process(false);
}

void EditorAssetLibraryItemDownload::configure(const String &p_title, int p_asset_id, const Ref<Texture> &p_preview, const String &p_download_url, const String &p_sha256_hash) {

	title->set_text(p_title);
	icon->set_texture(p_preview);
	asset_id = p_asset_id;
	if (!p_preview.is_valid())
		icon->set_texture(get_icon("GodotAssetDefault", "EditorIcons"));
	host = p_download_url;
	sha256 = p_sha256_hash;
	asset_installer->connect("confirmed", this, "_close");
	dismiss->set_normal_texture(get_icon("Close", "EditorIcons"));
	_make_request();
}

void EditorAssetLibraryItemDownload::_notification(int p_what) {

	if (p_what == NOTIFICATION_PROCESS) {

		progress->set_max(download->get_body_size());
		progress->set_value(download->get_downloaded_bytes());

		int cstatus = download->get_http_client_status();

		if (cstatus == HTTPClient::STATUS_BODY)
			status->set_text(TTR("Fetching:") + " " + String::humanize_size(download->get_downloaded_bytes()));

		if (cstatus != prev_status) {
			switch (cstatus) {

				case HTTPClient::STATUS_RESOLVING: {
					status->set_text(TTR("Resolving.."));
				} break;
				case HTTPClient::STATUS_CONNECTING: {
					status->set_text(TTR("Connecting.."));
				} break;
				case HTTPClient::STATUS_REQUESTING: {
					status->set_text(TTR("Requesting.."));
				} break;
				default: {}
			}
			prev_status = cstatus;
		}
	}
}
void EditorAssetLibraryItemDownload::_close() {

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->remove(download->get_download_file()); //clean up removed file
	memdelete(da);
	queue_delete();
}

void EditorAssetLibraryItemDownload::_install() {

	String file = download->get_download_file();

	if (external_install) {
		emit_signal("install_asset", file, title->get_text());
		return;
	}

	asset_installer->open(file, 1);
}

void EditorAssetLibraryItemDownload::_make_request() {
	download->cancel_request();
	download->set_download_file(EditorSettings::get_singleton()->get_settings_path().plus_file("tmp").plus_file("tmp_asset_" + itos(asset_id)) + ".zip");

	Error err = download->request(host);
	if (err != OK) {
		status->set_text(TTR("Error making request"));
	} else {
		set_process(true);
	}
}

void EditorAssetLibraryItemDownload::_bind_methods() {

	ClassDB::bind_method("_http_download_completed", &EditorAssetLibraryItemDownload::_http_download_completed);
	ClassDB::bind_method("_install", &EditorAssetLibraryItemDownload::_install);
	ClassDB::bind_method("_close", &EditorAssetLibraryItemDownload::_close);
	ClassDB::bind_method("_make_request", &EditorAssetLibraryItemDownload::_make_request);

	ADD_SIGNAL(MethodInfo("install_asset", PropertyInfo(Variant::STRING, "zip_path"), PropertyInfo(Variant::STRING, "name")));
}

EditorAssetLibraryItemDownload::EditorAssetLibraryItemDownload() {

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	icon = memnew(TextureRect);
	hb->add_child(icon);

	VBoxContainer *vb = memnew(VBoxContainer);
	hb->add_child(vb);
	vb->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	vb->add_child(title_hb);
	title = memnew(Label);
	title_hb->add_child(title);
	title->set_h_size_flags(SIZE_EXPAND_FILL);

	dismiss = memnew(TextureButton);
	dismiss->connect("pressed", this, "_close");
	title_hb->add_child(dismiss);

	title->set_clip_text(true);

	vb->add_spacer();

	status = memnew(Label(TTR("Idle")));
	vb->add_child(status);
	status->add_color_override("font_color", Color(0.5, 0.5, 0.5));
	progress = memnew(ProgressBar);
	vb->add_child(progress);

	HBoxContainer *hb2 = memnew(HBoxContainer);
	vb->add_child(hb2);
	hb2->add_spacer();

	install = memnew(Button);
	install->set_text(TTR("Install"));
	install->set_disabled(true);
	install->connect("pressed", this, "_install");

	retry = memnew(Button);
	retry->set_text(TTR("Retry"));
	retry->connect("pressed", this, "_make_request");

	hb2->add_child(retry);
	hb2->add_child(install);
	set_custom_minimum_size(Size2(250, 0));

	download = memnew(HTTPRequest);
	add_child(download);
	download->connect("request_completed", this, "_http_download_completed");

	download_error = memnew(AcceptDialog);
	add_child(download_error);
	download_error->set_title(TTR("Download Error"));

	asset_installer = memnew(EditorAssetInstaller);
	add_child(asset_installer);

	prev_status = -1;

	external_install = false;
}

////////////////////////////////////////////////////////////////////////////////
void EditorAssetLibrary::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_READY: {

			TextureRect *tf = memnew(TextureRect);
			tf->set_texture(get_icon("Error", "EditorIcons"));
			reverse->set_icon(get_icon("Sort", "EditorIcons"));

			error_hb->add_child(tf);
			error_label->raise();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {

			if (is_visible()) {
				_repository_changed(0); // Update when shown for the first time
			}
		} break;

		case NOTIFICATION_PROCESS: {

			HTTPClient::Status s = request->get_http_client_status();
			bool visible = s != HTTPClient::STATUS_DISCONNECTED;

			if (visible != load_status->is_visible()) {
				load_status->set_visible(visible);
			}

			if (visible) {
				switch (s) {

					case HTTPClient::STATUS_RESOLVING: {
						load_status->set_value(0.1);
					} break;
					case HTTPClient::STATUS_CONNECTING: {
						load_status->set_value(0.2);
					} break;
					case HTTPClient::STATUS_REQUESTING: {
						load_status->set_value(0.3);
					} break;
					case HTTPClient::STATUS_BODY: {
						load_status->set_value(0.4);
					} break;
					default: {}
				}
			}

			bool no_downloads = downloads_hb->get_child_count() == 0;
			if (no_downloads == downloads_scroll->is_visible()) {
				downloads_scroll->set_visible(!no_downloads);
			}

		} break;
		case NOTIFICATION_THEME_CHANGED: {

			library_scroll_bg->add_style_override("panel", get_stylebox("bg", "Tree"));
		} break;
	}
}

void EditorAssetLibrary::_install_asset() {

	ERR_FAIL_COND(!description);

	for (int i = 0; i < downloads_hb->get_child_count(); i++) {

		EditorAssetLibraryItemDownload *d = Object::cast_to<EditorAssetLibraryItemDownload>(downloads_hb->get_child(i));
		if (d && d->get_asset_id() == description->get_asset_id()) {

			if (EditorNode::get_singleton() != NULL)
				EditorNode::get_singleton()->show_warning(TTR("Download for this asset is already in progress!"));
			return;
		}
	}

	EditorAssetLibraryItemDownload *download = memnew(EditorAssetLibraryItemDownload);
	downloads_hb->add_child(download);
	download->configure(description->get_title(), description->get_asset_id(), description->get_preview_icon(), description->get_download_url(), description->get_sha256());

	if (templates_only) {
		download->set_external_install(true);
		download->connect("install_asset", this, "_install_external_asset");
	}
}

const char *EditorAssetLibrary::sort_key[SORT_MAX] = {
	"rating",
	"downloads",
	"name",
	"cost",
	"updated"
};

const char *EditorAssetLibrary::sort_text[SORT_MAX] = {
	"Rating",
	"Downloads",
	"Name",
	"Cost",
	"Updated"
};

const char *EditorAssetLibrary::support_key[SUPPORT_MAX] = {
	"official",
	"community",
	"testing"
};

void EditorAssetLibrary::_select_author(int p_id) {

	//opemn author window
}

void EditorAssetLibrary::_select_category(int p_id) {

	for (int i = 0; i < categories->get_item_count(); i++) {

		if (i == 0)
			continue;
		int id = categories->get_item_metadata(i);
		if (id == p_id) {
			categories->select(i);
			_search();
			break;
		}
	}
}
void EditorAssetLibrary::_select_asset(int p_id) {

	_api_request("asset/" + itos(p_id), REQUESTING_ASSET);

	/*
	if (description) {
		memdelete(description);
	}


	description = memnew( EditorAssetLibraryItemDescription );
	add_child(description);
	description->popup_centered_minsize();*/
}

void EditorAssetLibrary::_image_update(bool use_cache, bool final, const PoolByteArray &p_data, int p_queue_id) {
	Object *obj = ObjectDB::get_instance(image_queue[p_queue_id].target);

	if (obj) {
		bool image_set = false;
		PoolByteArray image_data = p_data;

		if (use_cache) {
			String cache_filename_base = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp").plus_file("assetimage_" + image_queue[p_queue_id].image_url.md5_text());

			FileAccess *file = FileAccess::open(cache_filename_base + ".data", FileAccess::READ);

			if (file) {
				PoolByteArray cached_data;
				int len = file->get_32();
				cached_data.resize(len);

				PoolByteArray::Write w = cached_data.write();
				file->get_buffer(w.ptr(), len);

				image_data = cached_data;
				file->close();
			}
		}

		int len = image_data.size();
		PoolByteArray::Read r = image_data.read();
		Ref<Image> image = Ref<Image>(memnew(Image(r.ptr(), len)));

		if (!image->empty()) {
			float max_height = 10000;
			switch (image_queue[p_queue_id].image_type) {
				case IMAGE_QUEUE_ICON: max_height = 80; break;
				case IMAGE_QUEUE_THUMBNAIL: max_height = 80; break;
				case IMAGE_QUEUE_SCREENSHOT: max_height = 345; break;
			}
			float scale_ratio = max_height / image->get_height();
			if (scale_ratio < 1) {
				image->resize(image->get_width() * scale_ratio, image->get_height() * scale_ratio, Image::INTERPOLATE_CUBIC);
			}

			Ref<ImageTexture> tex;
			tex.instance();
			tex->create_from_image(image);

			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, tex);
			image_set = true;
		}

		if (!image_set && final) {
			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, get_icon("ErrorSign", "EditorIcons"));
		}
	}
}

void EditorAssetLibrary::_image_request_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data, int p_queue_id) {

	ERR_FAIL_COND(!image_queue.has(p_queue_id));

	if (p_status == HTTPRequest::RESULT_SUCCESS) {

		print_line("GOT IMAGE YAY!");

		if (p_code != HTTPClient::RESPONSE_NOT_MODIFIED) {
			for (int i = 0; i < headers.size(); i++) {
				if (headers[i].findn("ETag:") == 0) { // Save etag
					String cache_filename_base = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp").plus_file("assetimage_" + image_queue[p_queue_id].image_url.md5_text());
					String new_etag = headers[i].substr(headers[i].find(":") + 1, headers[i].length()).strip_edges();
					FileAccess *file;

					file = FileAccess::open(cache_filename_base + ".etag", FileAccess::WRITE);
					if (file) {
						file->store_line(new_etag);
						file->close();
					}

					int len = p_data.size();
					PoolByteArray::Read r = p_data.read();
					file = FileAccess::open(cache_filename_base + ".data", FileAccess::WRITE);
					if (file) {
						file->store_32(len);
						file->store_buffer(r.ptr(), len);
						file->close();
					}

					break;
				}
			}
		}
		_image_update(p_code == HTTPClient::RESPONSE_NOT_MODIFIED, true, p_data, p_queue_id);

	} else {
		WARN_PRINTS("Error getting PNG file for asset id " + itos(image_queue[p_queue_id].asset_id));
		Object *obj = ObjectDB::get_instance(image_queue[p_queue_id].target);
		if (obj) {
			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, get_icon("ErrorSign", "EditorIcons"));
		}
	}

	image_queue[p_queue_id].request->queue_delete();
	image_queue.erase(p_queue_id);

	_update_image_queue();
}

void EditorAssetLibrary::_update_image_queue() {

	int max_images = 2;
	int current_images = 0;

	List<int> to_delete;
	for (Map<int, ImageQueue>::Element *E = image_queue.front(); E; E = E->next()) {
		if (!E->get().active && current_images < max_images) {

			String cache_filename_base = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp").plus_file("assetimage_" + E->get().image_url.md5_text());
			Vector<String> headers;

			if (FileAccess::exists(cache_filename_base + ".etag") && FileAccess::exists(cache_filename_base + ".data")) {
				FileAccess *file = FileAccess::open(cache_filename_base + ".etag", FileAccess::READ);
				if (file) {
					headers.push_back("If-None-Match: " + file->get_line());
					file->close();
				}
			}

			print_line("REQUEST ICON FOR: " + itos(E->get().asset_id));
			Error err = E->get().request->request(E->get().image_url, headers);
			if (err != OK) {
				to_delete.push_back(E->key());
			} else {
				E->get().active = true;
			}
			current_images++;
		} else if (E->get().active) {
			current_images++;
		}
	}

	while (to_delete.size()) {
		image_queue[to_delete.front()->get()].request->queue_delete();
		image_queue.erase(to_delete.front()->get());
		to_delete.pop_front();
	}
}

void EditorAssetLibrary::_request_image(ObjectID p_for, String p_image_url, ImageType p_type, int p_image_index) {

	ImageQueue iq;
	iq.image_url = p_image_url;
	iq.image_index = p_image_index;
	iq.image_type = p_type;
	iq.request = memnew(HTTPRequest);

	iq.target = p_for;
	iq.queue_id = ++last_queue_id;
	iq.active = false;

	iq.request->connect("request_completed", this, "_image_request_completed", varray(iq.queue_id));

	image_queue[iq.queue_id] = iq;

	add_child(iq.request);

	_image_update(true, false, PoolByteArray(), iq.queue_id);
	_update_image_queue();
}

void EditorAssetLibrary::_repository_changed(int p_repository_id) {
	host = repository->get_item_metadata(p_repository_id);
	print_line(".." + host);
	if (templates_only) {
		_api_request("configure", REQUESTING_CONFIG, "?type=project");
	} else {
		_api_request("configure", REQUESTING_CONFIG);
	}
}

void EditorAssetLibrary::_support_toggled(int p_support) {
	support->get_popup()->set_item_checked(p_support, !support->get_popup()->is_item_checked(p_support));
	_search();
}

void EditorAssetLibrary::_rerun_search(int p_ignore) {
	_search();
}

void EditorAssetLibrary::_search(int p_page) {

	String args;

	if (templates_only) {
		args += "?type=project&";
	} else {
		args += "?";
	}
	args += String() + "sort=" + sort_key[sort->get_selected()];

	String support_list;
	for (int i = 0; i < SUPPORT_MAX; i++) {
		if (support->get_popup()->is_item_checked(i)) {
			support_list += String(support_key[i]) + "+";
		}
	}
	if (support_list != String()) {
		args += "&support=" + support_list.substr(0, support_list.length() - 1);
	}

	if (categories->get_selected() > 0) {

		args += "&category=" + itos(categories->get_item_metadata(categories->get_selected()));
	}

	if (reverse->is_pressed()) {

		args += "&reverse=true";
	}

	if (filter->get_text() != String()) {
		args += "&filter=" + filter->get_text().http_escape();
	}

	if (p_page > 0) {
		args += "&page=" + itos(p_page);
	}

	_api_request("asset", REQUESTING_SEARCH, args);
}

HBoxContainer *EditorAssetLibrary::_make_pages(int p_page, int p_page_count, int p_page_len, int p_total_items, int p_current_items) {

	HBoxContainer *hbc = memnew(HBoxContainer);

	//do the mario
	int from = p_page - 5;
	if (from < 0)
		from = 0;
	int to = from + 10;
	if (to > p_page_count)
		to = p_page_count;

	Color gray = Color(0.65, 0.65, 0.65);

	hbc->add_spacer();
	hbc->add_constant_override("separation", 10);

	if (p_page != 0) {
		LinkButton *first = memnew(LinkButton);
		first->set_text(TTR("first"));
		first->add_color_override("font_color", gray);
		first->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		first->connect("pressed", this, "_search", varray(0));
		hbc->add_child(first);
	}

	if (p_page > 0) {
		LinkButton *prev = memnew(LinkButton);
		prev->set_text(TTR("prev"));
		prev->add_color_override("font_color", gray);
		prev->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		prev->connect("pressed", this, "_search", varray(p_page - 1));
		hbc->add_child(prev);
	}

	for (int i = from; i < to; i++) {

		if (i == p_page) {

			Label *current = memnew(Label);
			current->set_text(itos(i + 1));
			hbc->add_child(current);
		} else {

			LinkButton *current = memnew(LinkButton);
			current->add_color_override("font_color", gray);
			current->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
			current->set_text(itos(i + 1));
			current->connect("pressed", this, "_search", varray(i));

			hbc->add_child(current);
		}
	}

	if (p_page < p_page_count - 1) {
		LinkButton *next = memnew(LinkButton);
		next->set_text(TTR("next"));
		next->add_color_override("font_color", gray);
		next->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		next->connect("pressed", this, "_search", varray(p_page + 1));

		hbc->add_child(next);
	}

	if (p_page != p_page_count - 1) {
		LinkButton *last = memnew(LinkButton);
		last->set_text(TTR("last"));
		last->add_color_override("font_color", gray);
		last->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		hbc->add_child(last);
		last->connect("pressed", this, "_search", varray(p_page_count - 1));
	}

	Label *totals = memnew(Label);
	totals->set_text("( " + itos(from * p_page_len) + " - " + itos(from * p_page_len + p_current_items - 1) + " / " + itos(p_total_items) + " )");
	hbc->add_child(totals);

	hbc->add_spacer();

	return hbc;
}

void EditorAssetLibrary::_api_request(const String &p_request, RequestType p_request_type, const String &p_arguments) {

	if (requesting != REQUESTING_NONE) {
		request->cancel_request();
	}

	requesting = p_request_type;

	error_hb->hide();
	request->request(host + "/" + p_request + p_arguments);
}

void EditorAssetLibrary::_http_request_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data) {

	String str;

	{
		int datalen = p_data.size();
		PoolByteArray::Read r = p_data.read();
		str.parse_utf8((const char *)r.ptr(), datalen);
	}

	bool error_abort = true;

	switch (p_status) {

		case HTTPRequest::RESULT_CANT_RESOLVE: {
			error_label->set_text(TTR("Can't resolve hostname:") + " " + host);
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: {
			error_label->set_text(TTR("Connection error, please try again."));
		} break;
		case HTTPRequest::RESULT_SSL_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT: {
			error_label->set_text(TTR("Can't connect to host:") + " " + host);
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			error_label->set_text(TTR("No response from host:") + " " + host);
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			error_label->set_text(TTR("Request failed, return code:") + " " + itos(p_code));
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			error_label->set_text(TTR("Request failed, too many redirects"));

		} break;
		default: {
			if (p_code != 200) {
				error_label->set_text(TTR("Request failed, return code:") + " " + itos(p_code));
			} else {

				error_abort = false;
			}
		} break;
	}

	if (error_abort) {
		error_hb->show();
		return;
	}

	print_line("response: " + itos(p_status) + " code: " + itos(p_code));

	Dictionary d;
	{
		Variant js;
		String errs;
		int errl;
		JSON::parse(str, js, errs, errl);
		d = js;
	}

	print_line(Variant(d).get_construct_string());

	RequestType requested = requesting;
	requesting = REQUESTING_NONE;

	switch (requested) {
		case REQUESTING_CONFIG: {

			categories->clear();
			categories->add_item(TTR("All"));
			categories->set_item_metadata(0, 0);
			if (d.has("categories")) {
				Array clist = d["categories"];
				for (int i = 0; i < clist.size(); i++) {
					Dictionary cat = clist[i];
					if (!cat.has("name") || !cat.has("id"))
						continue;
					String name = cat["name"];
					int id = cat["id"];
					categories->add_item(name);
					categories->set_item_metadata(categories->get_item_count() - 1, id);
					category_map[cat["id"]] = name;
				}
			}

			_search();
		} break;
		case REQUESTING_SEARCH: {
			if (asset_items) {
				memdelete(asset_items);
			}

			if (asset_top_page) {
				memdelete(asset_top_page);
			}

			if (asset_bottom_page) {
				memdelete(asset_bottom_page);
			}

			int page = 0;
			int pages = 1;
			int page_len = 10;
			int total_items = 1;
			Array result;

			if (d.has("page")) {
				page = d["page"];
			}
			if (d.has("pages")) {
				pages = d["pages"];
			}
			if (d.has("page_length")) {
				page_len = d["page_length"];
			}
			if (d.has("total")) {
				total_items = d["total"];
			}
			if (d.has("result")) {
				result = d["result"];
			}

			asset_top_page = _make_pages(page, pages, page_len, total_items, result.size());
			library_vb->add_child(asset_top_page);

			asset_items = memnew(GridContainer);
			asset_items->set_columns(2);
			asset_items->add_constant_override("hseparation", 10);
			asset_items->add_constant_override("vseparation", 10);

			library_vb->add_child(asset_items);

			asset_bottom_page = _make_pages(page, pages, page_len, total_items, result.size());
			library_vb->add_child(asset_bottom_page);

			for (int i = 0; i < result.size(); i++) {

				Dictionary r = result[i];

				ERR_CONTINUE(!r.has("title"));
				ERR_CONTINUE(!r.has("asset_id"));
				ERR_CONTINUE(!r.has("author"));
				ERR_CONTINUE(!r.has("author_id"));
				ERR_CONTINUE(!r.has("category_id"));
				ERR_FAIL_COND(!category_map.has(r["category_id"]));
				ERR_CONTINUE(!r.has("rating"));
				ERR_CONTINUE(!r.has("cost"));

				EditorAssetLibraryItem *item = memnew(EditorAssetLibraryItem);
				asset_items->add_child(item);
				item->configure(r["title"], r["asset_id"], category_map[r["category_id"]], r["category_id"], r["author"], r["author_id"], r["rating"], r["cost"]);
				item->connect("asset_selected", this, "_select_asset");
				item->connect("author_selected", this, "_select_author");
				item->connect("category_selected", this, "_select_category");

				if (r.has("icon_url") && r["icon_url"] != "") {
					_request_image(item->get_instance_id(), r["icon_url"], IMAGE_QUEUE_ICON, 0);
				}
			}
		} break;
		case REQUESTING_ASSET: {
			Dictionary r = d;

			ERR_FAIL_COND(!r.has("title"));
			ERR_FAIL_COND(!r.has("asset_id"));
			ERR_FAIL_COND(!r.has("author"));
			ERR_FAIL_COND(!r.has("author_id"));
			ERR_FAIL_COND(!r.has("version"));
			ERR_FAIL_COND(!r.has("version_string"));
			ERR_FAIL_COND(!r.has("category_id"));
			ERR_FAIL_COND(!category_map.has(r["category_id"]));
			ERR_FAIL_COND(!r.has("rating"));
			ERR_FAIL_COND(!r.has("cost"));
			ERR_FAIL_COND(!r.has("description"));
			ERR_FAIL_COND(!r.has("download_url"));
			ERR_FAIL_COND(!r.has("download_hash"));
			ERR_FAIL_COND(!r.has("browse_url"));

			if (description) {
				memdelete(description);
			}

			description = memnew(EditorAssetLibraryItemDescription);
			add_child(description);
			description->popup_centered_minsize();
			description->connect("confirmed", this, "_install_asset");

			description->configure(r["title"], r["asset_id"], category_map[r["category_id"]], r["category_id"], r["author"], r["author_id"], r["rating"], r["cost"], r["version"], r["version_string"], r["description"], r["download_url"], r["browse_url"], r["download_hash"]);
			/*item->connect("asset_selected",this,"_select_asset");
			item->connect("author_selected",this,"_select_author");
			item->connect("category_selected",this,"_category_selected");*/

			if (r.has("icon_url") && r["icon_url"] != "") {
				_request_image(description->get_instance_id(), r["icon_url"], IMAGE_QUEUE_ICON, 0);
			}

			if (d.has("previews")) {
				Array previews = d["previews"];

				for (int i = 0; i < previews.size(); i++) {

					Dictionary p = previews[i];

					ERR_CONTINUE(!p.has("type"));
					ERR_CONTINUE(!p.has("link"));

					bool is_video = p.has("type") && String(p["type"]) == "video";
					String video_url;
					if (is_video && p.has("link")) {
						video_url = p["link"];
					}

					description->add_preview(i, is_video, video_url);

					if (p.has("thumbnail")) {
						_request_image(description->get_instance_id(), p["thumbnail"], IMAGE_QUEUE_THUMBNAIL, i);
					}
					if (is_video) {
						//_request_image(description->get_instance_id(),p["link"],IMAGE_QUEUE_SCREENSHOT,i);
					} else {
						_request_image(description->get_instance_id(), p["link"], IMAGE_QUEUE_SCREENSHOT, i);
					}
				}
			}
		} break;
		default: break;
	}
}

void EditorAssetLibrary::_asset_file_selected(const String &p_file) {

	if (asset_installer) {
		memdelete(asset_installer);
		asset_installer = NULL;
	}

	asset_installer = memnew(EditorAssetInstaller);
	add_child(asset_installer);
	asset_installer->open(p_file);
}

void EditorAssetLibrary::_asset_open() {

	asset_open->popup_centered_ratio();
}

void EditorAssetLibrary::_manage_plugins() {

	ProjectSettingsEditor::get_singleton()->popup_project_settings();
	ProjectSettingsEditor::get_singleton()->set_plugins_page();
}

void EditorAssetLibrary::_install_external_asset(String p_zip_path, String p_title) {

	emit_signal("install_asset", p_zip_path, p_title);
}

void EditorAssetLibrary::_bind_methods() {

	ClassDB::bind_method("_http_request_completed", &EditorAssetLibrary::_http_request_completed);
	ClassDB::bind_method("_select_asset", &EditorAssetLibrary::_select_asset);
	ClassDB::bind_method("_select_author", &EditorAssetLibrary::_select_author);
	ClassDB::bind_method("_select_category", &EditorAssetLibrary::_select_category);
	ClassDB::bind_method("_image_request_completed", &EditorAssetLibrary::_image_request_completed);
	ClassDB::bind_method("_search", &EditorAssetLibrary::_search, DEFVAL(0));
	ClassDB::bind_method("_install_asset", &EditorAssetLibrary::_install_asset);
	ClassDB::bind_method("_manage_plugins", &EditorAssetLibrary::_manage_plugins);
	ClassDB::bind_method("_asset_open", &EditorAssetLibrary::_asset_open);
	ClassDB::bind_method("_asset_file_selected", &EditorAssetLibrary::_asset_file_selected);
	ClassDB::bind_method("_repository_changed", &EditorAssetLibrary::_repository_changed);
	ClassDB::bind_method("_support_toggled", &EditorAssetLibrary::_support_toggled);
	ClassDB::bind_method("_rerun_search", &EditorAssetLibrary::_rerun_search);
	ClassDB::bind_method("_install_external_asset", &EditorAssetLibrary::_install_external_asset);

	ADD_SIGNAL(MethodInfo("install_asset", PropertyInfo(Variant::STRING, "zip_path"), PropertyInfo(Variant::STRING, "name")));
}

EditorAssetLibrary::EditorAssetLibrary(bool p_templates_only) {

	templates_only = p_templates_only;

	VBoxContainer *library_main = memnew(VBoxContainer);

	add_child(library_main);

	HBoxContainer *search_hb = memnew(HBoxContainer);

	library_main->add_child(search_hb);
	library_main->add_constant_override("separation", 10);

	search_hb->add_child(memnew(Label(TTR("Search:") + " ")));
	filter = memnew(LineEdit);
	search_hb->add_child(filter);
	filter->set_h_size_flags(SIZE_EXPAND_FILL);
	filter->connect("text_entered", this, "_search");
	search = memnew(Button(TTR("Search")));
	search->connect("pressed", this, "_search");
	search_hb->add_child(search);

	if (!p_templates_only)
		search_hb->add_child(memnew(VSeparator));

	Button *open_asset = memnew(Button);
	open_asset->set_text(TTR("Import"));
	search_hb->add_child(open_asset);
	open_asset->connect("pressed", this, "_asset_open");

	Button *plugins = memnew(Button);
	plugins->set_text(TTR("Plugins"));
	search_hb->add_child(plugins);
	plugins->connect("pressed", this, "_manage_plugins");

	if (p_templates_only) {
		open_asset->hide();
		plugins->hide();
	}

	HBoxContainer *search_hb2 = memnew(HBoxContainer);
	library_main->add_child(search_hb2);

	search_hb2->add_child(memnew(Label(TTR("Sort:") + " ")));
	sort = memnew(OptionButton);
	for (int i = 0; i < SORT_MAX; i++) {
		sort->add_item(sort_text[i]);
	}

	search_hb2->add_child(sort);

	sort->set_h_size_flags(SIZE_EXPAND_FILL);
	sort->connect("item_selected", this, "_rerun_search");

	reverse = memnew(ToolButton);
	reverse->set_toggle_mode(true);
	reverse->connect("toggled", this, "_rerun_search");
	//reverse->set_text(TTR("Reverse"));
	search_hb2->add_child(reverse);

	search_hb2->add_child(memnew(VSeparator));

	//search_hb2->add_spacer();

	search_hb2->add_child(memnew(Label(TTR("Category:") + " ")));
	categories = memnew(OptionButton);
	categories->add_item(TTR("All"));
	search_hb2->add_child(categories);
	categories->set_h_size_flags(SIZE_EXPAND_FILL);
	//search_hb2->add_spacer();
	categories->connect("item_selected", this, "_rerun_search");

	search_hb2->add_child(memnew(VSeparator));

	search_hb2->add_child(memnew(Label(TTR("Site:") + " ")));
	repository = memnew(OptionButton);

	// FIXME: Reenable me once GH-7147 is fixed.
	/*
	repository->add_item("godotengine.org");
	repository->set_item_metadata(0, "https://godotengine.org/asset-library/api");
	*/
	repository->add_item("localhost");
	repository->set_item_metadata(/*1*/ 0, "http://127.0.0.1/asset-library/api");
	repository->connect("item_selected", this, "_repository_changed");

	search_hb2->add_child(repository);
	repository->set_h_size_flags(SIZE_EXPAND_FILL);

	search_hb2->add_child(memnew(VSeparator));

	support = memnew(MenuButton);
	search_hb2->add_child(support);
	support->set_text(TTR("Support.."));
	support->get_popup()->add_check_item(TTR("Official"), SUPPORT_OFFICIAL);
	support->get_popup()->add_check_item(TTR("Community"), SUPPORT_COMMUNITY);
	support->get_popup()->add_check_item(TTR("Testing"), SUPPORT_TESTING);
	support->get_popup()->set_item_checked(SUPPORT_OFFICIAL, true);
	support->get_popup()->set_item_checked(SUPPORT_COMMUNITY, true);
	support->get_popup()->connect("id_pressed", this, "_support_toggled");

	/////////

	library_scroll_bg = memnew(PanelContainer);
	library_main->add_child(library_scroll_bg);
	library_scroll_bg->set_v_size_flags(SIZE_EXPAND_FILL);

	library_scroll = memnew(ScrollContainer);
	library_scroll->set_enable_v_scroll(true);
	library_scroll->set_enable_h_scroll(false);

	library_scroll_bg->add_child(library_scroll);

	Ref<StyleBoxEmpty> border2;
	border2.instance();
	border2->set_default_margin(MARGIN_LEFT, 15);
	border2->set_default_margin(MARGIN_RIGHT, 35);
	border2->set_default_margin(MARGIN_BOTTOM, 15);
	border2->set_default_margin(MARGIN_TOP, 15);

	PanelContainer *library_vb_border = memnew(PanelContainer);
	library_scroll->add_child(library_vb_border);
	library_vb_border->add_style_override("panel", border2);
	library_vb_border->set_h_size_flags(SIZE_EXPAND_FILL);
	library_vb_border->set_mouse_filter(MOUSE_FILTER_PASS);

	library_vb = memnew(VBoxContainer);
	library_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	library_vb_border->add_child(library_vb);
	//margin_panel->set_stop_mouse(false);

	asset_top_page = memnew(HBoxContainer);
	library_vb->add_child(asset_top_page);

	asset_items = memnew(GridContainer);
	asset_items->set_columns(2);
	asset_items->add_constant_override("hseparation", 10);
	asset_items->add_constant_override("vseparation", 10);

	library_vb->add_child(asset_items);

	asset_bottom_page = memnew(HBoxContainer);
	library_vb->add_child(asset_bottom_page);

	request = memnew(HTTPRequest);
	add_child(request);
	request->set_use_threads(EDITOR_DEF("asset_library/use_threads", true));
	request->connect("request_completed", this, "_http_request_completed");

	last_queue_id = 0;

	library_vb->add_constant_override("separation", 20);

	load_status = memnew(ProgressBar);
	load_status->set_min(0);
	load_status->set_max(1);
	load_status->set_step(0.001);
	library_main->add_child(load_status);

	error_hb = memnew(HBoxContainer);
	library_main->add_child(error_hb);
	error_label = memnew(Label);
	error_label->add_color_override("color", get_color("error_color", "Editor"));
	error_hb->add_child(error_label);

	description = NULL;

	set_process(true);

	downloads_scroll = memnew(ScrollContainer);
	downloads_scroll->set_enable_h_scroll(true);
	downloads_scroll->set_enable_v_scroll(false);
	library_main->add_child(downloads_scroll);
	downloads_hb = memnew(HBoxContainer);
	downloads_scroll->add_child(downloads_hb);

	asset_open = memnew(EditorFileDialog);

	asset_open->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	asset_open->add_filter("*.zip ; " + TTR("Assets ZIP File"));
	asset_open->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	add_child(asset_open);
	asset_open->connect("file_selected", this, "_asset_file_selected");

	asset_installer = NULL;
}

///////

void AssetLibraryEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		addon_library->show();
	} else {

		addon_library->hide();
	}
}

AssetLibraryEditorPlugin::AssetLibraryEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	addon_library = memnew(EditorAssetLibrary);
	addon_library->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->get_viewport()->add_child(addon_library);
	addon_library->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	addon_library->hide();
}

AssetLibraryEditorPlugin::~AssetLibraryEditorPlugin() {
}
