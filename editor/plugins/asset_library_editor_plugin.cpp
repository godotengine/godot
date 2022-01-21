/*************************************************************************/
/*  asset_library_editor_plugin.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/input/input.h"
#include "core/io/json.h"
#include "core/os/keyboard.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/project_settings_editor.h"

static inline void setup_http_request(HTTPRequest *request) {
	request->set_use_threads(EDITOR_DEF("asset_library/use_threads", true));

	const String proxy_host = EDITOR_DEF("network/http_proxy/host", "");
	const int proxy_port = EDITOR_DEF("network/http_proxy/port", -1);
	request->set_http_proxy(proxy_host, proxy_port);
	request->set_https_proxy(proxy_host, proxy_port);
}

void EditorAssetLibraryItem::configure(const String &p_title, int p_asset_id, const String &p_category, int p_category_id, const String &p_author, int p_author_id, const String &p_cost) {
	title->set_text(p_title);
	asset_id = p_asset_id;
	category->set_text(p_category);
	category_id = p_category_id;
	author->set_text(p_author);
	author_id = p_author_id;
	price->set_text(p_cost);
}

void EditorAssetLibraryItem::set_image(int p_type, int p_index, const Ref<Texture2D> &p_image) {
	ERR_FAIL_COND(p_type != EditorAssetLibrary::IMAGE_QUEUE_ICON);
	ERR_FAIL_COND(p_index != 0);

	icon->set_normal_texture(p_image);
}

void EditorAssetLibraryItem::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		icon->set_normal_texture(get_theme_icon(SNAME("ProjectIconLoading"), SNAME("EditorIcons")));
		category->add_theme_color_override("font_color", Color(0.5, 0.5, 0.5));
		author->add_theme_color_override("font_color", Color(0.5, 0.5, 0.5));
		price->add_theme_color_override("font_color", Color(0.5, 0.5, 0.5));
	}
}

void EditorAssetLibraryItem::_asset_clicked() {
	emit_signal(SNAME("asset_selected"), asset_id);
}

void EditorAssetLibraryItem::_category_clicked() {
	emit_signal(SNAME("category_selected"), category_id);
}

void EditorAssetLibraryItem::_author_clicked() {
	emit_signal(SNAME("author_selected"), author_id);
}

void EditorAssetLibraryItem::_bind_methods() {
	ClassDB::bind_method("set_image", &EditorAssetLibraryItem::set_image);
	ADD_SIGNAL(MethodInfo("asset_selected"));
	ADD_SIGNAL(MethodInfo("category_selected"));
	ADD_SIGNAL(MethodInfo("author_selected"));
}

EditorAssetLibraryItem::EditorAssetLibraryItem() {
	Ref<StyleBoxEmpty> border;
	border.instantiate();
	border->set_default_margin(SIDE_LEFT, 5 * EDSCALE);
	border->set_default_margin(SIDE_RIGHT, 5 * EDSCALE);
	border->set_default_margin(SIDE_BOTTOM, 5 * EDSCALE);
	border->set_default_margin(SIDE_TOP, 5 * EDSCALE);
	add_theme_style_override("panel", border);

	HBoxContainer *hb = memnew(HBoxContainer);
	// Add some spacing to visually separate the icon from the asset details.
	hb->add_theme_constant_override("separation", 15 * EDSCALE);
	add_child(hb);

	icon = memnew(TextureButton);
	icon->set_custom_minimum_size(Size2(64, 64) * EDSCALE);
	icon->set_default_cursor_shape(CURSOR_POINTING_HAND);
	icon->connect("pressed", callable_mp(this, &EditorAssetLibraryItem::_asset_clicked));

	hb->add_child(icon);

	VBoxContainer *vb = memnew(VBoxContainer);

	hb->add_child(vb);
	vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	title = memnew(LinkButton);
	title->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	title->connect("pressed", callable_mp(this, &EditorAssetLibraryItem::_asset_clicked));
	vb->add_child(title);

	category = memnew(LinkButton);
	category->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	category->connect("pressed", callable_mp(this, &EditorAssetLibraryItem::_category_clicked));
	vb->add_child(category);

	author = memnew(LinkButton);
	author->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	author->connect("pressed", callable_mp(this, &EditorAssetLibraryItem::_author_clicked));
	vb->add_child(author);

	price = memnew(Label);
	vb->add_child(price);

	set_custom_minimum_size(Size2(250, 100) * EDSCALE);
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
}

//////////////////////////////////////////////////////////////////////////////

void EditorAssetLibraryItemDescription::set_image(int p_type, int p_index, const Ref<Texture2D> &p_image) {
	switch (p_type) {
		case EditorAssetLibrary::IMAGE_QUEUE_ICON: {
			item->call("set_image", p_type, p_index, p_image);
			icon = p_image;
		} break;
		case EditorAssetLibrary::IMAGE_QUEUE_THUMBNAIL: {
			for (int i = 0; i < preview_images.size(); i++) {
				if (preview_images[i].id == p_index) {
					if (preview_images[i].is_video) {
						Ref<Image> overlay = previews->get_theme_icon(SNAME("PlayOverlay"), SNAME("EditorIcons"))->get_image();
						Ref<Image> thumbnail = p_image->get_image();
						thumbnail = thumbnail->duplicate();
						Point2 overlay_pos = Point2((thumbnail->get_width() - overlay->get_width()) / 2, (thumbnail->get_height() - overlay->get_height()) / 2);

						// Overlay and thumbnail need the same format for `blend_rect` to work.
						thumbnail->convert(Image::FORMAT_RGBA8);

						thumbnail->blend_rect(overlay, overlay->get_used_rect(), overlay_pos);

						Ref<ImageTexture> tex;
						tex.instantiate();
						tex->create_from_image(thumbnail);

						preview_images[i].button->set_icon(tex);
						// Make it clearer that clicking it will open an external link
						preview_images[i].button->set_default_cursor_shape(Control::CURSOR_POINTING_HAND);
					} else {
						preview_images[i].button->set_icon(p_image);
					}
					break;
				}
			}
		} break;
		case EditorAssetLibrary::IMAGE_QUEUE_SCREENSHOT: {
			for (int i = 0; i < preview_images.size(); i++) {
				if (preview_images[i].id == p_index) {
					preview_images.write[i].image = p_image;
					if (preview_images[i].button->is_pressed()) {
						_preview_click(p_index);
					}
					break;
				}
			}
		} break;
	}
}

void EditorAssetLibraryItemDescription::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			previews_bg->add_theme_style_override("panel", previews->get_theme_stylebox(SNAME("normal"), SNAME("TextEdit")));
		} break;
	}
}

void EditorAssetLibraryItemDescription::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_image"), &EditorAssetLibraryItemDescription::set_image);
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
					child_controls_changed();
				}
			} else {
				_link_click(preview_images[i].video_link);
			}
		} else {
			preview_images[i].button->set_pressed(false);
		}
	}
}

void EditorAssetLibraryItemDescription::configure(const String &p_title, int p_asset_id, const String &p_category, int p_category_id, const String &p_author, int p_author_id, const String &p_cost, int p_version, const String &p_version_string, const String &p_description, const String &p_download_url, const String &p_browse_url, const String &p_sha256_hash) {
	asset_id = p_asset_id;
	title = p_title;
	download_url = p_download_url;
	sha256 = p_sha256_hash;
	item->configure(p_title, p_asset_id, p_category, p_category_id, p_author, p_author_id, p_cost);
	description->clear();
	description->add_text(TTR("Version:") + " " + p_version_string + "\n");
	description->add_text(TTR("Contents:") + " ");
	description->push_meta(p_browse_url);
	description->add_text(TTR("View Files"));
	description->pop();
	description->add_text("\n" + TTR("Description:") + "\n\n");
	description->append_text(p_description);
	description->set_selection_enabled(true);
	set_title(p_title);
}

void EditorAssetLibraryItemDescription::add_preview(int p_id, bool p_video, const String &p_url) {
	Preview preview;
	preview.id = p_id;
	preview.video_link = p_url;
	preview.is_video = p_video;
	preview.button = memnew(Button);
	preview.button->set_icon(previews->get_theme_icon(SNAME("ThumbnailWait"), SNAME("EditorIcons")));
	preview.button->set_toggle_mode(true);
	preview.button->connect("pressed", callable_mp(this, &EditorAssetLibraryItemDescription::_preview_click), varray(p_id));
	preview_hb->add_child(preview.button);
	if (!p_video) {
		preview.image = previews->get_theme_icon(SNAME("ThumbnailWait"), SNAME("EditorIcons"));
	}
	preview_images.push_back(preview);
	if (preview_images.size() == 1 && !p_video) {
		_preview_click(p_id);
	}
}

EditorAssetLibraryItemDescription::EditorAssetLibraryItemDescription() {
	HBoxContainer *hbox = memnew(HBoxContainer);
	add_child(hbox);
	VBoxContainer *desc_vbox = memnew(VBoxContainer);
	hbox->add_child(desc_vbox);
	hbox->add_theme_constant_override("separation", 15 * EDSCALE);

	item = memnew(EditorAssetLibraryItem);

	desc_vbox->add_child(item);
	desc_vbox->set_custom_minimum_size(Size2(440 * EDSCALE, 0));

	description = memnew(RichTextLabel);
	desc_vbox->add_child(description);
	description->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	description->connect("meta_clicked", callable_mp(this, &EditorAssetLibraryItemDescription::_link_click));
	description->add_theme_constant_override("line_separation", Math::round(5 * EDSCALE));

	VBoxContainer *previews_vbox = memnew(VBoxContainer);
	hbox->add_child(previews_vbox);
	previews_vbox->add_theme_constant_override("separation", 15 * EDSCALE);
	previews_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	preview = memnew(TextureRect);
	previews_vbox->add_child(preview);
	preview->set_ignore_texture_size(true);
	preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	preview->set_custom_minimum_size(Size2(640 * EDSCALE, 345 * EDSCALE));

	previews_bg = memnew(PanelContainer);
	previews_vbox->add_child(previews_bg);
	previews_bg->set_custom_minimum_size(Size2(640 * EDSCALE, 101 * EDSCALE));

	previews = memnew(ScrollContainer);
	previews_bg->add_child(previews);
	previews->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	preview_hb = memnew(HBoxContainer);
	preview_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	previews->add_child(preview_hb);
	get_ok_button()->set_text(TTR("Download"));
	get_cancel_button()->set_text(TTR("Close"));
}

///////////////////////////////////////////////////////////////////////////////////

void EditorAssetLibraryItemDownload::_http_download_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data) {
	String error_text;

	switch (p_status) {
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED: {
			error_text = TTR("Connection error, please try again.");
			status->set_text(TTR("Can't connect."));
		} break;
		case HTTPRequest::RESULT_CANT_CONNECT:
		case HTTPRequest::RESULT_SSL_HANDSHAKE_ERROR: {
			error_text = TTR("Can't connect to host:") + " " + host;
			status->set_text(TTR("Can't connect."));
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			error_text = TTR("No response from host:") + " " + host;
			status->set_text(TTR("No response."));
		} break;
		case HTTPRequest::RESULT_CANT_RESOLVE: {
			error_text = TTR("Can't resolve hostname:") + " " + host;
			status->set_text(TTR("Can't resolve."));
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			error_text = TTR("Request failed, return code:") + " " + itos(p_code);
			status->set_text(TTR("Request failed."));
		} break;
		case HTTPRequest::RESULT_DOWNLOAD_FILE_CANT_OPEN:
		case HTTPRequest::RESULT_DOWNLOAD_FILE_WRITE_ERROR: {
			error_text = TTR("Cannot save response to:") + " " + download->get_download_file();
			status->set_text(TTR("Write error."));
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			error_text = TTR("Request failed, too many redirects");
			status->set_text(TTR("Redirect loop."));
		} break;
		case HTTPRequest::RESULT_TIMEOUT: {
			error_text = TTR("Request failed, timeout");
			status->set_text(TTR("Timeout."));
		} break;
		default: {
			if (p_code != 200) {
				error_text = TTR("Request failed, return code:") + " " + itos(p_code);
				status->set_text(TTR("Failed:") + " " + itos(p_code));
			} else if (!sha256.is_empty()) {
				String download_sha256 = FileAccess::get_sha256(download->get_download_file());
				if (sha256 != download_sha256) {
					error_text = TTR("Bad download hash, assuming file has been tampered with.") + "\n";
					error_text += TTR("Expected:") + " " + sha256 + "\n" + TTR("Got:") + " " + download_sha256;
					status->set_text(TTR("Failed SHA-256 hash check"));
				}
			}
		} break;
	}

	if (!error_text.is_empty()) {
		download_error->set_text(TTR("Asset Download Error:") + "\n" + error_text);
		download_error->popup_centered();
		// Let the user retry the download.
		retry_button->show();
		return;
	}

	install_button->set_disabled(false);
	status->set_text(TTR("Ready to install!"));
	// Make the progress bar invisible but don't reflow other Controls around it.
	progress->set_modulate(Color(0, 0, 0, 0));

	set_process(false);

	// Automatically prompt for installation once the download is completed.
	install();
}

void EditorAssetLibraryItemDownload::configure(const String &p_title, int p_asset_id, const Ref<Texture2D> &p_preview, const String &p_download_url, const String &p_sha256_hash) {
	title->set_text(p_title);
	icon->set_texture(p_preview);
	asset_id = p_asset_id;
	if (!p_preview.is_valid()) {
		icon->set_texture(get_theme_icon(SNAME("FileBrokenBigThumb"), SNAME("EditorIcons")));
	}
	host = p_download_url;
	sha256 = p_sha256_hash;
	_make_request();
}

void EditorAssetLibraryItemDownload::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("AssetLib")));
			status->add_theme_color_override("font_color", get_theme_color(SNAME("status_color"), SNAME("AssetLib")));
			dismiss_button->set_normal_texture(get_theme_icon(SNAME("dismiss"), SNAME("AssetLib")));
		} break;
		case NOTIFICATION_PROCESS: {
			// Make the progress bar visible again when retrying the download.
			progress->set_modulate(Color(1, 1, 1, 1));

			if (download->get_downloaded_bytes() > 0) {
				progress->set_max(download->get_body_size());
				progress->set_value(download->get_downloaded_bytes());
			}

			int cstatus = download->get_http_client_status();

			if (cstatus == HTTPClient::STATUS_BODY) {
				if (download->get_body_size() > 0) {
					status->set_text(vformat(
							TTR("Downloading (%s / %s)..."),
							String::humanize_size(download->get_downloaded_bytes()),
							String::humanize_size(download->get_body_size())));
				} else {
					// Total file size is unknown, so it cannot be displayed.
					progress->set_modulate(Color(0, 0, 0, 0));
					status->set_text(vformat(
							TTR("Downloading...") + " (%s)",
							String::humanize_size(download->get_downloaded_bytes())));
				}
			}

			if (cstatus != prev_status) {
				switch (cstatus) {
					case HTTPClient::STATUS_RESOLVING: {
						status->set_text(TTR("Resolving..."));
						progress->set_max(1);
						progress->set_value(0);
					} break;
					case HTTPClient::STATUS_CONNECTING: {
						status->set_text(TTR("Connecting..."));
						progress->set_max(1);
						progress->set_value(0);
					} break;
					case HTTPClient::STATUS_REQUESTING: {
						status->set_text(TTR("Requesting..."));
						progress->set_max(1);
						progress->set_value(0);
					} break;
					default: {
					}
				}
				prev_status = cstatus;
			}
		} break;
	}
}

void EditorAssetLibraryItemDownload::_close() {
	// Clean up downloaded file.
	DirAccess::remove_file_or_error(download->get_download_file());
	queue_delete();
}

bool EditorAssetLibraryItemDownload::can_install() const {
	return !install_button->is_disabled();
}

void EditorAssetLibraryItemDownload::install() {
	String file = download->get_download_file();

	if (external_install) {
		emit_signal(SNAME("install_asset"), file, title->get_text());
		return;
	}

	asset_installer->set_asset_name(title->get_text());
	asset_installer->open(file, 1);
}

void EditorAssetLibraryItemDownload::_make_request() {
	// Hide the Retry button if we've just pressed it.
	retry_button->hide();

	download->cancel_request();
	download->set_download_file(EditorPaths::get_singleton()->get_cache_dir().plus_file("tmp_asset_" + itos(asset_id)) + ".zip");

	Error err = download->request(host);
	if (err != OK) {
		status->set_text(TTR("Error making request"));
	} else {
		set_process(true);
	}
}

void EditorAssetLibraryItemDownload::_bind_methods() {
	ADD_SIGNAL(MethodInfo("install_asset", PropertyInfo(Variant::STRING, "zip_path"), PropertyInfo(Variant::STRING, "name")));
}

EditorAssetLibraryItemDownload::EditorAssetLibraryItemDownload() {
	panel = memnew(PanelContainer);
	add_child(panel);

	HBoxContainer *hb = memnew(HBoxContainer);
	panel->add_child(hb);
	icon = memnew(TextureRect);
	icon->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	icon->set_v_size_flags(0);
	hb->add_child(icon);

	VBoxContainer *vb = memnew(VBoxContainer);
	hb->add_child(vb);
	vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	vb->add_child(title_hb);
	title = memnew(Label);
	title_hb->add_child(title);
	title->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	dismiss_button = memnew(TextureButton);
	dismiss_button->connect("pressed", callable_mp(this, &EditorAssetLibraryItemDownload::_close));
	title_hb->add_child(dismiss_button);

	title->set_clip_text(true);

	vb->add_spacer();

	status = memnew(Label(TTR("Idle")));
	vb->add_child(status);
	progress = memnew(ProgressBar);
	vb->add_child(progress);

	HBoxContainer *hb2 = memnew(HBoxContainer);
	vb->add_child(hb2);
	hb2->add_spacer();

	install_button = memnew(Button);
	install_button->set_text(TTR("Install..."));
	install_button->set_disabled(true);
	install_button->connect("pressed", callable_mp(this, &EditorAssetLibraryItemDownload::install));

	retry_button = memnew(Button);
	retry_button->set_text(TTR("Retry"));
	retry_button->connect("pressed", callable_mp(this, &EditorAssetLibraryItemDownload::_make_request));
	// Only show the Retry button in case of a failure.
	retry_button->hide();

	hb2->add_child(retry_button);
	hb2->add_child(install_button);
	set_custom_minimum_size(Size2(310, 0) * EDSCALE);

	download = memnew(HTTPRequest);
	panel->add_child(download);
	download->connect("request_completed", callable_mp(this, &EditorAssetLibraryItemDownload::_http_download_completed));
	setup_http_request(download);

	download_error = memnew(AcceptDialog);
	panel->add_child(download_error);
	download_error->set_title(TTR("Download Error"));

	asset_installer = memnew(EditorAssetInstaller);
	panel->add_child(asset_installer);
	asset_installer->connect("confirmed", callable_mp(this, &EditorAssetLibraryItemDownload::_close));

	prev_status = -1;

	external_install = false;
}

////////////////////////////////////////////////////////////////////////////////
void EditorAssetLibrary::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			error_label->raise();
		} break;
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			error_tr->set_texture(get_theme_icon(SNAME("Error"), SNAME("EditorIcons")));
			filter->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
			library_scroll_bg->add_theme_style_override("panel", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
			downloads_scroll->add_theme_style_override("bg", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
			error_label->add_theme_color_override("color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				// Focus the search box automatically when switching to the Templates tab (in the Project Manager)
				// or switching to the AssetLib tab (in the editor).
				// The Project Manager's project filter box is automatically focused in the project manager code.
				filter->grab_focus();

				if (initial_loading) {
					_repository_changed(0); // Update when shown for the first time.
				}
			}
		} break;
		case NOTIFICATION_PROCESS: {
			HTTPClient::Status s = request->get_http_client_status();
			const bool loading = s != HTTPClient::STATUS_DISCONNECTED;

			if (loading) {
				library_scroll->set_modulate(Color(1, 1, 1, 0.5));
			} else {
				library_scroll->set_modulate(Color(1, 1, 1, 1));
			}

			const bool no_downloads = downloads_hb->get_child_count() == 0;
			if (no_downloads == downloads_scroll->is_visible()) {
				downloads_scroll->set_visible(!no_downloads);
			}

		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_update_repository_options();
			setup_http_request(request);
		} break;
	}
}

void EditorAssetLibrary::_update_repository_options() {
	Dictionary default_urls;
	default_urls["godotengine.org (Official)"] = "https://godotengine.org/asset-library/api";
	Dictionary available_urls = _EDITOR_DEF("asset_library/available_urls", default_urls, true);
	repository->clear();
	Array keys = available_urls.keys();
	for (int i = 0; i < keys.size(); i++) {
		String key = keys[i];
		repository->add_item(key);
		repository->set_item_metadata(i, available_urls[key]);
	}
}

void EditorAssetLibrary::unhandled_key_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	const Ref<InputEventKey> key = p_event;

	if (key.is_valid() && key->is_pressed()) {
		if (key->get_keycode_with_modifiers() == (KeyModifierMask::CMD | Key::F) && is_visible_in_tree()) {
			filter->grab_focus();
			filter->select_all();
			accept_event();
		}
	}
}

void EditorAssetLibrary::_install_asset() {
	ERR_FAIL_COND(!description);

	EditorAssetLibraryItemDownload *d = _get_asset_in_progress(description->get_asset_id());
	if (d) {
		d->install();
		return;
	}

	EditorAssetLibraryItemDownload *download = memnew(EditorAssetLibraryItemDownload);
	downloads_hb->add_child(download);
	download->configure(description->get_title(), description->get_asset_id(), description->get_preview_icon(), description->get_download_url(), description->get_sha256());

	if (templates_only) {
		download->set_external_install(true);
		download->connect("install_asset", callable_mp(this, &EditorAssetLibrary::_install_external_asset));
	}
}

const char *EditorAssetLibrary::sort_key[SORT_MAX] = {
	"updated",
	"updated",
	"name",
	"name",
	"cost",
	"cost",
};

const char *EditorAssetLibrary::sort_text[SORT_MAX] = {
	TTRC("Recently Updated"),
	TTRC("Least Recently Updated"),
	TTRC("Name (A-Z)"),
	TTRC("Name (Z-A)"),
	TTRC("License (A-Z)"), // "cost" stores the SPDX license name in the Godot Asset Library.
	TTRC("License (Z-A)"), // "cost" stores the SPDX license name in the Godot Asset Library.
};

const char *EditorAssetLibrary::support_key[SUPPORT_MAX] = {
	"official",
	"community",
	"testing",
};

void EditorAssetLibrary::_select_author(int p_id) {
	// Open author window.
}

void EditorAssetLibrary::_select_category(int p_id) {
	for (int i = 0; i < categories->get_item_count(); i++) {
		if (i == 0) {
			continue;
		}
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
}

void EditorAssetLibrary::_image_update(bool use_cache, bool final, const PackedByteArray &p_data, int p_queue_id) {
	Object *obj = ObjectDB::get_instance(image_queue[p_queue_id].target);

	if (obj) {
		bool image_set = false;
		PackedByteArray image_data = p_data;

		if (use_cache) {
			String cache_filename_base = EditorPaths::get_singleton()->get_cache_dir().plus_file("assetimage_" + image_queue[p_queue_id].image_url.md5_text());

			FileAccess *file = FileAccess::open(cache_filename_base + ".data", FileAccess::READ);

			if (file) {
				PackedByteArray cached_data;
				int len = file->get_32();
				cached_data.resize(len);

				uint8_t *w = cached_data.ptrw();
				file->get_buffer(w, len);

				image_data = cached_data;
				file->close();
				memdelete(file);
			}
		}

		int len = image_data.size();
		const uint8_t *r = image_data.ptr();
		Ref<Image> image = Ref<Image>(memnew(Image));

		uint8_t png_signature[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
		uint8_t jpg_signature[3] = { 255, 216, 255 };

		if (r) {
			if ((memcmp(&r[0], &png_signature[0], 8) == 0) && Image::_png_mem_loader_func) {
				image->copy_internals_from(Image::_png_mem_loader_func(r, len));
			} else if ((memcmp(&r[0], &jpg_signature[0], 3) == 0) && Image::_jpg_mem_loader_func) {
				image->copy_internals_from(Image::_jpg_mem_loader_func(r, len));
			}
		}

		if (!image->is_empty()) {
			switch (image_queue[p_queue_id].image_type) {
				case IMAGE_QUEUE_ICON:

					image->resize(64 * EDSCALE, 64 * EDSCALE, Image::INTERPOLATE_LANCZOS);

					break;
				case IMAGE_QUEUE_THUMBNAIL: {
					float max_height = 85 * EDSCALE;

					float scale_ratio = max_height / (image->get_height() * EDSCALE);
					if (scale_ratio < 1) {
						image->resize(image->get_width() * EDSCALE * scale_ratio, image->get_height() * EDSCALE * scale_ratio, Image::INTERPOLATE_LANCZOS);
					}
				} break;
				case IMAGE_QUEUE_SCREENSHOT: {
					float max_height = 397 * EDSCALE;

					float scale_ratio = max_height / (image->get_height() * EDSCALE);
					if (scale_ratio < 1) {
						image->resize(image->get_width() * EDSCALE * scale_ratio, image->get_height() * EDSCALE * scale_ratio, Image::INTERPOLATE_LANCZOS);
					}
				} break;
			}

			Ref<ImageTexture> tex;
			tex.instantiate();
			tex->create_from_image(image);

			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, tex);
			image_set = true;
		}

		if (!image_set && final) {
			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, get_theme_icon(SNAME("FileBrokenBigThumb"), SNAME("EditorIcons")));
		}
	}
}

void EditorAssetLibrary::_image_request_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data, int p_queue_id) {
	ERR_FAIL_COND(!image_queue.has(p_queue_id));

	if (p_status == HTTPRequest::RESULT_SUCCESS && p_code < HTTPClient::RESPONSE_BAD_REQUEST) {
		if (p_code != HTTPClient::RESPONSE_NOT_MODIFIED) {
			for (int i = 0; i < headers.size(); i++) {
				if (headers[i].findn("ETag:") == 0) { // Save etag
					String cache_filename_base = EditorPaths::get_singleton()->get_cache_dir().plus_file("assetimage_" + image_queue[p_queue_id].image_url.md5_text());
					String new_etag = headers[i].substr(headers[i].find(":") + 1, headers[i].length()).strip_edges();
					FileAccess *file;

					file = FileAccess::open(cache_filename_base + ".etag", FileAccess::WRITE);
					if (file) {
						file->store_line(new_etag);
						file->close();
						memdelete(file);
					}

					int len = p_data.size();
					const uint8_t *r = p_data.ptr();
					file = FileAccess::open(cache_filename_base + ".data", FileAccess::WRITE);
					if (file) {
						file->store_32(len);
						file->store_buffer(r, len);
						file->close();
						memdelete(file);
					}

					break;
				}
			}
		}
		_image_update(p_code == HTTPClient::RESPONSE_NOT_MODIFIED, true, p_data, p_queue_id);

	} else {
		WARN_PRINT("Error getting image file from URL: " + image_queue[p_queue_id].image_url);
		Object *obj = ObjectDB::get_instance(image_queue[p_queue_id].target);
		if (obj) {
			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, get_theme_icon(SNAME("FileBrokenBigThumb"), SNAME("EditorIcons")));
		}
	}

	image_queue[p_queue_id].request->queue_delete();
	image_queue.erase(p_queue_id);

	_update_image_queue();
}

void EditorAssetLibrary::_update_image_queue() {
	const int max_images = 6;
	int current_images = 0;

	List<int> to_delete;
	for (KeyValue<int, ImageQueue> &E : image_queue) {
		if (!E.value.active && current_images < max_images) {
			String cache_filename_base = EditorPaths::get_singleton()->get_cache_dir().plus_file("assetimage_" + E.value.image_url.md5_text());
			Vector<String> headers;

			if (FileAccess::exists(cache_filename_base + ".etag") && FileAccess::exists(cache_filename_base + ".data")) {
				FileAccess *file = FileAccess::open(cache_filename_base + ".etag", FileAccess::READ);
				if (file) {
					headers.push_back("If-None-Match: " + file->get_line());
					file->close();
					memdelete(file);
				}
			}

			Error err = E.value.request->request(E.value.image_url, headers);
			if (err != OK) {
				to_delete.push_back(E.key);
			} else {
				E.value.active = true;
			}
			current_images++;
		} else if (E.value.active) {
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
	setup_http_request(iq.request);

	iq.target = p_for;
	iq.queue_id = ++last_queue_id;
	iq.active = false;

	iq.request->connect("request_completed", callable_mp(this, &EditorAssetLibrary::_image_request_completed), varray(iq.queue_id));

	image_queue[iq.queue_id] = iq;

	add_child(iq.request);

	_image_update(true, false, PackedByteArray(), iq.queue_id);
	_update_image_queue();
}

void EditorAssetLibrary::_repository_changed(int p_repository_id) {
	host = repository->get_item_metadata(p_repository_id);
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

	// We use the "branch" version, i.e. major.minor, as patch releases should be compatible
	args += "&godot_version=" + String(VERSION_BRANCH);

	String support_list;
	for (int i = 0; i < SUPPORT_MAX; i++) {
		if (support->get_popup()->is_item_checked(i)) {
			support_list += String(support_key[i]) + "+";
		}
	}
	if (!support_list.is_empty()) {
		args += "&support=" + support_list.substr(0, support_list.length() - 1);
	}

	if (categories->get_selected() > 0) {
		args += "&category=" + itos(categories->get_item_metadata(categories->get_selected()));
	}

	// Sorting options with an odd index are always the reverse of the previous one
	if (sort->get_selected() % 2 == 1) {
		args += "&reverse=true";
	}

	if (!filter->get_text().is_empty()) {
		args += "&filter=" + filter->get_text().uri_encode();
	}

	if (p_page > 0) {
		args += "&page=" + itos(p_page);
	}

	_api_request("asset", REQUESTING_SEARCH, args);
}

void EditorAssetLibrary::_search_text_changed(const String &p_text) {
	filter_debounce_timer->start();
}

void EditorAssetLibrary::_filter_debounce_timer_timeout() {
	_search();
}

HBoxContainer *EditorAssetLibrary::_make_pages(int p_page, int p_page_count, int p_page_len, int p_total_items, int p_current_items) {
	HBoxContainer *hbc = memnew(HBoxContainer);

	if (p_page_count < 2) {
		return hbc;
	}

	//do the mario
	int from = p_page - 5;
	if (from < 0) {
		from = 0;
	}
	int to = from + 10;
	if (to > p_page_count) {
		to = p_page_count;
	}

	hbc->add_spacer();
	hbc->add_theme_constant_override("separation", 5 * EDSCALE);

	Button *first = memnew(Button);
	first->set_text(TTR("First"));
	if (p_page != 0) {
		first->connect("pressed", callable_mp(this, &EditorAssetLibrary::_search), varray(0));
	} else {
		first->set_disabled(true);
		first->set_focus_mode(Control::FOCUS_NONE);
	}
	hbc->add_child(first);

	Button *prev = memnew(Button);
	prev->set_text(TTR("Previous"));
	if (p_page > 0) {
		prev->connect("pressed", callable_mp(this, &EditorAssetLibrary::_search), varray(p_page - 1));
	} else {
		prev->set_disabled(true);
		prev->set_focus_mode(Control::FOCUS_NONE);
	}
	hbc->add_child(prev);
	hbc->add_child(memnew(VSeparator));

	for (int i = from; i < to; i++) {
		if (i == p_page) {
			Button *current = memnew(Button);
			// Keep the extended padding for the currently active page (see below).
			current->set_text(vformat(" %d ", i + 1));
			current->set_disabled(true);
			current->set_focus_mode(Control::FOCUS_NONE);

			hbc->add_child(current);
		} else {
			Button *current = memnew(Button);
			// Add padding to make page number buttons easier to click.
			current->set_text(vformat(" %d ", i + 1));
			current->connect("pressed", callable_mp(this, &EditorAssetLibrary::_search), varray(i));

			hbc->add_child(current);
		}
	}

	Button *next = memnew(Button);
	next->set_text(TTR("Next"));
	if (p_page < p_page_count - 1) {
		next->connect("pressed", callable_mp(this, &EditorAssetLibrary::_search), varray(p_page + 1));
	} else {
		next->set_disabled(true);
		next->set_focus_mode(Control::FOCUS_NONE);
	}
	hbc->add_child(memnew(VSeparator));
	hbc->add_child(next);

	Button *last = memnew(Button);
	last->set_text(TTR("Last"));
	if (p_page != p_page_count - 1) {
		last->connect("pressed", callable_mp(this, &EditorAssetLibrary::_search), varray(p_page_count - 1));
	} else {
		last->set_disabled(true);
		last->set_focus_mode(Control::FOCUS_NONE);
	}
	hbc->add_child(last);

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

void EditorAssetLibrary::_http_request_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data) {
	String str;

	{
		int datalen = p_data.size();
		const uint8_t *r = p_data.ptr();
		str.parse_utf8((const char *)r, datalen);
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

	Dictionary d;
	{
		JSON json;
		json.parse(str);
		d = json.get_data();
	}

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
					if (!cat.has("name") || !cat.has("id")) {
						continue;
					}
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
			initial_loading = false;

			// The loading text only needs to be displayed before the first page is loaded.
			// Therefore, we don't need to show it again.
			library_loading->hide();

			library_error->hide();

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
			asset_items->add_theme_constant_override("hseparation", 10 * EDSCALE);
			asset_items->add_theme_constant_override("vseparation", 10 * EDSCALE);

			library_vb->add_child(asset_items);

			asset_bottom_page = _make_pages(page, pages, page_len, total_items, result.size());
			library_vb->add_child(asset_bottom_page);

			if (result.is_empty()) {
				if (!filter->get_text().is_empty()) {
					library_error->set_text(
							vformat(TTR("No results for \"%s\"."), filter->get_text()));
				} else {
					// No results, even though the user didn't search for anything specific.
					// This is typically because the version number changed recently
					// and no assets compatible with the new version have been published yet.
					library_error->set_text(
							vformat(TTR("No results compatible with %s %s."), String(VERSION_SHORT_NAME).capitalize(), String(VERSION_BRANCH)));
				}
				library_error->show();
			}

			for (int i = 0; i < result.size(); i++) {
				Dictionary r = result[i];

				ERR_CONTINUE(!r.has("title"));
				ERR_CONTINUE(!r.has("asset_id"));
				ERR_CONTINUE(!r.has("author"));
				ERR_CONTINUE(!r.has("author_id"));
				ERR_CONTINUE(!r.has("category_id"));
				ERR_FAIL_COND(!category_map.has(r["category_id"]));
				ERR_CONTINUE(!r.has("cost"));

				EditorAssetLibraryItem *item = memnew(EditorAssetLibraryItem);
				asset_items->add_child(item);
				item->configure(r["title"], r["asset_id"], category_map[r["category_id"]], r["category_id"], r["author"], r["author_id"], r["cost"]);
				item->connect("asset_selected", callable_mp(this, &EditorAssetLibrary::_select_asset));
				item->connect("author_selected", callable_mp(this, &EditorAssetLibrary::_select_author));
				item->connect("category_selected", callable_mp(this, &EditorAssetLibrary::_select_category));

				if (r.has("icon_url") && !r["icon_url"].operator String().is_empty()) {
					_request_image(item->get_instance_id(), r["icon_url"], IMAGE_QUEUE_ICON, 0);
				}
			}

			if (!result.is_empty()) {
				library_scroll->set_v_scroll(0);
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
			description->popup_centered();
			description->connect("confirmed", callable_mp(this, &EditorAssetLibrary::_install_asset));

			description->configure(r["title"], r["asset_id"], category_map[r["category_id"]], r["category_id"], r["author"], r["author_id"], r["cost"], r["version"], r["version_string"], r["description"], r["download_url"], r["browse_url"], r["download_hash"]);

			EditorAssetLibraryItemDownload *download_item = _get_asset_in_progress(description->get_asset_id());
			if (download_item) {
				if (download_item->can_install()) {
					description->get_ok_button()->set_text(TTR("Install"));
					description->get_ok_button()->set_disabled(false);
				} else {
					description->get_ok_button()->set_text(TTR("Downloading..."));
					description->get_ok_button()->set_disabled(true);
				}
			} else {
				description->get_ok_button()->set_text(TTR("Download"));
				description->get_ok_button()->set_disabled(false);
			}

			if (r.has("icon_url") && !r["icon_url"].operator String().is_empty()) {
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

					if (!is_video) {
						_request_image(description->get_instance_id(), p["link"], IMAGE_QUEUE_SCREENSHOT, i);
					}
				}
			}
		} break;
		default:
			break;
	}
}

void EditorAssetLibrary::_asset_file_selected(const String &p_file) {
	if (asset_installer) {
		memdelete(asset_installer);
		asset_installer = nullptr;
	}

	asset_installer = memnew(EditorAssetInstaller);
	asset_installer->set_asset_name(p_file.get_basename());
	add_child(asset_installer);
	asset_installer->open(p_file);
}

void EditorAssetLibrary::_asset_open() {
	asset_open->popup_file_dialog();
}

void EditorAssetLibrary::_manage_plugins() {
	ProjectSettingsEditor::get_singleton()->popup_project_settings();
	ProjectSettingsEditor::get_singleton()->set_plugins_page();
}

EditorAssetLibraryItemDownload *EditorAssetLibrary::_get_asset_in_progress(int p_asset_id) const {
	for (int i = 0; i < downloads_hb->get_child_count(); i++) {
		EditorAssetLibraryItemDownload *d = Object::cast_to<EditorAssetLibraryItemDownload>(downloads_hb->get_child(i));
		if (d && d->get_asset_id() == p_asset_id) {
			return d;
		}
	}

	return nullptr;
}

void EditorAssetLibrary::_install_external_asset(String p_zip_path, String p_title) {
	emit_signal(SNAME("install_asset"), p_zip_path, p_title);
}

void EditorAssetLibrary::disable_community_support() {
	support->get_popup()->set_item_checked(SUPPORT_COMMUNITY, false);
}

void EditorAssetLibrary::_bind_methods() {
	ADD_SIGNAL(MethodInfo("install_asset", PropertyInfo(Variant::STRING, "zip_path"), PropertyInfo(Variant::STRING, "name")));
}

EditorAssetLibrary::EditorAssetLibrary(bool p_templates_only) {
	requesting = REQUESTING_NONE;
	templates_only = p_templates_only;
	initial_loading = true;

	VBoxContainer *library_main = memnew(VBoxContainer);

	add_child(library_main);

	HBoxContainer *search_hb = memnew(HBoxContainer);

	library_main->add_child(search_hb);
	library_main->add_theme_constant_override("separation", 10 * EDSCALE);

	filter = memnew(LineEdit);
	if (templates_only) {
		filter->set_placeholder(TTR("Search templates, projects, and demos"));
	} else {
		filter->set_placeholder(TTR("Search assets (excluding templates, projects, and demos)"));
	}
	filter->set_clear_button_enabled(true);
	search_hb->add_child(filter);
	filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filter->connect("text_changed", callable_mp(this, &EditorAssetLibrary::_search_text_changed));

	// Perform a search automatically if the user hasn't entered any text for a certain duration.
	// This way, the user doesn't need to press Enter to initiate their search.
	filter_debounce_timer = memnew(Timer);
	filter_debounce_timer->set_one_shot(true);
	filter_debounce_timer->set_wait_time(0.25);
	filter_debounce_timer->connect("timeout", callable_mp(this, &EditorAssetLibrary::_filter_debounce_timer_timeout));
	search_hb->add_child(filter_debounce_timer);

	if (!p_templates_only) {
		search_hb->add_child(memnew(VSeparator));
	}

	Button *open_asset = memnew(Button);
	open_asset->set_text(TTR("Import..."));
	search_hb->add_child(open_asset);
	open_asset->connect("pressed", callable_mp(this, &EditorAssetLibrary::_asset_open));

	Button *plugins = memnew(Button);
	plugins->set_text(TTR("Plugins..."));
	search_hb->add_child(plugins);
	plugins->connect("pressed", callable_mp(this, &EditorAssetLibrary::_manage_plugins));

	if (p_templates_only) {
		open_asset->hide();
		plugins->hide();
	}

	HBoxContainer *search_hb2 = memnew(HBoxContainer);
	library_main->add_child(search_hb2);

	search_hb2->add_child(memnew(Label(TTR("Sort:") + " ")));
	sort = memnew(OptionButton);
	for (int i = 0; i < SORT_MAX; i++) {
		sort->add_item(TTRGET(sort_text[i]));
	}

	search_hb2->add_child(sort);

	sort->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	sort->connect("item_selected", callable_mp(this, &EditorAssetLibrary::_rerun_search));

	search_hb2->add_child(memnew(VSeparator));

	search_hb2->add_child(memnew(Label(TTR("Category:") + " ")));
	categories = memnew(OptionButton);
	categories->add_item(TTR("All"));
	search_hb2->add_child(categories);
	categories->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	categories->connect("item_selected", callable_mp(this, &EditorAssetLibrary::_rerun_search));

	search_hb2->add_child(memnew(VSeparator));

	search_hb2->add_child(memnew(Label(TTR("Site:") + " ")));
	repository = memnew(OptionButton);

	_update_repository_options();

	repository->connect("item_selected", callable_mp(this, &EditorAssetLibrary::_repository_changed));

	search_hb2->add_child(repository);
	repository->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	search_hb2->add_child(memnew(VSeparator));

	support = memnew(MenuButton);
	search_hb2->add_child(support);
	support->set_text(TTR("Support"));
	support->get_popup()->set_hide_on_checkable_item_selection(false);
	support->get_popup()->add_check_item(TTR("Official"), SUPPORT_OFFICIAL);
	support->get_popup()->add_check_item(TTR("Community"), SUPPORT_COMMUNITY);
	support->get_popup()->add_check_item(TTR("Testing"), SUPPORT_TESTING);
	support->get_popup()->set_item_checked(SUPPORT_OFFICIAL, true);
	support->get_popup()->set_item_checked(SUPPORT_COMMUNITY, true);
	support->get_popup()->connect("id_pressed", callable_mp(this, &EditorAssetLibrary::_support_toggled));

	/////////

	library_scroll_bg = memnew(PanelContainer);
	library_main->add_child(library_scroll_bg);
	library_scroll_bg->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	library_scroll = memnew(ScrollContainer);
	library_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);

	library_scroll_bg->add_child(library_scroll);

	Ref<StyleBoxEmpty> border2;
	border2.instantiate();
	border2->set_default_margin(SIDE_LEFT, 15 * EDSCALE);
	border2->set_default_margin(SIDE_RIGHT, 35 * EDSCALE);
	border2->set_default_margin(SIDE_BOTTOM, 15 * EDSCALE);
	border2->set_default_margin(SIDE_TOP, 15 * EDSCALE);

	PanelContainer *library_vb_border = memnew(PanelContainer);
	library_scroll->add_child(library_vb_border);
	library_vb_border->add_theme_style_override("panel", border2);
	library_vb_border->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	library_vb = memnew(VBoxContainer);
	library_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	library_vb_border->add_child(library_vb);

	library_loading = memnew(Label(TTR("Loading...")));
	library_loading->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	library_vb->add_child(library_loading);

	library_error = memnew(Label);
	library_error->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	library_error->hide();
	library_vb->add_child(library_error);

	asset_top_page = memnew(HBoxContainer);
	library_vb->add_child(asset_top_page);

	asset_items = memnew(GridContainer);
	asset_items->set_columns(2);
	asset_items->add_theme_constant_override("hseparation", 10 * EDSCALE);
	asset_items->add_theme_constant_override("vseparation", 10 * EDSCALE);

	library_vb->add_child(asset_items);

	asset_bottom_page = memnew(HBoxContainer);
	library_vb->add_child(asset_bottom_page);

	request = memnew(HTTPRequest);
	add_child(request);
	setup_http_request(request);
	request->connect("request_completed", callable_mp(this, &EditorAssetLibrary::_http_request_completed));

	last_queue_id = 0;

	library_vb->add_theme_constant_override("separation", 20 * EDSCALE);

	error_hb = memnew(HBoxContainer);
	library_main->add_child(error_hb);
	error_label = memnew(Label);
	error_hb->add_child(error_label);
	error_tr = memnew(TextureRect);
	error_tr->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	error_hb->add_child(error_tr);

	description = nullptr;

	set_process(true);
	set_process_unhandled_key_input(true); // Global shortcuts since there is no main element to be focused.

	downloads_scroll = memnew(ScrollContainer);
	downloads_scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	library_main->add_child(downloads_scroll);
	downloads_hb = memnew(HBoxContainer);
	downloads_scroll->add_child(downloads_hb);

	asset_open = memnew(EditorFileDialog);

	asset_open->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	asset_open->add_filter("*.zip ; " + TTR("Assets ZIP File"));
	asset_open->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(asset_open);
	asset_open->connect("file_selected", callable_mp(this, &EditorAssetLibrary::_asset_file_selected));

	asset_installer = nullptr;
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
	editor->get_main_control()->add_child(addon_library);
	addon_library->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	addon_library->hide();
}

AssetLibraryEditorPlugin::~AssetLibraryEditorPlugin() {
}
