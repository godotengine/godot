/**************************************************************************/
/*  asset_library_editor_plugin.cpp                                       */
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

#include "asset_library_editor_plugin.h"

#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/io/stream_peer_tls.h"
#include "core/os/keyboard.h"
#include "core/version.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_paths.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/settings/project_settings_editor.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/style_box_texture.h"

static inline void setup_http_request(HTTPRequest *request) {
	request->set_use_threads(EDITOR_GET("asset_library/use_threads"));

	const String proxy_host = EDITOR_GET("network/http_proxy/host");
	const int proxy_port = EDITOR_GET("network/http_proxy/port");
	request->set_http_proxy(proxy_host, proxy_port);
	request->set_https_proxy(proxy_host, proxy_port);
}

void EditorAssetLibraryItem::configure(const String &p_title, const String &p_asset_id, const String &p_author, const String &p_author_id, const String &p_license_type, const String &p_license_url, int p_rating) {
	title_text = p_title;
	title->set_text(title_text);
	title->set_tooltip_text(title_text);
	asset_id = p_asset_id;
	author->set_text(p_author);
	author_id = p_author_id;
	license->set_text(p_license_type);
	license_url = p_license_url;
	rating_count->set_text(itos(p_rating));

	if (author_id.is_empty()) {
		author->set_disabled(true);
		author->set_mouse_filter(MOUSE_FILTER_IGNORE);
	}

	_calculate_misc_links_size();
}

void EditorAssetLibraryItem::set_image(int p_type, int p_index, const Ref<Texture2D> &p_image) {
	ERR_FAIL_COND(p_type != EditorAssetLibrary::IMAGE_QUEUE_THUMBNAIL);
	ERR_FAIL_COND(p_index != 0);

	icon->set_texture(p_image);
}

void EditorAssetLibraryItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			icon->set_texture(get_editor_theme_icon(SNAME("AssetThumbLoading")));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			author->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("faded_text"), SNAME("AssetLib")));
			license->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("faded_text"), SNAME("AssetLib")));
			rating_icon->set_texture(get_editor_theme_icon(SNAME("ThumbsUp")));

			_calculate_misc_links_size();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			_calculate_misc_links_size();
		} break;

		case NOTIFICATION_RESIZED: {
			calculate_misc_links_ratio();
		} break;
	}
}

void EditorAssetLibraryItem::_calculate_misc_links_size() {
	Ref<TextLine> text_buf;
	text_buf.instantiate();
	text_buf->add_string(author->get_text(), author->get_button_font(), author->get_button_font_size());
	author_width = text_buf->get_line_width();

	text_buf->clear();
	text_buf->add_string(license->get_text(), license->get_button_font(), license->get_button_font_size());
	price_width = text_buf->get_line_width();

	calculate_misc_links_ratio();
}

void EditorAssetLibraryItem::calculate_misc_links_ratio() {
	const int separators_width = 15 * EDSCALE;
	const float total_width = author_license_hbox->get_size().width - (separator->get_size().width + separators_width);
	if (total_width <= 0) {
		return;
	}

	float ratio_left = 1;
	// Make the ratios a fraction bigger, to avoid unnecessary trimming.
	const float extra_ratio = 4.0 / total_width;

	const float author_ratio = MIN(1, author_width / total_width);
	author->set_stretch_ratio(author_ratio + extra_ratio);
	ratio_left -= author_ratio;

	const float price_ratio = MIN(1, price_width / total_width);
	license->set_stretch_ratio(price_ratio + extra_ratio);
	ratio_left -= price_ratio;

	spacer->set_stretch_ratio(ratio_left);
}

void EditorAssetLibraryItem::_asset_clicked() {
	emit_signal(SNAME("asset_selected"), author_id + "/" + asset_id + "/");
}

void EditorAssetLibraryItem::_author_clicked() {
	OS::get_singleton()->shell_open("https://store-beta.godotengine.org/publisher/" + author_id.uri_encode() + "/");
}

void EditorAssetLibraryItem::_license_clicked() {
	ERR_FAIL_COND(!license_url.begins_with("http"));
	OS::get_singleton()->shell_open(license_url);
}

void EditorAssetLibraryItem::_bind_methods() {
	ClassDB::bind_method("set_image", &EditorAssetLibraryItem::set_image);
	ADD_SIGNAL(MethodInfo("asset_selected"));
	ADD_SIGNAL(MethodInfo("author_selected"));
}

EditorAssetLibraryItem::EditorAssetLibraryItem(bool p_clickable) {
	is_clickable = p_clickable;
	if (p_clickable) {
		button = memnew(Button);
		button->set_theme_type_variation(SceneStringName(FlatButton));
		add_child(button);
		button->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItem::_asset_clicked));
	}

	margin = memnew(MarginContainer);
	int margin_size = 5 * EDSCALE;
	margin->add_theme_constant_override(SNAME("margin_left"), margin_size);
	margin->add_theme_constant_override(SNAME("margin_right"), margin_size);
	margin->add_theme_constant_override(SNAME("margin_top"), margin_size);
	margin->add_theme_constant_override(SNAME("margin_bottom"), margin_size);
	margin->set_mouse_filter(MOUSE_FILTER_IGNORE);
	margin->set_clip_contents(true);
	add_child(margin);

	HBoxContainer *hb = memnew(HBoxContainer);
	// Add some spacing to visually separate the icon from the asset details.
	hb->add_theme_constant_override("separation", 15 * EDSCALE);
	hb->set_mouse_filter(MOUSE_FILTER_IGNORE);
	margin->add_child(hb);

	icon = memnew(TextureRect);
	icon->set_accessibility_name(TTRC("Thumbnail"));
	icon->set_custom_minimum_size(EditorAssetLibrary::THUMBNAIL_SIZE * EDSCALE);
	icon->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	icon->set_mouse_filter(MOUSE_FILTER_IGNORE);
	hb->add_child(icon);

	VBoxContainer *vb = memnew(VBoxContainer);
	vb->set_mouse_filter(MOUSE_FILTER_IGNORE);
	vb->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(vb);

	Ref<StyleBoxEmpty> label_margin;
	label_margin.instantiate();
	label_margin->set_content_margin_all(0);

	title = memnew(Label);
	title->add_theme_style_override(CoreStringName(normal), label_margin);
	title->set_accessibility_name(TTRC("Title"));
	title->set_auto_translate_mode(AutoTranslateMode::AUTO_TRANSLATE_MODE_DISABLED);
	title->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	title->set_mouse_filter(MOUSE_FILTER_IGNORE);
	title->set_focus_mode(FOCUS_ACCESSIBILITY);
	vb->add_child(title);

	author_license_hbox = memnew(HBoxContainer);
	author_license_hbox->add_theme_constant_override("separation", 5 * EDSCALE);
	author_license_hbox->set_mouse_filter(MOUSE_FILTER_IGNORE);
	vb->add_child(author_license_hbox);

	author = memnew(LinkButton);
	author->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	author->set_tooltip_text(TTRC("Author"));
	author->set_accessibility_name(TTRC("Author"));
	author->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	author_license_hbox->add_child(author);
	author->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItem::_author_clicked));

	separator = memnew(HSeparator);
	separator->set_mouse_filter(MOUSE_FILTER_IGNORE);
	author_license_hbox->add_child(separator);

	license = memnew(LinkButton);
	license->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	license->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	license->set_tooltip_text(TTRC("License"));
	license->set_accessibility_name(TTRC("License"));
	license->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	author_license_hbox->add_child(license);
	license->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItem::_license_clicked));

	spacer = memnew(Control);
	spacer->set_mouse_filter(MOUSE_FILTER_IGNORE);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	author_license_hbox->add_child(spacer);

	HBoxContainer *rating_hbox = memnew(HBoxContainer);
	rating_hbox->set_mouse_filter(MOUSE_FILTER_IGNORE);
	vb->add_child(rating_hbox);

	rating_icon = memnew(TextureRect);
	rating_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	rating_icon->set_mouse_filter(MOUSE_FILTER_IGNORE);
	rating_hbox->add_child(rating_icon);

	rating_count = memnew(Label);
	rating_count->set_mouse_filter(MOUSE_FILTER_STOP);
	rating_count->set_tooltip_text(TTRC("Review Score"));
	rating_count->set_accessibility_name(TTRC("Review score"));
	rating_hbox->add_child(rating_count);

	set_accessibility_name(TTRC("Open asset details"));
	set_custom_minimum_size(Size2(250, 80) * EDSCALE);
	set_h_size_flags(SIZE_EXPAND_FILL);
}

//////////////////////////////////////////////////////////////////////////////

Control *EditorAssetLibraryZoomMode::remove_previews() {
	ERR_FAIL_NULL_V(previews, nullptr);

	remove_child(previews);
	return previews;
}

void EditorAssetLibraryZoomMode::input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouse> m = p_event;
	if (m.is_valid()) {
		return;
	}

	if (p_event->is_action_pressed(SNAME("ui_cancel"))) {
		hide();
	}

	// Block inputs from going elsewhere.
	get_tree()->get_root()->set_input_as_handled();
}

EditorAssetLibraryZoomMode::EditorAssetLibraryZoomMode(Control *p_previews) {
	ERR_FAIL_COND(p_previews->get_parent());

	ColorRect *dim = memnew(ColorRect);
	dim->set_color(EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("base_color"), EditorStringName(Editor)));
	dim->set_anchors_preset(Control::PRESET_FULL_RECT);
	add_child(dim);

	previews = p_previews;
	add_child(previews);
	p_previews->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 40 * EDSCALE);

	set_process_input(true);
}

//////////////////////////////////////////////////////////////////////////////

void EditorAssetLibraryItemDescription::set_image(int p_type, int p_index, const Ref<Texture2D> &p_image) {
	switch (p_type) {
		case EditorAssetLibrary::IMAGE_QUEUE_THUMBNAIL: {
			item->call("set_image", p_type, p_index, p_image);
			icon = p_image;
		} break;

		case EditorAssetLibrary::IMAGE_QUEUE_VIDEO_THUMBNAIL:
		case EditorAssetLibrary::IMAGE_QUEUE_SCREENSHOT: {
			for (int i = 0; i < preview_images.size(); i++) {
				if (preview_images[i].id != p_index) {
					continue;
				}

				Button *button = preview_images[i].button;
				float button_texture_height = button->get_size().height - button->get_theme_stylebox(CoreStringName(normal), SNAME("Button"))->get_minimum_size().height;
				float scale_ratio = button_texture_height / p_image->get_height();
				button->set_custom_minimum_size(Size2(p_image->get_width() * scale_ratio * EDSCALE, 0));

				if (preview_images[i].is_video) {
					Ref<Image> overlay = previews->get_editor_theme_icon(SNAME("PlayOverlay"))->get_image();
					Ref<Image> thumbnail = p_image->get_image()->duplicate();
					Point2i overlay_pos = Point2i((p_image->get_width() - overlay->get_width()) / 2, (p_image->get_height() - overlay->get_height()) / 2);

					// Overlay and thumbnail need the same format for `blend_rect` to work.
					thumbnail->convert(Image::FORMAT_RGBA8);
					thumbnail->blend_rect(overlay, overlay->get_used_rect(), overlay_pos);
					button->set_button_icon(ImageTexture::create_from_image(thumbnail));

					// Make it clearer that clicking it will open an external link.
					button->set_default_cursor_shape(Control::CURSOR_POINTING_HAND);
				} else {
					button->set_button_icon(p_image);
				}

				preview_images.write[i].image = p_image;
				if (button->is_pressed()) {
					preview_click(p_index);
				}

				break;
			}
		} break;
	}
}

void EditorAssetLibraryItemDescription::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			connect(SceneStringName(confirmed), callable_mp(this, &EditorAssetLibraryItemDescription::_confirmed));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			version_label->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("bold"), EditorStringName(EditorFonts)));
			Ref<Texture2D> link_icon = get_editor_theme_icon(SNAME("ExternalLink"));
			store->set_button_icon(link_icon);
			source->set_button_icon(link_icon);
			previous_preview->set_button_icon(get_editor_theme_icon(SNAME("Back")));
			next_preview->set_button_icon(get_editor_theme_icon(SNAME("Forward")));
			previews_bg->add_theme_style_override(SceneStringName(panel), previews->get_theme_stylebox(CoreStringName(normal), SNAME("TextEdit")));
			zoom_button->set_button_icon(get_editor_theme_icon(SNAME("DistractionFree")));
		} break;

		case NOTIFICATION_READY: {
			int width = zoom_button->get_size().width;
			previous_preview->set_custom_minimum_size(Size2(width, 100 * EDSCALE));
			next_preview->set_custom_minimum_size(Size2(width, 100 * EDSCALE));
		} break;

		case NOTIFICATION_POST_POPUP: {
			callable_mp(item, &EditorAssetLibraryItem::calculate_misc_links_ratio).call_deferred();
		} break;
	}
}

void EditorAssetLibraryItemDescription::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_image"), &EditorAssetLibraryItemDescription::set_image);

	ADD_SIGNAL(MethodInfo("install_requested", PropertyInfo(Variant::STRING, "asset_id"), PropertyInfo(Variant::STRING, "version"), PropertyInfo(Variant::STRING, "dpownload_url"), PropertyInfo(Variant::STRING, "sha256")));
	ADD_SIGNAL(MethodInfo("tag_clicked", PropertyInfo(Variant::STRING, "tag")));
}

void EditorAssetLibraryItemDescription::_confirmed() {
	if (install_mode == MODE_INSTALL) {
		// It will just redirect to the install dialog.
		emit_signal(SNAME("install_requested"), asset_id, "", "", "");
		return;
	}

	Release release = releases[version_list->get_selected()];
	emit_signal(SNAME("install_requested"), asset_id, release.version, release.url, release.sha256);
}

void EditorAssetLibraryItemDescription::_store_pressed() {
	OS::get_singleton()->shell_open(store_url);
}

void EditorAssetLibraryItemDescription::_source_pressed() {
	OS::get_singleton()->shell_open(source_url);
}

void EditorAssetLibraryItemDescription::_link_click(const String &p_url) {
	if (p_url.begins_with("#")) {
		emit_signal("tag_clicked", p_url);
		return;
	}

	ERR_FAIL_COND(!p_url.begins_with("http"));
	OS::get_singleton()->shell_open(p_url);
}

void EditorAssetLibraryItemDescription::preview_click(int p_id) {
	for (int i = 0; i < preview_images.size(); i++) {
		if (preview_images[i].id != p_id) {
			continue;
		}

		preview_images[i].button->set_pressed(true);
		if (!preview_images[i].is_video) {
			if (preview_images[i].image.is_valid()) {
				preview->set_texture(preview_images[i].image);
				child_controls_changed();
			}
			preview_images[i].button->grab_focus(true);
		} else {
			_link_click(preview_images[i].video_link);
		}

		break;
	}
}

void EditorAssetLibraryItemDescription::_previous_preview_pressed() {
	List<BaseButton *> buttons;
	preview_group->get_buttons(&buttons);
	BaseButton *pressed = preview_group->get_pressed_button();
	if (pressed == buttons.front()->get()) {
		preview_click(buttons.back()->get()->get_index());
	} else {
		preview_click(pressed->get_index() - 1);
	}
}

void EditorAssetLibraryItemDescription::_next_preview_pressed() {
	List<BaseButton *> buttons;
	preview_group->get_buttons(&buttons);
	BaseButton *pressed = preview_group->get_pressed_button();
	if (pressed == buttons.back()->get()) {
		preview_click(buttons.front()->get()->get_index());
	} else {
		preview_click(pressed->get_index() + 1);
	}
}

void EditorAssetLibraryItemDescription::_zoom_toggled(bool p_pressed) {
	if (p_pressed) {
		root->remove_child(previews_vbox);
		zoom_mode = memnew(EditorAssetLibraryZoomMode(previews_vbox));
		get_tree()->get_root()->add_child(zoom_mode);
		zoom_mode->connect(SceneStringName(visibility_changed), callable_mp(Object::cast_to<BaseButton>(zoom_button), &BaseButton::set_pressed).bind(false));

		hide();
	} else {
		root->add_child(zoom_mode->remove_previews());
		zoom_mode->queue_free();
		zoom_mode = nullptr;

		show();
	}
}

void EditorAssetLibraryItemDescription::configure(const String &p_title, const String &p_asset_id, const String &p_author, const String &p_author_id, const String &p_license_type, const String &p_license_url, int p_rating, const String &p_description, const HashMap<String, String> &p_tags, const String &p_store_url, const String &p_source_url) {
	asset_id = p_asset_id;
	title = p_title;
	item->configure(p_title, p_asset_id, p_author, p_author_id, p_license_type, p_license_url, p_rating);

	releases.clear();

	version->show();
	version->set_text(TTR("Loading..."));
	version_list->hide();
	version_list->clear();

	store_url = p_store_url;

	source_url = p_source_url;
	source->set_visible(!p_source_url.is_empty());

	description->clear();
	description->push_bold();
	description->add_text(TTR("Description:") + "\n");
	description->pop();
	description->append_text(p_description);

	if (!p_tags.is_empty()) {
		description->append_text("\n\n[b]" + TTR("Tags:") + "[/b]");
		for (const KeyValue<String, String> &KV : p_tags) {
			description->add_text(" ");
			description->push_meta("#" + KV.value);
			description->add_text("#" + KV.key);
			description->pop();
		}
	}

	description->set_selection_enabled(true);
	description->set_context_menu_enabled(true);

	set_title(p_title);
	if (install_mode == MODE_DOWNLOAD) {
		get_ok_button()->set_disabled(true);
	}
}

void EditorAssetLibraryItemDescription::set_install_mode(InstallMode p_mode) {
	if (p_mode == install_mode) {
		return;
	}

	switch (p_mode) {
		case MODE_DOWNLOAD: {
			set_ok_button_text(TTRC("Download"));
			get_ok_button()->set_disabled(releases.is_empty());
			version_list->set_disabled(releases.is_empty());
		} break;

		case MODE_DOWNLOADING: {
			set_ok_button_text(TTRC("Downloading..."));
			get_ok_button()->set_disabled(true);
			version_list->set_disabled(true);
		} break;

		case MODE_INSTALL: {
			set_ok_button_text(TTRC("Install..."));
			get_ok_button()->set_disabled(false);
			version_list->set_disabled(true);
		} break;
	}

	install_mode = p_mode;
}

void EditorAssetLibraryItemDescription::add_release(const String &p_url, const String &p_version, const String &p_sha256) {
	Release release;
	release.url = p_url;
	release.version = p_version;
	release.sha256 = p_sha256;

	if (releases.is_empty()) {
		version->set_text(p_version);
		if (install_mode == MODE_DOWNLOAD) {
			get_ok_button()->set_disabled(false);
		}
	} else if (releases.size() == 1) {
		version->hide();
		version_list->set_text(releases[0].version);
		if (install_mode == MODE_DOWNLOAD) {
			version_list->set_disabled(false);
		}
		version_list->show();
	}

	version_list->add_item(p_version, releases.size());
	releases.append(release);
}

void EditorAssetLibraryItemDescription::add_preview(int p_id, bool p_video, const String &p_url, const String &p_thumbnail) {
	if (preview_images.is_empty()) {
		desc_vbox->set_h_size_flags(0);
		previews_vbox->show();
	}

	Preview new_preview;
	new_preview.id = p_id;
	new_preview.video_link = p_url;
	new_preview.is_video = p_video;
	new_preview.button = memnew(Button);
	new_preview.button->set_button_icon(previews->get_editor_theme_icon(SNAME("ThumbnailWait")));
	new_preview.button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	new_preview.button->set_expand_icon(true);
	new_preview.button->set_toggle_mode(!p_video);
	new_preview.button->set_theme_type_variation(SNAME("ThumbnailButton"));
	new_preview.button->set_custom_minimum_size(Size2(preview_hb->get_size().height, 0));
	preview_hb->add_child(new_preview.button);
	new_preview.button->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDescription::preview_click).bind(p_id));

	if (!p_video) {
		new_preview.button->set_button_group(preview_group);
		// Enable the preview arrows if more than one screenshot is available.
		if (previous_preview->is_disabled()) {
			List<BaseButton *> buttons;
			preview_group->get_buttons(&buttons);
			if (buttons.size() > 1) {
				previous_preview->set_disabled(false);
				next_preview->set_disabled(false);
			}
		}

		zoom_button->set_disabled(false);
	}

	preview_images.push_back(new_preview);
}

EditorAssetLibraryItemDescription::EditorAssetLibraryItemDescription() {
	root = memnew(HBoxContainer);
	root->add_theme_constant_override("separation", 15 * EDSCALE);
	add_child(root);

	desc_vbox = memnew(VBoxContainer);
	desc_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	desc_vbox->set_custom_minimum_size(Size2(440, 440) * EDSCALE);
	root->add_child(desc_vbox);

	item = memnew(EditorAssetLibraryItem);
	desc_vbox->add_child(item);

	HBoxContainer *contents = memnew(HBoxContainer);
	desc_vbox->add_child(contents);

	version_label = memnew(Label(TTRC("Version:")));
	contents->add_child(version_label);

	version = memnew(Label);
	contents->add_child(version);

	version_list = memnew(OptionButton);
	version_list->set_fit_to_longest_item(false);
	version_list->set_tooltip_text(TTRC("Download other versions."));
	version_list->hide(); // Will be shown if multiple versions are available.
	contents->add_child(version_list);

	contents->add_spacer();

	store = memnew(Button);
	store->set_text(TTRC("Store Page"));
	store->set_tooltip_text(TTRC("Open the web browser to show the asset in the online store page."));
	store->set_theme_type_variation(SceneStringName(FlatButton));
	contents->add_child(store);
	store->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDescription::_store_pressed));

	source = memnew(Button);
	source->set_text(TTRC("View Source"));
	source->set_tooltip_text(TTRC("Open the web browser to show a page with the source files."));
	source->set_theme_type_variation(SceneStringName(FlatButton));
	source->hide(); // Will be shown if the source link is available.
	contents->add_child(source);
	source->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDescription::_source_pressed));

	description = memnew(RichTextLabel);
	description->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	description->add_theme_constant_override(SceneStringName(line_separation), Math::round(5 * EDSCALE));
	desc_vbox->add_child(description);
	description->connect("meta_clicked", callable_mp(this, &EditorAssetLibraryItemDescription::_link_click));

	previews_vbox = memnew(VBoxContainer);
	previews_vbox->hide(); // Will be shown if we add any previews later.
	previews_vbox->add_theme_constant_override("separation", 15 * EDSCALE);
	previews_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	previews_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	root->add_child(previews_vbox);

	HBoxContainer *previews_hbox = memnew(HBoxContainer);
	previews_hbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	previews_vbox->add_child(previews_hbox);

	previous_preview = memnew(Button);
	previous_preview->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	previous_preview->set_disabled(true);
	previous_preview->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	previews_hbox->add_child(previous_preview);
	previous_preview->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDescription::_previous_preview_pressed));

	preview = memnew(TextureRect);
	preview->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	preview->set_custom_minimum_size(Size2(640, 345) * EDSCALE);
	preview->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	previews_hbox->add_child(preview);

	MarginContainer *mc = memnew(MarginContainer);
	previews_hbox->add_child(mc);

	next_preview = memnew(Button);
	next_preview->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	next_preview->set_disabled(true);
	next_preview->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	mc->add_child(next_preview);
	next_preview->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDescription::_next_preview_pressed));

	zoom_button = memnew(Button);
	zoom_button->set_toggle_mode(true);
	zoom_button->set_disabled(true);
	zoom_button->set_tooltip_text(TTRC("Toggle full view of preview images."));
	zoom_button->set_v_size_flags(Control::SIZE_SHRINK_END);
	mc->add_child(zoom_button);
	zoom_button->connect(SceneStringName(toggled), callable_mp(this, &EditorAssetLibraryItemDescription::_zoom_toggled));

	previews_bg = memnew(PanelContainer);
	previews_vbox->add_child(previews_bg);

	previews = memnew(ScrollContainer);
	previews->set_follow_focus(true);
	previews->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	previews_bg->add_child(previews);
	preview_hb = memnew(HBoxContainer);
	preview_hb->set_custom_minimum_size(Size2(620, 90) * EDSCALE);
	preview_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	previews->add_child(preview_hb);

	preview_group.instantiate();

	set_ok_button_text(TTRC("Download"));
	set_cancel_button_text(TTRC("Close"));
}

///////////////////////////////////////////////////////////////////////////////////

void EditorAssetLibraryItemDownload::_http_download_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data) {
	String error_text;

	switch (p_status) {
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED: {
			error_text = TTR("Connection error, please try again.");
			status->set_text(TTRC("Can't connect."));
		} break;
		case HTTPRequest::RESULT_CANT_CONNECT:
		case HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR: {
			error_text = TTR("Can't connect to host:") + " " + host;
			status->set_text(TTRC("Can't connect."));
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			error_text = TTR("No response from host:") + " " + host;
			status->set_text(TTRC("No response."));
		} break;
		case HTTPRequest::RESULT_CANT_RESOLVE: {
			error_text = TTR("Can't resolve hostname:") + " " + host;
			status->set_text(TTRC("Can't resolve."));
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			error_text = TTR("Request failed, return code:") + " " + itos(p_code);
			status->set_text(TTRC("Request failed."));
		} break;
		case HTTPRequest::RESULT_DOWNLOAD_FILE_CANT_OPEN:
		case HTTPRequest::RESULT_DOWNLOAD_FILE_WRITE_ERROR: {
			error_text = TTR("Cannot save response to:") + " " + download->get_download_file();
			status->set_text(TTRC("Write error."));
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			error_text = TTR("Request failed, too many redirects");
			status->set_text(TTRC("Redirect loop."));
		} break;
		case HTTPRequest::RESULT_TIMEOUT: {
			error_text = TTR("Request failed, timeout");
			status->set_text(TTRC("Timeout."));
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
					status->set_text(TTRC("Failed SHA-256 hash check"));
				}
			}
		} break;
	}

	progress->hide();

	set_process(false);

	if (!error_text.is_empty()) {
		download_error->set_text(TTR("Asset Download Error:") + "\n" + error_text);
		download_error->popup_centered();
		// Let the user retry the download.
		retry_button->show();
		return;
	}

	install_button->show();
	status->set_text(TTRC("Ready to install."));

	// Automatically prompt for installation once the download is completed
	// as long as the main window is focused, to not clash with other subwindows.
	if (get_window()->has_focus()) {
		install();
	}
}

void EditorAssetLibraryItemDownload::configure(const String &p_title, const String &p_asset_id, const String &p_version, const Ref<Texture2D> &p_preview, const String &p_download_url, const String &p_sha256) {
	title->set_text(p_title);
	version->set_text(p_version);
	icon->set_texture(p_preview);
	asset_id = p_asset_id;
	if (p_preview.is_null()) {
		icon->set_texture(get_editor_theme_icon(SNAME("FileBrokenBigThumb")));
	}
	host = p_download_url;
	sha256 = p_sha256;
	_make_request();
}

void EditorAssetLibraryItemDownload::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SceneStringName(panel), SNAME("AssetLib")));
			version->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("faded_text"), SNAME("AssetLib")));
			dismiss_button->set_texture_normal(get_theme_icon(SNAME("dismiss"), SNAME("AssetLib")));
			spacer->set_custom_minimum_size(Size2(0, 8 * EDSCALE));

			// Avoid sudden size changes by making the container have the same height as the buttons.
			Ref<StyleBoxFlat> button_style = get_theme_stylebox(CoreStringName(normal), SNAME("Button"));
			Ref<Font> font = get_theme_font(SceneStringName(font), SNAME("Button"));
			int font_size = get_theme_font_size(SceneStringName(font_size), SNAME("Button"));
			int button_height = button_style->get_minimum_size().height + font->get_height(font_size);
			progress_hbox->set_custom_minimum_size(Size2(0, button_height));
		} break;

		case NOTIFICATION_PROCESS: {
			progress->show();

			if (download->get_downloaded_bytes() > 0) {
				progress->set_max(download->get_body_size());
				progress->set_value(download->get_downloaded_bytes());
			}

			int cstatus = download->get_http_client_status();

			if (cstatus == HTTPClient::STATUS_BODY) {
				if (download->get_body_size() > 0) {
					progress->set_indeterminate(false);
					status->set_text(vformat(
							TTR("Downloading (%s / %s)..."),
							String::humanize_size(download->get_downloaded_bytes()),
							String::humanize_size(download->get_body_size())));
				} else {
					progress->set_indeterminate(true);
					status->set_text(vformat(
							TTR("Downloading...") + " (%s)",
							String::humanize_size(download->get_downloaded_bytes())));
				}
			}

			if (cstatus != prev_status) {
				switch (cstatus) {
					case HTTPClient::STATUS_RESOLVING: {
						status->set_text(TTRC("Resolving..."));
						progress->set_max(1);
						progress->set_value(0);
					} break;
					case HTTPClient::STATUS_CONNECTING: {
						status->set_text(TTRC("Connecting..."));
						progress->set_max(1);
						progress->set_value(0);
					} break;
					case HTTPClient::STATUS_REQUESTING: {
						status->set_text(TTRC("Requesting..."));
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
	queue_free();
}

bool EditorAssetLibraryItemDownload::can_install() const {
	return install_button->is_visible();
}

void EditorAssetLibraryItemDownload::install() {
	String file = download->get_download_file();

	if (external_install) {
		emit_signal(SNAME("install_asset"), file, title->get_text());
		return;
	}

	asset_installer->set_asset_name(title->get_text());
	asset_installer->open_asset(file, true);
}

void EditorAssetLibraryItemDownload::_make_request() {
	// Hide the Retry button if we've just pressed it.
	retry_button->hide();

	download->cancel_request();
	download->set_download_file(EditorPaths::get_singleton()->get_cache_dir().path_join("tmp_asset_" + asset_id) + ".zip");

	Error err = download->request(host);
	if (err != OK) {
		status->set_text(TTRC("Error making request"));
	} else {
		progress->set_indeterminate(true);
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
	vb->add_theme_constant_override("separation", 0);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	vb->add_child(title_hb);
	title = memnew(Label);
	title->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	title->set_theme_type_variation("LabelVMarginless");
	title->set_focus_mode(FOCUS_ACCESSIBILITY);
	title->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	title_hb->add_child(title);

	dismiss_button = memnew(TextureButton);
	dismiss_button->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDownload::_close));
	dismiss_button->set_accessibility_name(TTRC("Close"));
	title_hb->add_child(dismiss_button);

	version = memnew(Label);
	version->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	version->set_theme_type_variation("LabelVMarginless");
	vb->add_child(version);

	spacer = memnew(Control);
	vb->add_child(spacer);

	status = memnew(Label(TTRC("Idle")));
	vb->add_child(status);

	progress_hbox = memnew(HBoxContainer);
	vb->add_child(progress_hbox);

	progress = memnew(ProgressBar);
	progress->set_editor_preview_indeterminate(true);
	progress->hide();
	progress->set_h_size_flags(SIZE_EXPAND_FILL);
	progress_hbox->add_child(progress);

	retry_button = memnew(Button);
	retry_button->set_text(TTRC("Retry"));
	retry_button->hide(); // Only show the Retry button in case of a failure.
	retry_button->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_END);
	progress_hbox->add_child(retry_button);
	retry_button->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDownload::_make_request));

	install_button = memnew(Button);
	install_button->set_text(TTRC("Install..."));
	install_button->hide();
	install_button->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_END);
	progress_hbox->add_child(install_button);
	install_button->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryItemDownload::install));

	set_custom_minimum_size(Size2(400 * EDSCALE, 0));

	download = memnew(HTTPRequest);
	panel->add_child(download);
	download->connect("request_completed", callable_mp(this, &EditorAssetLibraryItemDownload::_http_download_completed));
	setup_http_request(download);

	download_error = memnew(AcceptDialog);
	download_error->set_title(TTRC("Download Error"));
	panel->add_child(download_error);

	asset_installer = memnew(EditorAssetInstaller);
	panel->add_child(asset_installer);
	asset_installer->connect(SceneStringName(confirmed), callable_mp(this, &EditorAssetLibraryItemDownload::_close));

	prev_status = -1;

	external_install = false;
}

////////////////////////////////////////////////////////////////////////////////
void EditorAssetLibrary::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("bg"), SNAME("AssetLib")));
			error_label->move_to_front();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (!initial_loading) {
				_search();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			error_tr->set_texture(get_editor_theme_icon(SNAME("Error")));
			filter->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			library_scroll->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SceneStringName(panel), SNAME("Tree")));
			downloads_scroll->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("downloads"), SNAME("AssetLib")));
			error_label->add_theme_color_override("color", get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
#ifndef ANDROID_ENABLED
				// Focus the search box automatically when switching to the Templates tab (in the Project Manager)
				// or switching to the AssetLib tab (in the editor).
				// The Project Manager's project filter box is automatically focused in the project manager code.
				filter->grab_focus();
#endif

				if (initial_loading) {
					_repository_changed(0); // Update when shown for the first time.
				}
			}
		} break;

		case NOTIFICATION_PROCESS: {
			// Check for finished image updates.
			List<int> to_delete;
			for (KeyValue<int, ImageQueue> &E : image_queue) {
				if (!E.value.update_finished) {
					continue;
				}

				Object *obj = ObjectDB::get_instance(E.value.target);
				if (obj) {
					if (E.value.texture.is_valid()) {
						obj->call("set_image", E.value.image_type, E.value.image_index, E.value.texture);
					} else {
						obj->call("set_image", E.value.image_type, E.value.image_index, get_editor_theme_icon(SNAME("FileBrokenBigThumb")));
					}
				}

				E.value.thread->wait_to_finish();
				E.value.request->queue_free();
				to_delete.push_back(E.key);
				_update_image_queue();
			}

			while (to_delete.size()) {
				image_queue[to_delete.front()->get()].request->queue_free();
				image_queue.erase(to_delete.front()->get());
				to_delete.pop_front();
			}

			if (image_queue.is_empty()) {
				set_process(false);
			}
		} break;

		case NOTIFICATION_RESIZED: {
			_update_asset_items_columns();
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (!EditorSettings::get_singleton()->check_changed_settings_in_group("asset_library") &&
					!EditorSettings::get_singleton()->check_changed_settings_in_group("network")) {
				break;
			}

			_update_repository_options();
			setup_http_request(request);

			const bool loading_blocked_new = ((int)EDITOR_GET("network/connection/network_mode") == EditorSettings::NETWORK_OFFLINE);
			if (loading_blocked_new != loading_blocked) {
				loading_blocked = loading_blocked_new;

				if (!loading_blocked && is_visible()) {
					_request_current_config(); // Reload config now that the network is available.
				}
			}
		} break;
	}
}

void EditorAssetLibrary::_update_repository_options() {
	// TODO: Move to editor_settings.cpp
	Dictionary default_urls;
	default_urls["godotengine.org (Official)"] = "https://store-beta.godotengine.org/api/v1";
	Dictionary available_urls = _EDITOR_DEF("asset_library/available_urls", default_urls, true);
	repository->clear();
	int i = 0;
	for (const KeyValue<Variant, Variant> &kv : available_urls) {
		repository->add_item(kv.key);
		repository->set_item_metadata(i, kv.value);
		i++;
	}
}

void EditorAssetLibrary::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	const Ref<InputEventKey> key = p_event;

	if (key.is_valid() && key->is_pressed()) {
		if (key->is_match(InputEventKey::create_reference(KeyModifierMask::CMD_OR_CTRL | Key::F)) && is_visible_in_tree()) {
			filter->grab_focus();
			filter->select_all();
			accept_event();
		}
	}
}

void EditorAssetLibrary::_install_asset(const String &p_asset_id, const String &p_version, const String &p_download_url, const String &p_sha256) {
	ERR_FAIL_NULL(description);

	EditorAssetLibraryItemDownload *d = _get_asset_in_progress(p_asset_id);
	if (d) {
		d->install();
		return;
	}

	EditorAssetLibraryItemDownload *download = memnew(EditorAssetLibraryItemDownload);
	downloads_hb->add_child(download);
	download->configure(description->get_title(), p_asset_id, p_version, description->get_preview_icon(), p_download_url, p_sha256);
	download->connect("tree_exited", callable_mp(this, &EditorAssetLibrary::_update_downloads_section));

	if (templates_only) {
		download->set_external_install(true);
		download->connect("install_asset", callable_mp(this, &EditorAssetLibrary::_install_external_asset));
	}
}

void EditorAssetLibrary::_tag_clicked(const String &p_tag) {
	description->hide();
	filter->set_text(p_tag);
	_search();
}

const char *EditorAssetLibrary::sort_key[SORT_MAX] = {
	"relevance",
	"updated_desc",
	"updated_asc",
	"reviews_desc",
	"reviews_asc",
	"created_desc",
	"created_asc",
};

const char *EditorAssetLibrary::sort_text[SORT_MAX] = {
	TTRC("Relevance"),
	TTRC("Updated (Newest First)"),
	TTRC("Updated (Oldest First)"),
	TTRC("Reviews (Highest Score First)"),
	TTRC("Reviews (Lowest Score First)"),
	TTRC("Created (Newest First)"),
	TTRC("Created (Oldest First)"),
};

void EditorAssetLibrary::_select_asset(const String &p_id) {
	_api_request("assets/" + p_id, REQUESTING_ASSET);
}

void EditorAssetLibrary::_image_update(void *p_image_queue) {
	ImageQueue *iq = static_cast<ImageQueue *>(p_image_queue);
	PackedByteArray image_data = iq->data;

	if (iq->use_cache) {
		String cache_filename_base = EditorPaths::get_singleton()->get_cache_dir().path_join("assetimage_" + iq->image_url.md5_text());

		Ref<FileAccess> file = FileAccess::open(cache_filename_base + ".data", FileAccess::READ);
		if (file.is_valid()) {
			PackedByteArray cached_data;
			int len = file->get_32();
			cached_data.resize(len);

			uint8_t *w = cached_data.ptrw();
			file->get_buffer(w, len);

			image_data = cached_data;
		}
	}

	int len = image_data.size();
	const uint8_t *r = image_data.ptr();
	Ref<Image> image = memnew(Image);

	uint8_t png_signature[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
	uint8_t jpg_signature[3] = { 255, 216, 255 };
	uint8_t webp_signature[4] = { 82, 73, 70, 70 };
	uint8_t bmp_signature[2] = { 66, 77 };

	if (r) {
		Ref<Image> parsed_image;

		if ((memcmp(&r[0], &png_signature[0], 8) == 0) && Image::_png_mem_loader_func) {
			parsed_image = Image::_png_mem_loader_func(r, len);
		} else if ((memcmp(&r[0], &jpg_signature[0], 3) == 0) && Image::_jpg_mem_loader_func) {
			parsed_image = Image::_jpg_mem_loader_func(r, len);
		} else if ((memcmp(&r[0], &webp_signature[0], 4) == 0) && Image::_webp_mem_loader_func) {
			parsed_image = Image::_webp_mem_loader_func(r, len);
		} else if ((memcmp(&r[0], &bmp_signature[0], 2) == 0) && Image::_bmp_mem_loader_func) {
			parsed_image = Image::_bmp_mem_loader_func(r, len);
		} else if (Image::_svg_scalable_mem_loader_func) {
			parsed_image = Image::_svg_scalable_mem_loader_func(r, len, 1.0);
		}

		if (parsed_image.is_null()) {
			if (is_print_verbose_enabled()) {
				ERR_PRINT(vformat("Asset Store: Invalid image downloaded from '%s' for asset # %d", iq->image_url, iq->asset_id));
			}
		} else {
			image->copy_internals_from(parsed_image);
		}
	}

	if (!image->is_empty()) {
		Size2 max_size;
		switch (iq->image_type) {
			case IMAGE_QUEUE_THUMBNAIL:
			case IMAGE_QUEUE_VIDEO_THUMBNAIL: {
				max_size = THUMBNAIL_SIZE;
			} break;

			case IMAGE_QUEUE_SCREENSHOT: {
				max_size.y = image->get_height();
			} break;
		}

		float scale_ratio = max_size.y / image->get_height();
		if (max_size.x > 0) {
			scale_ratio = MIN(scale_ratio, max_size.x / image->get_width());
		}
		if (scale_ratio < 1) {
			image->resize(image->get_width() * scale_ratio * EDSCALE, image->get_height() * scale_ratio * EDSCALE, Image::INTERPOLATE_LANCZOS);
		}

		iq->texture = ImageTexture::create_from_image(image);
	}

	iq->update_finished = true;
}

void EditorAssetLibrary::_image_request_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data, int p_queue_id) {
	ERR_FAIL_COND(!image_queue.has(p_queue_id));

	if (p_status == HTTPRequest::RESULT_SUCCESS && p_code < HTTPClient::RESPONSE_BAD_REQUEST) {
		if (p_code != HTTPClient::RESPONSE_NOT_MODIFIED) {
			for (int i = 0; i < headers.size(); i++) {
				if (headers[i].findn("ETag:") == 0) { // Save etag
					String cache_filename_base = EditorPaths::get_singleton()->get_cache_dir().path_join("assetimage_" + image_queue[p_queue_id].image_url.md5_text());
					String new_etag = headers[i].substr(headers[i].find_char(':') + 1).strip_edges();
					Ref<FileAccess> file = FileAccess::open(cache_filename_base + ".etag", FileAccess::WRITE);
					if (file.is_valid()) {
						file->store_line(new_etag);
					}

					int len = p_data.size();
					const uint8_t *r = p_data.ptr();
					file = FileAccess::open(cache_filename_base + ".data", FileAccess::WRITE);
					if (file.is_valid()) {
						file->store_32(len);
						file->store_buffer(r, len);
					}

					break;
				}
			}
		}

		image_queue[p_queue_id].data = const_cast<PackedByteArray &>(p_data);
		image_queue[p_queue_id].use_cache = p_code == HTTPClient::RESPONSE_NOT_MODIFIED;
		set_process(true);
		image_queue[p_queue_id].thread->start(_image_update, &image_queue[p_queue_id]);
	} else {
		if (is_print_verbose_enabled()) {
			WARN_PRINT(vformat("Asset Store: Error getting image from '%s' for asset # %d.", image_queue[p_queue_id].image_url, image_queue[p_queue_id].asset_id));
		}

		Object *obj = ObjectDB::get_instance(image_queue[p_queue_id].target);
		if (obj) {
			obj->call("set_image", image_queue[p_queue_id].image_type, image_queue[p_queue_id].image_index, get_editor_theme_icon(SNAME("FileBrokenBigThumb")));
		}

		image_queue[p_queue_id].request->queue_free();
		image_queue.erase(p_queue_id);
		_update_image_queue();
	}
}

void EditorAssetLibrary::_update_image_queue() {
	const int max_images = 6;
	int current_images = 0;

	List<int> to_delete;
	for (KeyValue<int, ImageQueue> &E : image_queue) {
		if (!E.value.active && current_images < max_images) {
			String cache_filename_base = EditorPaths::get_singleton()->get_cache_dir().path_join("assetimage_" + E.value.image_url.md5_text());
			Vector<String> headers;

			if (FileAccess::exists(cache_filename_base + ".etag") && FileAccess::exists(cache_filename_base + ".data")) {
				Ref<FileAccess> file = FileAccess::open(cache_filename_base + ".etag", FileAccess::READ);
				if (file.is_valid()) {
					headers.push_back("If-None-Match: " + file->get_line());
				}
			}

			Error err = E.value.request->request(E.value.image_url, headers);
			if (err != OK) {
				to_delete.push_back(E.key);
			} else {
				E.value.active = true;
			}
		}

		current_images++;
	}

	while (to_delete.size()) {
		image_queue[to_delete.front()->get()].request->queue_free();
		image_queue.erase(to_delete.front()->get());
		to_delete.pop_front();
	}
}

void EditorAssetLibrary::_request_image(ObjectID p_for, int p_asset_id, const String &p_image_url, ImageType p_type, int p_image_index) {
	// Remove extra spaces around the URL. This isn't strictly valid, but recoverable.
	String trimmed_url = p_image_url.strip_edges();
	if (trimmed_url != p_image_url && is_print_verbose_enabled()) {
		WARN_PRINT(vformat("Asset Store: Badly formatted image URL '%s' for asset # %d.", p_image_url, p_asset_id));
	}

	// Validate the image URL first.
	{
		String url_scheme;
		String url_host;
		int url_port;
		String url_path;
		String url_fragment;
		Error err = trimmed_url.parse_url(url_scheme, url_host, url_port, url_path, url_fragment);
		if (err != OK) {
			if (is_print_verbose_enabled()) {
				ERR_PRINT(vformat("Asset Store: Invalid image URL '%s' for asset # %d.", trimmed_url, p_asset_id));
			}

			Object *obj = ObjectDB::get_instance(p_for);
			if (obj) {
				obj->call("set_image", p_type, p_image_index, get_editor_theme_icon(SNAME("FileBrokenBigThumb")));
			}
			return;
		}
	}

	ImageQueue iq;
	iq.image_url = trimmed_url;
	iq.image_index = p_image_index;
	iq.image_type = p_type;
	iq.request = memnew(HTTPRequest);
	setup_http_request(iq.request);

	iq.target = p_for;
	iq.asset_id = p_asset_id;
	iq.queue_id = ++last_queue_id;
	iq.active = false;
	iq.thread = memnew(Thread);

	iq.request->connect("request_completed", callable_mp(this, &EditorAssetLibrary::_image_request_completed).bind(iq.queue_id));

	image_queue[iq.queue_id] = iq;
	add_child(iq.request);

	_update_image_queue();
}

void EditorAssetLibrary::_repository_changed(int p_repository_id) {
	_set_library_message(TTRC("Loading..."));

	asset_top_page->hide();
	asset_bottom_page->hide();
	asset_items->hide();

	filter->set_editable(false);
	sort->set_disabled(true);
	categories->set_disabled(true);

	host = repository->get_item_metadata(p_repository_id);
	_api_request("", REQUESTING_CHECK);
}

void EditorAssetLibrary::_licenses_id_pressed(int p_id) {
	licenses->get_popup()->set_item_checked(p_id, !licenses->get_popup()->is_item_checked(p_id));
	licenses_changed = true;
}

void EditorAssetLibrary::_licenses_popup_hide() {
	if (licenses_changed) {
		licenses_changed = false;
		_search();
	}
}

void EditorAssetLibrary::_search(int p_page) {
	ERR_FAIL_COND(p_page <= 0);

	String search = filter->get_text().to_lower();
	String args = "?query=" + search.uri_encode();

	if (templates_only) {
		args += "%23template";
	} else if (categories->get_selected() > 0) {
		args = args.replace("%23template", ""); // Bad user, no templates in projects!
		args += "%23" + (String)categories->get_item_metadata(categories->get_selected());
	}

	args += "&require_release=true";

	Dictionary version = Engine::get_singleton()->get_version_info();
	args += "&compatibility=" + (String)version["major"] + (String)version["minor"];
	if ((int)version["patch"] > 0) {
		args += (String)version["patch"];
	}

	args += String() + "&sort=" + sort_key[sort->get_selected()];

	int license_count = licenses->get_item_count();
	if (license_count > 0) {
		PopupMenu *popup = licenses->get_popup();
		for (int i = 0; i < license_count; i++) {
			if (popup->is_item_checked(i)) {
				args += "&licenses=" + (String)popup->get_item_metadata(i);
			}
		}
	}

	current_page = p_page;
	if (p_page > 1) {
		args += "&page=" + itos(p_page);
	}

	_api_request("search/query/" + args, REQUESTING_SEARCH);
}

void EditorAssetLibrary::_request_current_config() {
	_repository_changed(repository->get_selected());
}

HBoxContainer *EditorAssetLibrary::_make_pages(int p_page, int p_page_count, int p_page_len, int p_total_items, int p_current_items) {
	HBoxContainer *hbc = memnew(HBoxContainer);

	if (p_page_count < 2) {
		return hbc;
	}

	// 🎜 Do the Mario! Eat your arms, and then again... 🎜
	int from = p_page - (5 / EDSCALE);
	if (from < 1) {
		from = 1;
	}
	int to = from + (10 / EDSCALE);
	if (to > p_page_count) {
		to = p_page_count;
	}

	hbc->add_spacer();
	hbc->add_theme_constant_override("separation", 5 * EDSCALE);

	Button *first = memnew(Button);
	first->set_text(TTR("First", "Pagination"));
	first->set_theme_type_variation("PanelBackgroundButton");
	if (p_page != 1) {
		first->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_search).bind(1));
	} else {
		first->set_disabled(true);
		first->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	}
	hbc->add_child(first);

	Button *prev = memnew(Button);
	prev->set_text(TTR("Previous", "Pagination"));
	prev->set_theme_type_variation("PanelBackgroundButton");
	if (p_page > 1) {
		prev->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_search).bind(p_page - 1));
	} else {
		prev->set_disabled(true);
		prev->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	}
	hbc->add_child(prev);
	hbc->add_child(memnew(VSeparator));

	for (int i = from; i <= to; i++) {
		Button *current = memnew(Button);
		// Add padding to make page number buttons easier to click.
		current->set_text(vformat(" %d ", i));
		current->set_theme_type_variation("PanelBackgroundButton");
		if (i == p_page) {
			current->set_disabled(true);
			current->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
		} else {
			current->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_search).bind(i));
		}
		hbc->add_child(current);
	}

	Button *next = memnew(Button);
	next->set_text(TTR("Next", "Pagination"));
	next->set_theme_type_variation("PanelBackgroundButton");
	if (p_page < p_page_count) {
		next->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_search).bind(p_page + 1));
	} else {
		next->set_disabled(true);
		next->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	}
	hbc->add_child(memnew(VSeparator));
	hbc->add_child(next);

	Button *last = memnew(Button);
	last->set_text(TTR("Last", "Pagination"));
	last->set_theme_type_variation("PanelBackgroundButton");
	if (p_page != p_page_count) {
		last->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_search).bind(p_page_count));
	} else {
		last->set_disabled(true);
		last->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	}
	hbc->add_child(last);

	hbc->add_spacer();

	return hbc;
}

void EditorAssetLibrary::_api_request(const String &p_request, RequestType p_request_type, bool p_is_parallel) {
	if (!p_is_parallel) {
		if ((RequestType)request->get_meta("requesting") != REQUESTING_NONE) {
			request->cancel_request();
		}
		error_hb->hide();
	}

	if (loading_blocked) {
		_set_library_message_with_action(TTRC("The Asset Store requires an online connection and involves sending data over the internet."), TTRC("Go Online"), callable_mp(this, &EditorAssetLibrary::_force_online_mode));
		return;
	}

	HTTPRequest *requester = nullptr;
	if (p_is_parallel) {
		requester = memnew(HTTPRequest);
		add_child(requester);
		setup_http_request(requester);
		requester->connect("request_completed", callable_mp(this, &EditorAssetLibrary::_http_request_completed).bind(requester));
	} else {
		requester = request;
		// Make it clear that it's busy.
		library_scroll->set_modulate(Color(1, 1, 1, 0.5));
	}

	requester->set_meta("requesting", p_request_type);
	requester->request(host + "/" + p_request);
}

void EditorAssetLibrary::_http_request_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data, HTTPRequest *p_requester) {
	String str = String::utf8((const char *)p_data.ptr(), (int)p_data.size());
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
		case HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR:
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
			error_label->set_text(TTRC("Request failed, too many redirects"));

		} break;
		default: {
			if (p_code != 200) {
				error_label->set_text(TTR("Request failed, return code:") + " " + itos(p_code));
			} else {
				error_abort = false;
			}
		} break;
	}

	RequestType requested = p_requester->get_meta("requesting");
	if (p_requester != request) {
		// This was done as a parallel request, so free the node.
		p_requester->queue_free();
	} else {
		// Not busy anymore.
		library_scroll->set_modulate(Color(1, 1, 1));
	}

	if (error_abort) {
		if (requested == REQUESTING_CHECK) {
			_set_library_message_with_action(TTRC("Failed to verify repository."), TTRC("Retry"), callable_mp(this, &EditorAssetLibrary::_request_current_config));
		}
		error_hb->show();
		return;
	}

	Variant dt;
	{
		JSON json;
		json.parse(str);
		dt = json.get_data();
	}

	switch (requested) {
		case REQUESTING_CHECK: {
			if (!templates_only) {
				_api_request("tags/?featured_only=true", REQUESTING_TAGS, true);
			}
			_api_request("licenses/", REQUESTING_LICENSES, true);

			filter->set_editable(true);
			sort->set_disabled(false);

			_search();
		} break;

		case REQUESTING_TAGS: {
			categories->clear();

			if (templates_only) {
				categories->add_item(TTRC("Template"));
			} else {
				categories->add_item(TTRC("All"));
				categories->set_disabled(false);
			}

			Array arr = dt;
			for (int i = arr.size() - 1; i >= 0; i--) {
				Dictionary d = arr[i];
				if (!d.has("display_name") || !d.has("slug")) {
					continue;
				}

				String name = d["display_name"];
				String slug = d["slug"];

				// No temolates inside projects.
				if (slug == "template") {
					continue;
				}

				categories->add_item(name);
				categories->set_item_metadata(-1, slug);
			}
		} break;

		case REQUESTING_LICENSES: {
			Array arr = dt;
			PopupMenu *popup = licenses->get_popup();
			for (int i = 0; i < arr.size(); i++) {
				Dictionary d = arr[i];
				if (d.has("type")) {
					popup->add_check_item(d["type"]);
					popup->set_item_checked(-1, true);
					popup->set_item_metadata(-1, String(d["type"]).uri_encode());
				}
			}

			// TODO: Uncomment this once the API is updated.
			// licenses->set_disabled(false);
		} break;

		case REQUESTING_SEARCH: {
			initial_loading = false;

			if (asset_items) {
				memdelete(asset_items);
			}

			if (asset_top_page) {
				memdelete(asset_top_page);
			}

			if (asset_bottom_page) {
				memdelete(asset_bottom_page);
			}

			Dictionary d = dt;
			ERR_FAIL_COND(!d.has("count"));
			ERR_FAIL_COND(!d.has("hits"));

			int page_len = 24; // API's default batch size.
			int total_items = d["count"];
			int pages = total_items / page_len;
			current_page = MIN(current_page, pages);
			Array result = d["hits"];

			asset_top_page = _make_pages(current_page, pages, page_len, total_items, result.size());
			library_vb->add_child(asset_top_page);

			asset_items = memnew(GridContainer);
			_update_asset_items_columns();
			asset_items->add_theme_constant_override("h_separation", 10 * EDSCALE);
			asset_items->add_theme_constant_override("v_separation", 10 * EDSCALE);

			library_vb->add_child(asset_items);

			asset_bottom_page = _make_pages(current_page, pages, page_len, total_items, result.size());
			library_vb->add_child(asset_bottom_page);

			if (result.is_empty()) {
				if (!filter->get_text().is_empty()) {
					_set_library_message(
							vformat(TTR("No results for \"%s\"."), filter->get_text()));
				} else {
					// No results, even though the user didn't search for anything specific.
					// This is typically because the version number changed recently
					// and no assets compatible with the new version have been published yet.
					_set_library_message(
							vformat(TTR("No results compatible with %s %s."), String(GODOT_VERSION_SHORT_NAME).capitalize(), String(GODOT_VERSION_BRANCH)));
				}
			} else {
				library_message_box->hide();
			}

			for (int i = 0; i < result.size(); i++) {
				ERR_CONTINUE(!Dictionary(result[i]).has("asset"));
				Dictionary r = result[i].get("asset");

				ERR_CONTINUE(!r.has("name"));
				ERR_CONTINUE(!r.has("slug"));
				ERR_CONTINUE(!r.has("store_url"));
				ERR_CONTINUE(!r.has("license_type"));
				ERR_CONTINUE(!r.has("license_url"));
				ERR_CONTINUE(!r.has("reviews_score"));

				ERR_CONTINUE(!r.has("publisher"));
				Dictionary p = r["publisher"];
				ERR_CONTINUE(!p.has("name"));

				String author_id;
				// Don't allow to open profile links for alternative repositories.
				if (repository->get_selected() == 0) {
					ERR_CONTINUE(!p.has("slug"));
					author_id = p["slug"];
				}

				EditorAssetLibraryItem *item = memnew(EditorAssetLibraryItem(true));
				asset_items->add_child(item);
				asset_items->connect(SceneStringName(sort_children), callable_mp(item, &EditorAssetLibraryItem::calculate_misc_links_ratio));
				item->configure(r["name"], r["slug"], p["name"], author_id, r["license_type"], r["license_url"], r["reviews_score"]);
				item->connect("asset_selected", callable_mp(this, &EditorAssetLibrary::_select_asset));

				if (r.has("thumbnail") && !r["thumbnail"].operator String().is_empty()) {
					_request_image(item->get_instance_id(), r["slug"], r["thumbnail"], IMAGE_QUEUE_THUMBNAIL, 0);
				}
			}

			if (!result.is_empty()) {
				library_scroll->set_v_scroll(0);
			}
		} break;

		case REQUESTING_ASSET: {
			Dictionary d = dt;

			ERR_FAIL_COND(!d.has("name"));
			ERR_FAIL_COND(!d.has("slug"));
			ERR_FAIL_COND(!d.has("store_url"));
			ERR_FAIL_COND(!d.has("license_type"));
			ERR_FAIL_COND(!d.has("license_url"));
			ERR_FAIL_COND(!d.has("reviews_score"));
			ERR_FAIL_COND(!d.has("description"));
			ERR_FAIL_COND(!d.has("source"));
			ERR_FAIL_COND(!d.has("store_url"));

			ERR_FAIL_COND(!d.has("publisher"));
			Dictionary p = d["publisher"];
			ERR_FAIL_COND(!p.has("name"));
			ERR_FAIL_COND(!p.has("slug"));

			HashMap<String, String> tags;
			if (d.has("tags")) {
				Array t = d["tags"];
				for (const Variant &V : t) {
					const Dictionary tag = V;
					ERR_FAIL_COND(!tag.has("display_name"));
					ERR_FAIL_COND(!tag.has("slug"));

					tags[tag["display_name"]] = tag["slug"];
				}
			}

			if (description) {
				memdelete(description);
			}

			description = memnew(EditorAssetLibraryItemDescription);
			add_child(description);
			description->connect(SNAME("install_requested"), callable_mp(this, &EditorAssetLibrary::_install_asset));
			description->connect(SNAME("tag_clicked"), callable_mp(this, &EditorAssetLibrary::_tag_clicked));

			description->configure(d["name"], d["slug"], p["name"], p["slug"], d["license_type"], d["license_url"], d["reviews_score"], d["description"], tags, d["store_url"], d["source"]);

			EditorAssetLibraryItemDownload *download_item = _get_asset_in_progress(d["slug"]);
			if (download_item) {
				if (download_item->can_install()) {
					description->set_install_mode(EditorAssetLibraryItemDescription::MODE_INSTALL);
				} else {
					description->set_install_mode(EditorAssetLibraryItemDescription::MODE_DOWNLOADING);
				}
			}

			if (d.has("thumbnail") && !d["thumbnail"].operator String().is_empty()) {
				_request_image(description->get_instance_id(), d["slug"], d["thumbnail"], IMAGE_QUEUE_THUMBNAIL, 0);
			}

			int preview_index = 0;
			if (d.has("video_playback_url")) {
				String video = d["video_playback_url"];
				String thumb = d["video_thumbnail_url"];
				if (!video.is_empty() && !thumb.is_empty()) {
					description->add_preview(0, true, video, thumb);
					_request_image(description->get_instance_id(), d["slug"], thumb, IMAGE_QUEUE_VIDEO_THUMBNAIL, preview_index);
					preview_index = 1;
				}
			}

			if (d.has("media")) {
				Array previews = d["media"];
				for (int i = 0; i < previews.size(); i++) {
					description->add_preview(preview_index);
					if (i == 0) {
						description->preview_click(preview_index);
					}

					_request_image(description->get_instance_id(), d["slug"], previews[i], IMAGE_QUEUE_SCREENSHOT, preview_index);
					preview_index++;
				}
			}

			description->popup_centered();

			_api_request("releases/" + String(p["slug"]) + "/" + String(d["slug"]) + "/", REQUESTING_RELEASES);
		} break;

		case REQUESTING_RELEASES: {
			if (!description) {
				return;
			}

			Array arr = dt;
			// Iterate backwards, so the newer releases are added first.
			for (int i = arr.size() - 1; i >= 0; i--) {
				Dictionary d = arr[i];

				ERR_FAIL_COND(!d.has("download_url"));
				ERR_FAIL_COND(!d.has("version"));
				ERR_FAIL_COND(!d.has("stable"));

				String version = d["version"];
				if (!d["stable"]) {
					version += "(" + TTR("Unstable") + ")";
				}

				description->add_release(d["download_url"], version, "");
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
	asset_installer->set_asset_name(p_file);
	add_child(asset_installer);
	asset_installer->open_asset(p_file);
}

void EditorAssetLibrary::_asset_open() {
	asset_open->popup_file_dialog();
}

void EditorAssetLibrary::_manage_plugins() {
	ProjectSettingsEditor::get_singleton()->popup_project_settings(true);
	ProjectSettingsEditor::get_singleton()->set_plugins_page();
}

EditorAssetLibraryItemDownload *EditorAssetLibrary::_get_asset_in_progress(const String &p_asset_id) const {
	for (int i = 0; i < downloads_hb->get_child_count(); i++) {
		EditorAssetLibraryItemDownload *d = Object::cast_to<EditorAssetLibraryItemDownload>(downloads_hb->get_child(i));
		if (d && d->get_asset_id() == p_asset_id) {
			return d;
		}
	}

	return nullptr;
}

void EditorAssetLibrary::_install_external_asset(const String &p_zip_path, const String &p_title) {
	emit_signal(SNAME("install_asset"), p_zip_path, p_title);
}

void EditorAssetLibrary::_update_asset_items_columns() {
	int new_columns = get_size().x / (450.0 * EDSCALE);
	new_columns = MAX(1, new_columns);

	if (new_columns != asset_items->get_columns()) {
		asset_items->set_columns(new_columns);
	}
}

void EditorAssetLibrary::_update_downloads_section() {
	const bool has_downloads = downloads_hb->get_child_count() > 0;
	downloads_scroll->set_visible(has_downloads);
	library_mc->set_theme_type_variation(has_downloads ? "NoBorderHorizontal" : "NoBorderHorizontalBottom");
	library_scroll->set_scroll_hint_mode(has_downloads ? ScrollContainer::SCROLL_HINT_MODE_ALL : ScrollContainer::SCROLL_HINT_MODE_TOP_AND_LEFT);
}

void EditorAssetLibrary::_set_library_message(const String &p_message) {
	library_message->set_text(p_message);

	if (library_message_action.is_valid()) {
		library_message_button->disconnect(SceneStringName(pressed), library_message_action);
		library_message_action = Callable();
	}
	library_message_button->hide();

	library_message_box->show();
}

void EditorAssetLibrary::_set_library_message_with_action(const String &p_message, const String &p_action_text, const Callable &p_action) {
	library_message->set_text(p_message);

	library_message_button->set_text(p_action_text);
	if (library_message_action.is_valid()) {
		library_message_button->disconnect(SceneStringName(pressed), library_message_action);
		library_message_action = Callable();
	}
	library_message_action = p_action;
	library_message_button->connect(SceneStringName(pressed), library_message_action);
	library_message_button->show();

	library_message_box->show();
}

void EditorAssetLibrary::_force_online_mode() {
	EditorSettings::get_singleton()->set_setting("network/connection/network_mode", EditorSettings::NETWORK_ONLINE);
	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();
}

void EditorAssetLibrary::_bind_methods() {
	ADD_SIGNAL(MethodInfo("install_asset", PropertyInfo(Variant::STRING, "zip_path"), PropertyInfo(Variant::STRING, "name")));
}

EditorAssetLibrary::EditorAssetLibrary(bool p_templates_only) {
	templates_only = p_templates_only;
	loading_blocked = ((int)EDITOR_GET("network/connection/network_mode") == EditorSettings::NETWORK_OFFLINE);

	VBoxContainer *library_main = memnew(VBoxContainer);
	add_child(library_main);

	HBoxContainer *search_hb = memnew(HBoxContainer);

	library_main->add_child(search_hb);
	library_main->add_theme_constant_override("separation", 10 * EDSCALE);

	// Perform a search automatically if the user hasn't entered any text for a certain duration.
	// This way, the user doesn't need to press Enter to initiate their search.
	filter_debounce_timer = memnew(Timer);
	filter_debounce_timer->set_one_shot(true);
	filter_debounce_timer->set_wait_time(0.25);
	filter_debounce_timer->connect("timeout", callable_mp(this, &EditorAssetLibrary::_search).bind(1));
	search_hb->add_child(filter_debounce_timer);

	filter = memnew(LineEdit);
	if (templates_only) {
		filter->set_placeholder(TTRC("Search Templates. Use Tags by Including \"#tagname\""));
	} else {
		filter->set_placeholder(TTRC("Search Assets (Excluding Templates). Use Tags by Including \"#tagname\""));
	}
	filter->set_clear_button_enabled(true);
	search_hb->add_child(filter);
	filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filter->connect(SceneStringName(text_changed), callable_mp(filter_debounce_timer, &Timer::start).bind(-1).unbind(1));

	if (!p_templates_only) {
		search_hb->add_child(memnew(VSeparator));
	}

	Button *open_asset = memnew(Button);
	open_asset->set_text(TTRC("Import..."));
	search_hb->add_child(open_asset);
	open_asset->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_asset_open));

	Button *plugins = memnew(Button);
	plugins->set_text(TTRC("Plugins..."));
	search_hb->add_child(plugins);
	plugins->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibrary::_manage_plugins));

	if (p_templates_only) {
		open_asset->hide();
		plugins->hide();
	}

	HBoxContainer *search_hb2 = memnew(HBoxContainer);
	library_main->add_child(search_hb2);

	search_hb2->add_child(memnew(Label(TTRC("Sort:"))));
	sort = memnew(OptionButton);
	for (int i = 0; i < SORT_MAX; i++) {
		sort->add_item(sort_text[i]);
	}

	search_hb2->add_child(sort);

	sort->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	sort->set_clip_text(true);
	sort->connect(SceneStringName(item_selected), callable_mp(this, &EditorAssetLibrary::_search).bind(1).unbind(1));

	search_hb2->add_child(memnew(Label(TTRC("Category:"))));
	categories = memnew(OptionButton);
	if (p_templates_only) {
		categories->add_item(TTRC("Template"));
	} else {
		categories->add_item(TTRC("All"));
	}
	categories->set_disabled(true);
	categories->set_clip_text(true);
	categories->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_hb2->add_child(categories);
	categories->connect(SceneStringName(item_selected), callable_mp(this, &EditorAssetLibrary::_search).bind(1).unbind(1));

	search_hb2->add_child(memnew(Label(TTRC("Source:"))));
	repository = memnew(OptionButton);
	repository->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	repository->set_clip_text(true);
	search_hb2->add_child(repository);
	repository->connect(SceneStringName(item_selected), callable_mp(this, &EditorAssetLibrary::_repository_changed));
	_update_repository_options();

	search_hb2->add_child(memnew(VSeparator));

	licenses = memnew(MenuButton);
	licenses->set_text(TTRC("Licenses"));
	licenses->set_disabled(true);
	licenses->get_popup()->set_hide_on_checkable_item_selection(false);
	search_hb2->add_child(licenses);
	licenses->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &EditorAssetLibrary::_licenses_id_pressed));
	licenses->get_popup()->connect("popup_hide", callable_mp(this, &EditorAssetLibrary::_licenses_popup_hide));

	/////////

	library_mc = memnew(MarginContainer);
	library_mc->set_theme_type_variation("NoBorderHorizontalBottom");
	library_mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	library_main->add_child(library_mc);

	library_scroll = memnew(ScrollContainer);
	library_scroll->set_scroll_hint_mode(ScrollContainer::SCROLL_HINT_MODE_TOP_AND_LEFT);
	library_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	library_mc->add_child(library_scroll);

	Ref<StyleBoxEmpty> border2;
	border2.instantiate();
	border2->set_content_margin_individual(15 * EDSCALE, 15 * EDSCALE, 35 * EDSCALE, 15 * EDSCALE);

	PanelContainer *library_vb_border = memnew(PanelContainer);
	library_scroll->add_child(library_vb_border);
	library_vb_border->add_theme_style_override(SceneStringName(panel), border2);
	library_vb_border->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	library_vb = memnew(VBoxContainer);
	library_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	library_vb_border->add_child(library_vb);

	library_message_box = memnew(VBoxContainer);
	library_message_box->hide();
	library_vb->add_child(library_message_box);

	library_message = memnew(Label);
	library_message->set_focus_mode(FOCUS_ACCESSIBILITY);
	library_message->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	library_message_box->add_child(library_message);

	library_message_button = memnew(Button);
	library_message_button->set_h_size_flags(SIZE_SHRINK_CENTER);
	library_message_button->set_theme_type_variation("PanelBackgroundButton");
	library_message_box->add_child(library_message_button);

	asset_top_page = memnew(HBoxContainer);
	library_vb->add_child(asset_top_page);

	asset_items = memnew(GridContainer);
	asset_items->add_theme_constant_override("h_separation", 10 * EDSCALE);
	asset_items->add_theme_constant_override("v_separation", 10 * EDSCALE);

	library_vb->add_child(asset_items);

	asset_bottom_page = memnew(HBoxContainer);
	library_vb->add_child(asset_bottom_page);

	request = memnew(HTTPRequest);
	request->set_meta("requesting", REQUESTING_NONE);
	add_child(request);
	setup_http_request(request);
	request->connect("request_completed", callable_mp(this, &EditorAssetLibrary::_http_request_completed).bind(request));

	last_queue_id = 0;

	library_vb->add_theme_constant_override("separation", 20 * EDSCALE);

	error_hb = memnew(HBoxContainer);
	library_main->add_child(error_hb);
	error_label = memnew(Label);
	error_label->set_focus_mode(FOCUS_ACCESSIBILITY);
	error_hb->add_child(error_label);
	error_tr = memnew(TextureRect);
	error_tr->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	error_hb->add_child(error_tr);

	set_process_shortcut_input(true); // Global shortcuts since there is no main element to be focused.

	downloads_scroll = memnew(ScrollContainer);
	downloads_scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	downloads_scroll->set_theme_type_variation("ScrollContainerSecondary");
	library_main->add_child(downloads_scroll);
	downloads_hb = memnew(HBoxContainer);
	downloads_scroll->add_child(downloads_hb);
	downloads_hb->connect("child_entered_tree", callable_mp(this, &EditorAssetLibrary::_update_downloads_section).unbind(1));

	asset_open = memnew(EditorFileDialog);

	asset_open->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	asset_open->add_filter("*.zip", TTRC("Assets ZIP File"));
	asset_open->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(asset_open);
	asset_open->connect("file_selected", callable_mp(this, &EditorAssetLibrary::_asset_file_selected));
}

///////

bool AssetLibraryEditorPlugin::is_available() {
#ifdef WEB_ENABLED
	// Asset Store can't work on Web editor for now as most assets are sourced
	// directly from GitHub which does not set CORS.
	return false;
#else
	return StreamPeerTLS::is_available() && !Engine::get_singleton()->is_recovery_mode_hint();
#endif
}

const Ref<Texture2D> AssetLibraryEditorPlugin::get_plugin_icon() const {
	return EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("AssetStore"), EditorStringName(EditorIcons));
}

void AssetLibraryEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		addon_library->show();
	} else {
		addon_library->hide();
	}
}

AssetLibraryEditorPlugin::AssetLibraryEditorPlugin() {
	addon_library = memnew(EditorAssetLibrary);
	addon_library->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(addon_library);
	addon_library->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	addon_library->hide();
}
