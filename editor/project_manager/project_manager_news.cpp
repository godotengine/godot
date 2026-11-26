/**************************************************************************/
/*  project_manager_news.cpp                                              */
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

#include "project_manager_news.h"

#include "core/io/http_client.h"
#include "core/io/xml_parser.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "core/string/regex.h"
#include "core/string/translation.h"
#include "core/version.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/link_button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/theme/theme_db.h"

ProjectManagerNews::ProjectManagerNews() {
	set_name("ProjectManagerNews");
	set_v_size_flags(Control::SIZE_EXPAND_FILL);

	// News content.
	{
		news_scroll = memnew(ScrollContainer);
		news_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		news_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		add_child(news_scroll);

		MarginContainer *content_margin = memnew(MarginContainer);
		content_margin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		content_margin->set_v_size_flags(Control::SIZE_EXPAND_FILL);

		content_margin->add_theme_constant_override("margin_left", 16 * EDSCALE);
		content_margin->add_theme_constant_override("margin_top", 16 * EDSCALE);
		content_margin->add_theme_constant_override("margin_right", 16 * EDSCALE);
		content_margin->add_theme_constant_override("margin_bottom", 16 * EDSCALE);

		news_scroll->add_child(content_margin);

		VBoxContainer *content = memnew(VBoxContainer);
		content->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		content->add_theme_constant_override("separation", 12 * EDSCALE);
		content_margin->add_child(content);

		news_heading = memnew(Label);
		news_heading->set_text(TTR("Latest News"));
		news_heading->add_theme_font_size_override(SceneStringName(font_size), 18 * EDSCALE);
		news_heading->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		content->add_child(news_heading);

		news_list = memnew(VBoxContainer);
		news_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		news_list->add_theme_constant_override("separation", 16 * EDSCALE);
		content->add_child(news_list);

		news_idle = memnew(VBoxContainer);
		news_idle->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		news_idle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		news_idle->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		news_idle->add_theme_constant_override("separation", 8 * EDSCALE);
		news_idle->hide();
		content->add_child(news_idle);

		status_label = memnew(Label);
		status_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		status_label->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
		news_idle->add_child(status_label);
	}

	// Status/refresh controls.
	{
		HBoxContainer *controls = memnew(HBoxContainer);
		add_child(controls);

		refresh_button = memnew(Button);
		refresh_button->set_text(TTR("Refresh"));
		controls->add_child(refresh_button);

		refresh_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectManagerNews::_refresh_pressed));
	}

	_set_news_state(NEWS_STATE_IDLE);
}

ProjectManagerNews::~ProjectManagerNews() {
	if (http_client) {
		memdelete(http_client);
		http_client = nullptr;
	}
}

void ProjectManagerNews::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_update_theme();
		} break;

		case NOTIFICATION_PROCESS: {
			if (http_client) {
				_poll_http();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;
	}
}

void ProjectManagerNews::_update_theme() {
	if (!is_inside_tree()) {
		return;
	}

	if (refresh_button) {
		refresh_button->set_button_icon(get_editor_theme_icon("Reload"));
	}
}

void ProjectManagerNews::_set_news_state(NewsState p_state) {
	news_state = p_state;

	switch (news_state) {
		case NEWS_STATE_IDLE: {
			news_heading->hide();
			status_label->set_text(TTR("News requires online access."));
			status_label->show();
			refresh_button->hide();
			news_scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
			news_idle->show();
		} break;

		case NEWS_STATE_CONNECTING:
		case NEWS_STATE_LOADING: {
			news_heading->hide();
			status_label->set_text(TTR("Loading news..."));
			status_label->show();
			refresh_button->set_disabled(true);
			refresh_button->show();
			news_idle->show();
		} break;

		case NEWS_STATE_ERROR: {
			news_heading->hide();
			refresh_button->set_disabled(false);
			refresh_button->show();
			news_idle->show();
		} break;

		case NEWS_STATE_READY: {
			news_heading->show();
			status_label->set_text(vformat(TTR("%d news items"), news_items.size()));
			status_label->hide();
			refresh_button->set_disabled(false);
			refresh_button->show();
			news_scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_AUTO);
			news_idle->hide();
		} break;
	}
}

void ProjectManagerNews::load_news() {
	refresh();
}

void ProjectManagerNews::refresh() {
	if (is_loading()) {
		return;
	}

	const int network_mode = EDITOR_GET("network/connection/network_mode");

	if (network_mode == EditorSettings::NETWORK_OFFLINE) {
		_set_news_state(NEWS_STATE_IDLE);
		return;
	}

	_fetch_news();
}

bool ProjectManagerNews::is_loading() const {
	return news_state == NEWS_STATE_CONNECTING || news_state == NEWS_STATE_LOADING;
}

void ProjectManagerNews::_fetch_news() {
	if (http_client) {
		return;
	}

	_clear_news();
	response_body.clear();

	request_sent = false;
	response_received = false;

	http_client = HTTPClient::create();

	if (!http_client) {
		_show_error(TTR("Unable to create an HTTP client."));
		return;
	}

	http_client->set_blocking_mode(false);

	_set_news_state(NEWS_STATE_CONNECTING);

	http_start_time = OS::get_singleton()->get_ticks_msec();

	Error err = http_client->connect_to_host("godotengine.org", 443, TLSOptions::client());

	if (err != OK) {
		_show_error(TTR("Unable to connect to the news server."));
		return;
	}

	set_process(true);
}

void ProjectManagerNews::_poll_http() {
	ERR_FAIL_NULL(http_client);

	http_client->poll();

	const HTTPClient::Status status = http_client->get_status();

	switch (status) {
		case HTTPClient::STATUS_CONNECTED: {
			// If the request has already completed and the connection is
			// still alive, the response body has been fully received.
			if (request_sent) {
				if (!response_body.is_empty()) {
					_parse_news(response_body);
				} else {
					_show_error(TTR("The news server returned an empty response."));
				}
				return;
			}

			Vector<String> headers;
			headers.push_back("User-Agent: Godot/" + String(GODOT_VERSION_NAME));
			headers.push_back("Accept: application/rss+xml, application/xml, text/xml");

			Error err = http_client->request(HTTPClient::METHOD_GET, "/rss.xml", headers, nullptr, 0);

			if (err != OK) {
				_show_error(TTR("Unable to request the news feed."));
				return;
			}

			request_sent = true;

			http_start_time = OS::get_singleton()->get_ticks_msec();
			_set_news_state(NEWS_STATE_LOADING);
		} break;

		case HTTPClient::STATUS_REQUESTING: {
			// Waiting for the response headers.
		} break;

		case HTTPClient::STATUS_BODY: {
			if (!response_received) {
				response_received = true;

				const int response_code = http_client->get_response_code();

				if (response_code < 200 || response_code >= 300) {
					_show_error(vformat(TTR("The news server returned HTTP status %d."), response_code));
					return;
				}
			}

			PackedByteArray chunk = http_client->read_response_body_chunk();

			if (!chunk.is_empty()) {
				response_body.append_array(chunk);
				http_start_time = OS::get_singleton()->get_ticks_msec();
			}
		} break;

		case HTTPClient::STATUS_CONNECTION_ERROR:
		case HTTPClient::STATUS_CANT_CONNECT:
		case HTTPClient::STATUS_CANT_RESOLVE:
		case HTTPClient::STATUS_TLS_HANDSHAKE_ERROR: {
			_show_error(TTR("Unable to connect to the Godot news feed."));
			return;
		}

		case HTTPClient::STATUS_DISCONNECTED: {
			if (response_body.is_empty()) {
				_show_error(TTR("The news server returned an empty response."));
				return;
			}

			_parse_news(response_body);
			return;
		}

		default: {
			const uint64_t elapsed = OS::get_singleton()->get_ticks_msec() - http_start_time;

			if (elapsed >= 30000) {
				_show_error(TTR("Timed out while loading the news feed."));
				return;
			}
		} break;
	}
}

String ProjectManagerNews::_strip_html(const String &p_html) const {
	String text = p_html;

	// Convert common HTML block elements into paragraph/line breaks
	// before stripping the remaining tags.
	static const RegEx paragraph_breaks("(?i)<\\s*/\\s*(p|div|li|h[1-6])\\s*>");

	static const RegEx line_breaks("(?i)<\\s*br\\s*/?\\s*>");

	text = paragraph_breaks.sub(text, "\n\n", true);
	text = line_breaks.sub(text, "\n", true);

	// Remove all remaining HTML tags.
	static const RegEx tags("<[^>]*>");
	text = tags.sub(text, "", true);

	// Decode XML/HTML entities
	text = text.xml_unescape();

	PackedStringArray paragraphs = text.split("\n\n", true);

	String result;

	for (const String &paragraph : paragraphs) {
		PackedStringArray lines = paragraph.split("\n", true);
		String paragraph_text;

		for (const String &line : lines) {
			const String clean_line = line.strip_edges();

			if (clean_line.is_empty()) {
				continue;
			}

			if (!paragraph_text.is_empty()) {
				paragraph_text += " ";
			}

			paragraph_text += clean_line;
		}

		paragraph_text = paragraph_text.strip_edges();

		if (paragraph_text.is_empty()) {
			continue;
		}

		if (!result.is_empty()) {
			result += "\n\n";
		}

		result += paragraph_text;
	}

	return result.strip_edges();
}

String ProjectManagerNews::_limit_description(const String &p_description) const {
	PackedStringArray paragraphs = p_description.split("\n\n", true);

	String result;
	int paragraph_count = 0;

	for (const String &paragraph : paragraphs) {
		const String clean_paragraph = paragraph.strip_edges();

		if (clean_paragraph.is_empty()) {
			continue;
		}

		if (paragraph_count >= 2) {
			break;
		}

		if (!result.is_empty()) {
			result += "\n\n";
		}

		result += clean_paragraph;
		paragraph_count++;
	}

	return result;
}

String ProjectManagerNews::_format_pub_date(const String &p_date) const {
	static const RegEx date_regex(
			"^([A-Za-z]{3}),\\s+"
			"(\\d{1,2})\\s+"
			"([A-Za-z]{3})\\s+"
			"(\\d{4})\\s+"
			"(\\d{1,2}):(\\d{2})(?::\\d{2})?\\s+"
			"([+-]\\d{4})$");

	Ref<RegExMatch> match = date_regex.search(p_date.strip_edges());

	if (match.is_null()) {
		// If the server changes its date format, don't destroy the
		// original value. Just display what we received.
		return p_date.strip_edges();
	}

	const String weekday = match->get_string(1);
	const int day = match->get_string(2).to_int();
	const String month = match->get_string(3);
	const int year = match->get_string(4).to_int();
	const int hour = match->get_string(5).to_int();
	const int minute = match->get_string(6).to_int();

	// Convert 24-hour time to 12-hour time.
	const String period = hour >= 12 ? "PM" : "AM";
	int display_hour = hour % 12;

	if (display_hour == 0) {
		display_hour = 12;
	}

	String result = vformat("%s, %d %s %d %d", weekday, day, month, year, display_hour);

	if (minute != 0) {
		result += vformat(":%02d", minute);
	}

	result += " " + period;

	return result;
}

void ProjectManagerNews::_parse_news(const PackedByteArray &p_data) {
	_clear_news();

	Ref<XMLParser> parser;
	parser.instantiate();

	Error err = parser->open_buffer(p_data);

	if (err != OK) {
		_show_error(TTR("Unable to parse the news feed."));
		return;
	}

	bool inside_item = false;
	NewsItem current_item;
	String current_element;

	while (parser->read() == OK) {
		switch (parser->get_node_type()) {
			case XMLParser::NODE_ELEMENT: {
				current_element = parser->get_node_name();

				if (current_element == "item") {
					inside_item = true;
					current_item = NewsItem();
				}
			} break;

			case XMLParser::NODE_TEXT: {
				if (!inside_item) {
					break;
				}

				const String text = parser->get_node_data();

				if (current_element == "title") {
					current_item.title += text.strip_edges();
				} else if (current_element == "link") {
					current_item.link += text.strip_edges();
				} else if (current_element == "description") {
					current_item.description += text;
				} else if (current_element == "pubDate") {
					current_item.pub_date += text.strip_edges();
				}
			} break;

			case XMLParser::NODE_ELEMENT_END: {
				const String element_name = parser->get_node_name();

				if (element_name == "item") {
					inside_item = false;

					current_item.title = current_item.title.strip_edges();
					current_item.link = current_item.link.strip_edges();
					current_item.description = _limit_description(_strip_html(current_item.description));
					current_item.pub_date = _format_pub_date(current_item.pub_date);

					if (!current_item.title.is_empty() && !current_item.link.is_empty()) {
						news_items.push_back(current_item);
					}
				}

				current_element = String();
			} break;

			default:
				break;
		}
	}

	if (news_items.is_empty()) {
		_show_error(TTR("No news items were found."));
		return;
	}

	_show_news();
}

void ProjectManagerNews::_clear_news_list() {
	if (!news_list) {
		return;
	}

	while (news_list->get_child_count() > 0) {
		Node *child = news_list->get_child(0);
		news_list->remove_child(child);
		memdelete(child);
	}
}

void ProjectManagerNews::_clear_news() {
	news_items.clear();
	_clear_news_list();
}

void ProjectManagerNews::_show_news() {
	// We don't call _clear_news() here because it also clears news_items, which would erase
	// the parsed feed before we get a chance to create the UI.
	_clear_news_list();

	for (const NewsItem &item : news_items) {
		_create_news_item(item);
	}

	_set_news_state(NEWS_STATE_READY);

	if (http_client) {
		memdelete(http_client);
		http_client = nullptr;
	}

	request_sent = false;
	response_received = false;

	set_process(false);
}

bool ProjectManagerNews::has_news() const {
	return !news_items.is_empty();
}

void ProjectManagerNews::_create_news_item(const NewsItem &p_item) {
	PanelContainer *panel = memnew(PanelContainer);
	panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox("panel_container", "ProjectManager"));
	news_list->add_child(panel);

	MarginContainer *margin = memnew(MarginContainer);
	margin->add_theme_constant_override("margin_left", 12 * EDSCALE);
	margin->add_theme_constant_override("margin_top", 12 * EDSCALE);
	margin->add_theme_constant_override("margin_right", 12 * EDSCALE);
	margin->add_theme_constant_override("margin_bottom", 12 * EDSCALE);
	panel->add_child(margin);

	VBoxContainer *container = memnew(VBoxContainer);
	container->add_theme_constant_override("separation", 8 * EDSCALE);
	margin->add_child(container);

	LinkButton *title = memnew(LinkButton);
	title->set_text(p_item.title);
	title->set_text_direction(TEXT_DIRECTION_AUTO);
	title->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	title->add_theme_font_override(SceneStringName(font), get_theme_font("bold", EditorStringName(EditorFonts)));
	title->add_theme_font_size_override(SceneStringName(font_size), 16 * EDSCALE);
	container->add_child(title);

	title->connect(SceneStringName(pressed), callable_mp(this, &ProjectManagerNews::_open_news_link).bind(p_item.link));

	if (!p_item.pub_date.is_empty()) {
		Label *date = memnew(Label);
		date->set_text(p_item.pub_date);
		date->add_theme_color_override(SceneStringName(font_color), get_theme_color("font_placeholder_color", EditorStringName(Editor)));
		container->add_child(date);
	}

	if (!p_item.description.is_empty()) {
		Label *description = memnew(Label);
		description->set_text(p_item.description);
		description->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
		description->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
		description->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		container->add_child(description);
	}

	LinkButton *read_more = memnew(LinkButton);
	read_more->set_text(TTR("Read more"));
	read_more->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
	read_more->set_text_direction(TEXT_DIRECTION_AUTO);
	container->add_child(read_more);

	read_more->connect(SceneStringName(pressed), callable_mp(this, &ProjectManagerNews::_open_news_link).bind(p_item.link));
}

void ProjectManagerNews::_show_error(const String &p_message) {
	if (http_client) {
		memdelete(http_client);
		http_client = nullptr;
	}

	set_process(false);

	request_sent = false;
	response_received = false;

	_set_news_state(NEWS_STATE_ERROR);

	status_label->set_text(p_message);
}

void ProjectManagerNews::_open_news_link(const String &p_url) {
	OS::get_singleton()->shell_open(p_url);
}

void ProjectManagerNews::_refresh_pressed() {
	refresh();
}
