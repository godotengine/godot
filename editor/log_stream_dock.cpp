// Minimal LogStream dock UI.
#include "log_stream_dock.h"

#include "core/logstream/log_stream.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/spin_box.h"
#include "scene/main/timer.h"

LogStreamDock *LogStreamDock::singleton = nullptr;

LogStreamDock *LogStreamDock::get_singleton() {
	return singleton;
}

static Color _level_color(LogStreamEntry::Level p_level) {
	switch (p_level) {
		case LogStreamEntry::LEVEL_ERROR:
			return Color(1, 0.35, 0.35); // red
		case LogStreamEntry::LEVEL_WARNING:
			return Color(1, 0.75, 0.35); // orange/yellow
		case LogStreamEntry::LEVEL_INFO:
			return Color(0.8, 0.8, 0.8); // light gray
		case LogStreamEntry::LEVEL_DEBUG:
			return Color(0.65, 0.8, 1.0); // blue-ish
		case LogStreamEntry::LEVEL_VERBOSE:
			return Color(0.7, 0.7, 1.0); // soft blue
	}
	return Color(1, 1, 1);
}

static String _format_entry_bbcode(const LogStreamEntry &e) {
	const Color c = _level_color(e.level);
	String color_tag = vformat("[color=#%02x%02x%02x]", int(c.r * 255), int(c.g * 255), int(c.b * 255));
	String level = LogStreamEntry::level_to_string(e.level);
	String ts = e.timestamp_usec > 0 ? vformat("%0.3f", e.timestamp_usec / 1000000.0) : String();
	String line = color_tag;
	if (!ts.is_empty()) {
		line += vformat("[%s] ", ts);
	}
	line += vformat("[%s] %s", level, e.message);
	if (!e.function.is_empty()) {
		line += vformat(" (%s)", e.function);
	}
	if (!e.file.is_empty() && e.line > 0) {
		line += vformat(" [url=%s:%d]%s:%d[/url]", e.file, e.line, e.file, e.line);
	}
	if (!e.category.is_empty()) {
		line += vformat(" {%s}", e.category);
	}
	line += "[/color]";
	return line;
}

LogStreamDock::LogStreamDock() {
	singleton = this;

	VBoxContainer *root = memnew(VBoxContainer);
	add_child(root);

	HBoxContainer *controls = memnew(HBoxContainer);
	root->add_child(controls);

	chk_error = memnew(CheckBox);
	chk_error->set_text("Error");
	chk_error->set_pressed(true);
	controls->add_child(chk_error);

	chk_warning = memnew(CheckBox);
	chk_warning->set_text("Warn");
	chk_warning->set_pressed(true);
	controls->add_child(chk_warning);

	chk_info = memnew(CheckBox);
	chk_info->set_text("Info");
	chk_info->set_pressed(true);
	controls->add_child(chk_info);

	chk_debug = memnew(CheckBox);
	chk_debug->set_text("Debug");
	controls->add_child(chk_debug);

	chk_verbose = memnew(CheckBox);
	chk_verbose->set_text("Verbose");
	controls->add_child(chk_verbose);

	search_box = memnew(LineEdit);
	search_box->set_placeholder(TTR("Search..."));
	search_box->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
	search_box->connect(SceneStringName(text_changed), callable_mp(this, &LogStreamDock::_on_search_changed));
	controls->add_child(search_box);

	pause_button = memnew(Button);
	pause_button->set_text(TTR("Pause"));
	pause_button->connect(SceneStringName(pressed), callable_mp(this, &LogStreamDock::_on_pause_toggled));
	controls->add_child(pause_button);

	chk_autoscroll = memnew(CheckBox);
	chk_autoscroll->set_text(TTR("Auto-scroll"));
	chk_autoscroll->set_pressed(true);
	chk_autoscroll->connect(SceneStringName(pressed), callable_mp(this, &LogStreamDock::_on_autoscroll_toggled));
	controls->add_child(chk_autoscroll);

	clear_button = memnew(Button);
	clear_button->set_text(TTR("Clear"));
	clear_button->connect(SceneStringName(pressed), callable_mp(this, &LogStreamDock::_on_clear));
	controls->add_child(clear_button);

	export_button = memnew(Button);
	export_button->set_text(TTR("Export"));
	export_button->connect(SceneStringName(pressed), callable_mp(this, &LogStreamDock::_on_export));
	controls->add_child(export_button);

	Button *toggle_ts = memnew(Button);
	toggle_ts->set_text(TTR("TS"));
	toggle_ts->set_toggle_mode(true);
	toggle_ts->set_pressed(true);
	toggle_ts->connect(SceneStringName(pressed), callable_mp(this, &LogStreamDock::_on_toggle_timestamp));
	controls->add_child(toggle_ts);

	Button *toggle_cat = memnew(Button);
	toggle_cat->set_text(TTR("Cat"));
	toggle_cat->set_toggle_mode(true);
	toggle_cat->set_pressed(true);
	toggle_cat->connect(SceneStringName(pressed), callable_mp(this, &LogStreamDock::_on_toggle_category));
	controls->add_child(toggle_cat);

	max_display_spin = memnew(SpinBox);
	max_display_spin->set_min(100);
	max_display_spin->set_max(20000);
	max_display_spin->set_step(100);
	max_display_spin->set_value(max_display_entries);
	max_display_spin->set_custom_minimum_size(Size2(120 * EDSCALE, 0));
	max_display_spin->connect(SceneStringName(value_changed), callable_mp(this, &LogStreamDock::_on_max_display_changed));
	controls->add_child(max_display_spin);

	log_label = memnew(RichTextLabel);
	log_label->set_v_size_flags(SIZE_EXPAND_FILL);
	log_label->set_use_bbcode(true);
	log_label->set_scroll_active(true);
	log_label->set_selection_enabled(true);
	log_label->connect("meta_clicked", callable_mp(this, &LogStreamDock::_on_log_meta_clicked));
	root->add_child(log_label);

	timer = memnew(Timer);
	timer->set_wait_time(0.5);
	timer->set_one_shot(false);
	timer->set_autostart(true);
	timer->connect("timeout", callable_mp(this, &LogStreamDock::_on_timer_timeout));
	add_child(timer);
}

void LogStreamDock::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		_load_filter_state();
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		_save_filter_state();
		if (timer) {
			timer->stop();
		}
	}
}

void LogStreamDock::_on_timer_timeout() {
	if (paused) {
		return;
	}
	refresh();
}

void LogStreamDock::_on_pause_toggled() {
	paused = !paused;
	pause_button->set_text(paused ? TTR("Resume") : TTR("Pause"));
	if (!paused) {
		refresh();
	}
}

void LogStreamDock::_on_autoscroll_toggled() {
	paused = !chk_autoscroll->is_pressed();
	pause_button->set_text(paused ? TTR("Resume") : TTR("Pause"));
	if (!paused) {
		refresh();
	}
}

void LogStreamDock::_on_clear() {
	if (LogStreamRouter::get_singleton()) {
		LogStreamRouter::get_singleton()->clear_entries();
	}
	log_label->clear();
	last_seq_rendered = 0;
}

void LogStreamDock::_on_export() {
	if (!LogStreamRouter::get_singleton()) {
		return;
	}
	String export_path = "user://logs/logstream_export.jsonl";
	Ref<FileAccess> fa = FileAccess::open(export_path, FileAccess::WRITE);
	if (fa.is_null()) {
		return;
	}

	Vector<LogStreamEntry> entries = _filter_entries(LogStreamRouter::get_singleton()->get_entries_snapshot());
	for (const LogStreamEntry &e : entries) {
		Dictionary d;
		d["seq"] = (int64_t)e.seq;
		d["ts_usec"] = (int64_t)e.timestamp_usec;
		d["level"] = LogStreamEntry::level_to_string(e.level);
		d["message"] = e.message;
		d["file"] = e.file;
		d["line"] = e.line;
		d["function"] = e.function;
		d["category"] = e.category;
		d["stack"] = e.stack;
		d["project"] = e.project;
		d["session_id"] = e.session_id;
		fa->store_string(JSON::stringify(d) + "\n");
	}
	fa->flush();
}

void LogStreamDock::_on_search_changed(const String &p_text) {
	if (!paused) {
		refresh();
	}
	last_seq_rendered = 0; // force full redraw with new filter
}

void LogStreamDock::_on_toggle_timestamp() {
	show_timestamp = !show_timestamp;
	refresh();
}

void LogStreamDock::_on_toggle_category() {
	show_category = !show_category;
	refresh();
}

void LogStreamDock::_on_max_display_changed(double p_value) {
	max_display_entries = (int)p_value;
	if (!paused) {
		refresh();
	} else {
		last_seq_rendered = 0;
	}
}

void LogStreamDock::_on_log_meta_clicked(const Variant &p_meta) {
	String meta = p_meta;
	if (!meta.contains_char(':')) {
		return;
	}
	const PackedStringArray parts = meta.rsplit(":", true, 1);
	if (parts.size() != 2) {
		return;
	}
	String path = parts[0];
	int line = parts[1].to_int() - 1;

	if (path.begins_with("res://")) {
		if (ResourceLoader::exists(path)) {
			const Ref<Resource> res = ResourceLoader::load(path);
			if (res.is_valid()) {
				ScriptEditor::get_singleton()->edit(res, line, 0);
				return;
			}
		}
	}
	if (!ScriptEditorPlugin::open_in_external_editor(path, line, -1, true)) {
		OS::get_singleton()->shell_open(path);
	}
}

bool LogStreamDock::_allow_level(LogStreamEntry::Level p_level) const {
	switch (p_level) {
		case LogStreamEntry::LEVEL_ERROR:
			return chk_error->is_pressed();
		case LogStreamEntry::LEVEL_WARNING:
			return chk_warning->is_pressed();
		case LogStreamEntry::LEVEL_INFO:
			return chk_info->is_pressed();
		case LogStreamEntry::LEVEL_DEBUG:
			return chk_debug->is_pressed();
		case LogStreamEntry::LEVEL_VERBOSE:
			return chk_verbose->is_pressed();
	}
	return true;
}

Vector<LogStreamEntry> LogStreamDock::_filter_entries(const Vector<LogStreamEntry> &p_entries) const {
	Vector<LogStreamEntry> out;
	String query = search_box->get_text();
	for (const LogStreamEntry &e : p_entries) {
		if (!_allow_level(e.level)) {
			continue;
		}
		if (!query.is_empty()) {
			if (!e.message.containsn(query) && !e.file.containsn(query) && !e.category.containsn(query)) {
				continue;
			}
		}
		out.push_back(e);
	}
	return out;
}

void LogStreamDock::refresh() {
	if (!LogStreamRouter::get_singleton()) {
		return;
	}
	Vector<LogStreamEntry> all_entries = LogStreamRouter::get_singleton()->get_entries_snapshot();
	Vector<LogStreamEntry> entries = _filter_entries(all_entries);

	const bool full_redraw = last_seq_rendered == 0;
	if (full_redraw) {
		log_label->clear();
	}

	// Limit displayed entries to keep UI responsive.
	if (entries.size() > max_display_entries) {
		entries = entries.slice(entries.size() - max_display_entries, max_display_entries);
		log_label->clear();
	}

	for (const LogStreamEntry &e : entries) {
		if (!full_redraw && e.seq <= last_seq_rendered) {
			continue;
		}
		LogStreamEntry copy = e;
		if (!show_timestamp) {
			copy.timestamp_usec = 0;
		}
		if (!show_category) {
			copy.category = "";
		}
		log_label->append_text(_format_entry_bbcode(copy) + "\n");
		last_seq_rendered = MAX(last_seq_rendered, e.seq);
	}
	log_label->scroll_to_line(log_label->get_line_count());
}

void LogStreamDock::_load_filter_state() {
	EditorSettings *es = EditorSettings::get_singleton();
	if (!es) {
		return;
	}
	if (!bool(es->get_setting("logstream/ui/persist_filters"))) {
		return;
	}
	chk_error->set_pressed(bool(es->get_setting("logstream/ui/filter_error")));
	chk_warning->set_pressed(bool(es->get_setting("logstream/ui/filter_warning")));
	chk_info->set_pressed(bool(es->get_setting("logstream/ui/filter_info")));
	chk_debug->set_pressed(bool(es->get_setting("logstream/ui/filter_debug")));
	chk_verbose->set_pressed(bool(es->get_setting("logstream/ui/filter_verbose")));
	search_box->set_text(es->get_setting("logstream/ui/search"));
	show_timestamp = bool(es->get_setting("logstream/ui/show_timestamp"));
	show_category = bool(es->get_setting("logstream/ui/show_category"));
	max_display_entries = int(es->get_setting("logstream/ui/max_display"));
	if (max_display_spin) {
		max_display_spin->set_value(max_display_entries);
	}
	chk_autoscroll->set_pressed(es->get_setting("logstream/ui/auto_scroll"));
	paused = !chk_autoscroll->is_pressed();
	last_seq_rendered = 0;
}

void LogStreamDock::_save_filter_state() {
	EditorSettings *es = EditorSettings::get_singleton();
	if (!es) {
		return;
	}
	if (!bool(es->get_setting("logstream/ui/persist_filters"))) {
		return;
	}
	es->set_setting("logstream/ui/filter_error", chk_error->is_pressed());
	es->set_setting("logstream/ui/filter_warning", chk_warning->is_pressed());
	es->set_setting("logstream/ui/filter_info", chk_info->is_pressed());
	es->set_setting("logstream/ui/filter_debug", chk_debug->is_pressed());
	es->set_setting("logstream/ui/filter_verbose", chk_verbose->is_pressed());
	es->set_setting("logstream/ui/search", search_box->get_text());
	es->set_setting("logstream/ui/show_timestamp", show_timestamp);
	es->set_setting("logstream/ui/show_category", show_category);
	es->set_setting("logstream/ui/max_display", max_display_entries);
	es->set_setting("logstream/ui/auto_scroll", chk_autoscroll->is_pressed());
	es->save();
}

