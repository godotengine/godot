// Minimal LogStream dock UI.
#pragma once

#include "editor/docks/editor_dock.h"
#include "core/logstream/log_stream.h"

class Button;
class CheckBox;
class LineEdit;
class RichTextLabel;
class SpinBox;
class Timer;
class EditorSettings;
class ScriptEditorPlugin;

class LogStreamDock : public EditorDock {
	GDCLASS(LogStreamDock, EditorDock);

public:
	LogStreamDock();
	~LogStreamDock() override = default;

	static LogStreamDock *get_singleton();

	void refresh();

private:
	static LogStreamDock *singleton;

	RichTextLabel *log_label = nullptr;
	CheckBox *chk_error = nullptr;
	CheckBox *chk_warning = nullptr;
	CheckBox *chk_info = nullptr;
	CheckBox *chk_debug = nullptr;
	CheckBox *chk_verbose = nullptr;
	CheckBox *chk_autoscroll = nullptr;
	LineEdit *search_box = nullptr;
	Button *pause_button = nullptr;
	Button *clear_button = nullptr;
	Button *export_button = nullptr;
	SpinBox *max_display_spin = nullptr;
	Timer *timer = nullptr;
	bool paused = false;
	bool show_timestamp = true;
	bool show_category = true;
	uint64_t last_seq_rendered = 0;
	int max_display_entries = 2000;

	void _on_timer_timeout();
	void _on_pause_toggled();
	void _on_clear();
	void _on_export();
	void _on_search_changed(const String &p_text);
	void _on_log_meta_clicked(const Variant &p_meta);
	void _on_toggle_timestamp();
	void _on_toggle_category();
	void _on_autoscroll_toggled();
	void _on_max_display_changed(double p_value);

	bool _allow_level(LogStreamEntry::Level p_level) const;
	Vector<LogStreamEntry> _filter_entries(const Vector<LogStreamEntry> &p_entries) const;
	void _load_filter_state();
	void _save_filter_state();

protected:
	void _notification(int p_what);
};

