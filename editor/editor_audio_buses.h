/**************************************************************************/
/*  editor_audio_buses.h                                                  */
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

#ifndef EDITOR_AUDIO_BUSES_H
#define EDITOR_AUDIO_BUSES_H

#include "editor/editor_file_dialog.h"
#include "editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/slider.h"
#include "scene/gui/texture_progress.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"

class EditorAudioBuses;

class EditorAudioBus : public PanelContainer {
	GDCLASS(EditorAudioBus, PanelContainer);

	Ref<Texture> disabled_vu;
	LineEdit *track_name;
	MenuButton *bus_options;
	VSlider *slider;

	int cc;
	static const int CHANNELS_MAX = 4;

	struct {
		bool prev_active;

		float peak_l;
		float peak_r;

		TextureProgress *vu_l;
		TextureProgress *vu_r;
	} channel[CHANNELS_MAX];

	OptionButton *send;

	PopupMenu *effect_options;
	PopupMenu *bus_popup;
	PopupMenu *delete_effect_popup;

	Panel *audio_value_preview_box;
	Label *audio_value_preview_label;
	Timer *preview_timer;

	Button *solo;
	Button *mute;
	Button *bypass;

	Tree *effects;

	bool updating_bus;
	bool is_master;
	mutable bool hovering_drop;

	void _gui_input(const Ref<InputEvent> &p_event);
	void _unhandled_key_input(Ref<InputEvent> p_event);
	void _bus_popup_pressed(int p_option);

	void _name_changed(const String &p_new_name);
	void _name_focus_exit() { _name_changed(track_name->get_text()); }
	void _volume_changed(float p_normalized);
	float _normalized_volume_to_scaled_db(float normalized);
	float _scaled_db_to_normalized_volume(float db);
	void _show_value(float slider_value);
	void _hide_value_preview();
	void _solo_toggled();
	void _mute_toggled();
	void _bypass_toggled();
	void _send_selected(int p_which);
	void _effect_edited();
	void _effect_add(int p_which);
	void _effect_selected();
	void _delete_effect_pressed(int p_option);
	void _effect_rmb(const Vector2 &p_pos);
	void _update_visible_channels();

	virtual Variant get_drag_data(const Point2 &p_point);
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	friend class EditorAudioBuses;

	EditorAudioBuses *buses;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void update_bus();
	void update_send();

	EditorAudioBus(EditorAudioBuses *p_buses = nullptr, bool p_is_master = false);
};

class EditorAudioBusDrop : public Control {
	GDCLASS(EditorAudioBusDrop, Control);

	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

	mutable bool hovering_drop;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	EditorAudioBusDrop();
};

class EditorAudioBuses : public VBoxContainer {
	GDCLASS(EditorAudioBuses, VBoxContainer);

	HBoxContainer *top_hb;

	ScrollContainer *bus_scroll;
	HBoxContainer *bus_hb;

	EditorAudioBusDrop *drop_end;

	Label *file;

	Button *add;
	Button *load;
	Button *save_as;
	Button *_default;
	Button *_new;

	Timer *save_timer;
	String edited_path;

	void _add_bus();
	void _update_buses();
	void _update_bus(int p_index);
	void _update_sends();

	void _delete_bus(Object *p_which);
	void _duplicate_bus(int p_which);
	void _reset_bus_volume(Object *p_which);

	void _request_drop_end();
	void _drop_at_index(int p_bus, int p_index);

	void _server_save();

	void _select_layout();
	void _load_layout();
	void _save_as_layout();
	void _load_default_layout();
	void _new_layout();

	EditorFileDialog *file_dialog;
	bool new_layout;

	void _file_dialog_callback(const String &p_string);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void open_layout(const String &p_path);

	static EditorAudioBuses *register_editor();

	EditorAudioBuses();
};

class EditorAudioMeterNotches : public Control {
	GDCLASS(EditorAudioMeterNotches, Control);

private:
	struct AudioNotch {
		float relative_position;
		float db_value;
		bool render_db_value;

		_FORCE_INLINE_ AudioNotch(float r_pos, float db_v, bool rndr_val) {
			relative_position = r_pos;
			db_value = db_v;
			render_db_value = rndr_val;
		}

		_FORCE_INLINE_ AudioNotch(const AudioNotch &n) {
			relative_position = n.relative_position;
			db_value = n.db_value;
			render_db_value = n.render_db_value;
		}

		_FORCE_INLINE_ AudioNotch operator=(const EditorAudioMeterNotches::AudioNotch &n) {
			relative_position = n.relative_position;
			db_value = n.db_value;
			render_db_value = n.render_db_value;
			return *this;
		}

		_FORCE_INLINE_ AudioNotch() {}
	};

	List<AudioNotch> notches;

public:
	const float line_length = 5.0f;
	const float label_space = 2.0f;
	const float btm_padding = 9.0f;
	const float top_padding = 5.0f;
	Color notch_color;

	void add_notch(float p_normalized_offset, float p_db_value, bool p_render_value = false);
	Size2 get_minimum_size() const;

private:
	static void _bind_methods();
	void _notification(int p_what);
	void _draw_audio_notches();

public:
	EditorAudioMeterNotches();
};

class AudioBusesEditorPlugin : public EditorPlugin {
	GDCLASS(AudioBusesEditorPlugin, EditorPlugin);

	EditorAudioBuses *audio_bus_editor;

public:
	virtual String get_name() const { return "SampleLibrary"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	AudioBusesEditorPlugin(EditorAudioBuses *p_node);
	~AudioBusesEditorPlugin();
};

#endif // EDITOR_AUDIO_BUSES_H
