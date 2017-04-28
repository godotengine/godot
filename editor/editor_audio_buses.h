/*************************************************************************/
/*  editor_audio_buses.h                                                 */
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
#ifndef EDITORAUDIOBUSES_H
#define EDITORAUDIOBUSES_H

#include "editor/editor_file_dialog.h"
#include "editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/line_edit.h"
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

	GDCLASS(EditorAudioBus, PanelContainer)

	bool prev_active;
	float peak_l;
	float peak_r;

	Ref<Texture> disabled_vu;
	LineEdit *track_name;
	VSlider *slider;
	TextureProgress *vu_l;
	TextureProgress *vu_r;
	TextureRect *scale;
	OptionButton *send;

	PopupMenu *effect_options;
	PopupMenu *delete_popup;
	PopupMenu *delete_effect_popup;

	Button *solo;
	Button *mute;
	Button *bypass;

	Tree *effects;

	bool updating_bus;

	void _gui_input(const InputEvent &p_event);
	void _delete_pressed(int p_option);

	void _name_changed(const String &p_new_name);
	void _name_focus_exit() { _name_changed(track_name->get_text()); }
	void _volume_db_changed(float p_db);
	void _solo_toggled();
	void _mute_toggled();
	void _bypass_toggled();
	void _send_selected(int p_which);
	void _effect_edited();
	void _effect_add(int p_which);
	void _effect_selected();
	void _delete_effect_pressed(int p_option);
	void _effect_rmb(const Vector2 &p_pos);

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

	EditorAudioBus(EditorAudioBuses *p_buses = NULL);
};

class EditorAudioBusDrop : public Panel {

	GDCLASS(EditorAudioBusDrop, Panel);

	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

protected:
	static void _bind_methods();

public:
	EditorAudioBusDrop();
};

class EditorAudioBuses : public VBoxContainer {

	GDCLASS(EditorAudioBuses, VBoxContainer)

	HBoxContainer *top_hb;

	Button *add;
	ScrollContainer *bus_scroll;
	HBoxContainer *bus_hb;

	EditorAudioBusDrop *drop_end;

	Button *file;
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

#endif // EDITORAUDIOBUSES_H
