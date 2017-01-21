#ifndef EDITORAUDIOBUSES_H
#define EDITORAUDIOBUSES_H


#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/slider.h"
#include "scene/gui/texture_progress.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"
#include "scene/gui/option_button.h"

class EditorAudioBuses;

class EditorAudioBus : public PanelContainer {

	GDCLASS( EditorAudioBus, PanelContainer )

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

	Button *solo;
	Button *mute;
	Button *bypass;

	Tree *effects;

	bool updating_bus;

	void _name_changed(const String& p_new_name);
	void _name_focus_exit() { _name_changed(track_name->get_text()); }
	void _volume_db_changed(float p_db);
	void _solo_toggled();
	void _mute_toggled();
	void _bypass_toggled();
	void _send_selected(int p_which);
	void _effect_edited();
	void _effect_add(int p_which);
	void _effect_selected();

friend class EditorAudioBuses;

	EditorAudioBuses *buses;

protected:

	static void _bind_methods();
	void _notification(int p_what);
public:

	void update_bus();
	void update_send();

	EditorAudioBus(EditorAudioBuses *p_buses=NULL);
};


class EditorAudioBuses : public VBoxContainer  {

	GDCLASS(EditorAudioBuses,VBoxContainer)

	HBoxContainer *top_hb;

	Button *add;
	ToolButton *buses;
	ToolButton *groups;
	ScrollContainer *bus_scroll;
	HBoxContainer *bus_hb;
	ScrollContainer *group_scroll;
	HBoxContainer *group_hb;

	void _add_bus();
	void _update_buses();
	void _update_bus(int p_index);
	void _update_sends();


protected:

	static void _bind_methods();
	void _notification(int p_what);
public:



	static void register_editor();

	EditorAudioBuses();
};

#endif // EDITORAUDIOBUSES_H
