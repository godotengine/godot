#ifndef ANIMATION_STATE_MACHINE_EDITOR_H
#define ANIMATION_STATE_MACHINE_EDITOR_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/animation/animation_node_state_machine.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class AnimationNodeStateMachineEditor : public VBoxContainer {

	GDCLASS(AnimationNodeStateMachineEditor, VBoxContainer);

	Ref<AnimationNodeStateMachine> state_machine;

	ToolButton *tool_select;
	ToolButton *tool_create;
	ToolButton *tool_connect;
	LineEdit *name_edit;

	HBoxContainer *tool_erase_hb;
	ToolButton *tool_erase;
	ToolButton *tool_autoplay;
	ToolButton *tool_end;

	OptionButton *transition_mode;
	OptionButton *play_mode;

	HBoxContainer *goto_parent_hbox;
	ToolButton *goto_parent;

	PanelContainer *panel;

	StringName selected_node;

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;

	Control *state_machine_draw;
	Control *state_machine_play_pos;

	PanelContainer *error_panel;
	Label *error_label;

	bool updating;

	UndoRedo *undo_redo;

	static AnimationNodeStateMachineEditor *singleton;

	void _state_machine_gui_input(const Ref<InputEvent> &p_event);
	void _connection_draw(const Vector2 &p_from, const Vector2 &p_to, AnimationNodeStateMachineTransition::SwitchMode p_mode, bool p_enabled, bool p_selected, bool p_travel, bool p_auto_advance);
	void _state_machine_draw();
	void _state_machine_pos_draw();

	void _update_graph();

	PopupMenu *menu;
	PopupMenu *animations_menu;
	Vector<String> animations_to_add;

	Vector2 add_node_pos;

	bool dragging_selected_attempt;
	bool dragging_selected;
	Vector2 drag_from;
	Vector2 drag_ofs;
	StringName snap_x;
	StringName snap_y;

	bool connecting;
	StringName connecting_from;
	Vector2 connecting_to;
	StringName connecting_to_node;

	void _add_menu_type(int p_index);
	void _add_animation_type(int p_index);

	void _goto_parent();

	void _removed_from_graph();

	struct NodeRect {
		StringName node_name;
		Rect2 node;
		Rect2 play;
		Rect2 name;
		Rect2 edit;
	};

	Vector<NodeRect> node_rects;

	struct TransitionLine {
		StringName from_node;
		StringName to_node;
		Vector2 from;
		Vector2 to;
		AnimationNodeStateMachineTransition::SwitchMode mode;
		bool disabled;
		bool auto_advance;
		float width;
	};

	Vector<TransitionLine> transition_lines;

	StringName selected_transition_from;
	StringName selected_transition_to;

	bool over_text;
	StringName over_node;
	int over_node_what;

	String prev_name;
	void _name_edited(const String &p_text);
	void _open_editor(const String &p_name);
	void _scroll_changed(double);

	void _clip_src_line_to_rect(Vector2 &r_from, Vector2 &r_to, const Rect2 &p_rect);
	void _clip_dst_line_to_rect(Vector2 &r_from, Vector2 &r_to, const Rect2 &p_rect);

	void _erase_selected();
	void _update_mode();
	void _autoplay_selected();
	void _end_selected();

	bool last_active;
	StringName last_blend_from_node;
	StringName last_current_node;
	Vector<StringName> last_travel_path;
	float last_play_pos;

	float error_time;
	String error_text;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AnimationNodeStateMachineEditor *get_singleton() { return singleton; }
	void edit(AnimationNodeStateMachine *p_state_machine);
	AnimationNodeStateMachineEditor();
};

class AnimationNodeStateMachineEditorPlugin : public EditorPlugin {

	GDCLASS(AnimationNodeStateMachineEditorPlugin, EditorPlugin);

	AnimationNodeStateMachineEditor *anim_tree_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "StateMachine"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	AnimationNodeStateMachineEditorPlugin(EditorNode *p_node);
	~AnimationNodeStateMachineEditorPlugin();
};

#endif // ANIMATION_STATE_MACHINE_EDITOR_H
