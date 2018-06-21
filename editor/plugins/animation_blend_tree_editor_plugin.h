#ifndef ANIMATION_BLEND_TREE_EDITOR_PLUGIN_H
#define ANIMATION_BLEND_TREE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class AnimationNodeBlendTreeEditor : public VBoxContainer {

	GDCLASS(AnimationNodeBlendTreeEditor, VBoxContainer);

	Ref<AnimationNodeBlendTree> blend_tree;
	GraphEdit *graph;
	MenuButton *add_node;
	Button *goto_parent;

	PanelContainer *error_panel;
	Label *error_label;

	UndoRedo *undo_redo;

	AcceptDialog *filter_dialog;
	Tree *filters;
	CheckBox *filter_enabled;

	Map<StringName, ProgressBar *> animations;

	void _update_graph();

	struct AddOption {
		String name;
		String type;
		Ref<Script> script;
		AddOption(const String &p_name = String(), const String &p_type = String()) {
			name = p_name;
			type = p_type;
		}
	};

	Vector<AddOption> add_options;

	void _add_node(int p_idx);
	void _update_options_menu();

	static AnimationNodeBlendTreeEditor *singleton;

	void _node_dragged(const Vector2 &p_from, const Vector2 &p_to, Ref<AnimationNode> p_node);
	void _node_renamed(const String &p_text, Ref<AnimationNode> p_node);
	void _node_renamed_focus_out(Node *le, Ref<AnimationNode> p_node);

	bool updating;

	void _connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);
	void _disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);

	void _scroll_changed(const Vector2 &p_scroll);
	void _node_selected(Object *p_node);
	void _open_in_editor(const String &p_which);
	void _open_parent();
	void _anim_selected(int p_index, Array p_options, const String &p_node);
	void _delete_request(const String &p_which);
	void _oneshot_start(const StringName &p_name);
	void _oneshot_stop(const StringName &p_name);

	bool _update_filters(const Ref<AnimationNode> &anode);
	void _edit_filters(const String &p_which);
	void _filter_edited();
	void _filter_toggled();
	Ref<AnimationNode> _filter_edit;

	void _node_changed(ObjectID p_node);

	void _removed_from_graph();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AnimationNodeBlendTreeEditor *get_singleton() { return singleton; }

	void add_custom_type(const String &p_name, const Ref<Script> &p_script);
	void remove_custom_type(const Ref<Script> &p_script);

	virtual Size2 get_minimum_size() const;
	void edit(AnimationNodeBlendTree *p_blend_tree);
	AnimationNodeBlendTreeEditor();
};

class AnimationNodeBlendTreeEditorPlugin : public EditorPlugin {

	GDCLASS(AnimationNodeBlendTreeEditorPlugin, EditorPlugin);

	AnimationNodeBlendTreeEditor *anim_tree_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "BlendTree"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	AnimationNodeBlendTreeEditorPlugin(EditorNode *p_node);
	~AnimationNodeBlendTreeEditorPlugin();
};

#endif // ANIMATION_BLEND_TREE_EDITOR_PLUGIN_H
