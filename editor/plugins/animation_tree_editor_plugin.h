#ifndef ANIMATION_TREE_EDITOR_PLUGIN_H
#define ANIMATION_TREE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/animation/animation_tree.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class AnimationTreeNodeEditorPlugin : public VBoxContainer {
	GDCLASS(AnimationTreeNodeEditorPlugin, VBoxContainer)
public:
	virtual bool can_edit(const Ref<AnimationNode> &p_node) = 0;
	virtual void edit(const Ref<AnimationNode> &p_node) = 0;
};

class AnimationTreeEditor : public VBoxContainer {

	GDCLASS(AnimationTreeEditor, VBoxContainer);

	ScrollContainer *path_edit;
	HBoxContainer *path_hb;

	AnimationTree *tree;
	PanelContainer *editor_base;

	Vector<String> button_path;
	Vector<String> edited_path;
	Vector<AnimationTreeNodeEditorPlugin *> editors;

	void _update_path();
	void _about_to_show_root();
	ObjectID current_root;

	void _path_button_pressed(int p_path);

	static Vector<String> get_animation_list();

protected:
	void _notification(int p_what);
	static void _bind_methods();

	static AnimationTreeEditor *singleton;

public:
	AnimationTree *get_tree() { return tree; }
	void add_plugin(AnimationTreeNodeEditorPlugin *p_editor);
	void remove_plugin(AnimationTreeNodeEditorPlugin *p_editor);

	String get_base_path();

	bool can_edit(const Ref<AnimationNode> &p_node) const;

	void edit_path(const Vector<String> &p_path);
	Vector<String> get_edited_path() const;

	void enter_editor(const String &p_path = "");
	static AnimationTreeEditor *get_singleton() { return singleton; }
	void edit(AnimationTree *p_tree);
	AnimationTreeEditor();
};

class AnimationTreeEditorPlugin : public EditorPlugin {

	GDCLASS(AnimationTreeEditorPlugin, EditorPlugin);

	AnimationTreeEditor *anim_tree_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "AnimationTree"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	AnimationTreeEditorPlugin(EditorNode *p_node);
	~AnimationTreeEditorPlugin();
};

#endif // ANIMATION_TREE_EDITOR_PLUGIN_H
