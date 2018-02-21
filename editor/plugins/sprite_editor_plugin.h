#ifndef SPRITE_EDITOR_PLUGIN_H
#define SPRITE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/sprite.h"
#include "scene/gui/spin_box.h"

class SpriteEditor : public Control {

	GDCLASS(SpriteEditor, Control);

	enum Menu {
		MENU_OPTION_CREATE_MESH_2D,
	};

	Sprite *node;

	MenuButton *options;

	ConfirmationDialog *outline_dialog;

	AcceptDialog *err_dialog;

	ConfirmationDialog *debug_uv_dialog;
	Control *debug_uv;
	Vector<Vector2> uv_lines;

	Vector<Vector2> computed_vertices;
	Vector<Vector2> computed_uv;
	Vector<int> computed_indices;

	SpinBox *simplification;
	SpinBox *island_merging;
	Button *update_preview;

	void _menu_option(int p_option);

	//void _create_uv_lines();
	friend class SpriteEditorPlugin;

	void _debug_uv_draw();
	void _update_mesh_data();
	void _create_mesh_node();

protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(Sprite *p_sprite);
	SpriteEditor();
};

class SpriteEditorPlugin : public EditorPlugin {

	GDCLASS(SpriteEditorPlugin, EditorPlugin);

	SpriteEditor *sprite_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "Sprite"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	SpriteEditorPlugin(EditorNode *p_node);
	~SpriteEditorPlugin();
};

#endif // SPRITE_EDITOR_PLUGIN_H
