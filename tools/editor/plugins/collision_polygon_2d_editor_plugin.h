#ifndef COLLISION_POLYGON_2D_EDITOR_PLUGIN_H
#define COLLISION_POLYGON_2D_EDITOR_PLUGIN_H


#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/2d/collision_polygon_2d.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/button_group.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CanvasItemEditor;

class CollisionPolygon2DEditor : public HBoxContainer {

	OBJ_TYPE(CollisionPolygon2DEditor, HBoxContainer );

	UndoRedo *undo_redo;
	enum Mode {

		MODE_CREATE,
		MODE_EDIT,

	};

	Mode mode;

	ToolButton *button_create;
	ToolButton *button_edit;

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;
	Panel *panel;
	CollisionPolygon2D *node;
	MenuButton *options;

	int edited_point;
	Vector2 edited_point_pos;
	Vector<Vector2> pre_move_edit;
	Vector<Vector2> wip;
	bool wip_active;


	void _wip_close();
	void _canvas_draw();
	void _menu_option(int p_option);

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();
public:

	bool forward_input_event(const InputEvent& p_event);
	void edit(Node *p_collision_polygon);
	CollisionPolygon2DEditor(EditorNode *p_editor);
};

class CollisionPolygon2DEditorPlugin : public EditorPlugin {

	OBJ_TYPE( CollisionPolygon2DEditorPlugin, EditorPlugin );

	CollisionPolygon2DEditor *collision_polygon_editor;
	EditorNode *editor;

public:

	virtual bool forward_input_event(const InputEvent& p_event) { return collision_polygon_editor->forward_input_event(p_event); }

	virtual String get_name() const { return "CollisionPolygon2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	CollisionPolygon2DEditorPlugin(EditorNode *p_node);
	~CollisionPolygon2DEditorPlugin();

};

#endif // COLLISION_POLYGON_2D_EDITOR_PLUGIN_H
