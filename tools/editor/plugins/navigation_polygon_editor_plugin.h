#ifndef NAVIGATIONPOLYGONEDITORPLUGIN_H
#define NAVIGATIONPOLYGONEDITORPLUGIN_H



#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/2d/navigation_polygon.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/button_group.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CanvasItemEditor;

class NavigationPolygonEditor : public HBoxContainer {

	OBJ_TYPE(NavigationPolygonEditor, HBoxContainer );

	UndoRedo *undo_redo;
	enum Mode {

		MODE_CREATE,
		MODE_EDIT,

	};

	Mode mode;

	ToolButton *button_create;
	ToolButton *button_edit;

	ConfirmationDialog *create_nav;

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;
	Panel *panel;
	NavigationPolygonInstance *node;
	MenuButton *options;

	int edited_outline;
	int edited_point;
	Vector2 edited_point_pos;
	DVector<Vector2> pre_move_edit;
	Vector<Vector2> wip;
	bool wip_active;


	void _wip_close();
	void _canvas_draw();
	void _create_nav();

	void _menu_option(int p_option);

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();
public:

	Vector2 snap_point(const Vector2& p_point) const;
	bool forward_input_event(const InputEvent& p_event);
	void edit(Node *p_collision_polygon);
	NavigationPolygonEditor(EditorNode *p_editor);
};

class NavigationPolygonEditorPlugin : public EditorPlugin {

	OBJ_TYPE( NavigationPolygonEditorPlugin, EditorPlugin );

	NavigationPolygonEditor *collision_polygon_editor;
	EditorNode *editor;

public:

	virtual bool forward_input_event(const InputEvent& p_event) { return collision_polygon_editor->forward_input_event(p_event); }

	virtual String get_name() const { return "NavigationPolygonInstance"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	NavigationPolygonEditorPlugin(EditorNode *p_node);
	~NavigationPolygonEditorPlugin();

};


#endif // NAVIGATIONPOLYGONEDITORPLUGIN_H
