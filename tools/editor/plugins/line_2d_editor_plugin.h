#ifndef LINE_2D_EDITOR_PLUGIN_H
#define LINE_2D_EDITOR_PLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/2d/path_2d.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/button_group.h"
#include "scene/2d/line_2d.h"


class CanvasItemEditor;

class Line2DEditor : public HBoxContainer {
	GDCLASS(Line2DEditor, HBoxContainer)

public:
	bool forward_input_event(const InputEvent& p_event);
	void edit(Node *p_line2d);
	Line2DEditor(EditorNode *p_editor);

protected:
	void _node_removed(Node *p_node);
	void _notification(int p_what);

	Vector2 mouse_to_local_pos(Vector2 mpos);

	static void _bind_methods();

private:
	void _mode_selected(int p_mode);
	void _canvas_draw();
	void _node_visibility_changed();

	int get_point_index_at(Vector2 gpos);
	Vector2 mouse_to_local_pos(Vector2 gpos, bool alt);

	UndoRedo *undo_redo;

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;
	Panel *panel;
	Line2D *node;

	HBoxContainer *base_hb;
	Separator *sep;

	enum Mode {
		MODE_CREATE = 0,
		MODE_EDIT,
		MODE_DELETE,
		_MODE_COUNT
	};

	Mode mode;
	ToolButton* toolbar_buttons[_MODE_COUNT];

	bool _dragging;
	int action_point;
	Point2 moving_from;
	Point2 moving_screen_from;
};

class Line2DEditorPlugin : public EditorPlugin {
	GDCLASS( Line2DEditorPlugin, EditorPlugin )

public:
	virtual bool forward_canvas_input_event(const Transform2D& p_canvas_xform,const InputEvent& p_event) { return line2d_editor->forward_input_event(p_event); }

	virtual String get_name() const { return "Line2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	Line2DEditorPlugin(EditorNode *p_node);

private:
	Line2DEditor *line2d_editor;
	EditorNode *editor;

};

#endif // LINE_2D_EDITOR_PLUGIN_H

