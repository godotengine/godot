#ifndef COLLISION_SHAPE_2D_EDITOR_PLUGIN_H
#define COLLISION_SHAPE_2D_EDITOR_PLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"

#include "scene/2d/collision_shape_2d.h"

class CanvasItemEditor;

class CollisionShape2DEditor : public Control {
	OBJ_TYPE(CollisionShape2DEditor, Control);

	enum ShapeType {
		CAPSULE_SHAPE,
		CIRCLE_SHAPE,
		CONCAVE_POLYGON_SHAPE,
		CONVEX_POLYGON_SHAPE,
		LINE_SHAPE,
		RAY_SHAPE,
		RECTANGLE_SHAPE,
		SEGMENT_SHAPE
	};

	EditorNode* editor;
	UndoRedo* undo_redo;
	CanvasItemEditor* canvas_item_editor;
	CollisionShape2D* node;

	Vector<Point2> handles;

	int shape_type;
	int edit_handle;
	bool pressed;
	Variant original;

	Variant get_handle_value(int idx) const;
	void set_handle(int idx, Point2& p_point);
	void commit_handle(int idx, Variant& p_org);

	void _get_current_shape_type();
	void _canvas_draw();

protected:
	static void _bind_methods();

public:
	bool forward_input_event(const InputEvent& p_event);
	void edit(Node* p_node);

	CollisionShape2DEditor(EditorNode* p_editor);
};

class CollisionShape2DEditorPlugin : public EditorPlugin {
	OBJ_TYPE(CollisionShape2DEditorPlugin, EditorPlugin);

	CollisionShape2DEditor* collision_shape_2d_editor;
	EditorNode* editor;

public:
	virtual bool forward_input_event(const InputEvent& p_event) { return collision_shape_2d_editor->forward_input_event(p_event); }

	virtual String get_name() const { return "CollisionShape2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object* p_obj);
	virtual bool handles(Object* p_obj) const;
	virtual void make_visible(bool visible);

	CollisionShape2DEditorPlugin(EditorNode* p_editor);
	~CollisionShape2DEditorPlugin();
};

#endif //COLLISION_SHAPE_2D_EDITOR_PLUGIN_H
