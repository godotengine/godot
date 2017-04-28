#ifndef CURVE_EDITOR_PLUGIN_H
#define CURVE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"

class CurveTextureEdit : public Control {

	GDCLASS(CurveTextureEdit, Control);

	struct Point {

		float offset;
		float height;
		bool operator<(const Point &p_ponit) const {
			return offset < p_ponit.offset;
		}
	};

	bool grabbing;
	int grabbed;
	Vector<Point> points;
	float max, min;

	void _plot_curve(const Vector2 &p_a, const Vector2 &p_b, const Vector2 &p_c, const Vector2 &p_d);

protected:
	void _gui_input(const InputEvent &p_event);
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_range(float p_min, float p_max);
	void set_points(const Vector<Vector2> &p_points);
	Vector<Vector2> get_points() const;
	virtual Size2 get_minimum_size() const;
	CurveTextureEdit();
};

class CurveTextureEditorPlugin : public EditorPlugin {

	GDCLASS(CurveTextureEditorPlugin, EditorPlugin);

	CurveTextureEdit *curve_editor;
	Ref<CurveTexture> curve_texture_ref;
	EditorNode *editor;
	ToolButton *curve_button;

protected:
	static void _bind_methods();
	void _curve_changed();
	void _undo_redo_curve_texture(const PoolVector<Vector2> &points);
	void _curve_settings_changed();

public:
	virtual String get_name() const { return "CurveTexture"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	CurveTextureEditorPlugin(EditorNode *p_node);
	~CurveTextureEditorPlugin();
};

#endif // CURVE_EDITOR_PLUGIN_H
