#ifndef GRADIENT_TEXTURE_EDITOR_PLUGIN_H
#define GRADIENT_TEXTURE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/resources/texture.h"

class GradientTextureEdit : public Control {

	GDCLASS(GradientTextureEdit, Control);

	PopupPanel *popup;
	ColorPicker *picker;

	Ref<ImageTexture> checker;

	bool grabbing;
	int grabbed;
	Vector<GradientTexture::Point> points;

	void _draw_checker(int x, int y, int w, int h);
	void _color_changed(const Color &p_color);
	int _get_point_from_pos(int x);
	void _show_color_picker();

protected:
	void _gui_input(const InputEvent &p_event);
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_ramp(const Vector<float> &p_offsets, const Vector<Color> &p_colors);
	Vector<float> get_offsets() const;
	Vector<Color> get_colors() const;
	void set_points(Vector<GradientTexture::Point> &p_points);
	Vector<GradientTexture::Point> &get_points();
	virtual Size2 get_minimum_size() const;

	GradientTextureEdit();
	virtual ~GradientTextureEdit();
};

class GradientTextureEditorPlugin : public EditorPlugin {

	GDCLASS(GradientTextureEditorPlugin, EditorPlugin);

	bool _2d;
	Ref<GradientTexture> gradient_texture_ref;
	GradientTextureEdit *ramp_editor;
	EditorNode *editor;
	ToolButton *gradient_button;

protected:
	static void _bind_methods();
	void _ramp_changed();
	void _undo_redo_gradient_texture(const Vector<float> &offsets, const Vector<Color> &colors);

public:
	virtual String get_name() const { return "GradientTexture"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	GradientTextureEditorPlugin(EditorNode *p_node);
	~GradientTextureEditorPlugin();
};

#endif // GRADIENT_TEXTURE_EDITOR_PLUGIN_H
