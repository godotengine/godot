#ifndef SCENE_GUI_COLOR_RAMP_EDIT_H_
#define SCENE_GUI_COLOR_RAMP_EDIT_H_

#include "scene/gui/popup.h"
#include "scene/gui/color_picker.h"
#include "scene/resources/color_ramp.h"
#include "scene/resources/default_theme/theme_data.h"

#define POINT_WIDTH 8

class ColorRampEdit : public Control {

	OBJ_TYPE(ColorRampEdit,Control);

	PopupPanel *popup;
	ColorPicker *picker;

	Ref<ImageTexture> checker;

	bool grabbing;
	int grabbed;
	Vector<ColorRamp::Point> points;

	void _draw_checker(int x, int y, int w, int h);
	void _color_changed(const Color& p_color);
	int _get_point_from_pos(int x);
	void _show_color_picker();

protected:
	void _input_event(const InputEvent& p_event);
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_ramp(const Vector<float>& p_offsets,const Vector<Color>& p_colors);
	Vector<float> get_offsets() const;
	Vector<Color> get_colors() const;
	void set_points(Vector<ColorRamp::Point>& p_points);
	Vector<ColorRamp::Point>& get_points();
	virtual Size2 get_minimum_size() const;

	ColorRampEdit();
	virtual ~ColorRampEdit();
};

/*class  ColorRampEditPanel : public Panel
{
	OBJ_TYPE(ColorRampEditPanel, Panel );
};*/


#endif /* SCENE_GUI_COLOR_RAMP_EDIT_H_ */
