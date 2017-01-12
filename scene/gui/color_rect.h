#ifndef COLORRECT_H
#define COLORRECT_H

#include "scene/gui/control.h"

class ColorRect : public Control  {
	GDCLASS(ColorRect,Control)

	Color color;
protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_frame_color(const Color& p_color);
	Color get_frame_color() const;

	ColorRect();
};

#endif // COLORRECT_H
