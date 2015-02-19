#ifndef CANVASMODULATE_H
#define CANVASMODULATE_H

#include "scene/2d/node_2d.h"

class CanvasModulate : public Node2D {

	OBJ_TYPE(CanvasModulate,Node2D);

	Color color;
protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_color(const Color& p_color);
	Color get_color() const;

	CanvasModulate();
	~CanvasModulate();
};

#endif // CANVASMODULATE_H
