#ifndef PATCH_9_FRAME_H
#define PATCH_9_FRAME_H

#include "scene/gui/control.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Patch9Frame : public Control {

	OBJ_TYPE(Patch9Frame,Control);

	bool draw_center;
	int margin[4];
	Color modulate;
	Ref<Texture> texture;
protected:

	void _notification(int p_what);
	virtual Size2 get_minimum_size() const;
	static void _bind_methods();

public:

	void set_texture(const Ref<Texture>& p_tex);
	Ref<Texture> get_texture() const;

	void set_modulate(const Color& p_tex);
	Color get_modulate() const;

	void set_patch_margin(Margin p_margin,int p_size);
	int get_patch_margin(Margin p_margin) const;

	void set_draw_center(bool p_enable);
	bool get_draw_center() const;

	Patch9Frame();
	~Patch9Frame();

};
#endif // PATCH_9_FRAME_H
