#ifndef VIEWPORTCONTAINER_H
#define VIEWPORTCONTAINER_H

#include "scene/gui/container.h"

class ViewportContainer : public Container {

	GDCLASS( ViewportContainer, Container );

	bool stretch;
protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_stretch(bool p_enable);
	bool is_stretch_enabled() const;

	virtual Size2 get_minimum_size() const;

	ViewportContainer();
};

#endif // VIEWPORTCONTAINER_H
