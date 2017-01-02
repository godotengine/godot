#ifndef LISTENER_2D_H
#define LISTENER_2D_H

#include "scene/2d/node_2d.h"

class Listener2D : public Node2D {

	OBJ_TYPE(Listener2D, Node2D);
	
private:

	bool current;
	
protected:

	void _update_listener();

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _notification(int p_what);

	static void _bind_methods();

public:
	
	void set_current(bool p_current);
	bool is_current() const;

	Listener2D();
	~Listener2D();

};

#endif
