#ifndef LISTENER_2D_H
#define LISTENER_2D_H


#include "scene/2d/node_2d.h"
#include "scene/main/viewport.h"

class Listener2D : public Node2D {

	OBJ_TYPE(Listener2D, Node2D);
private:

	bool force_change;
	bool current;

	RID scenario_id;

	virtual bool _can_gizmo_scale() const;
	virtual RES _get_gizmo_geometry() const;

friend class Viewport;
	void _update_audio_listener_state();
protected:

	void _update_listener();
	virtual void _request_listener_update();

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

public:

	void make_current();
	void clear_current();
	bool is_current() const;

	virtual Matrix32 get_listener_transform() const;

	void set_visible_layers(uint32_t p_layers);
	uint32_t get_visible_layers() const;

	Vector<Plane> get_frustum() const;

	Listener2D();
	~Listener2D();

};

#endif
