#ifndef ROOT_MOTION_VIEW_H
#define ROOT_MOTION_VIEW_H

#include "scene/3d/visual_instance.h"

class RootMotionView : public VisualInstance {
	GDCLASS(RootMotionView, VisualInstance)
public:
	RID immediate;
	NodePath path;
	float cell_size;
	float radius;
	bool use_in_game;
	Color color;
	bool first;
	bool zero_y;

	Transform accumulated;

private:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_animation_path(const NodePath &p_path);
	NodePath get_animation_path() const;

	void set_color(const Color &p_path);
	Color get_color() const;

	void set_cell_size(float p_size);
	float get_cell_size() const;

	void set_radius(float p_radius);
	float get_radius() const;

	void set_zero_y(bool p_zero_y);
	bool get_zero_y() const;

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	RootMotionView();
	~RootMotionView();
};

#endif // ROOT_MOTION_VIEW_H
