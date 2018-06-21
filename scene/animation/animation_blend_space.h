#ifndef ANIMATION_BLEND_SPACE_H
#define ANIMATION_BLEND_SPACE_H

#include "scene/animation/animation_graph_player.h"

class AnimationNodeBlendSpace : public AnimationRootNode {
	GDCLASS(AnimationNodeBlendSpace, AnimationRootNode)

	enum {
		MAX_BLEND_POINTS = 64
	};

	struct BlendPoint {
		Ref<AnimationRootNode> node;
		Vector2 position;
	};

	BlendPoint blend_points[MAX_BLEND_POINTS];
	int blend_points_used;

	struct BlendTriangle {
		int points[3];
	};

	Vector<BlendTriangle> triangles;

	Vector2 blend_pos;
	Vector2 max_space;
	Vector2 min_space;
	Vector2 snap;
	String x_label;
	String y_label;

	void _add_blend_point(int p_index, const Ref<AnimationRootNode> &p_node);
	void _set_triangles(const Vector<int> &p_triangles);
	Vector<int> _get_triangles() const;

	void _blend_triangle(const Vector2 &p_pos, const Vector2 *p_points, float *r_weights);

protected:
	virtual void _validate_property(PropertyInfo &property) const;
	static void _bind_methods();

public:
	void add_blend_point(const Ref<AnimationRootNode> &p_node, const Vector2 &p_position, int p_at_index = -1);
	void set_blend_point_position(int p_point, const Vector2 &p_position);
	void set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node);
	Vector2 get_blend_point_position(int p_point) const;
	Ref<AnimationRootNode> get_blend_point_node(int p_point) const;
	void remove_blend_point(int p_point);
	int get_blend_point_count() const;

	bool has_triangle(int p_x, int p_y, int p_z) const;
	void add_triangle(int p_x, int p_y, int p_z, int p_at_index = -1);
	int get_triangle_point(int p_triangle, int p_point);
	void remove_triangle(int p_triangle);
	int get_triangle_count() const;

	void set_min_space(const Vector2 &p_min);
	Vector2 get_min_space() const;

	void set_max_space(const Vector2 &p_max);
	Vector2 get_max_space() const;

	void set_snap(const Vector2 &p_snap);
	Vector2 get_snap() const;

	void set_blend_pos(const Vector2 &p_pos);
	Vector2 get_blend_pos() const;

	void set_x_label(const String &p_label);
	String get_x_label() const;

	void set_y_label(const String &p_label);
	String get_y_label() const;

	float process(float p_time, bool p_seek);
	String get_caption() const;

	Vector2 get_closest_point(const Vector2 &p_point);

	AnimationNodeBlendSpace();
	~AnimationNodeBlendSpace();
};

#endif // ANIMATION_BLEND_SPACE_H
