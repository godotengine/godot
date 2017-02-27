#ifndef LINE2D_H
#define LINE2D_H

#include "node_2d.h"
#include "line_builder.h"


class Line2D : public Node2D {

	GDCLASS(Line2D, Node2D)

public:
	Line2D();

	void set_points(const PoolVector<Vector2> & p_points);
	PoolVector<Vector2> get_points() const;

	void set_point_pos(int i, Vector2 pos);
	Vector2 get_point_pos(int i) const;

	int get_point_count() const;

	void add_point(Vector2 pos);
	void remove_point(int i);

	void set_width(float width);
	float get_width() const;

	void set_default_color(Color color);
	Color get_default_color() const;

	void set_gradient(const Ref<ColorRamp>& gradient);
	Ref<ColorRamp> get_gradient() const;

	void set_texture(const Ref<Texture>& texture);
	Ref<Texture> get_texture() const;

	void set_texture_mode(const LineTextureMode mode);
	LineTextureMode get_texture_mode() const;

	void set_joint_mode(LineJointMode mode);
	LineJointMode get_joint_mode() const;

	void set_begin_cap_mode(LineCapMode mode);
	LineCapMode get_begin_cap_mode() const;

	void set_end_cap_mode(LineCapMode mode);
	LineCapMode get_end_cap_mode() const;

	void set_sharp_limit(float limit);
	float get_sharp_limit() const;

	void set_round_precision(int precision);
	int get_round_precision() const;

protected:
	void _notification(int p_what);
	void _draw();

	static void _bind_methods();

private:
	void _gradient_changed();

private:
	PoolVector<Vector2> _points;
	LineJointMode _joint_mode;
	LineCapMode _begin_cap_mode;
	LineCapMode _end_cap_mode;
	float _width;
	Color _default_color;
	Ref<ColorRamp> _gradient;
	Ref<Texture> _texture;
	LineTextureMode _texture_mode;
	float _sharp_limit;
	int _round_precision;

};

#endif // LINE2D_H
