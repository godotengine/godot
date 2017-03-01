#ifndef LINE_BUILDER_H
#define LINE_BUILDER_H

#include "math_2d.h"
#include "color.h"
#include "scene/resources/color_ramp.h"

enum LineJointMode {
	LINE_JOINT_SHARP = 0,
	LINE_JOINT_BEVEL,
	LINE_JOINT_ROUND
};

enum LineCapMode {
	LINE_CAP_NONE = 0,
	LINE_CAP_BOX,
	LINE_CAP_ROUND
};

enum LineTextureMode {
	LINE_TEXTURE_NONE = 0,
	LINE_TEXTURE_TILE
	// TODO STRETCH mode
};

class LineBuilder {
public:
	// TODO Move in a struct and reference it
	// Input
	Vector<Vector2> points;
	LineJointMode joint_mode;
	LineCapMode begin_cap_mode;
	LineCapMode end_cap_mode;
	float width;
	Color default_color;
	ColorRamp* gradient;
	LineTextureMode texture_mode;
	float sharp_limit;
	int round_precision;
	// TODO offset_joints option (offers alternative implementation of round joints)

	// TODO Move in a struct and reference it
	// Output
	Vector<Vector2> vertices;
	Vector<Color> colors;
	Vector<Vector2> uvs;
	Vector<int> indices;

	LineBuilder();

	void build();
	void clear_output();

private:
	enum Orientation {
		UP = 0,
		DOWN = 1
	};

	// Triangle-strip methods
	void strip_begin(Vector2 up, Vector2 down, Color color, float uvx);
	void strip_new_quad(Vector2 up, Vector2 down, Color color, float uvx);
	void strip_add_quad(Vector2 up, Vector2 down, Color color, float uvx);
	void strip_add_tri(Vector2 up, Orientation orientation);
	void strip_add_arc(Vector2 center, float angle_delta, Orientation orientation);

	void new_arc(Vector2 center, Vector2 vbegin, float angle_delta, Color color, Rect2 uv_rect);

private:
	bool _interpolate_color;
	int _last_index[2]; // Index of last up and down vertices of the strip

};


#endif // LINE_BUILDER_H
