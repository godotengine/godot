#ifndef LIGHT_2D_H
#define LIGHT_2D_H

#include "scene/2d/node_2d.h"

class Light2D : public Node2D {

	OBJ_TYPE(Light2D,Node2D);
public:

	enum LightBlendMode {
		LIGHT_BLEND_ADD,
		LIGHT_BLEND_SUB,
		LIGHT_BLEND_MULTIPLY,
		LIGHT_BLEND_DODGE,
		LIGHT_BLEND_BURN,
		LIGHT_BLEND_LIGHTEN,
		LIGHT_BLEND_DARKEN,
		LIGHT_BLEND_OVERLAY,
		LIGHT_BLEND_SCREEN,
	};

private:
	RID canvas_light;
	bool enabled;
	bool shadow;
	Color color;
	float height;
	int z_min;
	int z_max;
	int item_mask;
	LightBlendMode blend_mode;
	Ref<Texture> texture;
	Vector2 texture_offset;

protected:

	static void _bind_methods();
public:


	void set_enabled( bool p_enabled);
	bool is_enabled() const;

	void set_texture( const Ref<Texture>& p_texture);
	Ref<Texture> get_texture() const;

	void set_texture_offset( const Vector2& p_offset);
	Vector2 get_texture_offset() const;

	void set_color( const Color& p_color);
	Color get_color() const;

	void set_height( float p_height);
	float get_height() const;

	void set_z_range_min( int p_min_z);
	int get_z_range_min() const;

	void set_z_range_max( int p_max_z);
	int get_z_range_max() const;

	void set_item_mask( int p_mask);
	int get_item_mask() const;

	void set_blend_mode( LightBlendMode p_blend_mode );
	LightBlendMode get_blend_mode() const;

	void set_shadow_enabled( bool p_enabled);
	bool is_shadow_enabled() const;


	Light2D();
	~Light2D();
};


VARIANT_ENUM_CAST(Light2D::LightBlendMode);

#endif // LIGHT_2D_H
