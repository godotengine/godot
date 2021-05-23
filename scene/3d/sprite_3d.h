/*************************************************************************/
/*  sprite_3d.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef SPRITE_3D_H
#define SPRITE_3D_H

#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/sprite_frames.h"

class SpriteBase3D : public GeometryInstance3D {
	GDCLASS(SpriteBase3D, GeometryInstance3D);

	mutable Ref<TriangleMesh> triangle_mesh; //cached

public:
	enum DrawFlags {
		FLAG_TRANSPARENT,
		FLAG_SHADED,
		FLAG_DOUBLE_SIDED,
		FLAG_MAX

	};

	enum AlphaCutMode {
		ALPHA_CUT_DISABLED,
		ALPHA_CUT_DISCARD,
		ALPHA_CUT_OPAQUE_PREPASS
	};

private:
	bool color_dirty = true;
	Color color_accum;

	SpriteBase3D *parent_sprite = nullptr;
	List<SpriteBase3D *> children;
	List<SpriteBase3D *>::Element *pI = nullptr;

	bool centered = true;
	Point2 offset;

	bool hflip = false;
	bool vflip = false;

	Color modulate = Color(1, 1, 1, 1);
	float opacity = 1.0;

	Vector3::Axis axis = Vector3::AXIS_Z;
	float pixel_size = 0.01;
	AABB aabb;

	RID immediate;

	bool flags[FLAG_MAX];
	AlphaCutMode alpha_cut = ALPHA_CUT_DISABLED;
	StandardMaterial3D::BillboardMode billboard_mode = StandardMaterial3D::BILLBOARD_DISABLED;
	bool pending_update = false;
	void _im_update();

	void _propagate_color_changed();

protected:
	Color _get_color_accum();
	void _notification(int p_what);
	static void _bind_methods();
	virtual void _draw() = 0;
	_FORCE_INLINE_ void set_aabb(const AABB &p_aabb) { aabb = p_aabb; }
	_FORCE_INLINE_ RID &get_immediate() { return immediate; }
	void _queue_update();

public:
	void set_centered(bool p_center);
	bool is_centered() const;

	void set_offset(const Point2 &p_offset);
	Point2 get_offset() const;

	void set_flip_h(bool p_flip);
	bool is_flipped_h() const;

	void set_flip_v(bool p_flip);
	bool is_flipped_v() const;

	void set_region_enabled(bool p_region_enabled);
	bool is_region_enabled() const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_modulate(const Color &p_color);
	Color get_modulate() const;

	void set_opacity(float p_amount);
	float get_opacity() const;

	void set_pixel_size(float p_amount);
	float get_pixel_size() const;

	void set_axis(Vector3::Axis p_axis);
	Vector3::Axis get_axis() const;

	void set_draw_flag(DrawFlags p_flag, bool p_enable);
	bool get_draw_flag(DrawFlags p_flag) const;

	void set_alpha_cut_mode(AlphaCutMode p_mode);
	AlphaCutMode get_alpha_cut_mode() const;
	void set_billboard_mode(StandardMaterial3D::BillboardMode p_mode);
	StandardMaterial3D::BillboardMode get_billboard_mode() const;

	virtual Rect2 get_item_rect() const = 0;

	virtual AABB get_aabb() const override;
	virtual Vector<Face3> get_faces(uint32_t p_usage_flags) const override;
	Ref<TriangleMesh> generate_triangle_mesh() const;

	SpriteBase3D();
	~SpriteBase3D();
};

class Sprite3D : public SpriteBase3D {
	GDCLASS(Sprite3D, SpriteBase3D);
	Ref<Texture2D> texture;

	bool region_enabled;
	Rect2 region_rect;

	int frame;

	int vframes;
	int hframes;

	void _texture_changed();

protected:
	virtual void _draw() override;
	static void _bind_methods();

	virtual void _validate_property(PropertyInfo &property) const override;

public:
	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;

	void set_region_enabled(bool p_region_enabled);
	bool is_region_enabled() const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_frame(int p_frame);
	int get_frame() const;

	void set_frame_coords(const Vector2i &p_coord);
	Vector2i get_frame_coords() const;

	void set_vframes(int p_amount);
	int get_vframes() const;

	void set_hframes(int p_amount);
	int get_hframes() const;

	virtual Rect2 get_item_rect() const override;

	Sprite3D();
	//~Sprite3D();
};

class AnimatedSprite3D : public SpriteBase3D {
	GDCLASS(AnimatedSprite3D, SpriteBase3D);

	Ref<SpriteFrames> frames;
	bool playing = false;
	StringName animation = "default";
	int frame = 0;

	bool centered = true;

	float timeout = 0.0;

	bool hflip = true;
	bool vflip = true;

	Color modulate;

	void _res_changed();

	void _reset_timeout();
	void _set_playing(bool p_playing);
	bool _is_playing() const;

protected:
	virtual void _draw() override;
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const override;

public:
	void set_sprite_frames(const Ref<SpriteFrames> &p_frames);
	Ref<SpriteFrames> get_sprite_frames() const;

	void play(const StringName &p_animation = StringName());
	void stop();
	bool is_playing() const;

	void set_animation(const StringName &p_animation);
	StringName get_animation() const;

	void set_frame(int p_frame);
	int get_frame() const;

	virtual Rect2 get_item_rect() const override;

	TypedArray<String> get_configuration_warnings() const override;
	AnimatedSprite3D();
};

VARIANT_ENUM_CAST(SpriteBase3D::DrawFlags);
VARIANT_ENUM_CAST(SpriteBase3D::AlphaCutMode);
#endif // SPRITE_3D_H
