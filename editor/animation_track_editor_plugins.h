/**************************************************************************/
/*  animation_track_editor_plugins.h                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "editor/animation_preview.h"
#include "editor/animation_track_editor.h"
#include "editor/audio_stream_preview.h"

#define REGION_RESIZE_THRESHOLD 5.0
#define REGION_MAX_WIDTH 4.0
#define REGION_FONT_MARGIN 3.0
#define REGION_BG_COLOR Color(0.25, 0.25, 0.25)
#define REGION_EDGE_ALPHA 0.7

#define COLOR_EDIT_SAMPLE_INTERVAL 64
#define COLOR_EDIT_RECT_INTERVAL 2

struct Region {
	union {
		struct {
			real_t x;
			real_t width;
		};

		struct {
			real_t y;
			real_t height;
		};
	};

	Region(real_t p_x = 0.0, real_t p_width = 0.0) :
			x(p_x), width(p_width) {}
	real_t get_end() const { return x + width; }
	bool has_point(real_t p_point) const { return p_point >= x && p_point <= get_end(); }
};

class AnimationTrackEditClip : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditClip, AnimationTrackEdit);

private:
	ObjectID id;

	bool len_resizing = false;
	bool len_resizing_start = false;
	int len_resizing_index = 0;
	float len_resizing_from_px = 0.0f;
	float len_resizing_rel = 0.0f;
	bool over_drag_position = false;

	int handle_track_resizing(const Ref<InputEventMouseMotion> mm, const int p_index, const Rect2 p_global_rect, const int p_clip_left, const int p_clip_right);

	Region _calc_key_region(const int p_index, const float start_ofs, const float end_ofs, const float len, float __offset = 0) const;
	Region _clip_key_region(const Region &region, const int p_clip_left, const int p_clip_right);
	Region _calc_key_region_shift(const Region &orig_region, const Region &region) const;
	bool _is_key_region_outside(const Region &region, const int p_clip_left, const int p_clip_right) const;

public:
	void set_node(Object *p_object);
	ObjectID get_node_id() const { return id; }

	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void _preview_changed(ObjectID p_which);

public:
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

protected:
	virtual Ref<Resource> get_resource(const int p_index) const { return nullptr; } // resource of the key (AudioStream, Animation, ...)
	virtual float get_start_offset(const int p_index) const { return 0; } // start offset of the key
	virtual float get_end_offset(const int p_index) const { return 0; } // end offset of the key
	virtual float get_length(const int p_index) const { return 0; } // total length of the key
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) {} //sets the start offset of the key
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) {} //sets the end offset of the key
	virtual StringName get_edit_name(const int p_index) const; //name of the key

protected:
	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) {}
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {} // data to visualize the key if the key has a length
};

class AnimationTrackEditTypeAudio : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditTypeAudio, AnimationTrackEditClip);

	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual bool has_valid_key(const int p_index) const override;

protected:
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_start_offset(const int p_index) const override;
	virtual float get_end_offset(const int p_index) const override;
	virtual float get_length(const int p_index) const override;
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) override;

public:
	AnimationTrackEditTypeAudio();
};

class AnimationTrackEditTypeAnimation : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditTypeAnimation, AnimationTrackEditClip);

	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual bool has_valid_key(const int p_index) const override;

protected:
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_start_offset(const int p_index) const override;
	virtual float get_end_offset(const int p_index) const override;
	virtual float get_length(const int p_index) const override;
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual StringName get_edit_name(const int p_index) const override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) override;

public:
	AnimationTrackEditTypeAnimation();
};

class AnimationTrackEditAudio : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditAudio, AnimationTrackEditClip);

	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual bool has_valid_key(const int p_index) const override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_length(const int p_index) const override;

public:
	AnimationTrackEditAudio();
};

class AnimationTrackEditSubAnim : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditSubAnim, AnimationTrackEditClip);

	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual bool has_valid_key(const int p_index) const override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;

	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_length(const int p_index) const override;
	virtual StringName get_edit_name(const int p_index) const override;

public:
	AnimationTrackEditSubAnim();
};

class AnimationTrackEditBool : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditBool, AnimationTrackEdit);

private:
	Ref<Texture2D> icon_checked;
	Ref<Texture2D> icon_unchecked;

public:
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

public:
	AnimationTrackEditBool();
};

class AnimationTrackEditTypeMethod : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditTypeMethod, AnimationTrackEdit);

public:
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

public:
	StringName get_edit_name(const int p_index) const; //name of the key

protected:
	virtual void draw_key_link(int p_index, float p_pixels_sec, float p_x, float p_next_x, float p_clip_left, float p_clip_right) override;

private:
	// Helper
	String _make_method_text(const Dictionary &d) const;

public:
	AnimationTrackEditTypeMethod();
};

class AnimationTrackEditColor : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditColor, AnimationTrackEdit);

public:
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

protected:
	virtual void draw_key_link(int p_index, float p_pixels_sec, float p_x, float p_next_x, float p_clip_left, float p_clip_right) override;

public:
	AnimationTrackEditColor();
};

class AnimationTrackEditSpriteFrame : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditSpriteFrame, AnimationTrackEdit);

private:
	ObjectID id;
	bool is_coords = false;

	//Helper
	Rect2 _create_texture_region_sprite(int p_index, Object *object, const Ref<Texture2D> texture) const;
	Rect2 _create_region_animated_sprite(int p_index, Object *object, const Ref<Texture2D> texture) const;

public:
	void set_node(Object *p_object);
	ObjectID get_node_id() const { return id; }

public:
	void set_as_coords();

public:
	virtual bool has_valid_key(const int p_index) const override;

public:
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

protected:
	Ref<Resource> get_resource(const int p_index) const;

public:
	AnimationTrackEditSpriteFrame();
};

class AnimationTrackEditVolumeDB : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditVolumeDB, AnimationTrackEdit);

public:
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

protected:
	virtual void draw_bg(const float p_clip_left, const float p_clip_right) override;
	virtual void draw_fg(const float p_clip_left, const float p_clip_right) override;
	virtual void draw_key_link(int p_index, float p_pixels_sec, float p_x, float p_next_x, float p_clip_left, float p_clip_right) override;

public:
	AnimationTrackEditVolumeDB();
};

class AnimationTrackEditDefaultPlugin : public AnimationTrackEditPlugin {
	GDCLASS(AnimationTrackEditDefaultPlugin, AnimationTrackEditPlugin);

public:
	virtual AnimationTrackEdit *create_value_track_edit(Object *p_object, Variant::Type p_type, const String &p_property, PropertyHint p_hint, const String &p_hint_string, int p_usage) override;
	virtual AnimationTrackEdit *create_audio_track_edit() override;
	virtual AnimationTrackEdit *create_animation_track_edit(Object *p_object) override;
	virtual AnimationTrackEdit *create_method_track_edit() override;
};
