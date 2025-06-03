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

class AnimationTrackEditKey : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditKey, AnimationTrackEdit);

private:
	ObjectID id;

	bool len_resizing = false;
	bool len_resizing_start = false;
	int len_resizing_index = 0;
	float len_resizing_from_px = 0.0f;
	float len_resizing_rel = 0.0f;
	bool over_drag_position = false;

	int handle_track_resizing(const Ref<InputEventMouseMotion> mm, const float start_ofs, const float end_ofs, const float len, const int p_index, const float p_pixels_sec, const int p_x, const int p_clip_left, const int p_clip_right);

	Vector2 calc_key_region(const float start_ofs, const float end_ofs, const float len, const int p_index, const float p_pixels_sec, const int p_x);
	Vector2 clip_key_region(Vector2 region, int p_clip_left, int p_clip_right);
	void draw_key_region(Ref<Resource> resource, float start_ofs, float end_ofs, float len, int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);
	Vector2 calc_key_region_shift(Vector2 &orig_region, Vector2 &region);
	bool is_key_region_outside(const Vector2 &region, int p_clip_left, int p_clip_right);

protected:
	float key_scale = 1.0;

public:
	void set_node(Object *p_object);
	ObjectID get_node_id() const { return id; }

	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void _preview_changed(ObjectID p_which);

public:
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual bool is_key_selectable_by_distance() const override { return false; }
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;

protected:
	virtual Ref<Resource> get_resource(const int p_index) { return nullptr; } // resource of the key (AudioStream, Animation, ...)
	virtual float get_start_offset(const int p_index) { return 0; } // start offset of the key
	virtual float get_end_offset(const int p_index) { return 0; } // end offset of the key
	virtual float get_length(const int p_index) { return 0; } // total length of the key
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) {} //sets the start offset of the key
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) {} //sets the end offset of the key
	virtual StringName get_edit_name(const int p_index); //name of the key

protected:
	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) {}
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {} // data to visualize the key if the key has a length
};

class AnimationTrackEditTypeAudio : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditTypeAudio, AnimationTrackEditKey);

	virtual void _preview_changed(ObjectID p_which) override;

protected:
	virtual Ref<Resource> get_resource(const int p_index) override;
	virtual float get_start_offset(const int p_index) override;
	virtual float get_end_offset(const int p_index) override;
	virtual float get_length(const int p_index) override;
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) override;

public:
	AnimationTrackEditTypeAudio();
};

class AnimationTrackEditTypeAnimation : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditTypeAnimation, AnimationTrackEditKey);

	virtual void _preview_changed(ObjectID p_which) override;

protected:
	virtual Ref<Resource> get_resource(const int p_index) override;
	virtual float get_start_offset(const int p_index) override;
	virtual float get_end_offset(const int p_index) override;
	virtual float get_length(const int p_index) override;
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual StringName get_edit_name(const int p_index) override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) override;

public:
	AnimationTrackEditTypeAnimation();
};

class AnimationTrackEditAudio : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditAudio, AnimationTrackEditKey);

	virtual void _preview_changed(ObjectID p_which) override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual Ref<Resource> get_resource(const int p_index) override;
	virtual float get_length(const int p_index) override;

public:
	AnimationTrackEditAudio();
};

class AnimationTrackEditSubAnim : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditSubAnim, AnimationTrackEditKey);

	virtual void _preview_changed(ObjectID p_which) override;

protected:
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;

	virtual Ref<Resource> get_resource(const int p_index) override;
	virtual float get_length(const int p_index) override;
	virtual StringName get_edit_name(const int p_index) override;

public:
	AnimationTrackEditSubAnim();
};

class AnimationTrackEditBool : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditBool, AnimationTrackEditKey);

private:
	Ref<Texture2D> icon_checked;
	Ref<Texture2D> icon_unchecked;

public:
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;

public:
	AnimationTrackEditBool();
};

class AnimationTrackEditColor : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditColor, AnimationTrackEditKey);

public:
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;

protected:
	virtual void draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right) override;

public:
	AnimationTrackEditColor();
};

class AnimationTrackEditSpriteFrame : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditSpriteFrame, AnimationTrackEditKey);

private:
	bool is_coords = false;

public:
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;

public:
	void set_as_coords();

public:
	AnimationTrackEditSpriteFrame();
};

class AnimationTrackEditVolumeDB : public AnimationTrackEditKey {
	GDCLASS(AnimationTrackEditVolumeDB, AnimationTrackEditKey);

public:
	virtual int get_key_height() const override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;

protected:
	virtual void draw_bg(int p_clip_left, int p_clip_right) override;
	virtual void draw_fg(int p_clip_left, int p_clip_right) override;
	virtual void draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right) override;

public:
	AnimationTrackEditVolumeDB();
};

class AnimationTrackEditDefaultPlugin : public AnimationTrackEditPlugin {
	GDCLASS(AnimationTrackEditDefaultPlugin, AnimationTrackEditPlugin);

public:
	virtual AnimationTrackEdit *create_value_track_edit(Object *p_object, Variant::Type p_type, const String &p_property, PropertyHint p_hint, const String &p_hint_string, int p_usage) override;
	virtual AnimationTrackEdit *create_audio_track_edit() override;
	virtual AnimationTrackEdit *create_animation_track_edit(Object *p_object) override;
};
