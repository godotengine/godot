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

class AnimationTrackEditBase : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditBase, AnimationTrackEdit);

protected:
	ObjectID id;

protected:
	virtual void _preview_changed(ObjectID p_which) = 0;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) = 0;
	virtual int get_key_height() const = 0;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) = 0;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) = 0;
	virtual bool is_key_selectable_by_distance() const = 0;

protected:
	Rect2 get_key_rect_region(const float start_ofs, const float end_ofs, const float len, const int p_index, const float p_pixels_sec);
	virtual void draw_key_region(Ref<Resource> resource, float start_ofs, float end_ofs, float len, int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right); //B

public:
	void set_node(Object *p_object);
};

class AnimationTrackEditClip : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditClip, AnimationTrackEditBase);

protected:
	bool len_resizing = false;
	bool len_resizing_start = false;
	int len_resizing_index = 0;
	float len_resizing_from_px = 0.0f;
	float len_resizing_rel = 0.0f;
	bool over_drag_position = false;

	virtual void draw_key_region(Ref<Resource> resource, float start_ofs, float end_ofs, float len, int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;

public:
	virtual void gui_input(const Ref<InputEvent> &p_event) = 0;

protected:
	virtual void handle_data(const float ofs, const Ref<Resource> resource) = 0;

protected:
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);
	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const;

	bool handle_track_over(Ref<InputEventMouseMotion> mm, float start_ofs, float end_ofs, float len, int i);
};

class AnimationTrackEditTypeAudio : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditTypeAudio, AnimationTrackEditClip);

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override; //
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override; //
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void handle_data(const float ofs, const Ref<Resource> resource) override;

public:
	AnimationTrackEditTypeAudio();
};

class AnimationTrackEditBool : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditBool, AnimationTrackEditBase);
	Ref<Texture2D> icon_checked;
	Ref<Texture2D> icon_unchecked;

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

public:
	AnimationTrackEditBool();
};

class AnimationTrackEditColor : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditColor, AnimationTrackEditBase);

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

protected:
	virtual void draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right) override;

public:
	AnimationTrackEditColor();
};

class AnimationTrackEditAudio : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditAudio, AnimationTrackEditBase);

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

protected:
	virtual void draw_key_region(Ref<Resource> resource, float start_ofs, float end_ofs, float len, int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override; //C

public:
	AnimationTrackEditAudio();
};

class AnimationTrackEditSpriteFrame : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditSpriteFrame, AnimationTrackEdit);

private:
	bool is_coords = false;

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

public:
	void set_as_coords();

public:
	AnimationTrackEditSpriteFrame();
};

class AnimationTrackEditSubAnim : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditSubAnim, AnimationTrackEditBase);

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

protected:
	virtual void draw_key_region(Ref<Resource> resource, float start_ofs, float end_ofs, float len, int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override; //C

public:
	AnimationTrackEditSubAnim();
};

class AnimationTrackEditTypeAnimation : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditTypeAnimation, AnimationTrackEditClip);

	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void handle_data(const float ofs, const Ref<Resource> resource) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

public:
	AnimationTrackEditTypeAnimation();
};

class AnimationTrackEditVolumeDB : public AnimationTrackEditBase {
	GDCLASS(AnimationTrackEditVolumeDB, AnimationTrackEdit);

protected:
	virtual void _preview_changed(ObjectID p_which) override;

public:
	virtual void create_key_region(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual int get_key_height() const override;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec) override;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) override;
	virtual bool is_key_selectable_by_distance() const override;

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
