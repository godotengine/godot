/*************************************************************************/
/*  animation_track_editor_plugins.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ANIMATION_TRACK_EDITOR_PLUGINS_H
#define ANIMATION_TRACK_EDITOR_PLUGINS_H

#include "editor/animation_track_editor.h"

class AnimationTrackEditBool : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditBool, AnimationTrackEdit);
	Ref<Texture> icon_checked;
	Ref<Texture> icon_unchecked;

public:
	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);
};

class AnimationTrackEditColor : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditColor, AnimationTrackEdit);

public:
	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);
	virtual void draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right);
};

class AnimationTrackEditAudio : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditAudio, AnimationTrackEdit);

	ObjectID id;

	void _preview_changed(ObjectID p_which);

protected:
	static void _bind_methods();

public:
	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);

	void set_node(Object *p_object);

	AnimationTrackEditAudio();
};

class AnimationTrackEditSpriteFrame : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditSpriteFrame, AnimationTrackEdit);

	ObjectID id;
	bool is_coords;

public:
	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);

	void set_node(Object *p_object);
	void set_as_coords();

	AnimationTrackEditSpriteFrame() { is_coords = false; }
};

class AnimationTrackEditSubAnim : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditSubAnim, AnimationTrackEdit);

	ObjectID id;

public:
	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);

	void set_node(Object *p_object);
};

class AnimationTrackEditTypeAudio : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditTypeAudio, AnimationTrackEdit);

	void _preview_changed(ObjectID p_which);

	bool len_resizing;
	bool len_resizing_start;
	int len_resizing_index;
	float len_resizing_from_px;
	float len_resizing_rel;

protected:
	static void _bind_methods();

public:
	virtual void _gui_input(const Ref<InputEvent> &p_event);

	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);

	AnimationTrackEditTypeAudio();
};

class AnimationTrackEditTypeAnimation : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditTypeAnimation, AnimationTrackEdit);

	ObjectID id;

public:
	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);

	void set_node(Object *p_object);
	AnimationTrackEditTypeAnimation();
};

class AnimationTrackEditVolumeDB : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditVolumeDB, AnimationTrackEdit);

public:
	virtual void draw_bg(int p_clip_left, int p_clip_right);
	virtual void draw_fg(int p_clip_left, int p_clip_right);
	virtual int get_key_height() const;
	virtual void draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right);
};

class AnimationTrackEditDefaultPlugin : public AnimationTrackEditPlugin {
	GDCLASS(AnimationTrackEditDefaultPlugin, AnimationTrackEditPlugin);

public:
	virtual AnimationTrackEdit *create_value_track_edit(Object *p_object, Variant::Type p_type, const String &p_property, PropertyHint p_hint, const String &p_hint_string, int p_usage);
	virtual AnimationTrackEdit *create_audio_track_edit();
	virtual AnimationTrackEdit *create_animation_track_edit(Object *p_object);
};

#endif // ANIMATION_TRACK_EDITOR_PLUGINS_H
