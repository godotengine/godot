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

// Constants for region rendering and color editing
#define REGION_RESIZE_THRESHOLD 5.0
#define REGION_MAX_WIDTH 4.0
#define REGION_FONT_MARGIN 3.0
#define REGION_BG_COLOR Color(0.25, 0.25, 0.25)
#define REGION_EDGE_ALPHA 0.7
#define COLOR_EDIT_SAMPLE_INTERVAL 64
#define COLOR_EDIT_RECT_INTERVAL 2

// Struct for defining key regions in tracks
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

// Base class for clip-based animation track editing
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

	// Region handling methods
	int handle_track_resizing(const Ref<InputEventMouseMotion> mm, const int p_index, const int p_clip_left, const int p_clip_right);
	Region _calc_key_region(const int p_index, const float p_start_ofs, const float p_end_ofs, const float p_len, float p_offset = 0) const;
	Region _clip_key_region(const Region &p_region, const int p_clip_left, const int p_clip_right);
	Region _calc_key_region_shift(const Region &p_orig_region, const Region &p_region) const;
	bool _is_key_region_outside(const Region &p_region, const int p_clip_left, const int p_clip_right) const;

public:
	// Node management
	void set_node(Object *p_object);
	ObjectID get_node_id() const { return id; }

	// Input and preview handling
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void _preview_changed(ObjectID p_which);

	// Key dimension overrides
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;

protected:
	// Drawing and resource management
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;
	virtual Ref<Resource> get_resource(const int p_index) const { return nullptr; }
	virtual float get_start_offset(const int p_index) const { return 0; }
	virtual float get_end_offset(const int p_index) const { return 0; }
	virtual float get_length(const int p_index) const { return 0; }
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) {}
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) {}
	virtual StringName get_edit_name(const int p_index) const override;
	virtual float get_key_scale(const int p_index) const { return 1.0; }

	// Input handling overrides
	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) {}
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {}

public:
	AnimationTrackEditClip();
};

// Class for editing audio clip tracks
class AnimationTrackEditTypeAudio : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditTypeAudio, AnimationTrackEditClip);

protected:
	// Resource and preview handling
	virtual void _preview_changed(ObjectID p_which) override;
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_start_offset(const int p_index) const override;
	virtual float get_end_offset(const int p_index) const override;
	virtual float get_length(const int p_index) const override;
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) override;

public:
	// Validation and tooltip
	virtual bool has_valid_key(const int p_index) const override;
	virtual String _get_tooltip(const int p_index) const override;

public:
	AnimationTrackEditTypeAudio();
};

// Class for editing animation clip tracks
class AnimationTrackEditTypeAnimation : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditTypeAnimation, AnimationTrackEditClip);

protected:
	// Resource and preview handling
	virtual void _preview_changed(ObjectID p_which) override;
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_start_offset(const int p_index) const override;
	virtual float get_end_offset(const int p_index) const override;
	virtual float get_length(const int p_index) const override;
	virtual void set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual void set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) override;
	virtual StringName get_edit_name(const int p_index) const override;
	virtual float get_key_scale(const int p_index) const override;
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual void apply_data(const Ref<Resource> resource, const float time) override;

public:
	// Validation and tooltip
	virtual bool has_valid_key(const int p_index) const override;
	virtual String _get_tooltip(const int p_index) const override;

public:
	AnimationTrackEditTypeAnimation();
};

// Class for editing audio tracks
class AnimationTrackEditAudio : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditAudio, AnimationTrackEditClip);

protected:
	// Resource and preview handling
	virtual void _preview_changed(ObjectID p_which) override;
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_length(const int p_index) const override;

public:
	// Validation
	virtual bool has_valid_key(const int p_index) const override;

public:
	AnimationTrackEditAudio();
};

// Class for editing sub-animation tracks
class AnimationTrackEditSubAnim : public AnimationTrackEditClip {
	GDCLASS(AnimationTrackEditSubAnim, AnimationTrackEditClip);

protected:
	// Resource and preview handling
	virtual void _preview_changed(ObjectID p_which) override;
	virtual void get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) override;
	virtual Ref<Resource> get_resource(const int p_index) const override;
	virtual float get_length(const int p_index) const override;
	virtual StringName get_edit_name(const int p_index) const override;

public:
	// Validation
	virtual bool has_valid_key(const int p_index) const override;

public:
	AnimationTrackEditSubAnim();
};

// Class for editing boolean tracks
class AnimationTrackEditBool : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditBool, AnimationTrackEdit);

private:
	Ref<Texture2D> icon_checked;
	Ref<Texture2D> icon_unchecked;

public:
	// Key dimension overrides
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;

protected:
	// Drawing
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;

public:
	AnimationTrackEditBool();
};

// Class for editing method tracks
class AnimationTrackEditTypeMethod : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditTypeMethod, AnimationTrackEdit);

protected:
	// Drawing and tooltip
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;
	virtual void draw_key_link(const int p_index, const Rect2 &p_global_rect, const Rect2 &p_global_rect_next, const float p_clip_left, const float p_clip_right) override;
	String _make_method_text(const Dictionary &d) const;

public:
	// Key dimension and name overrides
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual StringName get_edit_name(const int p_index) const override;
	virtual String _get_tooltip(const int p_index) const override;

public:
	AnimationTrackEditTypeMethod();
};

// Class for editing color tracks
class AnimationTrackEditColor : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditColor, AnimationTrackEdit);

public:
	// Key dimension and link overrides
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual bool is_linked(const int p_index, const int p_index_next) const override;

protected:
	// Drawing
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;
	virtual void draw_key_link(const int p_index, const Rect2 &p_global_rect, const Rect2 &p_global_rect_next, const float p_clip_left, const float p_clip_right) override;

public:
	AnimationTrackEditColor();
};

// Class for editing sprite frame tracks
class AnimationTrackEditSpriteFrame : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditSpriteFrame, AnimationTrackEdit);

private:
	ObjectID id;
	bool is_coords = false;

	// Helper methods for texture regions
	Rect2 _create_texture_region_sprite(int p_index, Object *object, const Ref<Texture2D> texture) const;
	Rect2 _create_region_animated_sprite(int p_index, Object *object, const Ref<Texture2D> texture) const;

public:
	// Node management
	void set_node(Object *p_object);
	ObjectID get_node_id() const { return id; }
	void set_as_coords();

	// Validation and resource
	virtual bool has_valid_key(const int p_index) const override;
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;

protected:
	// Drawing and resource
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;
	Ref<Resource> get_resource(const int p_index) const;

public:
	AnimationTrackEditSpriteFrame();
};

// Class for editing volume dB tracks
class AnimationTrackEditVolumeDB : public AnimationTrackEdit {
	GDCLASS(AnimationTrackEditVolumeDB, AnimationTrackEdit);

public:
	// Key dimension and link overrides
	virtual float get_key_width(const int p_index) const override;
	virtual float get_key_height(const int p_index) const override;
	virtual bool is_linked(const int p_index, const int p_index_next) const override;
	virtual float get_key_y(const int p_index) const override;

protected:
	// Drawing
	virtual void draw_bg(const float p_clip_left, const float p_clip_right) override;
	virtual void draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) override;
	virtual void draw_key_link(const int p_index, const Rect2 &p_global_rect, const Rect2 &p_global_rect_next, const float p_clip_left, const float p_clip_right) override;
	virtual void draw_fg(const float p_clip_left, const float p_clip_right) override;

public:
	AnimationTrackEditVolumeDB();
};

// Default plugin for creating track edit controls
class AnimationTrackEditDefaultPlugin : public AnimationTrackEditPlugin {
	GDCLASS(AnimationTrackEditDefaultPlugin, AnimationTrackEditPlugin);

public:
	// Track creation methods
	virtual AnimationTrackEdit *create_value_track_edit(Object *p_object, Variant::Type p_type, const String &p_property, PropertyHint p_hint, const String &p_hint_string, int p_usage) override;
	virtual AnimationTrackEdit *create_audio_track_edit() override;
	virtual AnimationTrackEdit *create_animation_track_edit(Object *p_object) override;
	virtual AnimationTrackEdit *create_method_track_edit() override;
};
