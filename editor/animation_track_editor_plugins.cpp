/**************************************************************************/
/*  animation_track_editor_plugins.cpp                                    */
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

#include "animation_track_editor_plugins.h"

#include "editor/animation_preview.h"
#include "editor/audio_stream_preview.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/themes/editor_scale.h"
#include "scene/2d/animated_sprite_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/3d/sprite_3d.h"
#include "scene/animation/animation_player.h"
#include "servers/audio/audio_stream.h"

/// BOOL ///

AnimationTrackEditBool::AnimationTrackEditBool() {
	key_pivot.x = 0.5;
	key_pivot.y = 0.5;
}

float AnimationTrackEditBool::get_key_width(const int p_index) const {
	Ref<Texture2D> texture = get_theme_icon(SNAME("checked"), SNAME("CheckBox"));
	return texture->get_width();
}

float AnimationTrackEditBool::get_key_height(const int p_index) const {
	Ref<Texture2D> texture = get_theme_icon(SNAME("checked"), SNAME("CheckBox"));
	return texture->get_height();
}

void AnimationTrackEditBool::draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) {
	if (p_global_rect.size.is_zero_approx()) {
		AnimationTrackEdit::draw_key(p_index, p_global_rect, p_selected, p_clip_left, p_clip_right);
		return;
	}

	bool checked = get_animation()->track_get_key_value(get_track(), p_index);
	Ref<Texture2D> texture = get_theme_icon(checked ? "checked" : "unchecked", "CheckBox");

	Rect2 region;
	region.size = texture->get_size();
	editor->_draw_texture_region_clipped(this, texture, p_global_rect, region, p_clip_left, p_clip_right);
}

/// METHOD ///

AnimationTrackEditTypeMethod::AnimationTrackEditTypeMethod() {
	key_pivot.x = 0.5;
	key_pivot.y = 0.5;
}

float AnimationTrackEditTypeMethod::get_key_width(const int p_index) const {
	return AnimationTrackEdit::get_key_width(p_index);
}

float AnimationTrackEditTypeMethod::get_key_height(const int p_index) const {
	return AnimationTrackEdit::get_key_height(p_index);
}

void AnimationTrackEditTypeMethod::draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) {
	float clip_r = p_clip_right - REGION_FONT_MARGIN;
	if (get_animation()->track_get_key_count(get_track()) > p_index + 1) {
		Rect2 rect_next = get_global_key_rect(p_index + 1);
		clip_r = MIN(rect_next.position.x - REGION_FONT_MARGIN, clip_r);
	}

	float text_pos = p_global_rect.position.x + p_global_rect.size.width + REGION_FONT_MARGIN;
	float max_width = MAX(0.0, clip_r - text_pos);
	if (max_width > 0) {
		const Ref<Font> font = get_theme_font(SceneStringName(font), SNAME("Label"));
		const int font_size = get_theme_font_size(SceneStringName(font_size), SNAME("Label"));
		Color color = get_theme_color(SceneStringName(font_color), SNAME("Label"));
		color.a = 0.5;

		String method_name = get_edit_name(p_index);
		String edit_name = editor->_make_text_clipped(method_name, font, font_size, max_width);

		int f_h = int(get_size().height - font->get_height(font_size)) / 2 + font->get_ascent(font_size);
		draw_string(font, Vector2(text_pos, f_h), edit_name, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color);
	}

	AnimationTrackEdit::draw_key(p_index, p_global_rect, p_selected, p_clip_left, p_clip_right);
}

void AnimationTrackEditTypeMethod::draw_key_link(const int p_index, const Rect2 &p_global_rect, const Rect2 &p_global_rect_next, const float p_clip_left, const float p_clip_right) {
}

String AnimationTrackEditTypeMethod::_get_tooltip(const int p_index) const {
	String text = "";

	Dictionary d = get_key_value(p_index);
	text += _make_method_text(d) + "\n";

	return text;
}

StringName AnimationTrackEditTypeMethod::get_edit_name(const int p_index) const {
	Dictionary d = animation->track_get_key_value(get_track(), p_index);
	String method_name = _make_method_text(d);
	return method_name;
}

String AnimationTrackEditTypeMethod::_make_method_text(const Dictionary &d) const {
	String text;

	if (d.has("method")) {
		text += String(d["method"]);
	}
	text += "(";
	Vector<Variant> args;
	if (d.has("args")) {
		args = d["args"];
	}
	for (int i = 0; i < args.size(); i++) {
		if (i > 0) {
			text += ", ";
		}
		text += args[i].get_construct_string();
	}
	text += ")";

	return text;
}

/// COLOR ///

AnimationTrackEditColor::AnimationTrackEditColor() {
	key_pivot.x = 0.5;
	key_pivot.y = 0.5;
}

float AnimationTrackEditColor::get_key_width(const int p_index) const {
	return _get_theme_font_height(0.8);
}

float AnimationTrackEditColor::get_key_height(const int p_index) const {
	return _get_theme_font_height(0.8);
}

void AnimationTrackEditColor::draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) {
	Color color = get_key_value(p_index);
	editor->_draw_grid_clipped(this, p_global_rect, color, COLOR_EDIT_RECT_INTERVAL, p_clip_left, p_clip_right);
}

void AnimationTrackEditColor::draw_key_link(const int p_index, const Rect2 &p_global_rect, const Rect2 &p_global_rect_next, const float p_clip_left, const float p_clip_right) {
	int fh = get_key_height(p_index);

	fh /= 3;

	float x_from = p_global_rect.position.x + fh / 2.0 - 1;
	float x_to = p_global_rect_next.position.x - fh / 2.0 + 1;
	x_from = MAX(x_from, p_clip_left);
	x_to = MIN(x_to, p_clip_right);

	float y_from = (get_size().height - fh) / 2.0;

	if (x_from > p_clip_right || x_to < p_clip_left) {
		return;
	}

	Vector<Color> color_samples;
	color_samples.append(get_animation()->track_get_key_value(get_track(), p_index));

	if (get_animation()->track_get_type(get_track()) == Animation::TYPE_VALUE) {
		if (get_animation()->track_get_interpolation_type(get_track()) != Animation::INTERPOLATION_NEAREST &&
				(get_animation()->value_track_get_update_mode(get_track()) == Animation::UPDATE_CONTINUOUS ||
						get_animation()->value_track_get_update_mode(get_track()) == Animation::UPDATE_CAPTURE) &&
				!Math::is_zero_approx(get_animation()->track_get_key_transition(get_track(), p_index))) {
			float start_time = get_key_time(p_index);
			float end_time = get_key_time(p_index + 1);

			Color color_next = get_animation()->value_track_interpolate(get_track(), end_time);

			if (!color_samples[0].is_equal_approx(color_next)) {
				color_samples.resize(1 + (x_to - x_from) / COLOR_EDIT_SAMPLE_INTERVAL); // Make a color sample every 64 px.
				for (int i = 1; i < color_samples.size(); i++) {
					float j = i;
					color_samples.write[i] = get_animation()->value_track_interpolate(
							get_track(),
							Math::lerp(start_time, end_time, j / color_samples.size()));
				}
			}
			color_samples.append(color_next);
		} else {
			color_samples.append(color_samples[0]);
		}
	} else {
		color_samples.append(get_key_value(p_index + 1));
	}

	for (int i = 0; i < color_samples.size() - 1; i++) {
		Vector<Vector2> points = {
			Vector2(Math::lerp(x_from, x_to, float(i) / (color_samples.size() - 1)), y_from),
			Vector2(Math::lerp(x_from, x_to, float(i + 1) / (color_samples.size() - 1)), y_from),
			Vector2(Math::lerp(x_from, x_to, float(i + 1) / (color_samples.size() - 1)), y_from + fh),
			Vector2(Math::lerp(x_from, x_to, float(i) / (color_samples.size() - 1)), y_from + fh)
		};

		Vector<Color> colors = {
			color_samples[i],
			color_samples[i + 1],
			color_samples[i + 1],
			color_samples[i]
		};

		draw_primitive(points, colors, Vector<Vector2>());
	}
}

/// SPRITE FRAME / FRAME_COORDS ///

AnimationTrackEditSpriteFrame::AnimationTrackEditSpriteFrame() {
	key_pivot.x = 0.0;
	key_pivot.y = 0.5;
}

void AnimationTrackEditSpriteFrame::set_node(Object *p_object) {
	id = p_object->get_instance_id();
}

float AnimationTrackEditSpriteFrame::get_key_width(const int p_index) const {
	return _get_theme_font_height(2.0);
}

float AnimationTrackEditSpriteFrame::get_key_height(const int p_index) const {
	return _get_theme_font_height(2.0);
}

Ref<Resource> AnimationTrackEditSpriteFrame::get_resource(const int p_index) const {
	Object *object = ObjectDB::get_instance(get_node_id());

	if (!object) {
		return Ref<Resource>();
	}

	Ref<Texture2D> texture;
	if (Object::cast_to<Sprite2D>(object) || Object::cast_to<Sprite3D>(object)) {
		texture = object->call("get_texture");

	} else if (Object::cast_to<AnimatedSprite2D>(object) || Object::cast_to<AnimatedSprite3D>(object)) {
		List<StringName> animations;

		Ref<SpriteFrames> sprite_frames = object->call("get_sprite_frames");
		if (!sprite_frames.is_valid()) {
			return Ref<Texture2D>();
		}
		sprite_frames->get_animation_list(&animations);

		int frame = get_animation()->track_get_key_value(get_track(), p_index);
		String animation_name;
		if (animations.size() == 1) {
			animation_name = animations.front()->get();
		} else {
			// Go through other track to find if animation is set
			String animation_path = String(get_animation()->track_get_path(get_track()));
			animation_path = animation_path.replace(":frame", ":animation");
			int animation_track = get_animation()->find_track(animation_path, get_animation()->track_get_type(get_track()));
			float track_time = get_animation()->track_get_key_time(get_track(), p_index);
			int animation_index = get_animation()->track_find_key(animation_track, track_time);
			animation_name = get_animation()->track_get_key_value(animation_track, animation_index);
		}

		texture = sprite_frames->get_frame_texture(animation_name, frame);
	}

	return texture;
}

Rect2 AnimationTrackEditSpriteFrame::_create_texture_region_sprite(int p_index, Object *object, const Ref<Texture2D> texture) const {
	Rect2 region;

	if (texture.is_valid()) {
		int hframes = object->call("get_hframes");
		int vframes = object->call("get_vframes");

		Variant value = get_animation()->track_get_key_value(get_track(), p_index);
		Vector2 coords;
		if (is_coords) {
			coords = value;
		} else {
			int frame = value;
			coords.x = frame % hframes;
			coords.y = frame / hframes;
		}

		region.size = texture->get_size();

		if (bool(object->call("is_region_enabled"))) {
			region = Rect2(object->call("get_region_rect"));
		}

		if (hframes > 1) {
			region.size.x /= hframes;
		}
		if (vframes > 1) {
			region.size.y /= vframes;
		}

		region.position.x += region.size.x * coords.x;
		region.position.y += region.size.y * coords.y;
	}

	return region;
}

Rect2 AnimationTrackEditSpriteFrame::_create_region_animated_sprite(int p_index, Object *object, const Ref<Texture2D> texture) const {
	Rect2 region;

	if (texture.is_valid()) {
		region.size = texture->get_size();
	}

	return region;
}

void AnimationTrackEditSpriteFrame::draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) {
	Object *object = ObjectDB::get_instance(get_node_id());
	Ref<Resource> resource = get_resource(p_index);

	Ref<Texture2D> texture = resource;
	Rect2 region;

	if (Object::cast_to<Sprite2D>(object) || Object::cast_to<Sprite3D>(object)) {
		region = _create_texture_region_sprite(p_index, object, texture);

	} else if (Object::cast_to<AnimatedSprite2D>(object) || Object::cast_to<AnimatedSprite3D>(object)) {
		region = _create_region_animated_sprite(p_index, object, texture);
	}

	Rect2 rect = get_global_key_rect(p_index);

	Color accent = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
	Color bg = accent;
	bg.a = 0.15;
	editor->_draw_rect_clipped(this, rect, bg, true, p_clip_left, p_clip_right);
	editor->_draw_texture_region_clipped(this, texture, rect, region, p_clip_left, p_clip_right);
}

bool AnimationTrackEditSpriteFrame::has_valid_key(const int p_index) const {
	Object *object = ObjectDB::get_instance(get_node_id());
	if (!object) {
		return false;
	}

	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		return false;
	}

	return true;
}

void AnimationTrackEditSpriteFrame::set_as_coords() {
	is_coords = true;
}

/// SUB ANIMATION ///

AnimationTrackEditSubAnim::AnimationTrackEditSubAnim() {
	key_pivot.x = 0.0;
	key_pivot.y = 0.5;
}

bool AnimationTrackEditSubAnim::has_valid_key(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		return false;
	}

	StringName edit_name = get_edit_name(p_index);
	if (edit_name.is_empty() || edit_name == "[stop]") {
		return false;
	}

	return true;
}

void AnimationTrackEditSubAnim::get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {
	Ref<AnimationPreview> preview = AnimationPreviewGenerator::get_singleton()->generate_preview(resource);
	preview->create_key_region_data(points, rect, p_pixels_sec, start_ofs);
}

void AnimationTrackEditSubAnim::_preview_changed(ObjectID p_which) {
	Object *object = ObjectDB::get_instance(get_node_id());

	if (!object) {
		return;
	}

	StringName anim_name = object->call("get_animation");
	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(get_root()->get_node_or_null(get_animation()->track_get_path(get_track())));
	if (ap && ap->has_animation(anim_name)) {
		Ref<Animation> anim = ap->get_animation(anim_name);

		if (anim.is_valid() && anim->get_instance_id() == p_which) {
			queue_redraw();
		}
	}
}

Ref<Resource> AnimationTrackEditSubAnim::get_resource(const int p_index) const {
	StringName anim_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);

	if (String(anim_name) == "[stop]") {
		return Ref<Resource>();
	}

	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(get_root()->get_node_or_null(get_animation()->track_get_path(get_track())));
	if (!ap || !ap->has_animation(anim_name)) {
		return Ref<Resource>();
	}

	Ref<Animation> anim = ap->get_animation(anim_name);
	if (!anim.is_valid()) {
		return Ref<Resource>();
	}

	return anim;
}

float AnimationTrackEditSubAnim::get_length(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<Animation> anim = resource;
		return anim->get_length();
	}

	return AnimationTrackEditClip::get_length(p_index);
}

StringName AnimationTrackEditSubAnim::get_edit_name(const int p_index) const {
	StringName edit_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);
	if (!edit_name.is_empty()) {
		return edit_name;
	}

	return AnimationTrackEditClip::get_edit_name(p_index);
}

//// VOLUME DB ////

AnimationTrackEditVolumeDB::AnimationTrackEditVolumeDB() {
	key_pivot.x = 0.5;
	key_pivot.y = 0.5;
}

float AnimationTrackEditVolumeDB::get_key_width(const int p_index) const {
	return AnimationTrackEdit::get_key_width(p_index);
}

float AnimationTrackEditVolumeDB::get_key_height(const int p_index) const {
	return AnimationTrackEdit::get_key_height(p_index);
}

void AnimationTrackEditVolumeDB::draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) {
	AnimationTrackEdit::draw_key(p_index, p_global_rect, p_selected, p_clip_left, p_clip_right);
}

void AnimationTrackEditVolumeDB::draw_bg(const float p_clip_left, const float p_clip_right) {
	Ref<Texture2D> volume_texture = get_editor_theme_icon(SNAME("ColorTrackVu"));
	int tex_h = volume_texture->get_height();

	float y_from = (get_size().height - tex_h) / 2.0;
	float y_size = tex_h;

	Color color(1, 1, 1, 0.3);
	draw_texture_rect(volume_texture, Rect2(p_clip_left, y_from, p_clip_right - p_clip_left, y_from + y_size), false, color);
}

void AnimationTrackEditVolumeDB::draw_fg(const float p_clip_left, const float p_clip_right) {
	Ref<Texture2D> volume_texture = get_editor_theme_icon(SNAME("ColorTrackVu"));
	int tex_h = volume_texture->get_height();
	float y_from = (get_size().height - tex_h) / 2.0;
	float db0 = y_from + (24 / 80.0) * tex_h;

	editor->_draw_line_clipped(this, Vector2(p_clip_left, db0), Vector2(p_clip_right, db0), Color(1, 1, 1, 0.3), -1.0, p_clip_left, p_clip_right);
}

bool AnimationTrackEditVolumeDB::is_linked(const int p_index, const int p_index_next) const {
	return true;
}

float AnimationTrackEditVolumeDB::get_key_y(const int p_index) const {
	float db = get_key_value(p_index);

	float min_db = -60;
	float max_db = 24;
	float diff_db = max_db - min_db;

	db = CLAMP(db, min_db, max_db);
	float norm_h = (db - min_db) / diff_db; // [0, 1], 0 = -60 dB, 1 = 24 dB

	// Scale to small range around track_alignment
	float scale_factor = 0.5; // % of track height
	return (norm_h - 0.5) * scale_factor;
}

void AnimationTrackEditVolumeDB::draw_key_link(const int p_index, const Rect2 &p_global_rect, const Rect2 &p_global_rect_next, const float p_clip_left, const float p_clip_right) {
	Point2 center = p_global_rect.get_center();
	Point2 center_n = p_global_rect_next.get_center();

	Color color = get_theme_color(SceneStringName(font_color), SNAME("Label"));
	color.a *= REGION_EDGE_ALPHA;

	editor->_draw_line_clipped(this, center, center_n, color, 2, p_clip_left, p_clip_right);
}

/// AUDIO ///

AnimationTrackEditTypeAudio::AnimationTrackEditTypeAudio() {
	key_pivot.x = 0.0;
	key_pivot.y = 0.5;
	AudioStreamPreviewGenerator::get_singleton()->connect("preview_updated", callable_mp(this, &AnimationTrackEditTypeAudio::_preview_changed));
}

bool AnimationTrackEditTypeAudio::has_valid_key(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		return false;
	}

	return true;
}

String AnimationTrackEditTypeAudio::_get_tooltip(const int p_index) const {
	String text = "";

	String stream_name = get_edit_name(p_index);

	text += TTR("Stream:") + " " + stream_name + "\n";
	float so = get_start_offset(p_index);
	text += TTR("Start (s):") + " " + rtos(so) + "\n";
	float eo = get_end_offset(p_index);
	text += TTR("End (s):") + " " + rtos(eo) + "\n";

	return text;
}

void AnimationTrackEditTypeAudio::_preview_changed(ObjectID p_which) {
	AnimationTrackEditClip::_preview_changed(p_which);
}

void AnimationTrackEditTypeAudio::get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {
	Ref<AudioStreamPreview> preview = AudioStreamPreviewGenerator::get_singleton()->generate_preview(resource);
	preview->create_key_region_data(points, rect, p_pixels_sec, start_ofs);
}

void AnimationTrackEditTypeAudio::apply_data(const Ref<Resource> resource, const float time) {
	Ref<AudioStream> stream = resource;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(get_animation().ptr(), "audio_track_insert_key", get_track(), time, stream);
}

Ref<Resource> AnimationTrackEditTypeAudio::get_resource(const int p_index) const {
	return get_animation()->audio_track_get_key_stream(get_track(), p_index);
}

float AnimationTrackEditTypeAudio::get_start_offset(const int p_index) const {
	return get_animation()->audio_track_get_key_start_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAudio::get_end_offset(const int p_index) const {
	return get_animation()->audio_track_get_key_end_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAudio::get_length(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<AudioStream> stream = resource;
		return stream->get_length();
	}

	return AnimationTrackEditClip::get_length(p_index);
}

void AnimationTrackEditTypeAudio::set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(get_animation().ptr(), "audio_track_set_key_start_offset", get_track(), p_index, new_ofs);
	undo_redo->add_undo_method(get_animation().ptr(), "audio_track_set_key_start_offset", get_track(), p_index, prev_ofs);
}

void AnimationTrackEditTypeAudio::set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(get_animation().ptr(), "audio_track_set_key_end_offset", get_track(), p_index, new_ofs);
	undo_redo->add_undo_method(get_animation().ptr(), "audio_track_set_key_end_offset", get_track(), p_index, prev_ofs);
}

/// AUDIO ///

AnimationTrackEditAudio::AnimationTrackEditAudio() {
	key_pivot.x = 0.0;
	key_pivot.y = 0.5;
	AudioStreamPreviewGenerator::get_singleton()->connect("preview_updated", callable_mp(this, &AnimationTrackEditAudio::_preview_changed));
}

bool AnimationTrackEditAudio::has_valid_key(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		return false;
	}

	return true;
}

void AnimationTrackEditAudio::_preview_changed(ObjectID p_which) {
	Object *object = ObjectDB::get_instance(get_node_id());

	if (!object) {
		return;
	}

	Ref<Resource> resource = object->call("get_stream");

	if (resource.is_valid() && resource->get_instance_id() == p_which) {
		queue_redraw();
	}
}

void AnimationTrackEditAudio::get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {
	Ref<AudioStreamPreview> preview = AudioStreamPreviewGenerator::get_singleton()->generate_preview(resource);
	preview->create_key_region_data(points, rect, p_pixels_sec, start_ofs);
}

Ref<Resource> AnimationTrackEditAudio::get_resource(const int p_index) const {
	return get_animation()->audio_track_get_key_stream(get_track(), p_index);
}

float AnimationTrackEditAudio::get_length(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<AudioStream> stream = resource;
		return stream->get_length();
	}

	return AnimationTrackEditClip::get_length(p_index);
}

/// TYPE ANIMATION ///

AnimationTrackEditTypeAnimation::AnimationTrackEditTypeAnimation() {
	key_pivot.x = 0.0;
	key_pivot.y = 0.5;
	AnimationPreviewGenerator::get_singleton()->connect("preview_updated", callable_mp(this, &AnimationTrackEditTypeAnimation::_preview_changed));
}

bool AnimationTrackEditTypeAnimation::has_valid_key(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		return false;
	}

	StringName edit_name = get_edit_name(p_index);
	if (edit_name.is_empty() || edit_name == "[stop]") {
		return false;
	}

	return true;
}

float AnimationTrackEditTypeAnimation::get_key_scale(const int p_index) const {
	//Object *object = ObjectDB::get_instance(get_node_id());
	//AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(get_root()->get_node_or_null(get_animation()->track_get_path(get_track())));
	//if (ap) {
	//	return ap->get_speed_scale();
	//}
	return 1.0;
}

String AnimationTrackEditTypeAnimation::_get_tooltip(const int p_index) const {
	String text = "";

	String anim_name = get_edit_name(p_index);
	text += TTR("Animation Clip:") + " " + anim_name + "\n";
	float so = get_start_offset(p_index);
	text += TTR("Start (s):") + " " + rtos(so) + "\n";
	float eo = get_end_offset(p_index);
	text += TTR("End (s):") + " " + rtos(eo) + "\n";

	return text;
}

void AnimationTrackEditTypeAnimation::_preview_changed(ObjectID p_which) {
	AnimationTrackEditClip::_preview_changed(p_which);
}

void AnimationTrackEditTypeAnimation::get_key_region_data(Ref<Resource> resource, Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {
	Ref<AnimationPreview> preview = AnimationPreviewGenerator::get_singleton()->generate_preview(resource);
	preview->create_key_region_data(points, rect, p_pixels_sec, start_ofs);
}

void AnimationTrackEditTypeAnimation::apply_data(const Ref<Resource> resource, const float time) {
	Ref<Animation> anim = resource;

	StringName anim_name = anim->get_name();
	if (anim_name == StringName("[stop]")) {
		WARN_PRINT("Cannot insert [stop] animation key.");
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(get_animation().ptr(), "animation_track_insert_key", get_track(), time, anim_name);
}

Ref<Resource> AnimationTrackEditTypeAnimation::get_resource(const int p_index) const {
	StringName anim_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);

	if (String(anim_name) == "[stop]") {
		return Ref<Resource>();
	}

	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(get_root()->get_node_or_null(get_animation()->track_get_path(get_track())));
	if (!ap || !ap->has_animation(anim_name)) {
		return Ref<Resource>();
	}

	Ref<Animation> anim = ap->get_animation(anim_name);
	if (!anim.is_valid()) {
		return Ref<Resource>();
	}

	return anim;
}

float AnimationTrackEditTypeAnimation::get_start_offset(const int p_index) const {
	return get_animation()->animation_track_get_key_start_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAnimation::get_end_offset(const int p_index) const {
	return get_animation()->animation_track_get_key_end_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAnimation::get_length(const int p_index) const {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<Animation> anim = resource;
		return anim->get_length();
	}

	return AnimationTrackEditClip::get_length(p_index);
}

void AnimationTrackEditTypeAnimation::set_start_offset(const int p_index, const float prev_ofs, const float new_ofs) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(get_animation().ptr(), "animation_track_set_key_start_offset", get_track(), p_index, new_ofs);
	undo_redo->add_undo_method(get_animation().ptr(), "animation_track_set_key_start_offset", get_track(), p_index, prev_ofs);
}

void AnimationTrackEditTypeAnimation::set_end_offset(const int p_index, const float prev_ofs, const float new_ofs) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(get_animation().ptr(), "animation_track_set_key_end_offset", get_track(), p_index, new_ofs);
	undo_redo->add_undo_method(get_animation().ptr(), "animation_track_set_key_end_offset", get_track(), p_index, prev_ofs);
}

StringName AnimationTrackEditTypeAnimation::get_edit_name(const int p_index) const {
	StringName edit_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);
	if (!edit_name.is_empty()) {
		return edit_name;
	}

	return AnimationTrackEditClip::get_edit_name(p_index);
}

/// KEY ///

float AnimationTrackEditClip::get_key_width(const int p_index) const {
	float start_ofs = get_start_offset(p_index);
	float end_ofs = get_end_offset(p_index);
	float len = get_length(p_index);

	Region region = _calc_key_region(p_index, start_ofs, end_ofs, len);
	return region.width;
}

float AnimationTrackEditClip::get_key_height(const int p_index) const {
	return _get_theme_font_height(1.5);
}

int AnimationTrackEditClip::handle_track_resizing(const Ref<InputEventMouseMotion> mm, const int p_index, const Rect2 p_global_rect, const int p_clip_left, const int p_clip_right) {
	float len = get_length(p_index);
	float start_ofs = get_start_offset(p_index);
	float end_ofs = get_end_offset(p_index);

	float offset = _get_pixels_sec(p_index, true);
	Region region = _calc_key_region(p_index, start_ofs, end_ofs, len, offset);
	region = _clip_key_region(region, p_clip_left, p_clip_right);

	float region_begin = region.x;
	float region_end = (region.x + region.width);

	if (region_begin >= p_clip_left && region_end <= p_clip_right && region_begin <= region_end) {
		bool resize_start = false;
		int can_resize = false;

		float mouse_pos = mm->get_position().x;

		float diff_left = region_begin - mouse_pos;
		float diff_right = mouse_pos - region_end;

		float resize_threshold = REGION_RESIZE_THRESHOLD * EDSCALE;

		if (diff_left > 0) { // left outside clip
			if (Math::abs(diff_left) < resize_threshold) {
				resize_start = true;
				can_resize = true;
			}
		} else if (diff_right > 0) { // right outside clip
			if (Math::abs(diff_right) < resize_threshold) {
				resize_start = false;
				can_resize = true;
			}
		} else { // inside clip
			if (Math::abs(diff_left) < resize_threshold && Math::abs(diff_right) < resize_threshold) { // closest inside clip
				resize_start = Math::abs(diff_left) < Math::abs(diff_right);
				can_resize = true;
			} else if (Math::abs(diff_left) < resize_threshold) { // left inside clip
				resize_start = true;
				can_resize = true;
			} else if (Math::abs(diff_right) < resize_threshold) { // right inside clip
				resize_start = false;
				can_resize = true;
			}
		}

		if (can_resize) {
			len_resizing_start = resize_start;
			return p_index;
		}
	}

	return -1;
}

void AnimationTrackEditClip::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (!len_resizing && mm.is_valid()) {
		bool use_hsize_cursor = false;
		for (int p_index = 0; p_index < get_animation()->track_get_key_count(get_track()); p_index++) {
			Rect2 global_rect = get_global_key_rect(p_index);

			float clip_left = get_timeline()->get_name_limit();
			float clip_right = get_size().width - get_timeline()->get_buttons_width();

			int resizing_index = handle_track_resizing(mm, p_index, global_rect, clip_left, clip_right);
			if (resizing_index != -1) {
				if (!has_valid_key(resizing_index)) {
					AnimationTrackEdit::gui_input(p_event);
					return;
				}

				use_hsize_cursor = true;
				len_resizing_index = resizing_index;
			}
		}
		over_drag_position = use_hsize_cursor;
	}

	if (len_resizing && mm.is_valid()) {
		// Rezising index is some.
		len_resizing_rel += mm->get_relative().x;

		float ofs_local = len_resizing_rel / get_timeline()->get_zoom_scale();
		float prev_ofs_start = get_start_offset(len_resizing_index);
		float prev_ofs_end = get_end_offset(len_resizing_index);
		float len = get_length(len_resizing_index);

		float anim_len = len - prev_ofs_end - prev_ofs_start;

		if (len_resizing_start) {
			len_resizing_rel = CLAMP(ofs_local, -prev_ofs_start, MAX(0.0, anim_len)) * get_timeline()->get_zoom_scale();
		} else {
			len_resizing_rel = CLAMP(ofs_local, -(MAX(0.0, anim_len)), prev_ofs_end) * get_timeline()->get_zoom_scale();
		}

		queue_redraw();
		accept_event();
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && over_drag_position) {
		len_resizing = true;
		// In case if resizing index is not set yet reset the flag.
		if (len_resizing_index < 0) {
			len_resizing = false;
			return;
		}
		len_resizing_from_px = mb->get_position().x;
		len_resizing_rel = 0;

		emit_signal(SNAME("select_key"), len_resizing_index, true);

		queue_redraw();
		accept_event();
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	if (len_resizing && mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		if (len_resizing_rel == 0 || len_resizing_index < 0) {
			len_resizing = false;
			return;
		}

		if (len_resizing_start) {
			float ofs_local = len_resizing_rel / get_timeline()->get_zoom_scale();
			float prev_ofs = get_start_offset(len_resizing_index);
			float prev_time = get_animation()->track_get_key_time(get_track(), len_resizing_index);
			// Raster die neue Zeit
			float new_time = editor->snap_time(prev_time + ofs_local);
			float new_ofs = prev_ofs + (new_time - prev_time);
			if (Math::abs(prev_time - new_time) > CMP_EPSILON) { // Prüfe relevante Änderung
				float offset = new_time - prev_time;

				undo_redo->create_action(TTR("Change Track Clip Start Offset"));
				set_start_offset(len_resizing_index, prev_ofs, new_ofs);
				undo_redo->commit_action();

				emit_signal(SNAME("move_selection_begin"));
				emit_signal(SNAME("move_selection"), offset);
				emit_signal(SNAME("move_selection_commit"));
			}
		} else {
			float ofs_local = -len_resizing_rel / get_timeline()->get_zoom_scale();
			float prev_ofs = get_end_offset(len_resizing_index);
			// Raster die neue Zeit
			float new_time = editor->snap_time(prev_ofs + ofs_local);
			float new_ofs = new_time; // End-Offset ist absolut
			if (Math::abs(prev_ofs - new_ofs) > CMP_EPSILON) { // Prüfe relevante Änderung
				undo_redo->create_action(TTR("Change Track Clip End Offset"));
				set_end_offset(len_resizing_index, prev_ofs, new_ofs);
				undo_redo->commit_action();
			}
		}

		len_resizing_index = -1;
		len_resizing = false;
		queue_redraw();
		accept_event();
		return;
	}

	AnimationTrackEdit::gui_input(p_event);
}

Control::CursorShape AnimationTrackEditClip::get_cursor_shape(const Point2 &p_pos) const {
	if (over_drag_position || len_resizing) {
		return Control::CURSOR_HSIZE;
	} else {
		return get_default_cursor_shape();
	}
}

void AnimationTrackEditClip::draw_key(const int p_index, const Rect2 &p_global_rect, const bool p_selected, const float p_clip_left, const float p_clip_right) {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		AnimationTrackEdit::draw_key(p_index, p_global_rect, p_selected, p_clip_left, p_clip_right);
		return;
	}

	float start_ofs = get_start_offset(p_index);
	float end_ofs = get_end_offset(p_index);
	float len = get_length(p_index);

	float diff_start_ofs = 0;
	float diff_end_ofs = 0;

	float scale = get_timeline()->get_zoom_scale();

	if (len_resizing && p_index == len_resizing_index) {
		float ofs_local = len_resizing_rel / scale;
		float snapped_ofs = editor->snap_time(ofs_local);
		if (len_resizing_start) {
			diff_start_ofs = snapped_ofs;
		} else {
			diff_end_ofs = -snapped_ofs;
		}
	}

	float offset = p_global_rect.position.x + (diff_start_ofs * scale);
	float start_ofs_ = start_ofs + diff_start_ofs;
	float end_ofs_ = end_ofs + diff_end_ofs;

	Region orig_region = _calc_key_region(p_index, start_ofs_, end_ofs_, len, offset);

	bool is_outside = _is_key_region_outside(orig_region, p_clip_left, p_clip_right);
	if (is_outside) {
		return;
	}

	Region region = _clip_key_region(orig_region, p_clip_left, p_clip_right);

	float region_begin = region.x;
	float region_end = region.x + region.width;
	float region_width = region.width;

	Rect2 key_rect = get_global_key_rect(p_index);

	Color accent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));

	if (orig_region.width <= REGION_MAX_WIDTH) {
		Rect2 region_rect = Rect2(region_begin, key_rect.position.y, REGION_MAX_WIDTH, key_rect.size.y);
		draw_rect(region_rect, accent_color);

		if (p_selected) {
			draw_rect(region_rect, accent_color, false);
		}

		return;
	}

	Color bg_color = REGION_BG_COLOR;
	Rect2 rect = Rect2(region_begin, key_rect.position.y, region_width, key_rect.size.y);
	draw_rect(rect, bg_color);

	{
		Vector<Vector2> points;
		points.resize(Math::ceil(rect.size.x) * 2);

		Vector<Color> colors;
		if (p_selected) {
			colors = { bg_color.lightened(0.8) };
		} else {
			colors = { bg_color.lightened(0.2) };
		}

		Region region_shift = _calc_key_region_shift(orig_region, region);
		get_key_region_data(resource, points, rect, scale, start_ofs_ + region_shift.x / scale);

		if (!points.is_empty()) {
			draw_multiline_colors(points, colors);
		}
	}

	Color edge_color = Color(accent_color);
	edge_color.a = REGION_EDGE_ALPHA;
	if (start_ofs_ > 0 && region_begin > p_clip_left) {
		draw_rect(Rect2(region_begin, rect.position.y, 1, rect.size.y), edge_color);
	}
	if (end_ofs_ > 0 && region_end < p_clip_right) {
		draw_rect(Rect2(region_end, rect.position.y, 1, rect.size.y), edge_color);
	}

	if (region_width > REGION_MAX_WIDTH) {
		StringName edit_name = get_edit_name(p_index);
		if (!edit_name.is_empty()) {
			Color name_color;
			if (p_selected) {
				name_color = Color(accent_color);
				name_color.a = 0.0;
			} else {
				Color font_color = get_theme_color(SceneStringName(font_color), SNAME("Label"));
				name_color = font_color;
			}

			float max_width = region_width - REGION_FONT_MARGIN;

			Ref<Font> font = get_theme_font(SceneStringName(font), SNAME("Label"));
			int font_size = get_theme_font_size(SceneStringName(font_size), SNAME("Label"));

			int f_h = key_rect.position.y + key_rect.size.y / 2 - font->get_height(font_size) / 2 + font->get_ascent(font_size);
			draw_string(font, Point2(region_begin + REGION_FONT_MARGIN, f_h), editor->_make_text_clipped(edit_name, font, font_size, max_width), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, name_color);
		}
	}
}

bool AnimationTrackEditClip::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	if (p_point.x > get_timeline()->get_name_limit() && p_point.x < get_size().width - get_timeline()->get_buttons_width()) {
		Dictionary drag_data = p_data;
		if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
			Ref<Resource> res = drag_data["resource"];
			if (res.is_valid()) {
				return true;
			}
		}

		if (drag_data.has("type") && String(drag_data["type"]) == "files") {
			Vector<String> files = drag_data["files"];

			if (files.size() == 1) {
				Ref<Resource> res = ResourceLoader::load(files[0]);
				if (res.is_valid()) {
					return true;
				}
			}
		}
	}

	return AnimationTrackEdit::can_drop_data(p_point, p_data);
}

void AnimationTrackEditClip::drop_data(const Point2 &p_point, const Variant &p_data) {
	if (p_point.x > get_timeline()->get_name_limit() && p_point.x < get_size().width - get_timeline()->get_buttons_width()) {
		Ref<Resource> resource;
		Dictionary drag_data = p_data;
		if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
			resource = drag_data["resource"];
		} else if (drag_data.has("type") && String(drag_data["type"]) == "files") {
			Vector<String> files = drag_data["files"];

			if (files.size() == 1) {
				resource = ResourceLoader::load(files[0]);
			}
		}

		if (resource.is_valid()) {
			int x = p_point.x - get_timeline()->get_name_limit();
			float time = x / get_timeline()->get_zoom_scale();
			time += get_timeline()->get_value();

			time = get_editor()->snap_time(time);

			while (get_animation()->track_find_key(get_track(), time, Animation::FIND_MODE_APPROX) != -1) { //make sure insertion point is valid
				time += 0.0001;
			}

			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Add Track Clip"));
			apply_data(resource, time);
			undo_redo->add_undo_method(get_animation().ptr(), "track_remove_key_at_time", get_track(), time);
			undo_redo->commit_action();

			queue_redraw();
			return;
		}
	}

	AnimationTrackEdit::drop_data(p_point, p_data);
}

void AnimationTrackEditClip::set_node(Object *p_object) {
	id = p_object->get_instance_id();
}

StringName AnimationTrackEditClip::get_edit_name(const int p_index) const {
	String resource_name = "null";
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		if (resource->get_path().is_resource_file()) {
			resource_name = resource->get_path().get_file();
		} else if (!resource->get_name().is_empty()) {
			resource_name = resource->get_name();
		} else {
			resource_name = resource->get_class();
		}
	}
	return resource_name;
}

Region AnimationTrackEditClip::_calc_key_region(const int p_index, const float p_start_ofs, const float p_end_ofs, const float p_len, float p_offset) const {
	float anim_len = p_len - p_start_ofs - p_end_ofs;
	if (anim_len < 0) {
		anim_len = 0;
	}

	if (!(len_resizing && len_resizing_index == p_index)) {
		bool has_next_key = get_animation()->track_get_key_count(get_track()) > p_index + 1;
		if (has_next_key) {
			anim_len = MIN(anim_len, get_animation()->track_get_key_time(get_track(), p_index + 1) - get_animation()->track_get_key_time(get_track(), p_index));
		}
	}

	float scale = get_timeline()->get_zoom_scale();

	float pixel_len = anim_len * scale;
	float pixel_begin = p_offset;

	return Region(pixel_begin, pixel_len);
}

Region AnimationTrackEditClip::_clip_key_region(const Region &region, const int p_clip_left, const int p_clip_right) {
	float region_end = CLAMP(region.x + region.width, MAX(region.x, p_clip_left), p_clip_right);
	float region_start = CLAMP(region.x, p_clip_left, MIN(region_end, p_clip_right));
	ERR_FAIL_COND_V_MSG(region_start > region_end, region, "Clipped region is invalid (x > y).");

	return Region(region_start, region_end - region_start);
}

Region AnimationTrackEditClip::_calc_key_region_shift(const Region &orig_region, const Region &region) const {
	return Region(region.x - orig_region.x, (region.x + region.width) - (orig_region.x + orig_region.width));
}

bool AnimationTrackEditClip::_is_key_region_outside(const Region &region, const int p_clip_left, const int p_clip_right) const {
	return ((region.x + region.width) < p_clip_left || region.x > p_clip_right);
}

void AnimationTrackEditClip::_preview_changed(ObjectID p_which) {
	for (int p_index = 0; p_index < get_animation()->track_get_key_count(get_track()); p_index++) {
		Ref<Resource> resource = get_resource(p_index);
		if (resource.is_valid() && resource->get_instance_id() == p_which) {
			queue_redraw();
			return;
		}
	}
}

/// PLUGIN ///

AnimationTrackEdit *AnimationTrackEditDefaultPlugin::create_value_track_edit(Object *p_object, Variant::Type p_type, const String &p_property, PropertyHint p_hint, const String &p_hint_string, int p_usage) {
	if (p_property == "playing" && (p_object->is_class("AudioStreamPlayer") || p_object->is_class("AudioStreamPlayer2D") || p_object->is_class("AudioStreamPlayer3D"))) {
		AnimationTrackEditAudio *audio = memnew(AnimationTrackEditAudio);
		audio->set_node(p_object);
		return audio;
	}

	if (p_property == "frame" && (p_object->is_class("Sprite2D") || p_object->is_class("Sprite3D") || p_object->is_class("AnimatedSprite2D") || p_object->is_class("AnimatedSprite3D"))) {
		AnimationTrackEditSpriteFrame *sprite = memnew(AnimationTrackEditSpriteFrame);
		sprite->set_node(p_object);
		return sprite;
	}

	if (p_property == "frame_coords" && (p_object->is_class("Sprite2D") || p_object->is_class("Sprite3D"))) {
		AnimationTrackEditSpriteFrame *sprite = memnew(AnimationTrackEditSpriteFrame);
		sprite->set_as_coords();
		sprite->set_node(p_object);
		return sprite;
	}

	if (p_property == "current_animation" && (p_object->is_class("AnimationPlayer"))) {
		AnimationTrackEditSubAnim *player = memnew(AnimationTrackEditSubAnim);
		player->set_node(p_object);
		return player;
	}

	if (p_property == "volume_db") {
		AnimationTrackEditVolumeDB *vu = memnew(AnimationTrackEditVolumeDB);
		return vu;
	}

	if (p_type == Variant::BOOL) {
		return memnew(AnimationTrackEditBool);
	}
	if (p_type == Variant::COLOR) {
		return memnew(AnimationTrackEditColor);
	}

	return nullptr;
}

AnimationTrackEdit *AnimationTrackEditDefaultPlugin::create_audio_track_edit() {
	return memnew(AnimationTrackEditTypeAudio);
}

AnimationTrackEdit *AnimationTrackEditDefaultPlugin::create_animation_track_edit(Object *p_object) {
	AnimationTrackEditTypeAnimation *an = memnew(AnimationTrackEditTypeAnimation);
	an->set_node(p_object);
	return an;
}

AnimationTrackEdit *AnimationTrackEditDefaultPlugin::create_method_track_edit() {
	return memnew(AnimationTrackEditTypeMethod);
}
