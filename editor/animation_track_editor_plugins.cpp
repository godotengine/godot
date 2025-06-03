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
}

int AnimationTrackEditBool::get_key_height() const {
	Ref<Texture2D> checked = get_theme_icon(SNAME("checked"), SNAME("CheckBox"));
	return checked->get_height();
}

Rect2 AnimationTrackEditBool::get_key_rect(int p_index, float p_pixels_sec) {
	Ref<Texture2D> checked = get_theme_icon(SNAME("checked"), SNAME("CheckBox"));
	return Rect2(-checked->get_width() / 2, 0, checked->get_width(), get_size().height);
}

void AnimationTrackEditBool::draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) {
	bool checked = get_animation()->track_get_key_value(get_track(), p_index);
	Ref<Texture2D> icon = get_theme_icon(checked ? "checked" : "unchecked", "CheckBox");

	float x_from = p_x - icon->get_width() / 2;
	float x_to = p_x + icon->get_width() / 2;

	if (x_from > p_clip_right || x_to < p_clip_left) {
		return;
	}

	int h = int(get_size().height - icon->get_height()) / 2;
	Vector2 ofs(x_from, h);
	draw_texture(icon, ofs);

	if (p_selected) {
		Color color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		draw_color_rect_clipped(Rect2(ofs, icon->get_size()), color, false, p_clip_left, p_clip_right);
	}
}

/// COLOR ///

AnimationTrackEditColor::AnimationTrackEditColor() {
	key_scale = 0.8;
}

Rect2 AnimationTrackEditColor::get_key_rect(int p_index, float p_pixels_sec) {
	int fh = get_key_height();
	int h = get_size().height;
	return Rect2(-fh / 2, 0, fh, h);
}

void AnimationTrackEditColor::draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) {
	Color color = get_animation()->track_get_key_value(get_track(), p_index);
	int fh = get_key_height();

	Rect2 rect(Vector2(p_x - fh / 2, int(get_size().height - fh) / 2), Size2(fh, fh));

	draw_color_rect_clipped(Rect2(rect.position, rect.size / 2), Color(0.4, 0.4, 0.4), true, p_clip_left, p_clip_right);
	draw_color_rect_clipped(Rect2(rect.position + rect.size / 2, rect.size / 2), Color(0.4, 0.4, 0.4), true, p_clip_left, p_clip_right);
	draw_color_rect_clipped(Rect2(rect.position + Vector2(rect.size.x / 2, 0), rect.size / 2), Color(0.6, 0.6, 0.6), true, p_clip_left, p_clip_right);
	draw_color_rect_clipped(Rect2(rect.position + Vector2(0, rect.size.y / 2), rect.size / 2), Color(0.6, 0.6, 0.6), true, p_clip_left, p_clip_right);
	draw_color_rect_clipped(rect, color, true, p_clip_left, p_clip_right);

	if (p_selected) {
		Color accent = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		draw_color_rect_clipped(rect, accent, false, p_clip_left, p_clip_right);
	}
}

void AnimationTrackEditColor::draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right) {
	int fh = get_key_height();

	fh /= 3;

	int x_from = p_x + fh / 2 - 1;
	int x_to = p_next_x - fh / 2 + 1;
	x_from = MAX(x_from, p_clip_left);
	x_to = MIN(x_to, p_clip_right);

	int y_from = (get_size().height - fh) / 2;

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
			float start_time = get_animation()->track_get_key_time(get_track(), p_index);
			float end_time = get_animation()->track_get_key_time(get_track(), p_index + 1);

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
		color_samples.append(get_animation()->track_get_key_value(get_track(), p_index + 1));
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
	key_scale = 2.0;
}

Rect2 AnimationTrackEditSpriteFrame::get_key_rect(int p_index, float p_pixels_sec) {
	Object *object = ObjectDB::get_instance(get_node_id());

	if (!object) {
		return AnimationTrackEdit::get_key_rect(p_index, p_pixels_sec);
	}

	Size2 size;

	if (Object::cast_to<Sprite2D>(object) || Object::cast_to<Sprite3D>(object)) {
		Ref<Texture2D> texture = object->call("get_texture");
		if (texture.is_null()) {
			return AnimationTrackEdit::get_key_rect(p_index, p_pixels_sec);
		}

		size = texture->get_size();

		if (bool(object->call("is_region_enabled"))) {
			size = Rect2(object->call("get_region_rect")).size;
		}

		int hframes = object->call("get_hframes");
		int vframes = object->call("get_vframes");

		if (hframes > 1) {
			size.x /= hframes;
		}
		if (vframes > 1) {
			size.y /= vframes;
		}
	} else if (Object::cast_to<AnimatedSprite2D>(object) || Object::cast_to<AnimatedSprite3D>(object)) {
		Ref<SpriteFrames> sf = object->call("get_sprite_frames");
		if (sf.is_null()) {
			return AnimationTrackEdit::get_key_rect(p_index, p_pixels_sec);
		}

		List<StringName> animations;
		sf->get_animation_list(&animations);

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

		Ref<Texture2D> texture = sf->get_frame_texture(animation_name, frame);
		if (texture.is_null()) {
			return AnimationTrackEdit::get_key_rect(p_index, p_pixels_sec);
		}

		size = texture->get_size();
	}

	size = size.floor();

	int fh = get_key_height();
	int fw = fh * size.width / size.height;

	return Rect2(0, 0, fw, get_size().height);
}

void AnimationTrackEditSpriteFrame::draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) {
	Object *object = ObjectDB::get_instance(get_node_id());

	if (!object) {
		AnimationTrackEdit::draw_key(p_index, p_pixels_sec, p_x, p_selected, p_clip_left, p_clip_right);
		return;
	}

	Ref<Texture2D> texture;
	Rect2 region;

	if (Object::cast_to<Sprite2D>(object) || Object::cast_to<Sprite3D>(object)) {
		texture = object->call("get_texture");
		if (texture.is_null()) {
			AnimationTrackEdit::draw_key(p_index, p_pixels_sec, p_x, p_selected, p_clip_left, p_clip_right);
			return;
		}

		int hframes = object->call("get_hframes");
		int vframes = object->call("get_vframes");

		Vector2 coords;
		if (is_coords) {
			coords = get_animation()->track_get_key_value(get_track(), p_index);
		} else {
			int frame = get_animation()->track_get_key_value(get_track(), p_index);
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

	} else if (Object::cast_to<AnimatedSprite2D>(object) || Object::cast_to<AnimatedSprite3D>(object)) {
		Ref<SpriteFrames> sf = object->call("get_sprite_frames");
		if (sf.is_null()) {
			AnimationTrackEdit::draw_key(p_index, p_pixels_sec, p_x, p_selected, p_clip_left, p_clip_right);
			return;
		}

		List<StringName> animations;
		sf->get_animation_list(&animations);

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

		texture = sf->get_frame_texture(animation_name, frame);
		if (texture.is_null()) {
			AnimationTrackEdit::draw_key(p_index, p_pixels_sec, p_x, p_selected, p_clip_left, p_clip_right);
			return;
		}

		region.size = texture->get_size();
	}

	int height = get_key_height();
	int width = height * region.size.width / region.size.height;

	Rect2 rect(p_x, int(get_size().height - height) / 2, width, height);

	if (rect.position.x + rect.size.x < p_clip_left) {
		return;
	}

	if (rect.position.x > p_clip_right) {
		return;
	}

	Color accent = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
	Color bg = accent;
	bg.a = 0.15;

	draw_color_rect_clipped(rect, bg, true, p_clip_left, p_clip_right);

	draw_texture_region_clipped(texture, rect, region, p_clip_left, p_clip_right);

	if (p_selected) {
		draw_color_rect_clipped(rect, accent, false, p_clip_left, p_clip_right);
	}
}

void AnimationTrackEditSpriteFrame::set_as_coords() {
	is_coords = true;
}

/// SUB ANIMATION ///

AnimationTrackEditSubAnim::AnimationTrackEditSubAnim() {
	key_scale = 1.5;
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

Ref<Resource> AnimationTrackEditSubAnim::get_resource(const int p_index) {
	StringName anim_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);

	if (String(anim_name) == "[stop]") {
		return nullptr;
	}

	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(get_root()->get_node_or_null(get_animation()->track_get_path(get_track())));
	if (!ap || !ap->has_animation(anim_name)) {
		return nullptr;
	}

	Ref<Animation> anim = ap->get_animation(anim_name);
	if (!anim.is_valid()) {
		return nullptr;
	}

	return anim;
}

float AnimationTrackEditSubAnim::get_length(const int p_index) {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<Animation> anim = resource;
		return anim->get_length();
	}

	return AnimationTrackEditKey::get_length(p_index);
}

StringName AnimationTrackEditSubAnim::get_edit_name(const int p_index) {
	StringName edit_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);
	if (!edit_name.is_empty()) {
		return edit_name;
	}

	return AnimationTrackEditKey::get_edit_name(p_index);
}

//// VOLUME DB ////

AnimationTrackEditVolumeDB::AnimationTrackEditVolumeDB() {
	key_scale = 1.2;
}

int AnimationTrackEditVolumeDB::get_key_height() const {
	Ref<Texture2D> volume_texture = get_editor_theme_icon(SNAME("ColorTrackVu"));
	return volume_texture->get_height() * key_scale;
}

void AnimationTrackEditVolumeDB::draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) {
	if (p_index == 0) {
		draw_fg(p_clip_left, p_clip_right);
	} else if (p_index == 1) {
		draw_bg(p_clip_left, p_clip_right);
	}
}

void AnimationTrackEditVolumeDB::draw_bg(int p_clip_left, int p_clip_right) {
	Ref<Texture2D> volume_texture = get_editor_theme_icon(SNAME("ColorTrackVu"));
	int tex_h = volume_texture->get_height();

	int y_from = (get_size().height - tex_h) / 2;
	int y_size = tex_h;

	Color color(1, 1, 1, 0.3);
	draw_texture_rect(volume_texture, Rect2(p_clip_left, y_from, p_clip_right - p_clip_left, y_from + y_size), false, color);
}

void AnimationTrackEditVolumeDB::draw_fg(int p_clip_left, int p_clip_right) {
	Ref<Texture2D> volume_texture = get_editor_theme_icon(SNAME("ColorTrackVu"));
	int tex_h = volume_texture->get_height();
	int y_from = (get_size().height - tex_h) / 2;
	int db0 = y_from + (24 / 80.0) * tex_h;

	draw_line(Vector2(p_clip_left, db0), Vector2(p_clip_right, db0), Color(1, 1, 1, 0.3));
}

void AnimationTrackEditVolumeDB::draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right) {
	if (p_x > p_clip_right || p_next_x < p_clip_left) {
		return;
	}

	float db = get_animation()->track_get_key_value(get_track(), p_index);
	float db_n = get_animation()->track_get_key_value(get_track(), p_index + 1);

	db = CLAMP(db, -60, 24);
	db_n = CLAMP(db_n, -60, 24);

	float h = 1.0 - ((db + 60) / 84.0);
	float h_n = 1.0 - ((db_n + 60) / 84.0);

	int from_x = p_x;
	int to_x = p_next_x;

	if (from_x < p_clip_left) {
		h = Math::lerp(h, h_n, float(p_clip_left - from_x) / float(to_x - from_x));
		from_x = p_clip_left;
	}

	if (to_x > p_clip_right) {
		h_n = Math::lerp(h, h_n, float(p_clip_right - from_x) / float(to_x - from_x));
		to_x = p_clip_right;
	}

	Ref<Texture2D> volume_texture = get_editor_theme_icon(SNAME("ColorTrackVu"));
	int tex_h = volume_texture->get_height();

	int y_from = (get_size().height - tex_h) / 2;

	Color color = get_theme_color(SceneStringName(font_color), SNAME("Label"));
	color.a *= REGION_EDGE_ALPHA;

	draw_line(Point2(from_x, y_from + h * tex_h), Point2(to_x, y_from + h_n * tex_h), color, 2);
}

/// AUDIO ///

AnimationTrackEditTypeAudio::AnimationTrackEditTypeAudio() {
	key_scale = 1.5;
	AudioStreamPreviewGenerator::get_singleton()->connect("preview_updated", callable_mp(this, &AnimationTrackEditTypeAudio::_preview_changed));
}

void AnimationTrackEditTypeAudio::_preview_changed(ObjectID p_which) {
	AnimationTrackEditKey::_preview_changed(p_which);
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

Ref<Resource> AnimationTrackEditTypeAudio::get_resource(const int p_index) {
	return get_animation()->audio_track_get_key_stream(get_track(), p_index);
}

float AnimationTrackEditTypeAudio::get_start_offset(const int p_index) {
	return get_animation()->audio_track_get_key_start_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAudio::get_end_offset(const int p_index) {
	return get_animation()->audio_track_get_key_end_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAudio::get_length(const int p_index) {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<AudioStream> stream = resource;
		return stream->get_length();
	}

	return AnimationTrackEditKey::get_length(p_index);
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
	AudioStreamPreviewGenerator::get_singleton()->connect("preview_updated", callable_mp(this, &AnimationTrackEditAudio::_preview_changed));
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

Ref<Resource> AnimationTrackEditAudio::get_resource(const int p_index) {
	return get_animation()->audio_track_get_key_stream(get_track(), p_index);
}

float AnimationTrackEditAudio::get_length(const int p_index) {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<AudioStream> stream = resource;
		return stream->get_length();
	}

	return AnimationTrackEditKey::get_length(p_index);
}

/// TYPE ANIMATION ///

AnimationTrackEditTypeAnimation::AnimationTrackEditTypeAnimation() {
	key_scale = 1.5;
	AnimationPreviewGenerator::get_singleton()->connect("preview_updated", callable_mp(this, &AnimationTrackEditTypeAnimation::_preview_changed));
}

void AnimationTrackEditTypeAnimation::_preview_changed(ObjectID p_which) {
	AnimationTrackEditKey::_preview_changed(p_which);
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

Ref<Resource> AnimationTrackEditTypeAnimation::get_resource(const int p_index) {
	StringName anim_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);

	if (String(anim_name) == "[stop]") {
		return nullptr;
	}

	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(get_root()->get_node_or_null(get_animation()->track_get_path(get_track())));
	if (!ap || !ap->has_animation(anim_name)) {
		return nullptr;
	}

	Ref<Animation> anim = ap->get_animation(anim_name);
	if (!anim.is_valid()) {
		return nullptr;
	}

	return anim;
}

float AnimationTrackEditTypeAnimation::get_start_offset(const int p_index) {
	return get_animation()->animation_track_get_key_start_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAnimation::get_end_offset(const int p_index) {
	return get_animation()->animation_track_get_key_end_offset(get_track(), p_index);
}

float AnimationTrackEditTypeAnimation::get_length(const int p_index) {
	Ref<Resource> resource = get_resource(p_index);
	if (resource.is_valid()) {
		Ref<Animation> anim = resource;
		return anim->get_length();
	}

	return AnimationTrackEditKey::get_length(p_index);
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

StringName AnimationTrackEditTypeAnimation::get_edit_name(const int p_index) {
	StringName edit_name = get_animation()->animation_track_get_key_animation(get_track(), p_index);
	if (!edit_name.is_empty()) {
		return edit_name;
	}

	return AnimationTrackEditKey::get_edit_name(p_index);
}

/// BASE ///

bool AnimationTrackEditKey::handle_track_resizing(const Ref<InputEventMouseMotion> mm, const float start_ofs, const float end_ofs, const float len, const int p_index, const float p_pixels_sec, const int p_x, const int p_clip_left, const int p_clip_right) {
	Vector2 region = calc_key_region(start_ofs, end_ofs, len, p_index, p_pixels_sec, p_x);
	region = clip_key_region(region, p_clip_left, p_clip_right);

	int region_begin = region.x;
	int region_end = region.y;

	if (region_begin >= p_clip_left && region_end <= p_clip_right && region_begin <= region_end) { //if (region_begin >= p_clip_left && region_end <= p_clip_right && region_begin <= p_clip_right && region_end >= p_clip_left) {
		bool resize_start = false;
		bool can_resize = false;

		float diff_left = region_begin - mm->get_position().x;
		float diff_right = mm->get_position().x - region_end;

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
			len_resizing_index = p_index;
		}

		return can_resize;
	}

	return false;
}

void AnimationTrackEditKey::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (!len_resizing && mm.is_valid()) {
		bool use_hsize_cursor = false;
		for (int p_index = 0; p_index < get_animation()->track_get_key_count(get_track()); p_index++) {
			float len = get_length(p_index);
			if (len == 0) {
				continue;
			}

			float start_ofs = get_start_offset(p_index);
			float end_ofs = get_end_offset(p_index);
			float p_x = ((get_animation()->track_get_key_time(get_track(), p_index) - get_timeline()->get_value()) * get_timeline()->get_zoom_scale()) + get_timeline()->get_name_limit();
			float p_pixels_sec = get_timeline()->get_zoom_scale();
			float p_clip_left = get_timeline()->get_name_limit();
			float p_clip_right = get_size().width - get_timeline()->get_buttons_width();

			bool can_resize = handle_track_resizing(mm, start_ofs, end_ofs, len, p_index, p_pixels_sec, p_x, p_clip_left, p_clip_right);
			if (can_resize) {
				use_hsize_cursor = true;
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
			float new_ofs = prev_ofs + ofs_local;
			float new_time = prev_time + ofs_local;
			if (prev_time != new_time) {
				undo_redo->create_action(TTR("Change Track Clip Start Offset"));
				set_start_offset(len_resizing_index, prev_ofs, new_ofs);
				undo_redo->commit_action();

				emit_signal(SNAME("move_selection_begin"));
				emit_signal(SNAME("move_selection"), ofs_local);
				emit_signal(SNAME("move_selection_commit"));
			}
		} else {
			float ofs_local = -len_resizing_rel / get_timeline()->get_zoom_scale();
			float prev_ofs = get_end_offset(len_resizing_index);
			float new_ofs = prev_ofs + ofs_local;
			if (prev_ofs != new_ofs) {
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

Control::CursorShape AnimationTrackEditKey::get_cursor_shape(const Point2 &p_pos) const {
	if (over_drag_position || len_resizing) {
		return Control::CURSOR_HSIZE;
	} else {
		return get_default_cursor_shape();
	}
}

void AnimationTrackEditKey::draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		AnimationTrackEdit::draw_key(p_index, p_pixels_sec, p_x, p_selected, p_clip_left, p_clip_right);
		return;
	}

	float start_ofs = get_start_offset(p_index);
	float end_ofs = get_end_offset(p_index);
	float len = get_length(p_index);

	float diff_start_ofs = 0;
	float diff_end_ofs = 0;

	if (len_resizing && p_index == len_resizing_index) {
		float ofs_local = len_resizing_rel / get_timeline()->get_zoom_scale();
		if (len_resizing_start) {
			diff_start_ofs = ofs_local;
		} else {
			diff_end_ofs = -ofs_local;
		}
	}

	draw_key_region(resource, start_ofs + diff_start_ofs, end_ofs + diff_end_ofs, len, p_index, p_pixels_sec, p_x + (diff_start_ofs * p_pixels_sec), p_selected, p_clip_left, p_clip_right);
}

bool AnimationTrackEditKey::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
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

void AnimationTrackEditKey::drop_data(const Point2 &p_point, const Variant &p_data) {
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

Rect2 AnimationTrackEditKey::get_key_rect(int p_index, float p_pixels_sec) {
	Ref<Resource> resource = get_resource(p_index);
	if (!resource.is_valid()) {
		return AnimationTrackEdit::get_key_rect(p_index, p_pixels_sec);
	}

	float start_ofs = get_start_offset(p_index);
	float end_ofs = get_end_offset(p_index);
	float len = get_length(p_index);

	Vector2 region = calc_key_region(start_ofs, end_ofs, len, p_index, p_pixels_sec, 0);
	int h = get_size().height;
	return Rect2(region.x, 0, region.y, h);
}

void AnimationTrackEditKey::set_node(Object *p_object) {
	id = p_object->get_instance_id();
}

StringName AnimationTrackEditKey::get_edit_name(const int p_index) {
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

Vector2 AnimationTrackEditKey::calc_key_region(const float start_ofs, const float end_ofs, const float len, const int p_index, const float p_pixels_sec, const int p_x) {
	float anim_len = len - start_ofs - end_ofs;
	if (anim_len < 0) {
		WARN_PRINT("anim_len < 0");
		anim_len = 0;
	}

	if (!len_resizing) {
		if (get_animation()->track_get_key_count(get_track()) > p_index + 1) {
			anim_len = MIN(anim_len, get_animation()->track_get_key_time(get_track(), p_index + 1) - get_animation()->track_get_key_time(get_track(), p_index));
		}
	}

	float pixel_len = anim_len * p_pixels_sec;
	float pixel_begin = p_x;
	float pixel_end = p_x + pixel_len;

	return Vector2(pixel_begin, pixel_end);
}

Vector2 AnimationTrackEditKey::clip_key_region(Vector2 region, int p_clip_left, int p_clip_right) {
	region.y = CLAMP(region.y, MAX(region.x, p_clip_left), p_clip_right);
	region.x = CLAMP(region.x, p_clip_left, MIN(region.y, p_clip_right));
	ERR_FAIL_COND_V_MSG(region.x > region.y, region, "Clipped region is invalid (x > y).");
	return region;
}

Vector2 AnimationTrackEditKey::calc_key_region_shift(Vector2 &orig_region, Vector2 &region) {
	return Vector2(region.x - orig_region.x, region.y - orig_region.y);
}

bool AnimationTrackEditKey::is_key_region_outside(const Vector2 &region, int p_clip_left, int p_clip_right) {
	return (region.y < p_clip_left || region.x > p_clip_right);
}

void AnimationTrackEditKey::draw_key_region(Ref<Resource> resource, float start_ofs, float end_ofs, float len, int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right) {
	Vector2 orig_region = calc_key_region(start_ofs, end_ofs, len, p_index, p_pixels_sec, p_x);

	bool is_outside = is_key_region_outside(orig_region, p_clip_left, p_clip_right);
	if (is_outside) {
		return;
	}

	Vector2 region = clip_key_region(orig_region, p_clip_left, p_clip_right);

	int region_begin = region.x;
	int region_end = region.y;
	int region_width = region.y - region.x;

	int fh = get_key_height();
	float h = get_size().height;
	float fy = (h - fh) / 2;

	Color accent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));

	if (orig_region.y - orig_region.x <= REGION_MAX_WIDTH) {
		Rect2 rect2 = Rect2(region_begin, fy, REGION_MAX_WIDTH, fh);
		draw_rect(rect2, accent_color);

		if (p_selected) {
			draw_rect(rect2, accent_color, false);
		}

		return;
	}

	Color bg_color = REGION_BG_COLOR;
	Rect2 rect = Rect2(region_begin, fy, region_width, fh);
	draw_rect(rect, bg_color);

	Vector<Vector2> points;
	points.resize(region_width * 2);

	Vector<Color> colors;
	if (p_selected) {
		colors = { bg_color.lightened(0.8) };
	} else {
		colors = { bg_color.lightened(0.2) };
	}

	Vector2 region_shift = calc_key_region_shift(orig_region, region);
	get_key_region_data(resource, points, rect, p_pixels_sec, start_ofs + region_shift.x / p_pixels_sec);

	if (!points.is_empty()) {
		draw_multiline_colors(points, colors);
	}

	Color edge_color = Color(accent_color);
	edge_color.a = REGION_EDGE_ALPHA;
	if (start_ofs > 0 && region_begin > p_clip_left) {
		draw_rect(Rect2(region_begin, rect.position.y, 1, rect.size.y), edge_color);
	}
	if (end_ofs > 0 && region_end < p_clip_right) {
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

			int f_h = int(h - font->get_height(font_size)) / 2 + font->get_ascent(font_size);
			draw_string(font, Point2(region_begin + REGION_FONT_MARGIN, f_h), make_text_clipped(edit_name, font, font_size, max_width), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, name_color);
		}
	}

	if (p_selected) {
		draw_rect(rect, accent_color, false);
	}
}

void AnimationTrackEditKey::_preview_changed(ObjectID p_which) {
	for (int p_index = 0; p_index < get_animation()->track_get_key_count(get_track()); p_index++) {
		Ref<Resource> resource = get_resource(p_index);
		if (resource.is_valid() && resource->get_instance_id() == p_which) {
			queue_redraw();
			return;
		}
	}
}

int AnimationTrackEditKey::get_key_height() const {
	Ref<Font> font = get_theme_font(SceneStringName(font), SNAME("Label"));
	int font_size = get_theme_font_size(SceneStringName(font_size), SNAME("Label"));
	return int(font->get_height(font_size) * key_scale);
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
