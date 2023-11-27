/**************************************************************************/
/*  animation_bezier_editor.cpp                                           */
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

#include "animation_bezier_editor.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/view_panner.h"
#include "scene/resources/text_line.h"

#include <limits.h>

float AnimationBezierTrackEdit::_bezier_h_to_pixel(float p_h) {
	float h = p_h;
	h = (h - timeline_v_scroll) / timeline_v_zoom;
	h = (get_size().height / 2.0) - h;
	return h;
}

void AnimationBezierTrackEdit::_draw_track(int p_track, const Color &p_color) {
	float scale = timeline->get_zoom_scale();

	int limit = timeline->get_name_limit();
	int right_limit = get_size().width;

	//selection may have altered the order of keys
	RBMap<real_t, int> key_order;

	for (int i = 0; i < animation->track_get_key_count(p_track); i++) {
		real_t ofs = animation->track_get_key_time(p_track, i);
		if (moving_selection && selection.has(IntPair(p_track, i))) {
			ofs += moving_selection_offset.x;
		}

		key_order[ofs] = i;
	}

	for (RBMap<real_t, int>::Element *E = key_order.front(); E; E = E->next()) {
		int i = E->get();

		if (!E->next()) {
			break;
		}

		int i_n = E->next()->get();

		float offset = animation->track_get_key_time(p_track, i);
		float height = animation->bezier_track_get_key_value(p_track, i);
		Vector2 out_handle = animation->bezier_track_get_key_out_handle(p_track, i);
		if (p_track == moving_handle_track && (moving_handle == -1 || moving_handle == 1) && moving_handle_key == i) {
			out_handle = moving_handle_right;
		}

		if (moving_selection && selection.has(IntPair(p_track, i))) {
			offset += moving_selection_offset.x;
			height += moving_selection_offset.y;
		}

		out_handle += Vector2(offset, height);

		float offset_n = animation->track_get_key_time(p_track, i_n);
		float height_n = animation->bezier_track_get_key_value(p_track, i_n);
		Vector2 in_handle = animation->bezier_track_get_key_in_handle(p_track, i_n);
		if (p_track == moving_handle_track && (moving_handle == -1 || moving_handle == 1) && moving_handle_key == i_n) {
			in_handle = moving_handle_left;
		}

		if (moving_selection && selection.has(IntPair(p_track, i_n))) {
			offset_n += moving_selection_offset.x;
			height_n += moving_selection_offset.y;
		}

		in_handle += Vector2(offset_n, height_n);

		Vector2 start(offset, height);
		Vector2 end(offset_n, height_n);

		int from_x = (offset - timeline->get_value()) * scale + limit;
		int point_start = from_x;
		int to_x = (offset_n - timeline->get_value()) * scale + limit;
		int point_end = to_x;

		if (from_x > right_limit) { //not visible
			continue;
		}

		if (to_x < limit) { //not visible
			continue;
		}

		from_x = MAX(from_x, limit);
		to_x = MIN(to_x, right_limit);

		Vector<Vector2> lines;

		Vector2 prev_pos;

		for (int j = from_x; j <= to_x; j++) {
			float t = (j - limit) / scale + timeline->get_value();

			float h;

			if (j == point_end) {
				h = end.y; //make sure it always connects
			} else if (j == point_start) {
				h = start.y; //make sure it always connects
			} else { //custom interpolation, used because it needs to show paths affected by moving the selection or handles
				int iterations = 10;
				float low = 0;
				float high = 1;

				//narrow high and low as much as possible
				for (int k = 0; k < iterations; k++) {
					float middle = (low + high) / 2.0;

					Vector2 interp = start.bezier_interpolate(out_handle, in_handle, end, middle);

					if (interp.x < t) {
						low = middle;
					} else {
						high = middle;
					}
				}

				//interpolate the result:
				Vector2 low_pos = start.bezier_interpolate(out_handle, in_handle, end, low);
				Vector2 high_pos = start.bezier_interpolate(out_handle, in_handle, end, high);

				float c = (t - low_pos.x) / (high_pos.x - low_pos.x);

				h = low_pos.lerp(high_pos, c).y;
			}

			h = _bezier_h_to_pixel(h);

			Vector2 pos(j, h);

			if (j > from_x) {
				lines.push_back(prev_pos);
				lines.push_back(pos);
			}
			prev_pos = pos;
		}

		if (lines.size() >= 2) {
			draw_multiline(lines, p_color, Math::round(EDSCALE));
		}
	}
}

void AnimationBezierTrackEdit::_draw_line_clipped(const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, int p_clip_left, int p_clip_right) {
	Vector2 from = p_from;
	Vector2 to = p_to;

	if (from.x == to.x && from.y == to.y) {
		return;
	}
	if (to.x < from.x) {
		SWAP(to, from);
	}

	if (to.x < p_clip_left) {
		return;
	}

	if (from.x > p_clip_right) {
		return;
	}

	if (to.x > p_clip_right) {
		float c = (p_clip_right - from.x) / (to.x - from.x);
		to = from.lerp(to, c);
	}

	if (from.x < p_clip_left) {
		float c = (p_clip_left - from.x) / (to.x - from.x);
		from = from.lerp(to, c);
	}

	draw_line(from, to, p_color, Math::round(EDSCALE));
}

void AnimationBezierTrackEdit::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			panner->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/animation_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EDITOR_GET("editors/panning/simple_panning")));
		} break;

		case NOTIFICATION_ENTER_TREE: {
			panner->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/animation_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EDITOR_GET("editors/panning/simple_panning")));
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			bezier_icon = get_editor_theme_icon(SNAME("KeyBezierPoint"));
			bezier_handle_icon = get_editor_theme_icon(SNAME("KeyBezierHandle"));
			selected_icon = get_editor_theme_icon(SNAME("KeyBezierSelected"));
		} break;

		case NOTIFICATION_DRAW: {
			if (animation.is_null()) {
				return;
			}

			int limit = timeline->get_name_limit();

			if (has_focus()) {
				Color accent = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
				accent.a *= 0.7;
				draw_rect(Rect2(Point2(), get_size()), accent, false, Math::round(EDSCALE));
			}

			Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
			int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
			Color color = get_theme_color(SNAME("font_color"), SNAME("Label"));
			int hsep = get_theme_constant(SNAME("h_separation"), SNAME("ItemList"));
			int vsep = get_theme_constant(SNAME("v_separation"), SNAME("ItemList"));
			Color linecolor = color;
			linecolor.a = 0.2;

			draw_line(Point2(limit, 0), Point2(limit, get_size().height), linecolor, Math::round(EDSCALE));

			int right_limit = get_size().width;

			track_v_scroll_max = vsep;

			int vofs = vsep + track_v_scroll;
			int margin = 0;

			RBMap<int, Color> subtrack_colors;
			Color selected_track_color;
			subtracks.clear();
			subtrack_icons.clear();

			RBMap<String, Vector<int>> track_indices;
			int track_count = animation->get_track_count();
			for (int i = 0; i < track_count; ++i) {
				if (animation->track_get_type(i) != Animation::TrackType::TYPE_BEZIER) {
					continue;
				}

				String base_path = animation->track_get_path(i);
				if (is_filtered) {
					if (root && root->has_node(base_path)) {
						Node *node = root->get_node(base_path);
						if (!node) {
							continue; // No node, no filter.
						}
						if (!EditorNode::get_singleton()->get_editor_selection()->is_selected(node)) {
							continue; // Skip track due to not selected.
						}
					}
				}

				int end = base_path.find(":");
				if (end != -1) {
					base_path = base_path.substr(0, end + 1);
				}
				Vector<int> indices = track_indices.has(base_path) ? track_indices[base_path] : Vector<int>();
				indices.push_back(i);
				track_indices[base_path] = indices;
			}

			for (const KeyValue<String, Vector<int>> &E : track_indices) {
				String base_path = E.key;

				Vector<int> tracks = E.value;

				// NAMES AND ICON
				{
					NodePath path = animation->track_get_path(tracks[0]);

					Node *node = nullptr;

					if (root && root->has_node(path)) {
						node = root->get_node(path);
					}

					String text;

					if (node) {
						int ofs = 0;

						Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(node, "Node");

						text = node->get_name();
						ofs += hsep;

						TextLine text_buf = TextLine(text, font, font_size);
						text_buf.set_width(limit - ofs - icon->get_width() - hsep);

						int h = MAX(text_buf.get_size().y, icon->get_height());

						draw_texture(icon, Point2(ofs, vofs + int(h - icon->get_height()) / 2.0));
						ofs += icon->get_width();

						margin = icon->get_width();

						Vector2 string_pos = Point2(ofs, vofs);
						string_pos = string_pos.floor();
						text_buf.draw(get_canvas_item(), string_pos, color);

						vofs += h + vsep;
						track_v_scroll_max += h + vsep;
					}
				}

				Color dc = get_theme_color(SNAME("disabled_font_color"), EditorStringName(Editor));

				Ref<Texture2D> remove = get_editor_theme_icon(SNAME("Remove"));
				float remove_hpos = limit - hsep - remove->get_width();

				Ref<Texture2D> lock = get_editor_theme_icon(SNAME("Lock"));
				Ref<Texture2D> unlock = get_editor_theme_icon(SNAME("Unlock"));
				float lock_hpos = remove_hpos - hsep - lock->get_width();

				Ref<Texture2D> visibility_visible = get_editor_theme_icon(SNAME("GuiVisibilityVisible"));
				Ref<Texture2D> visibility_hidden = get_editor_theme_icon(SNAME("GuiVisibilityHidden"));
				float visibility_hpos = lock_hpos - hsep - visibility_visible->get_width();

				Ref<Texture2D> solo = get_editor_theme_icon(SNAME("AudioBusSolo"));
				float solo_hpos = visibility_hpos - hsep - solo->get_width();

				float buttons_width = remove->get_width() + lock->get_width() + visibility_visible->get_width() + solo->get_width() + hsep * 3;

				for (int i = 0; i < tracks.size(); ++i) {
					// RELATED TRACKS TITLES

					int current_track = tracks[i];

					String path = animation->track_get_path(current_track);
					path = path.replace_first(base_path, "");

					Color cc = color;
					TextLine text_buf = TextLine(path, font, font_size);
					text_buf.set_width(limit - margin - buttons_width);

					Rect2 rect = Rect2(margin, vofs, solo_hpos - hsep - solo->get_width(), text_buf.get_size().y + vsep);

					cc.a *= 0.7;
					float h;
					if (path.ends_with(":x")) {
						h = 0;
					} else if (path.ends_with(":y")) {
						h = 0.33f;
					} else if (path.ends_with(":z")) {
						h = 0.66f;
					} else {
						uint32_t hash = path.hash();
						hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
						hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
						hash = (hash >> 16) ^ hash;
						h = (hash % 65535) / 65536.0;
					}

					if (current_track != selected_track) {
						Color track_color;
						if (locked_tracks.has(current_track)) {
							track_color.set_hsv(h, 0, 0.4);
						} else {
							track_color.set_hsv(h, 0.2, 0.8);
						}
						track_color.a = 0.5;
						draw_rect(Rect2(0, vofs, margin - hsep, text_buf.get_size().y * 0.8), track_color);
						subtrack_colors[current_track] = track_color;

						subtracks[current_track] = rect;
					} else {
						Color ac = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
						ac.a = 0.5;
						draw_rect(rect, ac);
						if (locked_tracks.has(selected_track)) {
							selected_track_color.set_hsv(h, 0.0, 0.4);
						} else {
							selected_track_color.set_hsv(h, 0.8, 0.8);
						}
					}

					Vector2 string_pos = Point2(margin, vofs);
					text_buf.draw(get_canvas_item(), string_pos, cc);

					float icon_start_height = vofs + rect.size.y / 2.0;
					Rect2 remove_rect = Rect2(remove_hpos, icon_start_height - remove->get_height() / 2.0, remove->get_width(), remove->get_height());
					if (read_only) {
						draw_texture(remove, remove_rect.position, dc);
					} else {
						draw_texture(remove, remove_rect.position);
					}

					Rect2 lock_rect = Rect2(lock_hpos, icon_start_height - lock->get_height() / 2.0, lock->get_width(), lock->get_height());
					if (locked_tracks.has(current_track)) {
						draw_texture(lock, lock_rect.position);
					} else {
						draw_texture(unlock, lock_rect.position);
					}

					Rect2 visible_rect = Rect2(visibility_hpos, icon_start_height - visibility_visible->get_height() / 2.0, visibility_visible->get_width(), visibility_visible->get_height());
					if (hidden_tracks.has(current_track)) {
						draw_texture(visibility_hidden, visible_rect.position);
					} else {
						draw_texture(visibility_visible, visible_rect.position);
					}

					Rect2 solo_rect = Rect2(solo_hpos, icon_start_height - solo->get_height() / 2.0, solo->get_width(), solo->get_height());
					draw_texture(solo, solo_rect.position);

					RBMap<int, Rect2> track_icons;
					track_icons[REMOVE_ICON] = remove_rect;
					track_icons[LOCK_ICON] = lock_rect;
					track_icons[VISIBILITY_ICON] = visible_rect;
					track_icons[SOLO_ICON] = solo_rect;

					subtrack_icons[current_track] = track_icons;

					vofs += text_buf.get_size().y + vsep;
					track_v_scroll_max += text_buf.get_size().y + vsep;
				}
			}

			Color accent = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));

			{ //guides
				float min_left_scale = font->get_height(font_size) + vsep;

				float scale = (min_left_scale * 2) * timeline_v_zoom;
				float step = Math::pow(10.0, Math::round(Math::log(scale / 5.0) / Math::log(10.0))) * 5.0;
				scale = Math::snapped(scale, step);

				while (scale / timeline_v_zoom < min_left_scale * 2) {
					scale += step;
				}

				bool first = true;
				int prev_iv = 0;
				for (int i = font->get_height(font_size); i < get_size().height; i++) {
					float ofs = get_size().height / 2.0 - i;
					ofs *= timeline_v_zoom;
					ofs += timeline_v_scroll;

					int iv = int(ofs / scale);
					if (ofs < 0) {
						iv -= 1;
					}
					if (!first && iv != prev_iv) {
						Color lc = linecolor;
						lc.a *= 0.5;
						draw_line(Point2(limit, i), Point2(right_limit, i), lc, Math::round(EDSCALE));
						Color c = color;
						c.a *= 0.5;
						draw_string(font, Point2(limit + 8, i - 2), TS->format_number(rtos(Math::snapped((iv + 1) * scale, step))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, c);
					}

					first = false;
					prev_iv = iv;
				}
			}

			{ //draw OTHER curves

				float scale = timeline->get_zoom_scale();
				Ref<Texture2D> point = get_editor_theme_icon(SNAME("KeyValue"));
				for (const KeyValue<int, Color> &E : subtrack_colors) {
					if (hidden_tracks.has(E.key)) {
						continue;
					}
					_draw_track(E.key, E.value);

					for (int i = 0; i < animation->track_get_key_count(E.key); i++) {
						float offset = animation->track_get_key_time(E.key, i);
						float value = animation->bezier_track_get_key_value(E.key, i);

						Vector2 pos((offset - timeline->get_value()) * scale + limit, _bezier_h_to_pixel(value));

						if (pos.x >= limit && pos.x <= right_limit) {
							draw_texture(point, pos - point->get_size() / 2.0, E.value);
						}
					}
				}

				if (track_count > 0 && !hidden_tracks.has(selected_track)) {
					//draw edited curve
					_draw_track(selected_track, selected_track_color);
				}
			}

			//draw editor handles
			{
				edit_points.clear();
				float scale = timeline->get_zoom_scale();

				for (int i = 0; i < track_count; ++i) {
					if (animation->track_get_type(i) != Animation::TrackType::TYPE_BEZIER || hidden_tracks.has(i)) {
						continue;
					}

					if (hidden_tracks.has(i) || locked_tracks.has(i)) {
						continue;
					}

					int key_count = animation->track_get_key_count(i);
					String path = animation->track_get_path(i);

					if (is_filtered) {
						if (root && root->has_node(path)) {
							Node *node = root->get_node(path);
							if (!node) {
								continue; // No node, no filter.
							}
							if (!EditorNode::get_singleton()->get_editor_selection()->is_selected(node)) {
								continue; // Skip track due to not selected.
							}
						}
					}

					for (int j = 0; j < key_count; ++j) {
						float offset = animation->track_get_key_time(i, j);
						float value = animation->bezier_track_get_key_value(i, j);

						if (moving_selection && selection.has(IntPair(i, j))) {
							offset += moving_selection_offset.x;
							value += moving_selection_offset.y;
						}

						Vector2 pos((offset - timeline->get_value()) * scale + limit, _bezier_h_to_pixel(value));

						Vector2 in_vec = animation->bezier_track_get_key_in_handle(i, j);

						if ((moving_handle == 1 || moving_handle == -1) && moving_handle_track == i && moving_handle_key == j) {
							in_vec = moving_handle_left;
						}
						Vector2 pos_in(((offset + in_vec.x) - timeline->get_value()) * scale + limit, _bezier_h_to_pixel(value + in_vec.y));

						Vector2 out_vec = animation->bezier_track_get_key_out_handle(i, j);

						if ((moving_handle == 1 || moving_handle == -1) && moving_handle_track == i && moving_handle_key == j) {
							out_vec = moving_handle_right;
						}

						Vector2 pos_out(((offset + out_vec.x) - timeline->get_value()) * scale + limit, _bezier_h_to_pixel(value + out_vec.y));

						if (i == selected_track || selection.has(IntPair(i, j))) {
							_draw_line_clipped(pos, pos_in, accent, limit, right_limit);
							_draw_line_clipped(pos, pos_out, accent, limit, right_limit);
						}

						EditPoint ep;
						ep.track = i;
						ep.key = j;
						if (pos.x >= limit && pos.x <= right_limit) {
							ep.point_rect.position = (pos - bezier_icon->get_size() / 2.0).floor();
							ep.point_rect.size = bezier_icon->get_size();
							if (selection.has(IntPair(i, j))) {
								draw_texture(selected_icon, ep.point_rect.position);
								draw_string(font, ep.point_rect.position + Vector2(8, -font->get_height(font_size) - 8), TTR("Time:") + " " + TS->format_number(rtos(Math::snapped(offset, 0.0001))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, accent);
								draw_string(font, ep.point_rect.position + Vector2(8, -8), TTR("Value:") + " " + TS->format_number(rtos(Math::snapped(value, 0.001))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, accent);
							} else {
								Color track_color = Color(1, 1, 1, 1);
								if (i != selected_track) {
									track_color = subtrack_colors[i];
								}
								draw_texture(bezier_icon, ep.point_rect.position, track_color);
							}
							ep.point_rect = ep.point_rect.grow(ep.point_rect.size.width * 0.5);
						}
						ep.point_rect = ep.point_rect.grow(ep.point_rect.size.width * 0.5);

						if (i == selected_track || selection.has(IntPair(i, j))) {
							if (animation->bezier_track_get_key_handle_mode(i, j) != Animation::HANDLE_MODE_LINEAR) {
								if (pos_in.x >= limit && pos_in.x <= right_limit) {
									ep.in_rect.position = (pos_in - bezier_handle_icon->get_size() / 2.0).floor();
									ep.in_rect.size = bezier_handle_icon->get_size();
									draw_texture(bezier_handle_icon, ep.in_rect.position);
									ep.in_rect = ep.in_rect.grow(ep.in_rect.size.width * 0.5);
								}
								if (pos_out.x >= limit && pos_out.x <= right_limit) {
									ep.out_rect.position = (pos_out - bezier_handle_icon->get_size() / 2.0).floor();
									ep.out_rect.size = bezier_handle_icon->get_size();
									draw_texture(bezier_handle_icon, ep.out_rect.position);
									ep.out_rect = ep.out_rect.grow(ep.out_rect.size.width * 0.5);
								}
							}
						}
						if (!locked_tracks.has(i)) {
							edit_points.push_back(ep);
						}
					}
				}

				for (int i = 0; i < edit_points.size(); ++i) {
					if (edit_points[i].track == selected_track) {
						EditPoint ep = edit_points[i];
						edit_points.remove_at(i);
						edit_points.insert(0, ep);
					}
				}
			}

			if (box_selecting) {
				Vector2 bs_from = box_selection_from;
				Vector2 bs_to = box_selection_to;
				if (bs_from.x > bs_to.x) {
					SWAP(bs_from.x, bs_to.x);
				}
				if (bs_from.y > bs_to.y) {
					SWAP(bs_from.y, bs_to.y);
				}
				draw_rect(
						Rect2(bs_from, bs_to - bs_from),
						get_theme_color(SNAME("box_selection_fill_color"), EditorStringName(Editor)));
				draw_rect(
						Rect2(bs_from, bs_to - bs_from),
						get_theme_color(SNAME("box_selection_stroke_color"), EditorStringName(Editor)),
						false,
						Math::round(EDSCALE));
			}
		} break;
	}
}

Ref<Animation> AnimationBezierTrackEdit::get_animation() const {
	return animation;
}

void AnimationBezierTrackEdit::set_animation_and_track(const Ref<Animation> &p_animation, int p_track, bool p_read_only) {
	animation = p_animation;
	read_only = p_read_only;
	selected_track = p_track;
	queue_redraw();
}

Size2 AnimationBezierTrackEdit::get_minimum_size() const {
	return Vector2(1, 1);
}

void AnimationBezierTrackEdit::set_timeline(AnimationTimelineEdit *p_timeline) {
	timeline = p_timeline;
	timeline->connect("zoom_changed", callable_mp(this, &AnimationBezierTrackEdit::_zoom_changed));
	timeline->connect("name_limit_changed", callable_mp(this, &AnimationBezierTrackEdit::_zoom_changed));
}

void AnimationBezierTrackEdit::set_editor(AnimationTrackEditor *p_editor) {
	editor = p_editor;
	connect("clear_selection", callable_mp(editor, &AnimationTrackEditor::_clear_selection).bind(false));
	connect("select_key", callable_mp(editor, &AnimationTrackEditor::_key_selected), CONNECT_DEFERRED);
}

void AnimationBezierTrackEdit::_play_position_draw() {
	if (!animation.is_valid() || play_position_pos < 0) {
		return;
	}

	float scale = timeline->get_zoom_scale();
	int h = get_size().height;

	int limit = timeline->get_name_limit();

	int px = (-timeline->get_value() + play_position_pos) * scale + limit;

	if (px >= limit && px < (get_size().width)) {
		Color color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		play_position->draw_line(Point2(px, 0), Point2(px, h), color, Math::round(2 * EDSCALE));
	}
}

void AnimationBezierTrackEdit::set_play_position(real_t p_pos) {
	play_position_pos = p_pos;
	play_position->queue_redraw();
}

void AnimationBezierTrackEdit::update_play_position() {
	play_position->queue_redraw();
}

void AnimationBezierTrackEdit::set_root(Node *p_root) {
	root = p_root;
}

void AnimationBezierTrackEdit::set_filtered(bool p_filtered) {
	is_filtered = p_filtered;
	if (animation == nullptr) {
		return;
	}
	String base_path = animation->track_get_path(selected_track);
	if (is_filtered) {
		if (root && root->has_node(base_path)) {
			Node *node = root->get_node(base_path);
			if (!node || !EditorNode::get_singleton()->get_editor_selection()->is_selected(node)) {
				for (int i = 0; i < animation->get_track_count(); ++i) {
					if (animation->track_get_type(i) != Animation::TrackType::TYPE_BEZIER) {
						continue;
					}

					base_path = animation->track_get_path(i);
					if (root && root->has_node(base_path)) {
						node = root->get_node(base_path);
						if (!node) {
							continue; // No node, no filter.
						}
						if (!EditorNode::get_singleton()->get_editor_selection()->is_selected(node)) {
							continue; // Skip track due to not selected.
						}

						set_animation_and_track(animation, i, read_only);
						break;
					}
				}
			}
		}
	}
	queue_redraw();
}

void AnimationBezierTrackEdit::_zoom_changed() {
	queue_redraw();
	play_position->queue_redraw();
}

void AnimationBezierTrackEdit::_update_locked_tracks_after(int p_track) {
	if (locked_tracks.has(p_track)) {
		locked_tracks.erase(p_track);
	}

	Vector<int> updated_locked_tracks;
	for (const int &E : locked_tracks) {
		updated_locked_tracks.push_back(E);
	}
	locked_tracks.clear();
	for (int i = 0; i < updated_locked_tracks.size(); ++i) {
		if (updated_locked_tracks[i] > p_track) {
			locked_tracks.insert(updated_locked_tracks[i] - 1);
		} else {
			locked_tracks.insert(updated_locked_tracks[i]);
		}
	}
}

void AnimationBezierTrackEdit::_update_hidden_tracks_after(int p_track) {
	if (hidden_tracks.has(p_track)) {
		hidden_tracks.erase(p_track);
	}

	Vector<int> updated_hidden_tracks;
	for (const int &E : hidden_tracks) {
		updated_hidden_tracks.push_back(E);
	}
	hidden_tracks.clear();
	for (int i = 0; i < updated_hidden_tracks.size(); ++i) {
		if (updated_hidden_tracks[i] > p_track) {
			hidden_tracks.insert(updated_hidden_tracks[i] - 1);
		} else {
			hidden_tracks.insert(updated_hidden_tracks[i]);
		}
	}
}

String AnimationBezierTrackEdit::get_tooltip(const Point2 &p_pos) const {
	return Control::get_tooltip(p_pos);
}

void AnimationBezierTrackEdit::_clear_selection() {
	selection.clear();
	emit_signal(SNAME("clear_selection"));
	queue_redraw();
}

void AnimationBezierTrackEdit::_change_selected_keys_handle_mode(Animation::HandleMode p_mode, bool p_auto) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Update Selected Key Handles"));
	for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
		const IntPair track_key_pair = E->get();
		undo_redo->add_undo_method(editor, "_bezier_track_set_key_handle_mode", animation.ptr(), track_key_pair.first, track_key_pair.second, animation->bezier_track_get_key_handle_mode(track_key_pair.first, track_key_pair.second), Animation::HANDLE_SET_MODE_NONE);
		undo_redo->add_undo_method(animation.ptr(), "bezier_track_set_key_in_handle", track_key_pair.first, track_key_pair.second, animation->bezier_track_get_key_in_handle(track_key_pair.first, track_key_pair.second));
		undo_redo->add_undo_method(animation.ptr(), "bezier_track_set_key_out_handle", track_key_pair.first, track_key_pair.second, animation->bezier_track_get_key_out_handle(track_key_pair.first, track_key_pair.second));
		undo_redo->add_do_method(editor, "_bezier_track_set_key_handle_mode", animation.ptr(), track_key_pair.first, track_key_pair.second, p_mode, p_auto ? Animation::HANDLE_SET_MODE_AUTO : Animation::HANDLE_SET_MODE_RESET);
	}
	undo_redo->commit_action();
}

void AnimationBezierTrackEdit::_clear_selection_for_anim(const Ref<Animation> &p_anim) {
	if (!(animation == p_anim)) {
		return;
	}
	_clear_selection();
}

void AnimationBezierTrackEdit::_select_at_anim(const Ref<Animation> &p_anim, int p_track, real_t p_pos) {
	if (!(animation == p_anim)) {
		return;
	}

	int idx = animation->track_find_key(p_track, p_pos, Animation::FIND_MODE_APPROX);
	ERR_FAIL_COND(idx < 0);

	selection.insert(IntPair(p_track, idx));
	emit_signal(SNAME("select_key"), idx, true, p_track);
	queue_redraw();
}

void AnimationBezierTrackEdit::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (panner->gui_input(p_event)) {
		accept_event();
		return;
	}

	if (p_event->is_pressed()) {
		if (ED_GET_SHORTCUT("animation_editor/duplicate_selection")->matches_event(p_event)) {
			if (!read_only) {
				duplicate_selection();
			}
			accept_event();
		}

		if (ED_GET_SHORTCUT("animation_editor/delete_selection")->matches_event(p_event)) {
			if (!read_only) {
				delete_selection();
			}
			accept_event();
		}
	}

	Ref<InputEventKey> key_press = p_event;

	if (key_press.is_valid() && key_press->is_pressed()) {
		if (ED_GET_SHORTCUT("animation_bezier_editor/focus")->matches_event(p_event)) {
			SelectionSet focused_keys;
			if (selection.is_empty()) {
				for (int i = 0; i < edit_points.size(); ++i) {
					IntPair key_pair = IntPair(edit_points[i].track, edit_points[i].key);
					focused_keys.insert(key_pair);
				}
			} else {
				for (const IntPair &E : selection) {
					focused_keys.insert(E);
					if (E.second > 0) {
						IntPair previous_key = IntPair(E.first, E.second - 1);
						focused_keys.insert(previous_key);
					}
					if (E.second < animation->track_get_key_count(E.first) - 1) {
						IntPair next_key = IntPair(E.first, E.second + 1);
						focused_keys.insert(next_key);
					}
				}
			}
			if (focused_keys.is_empty()) {
				accept_event();
				return;
			}

			real_t minimum_time = INFINITY;
			real_t maximum_time = -INFINITY;
			real_t minimum_value = INFINITY;
			real_t maximum_value = -INFINITY;

			for (const IntPair &E : selection) {
				IntPair key_pair = E;

				real_t time = animation->track_get_key_time(key_pair.first, key_pair.second);
				real_t value = animation->bezier_track_get_key_value(key_pair.first, key_pair.second);

				minimum_time = MIN(time, minimum_time);
				maximum_time = MAX(time, maximum_time);
				minimum_value = MIN(value, minimum_value);
				maximum_value = MAX(value, maximum_value);
			}

			float width = get_size().width - timeline->get_name_limit() - timeline->get_buttons_width();
			float padding = width * 0.1;
			float desired_scale = (width - padding / 2.0) / (maximum_time - minimum_time);
			minimum_time = MAX(0, minimum_time - (padding / 2.0) / desired_scale);

			float zv = Math::pow(100 / desired_scale, 0.125f);
			if (zv < 1) {
				zv = Math::pow(desired_scale / 100, 0.125f) - 1;
				zv = 1 - zv;
			}
			float zoom_value = timeline->get_zoom()->get_max() - zv;

			if (Math::is_finite(minimum_time) && Math::is_finite(maximum_time) && maximum_time - minimum_time > CMP_EPSILON) {
				timeline->get_zoom()->set_value(zoom_value);
				timeline->call_deferred("set_value", minimum_time);
			}

			if (Math::is_finite(minimum_value) && Math::is_finite(maximum_value)) {
				timeline_v_scroll = (maximum_value + minimum_value) / 2.0;
				if (maximum_value - minimum_value > CMP_EPSILON) {
					timeline_v_zoom = (maximum_value - minimum_value) / ((get_size().height - timeline->get_size().height) * 0.9);
				}
			}

			queue_redraw();
			accept_event();
			return;
		} else if (ED_GET_SHORTCUT("animation_bezier_editor/select_all_keys")->matches_event(p_event)) {
			for (int i = 0; i < edit_points.size(); ++i) {
				selection.insert(IntPair(edit_points[i].track, edit_points[i].key));
			}

			queue_redraw();
			accept_event();
			return;
		} else if (ED_GET_SHORTCUT("animation_bezier_editor/deselect_all_keys")->matches_event(p_event)) {
			selection.clear();

			queue_redraw();
			accept_event();
			return;
		}
	}

	Ref<InputEventMouseButton> mb = p_event;
	int limit = timeline->get_name_limit();
	if (mb.is_valid() && mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
		menu_insert_key = mb->get_position();
		if (menu_insert_key.x >= limit && menu_insert_key.x <= get_size().width) {
			if (!read_only) {
				Vector2 popup_pos = get_screen_position() + mb->get_position();

				menu->clear();
				menu->add_icon_item(bezier_icon, TTR("Insert Key Here"), MENU_KEY_INSERT);
				if (selection.size()) {
					menu->add_separator();
					menu->add_icon_item(get_editor_theme_icon(SNAME("Duplicate")), TTR("Duplicate Selected Key(s)"), MENU_KEY_DUPLICATE);
					menu->add_separator();
					menu->add_icon_item(get_editor_theme_icon(SNAME("Remove")), TTR("Delete Selected Key(s)"), MENU_KEY_DELETE);
					menu->add_separator();
					menu->add_icon_item(get_editor_theme_icon(SNAME("BezierHandlesFree")), TTR("Make Handles Free"), MENU_KEY_SET_HANDLE_FREE);
					menu->add_icon_item(get_editor_theme_icon(SNAME("BezierHandlesLinear")), TTR("Make Handles Linear"), MENU_KEY_SET_HANDLE_LINEAR);
					menu->add_icon_item(get_editor_theme_icon(SNAME("BezierHandlesBalanced")), TTR("Make Handles Balanced"), MENU_KEY_SET_HANDLE_BALANCED);
					menu->add_icon_item(get_editor_theme_icon(SNAME("BezierHandlesMirror")), TTR("Make Handles Mirrored"), MENU_KEY_SET_HANDLE_MIRRORED);
					menu->add_separator();
					menu->add_icon_item(get_editor_theme_icon(SNAME("BezierHandlesBalanced")), TTR("Make Handles Balanced (Auto Tangent)"), MENU_KEY_SET_HANDLE_AUTO_BALANCED);
					menu->add_icon_item(get_editor_theme_icon(SNAME("BezierHandlesMirror")), TTR("Make Handles Mirrored (Auto Tangent)"), MENU_KEY_SET_HANDLE_AUTO_MIRRORED);
				}

				if (menu->get_item_count()) {
					menu->reset_size();
					menu->set_position(popup_pos);
					menu->popup();
				}
			}
		}
	}

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		for (const KeyValue<int, Rect2> &E : subtracks) {
			if (E.value.has_point(mb->get_position())) {
				if (!locked_tracks.has(E.key) && !hidden_tracks.has(E.key)) {
					set_animation_and_track(animation, E.key, read_only);
					_clear_selection();
				}
				return;
			}
		}

		for (const KeyValue<int, RBMap<int, Rect2>> &E : subtrack_icons) {
			int track = E.key;
			RBMap<int, Rect2> track_icons = E.value;
			for (const KeyValue<int, Rect2> &I : track_icons) {
				if (I.value.has_point(mb->get_position())) {
					if (I.key == REMOVE_ICON) {
						if (!read_only) {
							EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
							undo_redo->create_action("Remove Bezier Track");

							undo_redo->add_do_method(this, "_update_locked_tracks_after", track);
							undo_redo->add_do_method(this, "_update_hidden_tracks_after", track);

							undo_redo->add_do_method(animation.ptr(), "remove_track", track);

							undo_redo->add_undo_method(animation.ptr(), "add_track", Animation::TrackType::TYPE_BEZIER, track);
							undo_redo->add_undo_method(animation.ptr(), "track_set_path", track, animation->track_get_path(track));

							for (int i = 0; i < animation->track_get_key_count(track); ++i) {
								undo_redo->add_undo_method(
										this,
										"_bezier_track_insert_key",
										track,
										animation->track_get_key_time(track, i),
										animation->bezier_track_get_key_value(track, i),
										animation->bezier_track_get_key_in_handle(track, i),
										animation->bezier_track_get_key_out_handle(track, i),
										animation->bezier_track_get_key_handle_mode(track, i));
							}

							undo_redo->commit_action();

							selected_track = CLAMP(selected_track, 0, animation->get_track_count() - 1);
						}
						return;
					} else if (I.key == LOCK_ICON) {
						if (locked_tracks.has(track)) {
							locked_tracks.erase(track);
						} else {
							locked_tracks.insert(track);
							if (selected_track == track) {
								for (int i = 0; i < animation->get_track_count(); ++i) {
									if (!locked_tracks.has(i) && animation->track_get_type(i) == Animation::TrackType::TYPE_BEZIER) {
										set_animation_and_track(animation, i, read_only);
										break;
									}
								}
							}
						}
						queue_redraw();
						return;
					} else if (I.key == VISIBILITY_ICON) {
						if (hidden_tracks.has(track)) {
							hidden_tracks.erase(track);
						} else {
							hidden_tracks.insert(track);
							if (selected_track == track) {
								for (int i = 0; i < animation->get_track_count(); ++i) {
									if (!hidden_tracks.has(i) && animation->track_get_type(i) == Animation::TrackType::TYPE_BEZIER) {
										set_animation_and_track(animation, i, read_only);
										break;
									}
								}
							}
						}

						Vector<int> visible_tracks;
						for (int i = 0; i < animation->get_track_count(); ++i) {
							if (!hidden_tracks.has(i) && animation->track_get_type(i) == Animation::TrackType::TYPE_BEZIER) {
								visible_tracks.push_back(i);
							}
						}

						if (visible_tracks.size() == 1) {
							solo_track = visible_tracks[0];
						} else {
							solo_track = -1;
						}

						queue_redraw();
						return;
					} else if (I.key == SOLO_ICON) {
						if (solo_track == track) {
							solo_track = -1;

							hidden_tracks.clear();
						} else {
							if (hidden_tracks.has(track)) {
								hidden_tracks.erase(track);
							}
							for (int i = 0; i < animation->get_track_count(); ++i) {
								if (animation->track_get_type(i) == Animation::TrackType::TYPE_BEZIER) {
									if (i != track && !hidden_tracks.has(i)) {
										hidden_tracks.insert(i);
									}
								}
							}

							set_animation_and_track(animation, track, read_only);
							solo_track = track;
						}
						queue_redraw();
						return;
					}
					return;
				}
			}
		}

		for (int i = 0; i < edit_points.size(); i++) {
			//first check point
			//command makes it ignore the main point, so control point editors can be force-edited
			//path 2D editing in the 3D and 2D editors works the same way
			if (!mb->is_command_or_control_pressed()) {
				if (edit_points[i].point_rect.has_point(mb->get_position())) {
					IntPair pair = IntPair(edit_points[i].track, edit_points[i].key);
					if (mb->is_shift_pressed()) {
						//add to selection
						if (selection.has(pair)) {
							selection.erase(pair);
						} else {
							selection.insert(pair);
						}
						queue_redraw();
						select_single_attempt = IntPair(-1, -1);
					} else if (selection.has(pair)) {
						moving_selection_attempt = true;
						moving_selection = false;
						moving_selection_from_key = pair.second;
						moving_selection_from_track = pair.first;
						moving_handle_track = pair.first;
						moving_handle_left = animation->bezier_track_get_key_in_handle(pair.first, pair.second);
						moving_handle_right = animation->bezier_track_get_key_out_handle(pair.first, pair.second);
						moving_selection_offset = Vector2();
						select_single_attempt = pair;
						queue_redraw();
					} else {
						moving_selection_attempt = true;
						moving_selection = true;
						moving_selection_from_key = pair.second;
						moving_selection_from_track = pair.first;
						moving_selection_offset = Vector2();
						moving_handle_track = pair.first;
						moving_handle_left = animation->bezier_track_get_key_in_handle(pair.first, pair.second);
						moving_handle_right = animation->bezier_track_get_key_out_handle(pair.first, pair.second);
						selection.clear();
						selection.insert(pair);
						set_animation_and_track(animation, pair.first, read_only);
					}
					return;
				}
			}

			if (!read_only) {
				if (edit_points[i].in_rect.has_point(mb->get_position())) {
					moving_handle = -1;
					moving_handle_key = edit_points[i].key;
					moving_handle_track = edit_points[i].track;
					moving_handle_left = animation->bezier_track_get_key_in_handle(edit_points[i].track, edit_points[i].key);
					moving_handle_right = animation->bezier_track_get_key_out_handle(edit_points[i].track, edit_points[i].key);
					queue_redraw();
					return;
				}

				if (edit_points[i].out_rect.has_point(mb->get_position())) {
					moving_handle = 1;
					moving_handle_key = edit_points[i].key;
					moving_handle_track = edit_points[i].track;
					moving_handle_left = animation->bezier_track_get_key_in_handle(edit_points[i].track, edit_points[i].key);
					moving_handle_right = animation->bezier_track_get_key_out_handle(edit_points[i].track, edit_points[i].key);
					queue_redraw();
					return;
				}
			}
		}

		//insert new point
		if (mb->get_position().x >= limit && mb->get_position().x < get_size().width && mb->is_command_or_control_pressed()) {
			Array new_point;
			new_point.resize(5);

			float h = (get_size().height / 2.0 - mb->get_position().y) * timeline_v_zoom + timeline_v_scroll;

			new_point[0] = h;
			new_point[1] = -0.25;
			new_point[2] = 0;
			new_point[3] = 0.25;
			new_point[4] = 0;

			real_t time = ((mb->get_position().x - limit) / timeline->get_zoom_scale()) + timeline->get_value();
			while (animation->track_find_key(selected_track, time, Animation::FIND_MODE_APPROX) != -1) {
				time += 0.0001;
			}

			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Add Bezier Point"));
			undo_redo->add_do_method(animation.ptr(), "bezier_track_insert_key", selected_track, time, new_point);
			undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_time", selected_track, time);
			undo_redo->commit_action();

			//then attempt to move
			int index = animation->track_find_key(selected_track, time, Animation::FIND_MODE_APPROX);
			ERR_FAIL_COND(index == -1);
			_clear_selection();
			selection.insert(IntPair(selected_track, index));

			moving_selection_attempt = true;
			moving_selection = false;
			moving_selection_from_key = index;
			moving_selection_from_track = selected_track;
			moving_selection_offset = Vector2();
			select_single_attempt = IntPair(-1, -1);
			queue_redraw();

			return;
		}

		//box select
		if (mb->get_position().x >= limit && mb->get_position().x < get_size().width) {
			box_selecting_attempt = true;
			box_selecting = false;
			box_selecting_add = false;
			box_selection_from = mb->get_position();
			return;
		}
	}

	if (box_selecting_attempt && mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		if (box_selecting) {
			//do actual select
			if (!box_selecting_add) {
				_clear_selection();
			}

			Vector2 bs_from = box_selection_from;
			Vector2 bs_to = box_selection_to;
			if (bs_from.x > bs_to.x) {
				SWAP(bs_from.x, bs_to.x);
			}
			if (bs_from.y > bs_to.y) {
				SWAP(bs_from.y, bs_to.y);
			}
			Rect2 selection_rect(bs_from, bs_to - bs_from);

			bool track_set = false;
			for (int i = 0; i < edit_points.size(); i++) {
				if (edit_points[i].point_rect.intersects(selection_rect)) {
					selection.insert(IntPair(edit_points[i].track, edit_points[i].key));
					if (!track_set) {
						track_set = true;
						set_animation_and_track(animation, edit_points[i].track, read_only);
					}
				}
			}
		} else {
			_clear_selection(); //clicked and nothing happened, so clear the selection

			//select by clicking on curve
			int track_count = animation->get_track_count();

			real_t animation_length = animation->get_length();
			animation->set_length(real_t(INT_MAX)); //bezier_track_interpolate doesn't find keys if they exist beyond anim length

			real_t time = ((mb->get_position().x - limit) / timeline->get_zoom_scale()) + timeline->get_value();

			for (int i = 0; i < track_count; ++i) {
				if (animation->track_get_type(i) != Animation::TrackType::TYPE_BEZIER || hidden_tracks.has(i) || locked_tracks.has(i)) {
					continue;
				}

				float track_h = animation->bezier_track_interpolate(i, time);
				float track_height = _bezier_h_to_pixel(track_h);

				if (abs(mb->get_position().y - track_height) < 10) {
					set_animation_and_track(animation, i, read_only);
					break;
				}
			}

			animation->set_length(animation_length);
		}

		box_selecting_attempt = false;
		box_selecting = false;
		queue_redraw();
	}

	if (moving_selection_attempt && mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		if (!read_only) {
			if (moving_selection) {
				//combit it

				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				undo_redo->create_action(TTR("Move Bezier Points"));

				List<AnimMoveRestore> to_restore;
				List<Animation::HandleMode> to_restore_handle_modes;
				// 1-remove the keys
				for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
					undo_redo->add_do_method(animation.ptr(), "track_remove_key", E->get().first, E->get().second);
				}
				// 2- remove overlapped keys
				for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
					real_t newtime = editor->snap_time(animation->track_get_key_time(E->get().first, E->get().second) + moving_selection_offset.x);

					int idx = animation->track_find_key(E->get().first, newtime, Animation::FIND_MODE_APPROX);
					if (idx == -1) {
						continue;
					}

					if (selection.has(IntPair(E->get().first, idx))) {
						continue; //already in selection, don't save
					}

					undo_redo->add_do_method(animation.ptr(), "track_remove_key_at_time", E->get().first, newtime);
					AnimMoveRestore amr;

					amr.key = animation->track_get_key_value(E->get().first, idx);
					amr.track = E->get().first;
					amr.time = newtime;

					to_restore.push_back(amr);
					to_restore_handle_modes.push_back(animation->bezier_track_get_key_handle_mode(E->get().first, idx));
				}

				// 3-move the keys (re insert them)
				for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
					real_t newpos = editor->snap_time(animation->track_get_key_time(E->get().first, E->get().second) + moving_selection_offset.x);
					Array key = animation->track_get_key_value(E->get().first, E->get().second);
					real_t h = key[0];
					h += moving_selection_offset.y;
					key[0] = h;
					undo_redo->add_do_method(
							this,
							"_bezier_track_insert_key",
							E->get().first,
							newpos,
							key[0],
							Vector2(key[1], key[2]),
							Vector2(key[3], key[4]),
							animation->bezier_track_get_key_handle_mode(E->get().first, E->get().second));
				}

				// 4-(undo) remove inserted keys
				for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
					real_t newpos = editor->snap_time(animation->track_get_key_time(E->get().first, E->get().second) + moving_selection_offset.x);
					undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_time", E->get().first, newpos);
				}

				// 5-(undo) reinsert keys
				for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
					real_t oldpos = animation->track_get_key_time(E->get().first, E->get().second);
					Array key = animation->track_get_key_value(E->get().first, E->get().second);
					undo_redo->add_undo_method(
							this,
							"_bezier_track_insert_key",
							E->get().first,
							oldpos,
							key[0],
							Vector2(key[1], key[2]),
							Vector2(key[3], key[4]),
							animation->bezier_track_get_key_handle_mode(E->get().first, E->get().second));
				}

				// 6-(undo) reinsert overlapped keys
				for (int i = 0; i < to_restore.size(); i++) {
					const AnimMoveRestore &amr = to_restore[i];
					Array key = amr.key;
					undo_redo->add_undo_method(animation.ptr(), "track_insert_key", amr.track, amr.time, amr.key, 1);
					undo_redo->add_undo_method(
							this,
							"_bezier_track_insert_key",
							amr.track,
							amr.time,
							key[0],
							Vector2(key[1], key[2]),
							Vector2(key[3], key[4]),
							to_restore_handle_modes[i]);
				}

				undo_redo->add_do_method(this, "_clear_selection_for_anim", animation);
				undo_redo->add_undo_method(this, "_clear_selection_for_anim", animation);

				// 7-reselect

				for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
					real_t oldpos = animation->track_get_key_time(E->get().first, E->get().second);
					real_t newpos = editor->snap_time(oldpos + moving_selection_offset.x);

					undo_redo->add_do_method(this, "_select_at_anim", animation, E->get().first, newpos);
					undo_redo->add_undo_method(this, "_select_at_anim", animation, E->get().first, oldpos);
				}

				undo_redo->commit_action();

				moving_selection = false;
			} else if (select_single_attempt != IntPair(-1, -1)) {
				selection.clear();
				selection.insert(select_single_attempt);
				set_animation_and_track(animation, select_single_attempt.first, read_only);
			}

			moving_selection_attempt = false;
			queue_redraw();
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (moving_selection_attempt && mm.is_valid()) {
		if (!moving_selection) {
			moving_selection = true;
			select_single_attempt = IntPair(-1, -1);
		}

		float y = (get_size().height / 2.0 - mm->get_position().y) * timeline_v_zoom + timeline_v_scroll;
		float x = editor->snap_time(((mm->get_position().x - limit) / timeline->get_zoom_scale()) + timeline->get_value());

		if (!read_only) {
			moving_selection_offset = Vector2(x - animation->track_get_key_time(moving_selection_from_track, moving_selection_from_key), y - animation->bezier_track_get_key_value(moving_selection_from_track, moving_selection_from_key));
		}

		additional_moving_handle_lefts.clear();
		additional_moving_handle_rights.clear();

		queue_redraw();
	}

	if (box_selecting_attempt && mm.is_valid()) {
		if (!box_selecting) {
			box_selecting = true;
			box_selecting_add = mm->is_shift_pressed();
		}

		box_selection_to = mm->get_position();

		if (get_local_mouse_position().y < 0) {
			//avoid cursor from going too above, so it does not lose focus with viewport
			warp_mouse(Vector2(get_local_mouse_position().x, 0));
		}
		queue_redraw();
	}

	if ((moving_handle == 1 || moving_handle == -1) && mm.is_valid()) {
		float y = (get_size().height / 2.0 - mm->get_position().y) * timeline_v_zoom + timeline_v_scroll;
		float x = editor->snap_time((mm->get_position().x - timeline->get_name_limit()) / timeline->get_zoom_scale()) + timeline->get_value();

		Vector2 key_pos = Vector2(animation->track_get_key_time(selected_track, moving_handle_key), animation->bezier_track_get_key_value(selected_track, moving_handle_key));

		Vector2 moving_handle_value = Vector2(x, y) - key_pos;

		moving_handle_left = animation->bezier_track_get_key_in_handle(moving_handle_track, moving_handle_key);
		moving_handle_right = animation->bezier_track_get_key_out_handle(moving_handle_track, moving_handle_key);

		if (moving_handle == -1) {
			moving_handle_left = moving_handle_value;

			Animation::HandleMode handle_mode = animation->bezier_track_get_key_handle_mode(moving_handle_track, moving_handle_key);

			if (handle_mode == Animation::HANDLE_MODE_BALANCED) {
				real_t ratio = timeline->get_zoom_scale() * timeline_v_zoom;
				Transform2D xform;
				xform.set_scale(Vector2(1.0, 1.0 / ratio));

				Vector2 vec_out = xform.xform(moving_handle_right);
				Vector2 vec_in = xform.xform(moving_handle_left);

				moving_handle_right = xform.affine_inverse().xform(-vec_in.normalized() * vec_out.length());
			} else if (handle_mode == Animation::HANDLE_MODE_MIRRORED) {
				moving_handle_right = -moving_handle_left;
			}
		} else if (moving_handle == 1) {
			moving_handle_right = moving_handle_value;

			Animation::HandleMode handle_mode = animation->bezier_track_get_key_handle_mode(moving_handle_track, moving_handle_key);

			if (handle_mode == Animation::HANDLE_MODE_BALANCED) {
				real_t ratio = timeline->get_zoom_scale() * timeline_v_zoom;
				Transform2D xform;
				xform.set_scale(Vector2(1.0, 1.0 / ratio));

				Vector2 vec_in = xform.xform(moving_handle_left);
				Vector2 vec_out = xform.xform(moving_handle_right);

				moving_handle_left = xform.affine_inverse().xform(-vec_out.normalized() * vec_in.length());
			} else if (handle_mode == Animation::HANDLE_MODE_MIRRORED) {
				moving_handle_left = -moving_handle_right;
			}
		}
		queue_redraw();
	}

	if ((moving_handle == -1 || moving_handle == 1) && mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		if (!read_only) {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Move Bezier Points"));
			if (moving_handle == -1) {
				real_t ratio = timeline->get_zoom_scale() * timeline_v_zoom;
				undo_redo->add_do_method(animation.ptr(), "bezier_track_set_key_in_handle", moving_handle_track, moving_handle_key, moving_handle_left, ratio);
				undo_redo->add_undo_method(animation.ptr(), "bezier_track_set_key_in_handle", moving_handle_track, moving_handle_key, animation->bezier_track_get_key_in_handle(moving_handle_track, moving_handle_key), ratio);
			} else if (moving_handle == 1) {
				real_t ratio = timeline->get_zoom_scale() * timeline_v_zoom;
				undo_redo->add_do_method(animation.ptr(), "bezier_track_set_key_out_handle", moving_handle_track, moving_handle_key, moving_handle_right, ratio);
				undo_redo->add_undo_method(animation.ptr(), "bezier_track_set_key_out_handle", moving_handle_track, moving_handle_key, animation->bezier_track_get_key_out_handle(moving_handle_track, moving_handle_key), ratio);
			}
			undo_redo->commit_action();
			moving_handle = 0;
			queue_redraw();
		}
	}
}

void AnimationBezierTrackEdit::_pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event) {
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		if (mm->get_position().x > timeline->get_name_limit()) {
			timeline_v_scroll += p_scroll_vec.y * timeline_v_zoom;
			timeline_v_scroll = CLAMP(timeline_v_scroll, -100000, 100000);
			timeline->set_value(timeline->get_value() - p_scroll_vec.x / timeline->get_zoom_scale());
		} else {
			track_v_scroll += p_scroll_vec.y;
			if (track_v_scroll < -track_v_scroll_max) {
				track_v_scroll = -track_v_scroll_max;
			} else if (track_v_scroll > 0) {
				track_v_scroll = 0;
			}
		}
		queue_redraw();
	}
}

void AnimationBezierTrackEdit::_zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event) {
	const float v_zoom_orig = timeline_v_zoom;
	Ref<InputEventWithModifiers> iewm = p_event;
	if (iewm.is_valid() && iewm->is_alt_pressed()) {
		// Alternate zoom (doesn't affect timeline).
		timeline_v_zoom = CLAMP(timeline_v_zoom * p_zoom_factor, 0.000001, 100000);
	} else {
		timeline->get_zoom()->set_value(timeline->get_zoom()->get_value() / p_zoom_factor);
	}
	timeline_v_scroll = timeline_v_scroll + (p_origin.y - get_size().y / 2.0) * (timeline_v_zoom - v_zoom_orig);
	queue_redraw();
}

void AnimationBezierTrackEdit::_menu_selected(int p_index) {
	switch (p_index) {
		case MENU_KEY_INSERT: {
			if (animation->get_track_count() > 0) {
				Array new_point;
				new_point.resize(5);

				float h = (get_size().height / 2.0 - menu_insert_key.y) * timeline_v_zoom + timeline_v_scroll;

				new_point[0] = h;
				new_point[1] = -0.25;
				new_point[2] = 0;
				new_point[3] = 0.25;
				new_point[4] = 0;

				int limit = timeline->get_name_limit();

				real_t time = ((menu_insert_key.x - limit) / timeline->get_zoom_scale()) + timeline->get_value();

				while (animation->track_find_key(selected_track, time, Animation::FIND_MODE_APPROX) != -1) {
					time += 0.001;
				}

				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				undo_redo->create_action(TTR("Add Bezier Point"));
				undo_redo->add_do_method(animation.ptr(), "track_insert_key", selected_track, time, new_point);
				undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_time", selected_track, time);
				undo_redo->commit_action();
				queue_redraw();
			}
		} break;
		case MENU_KEY_DUPLICATE: {
			duplicate_selection();
		} break;
		case MENU_KEY_DELETE: {
			delete_selection();
		} break;
		case MENU_KEY_SET_HANDLE_FREE: {
			_change_selected_keys_handle_mode(Animation::HANDLE_MODE_FREE);
		} break;
		case MENU_KEY_SET_HANDLE_LINEAR: {
			_change_selected_keys_handle_mode(Animation::HANDLE_MODE_LINEAR);
		} break;
		case MENU_KEY_SET_HANDLE_BALANCED: {
			_change_selected_keys_handle_mode(Animation::HANDLE_MODE_BALANCED);
		} break;
		case MENU_KEY_SET_HANDLE_MIRRORED: {
			_change_selected_keys_handle_mode(Animation::HANDLE_MODE_MIRRORED);
		} break;
		case MENU_KEY_SET_HANDLE_AUTO_BALANCED: {
			_change_selected_keys_handle_mode(Animation::HANDLE_MODE_BALANCED, true);
		} break;
		case MENU_KEY_SET_HANDLE_AUTO_MIRRORED: {
			_change_selected_keys_handle_mode(Animation::HANDLE_MODE_MIRRORED, true);
		} break;
	}
}

void AnimationBezierTrackEdit::duplicate_selection() {
	if (selection.size() == 0) {
		return;
	}

	real_t top_time = 1e10;
	for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
		real_t t = animation->track_get_key_time(E->get().first, E->get().second);
		if (t < top_time) {
			top_time = t;
		}
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Animation Duplicate Keys"));

	List<Pair<int, real_t>> new_selection_values;

	for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
		real_t t = animation->track_get_key_time(E->get().first, E->get().second);
		real_t dst_time = t + (timeline->get_play_position() - top_time);
		int existing_idx = animation->track_find_key(E->get().first, dst_time, Animation::FIND_MODE_APPROX);

		undo_redo->add_do_method(animation.ptr(), "track_insert_key", E->get().first, dst_time, animation->track_get_key_value(E->get().first, E->get().second), animation->track_get_key_transition(E->get().first, E->get().second));
		undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_time", E->get().first, dst_time);

		Pair<int, real_t> p;
		p.first = E->get().first;
		p.second = dst_time;
		new_selection_values.push_back(p);

		if (existing_idx != -1) {
			undo_redo->add_undo_method(animation.ptr(), "track_insert_key", E->get().first, dst_time, animation->track_get_key_value(E->get().first, existing_idx), animation->track_get_key_transition(E->get().first, existing_idx));
		}
	}

	undo_redo->commit_action();

	//reselect duplicated

	selection.clear();
	for (const Pair<int, real_t> &E : new_selection_values) {
		int track = E.first;
		real_t time = E.second;

		int existing_idx = animation->track_find_key(track, time, Animation::FIND_MODE_APPROX);

		if (existing_idx == -1) {
			continue;
		}

		selection.insert(IntPair(track, existing_idx));
	}

	queue_redraw();
}

void AnimationBezierTrackEdit::delete_selection() {
	if (selection.size()) {
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Animation Delete Keys"));

		for (SelectionSet::Element *E = selection.back(); E; E = E->prev()) {
			undo_redo->add_do_method(animation.ptr(), "track_remove_key", E->get().first, E->get().second);
			undo_redo->add_undo_method(animation.ptr(), "track_insert_key", E->get().first, animation->track_get_key_time(E->get().first, E->get().second), animation->track_get_key_value(E->get().first, E->get().second), 1);
		}
		undo_redo->add_do_method(this, "_clear_selection_for_anim", animation);
		undo_redo->add_undo_method(this, "_clear_selection_for_anim", animation);
		undo_redo->commit_action();

		//selection.clear();
	}
}

void AnimationBezierTrackEdit::_bezier_track_insert_key(int p_track, double p_time, real_t p_value, const Vector2 &p_in_handle, const Vector2 &p_out_handle, const Animation::HandleMode p_handle_mode) {
	ERR_FAIL_COND(animation.is_null());
	int idx = animation->bezier_track_insert_key(p_track, p_time, p_value, p_in_handle, p_out_handle);
	animation->bezier_track_set_key_handle_mode(p_track, idx, p_handle_mode);
}

void AnimationBezierTrackEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_clear_selection"), &AnimationBezierTrackEdit::_clear_selection);
	ClassDB::bind_method(D_METHOD("_clear_selection_for_anim"), &AnimationBezierTrackEdit::_clear_selection_for_anim);
	ClassDB::bind_method(D_METHOD("_select_at_anim"), &AnimationBezierTrackEdit::_select_at_anim);
	ClassDB::bind_method(D_METHOD("_update_hidden_tracks_after"), &AnimationBezierTrackEdit::_update_hidden_tracks_after);
	ClassDB::bind_method(D_METHOD("_update_locked_tracks_after"), &AnimationBezierTrackEdit::_update_locked_tracks_after);
	ClassDB::bind_method(D_METHOD("_bezier_track_insert_key"), &AnimationBezierTrackEdit::_bezier_track_insert_key);

	ADD_SIGNAL(MethodInfo("timeline_changed", PropertyInfo(Variant::FLOAT, "position")));
	ADD_SIGNAL(MethodInfo("remove_request", PropertyInfo(Variant::INT, "track")));
	ADD_SIGNAL(MethodInfo("insert_key", PropertyInfo(Variant::FLOAT, "offset")));
	ADD_SIGNAL(MethodInfo("select_key", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::BOOL, "single"), PropertyInfo(Variant::INT, "track")));
	ADD_SIGNAL(MethodInfo("clear_selection"));
	ADD_SIGNAL(MethodInfo("close_request"));

	ADD_SIGNAL(MethodInfo("move_selection_begin"));
	ADD_SIGNAL(MethodInfo("move_selection", PropertyInfo(Variant::FLOAT, "offset")));
	ADD_SIGNAL(MethodInfo("move_selection_commit"));
	ADD_SIGNAL(MethodInfo("move_selection_cancel"));
}

AnimationBezierTrackEdit::AnimationBezierTrackEdit() {
	panner.instantiate();
	panner->set_callbacks(callable_mp(this, &AnimationBezierTrackEdit::_pan_callback), callable_mp(this, &AnimationBezierTrackEdit::_zoom_callback));

	play_position = memnew(Control);
	play_position->set_mouse_filter(MOUSE_FILTER_PASS);
	add_child(play_position);
	play_position->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	play_position->connect("draw", callable_mp(this, &AnimationBezierTrackEdit::_play_position_draw));
	set_focus_mode(FOCUS_CLICK);

	set_clip_contents(true);

	ED_SHORTCUT("animation_bezier_editor/focus", TTR("Focus"), Key::F);
	ED_SHORTCUT("animation_bezier_editor/select_all_keys", TTR("Select All Keys"), KeyModifierMask::CMD_OR_CTRL | Key::A);
	ED_SHORTCUT("animation_bezier_editor/deselect_all_keys", TTR("Deselect All Keys"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::A);

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("id_pressed", callable_mp(this, &AnimationBezierTrackEdit::_menu_selected));
}
