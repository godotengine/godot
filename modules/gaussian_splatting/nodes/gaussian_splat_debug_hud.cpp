#include "gaussian_splat_debug_hud.h"
#include "gaussian_splat_node_3d.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "scene/resources/font.h"
#include "scene/theme/theme_db.h"

void GaussianSplatDebugHUD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_corner", "corner"), &GaussianSplatDebugHUD::set_corner);
	ClassDB::bind_method(D_METHOD("get_corner"), &GaussianSplatDebugHUD::get_corner);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "corner", PROPERTY_HINT_ENUM, "Top Left,Top Right,Bottom Left,Bottom Right"), "set_corner", "get_corner");

	ClassDB::bind_method(D_METHOD("set_update_interval", "interval"), &GaussianSplatDebugHUD::set_update_interval);
	ClassDB::bind_method(D_METHOD("get_update_interval"), &GaussianSplatDebugHUD::get_update_interval);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "update_interval", PROPERTY_HINT_RANGE, "0.016,1.0,0.001"), "set_update_interval", "get_update_interval");

	ClassDB::bind_method(D_METHOD("set_font_size", "size"), &GaussianSplatDebugHUD::set_font_size);
	ClassDB::bind_method(D_METHOD("get_font_size"), &GaussianSplatDebugHUD::get_font_size);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "font_size", PROPERTY_HINT_RANGE, "8,32,1"), "set_font_size", "get_font_size");

	ClassDB::bind_method(D_METHOD("set_background_color", "color"), &GaussianSplatDebugHUD::set_background_color);
	ClassDB::bind_method(D_METHOD("get_background_color"), &GaussianSplatDebugHUD::get_background_color);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "background_color"), "set_background_color", "get_background_color");

	ClassDB::bind_method(D_METHOD("refresh_stats"), &GaussianSplatDebugHUD::refresh_stats);

	BIND_ENUM_CONSTANT(CORNER_TOP_LEFT);
	BIND_ENUM_CONSTANT(CORNER_TOP_RIGHT);
	BIND_ENUM_CONSTANT(CORNER_BOTTOM_LEFT);
	BIND_ENUM_CONSTANT(CORNER_BOTTOM_RIGHT);
}

GaussianSplatDebugHUD::GaussianSplatDebugHUD() {
	set_mouse_filter(MOUSE_FILTER_IGNORE);
	set_process(true);
}

GaussianSplatDebugHUD::~GaussianSplatDebugHUD() {
}

void GaussianSplatDebugHUD::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Try to get default font from theme
			if (!hud_font.is_valid()) {
				hud_font = ThemeDB::get_singleton()->get_fallback_font();
			}
		} break;

		case NOTIFICATION_PROCESS: {
			float delta = get_process_delta_time();

			time_since_update += delta;
			if (time_since_update >= update_interval) {
				time_since_update = 0.0f;
				_update_cached_stats();
				queue_redraw();
			}
		} break;

		case NOTIFICATION_DRAW: {
			_draw_hud();
		} break;
	}
}

void GaussianSplatDebugHUD::_update_cached_stats() {
	cached_hud_lines.clear();

	if (!splat_node) {
		return;
	}

	Ref<GaussianSplatRenderer> renderer = splat_node->get_renderer();
	if (!renderer.is_valid()) {
		return;
	}

	Dictionary stats = renderer->get_render_stats();
	Array hud_lines = stats.get(StringName("performance_hud_lines"), Array());
	if (hud_lines.is_empty()) {
		const bool show_performance_hud = stats.get(StringName("debug_show_performance_hud"), false);
		const bool show_residency_hud = stats.get(StringName("debug_show_residency_hud"), false);
		if (show_performance_hud || show_residency_hud) {
			cached_hud_lines.push_back("Collecting HUD metrics...");
		}
		return;
	}

	cached_hud_lines.resize(hud_lines.size());
	String *lines_write = cached_hud_lines.ptrw();
	for (int i = 0; i < hud_lines.size(); i++) {
		const Variant &line = hud_lines[i];
		lines_write[i] = line.get_type() == Variant::STRING ? String(line) : String(line.stringify());
	}
}

void GaussianSplatDebugHUD::_draw_hud() {
	if (!hud_font.is_valid() || cached_hud_lines.is_empty()) {
		return;
	}

	const Vector<String> &lines = cached_hud_lines;
	const int line_count = lines.size();
	line_size_scratch.resize(line_count);

	// Calculate HUD size
	float max_width = 0.0f;
	float total_height = 0.0f;
	for (int i = 0; i < line_count; i++) {
		const Vector2 text_size = hud_font->get_string_size(lines[i], HORIZONTAL_ALIGNMENT_LEFT, -1, font_size);
		line_size_scratch[i] = text_size;
		max_width = MAX(max_width, text_size.x);
		total_height += text_size.y;
		if (i + 1 < line_count) {
			total_height += line_spacing;
		}
	}

	Vector2 hud_size = Vector2(max_width + padding * 2, total_height + padding * 2);
	Vector2 hud_pos = _calculate_hud_position(hud_size);

	// Draw background
	Rect2 bg_rect(hud_pos, hud_size);
	draw_rect(bg_rect, background_color);

	// Draw border
	draw_rect(bg_rect, Color(0.3f, 0.3f, 0.3f, 0.8f), false, 1.0f);

	// Draw text lines
	float y_offset = hud_pos.y + padding;
	for (int i = 0; i < line_count; i++) {
		const String &line = lines[i];
		const Vector2 &text_size = line_size_scratch[i];
		Color color = text_color;
		if (i == 0) {
			color = highlight_color;
		} else if (line.begins_with("IO Error")) {
			color = warning_color;
		}
		Vector2 text_pos(hud_pos.x + padding, y_offset + text_size.y * 0.8f);
		draw_string(hud_font, text_pos, line, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color);
		y_offset += text_size.y + line_spacing;
	}
}

Vector2 GaussianSplatDebugHUD::_calculate_hud_position(const Vector2 &p_hud_size) const {
	Vector2 viewport_size = get_viewport_rect().size;

	switch (corner) {
		case CORNER_TOP_LEFT:
			return Vector2(margin, margin);
		case CORNER_TOP_RIGHT:
			return Vector2(viewport_size.x - p_hud_size.x - margin, margin);
		case CORNER_BOTTOM_LEFT:
			return Vector2(margin, viewport_size.y - p_hud_size.y - margin);
		case CORNER_BOTTOM_RIGHT:
			return Vector2(viewport_size.x - p_hud_size.x - margin, viewport_size.y - p_hud_size.y - margin);
		default:
			return Vector2(margin, margin);
	}
}

void GaussianSplatDebugHUD::set_splat_node(GaussianSplatNode3D *p_node) {
	splat_node = p_node;
	_update_cached_stats();
	queue_redraw();
}

void GaussianSplatDebugHUD::set_corner(Corner p_corner) {
	if (corner != p_corner) {
		corner = p_corner;
		queue_redraw();
	}
}

void GaussianSplatDebugHUD::set_update_interval(float p_interval) {
	update_interval = CLAMP(p_interval, 0.016f, 1.0f);
}

void GaussianSplatDebugHUD::set_font_size(int p_size) {
	font_size = CLAMP(p_size, 8, 32);
	queue_redraw();
}

void GaussianSplatDebugHUD::set_background_color(const Color &p_color) {
	background_color = p_color;
	queue_redraw();
}

void GaussianSplatDebugHUD::refresh_stats() {
	_update_cached_stats();
	queue_redraw();
}
