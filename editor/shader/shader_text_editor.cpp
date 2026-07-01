/**************************************************************************/
/*  shader_text_editor.cpp                                                */
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

#include "shader_text_editor.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "editor/docks/inspector_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/scene/material_editor_plugin.h"
#include "editor/script/script_editor_base.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/syntax_highlighters.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/shader.h"
#include "scene/resources/sky.h"
#include "scene/resources/style_box_flat.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/shader_preprocessor.h"
#include "servers/rendering/shader_types.h"

#include "modules/regex/regex.h"

/*** SHADER PREVIEW LINE LAYER ****/

void TextShaderPreviewLineLayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			line_color = Color(EditorNode::get_singleton()->get_editor_theme()->get_color(SceneStringName(font_color), EditorStringName(Editor)), 0.7);
		} break;
		case NOTIFICATION_DRAW: {
			int total_gutter_width = code_editor->get_line_start_margin();
			for (int i = 0; i < code_editor->get_gutter_count() - 1; i++) {
				if (code_editor->is_gutter_drawn(i)) {
					total_gutter_width += code_editor->get_gutter_width(i);
				}
			}

			const Rect2i visible_rect = scroll_container->get_global_rect();
			RenderingServer::get_singleton()->canvas_item_set_custom_rect(
					get_canvas_item(), true,
					Rect2(get_global_transform().affine_inverse().xform(Vector2(visible_rect.position)), visible_rect.size + Vector2(code_editor->get_total_gutter_width(), 0)));
			RenderingServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), true);

			int current_caret_line = code_editor->get_caret_line();
			int idx = 0;
			for (const KeyValue<int, TextShaderPreview *> &E : *previews) {
				idx++;

				const Control *panel_container = E.value->get_panel_container();
				float alpha = (E.key == current_caret_line) ? 0.2 : (E.value->is_hovered() ? 0.15 : (idx % 2 == 0 ? 0.1 : 0.05));
				const Color polygon_color = Color(line_color, alpha);
				E.value->update_panel_color(polygon_color);

				Point2i end_pos = code_editor->get_pos_at_line_column(E.key, 0);
				if (end_pos.x == -1) {
					continue;
				}

				Point2 preview_size = panel_container->get_size();
				const Point2 preview_bottom = panel_container->get_global_position() + preview_size;
				const Point2 preview_top = preview_bottom - Point2(0, preview_size.y);
				const Point2 line_bottom = code_editor->get_global_position() + Point2(0, end_pos.y);
				const Point2 line_top = line_bottom - Point2(0, code_editor->get_line_height());
				const Point2 line_bottom_gutter = line_bottom + Point2(total_gutter_width, 0);
				const Point2 line_top_gutter = line_top + Point2(total_gutter_width, 0);

				draw_polygon({ preview_top, line_top, line_top_gutter, line_bottom_gutter, line_bottom, preview_bottom }, { polygon_color, polygon_color, polygon_color, polygon_color, polygon_color, polygon_color });
			}
		} break;
	}
}

void TextShaderPreviewLineLayer::set_previews(HashMap<int, TextShaderPreview *> &p_previews) {
	previews = &p_previews;
}

void TextShaderPreviewLineLayer::set_code_editor(CodeEdit *p_code_editor) {
	code_editor = p_code_editor;
	code_editor->connect("caret_changed", callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw));
}

void TextShaderPreviewLineLayer::set_scroll_container(ScrollContainer *p_scroll_container) {
	scroll_container = p_scroll_container;
}

TextShaderPreviewLineLayer::TextShaderPreviewLineLayer() {
	set_as_top_level(true);
}

/***  SHADER PREVIEW ****/

class SquareMarginContainer : public MarginContainer {
public:
	Size2 get_minimum_size() const override {
		Size2 ms = MarginContainer::get_minimum_size();
		float side = MAX(get_size().x, ms.y);
		return Size2(ms.x, side);
	}
};

HashMap<String, String> TextShaderPreview::spatial_assignments = {
	{ "bool", "ALBEDO = vec3(float(%s)); ALPHA = 1.0;" },
	{ "int", "ALBEDO = vec3(float(%s)); ALPHA = 1.0;" },
	{ "float", "ALBEDO = vec3(%s); ALPHA = 1.0;" },
	{ "vec2", "ALBEDO = vec3(%s.rg, 0.0); ALPHA = 1.0;" },
	{ "vec3", "ALBEDO = %s; ALPHA = 1.0;" },
	{ "vec4", "vec4 __sp_v4 = %s; ALBEDO = __sp_v4.rgb; ALPHA = __sp_v4.a;" },
};

HashMap<String, String> TextShaderPreview::canvas_assignments = {
	{ "bool", "COLOR = vec4(vec3(float(%s)), 1.0);" },
	{ "int", "COLOR = vec4(vec3(float(%s)), 1.0);" },
	{ "float", "COLOR = vec4(vec3(%s), 1.0);" },
	{ "vec2", "COLOR = vec4(%s, 0.0, 1.0);" },
	{ "vec3", "COLOR = vec4(%s, 1.0);" },
	{ "vec4", "COLOR = %s;" },
};

HashMap<String, String> TextShaderPreview::builtin_spatial_types = {
	{ "NORMAL_MAP_DEPTH", "float" },
	{ "DEPTH", "float" },
	{ "ALPHA", "float" },
	{ "ALPHA_SCISSOR_THRESHOLD", "float" },
	{ "ALPHA_HASH_SCALE", "float" },
	{ "ALPHA_ANTIALIASING_EDGE", "float" },
	{ "PREMUL_ALPHA_FACTOR", "float" },
	{ "METALLIC", "float" },
	{ "SPECULAR", "float" },
	{ "ROUGHNESS", "float" },
	{ "RIM", "float" },
	{ "RIM_TINT", "float" },
	{ "CLEARCOAT", "float" },
	{ "CLEARCOAT_ROUGHNESS", "float" },
	{ "ANISOTROPY", "float" },
	{ "SSS_STRENGTH", "float" },
	{ "SSS_TRANSMITTANCE_DEPTH", "float" },
	{ "SSS_TRANSMITTANCE_BOOST", "float" },
	{ "AO", "float" },
	{ "AO_LIGHT_AFFECT", "float" },

	{ "ALPHA_TEXTURE_COORDINATE", "vec2" },
	{ "ANISOTROPY_FLOW", "vec2" },

	{ "NORMAL", "vec3" },
	{ "NORMAL_MAP", "vec3" },
	{ "LIGHT_VERTEX", "vec3" },
	{ "TANGENT", "vec3" },
	{ "BINORMAL", "vec3" },
	{ "ALBEDO", "vec3" },
	{ "BACKLIGHT", "vec3" },
	{ "EMISSION", "vec3" },
	{ "BENT_NORMAL_MAP", "vec3" },

	{ "FOG", "vec4" },
	{ "RADIANCE", "vec4" },
	{ "IRRADIANCE", "vec4" },
	{ "SSS_TRANSMITTANCE_COLOR", "vec4" },
};

HashMap<String, String> TextShaderPreview::builtin_canvas_types = {
	{ "NORMAL_MAP_DEPTH", "float" },

	{ "SHADOW_VERTEX", "vec2" },
	{ "VERTEX", "vec2" },

	{ "NORMAL", "vec3" },
	{ "NORMAL_MAP", "vec3" },
	{ "LIGHT_VERTEX", "vec3" },

	{ "COLOR", "vec4" },
};

TextShaderPreview::TextShaderPreview() {
	env.instantiate();
	Ref<Sky> sky = memnew(Sky);
	env->set_sky(sky);
	env->set_background(Environment::BG_COLOR);
	env->set_ambient_source(Environment::AMBIENT_SOURCE_SKY);
	env->set_reflection_source(Environment::REFLECTION_SOURCE_SKY);

	shader_material.instantiate();

	panel = memnew(PanelContainer);
	panel->set_custom_minimum_size(Size2(120, BUTTON_SIZE * 3) * EDSCALE); // Three buttons tall.
	panel->connect(SceneStringName(mouse_entered), callable_mp(this, &TextShaderPreview::_on_hover_enter));
	panel->connect(SceneStringName(mouse_exited), callable_mp(this, &TextShaderPreview::_on_hover_exit));
	add_child(panel);

	panel_style.instantiate();

	Control *fill = memnew(Control);
	panel->add_child(fill);

	surface_hover = memnew(Control);
	surface_hover->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	surface_hover->set_size(Size2(BUTTON_SIZE, 0) * EDSCALE);
	surface_hover->connect(SceneStringName(mouse_entered), callable_mp(this, &TextShaderPreview::_on_hover_enter));
	surface_hover->connect(SceneStringName(mouse_exited), callable_mp(this, &TextShaderPreview::_on_hover_exit));
	fill->add_child(surface_hover);

	delete_button = memnew(Button);
	delete_button->set_theme_type_variation("FlatMenuButton");
	delete_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	delete_button->set_custom_minimum_size(Size2(BUTTON_SIZE, BUTTON_SIZE) * EDSCALE);
	delete_button->set_anchors_and_offsets_preset(Control::PRESET_TOP_RIGHT);
	delete_button->set_anchor_and_offset(SIDE_LEFT, 1.0, -BUTTON_SIZE * EDSCALE);
	delete_button->set_anchor_and_offset(SIDE_TOP, 0.0, 0);
	delete_button->set_anchor_and_offset(SIDE_RIGHT, 1.0, 0);
	delete_button->set_anchor_and_offset(SIDE_BOTTOM, 0.0, BUTTON_SIZE * EDSCALE);
	delete_button->connect(SceneStringName(pressed), callable_mp(this, &TextShaderPreview::_delete_pressed));
	delete_button->hide();
	delete_button->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	surface_hover->add_child(delete_button);

	goto_button = memnew(Button);
	goto_button->set_theme_type_variation("FlatMenuButton");
	goto_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	goto_button->set_custom_minimum_size(Size2(BUTTON_SIZE, BUTTON_SIZE) * EDSCALE);
	goto_button->set_anchors_and_offsets_preset(Control::PRESET_CENTER_RIGHT);
	goto_button->set_anchor_and_offset(SIDE_LEFT, 1.0, -BUTTON_SIZE * EDSCALE);
	goto_button->set_anchor_and_offset(SIDE_TOP, 0.5, -BUTTON_SIZE / 2 * EDSCALE);
	goto_button->set_anchor_and_offset(SIDE_RIGHT, 1.0, 0);
	goto_button->set_anchor_and_offset(SIDE_BOTTOM, 0.5, BUTTON_SIZE / 2 * EDSCALE);
	goto_button->connect(SceneStringName(pressed), callable_mp(this, &TextShaderPreview::_goto_pressed));
	goto_button->hide();
	goto_button->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	surface_hover->add_child(goto_button);

	// Material

	surface_container = memnew(SquareMarginContainer);
	surface_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	panel->add_child(surface_container);

	surface = memnew(MaterialEditor);
	surface->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	surface->set_autohide_buttons(true);
	surface_container->add_child(surface);

	// Error

	error_container = memnew(MarginContainer);
	error_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	panel->add_child(error_container);

	VBoxContainer *error_box = memnew(VBoxContainer);
	error_box->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	error_container->add_child(error_box);

	error_icon = memnew(TextureRect);
	error_icon->set_stretch_mode(TextureRect::STRETCH_KEEP);
	error_box->add_child(error_icon);

	error_label = memnew(Label);
	error_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	error_label->set_v_size_flags(SIZE_EXPAND_FILL);
	error_label->add_theme_constant_override("line_spacing", 0);
	error_box->add_child(error_label);
}

void TextShaderPreview::_on_hover_enter() {
	hovered = true;
	delete_button->show();
	goto_button->show();

	DisplayServer::get_singleton()->cursor_set_shape(DisplayServerEnums::CURSOR_ARROW); // Since MaterialEditor doesn't set cursor.
}

void TextShaderPreview::_on_hover_exit() {
	hovered = false;
	delete_button->hide();
	goto_button->hide();
}

void TextShaderPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			panel->add_theme_style_override(SceneStringName(panel), panel_style);

			error_icon->set_texture(get_editor_theme_icon(SNAME("Error")));

			Ref<StyleBoxEmpty> no_margins;
			no_margins.instantiate();
			no_margins->set_content_margin_all(0);
			error_label->add_theme_style_override(CoreStringName(normal), no_margins);
			error_label->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));

			delete_button->set_button_icon(get_editor_theme_icon(SNAME("GuiClose")));

			goto_button->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("source"), EditorStringName(EditorFonts)));
			goto_button->add_theme_font_size_override(SceneStringName(font_size), Math::round(12 * EDSCALE));

			Ref<StyleBoxFlat> sq_hover = get_theme_stylebox(SceneStringName(hover), "FlatMenuButton")->duplicate();
			sq_hover->set_corner_radius_all(0);
			Ref<StyleBoxFlat> sq_pressed = get_theme_stylebox(SceneStringName(pressed), "FlatMenuButton")->duplicate();
			sq_pressed->set_corner_radius_all(0);
			Ref<StyleBoxFlat> sq_hover_pressed = get_theme_stylebox("hover_pressed", "FlatMenuButton")->duplicate();
			sq_hover_pressed->set_corner_radius_all(0);

			for (Button *btn : { delete_button, goto_button }) {
				btn->add_theme_style_override(SceneStringName(hover), sq_hover);
				btn->add_theme_style_override(SceneStringName(pressed), sq_pressed);
				btn->add_theme_style_override("hover_pressed", sq_hover_pressed);
			}

			surface_container->add_theme_constant_override("margin_left", 0 * EDSCALE);
			surface_container->add_theme_constant_override("margin_right", 0 * EDSCALE);
			surface_container->add_theme_constant_override("margin_top", 0 * EDSCALE);
			surface_container->add_theme_constant_override("margin_bottom", 0 * EDSCALE);

			error_container->add_theme_constant_override("margin_left", 12 * EDSCALE);
			error_container->add_theme_constant_override("margin_right", 12 * EDSCALE);
			error_container->add_theme_constant_override("margin_top", 12 * EDSCALE);
			error_container->add_theme_constant_override("margin_bottom", 12 * EDSCALE);
		} break;
	}
}

void TextShaderPreview::_bind_methods() {
	ADD_SIGNAL(MethodInfo("goto_btn_pressed"));
	ADD_SIGNAL(MethodInfo("remove_btn_pressed"));
}

void TextShaderPreview::_goto_pressed() {
	emit_signal("goto_btn_pressed");
}

void TextShaderPreview::_delete_pressed() {
	emit_signal("remove_btn_pressed");
}

String TextShaderPreview::_get_enclosing_function(const PackedStringArray &p_lines, int p_line) const {
	int brace_stack = 0;

	Ref<RegEx> regex;
	regex.instantiate();
	regex->compile(R"(void\s+(\w+)\s*\()");

	for (int i = p_line; i >= 0; i--) {
		// Strip comments and trailing whitespace.
		String clean_line = p_lines[i].split("//")[0].strip_edges();
		if (clean_line.is_empty()) {
			continue;
		}

		brace_stack += clean_line.count("}");
		brace_stack -= clean_line.count("{");

		if (brace_stack < 0) {
			Ref<RegExMatch> m = regex->search(clean_line);
			if (m.is_valid()) {
				return m->get_string(1);
			}
		}
	}

	return String(); // Global scope.
}

bool TextShaderPreview::_is_inside_loop(const PackedStringArray &p_lines, int p_line) const {
	int brace_stack = 0;

	Ref<RegEx> loop_regex;
	loop_regex.instantiate();
	loop_regex->compile(R"(\b(for|while|do)\b)");

	Ref<RegEx> func_regex;
	func_regex.instantiate();
	func_regex->compile(R"(\b(?!for\b|while\b|do\b|if\b|else\b|return\b|switch\b)\w+\s+\w+\s*\()");

	for (int i = p_line; i >= 0; i--) {
		String clean_line = p_lines[i].split("//")[0].strip_edges();
		if (clean_line.is_empty()) {
			continue;
		}

		brace_stack += clean_line.count("}");
		brace_stack -= clean_line.count("{");

		if (brace_stack < 0) {
			if (loop_regex->search(clean_line).is_valid()) {
				return true;
			}
			if (func_regex->search(clean_line).is_valid()) {
				return false;
			}

			brace_stack = 0;
		}
	}

	return false;
}

bool TextShaderPreview::_find_statement(const PackedStringArray &p_lines, int p_line, String &r_var_name, int &r_start, int &r_end) const {
	Ref<RegEx> var_regex;
	var_regex.instantiate();
	var_regex->compile(R"(([\w.]+)\s*([+\-*/%]?=)(?!=))");

	// Walk backward from the caret to find the line with the assignment operator.
	int start = p_line;
	Ref<RegExMatch> var_match = var_regex->search(p_lines[start]);
	while (!var_match.is_valid() && start > 0) {
		String current_line = p_lines[start].strip_edges();

		if (start < p_line && (current_line.is_empty() || current_line.ends_with(";") || current_line.ends_with("{") || current_line.ends_with("}"))) {
			return false;
		}

		start -= 1;
		var_match = var_regex->search(p_lines[start]);
	}

	if (!var_match.is_valid()) {
		return false;
	}

	// Flow control selection can't be previewed.
	Ref<RegEx> flow_regex;
	flow_regex.instantiate();
	flow_regex->compile(R"(^(else\s+)?(if|while|for)\b)");
	if (flow_regex->search(p_lines[start].strip_edges()).is_valid()) {
		return false;
	}

	int end = start;
	int max_scan = MIN(start + 20, p_lines.size() - 1);
	while (end < max_scan && !p_lines[end].strip_edges().ends_with(";")) {
		end += 1;
	}

	if (!p_lines[end].strip_edges().ends_with(";")) {
		return false;
	}

	if (p_line > end) {
		return false;
	}

	String full_captured_path = var_match->get_string(1); // e.g my_vec.xy.
	r_var_name = full_captured_path.split(".")[0]; // e.g my_vec.
	r_start = start;
	r_end = end;

	return true;
}

String TextShaderPreview::_find_var_type(const PackedStringArray &p_lines, const String &p_var_name, int p_line, bool p_mode_3d) {
	HashMap<String, String> &builtin_types = p_mode_3d ? builtin_spatial_types : builtin_canvas_types;
	if (builtin_types.has(p_var_name)) {
		return builtin_types[p_var_name];
	}

	Ref<RegEx> type_regex;
	type_regex.instantiate();

	// Matches a type keyword, followed by anything except a semicolon, then the variable name.
	// This safely handles: "float my_var;" and "float a, b, my_var;"
	type_regex->compile(R"(\b(float|vec2|vec3|vec4|int|bool)\b[^(;]*\b)" + p_var_name + R"(\b)");

	// Walk backwards from the end of the assignment statement.
	for (int i = p_line; i >= 0; i--) {
		// Strip out comments before checking so we don't catch commented-out declarations.
		String clean_line = p_lines[i].split("//")[0];

		Ref<RegExMatch> m = type_regex->search(clean_line);
		if (m.is_valid()) {
			return m->get_string(1);
		}
	}

	return String();
}

bool TextShaderPreview::_match_uniforms(const Ref<ShaderMaterial> &p_source, const Ref<ShaderMaterial> &p_target) const {
	if (p_source->get_shader().is_null() || p_target->get_shader().is_null()) {
		return false;
	}

	List<PropertyInfo> source_params;
	List<PropertyInfo> target_params;

	p_source->get_shader()->get_shader_uniform_list(&source_params);
	p_target->get_shader()->get_shader_uniform_list(&target_params);

	if (source_params.size() != target_params.size()) {
		return false;
	}

	RBSet<String> target_set;
	for (const PropertyInfo &p : target_params) {
		target_set.insert(p.name + itos(p.type));
	}

	for (const PropertyInfo &p : source_params) {
		String key = p.name + itos(p.type);
		if (!target_set.has(key)) {
			return false;
		}
	}

	return true;
}

void TextShaderPreview::_sync_shader_parameters(const Ref<ShaderMaterial> &p_source, Ref<ShaderMaterial> &p_target) {
	if (p_source->get_shader().is_null()) {
		return;
	}

	List<PropertyInfo> params;
	p_source->get_shader()->get_shader_uniform_list(&params);

	for (const PropertyInfo &p : params) {
		String param_name = p.name;
		Variant param_value = p_source->get_shader_parameter(param_name);

		p_target->set_shader_parameter(param_name, param_value);
	}
}

void TextShaderPreview::_reset_shader_parameters(Ref<ShaderMaterial> &p_target) {
	List<PropertyInfo> params;
	p_target->get_shader()->get_shader_uniform_list(&params);

	for (const PropertyInfo &p : params) {
		String param_name = p.name;
		p_target->set_shader_parameter(param_name, Variant());
	}
}

void TextShaderPreview::_show_error(const String &p_error) {
	surface->edit(Ref<Material>(), env);
	error_label->set_text(p_error);
	surface_container->hide();
	error_container->show();
}

void TextShaderPreview::show_shader_compile_error() {
	_show_error(TTRC("Shader must be compiled correctly."));
}

void TextShaderPreview::recompile(const String &p_code) {
	set_shader_code(p_code, line, in_comment);
}

Ref<ShaderMaterial> TextShaderPreview::_get_source_material() const {
	const Object *object = InspectorDock::get_inspector_singleton()->get_edited_object();
	if (!object) {
		return Ref<ShaderMaterial>();
	}

	const CanvasItem *ci = Object::cast_to<CanvasItem>(object);
	if (ci) {
		const Ref<ShaderMaterial> canvas_material = ci->get_material();
		if (canvas_material.is_valid() && _match_uniforms(canvas_material, shader_material)) {
			return canvas_material;
		}

		return Ref<ShaderMaterial>();
	}

	const GeometryInstance3D *gi = Object::cast_to<GeometryInstance3D>(object);
	if (gi) {
		const Ref<ShaderMaterial> material_overlay = gi->get_material_overlay();
		if (material_overlay.is_valid() && _match_uniforms(material_overlay, shader_material)) {
			return material_overlay;
		}

		const Ref<ShaderMaterial> material_override = gi->get_material_override();
		if (material_override.is_valid() && _match_uniforms(material_override, shader_material)) {
			return material_override;
		}

		const MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(object);
		if (mi) {
			const Ref<Mesh> mesh = mi->get_mesh();

			if (mesh.is_valid()) {
				for (int i = 0; i < mesh->get_surface_count(); i++) {
					const Ref<ShaderMaterial> surface_material = mi->get_surface_override_material(i);

					if (surface_material.is_valid() && _match_uniforms(surface_material, shader_material)) {
						return surface_material;
					}
				}
			}
		}
	}

	return Ref<ShaderMaterial>();
}

void TextShaderPreview::sync_shader_parameters() {
	if (shader_material->get_shader().is_null()) {
		return;
	}
	const Ref<ShaderMaterial> src_mat = _get_source_material();
	if (src_mat.is_valid()) {
		_sync_shader_parameters(src_mat, shader_material);
	} else {
		_reset_shader_parameters(shader_material);
	}
}

void TextShaderPreview::update_panel_color(const Color &p_color) {
	panel_style->set_bg_color(p_color);
}

PanelContainer *TextShaderPreview::get_panel_container() const {
	return panel;
}

Control *TextShaderPreview::get_hover_control() const {
	return surface_hover;
}

void TextShaderPreview::set_shader_code(const String &p_code, int p_line, bool p_in_comment) {
	line = p_line;
	in_comment = p_in_comment;
	goto_button->set_text(itos(line + 1));

	String shader_type = ShaderLanguage::get_shader_type(p_code);
	bool mode_3d = shader_type == "spatial";

	if (shader_type != "canvas_item" && !mode_3d) {
		_show_error(TTRC("For previews, shader type must be CanvasItem or Spatial."));
		return;
	}

	const PackedStringArray lines = p_code.split("\n");
	String enclosing_function = _get_enclosing_function(lines, p_line);

	if (enclosing_function != "fragment") {
		_show_error(TTRC("Previewed line must be inside fragment() function."));
		return;
	}

	if (_is_inside_loop(lines, p_line)) {
		_show_error(TTRC("Preview not supported inside loops."));
		return;
	}

	String var_name;
	int start;
	int end;

	if (in_comment || !_find_statement(lines, p_line, var_name, start, end)) {
		_show_error(TTRC("Previewed line must contain an assignment."));
		return;
	}

	String type = _find_var_type(lines, var_name, end, mode_3d);

	// All code before assignment stays as it was.
	PackedStringArray truncated_lines = lines.slice(0, end + 1);

	String injection;
	HashMap<String, String> &assignments = mode_3d ? spatial_assignments : canvas_assignments;
	if (!assignments.has(type)) {
		_show_error(TTRC("Preview only available for bool, int, float, vec2, vec3, vec4."));
		return;
	}
	injection = assignments[type].replace("%s", var_name);
	truncated_lines.append(injection);

	String full_truncated_text = "\n";
	full_truncated_text = full_truncated_text.join(truncated_lines);

	int open_braces = full_truncated_text.count("{");
	int closed_braces = full_truncated_text.count("}");
	int needed_closures = open_braces - closed_braces;

	for (int i = 0; i < needed_closures; i++) {
		full_truncated_text += "\n}";
	}

	Ref<Shader> shader;
	shader.instantiate();
	shader->set_code(full_truncated_text);
	shader_material->set_shader(shader);

	const Ref<ShaderMaterial> src_mat = _get_source_material();
	if (src_mat.is_valid()) {
		_sync_shader_parameters(src_mat, shader_material);
	} else {
		_reset_shader_parameters(shader_material);
	}

	error_container->hide();
	surface_container->show();
	surface->edit(shader_material.ptr(), env);
	surface->show(); // Edit may have called hide() earlier on failed compilation.
}

/*** SHADER SCRIPT EDITOR ****/

void ShaderTextEditor::EditMenusShTE::EditMenusShTE::_update_breakpoint_list() {
	EditMenusCEB::_update_breakpoint_list();
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/goto_previous_shader_preview"), DEBUG_GOTO_PREV_BREAKPOINT);
	breakpoints_menu->set_item_index(-1, 0);
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/goto_next_shader_preview"), DEBUG_GOTO_NEXT_BREAKPOINT);
	breakpoints_menu->set_item_index(-1, 0);
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/remove_all_shader_previews"), DEBUG_REMOVE_ALL_BREAKPOINTS);
	breakpoints_menu->set_item_index(-1, 0);
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/toggle_shader_preview"), DEBUG_TOGGLE_BREAKPOINT);
	breakpoints_menu->set_item_index(-1, 0);
}

static bool saved_warnings_enabled = false;
static bool saved_treat_warning_as_errors = false;
static HashMap<ShaderWarning::Code, bool> saved_warnings;
static uint32_t saved_warning_flags = 0U;

void ShaderTextEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			code_editor->get_text_editor()->add_theme_color_override("breakpoint_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SceneStringName(font_color), EditorStringName(Editor)));
			code_editor->get_text_editor()->add_theme_icon_override("breakpoint", get_editor_theme_icon(SNAME("GuiVisibilityVisible")));

			if (is_visible_in_tree()) {
				_load_theme_settings();
				if (warnings.size() > 0 && last_compile_result == OK) {
					warnings_panel->clear();
					_update_warning_panel();
				}
			}

			Ref<StyleBoxFlat> tab_style = get_theme_stylebox(SNAME("tab_selected"), "TabBar");
			Ref<StyleBoxFlat> preview_style = memnew(StyleBoxFlat);
			preview_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), EditorStringName(Editor)));
			preview_style->set_corner_radius_all(tab_style->get_corner_radius(CORNER_TOP_LEFT));
			preview_panel->add_theme_style_override(SceneStringName(panel), preview_style);

			update_params_btn->set_button_icon(get_editor_theme_icon(SNAME("Reload")));
			remove_all_btn->set_button_icon(get_editor_theme_icon(SNAME("Remove")));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible_in_tree() && preview_timer->is_inside_tree()) {
				preview_timer->start();
			}
		} break;

		case NOTIFICATION_RESIZED: {
			preview_timer->start();
		} break;
	}
}

void ShaderTextEditor::_shader_changed() {
	// This function is used for dependencies (include changing changes main shader and forces it to revalidate)
	if (block_shader_changed) {
		return;
	}
	dependencies_changed = true;
	_validate_script();
}

void ShaderTextEditor::goto_shader_preview(int p_line) {
	goto_line_centered(p_line);
}

void ShaderTextEditor::clear_previews() {
	for (KeyValue<int, TextShaderPreview *> pair : previews) {
		pair.value->queue_free();
	}
	previews.clear();
}

void ShaderTextEditor::redraw_preview_lines() {
	preview_line_layer->queue_redraw();
}

void ShaderTextEditor::recompile_previews() {
	for (KeyValue<int, TextShaderPreview *> &E : previews) {
		E.value->recompile(code_editor->get_text_editor()->get_text());
	}
}

void ShaderTextEditor::update_parameters() {
	for (KeyValue<int, TextShaderPreview *> &E : previews) {
		E.value->sync_shader_parameters();
	}
}

TextShaderPreviewLineLayer *ShaderTextEditor::get_preview_line_layer() const {
	return preview_line_layer;
}

TextShaderPreview *ShaderTextEditor::get_preview(int p_line) const {
	if (previews.has(p_line)) {
		return previews[p_line];
	}
	return nullptr;
}

void ShaderTextEditor::toggle_shader_preview(int p_line) {
	CodeEdit *tx = code_editor->get_text_editor();

	TextShaderPreview *preview = memnew(TextShaderPreview);
	previews.insert(p_line, preview);

	if (last_compile_result != OK) {
		preview->show_shader_compile_error();
	} else {
		preview->set_shader_code(tx->get_text(), p_line, tx->is_in_comment(p_line) != -1);
	}

	preview->connect("goto_btn_pressed", callable_mp(this, &ShaderTextEditor::goto_shader_preview).bind(p_line));
	preview->connect("remove_btn_pressed", callable_mp(this, &ShaderTextEditor::remove_shader_preview).bind(p_line));
	preview->get_panel_container()->connect(SceneStringName(mouse_entered), callable_mp((CanvasItem *)preview_line_layer, &CanvasItem::queue_redraw));
	preview->get_panel_container()->connect(SceneStringName(mouse_exited), callable_mp((CanvasItem *)preview_line_layer, &CanvasItem::queue_redraw));
	preview->get_hover_control()->connect(SceneStringName(mouse_entered), callable_mp((CanvasItem *)preview_line_layer, &CanvasItem::queue_redraw));
	preview->get_hover_control()->connect(SceneStringName(mouse_exited), callable_mp((CanvasItem *)preview_line_layer, &CanvasItem::queue_redraw));
	preview_vbox->add_child(preview);
}

void ShaderTextEditor::remove_shader_preview(int p_line) {
	set_breakpoint(p_line, false);
}

void ShaderTextEditor::_load_theme_settings() {
	CodeEdit *te = code_editor->get_text_editor();
	Color updated_marked_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
	if (updated_marked_line_color != marked_line_color) {
		for (int i = 0; i < te->get_line_count(); i++) {
			if (te->get_line_background_color(i) == marked_line_color) {
				te->set_line_background_color(i, updated_marked_line_color);
			}
		}
		marked_line_color = updated_marked_line_color;
	}

	te->clear_comment_delimiters();
	te->add_comment_delimiter("/*", "*/", false);
	te->add_comment_delimiter("//", "", true);

	if (!te->has_auto_brace_completion_open_key("/*")) {
		te->add_auto_brace_completion_pair("/*", "*/");
	}
	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

void ShaderTextEditor::_check_shader_mode() {
	String type = ShaderLanguage::get_shader_type(code_editor->get_text_editor()->get_text());

	Shader::Mode mode;
	Ref<Shader> shader = edited_res;
	if (shader.is_null()) {
		return;
	}

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else if (type == "sky") {
		mode = Shader::MODE_SKY;
	} else if (type == "fog") {
		mode = Shader::MODE_FOG;
	} else if (type == "texture_blit") {
		mode = Shader::MODE_TEXTURE_BLIT;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		block_shader_changed = true;
		shader->set_code(code_editor->get_text_editor()->get_text());
		block_shader_changed = false;
		_load_theme_settings();
	}
}

static ShaderLanguage::DataType _get_global_shader_uniform_type(const StringName &p_variable) {
	RSE::GlobalShaderParameterType gvt = RS::get_singleton()->global_shader_parameter_get_type(p_variable);
	return (ShaderLanguage::DataType)RS::global_shader_uniform_type_get_shader_datatype(gvt);
}

static String complete_from_path;

static void _complete_include_paths_search(EditorFileSystemDirectory *p_efsd, List<ScriptLanguage::CodeCompletionOption> *r_options) {
	if (!p_efsd) {
		return;
	}
	for (int i = 0; i < p_efsd->get_file_count(); i++) {
		if (p_efsd->get_file_type(i) == SNAME("ShaderInclude")) {
			String path = p_efsd->get_file_path(i);
			if (path.begins_with(complete_from_path)) {
				path = path.replace_first(complete_from_path, "");
			}
			r_options->push_back(ScriptLanguage::CodeCompletionOption(path, ScriptLanguage::CODE_COMPLETION_KIND_FILE_PATH));
		}
	}
	for (int j = 0; j < p_efsd->get_subdir_count(); j++) {
		_complete_include_paths_search(p_efsd->get_subdir(j), r_options);
	}
}

static void _complete_include_paths(List<ScriptLanguage::CodeCompletionOption> *r_options) {
	_complete_include_paths_search(EditorFileSystem::get_singleton()->get_filesystem(), r_options);
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) {
	List<ScriptLanguage::CodeCompletionOption> pp_options;
	List<ScriptLanguage::CodeCompletionOption> pp_defines;
	ShaderPreprocessor preprocessor;
	String code;
	String resource_path = edited_res->get_path();
	complete_from_path = resource_path.get_base_dir();
	if (!complete_from_path.ends_with("/")) {
		complete_from_path += "/";
	}
	preprocessor.preprocess(p_code, resource_path, code, nullptr, nullptr, nullptr, nullptr, &pp_options, &pp_defines, _complete_include_paths);
	complete_from_path = String();
	if (pp_options.size()) {
		for (const ScriptLanguage::CodeCompletionOption &E : pp_options) {
			r_options->push_back(E);
		}
		return;
	}

	ShaderLanguage sl;
	String calltip;
	ShaderLanguage::ShaderCompileInfo comp_info;
	comp_info.global_shader_uniform_type_func = _get_global_shader_uniform_type;

	Ref<Shader> shader = edited_res;
	if (shader.is_valid()) {
		_check_shader_mode();
		comp_info.functions = ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(shader->get_mode()));
		comp_info.render_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(shader->get_mode()));
		comp_info.stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(shader->get_mode()));
		comp_info.shader_types = ShaderTypes::get_singleton()->get_types();
	} else {
		comp_info.is_include = true;
	}

	sl.complete(code, comp_info, r_options, calltip);
	if (sl.get_completion_type() == ShaderLanguage::COMPLETION_IDENTIFIER) {
		for (const ScriptLanguage::CodeCompletionOption &E : pp_defines) {
			r_options->push_back(E);
		}
	}
	code_editor->get_text_editor()->set_code_hint(calltip);
}

void ShaderTextEditor::_update_warning_panel() {
	int warning_count = 0;

	warnings_panel->push_table(2);
	for (const ShaderWarning &w : warnings) {
		if (warning_count == 0) {
			if (saved_treat_warning_as_errors) {
				const String message = (w.get_message() + " " + TTR("Warnings should be fixed to prevent errors.")).replace("[", "[lb]");
				const String error_text = vformat(TTR("Error at line %d:"), w.get_line()) + " " + message;

				code_editor->set_error(error_text);
				code_editor->set_error_pos(w.get_line() - 1, 0);

				code_editor->get_text_editor()->set_line_background_color(w.get_line() - 1, marked_line_color);
			}
		}

		warning_count++;
		int line = w.get_line();

		// First cell.
		warnings_panel->push_cell();
		warnings_panel->push_color(warnings_panel->get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
		if (line != -1) {
			warnings_panel->push_meta(line - 1);
			warnings_panel->add_text(vformat(TTR("Line %d (%s):"), line, w.get_name()));
			warnings_panel->pop(); // Meta goto.
		} else {
			warnings_panel->add_text(w.get_name() + ":");
		}
		warnings_panel->pop(); // Color.
		warnings_panel->pop(); // Cell.

		// Second cell.
		warnings_panel->push_cell();
		warnings_panel->add_text(w.get_message());
		warnings_panel->pop(); // Cell.
	}
	warnings_panel->pop(); // Table.

	code_editor->set_warning_count(warning_count);
}

bool ShaderTextEditor::_edit_option(int p_option) {
	CodeEdit *tx = code_editor->get_text_editor();
	tx->apply_ime();

	switch (p_option) {
		case EDIT_COMPLETE:
		case EDIT_TOGGLE_COMMENT: {
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		}
	}

	switch (p_option) {
		case EDIT_TOGGLE_COMMENT: {
			if (edited_res.is_null()) {
				return true;
			}
			code_editor->toggle_inline_comment("//");
		} break;
		case EDIT_COMPLETE: {
			tx->request_code_completion();
		} break;
		default:
			CodeEditorBase::_edit_option(p_option);
	}
	return true;
}

void ShaderTextEditor::_validate_script() {
	apply_code();

	CodeEdit *te = code_editor->get_text_editor();
	String code = te->get_text();

	ShaderPreprocessor preprocessor;
	String code_pp;
	String error_pp;
	List<ShaderPreprocessor::FilePosition> err_positions;
	List<ShaderPreprocessor::Region> regions;

	Ref<Shader> shader = edited_res;
	Ref<ShaderInclude> shader_inc = edited_res;
	if (shader.is_valid()) {
		code_editor->get_text_editor()->set_draw_breakpoints_gutter(true);
	} else if (shader_inc.is_valid()) {
		code_editor->get_text_editor()->set_draw_breakpoints_gutter(false);
	}
	String filename = edited_res->get_path();
	last_compile_result = preprocessor.preprocess(code, filename, code_pp, &error_pp, &err_positions, &regions);

	for (int i = 0; i < code_editor->get_text_editor()->get_line_count(); i++) {
		code_editor->get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
	}

	Ref<GDShaderSyntaxHighlighter> sh = code_editor->get_text_editor()->get_syntax_highlighter();
	if (sh.is_valid()) {
		sh->clear_disabled_branch_regions();
		for (const ShaderPreprocessor::Region &region : regions) {
			if (!region.enabled && filename != region.file) {
				sh->add_disabled_branch_region(Point2i(region.from_line, region.to_line));
			}
		}
		sh->emit_changed();
	}

	code_editor->set_error("");
	code_editor->set_error_count(0);

	if (last_compile_result != OK) {
		// Preprocessor error.
		ERR_FAIL_COND(err_positions.is_empty());

		String err_text;
		const int err_line = err_positions.front()->get().line;
		if (err_positions.size() == 1) {
			// Error in the main file.
			const String message = error_pp.replace("[", "[lb]");

			err_text = vformat(TTR("Error at line %d:"), err_line) + " " + message;
		} else {
			// Error in an included file.
			const String inc_file = err_positions.back()->get().file.get_file();
			const int inc_line = err_positions.back()->get().line;
			const String message = error_pp.replace("[", "[lb]");

			err_text = vformat(TTR("Error at line %d in include %s:%d:"), err_line, inc_file, inc_line) + " " + message;
			code_editor->set_error_count(err_positions.size() - 1);
		}

		code_editor->set_error(err_text);
		code_editor->set_error_pos(err_line - 1, 0);

		for (int i = 0; i < code_editor->get_text_editor()->get_line_count(); i++) {
			code_editor->get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
		}
		code_editor->get_text_editor()->set_line_background_color(err_line - 1, marked_line_color);

		code_editor->set_warning_count(0);

		for (KeyValue<int, TextShaderPreview *> pair : previews) {
			pair.value->show_shader_compile_error();
		}
	} else {
		ShaderLanguage sl;

		sl.enable_warning_checking(saved_warnings_enabled);
		uint32_t flags = saved_warning_flags;
		if (shader.is_null()) {
			if (flags & ShaderWarning::UNUSED_CONSTANT) {
				flags &= ~(ShaderWarning::UNUSED_CONSTANT);
			}
			if (flags & ShaderWarning::UNUSED_FUNCTION) {
				flags &= ~(ShaderWarning::UNUSED_FUNCTION);
			}
			if (flags & ShaderWarning::UNUSED_STRUCT) {
				flags &= ~(ShaderWarning::UNUSED_STRUCT);
			}
			if (flags & ShaderWarning::UNUSED_UNIFORM) {
				flags &= ~(ShaderWarning::UNUSED_UNIFORM);
			}
			if (flags & ShaderWarning::UNUSED_VARYING) {
				flags &= ~(ShaderWarning::UNUSED_VARYING);
			}
		}
		sl.set_warning_flags(flags);

		ShaderLanguage::ShaderCompileInfo comp_info;
		comp_info.global_shader_uniform_type_func = _get_global_shader_uniform_type;

		if (shader.is_null()) {
			comp_info.is_include = true;
		} else {
			Shader::Mode mode = shader->get_mode();
			comp_info.functions = ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(mode));
			comp_info.render_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(mode));
			comp_info.stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(mode));
			comp_info.shader_types = ShaderTypes::get_singleton()->get_types();
		}

		code = code_pp;
		//compiler error
		last_compile_result = sl.compile(code, comp_info);

		if (last_compile_result != OK) {
			Vector<ShaderLanguage::FilePosition> include_positions = sl.get_include_positions();

			String err_text;
			int err_line;
			if (include_positions.size() > 1) {
				// Error in an included file.
				err_line = include_positions[0].line;

				const String inc_file = include_positions[include_positions.size() - 1].file;
				const int inc_line = include_positions[include_positions.size() - 1].line;
				const String message = sl.get_error_text().replace("[", "[lb]");

				err_text = vformat(TTR("Error at line %d in include %s:%d:"), err_line, inc_file, inc_line) + " " + message;
				code_editor->set_error_count(include_positions.size() - 1);
			} else {
				// Error in the main file.
				err_line = sl.get_error_line();

				const String message = sl.get_error_text().replace("[", "[lb]");

				err_text = vformat(TTR("Error at line %d:"), err_line) + " " + message;
				code_editor->set_error_count(0);
			}

			code_editor->set_error(err_text);
			code_editor->set_error_pos(err_line - 1, 0);

			code_editor->get_text_editor()->set_line_background_color(err_line - 1, marked_line_color);

			for (KeyValue<int, TextShaderPreview *> pair : previews) {
				pair.value->show_shader_compile_error();
			}
		} else {
			for (KeyValue<int, TextShaderPreview *> pair : previews) {
				pair.value->set_shader_code(code, pair.key, code_editor->get_text_editor()->is_in_comment(pair.key) != -1);
			}
		}

		if (warnings.size() > 0 || last_compile_result != OK) {
			warnings_panel->clear();
		}
		warnings.clear();
		for (List<ShaderWarning>::Element *E = sl.get_warnings_ptr(); E; E = E->next()) {
			warnings.push_back(E->get());
		}
		if (warnings.size() > 0 && last_compile_result == OK) {
			warnings.sort_custom<WarningsComparator>();
			_update_warning_panel();
		} else {
			code_editor->set_warning_count(0);
		}
	}

	validation_success = last_compile_result == OK;

	TextEditorBase::_validate_script();
}

void ShaderTextEditor::goto_line_centered(int p_line, int p_column) {
	code_editor->goto_line_centered(p_line, p_column);

	TextShaderPreview *preview = get_preview(p_line);
	if (preview) {
		preview_sbox->ensure_control_visible(preview);
	}
	preview_timer->start();
}

void ShaderTextEditor::_update_warnings(bool p_validate) {
	bool changed = false;

	bool warnings_enabled = GLOBAL_GET("debug/shader_language/warnings/enable").booleanize();
	if (warnings_enabled != saved_warnings_enabled) {
		saved_warnings_enabled = warnings_enabled;
		changed = true;
	}

	bool treat_warning_as_errors = GLOBAL_GET("debug/shader_language/warnings/treat_warnings_as_errors").booleanize();
	if (treat_warning_as_errors != saved_treat_warning_as_errors) {
		saved_treat_warning_as_errors = treat_warning_as_errors;
		changed = true;
	}

	bool update_flags = false;

	for (int i = 0; i < ShaderWarning::WARNING_MAX; i++) {
		ShaderWarning::Code code = (ShaderWarning::Code)i;
		bool value = GLOBAL_GET("debug/shader_language/warnings/" + ShaderWarning::get_name_from_code(code).to_lower());

		if (saved_warnings[code] != value) {
			saved_warnings[code] = value;
			update_flags = true;
			changed = true;
		}
	}

	if (update_flags) {
		saved_warning_flags = (uint32_t)ShaderWarning::get_flags_from_codemap(saved_warnings);
	}

	Ref<Shader> shader = edited_res;
	if (p_validate && changed && code_editor && shader.is_valid()) {
		code_editor->validate_script();
	}
}

void ShaderTextEditor::set_edited_resource(const Ref<Resource> &p_res) {
	Ref<Shader> shader = p_res;
	Ref<ShaderInclude> shader_inc = p_res;
	if (shader.is_valid()) {
		set_edited_resource(p_res, shader->get_code());
	} else if (shader_inc.is_valid()) {
		set_edited_resource(p_res, shader_inc->get_code());
	}
}

void ShaderTextEditor::set_edited_resource(const Ref<Resource> &p_res, const String &p_code) {
	if (p_res.is_null() || edited_res == p_res) {
		return;
	}

	Ref<Shader> shader = p_res;
	Ref<ShaderInclude> shader_inc = p_res;
	if (shader.is_null() && shader_inc.is_null()) {
		return;
	}
	p_res->disconnect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));

	edited_res = p_res;
	_load_theme_settings();

	code_editor->get_text_editor()->set_text(p_code);
	code_editor->get_text_editor()->clear_undo_history();
	callable_mp((TextEdit *)code_editor->get_text_editor(), &TextEdit::set_h_scroll).call_deferred(0);
	callable_mp((TextEdit *)code_editor->get_text_editor(), &TextEdit::set_v_scroll).call_deferred(0);
	code_editor->get_text_editor()->tag_saved_version();

	_validate_script();
	code_editor->update_line_and_column();

	p_res->connect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));
}

void ShaderTextEditor::apply_code() {
	String editor_code = code_editor->get_text_editor()->get_text();
	Ref<Shader> shader = edited_res;
	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		if (shader_code != editor_code || dependencies_changed) {
			block_shader_changed = true;
			shader->set_code(editor_code);
			block_shader_changed = false;
			shader->set_edited(true);
		}
	}
	Ref<ShaderInclude> shader_inc = edited_res;
	if (shader_inc.is_valid()) {
		String shader_inc_code = shader_inc->get_code();
		if (shader_inc_code != editor_code || dependencies_changed) {
			block_shader_changed = true;
			shader_inc->set_code(editor_code);
			block_shader_changed = false;
			shader_inc->set_edited(true);
		}
	}

	dependencies_changed = false;
	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

ScriptEditorBase *ShaderTextEditor::create_editor(const Ref<Resource> &p_resource) {
	// Check class name for Shader to ensure it is not a VisualShader even when the visual shader module is disabled
	if (p_resource->is_class("Shader") || Object::cast_to<ShaderInclude>(*p_resource)) {
		return memnew(ShaderTextEditor);
	}
	return nullptr;
}

void ShaderTextEditor::_breakpoint_toggled(int p_line) {
	if (!pending_update_shader_previews) {
		pending_update_shader_previews = true;
		callable_mp(this, &ShaderTextEditor::_update_shader_previews).call_deferred();
	}
}

void ShaderTextEditor::_update_shader_previews() {
	pending_update_shader_previews = false;

	const CodeEdit *ce = code_editor->get_text_editor();
	clear_previews();
	bool found = false;

	for (int i = 0; i < ce->get_line_count(); i++) {
		if (ce->is_line_breakpointed(i)) {
			found = true;
			toggle_shader_preview(i);
		}
	}

	if (!found) {
		preview_box->hide();
		return;
	}
	preview_box->show();

	preview_timer->start();
}

void ShaderTextEditor::register_editor() {
	ED_SHORTCUT("shader_text_editor/toggle_shader_preview", TTRC("Toggle Shader Preview"), Key::F9);
	ED_SHORTCUT_OVERRIDE("shader_text_editor/toggle_shader_preview", "macos", KeyModifierMask::META | KeyModifierMask::SHIFT | Key::B);

	ED_SHORTCUT("shader_text_editor/remove_all_shader_previews", TTRC("Remove All Shader Previews"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::F9);
	// Using Control for these shortcuts even on macOS because Command+Comma is taken for opening Editor Settings.
	ED_SHORTCUT("shader_text_editor/goto_next_shader_preview", TTRC("Go to Next Shader Preview"), KeyModifierMask::CTRL | Key::PERIOD);
	ED_SHORTCUT("shader_text_editor/goto_previous_shader_preview", TTRC("Go to Previous Shader Preview"), KeyModifierMask::CTRL | Key::COMMA);

	ScriptEditor::register_create_script_editor_function(create_editor);
}

ShaderTextEditor::ShaderTextEditor() {
	_update_warnings(false);

	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ShaderTextEditor::_update_warnings).bind(true));

	code_editor->get_text_editor()->connect("_fold_line_updated", callable_mp(this, &ShaderTextEditor::redraw_preview_lines), CONNECT_DEFERRED);
	code_editor->get_text_editor()->connect(SceneStringName(theme_changed), callable_mp(this, &ShaderTextEditor::redraw_preview_lines), CONNECT_DEFERRED);
	code_editor->get_text_editor()->get_v_scroll_bar()->connect(SceneStringName(value_changed), callable_mp(this, &ShaderTextEditor::redraw_preview_lines).unbind(1), CONNECT_DEFERRED);

	code_editor->get_text_editor()->set_draw_executing_lines_gutter(false);

	preview_line_layer = memnew(TextShaderPreviewLineLayer);
	preview_line_layer->set_previews(previews);
	preview_line_layer->set_code_editor(code_editor->get_text_editor());

	add_child(preview_line_layer);
	HSplitContainer *main_box = memnew(HSplitContainer);
	main_box->set_h_size_flags(SIZE_EXPAND_FILL);
	main_box->set_v_size_flags(SIZE_EXPAND_FILL);
	main_box->add_theme_constant_override("separation", 0);
	main_box->connect("dragged", callable_mp(this, &ShaderTextEditor::redraw_preview_lines).unbind(1), CONNECT_DEFERRED);

	preview_box = memnew(VBoxContainer);
	preview_box->hide();

	preview_panel = memnew(PanelContainer);
	preview_panel->set_v_size_flags(SIZE_EXPAND_FILL);
	preview_box->add_child(preview_panel);

	preview_sbox = memnew(ScrollContainer);
	preview_sbox->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	preview_sbox->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_AUTO);
	preview_sbox->get_v_scroll_bar()->connect(SceneStringName(value_changed), callable_mp(this, &ShaderTextEditor::redraw_preview_lines).unbind(1), CONNECT_DEFERRED);
	get_preview_line_layer()->set_scroll_container(preview_sbox);
	preview_panel->add_child(preview_sbox);

	HSplitContainer *preview_hsplit = memnew(HSplitContainer);
	preview_hsplit->set_v_size_flags(SIZE_EXPAND_FILL);
	preview_hsplit->set_h_size_flags(SIZE_EXPAND_FILL);
	preview_hsplit->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN_COLLAPSED);
	preview_hsplit->add_theme_constant_override("minimum_grab_thickness", 8 * EDSCALE);
	preview_sbox->add_child(preview_hsplit);

	preview_vbox = memnew(VBoxContainer);
	preview_vbox->add_theme_constant_override("separation", 0);
	preview_vbox->connect(SceneStringName(sort_children), callable_mp(this, &ShaderTextEditor::redraw_preview_lines));
	preview_hsplit->add_child(preview_vbox);

	Control *extended_gutter = memnew(Control);
	extended_gutter->set_custom_minimum_size(Size2(32 * EDSCALE, 0)); // To match BUTTON_SIZE in TextShaderPreview.
	extended_gutter->set_h_size_flags(SIZE_EXPAND_FILL);
	extended_gutter->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	preview_hsplit->add_child(extended_gutter);

	HBoxContainer *tools_row = memnew(HBoxContainer);
	tools_row->set_custom_minimum_size(Size2(0, 24 * EDSCALE)); // To match code editor's toolbar.
	preview_box->add_child(tools_row);

	update_params_btn = memnew(Button);
	update_params_btn->set_text(TTRC("Update Parameters"));
	update_params_btn->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD);
	update_params_btn->set_h_size_flags(SIZE_EXPAND_FILL);
	update_params_btn->set_theme_type_variation("FlatMenuButton");
	update_params_btn->connect(SceneStringName(pressed), callable_mp(this, &ShaderTextEditor::update_parameters));
	update_params_btn->set_tooltip_text(TTRC("Update shader parameters in previews to match the values in the current ShaderMaterial inspector."));
	tools_row->add_child(update_params_btn);

	remove_all_btn = memnew(Button);
	remove_all_btn->set_theme_type_variation("FlatMenuButton");
	remove_all_btn->connect(SceneStringName(pressed), callable_mp(this, &ShaderTextEditor::_edit_option).bind(DEBUG_REMOVE_ALL_BREAKPOINTS));
	remove_all_btn->set_tooltip_text(TTRC("Remove all line shader previews."));
	tools_row->add_child(remove_all_btn);

	main_box->add_child(preview_box);
	main_box->add_child(code_editor);

	editor_box->add_child(main_box);
	editor_box->add_child(warnings_panel);

	preview_timer = memnew(Timer);
	add_child(preview_timer);
	preview_timer->set_one_shot(true);
	preview_timer->set_wait_time(0.001);
	preview_timer->connect("timeout", callable_mp(this, &ShaderTextEditor::redraw_preview_lines));

	InspectorDock::get_inspector_singleton()->connect(SNAME("edited_object_changed"), callable_mp(this, &ShaderTextEditor::recompile_previews));
}
