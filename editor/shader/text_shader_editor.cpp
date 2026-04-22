/**************************************************************************/
/*  text_shader_editor.cpp                                                */
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

#include "text_shader_editor.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/version_generated.gen.h"
#include "editor/docks/inspector_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/scene/material_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "editor/themes/editor_theme_manager.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/split_container.h"
#include "scene/resources/sky.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/shader_preprocessor.h"
#include "servers/rendering/shader_types.h"

#include "modules/regex/regex.h"

/*** SHADER SYNTAX HIGHLIGHTER ****/

Dictionary GDShaderSyntaxHighlighter::_get_line_syntax_highlighting_impl(int p_line) {
	Dictionary color_map;

	for (const Point2i &region : disabled_branch_regions) {
		if (p_line >= region.x && p_line <= region.y) {
			// When "color_regions[0].p_start_key.length() > 2",
			// disabled_branch_region causes color_region to break.
			// This should be seen as a temporary solution.
			CodeHighlighter::_get_line_syntax_highlighting_impl(p_line);

			Dictionary highlighter_info;
			highlighter_info["color"] = disabled_branch_color;

			color_map[0] = highlighter_info;
			return color_map;
		}
	}

	return CodeHighlighter::_get_line_syntax_highlighting_impl(p_line);
}

void GDShaderSyntaxHighlighter::add_disabled_branch_region(const Point2i &p_region) {
	ERR_FAIL_COND(p_region.x < 0);
	ERR_FAIL_COND(p_region.y < 0);

	for (int i = 0; i < disabled_branch_regions.size(); i++) {
		ERR_FAIL_COND_MSG(disabled_branch_regions[i].x == p_region.x, "Branch region with a start line '" + itos(p_region.x) + "' already exists.");
	}

	Point2i disabled_branch_region;
	disabled_branch_region.x = p_region.x;
	disabled_branch_region.y = p_region.y;
	disabled_branch_regions.push_back(disabled_branch_region);

	clear_highlighting_cache();
}

void GDShaderSyntaxHighlighter::clear_disabled_branch_regions() {
	disabled_branch_regions.clear();
	clear_highlighting_cache();
}

void GDShaderSyntaxHighlighter::set_disabled_branch_color(const Color &p_color) {
	disabled_branch_color = p_color;
	clear_highlighting_cache();
}

/*** SHADER PREVIEW LINE LAYER ****/

void TextShaderPreviewLineLayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			line_color = Color(EditorNode::get_singleton()->get_editor_theme()->get_color(SceneStringName(font_color), EditorStringName(Editor)), 0.7);
		} break;
		case NOTIFICATION_DRAW: {
			const Rect2i visible_rect = scroll_container->get_global_rect();

			for (const KeyValue<int, TextShaderPreview *> &E : *previews) {
				const Control *surface_container = E.value->get_surface_container();

				Point2 start_pos = surface_container->get_global_position() + surface_container->get_size() * Point2(1.0, 0.5) + Point2(1.0, 0.0) * EDSCALE;
				if (!visible_rect.has_point(start_pos)) {
					continue;
				}

				Point2i end_pos = code_editor->get_pos_at_line_column(E.key, 0);
				if (end_pos.x == -1) {
					continue;
				}
				end_pos.y -= code_editor->get_line_height() / 2;
				end_pos.x = code_editor->get_line_start_margin() + code_editor->get_gutter_width(0) / 2.0;

				draw_line(start_pos, code_editor->get_global_position() + end_pos, line_color, EDSCALE, true);
			}
		} break;
	}
}

void TextShaderPreviewLineLayer::set_previews(HashMap<int, TextShaderPreview *> &p_previews) {
	previews = &p_previews;
}

void TextShaderPreviewLineLayer::set_code_editor(CodeEdit *p_code_editor) {
	code_editor = p_code_editor;
}

void TextShaderPreviewLineLayer::set_scroll_container(ScrollContainer *p_scroll_container) {
	scroll_container = p_scroll_container;
}

TextShaderPreviewLineLayer::TextShaderPreviewLineLayer() {
	set_as_top_level(true);
}

/***  SHADER PREVIEW ****/

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
	surface_container = memnew(MarginContainer);
	surface_container->set_custom_minimum_size(Size2(150, 150) * EDSCALE);
	add_child(surface_container);

	env.instantiate();
	Ref<Sky> sky = memnew(Sky);
	env->set_sky(sky);
	env->set_background(Environment::BG_COLOR);
	env->set_ambient_source(Environment::AMBIENT_SOURCE_SKY);
	env->set_reflection_source(Environment::REFLECTION_SOURCE_SKY);

	surface = memnew(MaterialEditor);
	surface_container->add_child(surface);

	error_label = memnew(Label);
	error_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	error_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	error_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	error_label->set_v_size_flags(SIZE_EXPAND_FILL);
	error_label->hide();
	surface_container->add_child(error_label);

	HBoxContainer *buttons_hbox = memnew(HBoxContainer);
	add_child(buttons_hbox);

	goto_button = memnew(Button);
	goto_button->connect(SceneStringName(pressed), callable_mp(this, &TextShaderPreview::_goto_pressed));
	goto_button->set_h_size_flags(SIZE_EXPAND_FILL);
	buttons_hbox->add_child(goto_button);

	delete_button = memnew(Button);
	delete_button->connect(SceneStringName(pressed), callable_mp(this, &TextShaderPreview::_delete_pressed));
	buttons_hbox->add_child(delete_button);

	shader_material.instantiate();
}

void TextShaderPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			error_label->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
			delete_button->set_button_icon(get_editor_theme_icon(SNAME("Close")));
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED: {
			goto_button->set_text(vformat(TTR("Go to Line %d"), line + 1));
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
	error_label->show();
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
					const Ref<ShaderMaterial> surface_material = Object::cast_to<ShaderMaterial>(mi->get_surface_override_material(i).ptr());

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

MarginContainer *TextShaderPreview::get_surface_container() const {
	return surface_container;
}

void TextShaderPreview::set_shader_code(const String &p_code, int p_line, bool p_in_comment) {
	line = p_line;
	in_comment = p_in_comment;
	goto_button->set_text(vformat(TTR("Go to Line %d"), line + 1));

	String shader_type = ShaderLanguage::get_shader_type(p_code);
	bool mode_3d = shader_type == "spatial";

	if (shader_type != "canvas_item" && !mode_3d) {
		_show_error(TTRC("Shader type must be either `canvas_item` or `spatial` to correctly set a preview."));
		return;
	}

	const PackedStringArray lines = p_code.split("\n");
	String enclosing_function = _get_enclosing_function(lines, p_line);

	if (enclosing_function != "fragment") {
		_show_error(TTRC("Preview only supports assignments in the `fragment()` function."));
		return;
	}

	if (_is_inside_loop(lines, p_line)) {
		_show_error(TTRC("Preview is not supported inside loops."));
		return;
	}

	String var_name;
	int start;
	int end;

	if (in_comment || !_find_statement(lines, p_line, var_name, start, end)) {
		_show_error(TTRC("The selected line needs to be an assignment."));
		return;
	}

	String type = _find_var_type(lines, var_name, end, mode_3d);

	// All code before assignment stays as it was.
	PackedStringArray truncated_lines = lines.slice(0, end + 1);

	String injection;
	HashMap<String, String> &assignments = mode_3d ? spatial_assignments : canvas_assignments;
	if (!assignments.has(type)) {
		_show_error(TTRC("Preview unavailable for current assignment.\nSupported types are: `bool`, `int`, `float`, `vec2`, `vec3`, `vec4`."));
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

	surface->show();
	error_label->hide();
	surface->edit(shader_material.ptr(), env);
}

/*** SHADER SCRIPT EDITOR ****/

static bool saved_warnings_enabled = false;
static bool saved_treat_warning_as_errors = false;
static HashMap<ShaderWarning::Code, bool> saved_warnings;
static uint32_t saved_warning_flags = 0U;

void ShaderTextEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			get_text_editor()->add_theme_color_override("breakpoint_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SceneStringName(font_color), EditorStringName(Editor)));
			get_text_editor()->add_theme_icon_override("breakpoint", get_editor_theme_icon(SNAME("GuiVisibilityVisible")));

			if (is_visible_in_tree()) {
				_load_theme_settings();
				if (warnings.size() > 0 && last_compile_result == OK) {
					warnings_panel->clear();
					_update_warning_panel();
				}
			}
		} break;
	}
}

Ref<Shader> ShaderTextEditor::get_edited_shader() const {
	return shader;
}

Ref<ShaderInclude> ShaderTextEditor::get_edited_shader_include() const {
	return shader_inc;
}

void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {
	set_edited_shader(p_shader, p_shader->get_code());
}

void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader, const String &p_code) {
	if (shader == p_shader) {
		return;
	}
	if (shader.is_valid()) {
		shader->disconnect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
	shader = p_shader;
	shader_inc = Ref<ShaderInclude>();

	set_edited_code(p_code);

	if (shader.is_valid()) {
		shader->connect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
}

void ShaderTextEditor::set_edited_shader_include(const Ref<ShaderInclude> &p_shader_inc) {
	set_edited_shader_include(p_shader_inc, p_shader_inc->get_code());
}

void ShaderTextEditor::_shader_changed() {
	// This function is used for dependencies (include changing changes main shader and forces it to revalidate)
	if (block_shader_changed) {
		return;
	}
	dependencies_version++;
	_validate_script();
}

void ShaderTextEditor::set_edited_shader_include(const Ref<ShaderInclude> &p_shader_inc, const String &p_code) {
	if (shader_inc == p_shader_inc) {
		return;
	}
	if (shader_inc.is_valid()) {
		shader_inc->disconnect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
	shader_inc = p_shader_inc;
	shader = Ref<Shader>();

	set_edited_code(p_code);

	if (shader_inc.is_valid()) {
		shader_inc->connect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
}

void ShaderTextEditor::set_edited_code(const String &p_code) {
	_load_theme_settings();

	get_text_editor()->set_text(p_code);
	get_text_editor()->clear_undo_history();
	callable_mp((TextEdit *)get_text_editor(), &TextEdit::set_h_scroll).call_deferred(0);
	callable_mp((TextEdit *)get_text_editor(), &TextEdit::set_v_scroll).call_deferred(0);
	get_text_editor()->tag_saved_version();

	_validate_script();
	_line_col_changed();
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
		E.value->recompile(get_text_editor()->get_text());
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
	CodeEdit *tx = get_text_editor();

	TextShaderPreview *preview = memnew(TextShaderPreview);
	previews.insert(p_line, preview);

	if (last_compile_result != OK) {
		preview->show_shader_compile_error();
	} else {
		preview->set_shader_code(tx->get_text(), p_line, tx->is_in_comment(p_line) != -1);
	}

	preview->connect("goto_btn_pressed", callable_mp(this, &ShaderTextEditor::goto_shader_preview).bind(p_line));
	preview->connect("remove_btn_pressed", callable_mp(this, &ShaderTextEditor::remove_shader_preview).bind(p_line));
	preview_box->add_child(preview);
}

void ShaderTextEditor::remove_shader_preview(int p_line) {
	get_text_editor()->set_line_as_breakpoint(p_line, false);
}

void ShaderTextEditor::set_preview_box(Control *p_box) {
	preview_box = p_box;
}

void ShaderTextEditor::reload_text() {
	ERR_FAIL_COND(shader.is_null() && shader_inc.is_null());

	String code;
	if (shader.is_valid()) {
		code = shader->get_code();
	} else {
		code = shader_inc->get_code();
	}

	CodeEdit *te = get_text_editor();
	int column = te->get_caret_column();
	int row = te->get_caret_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(code);
	te->set_caret_line(row);
	te->set_caret_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	update_line_and_column();
}

void ShaderTextEditor::set_warnings_panel(RichTextLabel *p_warnings_panel) {
	warnings_panel = p_warnings_panel;
}

void ShaderTextEditor::_load_theme_settings() {
	CodeEdit *te = get_text_editor();
	Color updated_marked_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
	if (updated_marked_line_color != marked_line_color) {
		for (int i = 0; i < te->get_line_count(); i++) {
			if (te->get_line_background_color(i) == marked_line_color) {
				te->set_line_background_color(i, updated_marked_line_color);
			}
		}
		marked_line_color = updated_marked_line_color;
	}

	syntax_highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));
	syntax_highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
	syntax_highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/function_color"));
	syntax_highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/member_variable_color"));

	syntax_highlighter->clear_keyword_colors();

	const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
	const Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);

	for (const String &E : keywords) {
		if (ShaderLanguage::is_control_flow_keyword(E)) {
			syntax_highlighter->add_keyword_color(E, control_flow_keyword_color);
		} else {
			syntax_highlighter->add_keyword_color(E, keyword_color);
		}
	}

	List<String> pp_keywords;
	ShaderPreprocessor::get_keyword_list(&pp_keywords, false);

	for (const String &E : pp_keywords) {
		syntax_highlighter->add_keyword_color(E, control_flow_keyword_color);
	}

	// Colorize built-ins like `COLOR` differently to make them easier
	// to distinguish from keywords at a quick glance.

	List<String> built_ins;

	if (shader_inc.is_valid()) {
		for (int i = 0; i < RSE::SHADER_MAX; i++) {
			for (const KeyValue<StringName, ShaderLanguage::FunctionInfo> &E : ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(i))) {
				for (const KeyValue<StringName, ShaderLanguage::BuiltInInfo> &F : E.value.built_ins) {
					built_ins.push_back(F.key);
				}
			}

			{
				const Vector<ShaderLanguage::ModeInfo> &render_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(i));

				for (const ShaderLanguage::ModeInfo &mode_info : render_modes) {
					if (!mode_info.options.is_empty()) {
						for (const StringName &option : mode_info.options) {
							built_ins.push_back(String(mode_info.name) + "_" + String(option));
						}
					} else {
						built_ins.push_back(String(mode_info.name));
					}
				}
			}

			{
				const Vector<ShaderLanguage::ModeInfo> &stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(i));

				for (const ShaderLanguage::ModeInfo &mode_info : stencil_modes) {
					if (!mode_info.options.is_empty()) {
						for (const StringName &option : mode_info.options) {
							built_ins.push_back(String(mode_info.name) + "_" + String(option));
						}
					} else {
						built_ins.push_back(String(mode_info.name));
					}
				}
			}
		}
	} else if (shader.is_valid()) {
		for (const KeyValue<StringName, ShaderLanguage::FunctionInfo> &E : ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(shader->get_mode()))) {
			for (const KeyValue<StringName, ShaderLanguage::BuiltInInfo> &F : E.value.built_ins) {
				built_ins.push_back(F.key);
			}
		}

		{
			const Vector<ShaderLanguage::ModeInfo> &shader_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(shader->get_mode()));

			for (const ShaderLanguage::ModeInfo &mode_info : shader_modes) {
				if (!mode_info.options.is_empty()) {
					for (const StringName &option : mode_info.options) {
						built_ins.push_back(String(mode_info.name) + "_" + String(option));
					}
				} else {
					built_ins.push_back(String(mode_info.name));
				}
			}
		}

		{
			const Vector<ShaderLanguage::ModeInfo> &stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(shader->get_mode()));

			for (const ShaderLanguage::ModeInfo &mode_info : stencil_modes) {
				if (!mode_info.options.is_empty()) {
					for (const StringName &option : mode_info.options) {
						built_ins.push_back(String(mode_info.name) + "_" + String(option));
					}
				} else {
					built_ins.push_back(String(mode_info.name));
				}
			}
		}
	}

	const Color user_type_color = EDITOR_GET("text_editor/theme/highlighting/user_type_color");

	for (const String &E : built_ins) {
		syntax_highlighter->add_keyword_color(E, user_type_color);
	}

	// Colorize comments.
	const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	syntax_highlighter->clear_color_regions();
	syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
	syntax_highlighter->add_color_region("//", "", comment_color, true);

	const Color doc_comment_color = EDITOR_GET("text_editor/theme/highlighting/doc_comment_color");
	syntax_highlighter->add_color_region("/**", "*/", doc_comment_color, false);
	// "/**/" will be treated as the start of the "/**" region, this line is guaranteed to end the color_region.
	syntax_highlighter->add_color_region("/**/", "", comment_color, true);

	// Disabled preprocessor branches use translucent text color to be easier to distinguish from comments.
	syntax_highlighter->set_disabled_branch_color(Color(EDITOR_GET("text_editor/theme/highlighting/text_color")) * Color(1, 1, 1, 0.5));

	te->clear_comment_delimiters();
	te->add_comment_delimiter("/*", "*/", false);
	te->add_comment_delimiter("//", "", true);

	if (!te->has_auto_brace_completion_open_key("/*")) {
		te->add_auto_brace_completion_pair("/*", "*/");
	}

	// Colorize preprocessor include strings.
	const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	syntax_highlighter->add_color_region("\"", "\"", string_color, false);
	syntax_highlighter->set_uint_suffix_enabled(true);
}

void ShaderTextEditor::_check_shader_mode() {
	String type = ShaderLanguage::get_shader_type(get_text_editor()->get_text());

	Shader::Mode mode;

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
		set_block_shader_changed(true);
		shader->set_code(get_text_editor()->get_text());
		set_block_shader_changed(false);
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

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options) {
	List<ScriptLanguage::CodeCompletionOption> pp_options;
	List<ScriptLanguage::CodeCompletionOption> pp_defines;
	ShaderPreprocessor preprocessor;
	String code;
	String resource_path = (shader.is_valid() ? shader->get_path() : shader_inc->get_path());
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

	if (shader.is_null()) {
		comp_info.is_include = true;

		sl.complete(code, comp_info, r_options, calltip);
		if (sl.get_completion_type() == ShaderLanguage::COMPLETION_IDENTIFIER) {
			for (const ScriptLanguage::CodeCompletionOption &E : pp_defines) {
				r_options->push_back(E);
			}
		}

		get_text_editor()->set_code_hint(calltip);
		return;
	}
	_check_shader_mode();
	comp_info.functions = ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(shader->get_mode()));
	comp_info.render_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(shader->get_mode()));
	comp_info.stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(shader->get_mode()));
	comp_info.shader_types = ShaderTypes::get_singleton()->get_types();

	sl.complete(code, comp_info, r_options, calltip);
	if (sl.get_completion_type() == ShaderLanguage::COMPLETION_IDENTIFIER) {
		for (const ScriptLanguage::CodeCompletionOption &E : pp_defines) {
			r_options->push_back(E);
		}
	}

	get_text_editor()->set_code_hint(calltip);
}

void ShaderTextEditor::_validate_script() {
	emit_signal(CoreStringName(script_changed)); // Ensure to notify that it changed, so it is applied

	String code;

	if (shader.is_valid()) {
		_check_shader_mode();
		code = shader->get_code();
	} else {
		code = shader_inc->get_code();
	}

	ShaderPreprocessor preprocessor;
	String code_pp;
	String error_pp;
	List<ShaderPreprocessor::FilePosition> err_positions;
	List<ShaderPreprocessor::Region> regions;
	String filename;
	if (shader.is_valid()) {
		filename = shader->get_path();
		get_text_editor()->set_draw_breakpoints_gutter(true);
	} else if (shader_inc.is_valid()) {
		filename = shader_inc->get_path();
		get_text_editor()->set_draw_breakpoints_gutter(false);
	}
	last_compile_result = preprocessor.preprocess(code, filename, code_pp, &error_pp, &err_positions, &regions);

	for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
		get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
	}

	syntax_highlighter->clear_disabled_branch_regions();
	for (const ShaderPreprocessor::Region &region : regions) {
		if (!region.enabled) {
			if (filename != region.file) {
				continue;
			}
			syntax_highlighter->add_disabled_branch_region(Point2i(region.from_line, region.to_line));
		}
	}

	set_error("");
	set_error_count(0);

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
			set_error_count(err_positions.size() - 1);
		}

		set_error(err_text);
		set_error_pos(err_line - 1, 0);

		for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
			get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
		}
		get_text_editor()->set_line_background_color(err_line - 1, marked_line_color);

		set_warning_count(0);

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
				set_error_count(include_positions.size() - 1);
			} else {
				// Error in the main file.
				err_line = sl.get_error_line();

				const String message = sl.get_error_text().replace("[", "[lb]");

				err_text = vformat(TTR("Error at line %d:"), err_line) + " " + message;
				set_error_count(0);
			}

			set_error(err_text);
			set_error_pos(err_line - 1, 0);

			get_text_editor()->set_line_background_color(err_line - 1, marked_line_color);

			for (KeyValue<int, TextShaderPreview *> pair : previews) {
				pair.value->show_shader_compile_error();
			}
		} else {
			set_error("");

			for (KeyValue<int, TextShaderPreview *> pair : previews) {
				pair.value->set_shader_code(code, pair.key, get_text_editor()->is_in_comment(pair.key) != -1);
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
			set_warning_count(0);
		}
	}

	emit_signal(SNAME("script_validated"), last_compile_result == OK); // Notify that validation finished, to update the list of scripts
}

void ShaderTextEditor::_update_warning_panel() {
	int warning_count = 0;

	warnings_panel->push_table(2);
	for (const ShaderWarning &w : warnings) {
		if (warning_count == 0) {
			if (saved_treat_warning_as_errors) {
				const String message = (w.get_message() + " " + TTR("Warnings should be fixed to prevent errors.")).replace("[", "[lb]");
				const String error_text = vformat(TTR("Error at line %d:"), w.get_line()) + " " + message;

				set_error(error_text);
				set_error_pos(w.get_line() - 1, 0);

				get_text_editor()->set_line_background_color(w.get_line() - 1, marked_line_color);
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

	set_warning_count(warning_count);
}

void ShaderTextEditor::_bind_methods() {
	ADD_SIGNAL(MethodInfo("script_validated", PropertyInfo(Variant::BOOL, "valid")));
}

ShaderTextEditor::ShaderTextEditor() {
	syntax_highlighter.instantiate();
	get_text_editor()->set_syntax_highlighter(syntax_highlighter);

	preview_line_layer = memnew(TextShaderPreviewLineLayer);
	preview_line_layer->set_previews(previews);
	preview_line_layer->set_code_editor(get_text_editor());
	add_child(preview_line_layer);

	InspectorDock::get_inspector_singleton()->connect(SNAME("edited_object_changed"), callable_mp(this, &ShaderTextEditor::recompile_previews));
}

/*** SCRIPT EDITOR ******/

void TextShaderEditor::_menu_option(int p_option) {
	CodeEdit *tx = code_editor->get_text_editor();
	tx->apply_ime();

	switch (p_option) {
		case EDIT_UNDO: {
			tx->undo();
		} break;
		case EDIT_REDO: {
			tx->redo();
		} break;
		case EDIT_CUT: {
			tx->cut();
		} break;
		case EDIT_COPY: {
			tx->copy();
		} break;
		case EDIT_PASTE: {
			tx->paste();
		} break;
		case EDIT_SELECT_ALL: {
			tx->select_all();
		} break;
		case EDIT_MOVE_LINE_UP: {
			tx->move_lines_up();
		} break;
		case EDIT_MOVE_LINE_DOWN: {
			tx->move_lines_down();
		} break;
		case EDIT_INDENT: {
			if (shader.is_null() && shader_inc.is_null()) {
				return;
			}
			tx->indent_lines();
		} break;
		case EDIT_UNINDENT: {
			if (shader.is_null() && shader_inc.is_null()) {
				return;
			}
			tx->unindent_lines();
		} break;
		case EDIT_DELETE_LINE: {
			tx->delete_lines();
		} break;
		case EDIT_DUPLICATE_SELECTION: {
			tx->duplicate_selection();
		} break;
		case EDIT_DUPLICATE_LINES: {
			tx->duplicate_lines();
		} break;
		case EDIT_TOGGLE_WORD_WRAP: {
			TextEdit::LineWrappingMode wrap = tx->get_line_wrapping_mode();
			tx->set_line_wrapping_mode(wrap == TextEdit::LINE_WRAPPING_BOUNDARY ? TextEdit::LINE_WRAPPING_NONE : TextEdit::LINE_WRAPPING_BOUNDARY);
		} break;
		case EDIT_TOGGLE_COMMENT: {
			if (shader.is_null() && shader_inc.is_null()) {
				return;
			}
			code_editor->toggle_inline_comment("//");
		} break;
		case EDIT_COMPLETE: {
			tx->request_code_completion();
		} break;
		case SEARCH_FIND: {
			code_editor->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {
			code_editor->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {
			code_editor->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {
			code_editor->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_GOTO_LINE: {
			goto_line_popup->popup_find_line(code_editor);
		} break;
		case BOOKMARK_TOGGLE: {
			code_editor->toggle_bookmark();
		} break;
		case BOOKMARK_GOTO_NEXT: {
			code_editor->goto_next_bookmark();
		} break;
		case BOOKMARK_GOTO_PREV: {
			code_editor->goto_prev_bookmark();
		} break;
		case BOOKMARK_REMOVE_ALL: {
			code_editor->remove_all_bookmarks();
		} break;
		case PREVIEW_TOGGLE: {
			Vector<int> sorted_carets = tx->get_sorted_carets();
			int last_line = -1;

			for (const int &c : sorted_carets) {
				int from = tx->get_selection_from_line(c);
				from += from == last_line ? 1 : 0;

				int to = tx->get_selection_to_line(c);
				if (to < from) {
					continue;
				}

				// Check first if there's any lines with breakpoints in the selection.
				bool selection_has_breakpoints = false;
				for (int line = from; line <= to; line++) {
					if (tx->is_line_breakpointed(line)) {
						selection_has_breakpoints = true;
						break;
					}
				}

				// Set breakpoint on caret or remove all bookmarks from the selection.
				if (!selection_has_breakpoints) {
					if (tx->get_caret_line(c) != last_line) {
						tx->set_line_as_breakpoint(tx->get_caret_line(c), true);
					}
				} else {
					for (int line = from; line <= to; line++) {
						tx->set_line_as_breakpoint(line, false);
					}
				}

				last_line = to;
			}
		} break;
		case PREVIEW_REMOVE_ALL: {
			PackedInt32Array bpoints = tx->get_breakpointed_lines();

			for (int i = 0; i < bpoints.size(); i++) {
				int line = bpoints[i];
				bool dobreak = !tx->is_line_breakpointed(line);

				tx->set_line_as_breakpoint(line, dobreak);
			}
		} break;
		case PREVIEW_GOTO_NEXT: {
			PackedInt32Array bpoints = tx->get_breakpointed_lines();
			if (bpoints.is_empty()) {
				return;
			}

			int current_line = tx->get_caret_line();
			int bpoint_idx = 0;
			if (current_line < (int)bpoints[bpoints.size() - 1]) {
				while (bpoint_idx < bpoints.size() && bpoints[bpoint_idx] <= current_line) {
					bpoint_idx++;
				}
			}
			_focus_preview_line(bpoints[bpoint_idx]);
		} break;
		case PREVIEW_GOTO_PREV: {
			PackedInt32Array bpoints = tx->get_breakpointed_lines();
			if (bpoints.is_empty()) {
				return;
			}

			int current_line = tx->get_caret_line();
			int bpoint_idx = bpoints.size() - 1;
			if (current_line > (int)bpoints[0]) {
				while (bpoint_idx >= 0 && bpoints[bpoint_idx] >= current_line) {
					bpoint_idx--;
				}
			}
			_focus_preview_line(bpoints[bpoint_idx]);
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open(vformat("%s/tutorials/shaders/shader_reference/index.html", GODOT_VERSION_DOCS_URL));
		} break;
		case EDIT_EMOJI_AND_SYMBOL: {
			tx->show_emoji_and_symbol_picker();
		} break;
		case EDIT_JOIN_LINES: {
			tx->join_lines();
		} break;
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
	}
}

void TextShaderEditor::_prepare_edit_menu() {
	const CodeEdit *tx = code_editor->get_text_editor();
	PopupMenu *popup = edit_menu->get_popup();
	popup->set_item_disabled(popup->get_item_index(EDIT_UNDO), !tx->has_undo());
	popup->set_item_disabled(popup->get_item_index(EDIT_REDO), !tx->has_redo());
}

void TextShaderEditor::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (EditorThemeManager::is_generated_theme_outdated() ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/editor/fonts") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("text_editor")) {
				_apply_editor_settings();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible_in_tree() && preview_timer->is_inside_tree()) {
				preview_timer->start();
			}
		} break;

		case NOTIFICATION_RESIZED: {
			preview_timer->start();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			site_search->set_button_icon(get_editor_theme_icon(SNAME("ExternalLink")));
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_IN: {
			_check_for_external_edit();
		} break;
	}
}

void TextShaderEditor::_apply_editor_settings() {
	code_editor->update_editor_settings();

	trim_trailing_whitespace_on_save = EDITOR_GET("text_editor/behavior/files/trim_trailing_whitespace_on_save");
	trim_final_newlines_on_save = EDITOR_GET("text_editor/behavior/files/trim_final_newlines_on_save");
}

void TextShaderEditor::_show_warnings_panel(bool p_show) {
	warnings_panel->set_visible(p_show);
}

void TextShaderEditor::_warning_clicked(const Variant &p_line) {
	if (p_line.get_type() == Variant::INT) {
		code_editor->goto_line_centered(p_line.operator int64_t());
	}
}

void TextShaderEditor::_bind_methods() {
	ClassDB::bind_method("_show_warnings_panel", &TextShaderEditor::_show_warnings_panel);
	ClassDB::bind_method("_warning_clicked", &TextShaderEditor::_warning_clicked);

	ADD_SIGNAL(MethodInfo("validation_changed"));
}

void TextShaderEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	code_editor->goto_line_selection(p_line, p_begin, p_end);
}

void TextShaderEditor::_project_settings_changed() {
	_update_warnings(true);
}

void TextShaderEditor::_focus_preview_line(int p_line) {
	code_editor->goto_line_centered(p_line);

	TextShaderPreview *preview = code_editor->get_preview(p_line);
	if (preview) {
		preview_sbox->ensure_control_visible(preview);
	}
	preview_timer->start();
}

void TextShaderEditor::_update_warnings(bool p_validate) {
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

	if (p_validate && changed && code_editor && code_editor->get_edited_shader().is_valid()) {
		code_editor->validate_script();
	}
}

void TextShaderEditor::_check_for_external_edit() {
	bool use_autoreload = bool(EDITOR_GET("text_editor/behavior/files/auto_reload_scripts_on_external_change"));

	if (shader_inc.is_valid()) {
		if (shader_inc->get_last_modified_time() != FileAccess::get_modified_time(shader_inc->get_path())) {
			if (use_autoreload) {
				_reload_shader_include_from_disk();
			} else {
				callable_mp((Window *)disk_changed, &Window::popup_centered).call_deferred(Size2i());
			}
		}
		return;
	}

	if (shader.is_null() || shader->is_built_in()) {
		return;
	}

	if (shader->get_last_modified_time() != FileAccess::get_modified_time(shader->get_path())) {
		if (use_autoreload) {
			_reload_shader_from_disk();
		} else {
			callable_mp((Window *)disk_changed, &Window::popup_centered).call_deferred(Size2i());
		}
	}
}

void TextShaderEditor::_reload_shader_from_disk() {
	Ref<Shader> rel_shader = ResourceLoader::load(shader->get_path(), shader->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
	ERR_FAIL_COND(rel_shader.is_null());

	code_editor->set_block_shader_changed(true);
	shader->set_code(rel_shader->get_code());
	code_editor->set_block_shader_changed(false);
	shader->set_last_modified_time(rel_shader->get_last_modified_time());
	code_editor->reload_text();
}

void TextShaderEditor::_reload_shader_include_from_disk() {
	Ref<ShaderInclude> rel_shader_include = ResourceLoader::load(shader_inc->get_path(), shader_inc->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
	ERR_FAIL_COND(rel_shader_include.is_null());

	code_editor->set_block_shader_changed(true);
	shader_inc->set_code(rel_shader_include->get_code());
	code_editor->set_block_shader_changed(false);
	shader_inc->set_last_modified_time(rel_shader_include->get_last_modified_time());
	code_editor->reload_text();
}

void TextShaderEditor::_reload() {
	if (shader.is_valid()) {
		_reload_shader_from_disk();
	} else if (shader_inc.is_valid()) {
		_reload_shader_include_from_disk();
	}
}

void TextShaderEditor::edit_shader(const Ref<Shader> &p_shader) {
	if (p_shader.is_null() || !p_shader->is_text_shader()) {
		return;
	}

	if (shader == p_shader) {
		return;
	}

	shader = p_shader;
	shader_inc = Ref<ShaderInclude>();

	code_editor->set_edited_shader(shader);
}

void TextShaderEditor::edit_shader_include(const Ref<ShaderInclude> &p_shader_inc) {
	if (p_shader_inc.is_null()) {
		return;
	}

	if (shader_inc == p_shader_inc) {
		return;
	}

	shader_inc = p_shader_inc;
	shader = Ref<Shader>();

	code_editor->set_edited_shader_include(p_shader_inc);
}

void TextShaderEditor::use_menu_bar(MenuButton *p_file_menu) {
	p_file_menu->set_switch_on_hover(true);
	menu_bar_hbox->add_child(p_file_menu);
	menu_bar_hbox->move_child(p_file_menu, 0);
}

void TextShaderEditor::save_external_data(const String &p_str) {
	if (shader.is_null() && shader_inc.is_null()) {
		disk_changed->hide();
		return;
	}

	if (trim_trailing_whitespace_on_save) {
		trim_trailing_whitespace();
	}

	if (trim_final_newlines_on_save) {
		trim_final_newlines();
	}

	apply_shaders();

	Ref<Shader> edited_shader = code_editor->get_edited_shader();
	if (edited_shader.is_valid()) {
		ResourceSaver::save(edited_shader);
	}
	if (shader.is_valid() && shader != edited_shader) {
		ResourceSaver::save(shader);
	}

	Ref<ShaderInclude> edited_shader_inc = code_editor->get_edited_shader_include();
	if (edited_shader_inc.is_valid()) {
		ResourceSaver::save(edited_shader_inc);
	}
	if (shader_inc.is_valid() && shader_inc != edited_shader_inc) {
		ResourceSaver::save(shader_inc);
	}
	code_editor->get_text_editor()->tag_saved_version();

	disk_changed->hide();
}

void TextShaderEditor::trim_trailing_whitespace() {
	code_editor->trim_trailing_whitespace();
}

void TextShaderEditor::trim_final_newlines() {
	code_editor->trim_final_newlines();
}

void TextShaderEditor::set_toggle_list_control(Control *p_toggle_list_control) {
	code_editor->set_toggle_list_control(p_toggle_list_control);
}

void TextShaderEditor::update_toggle_files_button() {
	code_editor->update_toggle_files_button();
}

void TextShaderEditor::validate_script() {
	code_editor->_validate_script();
}

bool TextShaderEditor::is_unsaved() const {
	return code_editor->get_text_editor()->get_saved_version() != code_editor->get_text_editor()->get_version();
}

void TextShaderEditor::tag_saved_version() {
	code_editor->get_text_editor()->tag_saved_version();
}

void TextShaderEditor::apply_shaders() {
	String editor_code = code_editor->get_text_editor()->get_text();
	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		if (shader_code != editor_code || dependencies_version != code_editor->get_dependencies_version()) {
			code_editor->set_block_shader_changed(true);
			shader->set_code(editor_code);
			code_editor->set_block_shader_changed(false);
			shader->set_edited(true);
		}
	}
	if (shader_inc.is_valid()) {
		String shader_inc_code = shader_inc->get_code();
		if (shader_inc_code != editor_code || dependencies_version != code_editor->get_dependencies_version()) {
			code_editor->set_block_shader_changed(true);
			shader_inc->set_code(editor_code);
			code_editor->set_block_shader_changed(false);
			shader_inc->set_edited(true);
		}
	}

	dependencies_version = code_editor->get_dependencies_version();
}

void TextShaderEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {
	Ref<InputEventMouseButton> mb = ev;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
			CodeEdit *tx = code_editor->get_text_editor();

			tx->apply_ime();

			Point2i pos = tx->get_line_column_at_pos(mb->get_global_position() - tx->get_global_position());
			int row = pos.y;
			int col = pos.x;
			tx->set_move_caret_on_right_click_enabled(EDITOR_GET("text_editor/behavior/navigation/move_caret_on_right_click"));

			if (tx->is_move_caret_on_right_click_enabled()) {
				tx->remove_secondary_carets();
				if (tx->has_selection()) {
					int from_line = tx->get_selection_from_line();
					int to_line = tx->get_selection_to_line();
					int from_column = tx->get_selection_from_column();
					int to_column = tx->get_selection_to_column();

					if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
						// Right click is outside the selected text
						tx->deselect();
					}
				}
				if (!tx->has_selection()) {
					tx->set_caret_line(row, true, false, -1);
					tx->set_caret_column(col);
				}
			}
			_make_context_menu(tx->has_selection(), get_local_mouse_position());
		}
	}

	Ref<InputEventKey> k = ev;
	if (k.is_valid() && k->is_pressed() && k->is_action("ui_menu", true)) {
		CodeEdit *tx = code_editor->get_text_editor();
		tx->adjust_viewport_to_caret();
		_make_context_menu(tx->has_selection(), (get_global_transform().inverse() * tx->get_global_transform()).xform(tx->get_caret_draw_pos()));
		context_menu->grab_focus();
	}
}

void TextShaderEditor::_on_shader_preview_toggled(int p_line) {
	if (!pending_update_shader_previews) {
		pending_update_shader_previews = true;
		callable_mp(this, &TextShaderEditor::_update_shader_previews).call_deferred();
	}
}

void TextShaderEditor::_update_shader_previews() {
	pending_update_shader_previews = false;

	const CodeEdit *ce = code_editor->get_text_editor();
	code_editor->clear_previews();
	bool found = false;

	for (int i = 0; i < ce->get_line_count(); i++) {
		if (ce->is_line_breakpointed(i)) {
			found = true;
			code_editor->toggle_shader_preview(i);
		}
	}

	code_editor->redraw_preview_lines();

	if (!found) {
		preview_box->hide();
		return;
	}
	preview_box->show();
}

void TextShaderEditor::_update_bookmark_list() {
	bookmarks_menu->clear();

	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_bookmarks"), BOOKMARK_REMOVE_ALL);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_bookmark"), BOOKMARK_GOTO_NEXT);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_bookmark"), BOOKMARK_GOTO_PREV);

	PackedInt32Array bookmark_list = code_editor->get_text_editor()->get_bookmarked_lines();
	if (bookmark_list.is_empty()) {
		return;
	}

	bookmarks_menu->add_separator();

	for (int i = 0; i < bookmark_list.size(); i++) {
		String line = code_editor->get_text_editor()->get_line(bookmark_list[i]).strip_edges();
		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		bookmarks_menu->add_item(String::num_int64(bookmark_list[i] + 1) + " - \"" + line + "\"");
		bookmarks_menu->set_item_metadata(-1, bookmark_list[i]);
	}
}

void TextShaderEditor::_bookmark_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_menu_option(bookmarks_menu->get_item_id(p_idx));
	} else {
		code_editor->goto_line(bookmarks_menu->get_item_metadata(p_idx));
	}
}

void TextShaderEditor::_update_shader_preview_list() {
	previews_menu->clear();
	previews_menu->reset_size();

	previews_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/toggle_shader_preview"), PREVIEW_TOGGLE);
	previews_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/remove_all_shader_previews"), PREVIEW_REMOVE_ALL);
	previews_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/goto_next_shader_preview"), PREVIEW_GOTO_NEXT);
	previews_menu->add_shortcut(ED_GET_SHORTCUT("shader_text_editor/goto_previous_shader_preview"), PREVIEW_GOTO_PREV);

	PackedInt32Array breakpoint_list = get_code_editor()->get_text_editor()->get_breakpointed_lines();
	if (breakpoint_list.is_empty()) {
		return;
	}

	previews_menu->add_separator();

	for (int i = 0; i < breakpoint_list.size(); i++) {
		// Strip edges to remove spaces or tabs.
		// Also replace any tabs by spaces, since we can't print tabs in the menu.
		String line = get_code_editor()->get_text_editor()->get_line(breakpoint_list[i]).replace("\t", "  ").strip_edges();

		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		previews_menu->add_item(String::num_int64(breakpoint_list[i] + 1) + " - `" + line + "`");
		previews_menu->set_item_metadata(-1, breakpoint_list[i]);
	}
}

void TextShaderEditor::_shader_preview_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_menu_option(previews_menu->get_item_id(p_idx));
	} else {
		_focus_preview_line(previews_menu->get_item_metadata(p_idx));
	}
}

void TextShaderEditor::_make_context_menu(bool p_selection, Vector2 p_position) {
	context_menu->clear();
	if (DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_EMOJI_AND_SYMBOL_PICKER)) {
		context_menu->add_item(TTRC("Emoji & Symbols"), EDIT_EMOJI_AND_SYMBOL);
		context_menu->add_separator();
	}
	if (p_selection) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("ui_cut"), EDIT_CUT);
		context_menu->add_shortcut(ED_GET_SHORTCUT("ui_copy"), EDIT_COPY);
	}

	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_paste"), EDIT_PASTE);
	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_text_select_all"), EDIT_SELECT_ALL);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent"), EDIT_INDENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unindent"), EDIT_UNINDENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);

	context_menu->set_item_disabled(context_menu->get_item_index(EDIT_UNDO), !code_editor->get_text_editor()->has_undo());
	context_menu->set_item_disabled(context_menu->get_item_index(EDIT_REDO), !code_editor->get_text_editor()->has_redo());

	context_menu->set_position(get_screen_position() + p_position);
	context_menu->reset_size();
	context_menu->popup();
}

void TextShaderEditor::register_editor() {
	ED_SHORTCUT("shader_text_editor/toggle_shader_preview", TTRC("Toggle Shader Preview"), Key::F9);
	ED_SHORTCUT_OVERRIDE("shader_text_editor/toggle_shader_preview", "macos", KeyModifierMask::META | KeyModifierMask::SHIFT | Key::B);

	ED_SHORTCUT("shader_text_editor/remove_all_shader_previews", TTRC("Remove All Shader Previews"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::F9);
	// Using Control for these shortcuts even on macOS because Command+Comma is taken for opening Editor Settings.
	ED_SHORTCUT("shader_text_editor/goto_next_shader_preview", TTRC("Go to Next Shader Preview"), KeyModifierMask::CTRL | Key::PERIOD);
	ED_SHORTCUT("shader_text_editor/goto_previous_shader_preview", TTRC("Go to Previous Shader Preview"), KeyModifierMask::CTRL | Key::COMMA);
}

TextShaderEditor::TextShaderEditor() {
	_update_warnings(false);

	code_editor = memnew(ShaderTextEditor);

	code_editor->connect("script_validated", callable_mp(this, &TextShaderEditor::_script_validated));

	code_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	code_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	code_editor->connect("show_warnings_panel", callable_mp(this, &TextShaderEditor::_show_warnings_panel));
	code_editor->connect(CoreStringName(script_changed), callable_mp(this, &TextShaderEditor::apply_shaders));
	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &TextShaderEditor::_project_settings_changed));

	code_editor->get_text_editor()->set_symbol_lookup_on_click_enabled(true);
	code_editor->get_text_editor()->set_context_menu_enabled(false);
	code_editor->get_text_editor()->set_draw_executing_lines_gutter(false);
	code_editor->get_text_editor()->connect(SceneStringName(gui_input), callable_mp(this, &TextShaderEditor::_text_edit_gui_input));
	code_editor->get_text_editor()->connect("breakpoint_toggled", callable_mp(this, &TextShaderEditor::_on_shader_preview_toggled));
	code_editor->get_text_editor()->connect("_fold_line_updated", callable_mp(code_editor, &ShaderTextEditor::redraw_preview_lines));
	code_editor->get_text_editor()->connect("theme_changed", callable_mp(code_editor, &ShaderTextEditor::redraw_preview_lines), CONNECT_DEFERRED);
	code_editor->get_text_editor()->get_v_scroll_bar()->connect(SceneStringName(value_changed), callable_mp(code_editor, &ShaderTextEditor::redraw_preview_lines).unbind(1));

	code_editor->update_editor_settings();

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect(SceneStringName(id_pressed), callable_mp(this, &TextShaderEditor::_menu_option));

	VBoxContainer *main_container = memnew(VBoxContainer);
	main_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	menu_bar_hbox = memnew(HBoxContainer);

	edit_menu = memnew(MenuButton);
	edit_menu->set_flat(false);
	edit_menu->set_theme_type_variation("FlatMenuButton");
	edit_menu->set_shortcut_context(this);
	edit_menu->set_text(TTRC("Edit"));
	edit_menu->set_switch_on_hover(true);
	edit_menu->connect("about_to_popup", callable_mp(this, &TextShaderEditor::_prepare_edit_menu));

	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_cut"), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_copy"), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_paste"), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_select_all"), EDIT_SELECT_ALL);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent"), EDIT_INDENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unindent"), EDIT_UNINDENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/join_lines"), EDIT_JOIN_LINES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/duplicate_selection"), EDIT_DUPLICATE_SELECTION);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/duplicate_lines"), EDIT_DUPLICATE_LINES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_word_wrap"), EDIT_TOGGLE_WORD_WRAP);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_completion_query"), EDIT_COMPLETE);
	edit_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &TextShaderEditor::_menu_option));

	search_menu = memnew(MenuButton);
	search_menu->set_flat(false);
	search_menu->set_theme_type_variation("FlatMenuButton");
	search_menu->set_shortcut_context(this);
	search_menu->set_text(TTRC("Search"));
	search_menu->set_switch_on_hover(true);

	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &TextShaderEditor::_menu_option));

	MenuButton *goto_menu = memnew(MenuButton);
	goto_menu->set_flat(false);
	goto_menu->set_theme_type_variation("FlatMenuButton");
	goto_menu->set_shortcut_context(this);
	goto_menu->set_text(TTRC("Go To"));
	goto_menu->set_switch_on_hover(true);
	goto_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &TextShaderEditor::_menu_option));

	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	goto_menu->get_popup()->add_separator();

	bookmarks_menu = memnew(PopupMenu);
	goto_menu->get_popup()->add_submenu_node_item(TTRC("Bookmarks"), bookmarks_menu);
	_update_bookmark_list();
	bookmarks_menu->connect("about_to_popup", callable_mp(this, &TextShaderEditor::_update_bookmark_list));
	bookmarks_menu->connect("index_pressed", callable_mp(this, &TextShaderEditor::_bookmark_item_pressed));

	previews_menu = memnew(PopupMenu);
	goto_menu->get_popup()->add_submenu_node_item(TTRC("Shader Previews"), previews_menu);
	_update_shader_preview_list();
	previews_menu->connect("about_to_popup", callable_mp(this, &TextShaderEditor::_update_shader_preview_list));
	previews_menu->connect("index_pressed", callable_mp(this, &TextShaderEditor::_shader_preview_item_pressed));

	add_child(main_container);
	main_container->add_child(menu_bar_hbox);
	menu_bar_hbox->add_child(edit_menu);
	menu_bar_hbox->add_child(search_menu);
	menu_bar_hbox->add_child(goto_menu);
	menu_bar_hbox->add_spacer();

	site_search = memnew(Button);
	site_search->set_theme_type_variation(SceneStringName(FlatButton));
	site_search->connect(SceneStringName(pressed), callable_mp(this, &TextShaderEditor::_menu_option).bind(HELP_DOCS));
	site_search->set_text(TTRC("Online Docs"));
	site_search->set_tooltip_text(TTRC("Open Godot online documentation."));
	menu_bar_hbox->add_child(site_search);

	menu_bar_hbox->add_theme_style_override(SceneStringName(panel), EditorNode::get_singleton()->get_editor_theme()->get_stylebox(SNAME("ScriptEditorPanel"), EditorStringName(EditorStyles)));

	HSplitContainer *main_box = memnew(HSplitContainer);
	main_box->set_h_size_flags(SIZE_EXPAND_FILL);
	main_box->set_v_size_flags(SIZE_EXPAND_FILL);

	preview_box = memnew(VBoxContainer);
	preview_box->set_v_size_flags(SIZE_EXPAND_FILL);
	preview_box->hide();

	Button *update_params_btn = memnew(Button);
	update_params_btn->set_text(TTRC("Update Parameters"));
	preview_box->add_child(update_params_btn);
	update_params_btn->connect(SceneStringName(pressed), callable_mp(code_editor, &ShaderTextEditor::update_parameters));
	update_params_btn->set_tooltip_text(TTRC("Updates shader parameters in previews to match the values in the current `ShaderMaterial` inspector."));

	preview_sbox = memnew(ScrollContainer);
	preview_sbox->set_v_size_flags(SIZE_EXPAND_FILL);
	preview_sbox->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	preview_sbox->get_v_scroll_bar()->connect(SceneStringName(value_changed), callable_mp(code_editor, &ShaderTextEditor::redraw_preview_lines).unbind(1));
	preview_sbox->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_RESERVE);
	code_editor->get_preview_line_layer()->set_scroll_container(preview_sbox);

	VBoxContainer *preview_vbox = memnew(VBoxContainer);
	preview_vbox->connect(SceneStringName(sort_children), callable_mp(code_editor, &ShaderTextEditor::redraw_preview_lines));
	preview_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
	code_editor->set_preview_box(preview_vbox);
	preview_sbox->add_child(preview_vbox);
	preview_box->add_child(preview_sbox);
	main_box->add_child(preview_box);
	main_box->add_child(code_editor);

	VSplitContainer *editor_box = memnew(VSplitContainer);
	main_container->add_child(editor_box);
	editor_box->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	editor_box->set_v_size_flags(SIZE_EXPAND_FILL);
	editor_box->add_child(main_box);

	FindReplaceBar *bar = memnew(FindReplaceBar);
	main_container->add_child(bar);
	bar->hide();
	code_editor->set_find_replace_bar(bar);

	warnings_panel = memnew(RichTextLabel);
	warnings_panel->set_custom_minimum_size(Size2(0, 100 * EDSCALE));
	warnings_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	warnings_panel->set_meta_underline(true);
	warnings_panel->set_selection_enabled(true);
	warnings_panel->set_context_menu_enabled(true);
	warnings_panel->set_focus_mode(FOCUS_CLICK);
	warnings_panel->hide();
	warnings_panel->connect("meta_clicked", callable_mp(this, &TextShaderEditor::_warning_clicked));
	editor_box->add_child(warnings_panel);
	code_editor->set_warnings_panel(warnings_panel);

	goto_line_popup = memnew(GotoLinePopup);
	add_child(goto_line_popup);
	code_editor->connect("show_goto_popup", callable_mp(this, &TextShaderEditor::_menu_option).bind(SEARCH_GOTO_LINE));

	disk_changed = memnew(ConfirmationDialog);

	VBoxContainer *vbc = memnew(VBoxContainer);
	disk_changed->add_child(vbc);

	Label *dl = memnew(Label);
	dl->set_focus_mode(FOCUS_ACCESSIBILITY);
	dl->set_text(TTRC("This shader has been modified on disk.\nWhat action should be taken?"));
	vbc->add_child(dl);

	disk_changed->connect(SceneStringName(confirmed), callable_mp(this, &TextShaderEditor::_reload));
	disk_changed->set_ok_button_text(TTRC("Reload"));

	disk_changed->add_button(TTRC("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
	disk_changed->connect("custom_action", callable_mp(this, &TextShaderEditor::save_external_data));

	add_child(disk_changed);

	preview_timer = memnew(Timer);
	add_child(preview_timer);
	preview_timer->set_one_shot(true);
	preview_timer->set_wait_time(0.001);
	preview_timer->connect("timeout", callable_mp(code_editor, &ShaderTextEditor::redraw_preview_lines));

	_apply_editor_settings();
	code_editor->show_toggle_files_button(); // TODO: Disabled for now, because it doesn't work properly.
}
