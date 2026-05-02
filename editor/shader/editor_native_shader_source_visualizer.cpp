/**************************************************************************/
/*  editor_native_shader_source_visualizer.cpp                            */
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

#include "editor_native_shader_source_visualizer.h"

#include "core/object/class_db.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/code_edit.h"
#include "scene/gui/tab_container.h"
#include "scene/resources/syntax_highlighter.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/shader_language.h"

void EditorNativeShaderSourceVisualizer::_load_theme_settings() {
    syntax_highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));
    syntax_highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
    syntax_highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/function_color"));
    syntax_highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/member_variable_color"));

    syntax_highlighter->clear_keyword_colors();

    List<String> keywords;
    ShaderLanguage::get_keyword_list(&keywords);
    const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
    const Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");

    for (const String &keyword : keywords) {
        if (ShaderLanguage::is_control_flow_keyword(keyword)) {
            syntax_highlighter->add_keyword_color(keyword, control_flow_keyword_color);
        } else {
            syntax_highlighter->add_keyword_color(keyword, keyword_color);
        }
    }

    // Colorize comments.
    const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
    syntax_highlighter->clear_color_regions();
    syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
    syntax_highlighter->add_color_region("//", "", comment_color, true);

    // Colorize preprocessor statements.
    const Color user_type_color = EDITOR_GET("text_editor/theme/highlighting/user_type_color");
    syntax_highlighter->add_color_region("#", "", user_type_color, true);

    syntax_highlighter->set_uint_suffix_enabled(true);
}

void EditorNativeShaderSourceVisualizer::_inspect_shader(RID p_shader) {
    if (versions) {
        memdelete(versions);
        versions = nullptr;
    }

    RenderingServerTypes::ShaderNativeSourceCode nsc = RS::get_singleton()->shader_get_native_source_code(p_shader);

    _load_theme_settings();

    // Cache theme and settings to avoid redundant dictionary lookups in the nested loops.
    Ref<Font> source_font = get_theme_font("source", EditorStringName(EditorFonts));
    int source_font_size = get_theme_font_size("source_size", EditorStringName(EditorFonts));
    
    // Appearance properties
    TextEdit::CaretType caret_type = (TextEdit::CaretType)(int)EDITOR_GET("text_editor/appearance/caret/type");
    bool caret_blink = EDITOR_GET("text_editor/appearance/caret/caret_blink");
    float caret_blink_interval = EDITOR_GET("text_editor/appearance/caret/caret_blink_interval");
    bool highlight_current_line = EDITOR_GET("text_editor/appearance/caret/highlight_current_line");
    bool highlight_all_occurrences = EDITOR_GET("text_editor/appearance/caret/highlight_all_occurrences");
    bool show_line_numbers = EDITOR_GET("text_editor/appearance/gutters/show_line_numbers");
    bool line_numbers_zero_padded = EDITOR_GET("text_editor/appearance/gutters/line_numbers_zero_padded");
    bool show_minimap = EDITOR_GET("text_editor/appearance/minimap/show_minimap");
    int minimap_width = (int)EDITOR_GET("text_editor/appearance/minimap/minimap_width") * EDSCALE;
    bool code_folding = EDITOR_GET("text_editor/appearance/lines/code_folding");
    TextEdit::LineWrappingMode word_wrap = (TextEdit::LineWrappingMode)(int)EDITOR_GET("text_editor/appearance/lines/word_wrap");
    TextServer::AutowrapMode autowrap_mode = (TextServer::AutowrapMode)(int)EDITOR_GET("text_editor/appearance/lines/autowrap_mode");
    bool draw_tabs = EDITOR_GET("text_editor/appearance/whitespace/draw_tabs");
    bool draw_spaces = EDITOR_GET("text_editor/appearance/whitespace/draw_spaces");
    int line_spacing = EDITOR_GET("text_editor/appearance/whitespace/line_spacing");

    // Behavior properties
    bool scroll_past_end = EDITOR_GET("text_editor/behavior/navigation/scroll_past_end_of_file");
    bool smooth_scrolling = EDITOR_GET("text_editor/behavior/navigation/smooth_scrolling");
    float v_scroll_speed = EDITOR_GET("text_editor/behavior/navigation/v_scroll_speed");
    bool drag_and_drop_selection = EDITOR_GET("text_editor/behavior/navigation/drag_and_drop_selection");
    int indent_size = EDITOR_GET("text_editor/behavior/indent/size");
    bool auto_indent = EDITOR_GET("text_editor/behavior/indent/auto_indent");
    bool indent_wrapped_lines = EDITOR_GET("text_editor/behavior/indent/indent_wrapped_lines");

    versions = memnew(TabContainer);
    versions->set_theme_type_variation("TabContainerInner");
    versions->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
    versions->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    versions->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    
    for (int i = 0; i < nsc.versions.size(); i++) {
        const auto &version_data = nsc.versions[i];

        TabContainer *vtab = memnew(TabContainer);
        vtab->set_theme_type_variation("TabContainerInner");
        vtab->set_name("Version " + itos(i));
        vtab->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
        vtab->set_v_size_flags(Control::SIZE_EXPAND_FILL);
        vtab->set_h_size_flags(Control::SIZE_EXPAND_FILL);
        versions->add_child(vtab);
        
        for (int j = 0; j < version_data.stages.size(); j++) {
            const auto &stage_data = version_data.stages[j];

            CodeEdit *code_edit = memnew(CodeEdit);
            code_edit->set_editable(false);
            code_edit->set_syntax_highlighter(syntax_highlighter);
            code_edit->add_theme_font_override(SceneStringName(font), source_font);
            code_edit->add_theme_font_size_override(SceneStringName(font_size), source_font_size);
            code_edit->add_theme_constant_override("line_spacing", line_spacing);

            // Appearance: Caret
            code_edit->set_caret_type(caret_type);
            code_edit->set_caret_blink_enabled(caret_blink);
            code_edit->set_caret_blink_interval(caret_blink_interval);
            code_edit->set_highlight_current_line(highlight_current_line);
            code_edit->set_highlight_all_occurrences(highlight_all_occurrences);

            // Appearance: Gutters
            code_edit->set_draw_line_numbers(show_line_numbers);
            code_edit->set_line_numbers_zero_padded(line_numbers_zero_padded);

            // Appearance: Minimap
            code_edit->set_draw_minimap(show_minimap);
            code_edit->set_minimap_width(minimap_width);

            // Appearance: Lines
            code_edit->set_line_folding_enabled(code_folding);
            code_edit->set_draw_fold_gutter(code_folding);
            code_edit->set_line_wrapping_mode(word_wrap);
            code_edit->set_autowrap_mode(autowrap_mode);

            // Appearance: Whitespace
            code_edit->set_draw_tabs(draw_tabs);
            code_edit->set_draw_spaces(draw_spaces);

            // Behavior: Navigation
            code_edit->set_scroll_past_end_of_file_enabled(scroll_past_end);
            code_edit->set_smooth_scroll_enabled(smooth_scrolling);
            code_edit->set_v_scroll_speed(v_scroll_speed);
            code_edit->set_drag_and_drop_selection_enabled(drag_and_drop_selection);

            // Behavior: Indent
            code_edit->set_indent_size(indent_size);
            code_edit->set_auto_indent_enabled(auto_indent);
            code_edit->set_indent_wrapped_lines(indent_wrapped_lines);

            code_edit->set_name(stage_data.name);
            code_edit->set_text(stage_data.code);
            code_edit->set_v_size_flags(Control::SIZE_EXPAND_FILL);
            code_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
            
            vtab->add_child(code_edit);
        }
    }
    
    add_child(versions);
    popup_centered_ratio();
}

void EditorNativeShaderSourceVisualizer::_bind_methods() {
    ClassDB::bind_method("_inspect_shader", &EditorNativeShaderSourceVisualizer::_inspect_shader);
}

EditorNativeShaderSourceVisualizer::EditorNativeShaderSourceVisualizer() {
    syntax_highlighter.instantiate();

    add_to_group("_native_shader_source_visualizer");
    set_title(TTR("Native Shader Source Inspector"));
}
