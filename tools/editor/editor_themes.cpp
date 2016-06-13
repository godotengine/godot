/*************************************************************************/
/*  editor_themes.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "editor_themes.h"
#include "editor_icons.h"
#include "editor_fonts.h"
#include "editor_settings.h"

Ref<Theme> editor_create_theme()
{
	Ref<Theme> theme = Ref<Theme>( memnew( Theme ) );
	editor_register_icons(theme);
	editor_register_fonts(theme);

	//theme->set_icon("folder","EditorFileDialog",Theme::get_default()->get_icon("folder","EditorFileDialog"));
	//theme->set_color("files_disabled","EditorFileDialog",Color(0,0,0,0.7));

	String global_font = EditorSettings::get_singleton()->get("global/custom_font");
	if (global_font!="") {
		Ref<Font> fnt = ResourceLoader::load(global_font);
		if (fnt.is_valid()) {
			theme->set_default_theme_font(fnt);
		}
	}

	Ref<StyleBoxTexture> focus_sbt=memnew( StyleBoxTexture );
	focus_sbt->set_texture(theme->get_icon("EditorFocus","EditorIcons"));
	for(int i=0;i<4;i++) {
		focus_sbt->set_margin_size(Margin(i),16);
		focus_sbt->set_default_margin(Margin(i),16);
	}
	focus_sbt->set_draw_center(false);
	theme->set_stylebox("EditorFocus","EditorStyles",focus_sbt);

	return theme;
}
