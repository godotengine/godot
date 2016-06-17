/*************************************************/
/*  default_theme.h                              */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef DEFAULT_THEME_H
#define DEFAULT_THEME_H

#include "scene/resources/theme.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

void fill_default_theme(Ref<Theme>& theme,const Ref<Font> & default_font,const Ref<Font> & large_font,Ref<Texture>& default_icon, Ref<StyleBox>& default_style,bool p_hidpi);
void make_default_theme(bool p_hidpi, Ref<Font> p_font);
void clear_default_theme();

#endif
