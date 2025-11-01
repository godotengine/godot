/**************************************************************************/
/*  safe_area_container.cpp                                               */
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

#include "safe_area_container.h"


void SafeAreaContainer::_notification(int p_what) {
    if (p_what == NOTIFICATION_ENTER_TREE) {
        if (Engine::get_singleton()->is_editor_hint()) {
            return;
        }
        
        callable_mp(this, &SafeAreaContainer::_update_safe_area).call_deferred();
    }
}

void SafeAreaContainer::_update_safe_area() {
    Size2i window_size = DisplayServer::get_singleton()->window_get_size();

    if (Engine::get_singleton()->has_safe_area_override_set()) {
        Engine::SafeAreaInsets safe_area = Engine::get_singleton()->get_safe_area_override();
        add_theme_constant_override("margin_left", safe_area.left);
        add_theme_constant_override("margin_top", safe_area.top);
        add_theme_constant_override("margin_right", safe_area.right);
        add_theme_constant_override("margin_bottom", safe_area.bottom);
    } else {
        Rect2i safe_area = DisplayServer::get_singleton()->get_display_safe_area();
        add_theme_constant_override("margin_left", safe_area.position.x);
        add_theme_constant_override("margin_top", safe_area.position.y);
        add_theme_constant_override("margin_right", window_size.width - safe_area.get_end().x);
        add_theme_constant_override("margin_bottom", window_size.height - safe_area.get_end().y);
    }

    
}