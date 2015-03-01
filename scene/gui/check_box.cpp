/*************************************************************************/
/*  check_button.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "check_box.h"

#include "servers/visual_server.h"
#include "button_group.h"

void CheckBox::_bind_methods()
{
    ObjectTypeDB::bind_method(_MD("set_pressed","pressed"),&CheckBox::toggled);
}

bool CheckBox::is_radio()
{
    Node* parent = this;
    do {
        parent = parent->get_parent();
        if (dynamic_cast< ButtonGroup* >( parent))
            break;
    } while (parent != nullptr);

    return (parent != nullptr);
}

void CheckBox::update_icon(bool p_pressed)
{
    if (is_radio())
        set_icon(Control::get_icon(p_pressed ? "radio_checked" : "radio_unchecked"));
    else
        set_icon(Control::get_icon(p_pressed ? "checked" : "unchecked"));
}

void CheckBox::toggled(bool p_pressed)
{
    update_icon();
    BaseButton::toggled(p_pressed);
}

CheckBox::CheckBox()
{
    set_toggle_mode(true);
    set_text_align(ALIGN_LEFT);

    update_icon(is_pressed());

}

CheckBox::~CheckBox()
{
}
