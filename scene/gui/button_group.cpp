/*************************************************************************/
/*  button_group.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "button_group.h"

#if 0
#include "base_button.h"

void ButtonGroup::_add_button(BaseButton *p_button) {

	buttons.insert(p_button);
	p_button->set_toggle_mode(true);
	p_button->set_click_on_press(true);
	p_button->connect("pressed",this,"_pressed",make_binds(p_button));

}

void ButtonGroup::_remove_button(BaseButton *p_button){

	buttons.erase(p_button);
	p_button->disconnect("pressed",this,"_pressed");

}

void ButtonGroup::set_pressed_button(BaseButton *p_button) {

	_pressed(p_button);
}

void ButtonGroup::_pressed(Object *p_button) {

	ERR_FAIL_NULL(p_button);
	BaseButton *b=p_button->cast_to<BaseButton>();
	ERR_FAIL_COND(!b);

	for(Set<BaseButton*>::Element *E=buttons.front();E;E=E->next()) {

		BaseButton *bb=E->get();
		bb->set_pressed( b==bb );
		if (b==bb){
			emit_signal("button_selected", b);
		}
	}
}

Array ButtonGroup::_get_button_list() const {

	List<BaseButton*> b;
	get_button_list(&b);

	b.sort_custom<Node::Comparator>();

	Array arr;
	arr.resize(b.size());

	int idx=0;

	for(List<BaseButton*>::Element *E=b.front();E;E=E->next(),idx++) {

		arr[idx]=E->get();
	}

	return arr;
}

void ButtonGroup::get_button_list(List<BaseButton*> *p_buttons) const {

	for(Set<BaseButton*>::Element *E=buttons.front();E;E=E->next()) {

		p_buttons->push_back(E->get());
	}
}

BaseButton *ButtonGroup::get_pressed_button() const {

	for(Set<BaseButton*>::Element *E=buttons.front();E;E=E->next()) {

		if (E->get()->is_pressed())
			return E->get();
	}

	return NULL;
}

BaseButton *ButtonGroup::get_focused_button() const{

	for(Set<BaseButton*>::Element *E=buttons.front();E;E=E->next()) {

		if (E->get()->has_focus())
			return E->get();
	}

	return NULL;

}

int ButtonGroup::get_pressed_button_index() const {
	//in tree order, this is bizarre

	ERR_FAIL_COND_V(!is_inside_tree(),0);

	BaseButton *pressed = get_pressed_button();
	if (!pressed)
		return -1;

	List<BaseButton*> blist;
	for(Set<BaseButton*>::Element *E=buttons.front();E;E=E->next()) {

		blist.push_back(E->get());

	}

	blist.sort_custom<Node::Comparator>();

	int idx=0;
	for(List<BaseButton*>::Element *E=blist.front();E;E=E->next()) {

		if (E->get()==pressed)
			return idx;

		idx++;
	}

	return -1;
}

void ButtonGroup::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_pressed_button:BaseButton"),&ButtonGroup::get_pressed_button);
	ClassDB::bind_method(D_METHOD("get_pressed_button_index"),&ButtonGroup::get_pressed_button_index);
	ClassDB::bind_method(D_METHOD("get_focused_button:BaseButton"),&ButtonGroup::get_focused_button);
	ClassDB::bind_method(D_METHOD("get_button_list"),&ButtonGroup::_get_button_list);
	ClassDB::bind_method(D_METHOD("_pressed"),&ButtonGroup::_pressed);
	ClassDB::bind_method(D_METHOD("set_pressed_button","button:BaseButton"),&ButtonGroup::_pressed);

	ADD_SIGNAL( MethodInfo("button_selected",PropertyInfo(Variant::OBJECT,"button",PROPERTY_HINT_RESOURCE_TYPE,"BaseButton")));
}

ButtonGroup::ButtonGroup() : BoxContainer(true)
{
}
#endif
