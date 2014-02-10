/*************************************************************************/
/*  center_container.cpp                                                 */
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
#include "center_container.h"


Size2 CenterContainer::get_minimum_size() const {


	Size2 ms;
	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (c->is_hidden())
			continue;
		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width , minsize.width);
		ms.height = MAX(ms.height , minsize.height);


	}

	return ms;

}

void CenterContainer::_notification(int p_what) {

	if (p_what==NOTIFICATION_SORT_CHILDREN) {

		Size2 size = get_size();
		for(int i=0;i<get_child_count();i++) {

			Control *c = get_child(i)->cast_to<Control>();
			if (!c)
				continue;
			if (c->is_set_as_toplevel())
				continue;
			Size2 minsize = c->get_combined_minimum_size();
			Point2 ofs = ((size - minsize)/2.0).floor();
			fit_child_in_rect(c,Rect2(ofs,minsize));

		}
	}
}

CenterContainer::CenterContainer()
{
}
