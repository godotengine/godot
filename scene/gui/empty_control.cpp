/*************************************************************************/
/*  empty_control.cpp                                                    */
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
#include "empty_control.h"

Size2 EmptyControl::get_minimum_size() const {

	return minsize;
}

void EmptyControl::set_minsize(const Size2& p_size) {

	minsize=p_size;
	minimum_size_changed();
}

Size2 EmptyControl::get_minsize() const {

	return minsize;
}


void EmptyControl::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_minsize","minsize"),&EmptyControl::set_minsize);
	ObjectTypeDB::bind_method(_MD("get_minsize"),&EmptyControl::get_minsize);

	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"minsize"), _SCS("set_minsize"),_SCS("get_minsize") );
}

EmptyControl::EmptyControl()
{
}
