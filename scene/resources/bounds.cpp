/*************************************************************************/
/*  bounds.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "bounds.h"


void Bounds::_bind_methods() {

	ClassDB::bind_method( _MD("set_bsp_tree","bsp_tree"),&Bounds::set_bsp_tree);
	ClassDB::bind_method( _MD("get_bsp_tree"),&Bounds::get_bsp_tree );

	ADD_PROPERTY( PropertyInfo( Variant::ARRAY, "bsp_tree" ), _SCS("set_bsp_tree"), _SCS("get_bsp_tree"));

}

void Bounds::set_bsp_tree(const BSP_Tree& p_bsp_tree) {

	bsp_tree=p_bsp_tree;
}

BSP_Tree Bounds::get_bsp_tree() const {

	return bsp_tree;
}


Bounds::Bounds()
{
}
