/*************************************************************************/
/*  test_containers.cpp                                                  */
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
#include "test_containers.h"

#include "dvector.h"
#include "math_funcs.h"
#include "print_string.h"
#include "servers/visual/default_mouse_cursor.xpm"
#include "set.h"

#include "image.h"
#include "list.h"
#include "variant.h"

namespace TestContainers {

MainLoop *test() {

	/*
	HashMap<int,int> int_map;

	for (int i=0;i<68000;i++) {

		int num=(int)Math::random(0,1024);
		int_map[i]=num;
	}
	*/

	{

		Image img;
		img.create(default_mouse_cursor_xpm);

		{
			for (int i = 0; i < 8; i++) {

				Image mipmap;
				//img.make_mipmap(mipmap);
				img = mipmap;
				if (img.get_width() <= 4) break;
			};
		};
	};

#if 0
	Set<int> set;

	print_line("Begin Insert");
	for (int i=0;i<1100;i++) {

		int num=i;//(int)Math::random(0,1024);
		//print_line("inserting "+itos(num));
		set.insert( num );
	}

	/*
	for (int i=0;i<400;i++) {

		int num=(int)Math::random(0,1024);
		set.erase(num);
	}
	*/
	//set.print_tree();

	for(Set<int>::Element *I=set.front();I;I=I->next()) {

		print_line("inserted "+itos(I->get())+" prev is "+itos(I->prev()?I->prev()->get():-100));

	}

	print_line("depth is "+itos(set.calculate_depth()));
	print_line("Insert Success");
#endif

	return NULL;
}
}
