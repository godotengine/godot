/*************************************************************************/
/*  ip_address.cpp                                                       */
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
#include "ip_address.h"
/*
IP_Address::operator Variant() const {

	return operator String();
}*/
IP_Address::operator String() const {

	return itos(field[0])+"."+itos(field[1])+"."+itos(field[2])+"."+itos(field[3]);
}

IP_Address::IP_Address(const String& p_string) {

	host=0;
	int slices = p_string.get_slice_count(".");
	if (slices!=4) {
		ERR_EXPLAIN("Invalid IP Address String: "+p_string);
		ERR_FAIL();
	}
	for(int i=0;i<4;i++) {

		field[i]=p_string.get_slicec('.',i).to_int();
	}
}

IP_Address::IP_Address(uint8_t p_a,uint8_t p_b,uint8_t p_c,uint8_t p_d) {

	field[0]=p_a;
	field[1]=p_b;
	field[2]=p_c;
	field[3]=p_d;
}
