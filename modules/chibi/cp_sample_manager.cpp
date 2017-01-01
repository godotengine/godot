/*************************************************************************/
/*  cp_sample_manager.cpp                                                */
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
#include "cp_sample_manager.h"


CPSampleManager * CPSampleManager::singleton=NULL;


void CPSampleManager::copy_to(CPSample_ID p_from,CPSample_ID &p_to) {
	
	ERR_FAIL_COND(!check( p_from ));

	
	if (p_to.is_null()) {
		
		p_to=create( is_16bits( p_from), is_stereo( p_from), get_size(p_from));
	} else {
		
		recreate( p_to, is_16bits( p_from), is_stereo( p_from), get_size(p_from));
		
	}
	
	int len=get_size( p_from );
	int ch=is_stereo( p_from ) ? 2 : 1;
	
	for (int c=0;c<ch;c++) {
		
		for (int i=0;i<len;i++) {
			
			int16_t s=get_data( p_from, i, c );
			set_data( p_to, i, s, c );
		}
	}
	
	set_loop_type( p_to, get_loop_type( p_from ) );
	set_loop_begin( p_to, get_loop_begin( p_from ) );
	set_loop_end( p_to, get_loop_end( p_from ) );
	set_c5_freq( p_to, get_c5_freq( p_from ) );
	
		
	
}

CPSampleManager::CPSampleManager() {

	singleton=this;
}

CPSampleManager *CPSampleManager::get_singleton() {

	return singleton;
}
