/*************************************************************************/
/*  copymem.h                                                            */
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
#ifndef COPYMEM_H
#define COPYMEM_H

#include "typedefs.h"

///@TODO use optimized routines for this, depending on platform. these are just the standard ones

#define copymem(m_to,m_from,m_count) \
	do {			 	\
		unsigned char * _from=(unsigned char*)m_from;	\
		unsigned char * _to=(unsigned char*)m_to;	\
		int _count=m_count;		\
		for (int _i=0;_i<_count;_i++)	\
			_to[_i]=_from[_i];	\
	} while (0);
	/*
					case 0:	*_dto++ = *_dfrom++;	\
					case 7:	*_dto++ = *_dfrom++;	\
							case 6:	*_dto++ = *_dfrom++;	\
									case 5:	*_dto++ = *_dfrom++;	\
										case 4:	*_dto++ = *_dfrom++;	\
										case 3:	*_dto++ = *_dfrom++;	\
										case 2:	*_dto++ = *_dfrom++;	\
										case 1:	*_dto++ = *_dfrom++;	\
	*/
#define movemem_duff(m_to, m_from, m_count) \
	do { \
		if (m_to<m_from) {	\
			unsigned char* _dto = (unsigned char*)m_to;	\
			unsigned char* _dfrom = (unsigned char*)m_from;	\
			int n = (m_count + 7) / 8;	\
			switch (m_count % 8) {	\
				do {					\
					case 0:	*_dto++ = *_dfrom++;	\
					case 7:	*_dto++ = *_dfrom++;	\
					case 6:	*_dto++ = *_dfrom++;	\
					case 5:	*_dto++ = *_dfrom++;	\
					case 4:	*_dto++ = *_dfrom++;	\
					case 3:	*_dto++ = *_dfrom++;	\
					case 2:	*_dto++ = *_dfrom++;	\
					case 1:	*_dto++ = *_dfrom++;	\
				} while (--n > 0);			\
			};						\
		} else if (m_to>m_from) {					\
			unsigned char* _dto = &((unsigned char*)m_to)[m_count-1];	\
			unsigned char* _dfrom = &((unsigned char*)m_from)[m_count-1];	\
			int n = (m_count + 7) / 8;	\
			switch (m_count % 8) {	\
				do {	\
					case 0:	*_dto-- = *_dfrom--;	\
					case 7:	*_dto-- = *_dfrom--;	\
					case 6:	*_dto-- = *_dfrom--;	\
					case 5:	*_dto-- = *_dfrom--;	\
					case 4:	*_dto-- = *_dfrom--;	\
					case 3:	*_dto-- = *_dfrom--;	\
					case 2:	*_dto-- = *_dfrom--;	\
					case 1:	*_dto-- = *_dfrom--;	\
				} while (--n > 0);			\
			};						\
		}                           				\
	} while(0)                           				\

#define movemem_conventional(m_to,m_from,m_count) \
	do {			 	\
		if (m_to<m_from) {					\
			unsigned char * _from=(unsigned char*)m_from;	\
			unsigned char * _to=(unsigned char*)m_to;	\
			int _count=m_count;				\
			for (int _i=0;_i<_count;_i++)			\
				_to[_i]=_from[_i];			\
									\
		} else if (m_to>m_from) {				\
			unsigned char * _from=(unsigned char*)m_from;	\
			unsigned char * _to=(unsigned char*)m_to;	\
			int _count=m_count;				\
			while (_count--) 				\
				_to[_count]=_from[_count];		\
									\
									\
		}							\
	} while (0);							\

void movemem_system(void*,void*,int);

#define movemem movemem_system


void zeromem(void* p_mem,size_t p_bytes);

#endif

