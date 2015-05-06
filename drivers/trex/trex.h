#ifndef _TREX_H_
#define _TREX_H_
/***************************************************************
	T-Rex a tiny regular expression library

	Copyright (C) 2003-2006 Alberto Demichelis

	This software is provided 'as-is', without any express 
	or implied warranty. In no event will the authors be held 
	liable for any damages arising from the use of this software.

	Permission is granted to anyone to use this software for 
	any purpose, including commercial applications, and to alter
	it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented;
		you must not claim that you wrote the original software.
		If you use this software in a product, an acknowledgment
		in the product documentation would be appreciated but
		is not required.

		2. Altered source versions must be plainly marked as such,
		and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any
		source distribution.

****************************************************************/

#define _UNICODE


#ifdef _UNICODE
#define TRexChar wchar_t
#define MAX_CHAR 0xFFFF
#define _TREXC(c) L##c 
#define trex_strlen wcslen
#define trex_printf wprintf
#else
#define TRexChar char
#define MAX_CHAR 0xFF
#define _TREXC(c) (c) 
#define trex_strlen strlen
#define trex_printf printf
#endif

#ifndef TREX_API
#define TREX_API extern
#endif

#define TRex_True 1
#define TRex_False 0

typedef unsigned int TRexBool;
typedef struct TRex TRex;

typedef struct {
	const TRexChar *begin;
	int len;
} TRexMatch;

TREX_API TRex *trex_compile(const TRexChar *pattern,const TRexChar **error);
TREX_API void trex_free(TRex *exp);
TREX_API TRexBool trex_match(TRex* exp,const TRexChar* text);
TREX_API TRexBool trex_search(TRex* exp,const TRexChar* text, const TRexChar** out_begin, const TRexChar** out_end);
TREX_API TRexBool trex_searchrange(TRex* exp,const TRexChar* text_begin,const TRexChar* text_end,const TRexChar** out_begin, const TRexChar** out_end);
TREX_API int trex_getsubexpcount(TRex* exp);
TREX_API TRexBool trex_getsubexp(TRex* exp, int n, TRexMatch *subexp);

#endif
