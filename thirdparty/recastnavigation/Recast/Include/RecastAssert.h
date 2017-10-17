//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef RECASTASSERT_H
#define RECASTASSERT_H

// Note: This header file's only purpose is to include define assert.
// Feel free to change the file and include your own implementation instead.

#ifdef NDEBUG

// From http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
#	define rcAssert(x) do { (void)sizeof(x); } while((void)(__LINE__==-1),false)

#else

/// An assertion failure function.
//  @param[in]		expression  asserted expression.
//  @param[in]		file  Filename of the failed assertion.
//  @param[in]		line  Line number of the failed assertion.
///  @see rcAssertFailSetCustom
typedef void (rcAssertFailFunc)(const char* expression, const char* file, int line);

/// Sets the base custom assertion failure function to be used by Recast.
///  @param[in]		assertFailFunc	The function to be used in case of failure of #dtAssert
void rcAssertFailSetCustom(rcAssertFailFunc *assertFailFunc);

/// Gets the base custom assertion failure function to be used by Recast.
rcAssertFailFunc* rcAssertFailGetCustom();

#	include <assert.h> 
#	define rcAssert(expression) \
		{ \
			rcAssertFailFunc* failFunc = rcAssertFailGetCustom(); \
			if(failFunc == NULL) { assert(expression); } \
			else if(!(expression)) { (*failFunc)(#expression, __FILE__, __LINE__); } \
		}

#endif

#endif // RECASTASSERT_H
