/*************************************************************************/
/* Copyright (c) 2015 dx, http://kaimi.ru                                */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person           */
/* obtaining a copy of this software and associated documentation        */
/* files (the "Software"), to deal in the Software without               */
/* restriction, including without limitation the rights to use,          */
/* copy, modify, merge, publish, distribute, sublicense, and/or          */
/* sell copies of the Software, and to permit persons to whom the        */
/* Software is furnished to do so, subject to the following conditions:  */
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
#pragma once

#define U16TEXT(t) reinterpret_cast<const unicode16_t*>( t )

#define StringFileInfo U16TEXT("S\0t\0r\0i\0n\0g\0F\0i\0l\0e\0I\0n\0f\0o\0\0")
#define SizeofStringFileInfo sizeof("S\0t\0r\0i\0n\0g\0F\0i\0l\0e\0I\0n\0f\0o\0\0")
#define VarFileInfo U16TEXT("V\0a\0r\0F\0i\0l\0e\0I\0n\0f\0o\0\0")
#define Translation U16TEXT("T\0r\0a\0n\0s\0l\0a\0t\0i\0o\0n\0\0")

#define VarFileInfoAligned U16TEXT("V\0a\0r\0F\0i\0l\0e\0I\0n\0f\0o\0\0\0\0")
#define TranslationAligned U16TEXT("T\0r\0a\0n\0s\0l\0a\0t\0i\0o\0n\0\0\0\0")
#define SizeofVarFileInfoAligned sizeof("V\0a\0r\0F\0i\0l\0e\0I\0n\0f\0o\0\0\0\0")
#define SizeofTranslationAligned sizeof("T\0r\0a\0n\0s\0l\0a\0t\0i\0o\0n\0\0\0\0")
