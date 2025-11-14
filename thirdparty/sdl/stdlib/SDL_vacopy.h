/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

// Do our best to make sure va_copy is working
#if defined(_MSC_VER) && _MSC_VER <= 1800
// Visual Studio 2013 tries to link with _vacopy in the C runtime. Newer versions do an inline assignment
#undef va_copy
#define va_copy(dst, src) dst = src

#elif defined(__GNUC__) && (__GNUC__ < 3)
#define va_copy(dst, src) __va_copy(dst, src)
#endif
