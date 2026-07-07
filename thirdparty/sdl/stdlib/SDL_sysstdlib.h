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

#ifndef SDL_sysstdlib_h_
#define SDL_sysstdlib_h_

// most things you might need internally in here are public APIs, this is
// just a few special pieces right now.

// this expects `from` to be a Unicode codepoint, and `to` to point to AT LEAST THREE Uint32s.
int SDL_CaseFoldUnicode(Uint32 from, Uint32 *to);

#endif

