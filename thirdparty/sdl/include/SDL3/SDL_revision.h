/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

/* WIKI CATEGORY: Version */

/*
 * SDL_revision.h contains the SDL revision, which might be defined on the
 * compiler command line, or generated right into the header itself by the
 * build system.
 */

#ifndef SDL_revision_h_
#define SDL_revision_h_

#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * This macro is a string describing the source at a particular point in
 * development.
 *
 * This string is often generated from revision control's state at build time.
 *
 * This string can be quite complex and does not follow any standard. For
 * example, it might be something like "SDL-prerelease-3.1.1-47-gf687e0732".
 * It might also be user-defined at build time, so it's best to treat it as a
 * clue in debugging forensics and not something the app will parse in any
 * way.
 *
 * SDL_revision.h must be included in your program explicitly if you want
 * access to the SDL_REVISION constant.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_REVISION "Some arbitrary string decided at SDL build time"
#elif defined(SDL_VENDOR_INFO)
#define SDL_REVISION SDL_VENDOR_INFO
#else
#define SDL_REVISION ""
#endif

#endif /* SDL_revision_h_ */
