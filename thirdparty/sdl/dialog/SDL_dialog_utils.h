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
#include "SDL_internal.h"

/* The following are utility functions to help implementations.
   They are ordered by scope largeness, decreasing. All implementations
   should use them, as they check for invalid filters. Where they are unused,
   the validate_* function further down below should be used. */

/* Transform the name given in argument into something viable for the engine.
   Useful if there are special characters to avoid on certain platforms (such
   as "|" with Zenity). */
typedef char *(NameTransform)(const char * name);

// Converts all the filters into a single string.
// <prefix>[filter]{<separator>[filter]...}<suffix>
char *convert_filters(const SDL_DialogFileFilter *filters, int nfilters,
                      NameTransform ntf, const char *prefix,
                      const char *separator, const char *suffix,
                      const char *filt_prefix, const char *filt_separator,
                      const char *filt_suffix, const char *ext_prefix,
                      const char *ext_separator, const char *ext_suffix);

// Converts one filter into a single string.
// <prefix>[filter name]<separator>[filter extension list]<suffix>
char *convert_filter(SDL_DialogFileFilter filter, NameTransform ntf,
                     const char *prefix, const char *separator,
                     const char *suffix, const char *ext_prefix,
                     const char *ext_separator, const char *ext_suffix);

// Converts the extension list of a filter into a single string.
// <prefix>[extension]{<separator>[extension]...}<suffix>
char *convert_ext_list(const char *list, const char *prefix,
                       const char *separator, const char *suffix);

/* Must be used if convert_* functions aren't used */
// Returns an error message if there's a problem, NULL otherwise
const char *validate_filters(const SDL_DialogFileFilter *filters,
                             int nfilters);

const char *validate_list(const char *list);
