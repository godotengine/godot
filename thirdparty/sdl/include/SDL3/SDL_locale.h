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

/**
 * # CategoryLocale
 *
 * SDL locale services.
 *
 * This provides a way to get a list of preferred locales (language plus
 * country) for the user. There is exactly one function:
 * SDL_GetPreferredLocales(), which handles all the heavy lifting, and offers
 * documentation on all the strange ways humans might have configured their
 * language settings.
 */

#ifndef SDL_locale_h
#define SDL_locale_h

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
/* *INDENT-OFF* */
extern "C" {
/* *INDENT-ON* */
#endif

/**
 * A struct to provide locale data.
 *
 * Locale data is split into a spoken language, like English, and an optional
 * country, like Canada. The language will be in ISO-639 format (so English
 * would be "en"), and the country, if not NULL, will be an ISO-3166 country
 * code (so Canada would be "CA").
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPreferredLocales
 */
typedef struct SDL_Locale
{
    const char *language;  /**< A language name, like "en" for English. */
    const char *country;  /**< A country, like "US" for America. Can be NULL. */
} SDL_Locale;

/**
 * Report the user's preferred locale.
 *
 * Returned language strings are in the format xx, where 'xx' is an ISO-639
 * language specifier (such as "en" for English, "de" for German, etc).
 * Country strings are in the format YY, where "YY" is an ISO-3166 country
 * code (such as "US" for the United States, "CA" for Canada, etc). Country
 * might be NULL if there's no specific guidance on them (so you might get {
 * "en", "US" } for American English, but { "en", NULL } means "English
 * language, generically"). Language strings are never NULL, except to
 * terminate the array.
 *
 * Please note that not all of these strings are 2 characters; some are three
 * or more.
 *
 * The returned list of locales are in the order of the user's preference. For
 * example, a German citizen that is fluent in US English and knows enough
 * Japanese to navigate around Tokyo might have a list like: { "de", "en_US",
 * "jp", NULL }. Someone from England might prefer British English (where
 * "color" is spelled "colour", etc), but will settle for anything like it: {
 * "en_GB", "en", NULL }.
 *
 * This function returns NULL on error, including when the platform does not
 * supply this information at all.
 *
 * This might be a "slow" call that has to query the operating system. It's
 * best to ask for this once and save the results. However, this list can
 * change, usually because the user has changed a system preference outside of
 * your program; SDL will send an SDL_EVENT_LOCALE_CHANGED event in this case,
 * if possible, and you can call this function again to get an updated copy of
 * preferred locales.
 *
 * \param count a pointer filled in with the number of locales returned, may
 *              be NULL.
 * \returns a NULL terminated array of locale pointers, or NULL on failure;
 *          call SDL_GetError() for more information. This is a single
 *          allocation that should be freed with SDL_free() when it is no
 *          longer needed.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_Locale ** SDLCALL SDL_GetPreferredLocales(int *count);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
/* *INDENT-OFF* */
}
/* *INDENT-ON* */
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_locale_h */
