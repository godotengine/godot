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

#include "SDL_dialog_utils.h"

char *convert_filters(const SDL_DialogFileFilter *filters, int nfilters,
                      NameTransform ntf, const char *prefix,
                      const char *separator, const char *suffix,
                      const char *filt_prefix, const char *filt_separator,
                      const char *filt_suffix, const char *ext_prefix,
                      const char *ext_separator, const char *ext_suffix)
{
    char *combined;
    char *new_combined;
    char *converted;
    const char *terminator;
    size_t new_length;
    int i;

    if (!filters) {
        SDL_SetError("Called convert_filters() with NULL filters (SDL bug)");
        return NULL;
    }

    combined = SDL_strdup(prefix);

    if (!combined) {
        return NULL;
    }

    for (i = 0; i < nfilters; i++) {
        const SDL_DialogFileFilter *f = &filters[i];

        converted = convert_filter(*f, ntf, filt_prefix, filt_separator,
                                   filt_suffix, ext_prefix, ext_separator,
                                   ext_suffix);

        if (!converted) {
            SDL_free(combined);
            return NULL;
        }

        terminator = ((i + 1) < nfilters) ? separator : suffix;
        new_length = SDL_strlen(combined) + SDL_strlen(converted)
                   + SDL_strlen(terminator) + 1;

        new_combined = (char *)SDL_realloc(combined, new_length);

        if (!new_combined) {
            SDL_free(converted);
            SDL_free(combined);
            return NULL;
        }

        combined = new_combined;

        SDL_strlcat(combined, converted, new_length);
        SDL_strlcat(combined, terminator, new_length);
        SDL_free(converted);
    }

    new_length = SDL_strlen(combined) + SDL_strlen(suffix) + 1;

    new_combined = (char *)SDL_realloc(combined, new_length);

    if (!new_combined) {
        SDL_free(combined);
        return NULL;
    }

    combined = new_combined;

    SDL_strlcat(combined, suffix, new_length);

    return combined;
}

char *convert_filter(SDL_DialogFileFilter filter, NameTransform ntf,
                     const char *prefix, const char *separator,
                     const char *suffix, const char *ext_prefix,
                     const char *ext_separator, const char *ext_suffix)
{
    char *converted;
    char *name_filtered;
    size_t total_length;
    char *list;

    list = convert_ext_list(filter.pattern, ext_prefix, ext_separator,
                            ext_suffix);

    if (!list) {
        return NULL;
    }

    if (ntf) {
        name_filtered = ntf(filter.name);
    } else {
        // Useless strdup, but easier to read and maintain code this way
        name_filtered = SDL_strdup(filter.name);
    }

    if (!name_filtered) {
        SDL_free(list);
        return NULL;
    }

    total_length = SDL_strlen(prefix) + SDL_strlen(name_filtered)
                 + SDL_strlen(separator) + SDL_strlen(list)
                 + SDL_strlen(suffix) + 1;

    converted = (char *) SDL_malloc(total_length);

    if (!converted) {
        SDL_free(list);
        SDL_free(name_filtered);
        return NULL;
    }

    SDL_snprintf(converted, total_length, "%s%s%s%s%s", prefix, name_filtered,
                 separator, list, suffix);

    SDL_free(list);
    SDL_free(name_filtered);

    return converted;
}

char *convert_ext_list(const char *list, const char *prefix,
                       const char *separator, const char *suffix)
{
    char *converted;
    int semicolons;
    size_t total_length;

    semicolons = 0;

    for (const char *c = list; *c; c++) {
        semicolons += (*c == ';');
    }

    total_length =
        SDL_strlen(list) - semicolons // length of list contents
      + semicolons * SDL_strlen(separator) // length of separators
      + SDL_strlen(prefix) + SDL_strlen(suffix) // length of prefix/suffix
      + 1; // terminating null byte

    converted = (char *) SDL_malloc(total_length);

    if (!converted) {
        return NULL;
    }

    *converted = '\0';

    SDL_strlcat(converted, prefix, total_length);

    /* Some platforms may prefer to handle the asterisk manually, but this
       function offers to handle it for ease of use. */
    if (SDL_strcmp(list, "*") == 0) {
        SDL_strlcat(converted, "*", total_length);
    } else {
        for (const char *c = list; *c; c++) {
            if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z')
             || (*c >= '0' && *c <= '9') || *c == '-' || *c == '_'
             || *c == '.') {
                char str[2];
                str[0] = *c;
                str[1] = '\0';
                SDL_strlcat(converted, str, total_length);
            } else if (*c == ';') {
                if (c == list || c[-1] == ';') {
                    SDL_SetError("Empty pattern not allowed");
                    SDL_free(converted);
                    return NULL;
                }

                SDL_strlcat(converted, separator, total_length);
            } else {
                SDL_SetError("Invalid character '%c' in pattern (Only [a-zA-Z0-9_.-] allowed, or a single *)", *c);
                SDL_free(converted);
                return NULL;
            }
        }
    }

    if (list[SDL_strlen(list) - 1] == ';') {
        SDL_SetError("Empty pattern not allowed");
        SDL_free(converted);
        return NULL;
    }

    SDL_strlcat(converted, suffix, total_length);

    return converted;
}

const char *validate_filters(const SDL_DialogFileFilter *filters, int nfilters)
{
    if (filters) {
        for (int i = 0; i < nfilters; i++) {
             const char *msg = validate_list(filters[i].pattern);

             if (msg) {
                 return msg;
             }
        }
    }

    return NULL;
}

const char *validate_list(const char *list)
{
    if (SDL_strcmp(list, "*") == 0) {
        return NULL;
    } else {
        for (const char *c = list; *c; c++) {
            if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z')
             || (*c >= '0' && *c <= '9') || *c == '-' || *c == '_'
             || *c == '.') {
                continue;
            } else if (*c == ';') {
                if (c == list || c[-1] == ';') {
                    return "Empty pattern not allowed";
                }
            } else {
                return "Invalid character in pattern (Only [a-zA-Z0-9_.-] allowed, or a single *)";
            }
        }
    }

    if (list[SDL_strlen(list) - 1] == ';') {
        return "Empty pattern not allowed";
    }

    return NULL;
}
