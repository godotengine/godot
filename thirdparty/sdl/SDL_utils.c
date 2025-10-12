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

#if defined(HAVE_GETHOSTNAME) && !defined(SDL_PLATFORM_WINDOWS)
#include <unistd.h>
#endif

#include "joystick/SDL_joystick_c.h" // For SDL_GetGamepadTypeFromVIDPID()


// Common utility functions that aren't in the public API

int SDL_powerof2(int x)
{
    int value;

    if (x <= 0) {
        // Return some sane value - we shouldn't hit this in our use cases
        return 1;
    }

    // This trick works for 32-bit values
    {
        SDL_COMPILE_TIME_ASSERT(SDL_powerof2, sizeof(x) == sizeof(Uint32));
    }
    value = x;
    value -= 1;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value += 1;

    return value;
}

Uint32 SDL_CalculateGCD(Uint32 a, Uint32 b)
{
    if (b == 0) {
        return a;
    }
    return SDL_CalculateGCD(b, (a % b));
}

// Algorithm adapted with thanks from John Cook's blog post:
// http://www.johndcook.com/blog/2010/10/20/best-rational-approximation
void SDL_CalculateFraction(float x, int *numerator, int *denominator)
{
    const int N = 1000;
    int a = 0, b = 1;
    int c = 1, d = 0;

    while (b <= N && d <= N) {
        float mediant = (float)(a + c) / (b + d);
        if (x == mediant) {
            if (b + d <= N) {
                *numerator = a + c;
                *denominator = b + d;
            } else if (d > b) {
                *numerator = c;
                *denominator = d;
            } else {
                *numerator = a;
                *denominator = b;
            }
            return;
        } else if (x > mediant) {
            a = a + c;
            b = b + d;
        } else {
            c = a + c;
            d = b + d;
        }
    }
    if (b > N) {
        *numerator = c;
        *denominator = d;
    } else {
        *numerator = a;
        *denominator = b;
    }
}

bool SDL_startswith(const char *string, const char *prefix)
{
    if (SDL_strncmp(string, prefix, SDL_strlen(prefix)) == 0) {
        return true;
    }
    return false;
}

bool SDL_endswith(const char *string, const char *suffix)
{
    size_t string_length = string ? SDL_strlen(string) : 0;
    size_t suffix_length = suffix ? SDL_strlen(suffix) : 0;

    if (suffix_length > 0 && suffix_length <= string_length) {
        if (SDL_memcmp(string + string_length - suffix_length, suffix, suffix_length) == 0) {
            return true;
        }
    }
    return false;
}

SDL_COMPILE_TIME_ASSERT(sizeof_object_id, sizeof(int) == sizeof(Uint32));

Uint32 SDL_GetNextObjectID(void)
{
    static SDL_AtomicInt last_id;

    Uint32 id = (Uint32)SDL_AtomicIncRef(&last_id) + 1;
    if (id == 0) {
        id = (Uint32)SDL_AtomicIncRef(&last_id) + 1;
    }
    return id;
}

static SDL_InitState SDL_objects_init;
static SDL_HashTable *SDL_objects;

static Uint32 SDLCALL SDL_HashObject(void *unused, const void *key)
{
    return (Uint32)(uintptr_t)key;
}

static bool SDL_KeyMatchObject(void *unused, const void *a, const void *b)
{
    return (a == b);
}

void SDL_SetObjectValid(void *object, SDL_ObjectType type, bool valid)
{
    SDL_assert(object != NULL);

    if (SDL_ShouldInit(&SDL_objects_init)) {
        SDL_objects = SDL_CreateHashTable(0, true, SDL_HashObject, SDL_KeyMatchObject, NULL, NULL);
        const bool initialized = (SDL_objects != NULL);
        SDL_SetInitialized(&SDL_objects_init, initialized);
        if (!initialized) {
            return;
        }
    }

    if (valid) {
        SDL_InsertIntoHashTable(SDL_objects, object, (void *)(uintptr_t)type, true);
    } else {
        SDL_RemoveFromHashTable(SDL_objects, object);
    }
}

bool SDL_ObjectValid(void *object, SDL_ObjectType type)
{
    if (!object) {
        return false;
    }

    const void *object_type;
    if (!SDL_FindInHashTable(SDL_objects, object, &object_type)) {
        return false;
    }

    return (((SDL_ObjectType)(uintptr_t)object_type) == type);
}

typedef struct GetOneObjectData
{
    const SDL_ObjectType type;
    void **objects;
    const int count;
    int num_objects;
} GetOneObjectData;

static bool SDLCALL GetOneObject(void *userdata, const SDL_HashTable *table, const void *object, const void *object_type)
{
    GetOneObjectData *data = (GetOneObjectData *) userdata;
    if ((SDL_ObjectType)(uintptr_t)object_type == data->type) {
        if (data->num_objects < data->count) {
            data->objects[data->num_objects] = (void *)object;
        }
        ++data->num_objects;
    }
    return true;  // keep iterating.
}


int SDL_GetObjects(SDL_ObjectType type, void **objects, int count)
{
    GetOneObjectData data = { type, objects, count, 0 };
    SDL_IterateHashTable(SDL_objects, GetOneObject, &data);
    return data.num_objects;
}

static bool SDLCALL LogOneLeakedObject(void *userdata, const SDL_HashTable *table, const void *object, const void *object_type)
{
    const char *type = "unknown object";
    switch ((SDL_ObjectType)(uintptr_t)object_type) {
        #define SDLOBJTYPECASE(typ, name) case SDL_OBJECT_TYPE_##typ: type = name; break
        SDLOBJTYPECASE(WINDOW, "SDL_Window");
        SDLOBJTYPECASE(RENDERER, "SDL_Renderer");
        SDLOBJTYPECASE(TEXTURE, "SDL_Texture");
        SDLOBJTYPECASE(JOYSTICK, "SDL_Joystick");
        SDLOBJTYPECASE(GAMEPAD, "SDL_Gamepad");
        SDLOBJTYPECASE(HAPTIC, "SDL_Haptic");
        SDLOBJTYPECASE(SENSOR, "SDL_Sensor");
        SDLOBJTYPECASE(HIDAPI_DEVICE, "hidapi device");
        SDLOBJTYPECASE(HIDAPI_JOYSTICK, "hidapi joystick");
        SDLOBJTYPECASE(THREAD, "thread");
        SDLOBJTYPECASE(TRAY, "SDL_Tray");
        #undef SDLOBJTYPECASE
        default: break;
    }
    SDL_Log("Leaked %s (%p)", type, object);
    return true;  // keep iterating.
}

void SDL_SetObjectsInvalid(void)
{
    if (SDL_ShouldQuit(&SDL_objects_init)) {
        // Log any leaked objects
        SDL_IterateHashTable(SDL_objects, LogOneLeakedObject, NULL);
        SDL_assert(SDL_HashTableEmpty(SDL_objects));
        SDL_DestroyHashTable(SDL_objects);
        SDL_objects = NULL;
        SDL_SetInitialized(&SDL_objects_init, false);
    }
}

static int SDL_URIDecode(const char *src, char *dst, int len)
{
    int ri, wi, di;
    char decode = '\0';
    if (!src || !dst || len < 0) {
        return -1;
    }
    if (len == 0) {
        len = (int)SDL_strlen(src);
    }
    for (ri = 0, wi = 0, di = 0; ri < len && wi < len; ri += 1) {
        if (di == 0) {
            // start decoding
            if (src[ri] == '%') {
                decode = '\0';
                di += 1;
                continue;
            }
            // normal write
            dst[wi] = src[ri];
            wi += 1;
        } else if (di == 1 || di == 2) {
            char off = '\0';
            char isa = src[ri] >= 'a' && src[ri] <= 'f';
            char isA = src[ri] >= 'A' && src[ri] <= 'F';
            char isn = src[ri] >= '0' && src[ri] <= '9';
            if (!(isa || isA || isn)) {
                // not a hexadecimal
                int sri;
                for (sri = ri - di; sri <= ri; sri += 1) {
                    dst[wi] = src[sri];
                    wi += 1;
                }
                di = 0;
                continue;
            }
            // itsy bitsy magicsy
            if (isn) {
                off = 0 - '0';
            } else if (isa) {
                off = 10 - 'a';
            } else if (isA) {
                off = 10 - 'A';
            }
            decode |= (src[ri] + off) << (2 - di) * 4;
            if (di == 2) {
                dst[wi] = decode;
                wi += 1;
                di = 0;
            } else {
                di += 1;
            }
        }
    }
    dst[wi] = '\0';
    return wi;
}

int SDL_URIToLocal(const char *src, char *dst)
{
    if (SDL_memcmp(src, "file:/", 6) == 0) {
        src += 6; // local file?
    } else if (SDL_strstr(src, ":/") != NULL) {
        return -1; // wrong scheme
    }

    bool local = src[0] != '/' || (src[0] != '\0' && src[1] == '/');

    // Check the hostname, if present. RFC 3986 states that the hostname component of a URI is not case-sensitive.
    if (!local && src[0] == '/' && src[2] != '/') {
        char *hostname_end = SDL_strchr(src + 1, '/');
        if (hostname_end) {
            const size_t src_len = hostname_end - (src + 1);
            size_t hostname_len;

#if defined(HAVE_GETHOSTNAME) && !defined(SDL_PLATFORM_WINDOWS)
            char hostname[257];
            if (gethostname(hostname, 255) == 0) {
                hostname[256] = '\0';
                hostname_len = SDL_strlen(hostname);
                if (hostname_len == src_len && SDL_strncasecmp(src + 1, hostname, src_len) == 0) {
                    src = hostname_end + 1;
                    local = true;
                }
            }
#endif

            if (!local) {
                static const char *localhost = "localhost";
                hostname_len = SDL_strlen(localhost);
                if (hostname_len == src_len && SDL_strncasecmp(src + 1, localhost, src_len) == 0) {
                    src = hostname_end + 1;
                    local = true;
                }
            }
        }
    }

    if (local) {
        // Convert URI escape sequences to real characters
        if (src[0] == '/') {
            src++;
        } else {
            src--;
        }
        return SDL_URIDecode(src, dst, 0);
    }
    return -1;
}

// This is a set of per-thread persistent strings that we can return from the SDL API.
// This is used for short strings that might persist past the lifetime of the object
// they are related to.

static SDL_TLSID SDL_string_storage;

static void SDL_FreePersistentStrings( void *value )
{
    SDL_HashTable *strings = (SDL_HashTable *)value;
    SDL_DestroyHashTable(strings);
}

const char *SDL_GetPersistentString(const char *string)
{
    if (!string) {
        return NULL;
    }
    if (!*string) {
        return "";
    }

    SDL_HashTable *strings = (SDL_HashTable *)SDL_GetTLS(&SDL_string_storage);
    if (!strings) {
        strings = SDL_CreateHashTable(0, false, SDL_HashString, SDL_KeyMatchString, SDL_DestroyHashValue, NULL);
        if (!strings) {
            return NULL;
        }

        SDL_SetTLS(&SDL_string_storage, strings, SDL_FreePersistentStrings);
    }

    const char *result;
    if (!SDL_FindInHashTable(strings, string, (const void **)&result)) {
        char *new_string = SDL_strdup(string);
        if (!new_string) {
            return NULL;
        }

        // If the hash table insert fails, at least we can return the string we allocated
        SDL_InsertIntoHashTable(strings, new_string, new_string, false);
        result = new_string;
    }
    return result;
}

static int PrefixMatch(const char *a, const char *b)
{
    int matchlen = 0;
    // Fixes the "HORI HORl Taiko No Tatsujin Drum Controller"
    if (SDL_strncmp(a, "HORI ", 5) == 0 && SDL_strncmp(b, "HORl ", 5) == 0) {
        return 5;
    }
    while (*a && *b) {
        if (SDL_tolower((unsigned char)*a++) == SDL_tolower((unsigned char)*b++)) {
            ++matchlen;
        } else {
            break;
        }
    }
    return matchlen;
}

char *SDL_CreateDeviceName(Uint16 vendor, Uint16 product, const char *vendor_name, const char *product_name, const char *default_name)
{
    static struct
    {
        const char *prefix;
        const char *replacement;
    } replacements[] = {
        { "8BitDo Tech Ltd", "8BitDo" },
        { "ASTRO Gaming", "ASTRO" },
        { "Bensussen Deutsch & Associates,Inc.(BDA)", "BDA" },
        { "Guangzhou Chicken Run Network Technology Co., Ltd.", "GameSir" },
        { "HORI CO.,LTD.", "HORI" },
        { "HORI CO.,LTD", "HORI" },
        { "Mad Catz Inc.", "Mad Catz" },
        { "Nintendo Co., Ltd.", "Nintendo" },
        { "NVIDIA Corporation ", "" },
        { "Performance Designed Products", "PDP" },
        { "QANBA USA, LLC", "Qanba" },
        { "QANBA USA,LLC", "Qanba" },
        { "Unknown ", "" },
    };
    char *name = NULL;
    size_t i, len;

    if (!vendor_name) {
        vendor_name = "";
    }
    if (!product_name) {
        product_name = "";
    }

    while (*vendor_name == ' ') {
        ++vendor_name;
    }
    while (*product_name == ' ') {
        ++product_name;
    }

    if (*vendor_name && *product_name) {
        len = (SDL_strlen(vendor_name) + 1 + SDL_strlen(product_name) + 1);
        name = (char *)SDL_malloc(len);
        if (name) {
            (void)SDL_snprintf(name, len, "%s %s", vendor_name, product_name);
        }
    } else if (*product_name) {
        name = SDL_strdup(product_name);
    } else if (vendor || product) {
        // Couldn't find a controller name, try to give it one based on device type
        switch (SDL_GetGamepadTypeFromVIDPID(vendor, product, NULL, true)) {
        case SDL_GAMEPAD_TYPE_XBOX360:
            name = SDL_strdup("Xbox 360 Controller");
            break;
        case SDL_GAMEPAD_TYPE_XBOXONE:
            name = SDL_strdup("Xbox One Controller");
            break;
        case SDL_GAMEPAD_TYPE_PS3:
            name = SDL_strdup("PS3 Controller");
            break;
        case SDL_GAMEPAD_TYPE_PS4:
            name = SDL_strdup("PS4 Controller");
            break;
        case SDL_GAMEPAD_TYPE_PS5:
            name = SDL_strdup("DualSense Wireless Controller");
            break;
        case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
            name = SDL_strdup("Nintendo Switch Pro Controller");
            break;
        default:
            len = (6 + 1 + 6 + 1);
            name = (char *)SDL_malloc(len);
            if (name) {
                (void)SDL_snprintf(name, len, "0x%.4x/0x%.4x", vendor, product);
            }
            break;
        }
    } else if (default_name) {
        name = SDL_strdup(default_name);
    }

    if (!name) {
        return NULL;
    }

    // Trim trailing whitespace
    for (len = SDL_strlen(name); (len > 0 && name[len - 1] == ' '); --len) {
        // continue
    }
    name[len] = '\0';

    // Compress duplicate spaces
    for (i = 0; i < (len - 1);) {
        if (name[i] == ' ' && name[i + 1] == ' ') {
            SDL_memmove(&name[i], &name[i + 1], (len - i));
            --len;
        } else {
            ++i;
        }
    }

    // Perform any manufacturer replacements
    for (i = 0; i < SDL_arraysize(replacements); ++i) {
        size_t prefixlen = SDL_strlen(replacements[i].prefix);
        if (SDL_strncasecmp(name, replacements[i].prefix, prefixlen) == 0) {
            size_t replacementlen = SDL_strlen(replacements[i].replacement);
            if (replacementlen <= prefixlen) {
                SDL_memcpy(name, replacements[i].replacement, replacementlen);
                SDL_memmove(name + replacementlen, name + prefixlen, (len - prefixlen) + 1);
                len -= (prefixlen - replacementlen);
            } else {
                // FIXME: Need to handle the expand case by reallocating the string
            }
            break;
        }
    }

    /* Remove duplicate manufacturer or product in the name
     * e.g. Razer Razer Raiju Tournament Edition Wired
     */
    for (i = 1; i < (len - 1); ++i) {
        int matchlen = PrefixMatch(name, &name[i]);
        while (matchlen > 0) {
            if (name[matchlen] == ' ' || name[matchlen] == '-') {
                SDL_memmove(name, name + matchlen + 1, len - matchlen);
                break;
            }
            --matchlen;
        }
        if (matchlen > 0) {
            // We matched the manufacturer's name and removed it
            break;
        }
    }

    return name;
}
