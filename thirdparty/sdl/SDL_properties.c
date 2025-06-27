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

#include "SDL_hints_c.h"
#include "SDL_properties_c.h"


typedef struct
{
    SDL_PropertyType type;

    union {
        void *pointer_value;
        char *string_value;
        Sint64 number_value;
        float float_value;
        bool boolean_value;
    } value;

    char *string_storage;

    SDL_CleanupPropertyCallback cleanup;
    void *userdata;
} SDL_Property;

typedef struct
{
    SDL_HashTable *props;
    SDL_Mutex *lock;
} SDL_Properties;

static SDL_InitState SDL_properties_init;
static SDL_HashTable *SDL_properties;
static SDL_AtomicU32 SDL_last_properties_id;
static SDL_AtomicU32 SDL_global_properties;


static void SDL_FreePropertyWithCleanup(const void *key, const void *value, void *data, bool cleanup)
{
    SDL_Property *property = (SDL_Property *)value;
    if (property) {
        switch (property->type) {
        case SDL_PROPERTY_TYPE_POINTER:
            if (property->cleanup && cleanup) {
                property->cleanup(property->userdata, property->value.pointer_value);
            }
            break;
        case SDL_PROPERTY_TYPE_STRING:
            SDL_free(property->value.string_value);
            break;
        default:
            break;
        }
        SDL_free(property->string_storage);
    }
    SDL_free((void *)key);
    SDL_free((void *)value);
}

static void SDLCALL SDL_FreeProperty(void *data, const void *key, const void *value)
{
    SDL_FreePropertyWithCleanup(key, value, data, true);
}

static void SDL_FreeProperties(SDL_Properties *properties)
{
    if (properties) {
        SDL_DestroyHashTable(properties->props);
        SDL_DestroyMutex(properties->lock);
        SDL_free(properties);
    }
}

bool SDL_InitProperties(void)
{
    if (!SDL_ShouldInit(&SDL_properties_init)) {
        return true;
    }

    SDL_properties = SDL_CreateHashTable(0, true, SDL_HashID, SDL_KeyMatchID, NULL, NULL);
    const bool initialized = (SDL_properties != NULL);
    SDL_SetInitialized(&SDL_properties_init, initialized);
    return initialized;
}

static bool SDLCALL FreeOneProperties(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    SDL_FreeProperties((SDL_Properties *)value);
    return true;  // keep iterating.
}

void SDL_QuitProperties(void)
{
    if (!SDL_ShouldQuit(&SDL_properties_init)) {
        return;
    }

    SDL_PropertiesID props;
    do {
        props = SDL_GetAtomicU32(&SDL_global_properties);
    } while (!SDL_CompareAndSwapAtomicU32(&SDL_global_properties, props, 0));

    if (props) {
        SDL_DestroyProperties(props);
    }

    // this can't just DestroyHashTable with SDL_FreeProperties as the destructor, because
    //  other destructors under this might cause use to attempt a recursive lock on SDL_properties,
    //  which isn't allowed with rwlocks. So manually iterate and free everything.
    SDL_HashTable *properties = SDL_properties;
    SDL_properties = NULL;
    SDL_IterateHashTable(properties, FreeOneProperties, NULL);
    SDL_DestroyHashTable(properties);

    SDL_SetInitialized(&SDL_properties_init, false);
}

static bool SDL_CheckInitProperties(void)
{
    return SDL_InitProperties();
}

SDL_PropertiesID SDL_GetGlobalProperties(void)
{
    SDL_PropertiesID props = SDL_GetAtomicU32(&SDL_global_properties);
    if (!props) {
        props = SDL_CreateProperties();
        if (!SDL_CompareAndSwapAtomicU32(&SDL_global_properties, 0, props)) {
            // Somebody else created global properties before us, just use those
            SDL_DestroyProperties(props);
            props = SDL_GetAtomicU32(&SDL_global_properties);
        }
    }
    return props;
}

SDL_PropertiesID SDL_CreateProperties(void)
{
    if (!SDL_CheckInitProperties()) {
        return 0;
    }

    SDL_Properties *properties = (SDL_Properties *)SDL_calloc(1, sizeof(*properties));
    if (!properties) {
        return 0;
    }

    properties->lock = SDL_CreateMutex();
    if (!properties->lock) {
        SDL_free(properties);
        return 0;
    }

    properties->props = SDL_CreateHashTable(0, false, SDL_HashString, SDL_KeyMatchString, SDL_FreeProperty, NULL);
    if (!properties->props) {
        SDL_DestroyMutex(properties->lock);
        SDL_free(properties);
        return 0;
    }

    SDL_PropertiesID props = 0;
    while (true) {
        props = (SDL_GetAtomicU32(&SDL_last_properties_id) + 1);
        if (props == 0) {
            continue;
        } else if (SDL_CompareAndSwapAtomicU32(&SDL_last_properties_id, props - 1, props)) {
            break;
        }
    }

    SDL_assert(!SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, NULL));  // should NOT be in the hash table already.

    if (!SDL_InsertIntoHashTable(SDL_properties, (const void *)(uintptr_t)props, properties, false)) {
        SDL_FreeProperties(properties);
        return 0;
    }

    return props;  // All done!
}

typedef struct CopyOnePropertyData
{
    SDL_Properties *dst_properties;
    bool result;
} CopyOnePropertyData;

static bool SDLCALL CopyOneProperty(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    const SDL_Property *src_property = (const SDL_Property *)value;
    if (src_property->cleanup) {
        // Can't copy properties with cleanup functions, we don't know how to duplicate the data
        return true;  // keep iterating.
    }

    CopyOnePropertyData *data = (CopyOnePropertyData *) userdata;
    SDL_Properties *dst_properties = data->dst_properties;
    const char *src_name = (const char *)key;
    SDL_Property *dst_property;

    char *dst_name = SDL_strdup(src_name);
    if (!dst_name) {
        data->result = false;
        return true; // keep iterating (I guess...?)
    }

    dst_property = (SDL_Property *)SDL_malloc(sizeof(*dst_property));
    if (!dst_property) {
        SDL_free(dst_name);
        data->result = false;
        return true; // keep iterating (I guess...?)
    }

    SDL_copyp(dst_property, src_property);
    if (src_property->type == SDL_PROPERTY_TYPE_STRING) {
        dst_property->value.string_value = SDL_strdup(src_property->value.string_value);
        if (!dst_property->value.string_value) {
            SDL_free(dst_name);
            SDL_free(dst_property);
            data->result = false;
            return true; // keep iterating (I guess...?)
        }
    }

    if (!SDL_InsertIntoHashTable(dst_properties->props, dst_name, dst_property, true)) {
        SDL_FreePropertyWithCleanup(dst_name, dst_property, NULL, false);
        data->result = false;
    }

    return true;  // keep iterating.
}

bool SDL_CopyProperties(SDL_PropertiesID src, SDL_PropertiesID dst)
{
    if (!src) {
        return SDL_InvalidParamError("src");
    }
    if (!dst) {
        return SDL_InvalidParamError("dst");
    }

    SDL_Properties *src_properties = NULL;
    SDL_Properties *dst_properties = NULL;

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)src, (const void **)&src_properties);
    if (!src_properties) {
        return SDL_InvalidParamError("src");
    }
    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)dst, (const void **)&dst_properties);
    if (!dst_properties) {
        return SDL_InvalidParamError("dst");
    }

    bool result = true;
    SDL_LockMutex(src_properties->lock);
    SDL_LockMutex(dst_properties->lock);
    {
        CopyOnePropertyData data = { dst_properties, true };
        SDL_IterateHashTable(src_properties->props, CopyOneProperty, &data);
        result = data.result;
    }
    SDL_UnlockMutex(dst_properties->lock);
    SDL_UnlockMutex(src_properties->lock);

    return result;
}

bool SDL_LockProperties(SDL_PropertiesID props)
{
    SDL_Properties *properties = NULL;

    if (!props) {
        return SDL_InvalidParamError("props");
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return SDL_InvalidParamError("props");
    }

    SDL_LockMutex(properties->lock);
    return true;
}

void SDL_UnlockProperties(SDL_PropertiesID props)
{
    SDL_Properties *properties = NULL;

    if (!props) {
        return;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return;
    }

    SDL_UnlockMutex(properties->lock);
}

static bool SDL_PrivateSetProperty(SDL_PropertiesID props, const char *name, SDL_Property *property)
{
    SDL_Properties *properties = NULL;
    bool result = true;

    if (!props) {
        SDL_FreePropertyWithCleanup(NULL, property, NULL, true);
        return SDL_InvalidParamError("props");
    }
    if (!name || !*name) {
        SDL_FreePropertyWithCleanup(NULL, property, NULL, true);
        return SDL_InvalidParamError("name");
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        SDL_FreePropertyWithCleanup(NULL, property, NULL, true);
        return SDL_InvalidParamError("props");
    }

    SDL_LockMutex(properties->lock);
    {
        SDL_RemoveFromHashTable(properties->props, name);
        if (property) {
            char *key = SDL_strdup(name);
            if (!key || !SDL_InsertIntoHashTable(properties->props, key, property, false)) {
                SDL_FreePropertyWithCleanup(key, property, NULL, true);
                result = false;
            }
        }
    }
    SDL_UnlockMutex(properties->lock);

    return result;
}

bool SDL_SetPointerPropertyWithCleanup(SDL_PropertiesID props, const char *name, void *value, SDL_CleanupPropertyCallback cleanup, void *userdata)
{
    SDL_Property *property;

    if (!value) {
        if (cleanup) {
            cleanup(userdata, value);
        }
        return SDL_ClearProperty(props, name);
    }

    property = (SDL_Property *)SDL_calloc(1, sizeof(*property));
    if (!property) {
        if (cleanup) {
            cleanup(userdata, value);
        }
        SDL_FreePropertyWithCleanup(NULL, property, NULL, false);
        return false;
    }
    property->type = SDL_PROPERTY_TYPE_POINTER;
    property->value.pointer_value = value;
    property->cleanup = cleanup;
    property->userdata = userdata;
    return SDL_PrivateSetProperty(props, name, property);
}

bool SDL_SetPointerProperty(SDL_PropertiesID props, const char *name, void *value)
{
    SDL_Property *property;

    if (!value) {
        return SDL_ClearProperty(props, name);
    }

    property = (SDL_Property *)SDL_calloc(1, sizeof(*property));
    if (!property) {
        return false;
    }
    property->type = SDL_PROPERTY_TYPE_POINTER;
    property->value.pointer_value = value;
    return SDL_PrivateSetProperty(props, name, property);
}

static void SDLCALL CleanupFreeableProperty(void *userdata, void *value)
{
    SDL_free(value);
}

bool SDL_SetFreeableProperty(SDL_PropertiesID props, const char *name, void *value)
{
    return SDL_SetPointerPropertyWithCleanup(props, name, value, CleanupFreeableProperty, NULL);
}

static void SDLCALL CleanupSurface(void *userdata, void *value)
{
    SDL_Surface *surface = (SDL_Surface *)value;

    //SDL_DestroySurface(surface);
}

bool SDL_SetSurfaceProperty(SDL_PropertiesID props, const char *name, SDL_Surface *surface)
{
    return SDL_SetPointerPropertyWithCleanup(props, name, surface, CleanupSurface, NULL);
}

bool SDL_SetStringProperty(SDL_PropertiesID props, const char *name, const char *value)
{
    SDL_Property *property;

    if (!value) {
        return SDL_ClearProperty(props, name);
    }

    property = (SDL_Property *)SDL_calloc(1, sizeof(*property));
    if (!property) {
        return false;
    }
    property->type = SDL_PROPERTY_TYPE_STRING;
    property->value.string_value = SDL_strdup(value);
    if (!property->value.string_value) {
        SDL_free(property);
        return false;
    }
    return SDL_PrivateSetProperty(props, name, property);
}

bool SDL_SetNumberProperty(SDL_PropertiesID props, const char *name, Sint64 value)
{
    SDL_Property *property = (SDL_Property *)SDL_calloc(1, sizeof(*property));
    if (!property) {
        return false;
    }
    property->type = SDL_PROPERTY_TYPE_NUMBER;
    property->value.number_value = value;
    return SDL_PrivateSetProperty(props, name, property);
}

bool SDL_SetFloatProperty(SDL_PropertiesID props, const char *name, float value)
{
    SDL_Property *property = (SDL_Property *)SDL_calloc(1, sizeof(*property));
    if (!property) {
        return false;
    }
    property->type = SDL_PROPERTY_TYPE_FLOAT;
    property->value.float_value = value;
    return SDL_PrivateSetProperty(props, name, property);
}

bool SDL_SetBooleanProperty(SDL_PropertiesID props, const char *name, bool value)
{
    SDL_Property *property = (SDL_Property *)SDL_calloc(1, sizeof(*property));
    if (!property) {
        return false;
    }
    property->type = SDL_PROPERTY_TYPE_BOOLEAN;
    property->value.boolean_value = value ? true : false;
    return SDL_PrivateSetProperty(props, name, property);
}

bool SDL_HasProperty(SDL_PropertiesID props, const char *name)
{
    return (SDL_GetPropertyType(props, name) != SDL_PROPERTY_TYPE_INVALID);
}

SDL_PropertyType SDL_GetPropertyType(SDL_PropertiesID props, const char *name)
{
    SDL_Properties *properties = NULL;
    SDL_PropertyType type = SDL_PROPERTY_TYPE_INVALID;

    if (!props) {
        return SDL_PROPERTY_TYPE_INVALID;
    }
    if (!name || !*name) {
        return SDL_PROPERTY_TYPE_INVALID;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return SDL_PROPERTY_TYPE_INVALID;
    }

    SDL_LockMutex(properties->lock);
    {
        SDL_Property *property = NULL;
        if (SDL_FindInHashTable(properties->props, name, (const void **)&property)) {
            type = property->type;
        }
    }
    SDL_UnlockMutex(properties->lock);

    return type;
}

void *SDL_GetPointerProperty(SDL_PropertiesID props, const char *name, void *default_value)
{
    SDL_Properties *properties = NULL;
    void *value = default_value;

    if (!props) {
        return value;
    }
    if (!name || !*name) {
        return value;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return value;
    }

    // Note that taking the lock here only guarantees that we won't read the
    // hashtable while it's being modified. The value itself can easily be
    // freed from another thread after it is returned here.
    SDL_LockMutex(properties->lock);
    {
        SDL_Property *property = NULL;
        if (SDL_FindInHashTable(properties->props, name, (const void **)&property)) {
            if (property->type == SDL_PROPERTY_TYPE_POINTER) {
                value = property->value.pointer_value;
            }
        }
    }
    SDL_UnlockMutex(properties->lock);

    return value;
}

const char *SDL_GetStringProperty(SDL_PropertiesID props, const char *name, const char *default_value)
{
    SDL_Properties *properties = NULL;
    const char *value = default_value;

    if (!props) {
        return value;
    }
    if (!name || !*name) {
        return value;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return value;
    }

    SDL_LockMutex(properties->lock);
    {
        SDL_Property *property = NULL;
        if (SDL_FindInHashTable(properties->props, name, (const void **)&property)) {
            switch (property->type) {
            case SDL_PROPERTY_TYPE_STRING:
                value = property->value.string_value;
                break;
            case SDL_PROPERTY_TYPE_NUMBER:
                if (property->string_storage) {
                    value = property->string_storage;
                } else {
                    SDL_asprintf(&property->string_storage, "%" SDL_PRIs64, property->value.number_value);
                    if (property->string_storage) {
                        value = property->string_storage;
                    }
                }
                break;
            case SDL_PROPERTY_TYPE_FLOAT:
                if (property->string_storage) {
                    value = property->string_storage;
                } else {
                    SDL_asprintf(&property->string_storage, "%f", property->value.float_value);
                    if (property->string_storage) {
                        value = property->string_storage;
                    }
                }
                break;
            case SDL_PROPERTY_TYPE_BOOLEAN:
                value = property->value.boolean_value ? "true" : "false";
                break;
            default:
                break;
            }
        }
    }
    SDL_UnlockMutex(properties->lock);

    return value;
}

Sint64 SDL_GetNumberProperty(SDL_PropertiesID props, const char *name, Sint64 default_value)
{
    SDL_Properties *properties = NULL;
    Sint64 value = default_value;

    if (!props) {
        return value;
    }
    if (!name || !*name) {
        return value;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return value;
    }

    SDL_LockMutex(properties->lock);
    {
        SDL_Property *property = NULL;
        if (SDL_FindInHashTable(properties->props, name, (const void **)&property)) {
            switch (property->type) {
            case SDL_PROPERTY_TYPE_STRING:
                value = (Sint64)SDL_strtoll(property->value.string_value, NULL, 0);
                break;
            case SDL_PROPERTY_TYPE_NUMBER:
                value = property->value.number_value;
                break;
            case SDL_PROPERTY_TYPE_FLOAT:
                value = (Sint64)SDL_round((double)property->value.float_value);
                break;
            case SDL_PROPERTY_TYPE_BOOLEAN:
                value = property->value.boolean_value;
                break;
            default:
                break;
            }
        }
    }
    SDL_UnlockMutex(properties->lock);

    return value;
}

float SDL_GetFloatProperty(SDL_PropertiesID props, const char *name, float default_value)
{
    SDL_Properties *properties = NULL;
    float value = default_value;

    if (!props) {
        return value;
    }
    if (!name || !*name) {
        return value;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return value;
    }

    SDL_LockMutex(properties->lock);
    {
        SDL_Property *property = NULL;
        if (SDL_FindInHashTable(properties->props, name, (const void **)&property)) {
            switch (property->type) {
            case SDL_PROPERTY_TYPE_STRING:
                value = (float)SDL_atof(property->value.string_value);
                break;
            case SDL_PROPERTY_TYPE_NUMBER:
                value = (float)property->value.number_value;
                break;
            case SDL_PROPERTY_TYPE_FLOAT:
                value = property->value.float_value;
                break;
            case SDL_PROPERTY_TYPE_BOOLEAN:
                value = (float)property->value.boolean_value;
                break;
            default:
                break;
            }
        }
    }
    SDL_UnlockMutex(properties->lock);

    return value;
}

bool SDL_GetBooleanProperty(SDL_PropertiesID props, const char *name, bool default_value)
{
    SDL_Properties *properties = NULL;
    bool value = default_value ? true : false;

    if (!props) {
        return value;
    }
    if (!name || !*name) {
        return value;
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return value;
    }

    SDL_LockMutex(properties->lock);
    {
        SDL_Property *property = NULL;
        if (SDL_FindInHashTable(properties->props, name, (const void **)&property)) {
            switch (property->type) {
            case SDL_PROPERTY_TYPE_STRING:
                value = SDL_GetStringBoolean(property->value.string_value, default_value);
                break;
            case SDL_PROPERTY_TYPE_NUMBER:
                value = (property->value.number_value != 0);
                break;
            case SDL_PROPERTY_TYPE_FLOAT:
                value = (property->value.float_value != 0.0f);
                break;
            case SDL_PROPERTY_TYPE_BOOLEAN:
                value = property->value.boolean_value;
                break;
            default:
                break;
            }
        }
    }
    SDL_UnlockMutex(properties->lock);

    return value;
}

bool SDL_ClearProperty(SDL_PropertiesID props, const char *name)
{
    return SDL_PrivateSetProperty(props, name, NULL);
}

typedef struct EnumerateOnePropertyData
{
    SDL_EnumeratePropertiesCallback callback;
    void *userdata;
    SDL_PropertiesID props;
} EnumerateOnePropertyData;


static bool SDLCALL EnumerateOneProperty(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    (void) table;
    (void) value;
    const EnumerateOnePropertyData *data = (const EnumerateOnePropertyData *) userdata;
    data->callback(data->userdata, data->props, (const char *)key);
    return true;  // keep iterating.
}

bool SDL_EnumerateProperties(SDL_PropertiesID props, SDL_EnumeratePropertiesCallback callback, void *userdata)
{
    SDL_Properties *properties = NULL;

    if (!props) {
        return SDL_InvalidParamError("props");
    }
    if (!callback) {
        return SDL_InvalidParamError("callback");
    }

    SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties);
    if (!properties) {
        return SDL_InvalidParamError("props");
    }

    SDL_LockMutex(properties->lock);
    {
        EnumerateOnePropertyData data = { callback, userdata, props };
        SDL_IterateHashTable(properties->props, EnumerateOneProperty, &data);
    }
    SDL_UnlockMutex(properties->lock);

    return true;
}

static void SDLCALL SDL_DumpPropertiesCallback(void *userdata, SDL_PropertiesID props, const char *name)
{
    switch (SDL_GetPropertyType(props, name)) {
    case SDL_PROPERTY_TYPE_POINTER:
        SDL_Log("%s: %p", name, SDL_GetPointerProperty(props, name, NULL));
        break;
    case SDL_PROPERTY_TYPE_STRING:
        SDL_Log("%s: \"%s\"", name, SDL_GetStringProperty(props, name, ""));
        break;
    case SDL_PROPERTY_TYPE_NUMBER:
        {
            Sint64 value = SDL_GetNumberProperty(props, name, 0);
            SDL_Log("%s: %" SDL_PRIs64 " (%" SDL_PRIx64 ")", name, value, value);
        }
        break;
    case SDL_PROPERTY_TYPE_FLOAT:
        SDL_Log("%s: %g", name, SDL_GetFloatProperty(props, name, 0.0f));
        break;
    case SDL_PROPERTY_TYPE_BOOLEAN:
        SDL_Log("%s: %s", name, SDL_GetBooleanProperty(props, name, false) ? "true" : "false");
        break;
    default:
        SDL_Log("%s UNKNOWN TYPE", name);
        break;
    }
}

bool SDL_DumpProperties(SDL_PropertiesID props)
{
    return SDL_EnumerateProperties(props, SDL_DumpPropertiesCallback, NULL);
}

void SDL_DestroyProperties(SDL_PropertiesID props)
{
    if (props) {
        // this can't just use RemoveFromHashTable with SDL_FreeProperties as the destructor, because
        //  other destructors under this might cause use to attempt a recursive lock on SDL_properties,
        //  which isn't allowed with rwlocks. So manually look it up and remove/free it.
        SDL_Properties *properties = NULL;
        if (SDL_FindInHashTable(SDL_properties, (const void *)(uintptr_t)props, (const void **)&properties)) {
            SDL_FreeProperties(properties);
            SDL_RemoveFromHashTable(SDL_properties, (const void *)(uintptr_t)props);
        }
    }
}
