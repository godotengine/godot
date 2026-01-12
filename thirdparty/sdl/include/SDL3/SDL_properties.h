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
 * # CategoryProperties
 *
 * A property is a variable that can be created and retrieved by name at
 * runtime.
 *
 * All properties are part of a property group (SDL_PropertiesID). A property
 * group can be created with the SDL_CreateProperties function and destroyed
 * with the SDL_DestroyProperties function.
 *
 * Properties can be added to and retrieved from a property group through the
 * following functions:
 *
 * - SDL_SetPointerProperty and SDL_GetPointerProperty operate on `void*`
 *   pointer types.
 * - SDL_SetStringProperty and SDL_GetStringProperty operate on string types.
 * - SDL_SetNumberProperty and SDL_GetNumberProperty operate on signed 64-bit
 *   integer types.
 * - SDL_SetFloatProperty and SDL_GetFloatProperty operate on floating point
 *   types.
 * - SDL_SetBooleanProperty and SDL_GetBooleanProperty operate on boolean
 *   types.
 *
 * Properties can be removed from a group by using SDL_ClearProperty.
 */


#ifndef SDL_properties_h_
#define SDL_properties_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * SDL properties ID
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint32 SDL_PropertiesID;

/**
 * SDL property type
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_PropertyType
{
    SDL_PROPERTY_TYPE_INVALID,
    SDL_PROPERTY_TYPE_POINTER,
    SDL_PROPERTY_TYPE_STRING,
    SDL_PROPERTY_TYPE_NUMBER,
    SDL_PROPERTY_TYPE_FLOAT,
    SDL_PROPERTY_TYPE_BOOLEAN
} SDL_PropertyType;

/**
 * Get the global SDL properties.
 *
 * \returns a valid property ID on success or 0 on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_PropertiesID SDLCALL SDL_GetGlobalProperties(void);

/**
 * Create a group of properties.
 *
 * All properties are automatically destroyed when SDL_Quit() is called.
 *
 * \returns an ID for a new group of properties, or 0 on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_DestroyProperties
 */
extern SDL_DECLSPEC SDL_PropertiesID SDLCALL SDL_CreateProperties(void);

/**
 * Copy a group of properties.
 *
 * Copy all the properties from one group of properties to another, with the
 * exception of properties requiring cleanup (set using
 * SDL_SetPointerPropertyWithCleanup()), which will not be copied. Any
 * property that already exists on `dst` will be overwritten.
 *
 * \param src the properties to copy.
 * \param dst the destination properties.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CopyProperties(SDL_PropertiesID src, SDL_PropertiesID dst);

/**
 * Lock a group of properties.
 *
 * Obtain a multi-threaded lock for these properties. Other threads will wait
 * while trying to lock these properties until they are unlocked. Properties
 * must be unlocked before they are destroyed.
 *
 * The lock is automatically taken when setting individual properties, this
 * function is only needed when you want to set several properties atomically
 * or want to guarantee that properties being queried aren't freed in another
 * thread.
 *
 * \param props the properties to lock.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_UnlockProperties
 */
extern SDL_DECLSPEC bool SDLCALL SDL_LockProperties(SDL_PropertiesID props);

/**
 * Unlock a group of properties.
 *
 * \param props the properties to unlock.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_LockProperties
 */
extern SDL_DECLSPEC void SDLCALL SDL_UnlockProperties(SDL_PropertiesID props);

/**
 * A callback used to free resources when a property is deleted.
 *
 * This should release any resources associated with `value` that are no
 * longer needed.
 *
 * This callback is set per-property. Different properties in the same group
 * can have different cleanup callbacks.
 *
 * This callback will be called _during_ SDL_SetPointerPropertyWithCleanup if
 * the function fails for any reason.
 *
 * \param userdata an app-defined pointer passed to the callback.
 * \param value the pointer assigned to the property to clean up.
 *
 * \threadsafety This callback may fire without any locks held; if this is a
 *               concern, the app should provide its own locking.
 *
 * \since This datatype is available since SDL 3.2.0.
 *
 * \sa SDL_SetPointerPropertyWithCleanup
 */
typedef void (SDLCALL *SDL_CleanupPropertyCallback)(void *userdata, void *value);

/**
 * Set a pointer property in a group of properties with a cleanup function
 * that is called when the property is deleted.
 *
 * The cleanup function is also called if setting the property fails for any
 * reason.
 *
 * For simply setting basic data types, like numbers, bools, or strings, use
 * SDL_SetNumberProperty, SDL_SetBooleanProperty, or SDL_SetStringProperty
 * instead, as those functions will handle cleanup on your behalf. This
 * function is only for more complex, custom data.
 *
 * \param props the properties to modify.
 * \param name the name of the property to modify.
 * \param value the new value of the property, or NULL to delete the property.
 * \param cleanup the function to call when this property is deleted, or NULL
 *                if no cleanup is necessary.
 * \param userdata a pointer that is passed to the cleanup function.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPointerProperty
 * \sa SDL_SetPointerProperty
 * \sa SDL_CleanupPropertyCallback
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetPointerPropertyWithCleanup(SDL_PropertiesID props, const char *name, void *value, SDL_CleanupPropertyCallback cleanup, void *userdata);

/**
 * Set a pointer property in a group of properties.
 *
 * \param props the properties to modify.
 * \param name the name of the property to modify.
 * \param value the new value of the property, or NULL to delete the property.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPointerProperty
 * \sa SDL_HasProperty
 * \sa SDL_SetBooleanProperty
 * \sa SDL_SetFloatProperty
 * \sa SDL_SetNumberProperty
 * \sa SDL_SetPointerPropertyWithCleanup
 * \sa SDL_SetStringProperty
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetPointerProperty(SDL_PropertiesID props, const char *name, void *value);

/**
 * Set a string property in a group of properties.
 *
 * This function makes a copy of the string; the caller does not have to
 * preserve the data after this call completes.
 *
 * \param props the properties to modify.
 * \param name the name of the property to modify.
 * \param value the new value of the property, or NULL to delete the property.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetStringProperty
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetStringProperty(SDL_PropertiesID props, const char *name, const char *value);

/**
 * Set an integer property in a group of properties.
 *
 * \param props the properties to modify.
 * \param name the name of the property to modify.
 * \param value the new value of the property.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetNumberProperty
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetNumberProperty(SDL_PropertiesID props, const char *name, Sint64 value);

/**
 * Set a floating point property in a group of properties.
 *
 * \param props the properties to modify.
 * \param name the name of the property to modify.
 * \param value the new value of the property.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetFloatProperty
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetFloatProperty(SDL_PropertiesID props, const char *name, float value);

/**
 * Set a boolean property in a group of properties.
 *
 * \param props the properties to modify.
 * \param name the name of the property to modify.
 * \param value the new value of the property.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetBooleanProperty
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetBooleanProperty(SDL_PropertiesID props, const char *name, bool value);

/**
 * Return whether a property exists in a group of properties.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \returns true if the property exists, or false if it doesn't.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPropertyType
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasProperty(SDL_PropertiesID props, const char *name);

/**
 * Get the type of a property in a group of properties.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \returns the type of the property, or SDL_PROPERTY_TYPE_INVALID if it is
 *          not set.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasProperty
 */
extern SDL_DECLSPEC SDL_PropertyType SDLCALL SDL_GetPropertyType(SDL_PropertiesID props, const char *name);

/**
 * Get a pointer property from a group of properties.
 *
 * By convention, the names of properties that SDL exposes on objects will
 * start with "SDL.", and properties that SDL uses internally will start with
 * "SDL.internal.". These should be considered read-only and should not be
 * modified by applications.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \param default_value the default value of the property.
 * \returns the value of the property, or `default_value` if it is not set or
 *          not a pointer property.
 *
 * \threadsafety It is safe to call this function from any thread, although
 *               the data returned is not protected and could potentially be
 *               freed if you call SDL_SetPointerProperty() or
 *               SDL_ClearProperty() on these properties from another thread.
 *               If you need to avoid this, use SDL_LockProperties() and
 *               SDL_UnlockProperties().
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetBooleanProperty
 * \sa SDL_GetFloatProperty
 * \sa SDL_GetNumberProperty
 * \sa SDL_GetPropertyType
 * \sa SDL_GetStringProperty
 * \sa SDL_HasProperty
 * \sa SDL_SetPointerProperty
 */
extern SDL_DECLSPEC void * SDLCALL SDL_GetPointerProperty(SDL_PropertiesID props, const char *name, void *default_value);

/**
 * Get a string property from a group of properties.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \param default_value the default value of the property.
 * \returns the value of the property, or `default_value` if it is not set or
 *          not a string property.
 *
 * \threadsafety It is safe to call this function from any thread, although
 *               the data returned is not protected and could potentially be
 *               freed if you call SDL_SetStringProperty() or
 *               SDL_ClearProperty() on these properties from another thread.
 *               If you need to avoid this, use SDL_LockProperties() and
 *               SDL_UnlockProperties().
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPropertyType
 * \sa SDL_HasProperty
 * \sa SDL_SetStringProperty
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetStringProperty(SDL_PropertiesID props, const char *name, const char *default_value);

/**
 * Get a number property from a group of properties.
 *
 * You can use SDL_GetPropertyType() to query whether the property exists and
 * is a number property.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \param default_value the default value of the property.
 * \returns the value of the property, or `default_value` if it is not set or
 *          not a number property.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPropertyType
 * \sa SDL_HasProperty
 * \sa SDL_SetNumberProperty
 */
extern SDL_DECLSPEC Sint64 SDLCALL SDL_GetNumberProperty(SDL_PropertiesID props, const char *name, Sint64 default_value);

/**
 * Get a floating point property from a group of properties.
 *
 * You can use SDL_GetPropertyType() to query whether the property exists and
 * is a floating point property.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \param default_value the default value of the property.
 * \returns the value of the property, or `default_value` if it is not set or
 *          not a float property.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPropertyType
 * \sa SDL_HasProperty
 * \sa SDL_SetFloatProperty
 */
extern SDL_DECLSPEC float SDLCALL SDL_GetFloatProperty(SDL_PropertiesID props, const char *name, float default_value);

/**
 * Get a boolean property from a group of properties.
 *
 * You can use SDL_GetPropertyType() to query whether the property exists and
 * is a boolean property.
 *
 * \param props the properties to query.
 * \param name the name of the property to query.
 * \param default_value the default value of the property.
 * \returns the value of the property, or `default_value` if it is not set or
 *          not a boolean property.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPropertyType
 * \sa SDL_HasProperty
 * \sa SDL_SetBooleanProperty
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetBooleanProperty(SDL_PropertiesID props, const char *name, bool default_value);

/**
 * Clear a property from a group of properties.
 *
 * \param props the properties to modify.
 * \param name the name of the property to clear.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ClearProperty(SDL_PropertiesID props, const char *name);

/**
 * A callback used to enumerate all the properties in a group of properties.
 *
 * This callback is called from SDL_EnumerateProperties(), and is called once
 * per property in the set.
 *
 * \param userdata an app-defined pointer passed to the callback.
 * \param props the SDL_PropertiesID that is being enumerated.
 * \param name the next property name in the enumeration.
 *
 * \threadsafety SDL_EnumerateProperties holds a lock on `props` during this
 *               callback.
 *
 * \since This datatype is available since SDL 3.2.0.
 *
 * \sa SDL_EnumerateProperties
 */
typedef void (SDLCALL *SDL_EnumeratePropertiesCallback)(void *userdata, SDL_PropertiesID props, const char *name);

/**
 * Enumerate the properties contained in a group of properties.
 *
 * The callback function is called for each property in the group of
 * properties. The properties are locked during enumeration.
 *
 * \param props the properties to query.
 * \param callback the function to call for each property.
 * \param userdata a pointer that is passed to `callback`.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_EnumerateProperties(SDL_PropertiesID props, SDL_EnumeratePropertiesCallback callback, void *userdata);

/**
 * Destroy a group of properties.
 *
 * All properties are deleted and their cleanup functions will be called, if
 * any.
 *
 * \param props the properties to destroy.
 *
 * \threadsafety This function should not be called while these properties are
 *               locked or other threads might be setting or getting values
 *               from these properties.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateProperties
 */
extern SDL_DECLSPEC void SDLCALL SDL_DestroyProperties(SDL_PropertiesID props);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_properties_h_ */
