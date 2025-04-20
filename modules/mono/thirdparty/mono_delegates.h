// Adapted from monovm.h and assembly-functions.h to match coreclr_delegates.h.

// https://github.com/dotnet/runtime/blob/27a7fe5c4bbe0762c231b2a46162e60ee04f3cde/src/mono/mono/mini/monovm.h
// https://github.com/dotnet/runtime/blob/27a7fe5c4bbe0762c231b2a46162e60ee04f3cde/src/native/public/mono/metadata/details/assembly-functions.h

#ifndef _MONO_DELEGATES_H_
#define _MONO_DELEGATES_H_

#include "mono_types.h"

typedef MonoAssembly *(*MonoAssemblyPreLoadFunc)(
		MonoAssemblyName *aname,
		char **assemblies_path,
		void* user_data);

typedef void (*mono_install_assembly_preload_hook_fn)(
		MonoAssemblyPreLoadFunc func,
		void *user_data);

typedef const char *(*mono_assembly_name_get_name_fn)(MonoAssemblyName *aname);

typedef const char *(*mono_assembly_name_get_culture_fn)(MonoAssemblyName *aname);

typedef MonoImage *(*mono_image_open_from_data_with_name_fn)(
		char *data,
		uint32_t data_len,
		mono_bool need_copy,
		/*out*/ MonoImageOpenStatus *status,
		mono_bool refonly,
		const char *name);

typedef MonoAssembly *(*mono_assembly_load_from_full_fn)(
		MonoImage *image,
		const char *fname,
		/*out*/ MonoImageOpenStatus *status,
		mono_bool refonly);

#endif // _MONO_DELEGATES_H_
