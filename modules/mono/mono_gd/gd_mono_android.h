#ifndef GD_MONO_ANDROID_H
#define GD_MONO_ANDROID_H

#if defined(ANDROID_ENABLED)

#include "core/ustring.h"

namespace GDMonoAndroid {

String get_app_native_lib_dir();

void register_android_dl_fallback();

} // namespace GDMonoAndroid

#endif // ANDROID_ENABLED

#endif // GD_MONO_ANDROID_H
