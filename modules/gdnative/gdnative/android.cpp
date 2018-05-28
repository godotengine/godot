#include "gdnative/android.h"

#ifdef __ANDROID__
#include "platform/android/thread_jandroid.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

JNIEnv* GDAPI godot_android_get_env() {
#ifdef __ANDROID__
    return ThreadAndroid::get_env();
#else
    return nullptr;
#endif
}

#ifdef __cplusplus
}
#endif
