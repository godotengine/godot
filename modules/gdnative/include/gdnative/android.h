
#ifndef GODOT_GDNATIVE_ANDRIOD_H
#define GODOT_GDNATIVE_ANDRIOD_H

#include <gdnative/gdnative.h>

#ifdef __ANDROID__
#include <jni.h>
#else
    using JNIEnv = void;
#endif

#ifdef __cplusplus
extern "C" {
#endif

JNIEnv* GDAPI godot_android_get_env();

#ifdef __cplusplus
}
#endif

#endif
