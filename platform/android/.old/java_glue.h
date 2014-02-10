#ifndef JAVA_GLUE_H
#define JAVA_GLUE_H

#include <jni.h>
#include <android/log.h>


extern "C" {
    JNIEXPORT void JNICALL Java_com_android_godot_GodotLib_init(JNIEnv * env, jobject obj,  jint width, jint height);
    JNIEXPORT void JNICALL Java_com_android_godot_GodotLib_step(JNIEnv * env, jobject obj);
    JNIEXPORT void JNICALL Java_com_android_godot_GodotLib_touch(JNIEnv * env, jobject obj, jint ev,jint pointer, jint count, jintArray positions);
};



#endif // JAVA_GLUE_H
