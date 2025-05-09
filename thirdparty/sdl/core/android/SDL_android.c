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

#ifdef SDL_PLATFORM_ANDROID

#include "SDL_android.h"

#include "../../events/SDL_events_c.h"
#include "../../video/android/SDL_androidkeyboard.h"
#include "../../video/android/SDL_androidmouse.h"
#include "../../video/android/SDL_androidtouch.h"
#include "../../video/android/SDL_androidpen.h"
#include "../../video/android/SDL_androidvideo.h"
#include "../../video/android/SDL_androidwindow.h"
#include "../../joystick/android/SDL_sysjoystick_c.h"
#include "../../haptic/android/SDL_syshaptic_c.h"
#include "../../hidapi/android/hid.h"
#include "../../SDL_hints_c.h"

#include <android/log.h>
#include <android/configuration.h>
#include <android/asset_manager_jni.h>
#include <sys/system_properties.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>

#define SDL_JAVA_PREFIX                               org_libsdl_app
#define CONCAT1(prefix, class, function)              CONCAT2(prefix, class, function)
#define CONCAT2(prefix, class, function)              Java_##prefix##_##class##_##function
#define SDL_JAVA_INTERFACE(function)                  CONCAT1(SDL_JAVA_PREFIX, SDLActivity, function)
#define SDL_JAVA_AUDIO_INTERFACE(function)            CONCAT1(SDL_JAVA_PREFIX, SDLAudioManager, function)
#define SDL_JAVA_CONTROLLER_INTERFACE(function)       CONCAT1(SDL_JAVA_PREFIX, SDLControllerManager, function)
#define SDL_JAVA_INTERFACE_INPUT_CONNECTION(function) CONCAT1(SDL_JAVA_PREFIX, SDLInputConnection, function)

// Audio encoding definitions
#define ENCODING_PCM_8BIT  3
#define ENCODING_PCM_16BIT 2
#define ENCODING_PCM_FLOAT 4

// Java class SDLActivity
JNIEXPORT jstring JNICALL SDL_JAVA_INTERFACE(nativeGetVersion)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetupJNI)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeInitMainThread)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeCleanupMainThread)(
    JNIEnv *env, jclass cls);

JNIEXPORT int JNICALL SDL_JAVA_INTERFACE(nativeRunMain)(
    JNIEnv *env, jclass cls,
    jstring library, jstring function, jobject array);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeDropFile)(
    JNIEnv *env, jclass jcls,
    jstring filename);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetScreenResolution)(
    JNIEnv *env, jclass jcls,
    jint surfaceWidth, jint surfaceHeight,
    jint deviceWidth, jint deviceHeight, jfloat density, jfloat rate);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeResize)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeSurfaceCreated)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeSurfaceChanged)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeSurfaceDestroyed)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeKeyDown)(
    JNIEnv *env, jclass jcls,
    jint keycode);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeKeyUp)(
    JNIEnv *env, jclass jcls,
    jint keycode);

JNIEXPORT jboolean JNICALL SDL_JAVA_INTERFACE(onNativeSoftReturnKey)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeKeyboardFocusLost)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeTouch)(
    JNIEnv *env, jclass jcls,
    jint touch_device_id_in, jint pointer_finger_id_in,
    jint action, jfloat x, jfloat y, jfloat p);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeMouse)(
    JNIEnv *env, jclass jcls,
    jint button, jint action, jfloat x, jfloat y, jboolean relative);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativePen)(
    JNIEnv *env, jclass jcls,
    jint pen_id_in, jint button, jint action, jfloat x, jfloat y, jfloat p);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeAccel)(
    JNIEnv *env, jclass jcls,
    jfloat x, jfloat y, jfloat z);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeClipboardChanged)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeLowMemory)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeLocaleChanged)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeDarkModeChanged)(
    JNIEnv *env, jclass cls, jboolean enabled);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSendQuit)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeQuit)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativePause)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeResume)(
    JNIEnv *env, jclass cls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeFocusChanged)(
    JNIEnv *env, jclass cls, jboolean hasFocus);

JNIEXPORT jstring JNICALL SDL_JAVA_INTERFACE(nativeGetHint)(
    JNIEnv *env, jclass cls,
    jstring name);

JNIEXPORT jboolean JNICALL SDL_JAVA_INTERFACE(nativeGetHintBoolean)(
    JNIEnv *env, jclass cls,
    jstring name, jboolean default_value);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetenv)(
    JNIEnv *env, jclass cls,
    jstring name, jstring value);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetNaturalOrientation)(
    JNIEnv *env, jclass cls,
    jint orientation);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeRotationChanged)(
    JNIEnv *env, jclass cls,
    jint rotation);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeInsetsChanged)(
    JNIEnv *env, jclass cls,
    jint left, jint right, jint top, jint bottom);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeAddTouch)(
    JNIEnv *env, jclass cls,
    jint touchId, jstring name);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativePermissionResult)(
    JNIEnv *env, jclass cls,
    jint requestCode, jboolean result);

JNIEXPORT jboolean JNICALL SDL_JAVA_INTERFACE(nativeAllowRecreateActivity)(
    JNIEnv *env, jclass jcls);

JNIEXPORT int JNICALL SDL_JAVA_INTERFACE(nativeCheckSDLThreadCounter)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeFileDialog)(
    JNIEnv *env, jclass jcls,
    jint requestCode, jobjectArray fileList, jint filter);

static JNINativeMethod SDLActivity_tab[] = {
    { "nativeGetVersion", "()Ljava/lang/String;", SDL_JAVA_INTERFACE(nativeGetVersion) },
    { "nativeSetupJNI", "()I", SDL_JAVA_INTERFACE(nativeSetupJNI) },
    { "nativeInitMainThread", "()V", SDL_JAVA_INTERFACE(nativeInitMainThread) },
    { "nativeCleanupMainThread", "()V", SDL_JAVA_INTERFACE(nativeCleanupMainThread) },
    { "nativeRunMain", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/Object;)I", SDL_JAVA_INTERFACE(nativeRunMain) },
    { "onNativeDropFile", "(Ljava/lang/String;)V", SDL_JAVA_INTERFACE(onNativeDropFile) },
    { "nativeSetScreenResolution", "(IIIIFF)V", SDL_JAVA_INTERFACE(nativeSetScreenResolution) },
    { "onNativeResize", "()V", SDL_JAVA_INTERFACE(onNativeResize) },
    { "onNativeSurfaceCreated", "()V", SDL_JAVA_INTERFACE(onNativeSurfaceCreated) },
    { "onNativeSurfaceChanged", "()V", SDL_JAVA_INTERFACE(onNativeSurfaceChanged) },
    { "onNativeSurfaceDestroyed", "()V", SDL_JAVA_INTERFACE(onNativeSurfaceDestroyed) },
    { "onNativeKeyDown", "(I)V", SDL_JAVA_INTERFACE(onNativeKeyDown) },
    { "onNativeKeyUp", "(I)V", SDL_JAVA_INTERFACE(onNativeKeyUp) },
    { "onNativeSoftReturnKey", "()Z", SDL_JAVA_INTERFACE(onNativeSoftReturnKey) },
    { "onNativeKeyboardFocusLost", "()V", SDL_JAVA_INTERFACE(onNativeKeyboardFocusLost) },
    { "onNativeTouch", "(IIIFFF)V", SDL_JAVA_INTERFACE(onNativeTouch) },
    { "onNativeMouse", "(IIFFZ)V", SDL_JAVA_INTERFACE(onNativeMouse) },
    { "onNativePen", "(IIIFFF)V", SDL_JAVA_INTERFACE(onNativePen) },
    { "onNativeAccel", "(FFF)V", SDL_JAVA_INTERFACE(onNativeAccel) },
    { "onNativeClipboardChanged", "()V", SDL_JAVA_INTERFACE(onNativeClipboardChanged) },
    { "nativeLowMemory", "()V", SDL_JAVA_INTERFACE(nativeLowMemory) },
    { "onNativeLocaleChanged", "()V", SDL_JAVA_INTERFACE(onNativeLocaleChanged) },
    { "onNativeDarkModeChanged", "(Z)V", SDL_JAVA_INTERFACE(onNativeDarkModeChanged) },
    { "nativeSendQuit", "()V", SDL_JAVA_INTERFACE(nativeSendQuit) },
    { "nativeQuit", "()V", SDL_JAVA_INTERFACE(nativeQuit) },
    { "nativePause", "()V", SDL_JAVA_INTERFACE(nativePause) },
    { "nativeResume", "()V", SDL_JAVA_INTERFACE(nativeResume) },
    { "nativeFocusChanged", "(Z)V", SDL_JAVA_INTERFACE(nativeFocusChanged) },
    { "nativeGetHint", "(Ljava/lang/String;)Ljava/lang/String;", SDL_JAVA_INTERFACE(nativeGetHint) },
    { "nativeGetHintBoolean", "(Ljava/lang/String;Z)Z", SDL_JAVA_INTERFACE(nativeGetHintBoolean) },
    { "nativeSetenv", "(Ljava/lang/String;Ljava/lang/String;)V", SDL_JAVA_INTERFACE(nativeSetenv) },
    { "nativeSetNaturalOrientation", "(I)V", SDL_JAVA_INTERFACE(nativeSetNaturalOrientation) },
    { "onNativeRotationChanged", "(I)V", SDL_JAVA_INTERFACE(onNativeRotationChanged) },
    { "onNativeInsetsChanged", "(IIII)V", SDL_JAVA_INTERFACE(onNativeInsetsChanged) },
    { "nativeAddTouch", "(ILjava/lang/String;)V", SDL_JAVA_INTERFACE(nativeAddTouch) },
    { "nativePermissionResult", "(IZ)V", SDL_JAVA_INTERFACE(nativePermissionResult) },
    { "nativeAllowRecreateActivity", "()Z", SDL_JAVA_INTERFACE(nativeAllowRecreateActivity) },
    { "nativeCheckSDLThreadCounter", "()I", SDL_JAVA_INTERFACE(nativeCheckSDLThreadCounter) },
    { "onNativeFileDialog", "(I[Ljava/lang/String;I)V", SDL_JAVA_INTERFACE(onNativeFileDialog) }
};

// Java class SDLInputConnection
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE_INPUT_CONNECTION(nativeCommitText)(
    JNIEnv *env, jclass cls,
    jstring text, jint newCursorPosition);

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE_INPUT_CONNECTION(nativeGenerateScancodeForUnichar)(
    JNIEnv *env, jclass cls,
    jchar chUnicode);

static JNINativeMethod SDLInputConnection_tab[] = {
    { "nativeCommitText", "(Ljava/lang/String;I)V", SDL_JAVA_INTERFACE_INPUT_CONNECTION(nativeCommitText) },
    { "nativeGenerateScancodeForUnichar", "(C)V", SDL_JAVA_INTERFACE_INPUT_CONNECTION(nativeGenerateScancodeForUnichar) }
};

// Java class SDLAudioManager
JNIEXPORT void JNICALL SDL_JAVA_AUDIO_INTERFACE(nativeSetupJNI)(
    JNIEnv *env, jclass jcls);

JNIEXPORT void JNICALL
    SDL_JAVA_AUDIO_INTERFACE(nativeAddAudioDevice)(JNIEnv *env, jclass jcls, jboolean recording, jstring name,
                                             jint device_id);

JNIEXPORT void JNICALL
    SDL_JAVA_AUDIO_INTERFACE(nativeRemoveAudioDevice)(JNIEnv *env, jclass jcls, jboolean recording,
                                                jint device_id);

static JNINativeMethod SDLAudioManager_tab[] = {
    { "nativeSetupJNI", "()I", SDL_JAVA_AUDIO_INTERFACE(nativeSetupJNI) },
    { "nativeAddAudioDevice", "(ZLjava/lang/String;I)V", SDL_JAVA_AUDIO_INTERFACE(nativeAddAudioDevice) },
    { "nativeRemoveAudioDevice", "(ZI)V", SDL_JAVA_AUDIO_INTERFACE(nativeRemoveAudioDevice) }
};

// Java class SDLControllerManager
JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeSetupJNI)(
    JNIEnv *env, jclass jcls);

JNIEXPORT jboolean JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativePadDown)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint keycode);

JNIEXPORT jboolean JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativePadUp)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint keycode);

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativeJoy)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint axis, jfloat value);

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativeHat)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint hat_id, jint x, jint y);

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeAddJoystick)(
    JNIEnv *env, jclass jcls,
    jint device_id, jstring device_name, jstring device_desc, jint vendor_id, jint product_id,
    jint button_mask, jint naxes, jint axis_mask, jint nhats, jboolean can_rumble);

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeRemoveJoystick)(
    JNIEnv *env, jclass jcls,
    jint device_id);

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeAddHaptic)(
    JNIEnv *env, jclass jcls,
    jint device_id, jstring device_name);

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeRemoveHaptic)(
    JNIEnv *env, jclass jcls,
    jint device_id);

static JNINativeMethod SDLControllerManager_tab[] = {
    { "nativeSetupJNI", "()I", SDL_JAVA_CONTROLLER_INTERFACE(nativeSetupJNI) },
    { "onNativePadDown", "(II)Z", SDL_JAVA_CONTROLLER_INTERFACE(onNativePadDown) },
    { "onNativePadUp", "(II)Z", SDL_JAVA_CONTROLLER_INTERFACE(onNativePadUp) },
    { "onNativeJoy", "(IIF)V", SDL_JAVA_CONTROLLER_INTERFACE(onNativeJoy) },
    { "onNativeHat", "(IIII)V", SDL_JAVA_CONTROLLER_INTERFACE(onNativeHat) },
    { "nativeAddJoystick", "(ILjava/lang/String;Ljava/lang/String;IIIIIIZ)V", SDL_JAVA_CONTROLLER_INTERFACE(nativeAddJoystick) },
    { "nativeRemoveJoystick", "(I)V", SDL_JAVA_CONTROLLER_INTERFACE(nativeRemoveJoystick) },
    { "nativeAddHaptic", "(ILjava/lang/String;)V", SDL_JAVA_CONTROLLER_INTERFACE(nativeAddHaptic) },
    { "nativeRemoveHaptic", "(I)V", SDL_JAVA_CONTROLLER_INTERFACE(nativeRemoveHaptic) }
};

// Uncomment this to log messages entering and exiting methods in this file
// #define DEBUG_JNI

static void checkJNIReady(void);

/*******************************************************************************
 This file links the Java side of Android with libsdl
*******************************************************************************/
#include <jni.h>

/*******************************************************************************
                               Globals
*******************************************************************************/
static pthread_key_t mThreadKey;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;
static JavaVM *mJavaVM = NULL;

// Main activity
static jclass mActivityClass;

// method signatures
static jmethodID midClipboardGetText;
static jmethodID midClipboardHasText;
static jmethodID midClipboardSetText;
static jmethodID midCreateCustomCursor;
static jmethodID midDestroyCustomCursor;
static jmethodID midGetContext;
static jmethodID midGetManifestEnvironmentVariables;
static jmethodID midGetNativeSurface;
static jmethodID midInitTouch;
static jmethodID midIsAndroidTV;
static jmethodID midIsChromebook;
static jmethodID midIsDeXMode;
static jmethodID midIsScreenKeyboardShown;
static jmethodID midIsTablet;
static jmethodID midManualBackButton;
static jmethodID midMinimizeWindow;
static jmethodID midOpenURL;
static jmethodID midRequestPermission;
static jmethodID midShowToast;
static jmethodID midSendMessage;
static jmethodID midSetActivityTitle;
static jmethodID midSetCustomCursor;
static jmethodID midSetOrientation;
static jmethodID midSetRelativeMouseEnabled;
static jmethodID midSetSystemCursor;
static jmethodID midSetWindowStyle;
static jmethodID midShouldMinimizeOnFocusLoss;
static jmethodID midShowTextInput;
static jmethodID midSupportsRelativeMouse;
static jmethodID midOpenFileDescriptor;
static jmethodID midShowFileDialog;

// audio manager
static jclass mAudioManagerClass;

// method signatures
static jmethodID midRegisterAudioDeviceCallback;
static jmethodID midUnregisterAudioDeviceCallback;
static jmethodID midAudioSetThreadPriority;

// controller manager
static jclass mControllerManagerClass;

// method signatures
static jmethodID midPollInputDevices;
static jmethodID midPollHapticDevices;
static jmethodID midHapticRun;
static jmethodID midHapticRumble;
static jmethodID midHapticStop;

// Accelerometer data storage
static SDL_DisplayOrientation displayNaturalOrientation;
static SDL_DisplayOrientation displayCurrentOrientation;
static float fLastAccelerometer[3];
static bool bHasNewData;

static bool bHasEnvironmentVariables;

// Android AssetManager
static void Internal_Android_Create_AssetManager(void);
static void Internal_Android_Destroy_AssetManager(void);
static AAssetManager *asset_manager = NULL;
static jobject javaAssetManagerRef = 0;

static SDL_Mutex *Android_ActivityMutex = NULL;
static SDL_Mutex *Android_LifecycleMutex = NULL;
static SDL_Semaphore *Android_LifecycleEventSem = NULL;
static SDL_AndroidLifecycleEvent Android_LifecycleEvents[SDL_NUM_ANDROID_LIFECYCLE_EVENTS];
static int Android_NumLifecycleEvents;

/*******************************************************************************
                 Functions called by JNI
*******************************************************************************/

/* From http://developer.android.com/guide/practices/jni.html
 * All threads are Linux threads, scheduled by the kernel.
 * They're usually started from managed code (using Thread.start), but they can also be created elsewhere and then
 * attached to the JavaVM. For example, a thread started with pthread_create can be attached with the
 * JNI AttachCurrentThread or AttachCurrentThreadAsDaemon functions. Until a thread is attached, it has no JNIEnv,
 * and cannot make JNI calls.
 * Attaching a natively-created thread causes a java.lang.Thread object to be constructed and added to the "main"
 * ThreadGroup, making it visible to the debugger. Calling AttachCurrentThread on an already-attached thread
 * is a no-op.
 * Note: You can call this function any number of times for the same thread, there's no harm in it
 */

/* From http://developer.android.com/guide/practices/jni.html
 * Threads attached through JNI must call DetachCurrentThread before they exit. If coding this directly is awkward,
 * in Android 2.0 (Eclair) and higher you can use pthread_key_create to define a destructor function that will be
 * called before the thread exits, and call DetachCurrentThread from there. (Use that key with pthread_setspecific
 * to store the JNIEnv in thread-local-storage; that way it'll be passed into your destructor as the argument.)
 * Note: The destructor is not called unless the stored value is != NULL
 * Note: You can call this function any number of times for the same thread, there's no harm in it
 *       (except for some lost CPU cycles)
 */

// Set local storage value
static bool Android_JNI_SetEnv(JNIEnv *env)
{
    int status = pthread_setspecific(mThreadKey, env);
    if (status < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed pthread_setspecific() in Android_JNI_SetEnv() (err=%d)", status);
        return false;
    }
    return true;
}

// Get local storage value
JNIEnv *Android_JNI_GetEnv(void)
{
    // Get JNIEnv from the Thread local storage
    JNIEnv *env = pthread_getspecific(mThreadKey);
    if (!env) {
        // If it fails, try to attach ! (e.g the thread isn't created with SDL_CreateThread()
        int status;

        // There should be a JVM
        if (!mJavaVM) {
            __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed, there is no JavaVM");
            return NULL;
        }

        /* Attach the current thread to the JVM and get a JNIEnv.
         * It will be detached by pthread_create destructor 'Android_JNI_ThreadDestroyed' */
        status = (*mJavaVM)->AttachCurrentThread(mJavaVM, &env, NULL);
        if (status < 0) {
            __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed to attach current thread (err=%d)", status);
            return NULL;
        }

        // Save JNIEnv into the Thread local storage
        if (!Android_JNI_SetEnv(env)) {
            return NULL;
        }
    }

    return env;
}

// Set up an external thread for using JNI with Android_JNI_GetEnv()
bool Android_JNI_SetupThread(void)
{
    JNIEnv *env;
    int status;

    // There should be a JVM
    if (!mJavaVM) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed, there is no JavaVM");
        return false;
    }

    /* Attach the current thread to the JVM and get a JNIEnv.
     * It will be detached by pthread_create destructor 'Android_JNI_ThreadDestroyed' */
    status = (*mJavaVM)->AttachCurrentThread(mJavaVM, &env, NULL);
    if (status < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed to attach current thread (err=%d)", status);
        return false;
    }

    // Save JNIEnv into the Thread local storage
    if (!Android_JNI_SetEnv(env)) {
        return false;
    }

    return true;
}

// Destructor called for each thread where mThreadKey is not NULL
static void Android_JNI_ThreadDestroyed(void *value)
{
    // The thread is being destroyed, detach it from the Java VM and set the mThreadKey value to NULL as required
    JNIEnv *env = (JNIEnv *)value;
    if (env) {
        (*mJavaVM)->DetachCurrentThread(mJavaVM);
        Android_JNI_SetEnv(NULL);
    }
}

// Creation of local storage mThreadKey
static void Android_JNI_CreateKey(void)
{
    int status = pthread_key_create(&mThreadKey, Android_JNI_ThreadDestroyed);
    if (status < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Error initializing mThreadKey with pthread_key_create() (err=%d)", status);
    }
}

static void Android_JNI_CreateKey_once(void)
{
    int status = pthread_once(&key_once, Android_JNI_CreateKey);
    if (status < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Error initializing mThreadKey with pthread_once() (err=%d)", status);
    }
}

static void register_methods(JNIEnv *env, const char *classname, JNINativeMethod *methods, int nb)
{
    jclass clazz = (*env)->FindClass(env, classname);
    if (!clazz || (*env)->RegisterNatives(env, clazz, methods, nb) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed to register methods of %s", classname);
        return;
    }
}

// Library init
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
    JNIEnv *env = NULL;

    mJavaVM = vm;

    if ((*mJavaVM)->GetEnv(mJavaVM, (void **)&env, JNI_VERSION_1_4) != JNI_OK) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "Failed to get JNI Env");
        return JNI_VERSION_1_4;
    }

    register_methods(env, "org/libsdl/app/SDLActivity", SDLActivity_tab, SDL_arraysize(SDLActivity_tab));
    register_methods(env, "org/libsdl/app/SDLInputConnection", SDLInputConnection_tab, SDL_arraysize(SDLInputConnection_tab));
    register_methods(env, "org/libsdl/app/SDLAudioManager", SDLAudioManager_tab, SDL_arraysize(SDLAudioManager_tab));
    register_methods(env, "org/libsdl/app/SDLControllerManager", SDLControllerManager_tab, SDL_arraysize(SDLControllerManager_tab));
    register_methods(env, "org/libsdl/app/HIDDeviceManager", HIDDeviceManager_tab, SDL_arraysize(HIDDeviceManager_tab));

    return JNI_VERSION_1_4;
}

void checkJNIReady(void)
{
    if (!mActivityClass || !mAudioManagerClass || !mControllerManagerClass) {
        // We aren't fully initialized, let's just return.
        return;
    }

    SDL_SetMainReady();
}

// Get SDL version -- called before SDL_main() to verify JNI bindings
JNIEXPORT jstring JNICALL SDL_JAVA_INTERFACE(nativeGetVersion)(JNIEnv *env, jclass cls)
{
    char version[128];

    SDL_snprintf(version, sizeof(version), "%d.%d.%d", SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_MICRO_VERSION);

    return (*env)->NewStringUTF(env, version);
}

// Activity initialization -- called before SDL_main() to initialize JNI bindings
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetupJNI)(JNIEnv *env, jclass cls)
{
    __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "nativeSetupJNI()");

    // Start with a clean slate
    SDL_ClearError();

    /*
     * Create mThreadKey so we can keep track of the JNIEnv assigned to each thread
     * Refer to http://developer.android.com/guide/practices/design/jni.html for the rationale behind this
     */
    Android_JNI_CreateKey_once();

    // Save JNIEnv of SDLActivity
    Android_JNI_SetEnv(env);

    if (!mJavaVM) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "failed to found a JavaVM");
    }

    /* Use a mutex to prevent concurrency issues between Java Activity and Native thread code, when using 'Android_Window'.
     * (Eg. Java sending Touch events, while native code is destroying the main SDL_Window. )
     */
    if (!Android_ActivityMutex) {
        Android_ActivityMutex = SDL_CreateMutex(); // Could this be created twice if onCreate() is called a second time ?
    }

    if (!Android_ActivityMutex) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "failed to create Android_ActivityMutex mutex");
    }

    Android_LifecycleMutex = SDL_CreateMutex();
    if (!Android_LifecycleMutex) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "failed to create Android_LifecycleMutex mutex");
    }

    Android_LifecycleEventSem = SDL_CreateSemaphore(0);
    if (!Android_LifecycleEventSem) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "failed to create Android_LifecycleEventSem semaphore");
    }

    mActivityClass = (jclass)((*env)->NewGlobalRef(env, cls));

    midClipboardGetText = (*env)->GetStaticMethodID(env, mActivityClass, "clipboardGetText", "()Ljava/lang/String;");
    midClipboardHasText = (*env)->GetStaticMethodID(env, mActivityClass, "clipboardHasText", "()Z");
    midClipboardSetText = (*env)->GetStaticMethodID(env, mActivityClass, "clipboardSetText", "(Ljava/lang/String;)V");
    midCreateCustomCursor = (*env)->GetStaticMethodID(env, mActivityClass, "createCustomCursor", "([IIIII)I");
    midDestroyCustomCursor = (*env)->GetStaticMethodID(env, mActivityClass, "destroyCustomCursor", "(I)V");
    midGetContext = (*env)->GetStaticMethodID(env, mActivityClass, "getContext", "()Landroid/content/Context;");
    midGetManifestEnvironmentVariables = (*env)->GetStaticMethodID(env, mActivityClass, "getManifestEnvironmentVariables", "()Z");
    midGetNativeSurface = (*env)->GetStaticMethodID(env, mActivityClass, "getNativeSurface", "()Landroid/view/Surface;");
    midInitTouch = (*env)->GetStaticMethodID(env, mActivityClass, "initTouch", "()V");
    midIsAndroidTV = (*env)->GetStaticMethodID(env, mActivityClass, "isAndroidTV", "()Z");
    midIsChromebook = (*env)->GetStaticMethodID(env, mActivityClass, "isChromebook", "()Z");
    midIsDeXMode = (*env)->GetStaticMethodID(env, mActivityClass, "isDeXMode", "()Z");
    midIsScreenKeyboardShown = (*env)->GetStaticMethodID(env, mActivityClass, "isScreenKeyboardShown", "()Z");
    midIsTablet = (*env)->GetStaticMethodID(env, mActivityClass, "isTablet", "()Z");
    midManualBackButton = (*env)->GetStaticMethodID(env, mActivityClass, "manualBackButton", "()V");
    midMinimizeWindow = (*env)->GetStaticMethodID(env, mActivityClass, "minimizeWindow", "()V");
    midOpenURL = (*env)->GetStaticMethodID(env, mActivityClass, "openURL", "(Ljava/lang/String;)Z");
    midRequestPermission = (*env)->GetStaticMethodID(env, mActivityClass, "requestPermission", "(Ljava/lang/String;I)V");
    midShowToast = (*env)->GetStaticMethodID(env, mActivityClass, "showToast", "(Ljava/lang/String;IIII)Z");
    midSendMessage = (*env)->GetStaticMethodID(env, mActivityClass, "sendMessage", "(II)Z");
    midSetActivityTitle = (*env)->GetStaticMethodID(env, mActivityClass, "setActivityTitle", "(Ljava/lang/String;)Z");
    midSetCustomCursor = (*env)->GetStaticMethodID(env, mActivityClass, "setCustomCursor", "(I)Z");
    midSetOrientation = (*env)->GetStaticMethodID(env, mActivityClass, "setOrientation", "(IIZLjava/lang/String;)V");
    midSetRelativeMouseEnabled = (*env)->GetStaticMethodID(env, mActivityClass, "setRelativeMouseEnabled", "(Z)Z");
    midSetSystemCursor = (*env)->GetStaticMethodID(env, mActivityClass, "setSystemCursor", "(I)Z");
    midSetWindowStyle = (*env)->GetStaticMethodID(env, mActivityClass, "setWindowStyle", "(Z)V");
    midShouldMinimizeOnFocusLoss = (*env)->GetStaticMethodID(env, mActivityClass, "shouldMinimizeOnFocusLoss", "()Z");
    midShowTextInput = (*env)->GetStaticMethodID(env, mActivityClass, "showTextInput", "(IIIII)Z");
    midSupportsRelativeMouse = (*env)->GetStaticMethodID(env, mActivityClass, "supportsRelativeMouse", "()Z");
    midOpenFileDescriptor = (*env)->GetStaticMethodID(env, mActivityClass, "openFileDescriptor", "(Ljava/lang/String;Ljava/lang/String;)I");
    midShowFileDialog = (*env)->GetStaticMethodID(env, mActivityClass, "showFileDialog", "([Ljava/lang/String;ZZI)Z");

    if (!midClipboardGetText ||
        !midClipboardHasText ||
        !midClipboardSetText ||
        !midCreateCustomCursor ||
        !midDestroyCustomCursor ||
        !midGetContext ||
        !midGetManifestEnvironmentVariables ||
        !midGetNativeSurface ||
        !midInitTouch ||
        !midIsAndroidTV ||
        !midIsChromebook ||
        !midIsDeXMode ||
        !midIsScreenKeyboardShown ||
        !midIsTablet ||
        !midManualBackButton ||
        !midMinimizeWindow ||
        !midOpenURL ||
        !midRequestPermission ||
        !midShowToast ||
        !midSendMessage ||
        !midSetActivityTitle ||
        !midSetCustomCursor ||
        !midSetOrientation ||
        !midSetRelativeMouseEnabled ||
        !midSetSystemCursor ||
        !midSetWindowStyle ||
        !midShouldMinimizeOnFocusLoss ||
        !midShowTextInput ||
        !midSupportsRelativeMouse ||
        !midOpenFileDescriptor ||
        !midShowFileDialog) {
        __android_log_print(ANDROID_LOG_WARN, "SDL", "Missing some Java callbacks, do you have the latest version of SDLActivity.java?");
    }

    checkJNIReady();
}

// Audio initialization -- called before SDL_main() to initialize JNI bindings
JNIEXPORT void JNICALL SDL_JAVA_AUDIO_INTERFACE(nativeSetupJNI)(JNIEnv *env, jclass cls)
{
    __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "AUDIO nativeSetupJNI()");

    mAudioManagerClass = (jclass)((*env)->NewGlobalRef(env, cls));

    midRegisterAudioDeviceCallback = (*env)->GetStaticMethodID(env, mAudioManagerClass,
                                                         "registerAudioDeviceCallback",
                                                         "()V");
    midUnregisterAudioDeviceCallback = (*env)->GetStaticMethodID(env, mAudioManagerClass,
                                                         "unregisterAudioDeviceCallback",
                                                         "()V");
    midAudioSetThreadPriority = (*env)->GetStaticMethodID(env, mAudioManagerClass,
                                                          "audioSetThreadPriority", "(ZI)V");

    if (!midRegisterAudioDeviceCallback || !midUnregisterAudioDeviceCallback || !midAudioSetThreadPriority) {
        __android_log_print(ANDROID_LOG_WARN, "SDL",
                            "Missing some Java callbacks, do you have the latest version of SDLAudioManager.java?");
    }

    checkJNIReady();
}

// Controller initialization -- called before SDL_main() to initialize JNI bindings
JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeSetupJNI)(JNIEnv *env, jclass cls)
{
    __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "CONTROLLER nativeSetupJNI()");

    mControllerManagerClass = (jclass)((*env)->NewGlobalRef(env, cls));

    midPollInputDevices = (*env)->GetStaticMethodID(env, mControllerManagerClass,
                                                    "pollInputDevices", "()V");
    midPollHapticDevices = (*env)->GetStaticMethodID(env, mControllerManagerClass,
                                                     "pollHapticDevices", "()V");
    midHapticRun = (*env)->GetStaticMethodID(env, mControllerManagerClass,
                                             "hapticRun", "(IFI)V");
    midHapticRumble = (*env)->GetStaticMethodID(env, mControllerManagerClass,
                                             "hapticRumble", "(IFFI)V");
    midHapticStop = (*env)->GetStaticMethodID(env, mControllerManagerClass,
                                              "hapticStop", "(I)V");

    if (!midPollInputDevices || !midPollHapticDevices || !midHapticRun || !midHapticRumble || !midHapticStop) {
        __android_log_print(ANDROID_LOG_WARN, "SDL", "Missing some Java callbacks, do you have the latest version of SDLControllerManager.java?");
    }

    checkJNIReady();
}

// SDL main function prototype
typedef int (*SDL_main_func)(int argc, char *argv[]);

static int run_count = 0;
static bool allow_recreate_activity;
static bool allow_recreate_activity_set;

JNIEXPORT int JNICALL SDL_JAVA_INTERFACE(nativeCheckSDLThreadCounter)(
    JNIEnv *env, jclass jcls)
{
    int tmp = run_count;
    run_count += 1;
    return tmp;
}

void Android_SetAllowRecreateActivity(bool enabled)
{
    allow_recreate_activity = enabled;
    allow_recreate_activity_set = true;
}

JNIEXPORT jboolean JNICALL SDL_JAVA_INTERFACE(nativeAllowRecreateActivity)(
    JNIEnv *env, jclass jcls)
{
    return allow_recreate_activity;
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeInitMainThread)(
    JNIEnv *env, jclass jcls)
{
    __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "nativeInitSDLThread() %d time", run_count);
    run_count += 1;

    // Save JNIEnv of SDLThread
    Android_JNI_SetEnv(env);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeCleanupMainThread)(
    JNIEnv *env, jclass jcls)
{
    /* This is a Java thread, it doesn't need to be Detached from the JVM.
     * Set to mThreadKey value to NULL not to call pthread_create destructor 'Android_JNI_ThreadDestroyed' */
    Android_JNI_SetEnv(NULL);
}

// Start up the SDL app
JNIEXPORT int JNICALL SDL_JAVA_INTERFACE(nativeRunMain)(JNIEnv *env, jclass cls, jstring library, jstring function, jobject array)
{
    int status = -1;
    const char *library_file;
    void *library_handle;

    library_file = (*env)->GetStringUTFChars(env, library, NULL);
    library_handle = dlopen(library_file, RTLD_GLOBAL);

    if (library_handle == NULL) {
        /* When deploying android app bundle format uncompressed native libs may not extract from apk to filesystem.
           In this case we should use lib name without path. https://bugzilla.libsdl.org/show_bug.cgi?id=4739 */
        const char *library_name = SDL_strrchr(library_file, '/');
        if (library_name && *library_name) {
            library_name += 1;
            library_handle = dlopen(library_name, RTLD_GLOBAL);
        }
    }

    if (library_handle) {
        const char *function_name;
        SDL_main_func SDL_main;

        function_name = (*env)->GetStringUTFChars(env, function, NULL);
        SDL_main = (SDL_main_func)dlsym(library_handle, function_name);
        if (SDL_main) {
            int i;
            int argc;
            int len;
            char **argv;
            bool isstack;

            // Prepare the arguments.
            len = (*env)->GetArrayLength(env, array);
            argv = SDL_small_alloc(char *, 1 + len + 1, &isstack); // !!! FIXME: check for NULL
            argc = 0;
            /* Use the name "app_process" so PHYSFS_platformCalcBaseDir() works.
               https://github.com/love2d/love-android/issues/24
             */
            argv[argc++] = SDL_strdup("app_process");
            for (i = 0; i < len; ++i) {
                char *arg = NULL;
                jstring string = (*env)->GetObjectArrayElement(env, array, i);
                if (string) {
                    const char *utf = (*env)->GetStringUTFChars(env, string, 0);
                    if (utf) {
                        arg = SDL_strdup(utf);
                        (*env)->ReleaseStringUTFChars(env, string, utf);
                    }
                    (*env)->DeleteLocalRef(env, string);
                }
                if (arg == NULL) {
                    arg = SDL_strdup("");
                }
                argv[argc++] = arg;
            }
            argv[argc] = NULL;

            // Run the application.
            status = SDL_main(argc, argv);

            // Release the arguments.
            for (i = 0; i < argc; ++i) {
                SDL_free(argv[i]);
            }
            SDL_small_free(argv, isstack);

        } else {
            __android_log_print(ANDROID_LOG_ERROR, "SDL", "nativeRunMain(): Couldn't find function %s in library %s", function_name, library_file);
        }
        (*env)->ReleaseStringUTFChars(env, function, function_name);

        dlclose(library_handle);

    } else {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "nativeRunMain(): Couldn't load library %s", library_file);
    }
    (*env)->ReleaseStringUTFChars(env, library, library_file);

    // Do not issue an exit or the whole application will terminate instead of just the SDL thread
    // exit(status);

    return status;
}

static int FindLifecycleEvent(SDL_AndroidLifecycleEvent event)
{
    for (int index = 0; index < Android_NumLifecycleEvents; ++index) {
        if (Android_LifecycleEvents[index] == event) {
            return index;
        }
    }
    return -1;
}

static void RemoveLifecycleEvent(int index)
{
    if (index < Android_NumLifecycleEvents - 1) {
        SDL_memmove(&Android_LifecycleEvents[index], &Android_LifecycleEvents[index+1], (Android_NumLifecycleEvents - index - 1) * sizeof(Android_LifecycleEvents[index]));
    }
    --Android_NumLifecycleEvents;
}

void Android_SendLifecycleEvent(SDL_AndroidLifecycleEvent event)
{
    SDL_LockMutex(Android_LifecycleMutex);
    {
        int index;
        bool add_event = true;

        switch (event) {
        case SDL_ANDROID_LIFECYCLE_WAKE:
            // We don't need more than one wake queued
            index = FindLifecycleEvent(SDL_ANDROID_LIFECYCLE_WAKE);
            if (index >= 0) {
                add_event = false;
            }
            break;
        case SDL_ANDROID_LIFECYCLE_PAUSE:
            // If we have a resume queued, just stay in the paused state
            index = FindLifecycleEvent(SDL_ANDROID_LIFECYCLE_RESUME);
            if (index >= 0) {
                RemoveLifecycleEvent(index);
                add_event = false;
            }
            break;
        case SDL_ANDROID_LIFECYCLE_RESUME:
            // If we have a pause queued, just stay in the resumed state
            index = FindLifecycleEvent(SDL_ANDROID_LIFECYCLE_PAUSE);
            if (index >= 0) {
                RemoveLifecycleEvent(index);
                add_event = false;
            }
            break;
        case SDL_ANDROID_LIFECYCLE_LOWMEMORY:
            // We don't need more than one low memory event queued
            index = FindLifecycleEvent(SDL_ANDROID_LIFECYCLE_LOWMEMORY);
            if (index >= 0) {
                add_event = false;
            }
            break;
        case SDL_ANDROID_LIFECYCLE_DESTROY:
            // Remove all other events, we're done!
            while (Android_NumLifecycleEvents > 0) {
                RemoveLifecycleEvent(0);
            }
            break;
        default:
            SDL_assert(!"Sending unexpected lifecycle event");
            add_event = false;
            break;
        }

        if (add_event) {
            SDL_assert(Android_NumLifecycleEvents < SDL_arraysize(Android_LifecycleEvents));
            Android_LifecycleEvents[Android_NumLifecycleEvents++] = event;
            SDL_SignalSemaphore(Android_LifecycleEventSem);
        }
    }
    SDL_UnlockMutex(Android_LifecycleMutex);
}

bool Android_WaitLifecycleEvent(SDL_AndroidLifecycleEvent *event, Sint64 timeoutNS)
{
    bool got_event = false;

    while (!got_event && SDL_WaitSemaphoreTimeoutNS(Android_LifecycleEventSem, timeoutNS)) {
        SDL_LockMutex(Android_LifecycleMutex);
        {
            if (Android_NumLifecycleEvents > 0) {
                *event = Android_LifecycleEvents[0];
                RemoveLifecycleEvent(0);
                got_event = true;
            }
        }
        SDL_UnlockMutex(Android_LifecycleMutex);
    }
    return got_event;
}

void Android_LockActivityMutex(void)
{
    SDL_LockMutex(Android_ActivityMutex);
}

void Android_UnlockActivityMutex(void)
{
    SDL_UnlockMutex(Android_ActivityMutex);
}

// Drop file
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeDropFile)(
    JNIEnv *env, jclass jcls,
    jstring filename)
{
    const char *path = (*env)->GetStringUTFChars(env, filename, NULL);
    SDL_SendDropFile(NULL, NULL, path);
    (*env)->ReleaseStringUTFChars(env, filename, path);
    SDL_SendDropComplete(NULL);
}

// Set screen resolution
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetScreenResolution)(
    JNIEnv *env, jclass jcls,
    jint surfaceWidth, jint surfaceHeight,
    jint deviceWidth, jint deviceHeight, jfloat density, jfloat rate)
{
    SDL_LockMutex(Android_ActivityMutex);

    Android_SetScreenResolution(surfaceWidth, surfaceHeight, deviceWidth, deviceHeight, density, rate);

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Resize
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeResize)(
    JNIEnv *env, jclass jcls)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (Android_Window) {
        Android_SendResize(Android_Window);
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetNaturalOrientation)(
    JNIEnv *env, jclass jcls,
    jint orientation)
{
    displayNaturalOrientation = (SDL_DisplayOrientation)orientation;
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeRotationChanged)(
    JNIEnv *env, jclass jcls,
    jint rotation)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (displayNaturalOrientation == SDL_ORIENTATION_LANDSCAPE) {
        rotation += 90;
    }

    switch (rotation % 360) {
    case 0:
        displayCurrentOrientation = SDL_ORIENTATION_PORTRAIT;
        break;
    case 90:
        displayCurrentOrientation = SDL_ORIENTATION_LANDSCAPE;
        break;
    case 180:
        displayCurrentOrientation = SDL_ORIENTATION_PORTRAIT_FLIPPED;
        break;
    case 270:
        displayCurrentOrientation = SDL_ORIENTATION_LANDSCAPE_FLIPPED;
        break;
    default:
        displayCurrentOrientation = SDL_ORIENTATION_UNKNOWN;
        break;
    }

    Android_SetOrientation(displayCurrentOrientation);

    SDL_UnlockMutex(Android_ActivityMutex);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeInsetsChanged)(
    JNIEnv *env, jclass jcls,
    jint left, jint right, jint top, jint bottom)
{
    SDL_LockMutex(Android_ActivityMutex);

    Android_SetWindowSafeAreaInsets(left, right, top, bottom);

    SDL_UnlockMutex(Android_ActivityMutex);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeAddTouch)(
    JNIEnv *env, jclass cls,
    jint touchId, jstring name)
{
    const char *utfname = (*env)->GetStringUTFChars(env, name, NULL);

    SDL_AddTouch((SDL_TouchID)touchId, SDL_TOUCH_DEVICE_DIRECT, utfname);

    (*env)->ReleaseStringUTFChars(env, name, utfname);
}

JNIEXPORT void JNICALL
SDL_JAVA_AUDIO_INTERFACE(nativeAddAudioDevice)(JNIEnv *env, jclass jcls, jboolean recording,
                                         jstring name, jint device_id)
{
#if ALLOW_MULTIPLE_ANDROID_AUDIO_DEVICES
    if (SDL_GetCurrentAudioDriver() != NULL) {
        void *handle = (void *)((size_t)device_id);
        if (!SDL_FindPhysicalAudioDeviceByHandle(handle)) {
            const char *utf8name = (*env)->GetStringUTFChars(env, name, NULL);
            SDL_AddAudioDevice(recording, SDL_strdup(utf8name), NULL, handle);
            (*env)->ReleaseStringUTFChars(env, name, utf8name);
        }
    }
#endif
}

JNIEXPORT void JNICALL
SDL_JAVA_AUDIO_INTERFACE(nativeRemoveAudioDevice)(JNIEnv *env, jclass jcls, jboolean recording,
                                            jint device_id)
{
#if ALLOW_MULTIPLE_ANDROID_AUDIO_DEVICES
    if (SDL_GetCurrentAudioDriver() != NULL) {
        SDL_Log("Removing device with handle %d, recording %d", device_id, recording);
        SDL_AudioDeviceDisconnected(SDL_FindPhysicalAudioDeviceByHandle((void *)((size_t)device_id)));
    }
#endif
}

// Paddown
JNIEXPORT jboolean JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativePadDown)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint keycode)
{
#ifdef SDL_JOYSTICK_ANDROID
    return Android_OnPadDown(device_id, keycode);
#else
    return false;
#endif // SDL_JOYSTICK_ANDROID
}

// Padup
JNIEXPORT jboolean JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativePadUp)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint keycode)
{
#ifdef SDL_JOYSTICK_ANDROID
    return Android_OnPadUp(device_id, keycode);
#else
    return false;
#endif // SDL_JOYSTICK_ANDROID
}

// Joy
JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativeJoy)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint axis, jfloat value)
{
#ifdef SDL_JOYSTICK_ANDROID
    Android_OnJoy(device_id, axis, value);
#endif // SDL_JOYSTICK_ANDROID
}

// POV Hat
JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(onNativeHat)(
    JNIEnv *env, jclass jcls,
    jint device_id, jint hat_id, jint x, jint y)
{
#ifdef SDL_JOYSTICK_ANDROID
    Android_OnHat(device_id, hat_id, x, y);
#endif // SDL_JOYSTICK_ANDROID
}

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeAddJoystick)(
    JNIEnv *env, jclass jcls,
    jint device_id, jstring device_name, jstring device_desc,
    jint vendor_id, jint product_id,
    jint button_mask, jint naxes, jint axis_mask, jint nhats, jboolean can_rumble)
{
#ifdef SDL_JOYSTICK_ANDROID
    const char *name = (*env)->GetStringUTFChars(env, device_name, NULL);
    const char *desc = (*env)->GetStringUTFChars(env, device_desc, NULL);

    Android_AddJoystick(device_id, name, desc, vendor_id, product_id, button_mask, naxes, axis_mask, nhats, can_rumble);

    (*env)->ReleaseStringUTFChars(env, device_name, name);
    (*env)->ReleaseStringUTFChars(env, device_desc, desc);
#endif // SDL_JOYSTICK_ANDROID
}

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeRemoveJoystick)(
    JNIEnv *env, jclass jcls,
    jint device_id)
{
#ifdef SDL_JOYSTICK_ANDROID
    Android_RemoveJoystick(device_id);
#endif // SDL_JOYSTICK_ANDROID
}

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeAddHaptic)(
    JNIEnv *env, jclass jcls, jint device_id, jstring device_name)
{
#ifdef SDL_HAPTIC_ANDROID
    const char *name = (*env)->GetStringUTFChars(env, device_name, NULL);

    Android_AddHaptic(device_id, name);

    (*env)->ReleaseStringUTFChars(env, device_name, name);
#endif // SDL_HAPTIC_ANDROID
}

JNIEXPORT void JNICALL SDL_JAVA_CONTROLLER_INTERFACE(nativeRemoveHaptic)(
    JNIEnv *env, jclass jcls, jint device_id)
{
#ifdef SDL_HAPTIC_ANDROID
    Android_RemoveHaptic(device_id);
#endif
}

// Called from surfaceCreated()
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeSurfaceCreated)(JNIEnv *env, jclass jcls)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (Android_Window) {
        SDL_WindowData *data = Android_Window->internal;

        data->native_window = Android_JNI_GetNativeWindow();
        SDL_SetPointerProperty(SDL_GetWindowProperties(Android_Window), SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, data->native_window);
        if (data->native_window == NULL) {
            SDL_SetError("Could not fetch native window from UI thread");
        }
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Called from surfaceChanged()
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeSurfaceChanged)(JNIEnv *env, jclass jcls)
{
    SDL_LockMutex(Android_ActivityMutex);

#ifdef SDL_VIDEO_OPENGL_EGL
    if (Android_Window && (Android_Window->flags & SDL_WINDOW_OPENGL)) {
        SDL_VideoDevice *_this = SDL_GetVideoDevice();
        SDL_WindowData *data = Android_Window->internal;

        // If the surface has been previously destroyed by onNativeSurfaceDestroyed, recreate it here
        if (data->egl_surface == EGL_NO_SURFACE) {
            data->egl_surface = SDL_EGL_CreateSurface(_this, Android_Window, (NativeWindowType)data->native_window);
            SDL_SetPointerProperty(SDL_GetWindowProperties(Android_Window), SDL_PROP_WINDOW_ANDROID_SURFACE_POINTER, data->egl_surface);
        }

        // GL Context handling is done in the event loop because this function is run from the Java thread
    }
#endif

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Called from surfaceDestroyed()
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeSurfaceDestroyed)(JNIEnv *env, jclass jcls)
{
    int nb_attempt = 50;

retry:

    SDL_LockMutex(Android_ActivityMutex);

    if (Android_Window) {
        SDL_WindowData *data = Android_Window->internal;

        // Wait for Main thread being paused and context un-activated to release 'egl_surface'
        if ((Android_Window->flags & SDL_WINDOW_OPENGL) && !data->backup_done) {
            nb_attempt -= 1;
            if (nb_attempt == 0) {
                SDL_SetError("Try to release egl_surface with context probably still active");
            } else {
                SDL_UnlockMutex(Android_ActivityMutex);
                SDL_Delay(10);
                goto retry;
            }
        }

#ifdef SDL_VIDEO_OPENGL_EGL
        if (data->egl_surface != EGL_NO_SURFACE) {
            SDL_EGL_DestroySurface(SDL_GetVideoDevice(), data->egl_surface);
            data->egl_surface = EGL_NO_SURFACE;
        }
#endif

        if (data->native_window) {
            ANativeWindow_release(data->native_window);
            data->native_window = NULL;
        }

        // GL Context handling is done in the event loop because this function is run from the Java thread
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Keydown
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeKeyDown)(
    JNIEnv *env, jclass jcls,
    jint keycode)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (Android_Window) {
        Android_OnKeyDown(keycode);
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Keyup
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeKeyUp)(
    JNIEnv *env, jclass jcls,
    jint keycode)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (Android_Window) {
        Android_OnKeyUp(keycode);
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Virtual keyboard return key might stop text input
JNIEXPORT jboolean JNICALL SDL_JAVA_INTERFACE(onNativeSoftReturnKey)(
    JNIEnv *env, jclass jcls)
{
    if (SDL_GetHintBoolean(SDL_HINT_RETURN_KEY_HIDES_IME, false)) {
        SDL_StopTextInput(Android_Window);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// Keyboard Focus Lost
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeKeyboardFocusLost)(
    JNIEnv *env, jclass jcls)
{
    // Calling SDL_StopTextInput will take care of hiding the keyboard and cleaning up the DummyText widget
    SDL_StopTextInput(Android_Window);
}

// Touch
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeTouch)(
    JNIEnv *env, jclass jcls,
    jint touch_device_id_in, jint pointer_finger_id_in,
    jint action, jfloat x, jfloat y, jfloat p)
{
    SDL_LockMutex(Android_ActivityMutex);

    Android_OnTouch(Android_Window, touch_device_id_in, pointer_finger_id_in, action, x, y, p);

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Mouse
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeMouse)(
    JNIEnv *env, jclass jcls,
    jint button, jint action, jfloat x, jfloat y, jboolean relative)
{
    SDL_LockMutex(Android_ActivityMutex);

    Android_OnMouse(Android_Window, button, action, x, y, relative);

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Pen
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativePen)(
    JNIEnv *env, jclass jcls,
    jint pen_id_in, jint button, jint action, jfloat x, jfloat y, jfloat p)
{
    SDL_LockMutex(Android_ActivityMutex);

    Android_OnPen(Android_Window, pen_id_in, button, action, x, y, p);

    SDL_UnlockMutex(Android_ActivityMutex);
}

// Accelerometer
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeAccel)(
    JNIEnv *env, jclass jcls,
    jfloat x, jfloat y, jfloat z)
{
    fLastAccelerometer[0] = x;
    fLastAccelerometer[1] = y;
    fLastAccelerometer[2] = z;
    bHasNewData = true;
}

// Clipboard
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeClipboardChanged)(
    JNIEnv *env, jclass jcls)
{
    // TODO: compute new mime types
    SDL_SendClipboardUpdate(false, NULL, 0);
}

// Low memory
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeLowMemory)(
    JNIEnv *env, jclass cls)
{
    Android_SendLifecycleEvent(SDL_ANDROID_LIFECYCLE_LOWMEMORY);
}

/* Locale
 * requires android:configChanges="layoutDirection|locale" in AndroidManifest.xml */
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeLocaleChanged)(
    JNIEnv *env, jclass cls)
{
    SDL_SendAppEvent(SDL_EVENT_LOCALE_CHANGED);
}

// Dark mode
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeDarkModeChanged)(
    JNIEnv *env, jclass cls, jboolean enabled)
{
    Android_SetDarkMode(enabled);
}

// Send Quit event to "SDLThread" thread
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSendQuit)(
    JNIEnv *env, jclass cls)
{
    Android_SendLifecycleEvent(SDL_ANDROID_LIFECYCLE_DESTROY);
}

// Activity ends
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeQuit)(
    JNIEnv *env, jclass cls)
{
    const char *str;

    if (Android_ActivityMutex) {
        SDL_DestroyMutex(Android_ActivityMutex);
        Android_ActivityMutex = NULL;
    }

    if (Android_LifecycleMutex) {
        SDL_DestroyMutex(Android_LifecycleMutex);
        Android_LifecycleMutex = NULL;
    }

    if (Android_LifecycleEventSem) {
        SDL_DestroySemaphore(Android_LifecycleEventSem);
        Android_LifecycleEventSem = NULL;
    }

    Android_NumLifecycleEvents = 0;

    Internal_Android_Destroy_AssetManager();

    str = SDL_GetError();
    if (str && str[0]) {
        __android_log_print(ANDROID_LOG_ERROR, "SDL", "SDLActivity thread ends (error=%s)", str);
    } else {
        __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "SDLActivity thread ends");
    }
}

// Pause
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativePause)(
    JNIEnv *env, jclass cls)
{
    __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "nativePause()");

    Android_SendLifecycleEvent(SDL_ANDROID_LIFECYCLE_PAUSE);
}

// Resume
JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeResume)(
    JNIEnv *env, jclass cls)
{
    __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "nativeResume()");

    Android_SendLifecycleEvent(SDL_ANDROID_LIFECYCLE_RESUME);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeFocusChanged)(
    JNIEnv *env, jclass cls, jboolean hasFocus)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (Android_Window) {
        __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "nativeFocusChanged()");
        SDL_SendWindowEvent(Android_Window, (hasFocus ? SDL_EVENT_WINDOW_FOCUS_GAINED : SDL_EVENT_WINDOW_FOCUS_LOST), 0, 0);
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE_INPUT_CONNECTION(nativeCommitText)(
    JNIEnv *env, jclass cls,
    jstring text, jint newCursorPosition)
{
    const char *utftext = (*env)->GetStringUTFChars(env, text, NULL);

    SDL_SendKeyboardText(utftext);

    (*env)->ReleaseStringUTFChars(env, text, utftext);
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE_INPUT_CONNECTION(nativeGenerateScancodeForUnichar)(
    JNIEnv *env, jclass cls,
    jchar chUnicode)
{
    SDL_SendKeyboardUnicodeKey(0, chUnicode);
}

JNIEXPORT jstring JNICALL SDL_JAVA_INTERFACE(nativeGetHint)(
    JNIEnv *env, jclass cls,
    jstring name)
{
    const char *utfname = (*env)->GetStringUTFChars(env, name, NULL);
    const char *hint = SDL_GetHint(utfname);

    jstring result = (*env)->NewStringUTF(env, hint);
    (*env)->ReleaseStringUTFChars(env, name, utfname);

    return result;
}

JNIEXPORT jboolean JNICALL SDL_JAVA_INTERFACE(nativeGetHintBoolean)(
    JNIEnv *env, jclass cls,
    jstring name, jboolean default_value)
{
    jboolean result;

    const char *utfname = (*env)->GetStringUTFChars(env, name, NULL);
    result = SDL_GetHintBoolean(utfname, default_value);
    (*env)->ReleaseStringUTFChars(env, name, utfname);

    return result;
}

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativeSetenv)(
    JNIEnv *env, jclass cls,
    jstring name, jstring value)
{
    const char *utfname = (*env)->GetStringUTFChars(env, name, NULL);
    const char *utfvalue = (*env)->GetStringUTFChars(env, value, NULL);

    // This is only called at startup, to initialize the environment
    // Note that we call setenv() directly to avoid affecting SDL environments
    setenv(utfname, utfvalue, 1); // This should NOT be SDL_setenv()

    if (SDL_strcmp(utfname, SDL_HINT_ANDROID_ALLOW_RECREATE_ACTIVITY) == 0) {
        // Special handling for this hint, which needs to persist outside the normal application flow
        // Only set this the first time we run, in case it's been set by the application via SDL_SetHint()
        if (!allow_recreate_activity_set) {
            Android_SetAllowRecreateActivity(SDL_GetStringBoolean(utfvalue, false));
        }
    }

    (*env)->ReleaseStringUTFChars(env, name, utfname);
    (*env)->ReleaseStringUTFChars(env, value, utfvalue);
}

/*******************************************************************************
             Functions called by SDL into Java
*******************************************************************************/

static SDL_AtomicInt s_active;
struct LocalReferenceHolder
{
    JNIEnv *m_env;
    const char *m_func;
};

static struct LocalReferenceHolder LocalReferenceHolder_Setup(const char *func)
{
    struct LocalReferenceHolder refholder;
    refholder.m_env = NULL;
    refholder.m_func = func;
#ifdef DEBUG_JNI
    SDL_Log("Entering function %s", func);
#endif
    return refholder;
}

static bool LocalReferenceHolder_Init(struct LocalReferenceHolder *refholder, JNIEnv *env)
{
    const int capacity = 16;
    if ((*env)->PushLocalFrame(env, capacity) < 0) {
        SDL_SetError("Failed to allocate enough JVM local references");
        return false;
    }
    SDL_AtomicIncRef(&s_active);
    refholder->m_env = env;
    return true;
}

static void LocalReferenceHolder_Cleanup(struct LocalReferenceHolder *refholder)
{
#ifdef DEBUG_JNI
    SDL_Log("Leaving function %s", refholder->m_func);
#endif
    if (refholder->m_env) {
        JNIEnv *env = refholder->m_env;
        (*env)->PopLocalFrame(env, NULL);
        SDL_AtomicDecRef(&s_active);
    }
}

ANativeWindow *Android_JNI_GetNativeWindow(void)
{
    ANativeWindow *anw = NULL;
    jobject s;
    JNIEnv *env = Android_JNI_GetEnv();

    s = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetNativeSurface);
    if (s) {
        anw = ANativeWindow_fromSurface(env, s);
        (*env)->DeleteLocalRef(env, s);
    }

    return anw;
}

void Android_JNI_SetActivityTitle(const char *title)
{
    JNIEnv *env = Android_JNI_GetEnv();

    jstring jtitle = (*env)->NewStringUTF(env, title);
    (*env)->CallStaticBooleanMethod(env, mActivityClass, midSetActivityTitle, jtitle);
    (*env)->DeleteLocalRef(env, jtitle);
}

void Android_JNI_SetWindowStyle(bool fullscreen)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mActivityClass, midSetWindowStyle, fullscreen ? 1 : 0);
}

void Android_JNI_SetOrientation(int w, int h, int resizable, const char *hint)
{
    JNIEnv *env = Android_JNI_GetEnv();

    jstring jhint = (*env)->NewStringUTF(env, (hint ? hint : ""));
    (*env)->CallStaticVoidMethod(env, mActivityClass, midSetOrientation, w, h, (resizable ? 1 : 0), jhint);
    (*env)->DeleteLocalRef(env, jhint);
}

SDL_DisplayOrientation Android_JNI_GetDisplayNaturalOrientation(void)
{
    return displayNaturalOrientation;
}

SDL_DisplayOrientation Android_JNI_GetDisplayCurrentOrientation(void)
{
    return displayCurrentOrientation;
}

void Android_JNI_MinizeWindow(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mActivityClass, midMinimizeWindow);
}

bool Android_JNI_ShouldMinimizeOnFocusLoss(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midShouldMinimizeOnFocusLoss);
}

bool Android_JNI_GetAccelerometerValues(float values[3])
{
    bool result = false;

    if (bHasNewData) {
        int i;
        for (i = 0; i < 3; ++i) {
            values[i] = fLastAccelerometer[i];
        }
        bHasNewData = false;
        result = true;
    }

    return result;
}

/*
 * Audio support
 */
void Android_StartAudioHotplug(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    JNIEnv *env = Android_JNI_GetEnv();
    // this will fire the callback for each existing device right away (which will eventually SDL_AddAudioDevice), and again later when things change.
    (*env)->CallStaticVoidMethod(env, mAudioManagerClass, midRegisterAudioDeviceCallback);
    *default_playback = *default_recording = NULL;  // !!! FIXME: how do you decide the default device id?
}

void Android_StopAudioHotplug(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mAudioManagerClass, midUnregisterAudioDeviceCallback);
}

static void Android_JNI_AudioSetThreadPriority(int recording, int device_id)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mAudioManagerClass, midAudioSetThreadPriority, recording, device_id);
}

void Android_AudioThreadInit(SDL_AudioDevice *device)
{
    Android_JNI_AudioSetThreadPriority((int) device->recording, (int)device->instance_id);
}

// Test for an exception and call SDL_SetError with its detail if one occurs
// If the parameter silent is truthy then SDL_SetError() will not be called.
static bool Android_JNI_ExceptionOccurred(bool silent)
{
    JNIEnv *env = Android_JNI_GetEnv();
    jthrowable exception;

    // Detect mismatch LocalReferenceHolder_Init/Cleanup
    SDL_assert(SDL_GetAtomicInt(&s_active) > 0);

    exception = (*env)->ExceptionOccurred(env);
    if (exception != NULL) {
        jmethodID mid;

        // Until this happens most JNI operations have undefined behaviour
        (*env)->ExceptionClear(env);

        if (!silent) {
            jclass exceptionClass = (*env)->GetObjectClass(env, exception);
            jclass classClass = (*env)->FindClass(env, "java/lang/Class");
            jstring exceptionName;
            const char *exceptionNameUTF8;
            jstring exceptionMessage;

            mid = (*env)->GetMethodID(env, classClass, "getName", "()Ljava/lang/String;");
            exceptionName = (jstring)(*env)->CallObjectMethod(env, exceptionClass, mid);
            exceptionNameUTF8 = (*env)->GetStringUTFChars(env, exceptionName, 0);

            mid = (*env)->GetMethodID(env, exceptionClass, "getMessage", "()Ljava/lang/String;");
            exceptionMessage = (jstring)(*env)->CallObjectMethod(env, exception, mid);

            if (exceptionMessage != NULL) {
                const char *exceptionMessageUTF8 = (*env)->GetStringUTFChars(env, exceptionMessage, 0);
                SDL_SetError("%s: %s", exceptionNameUTF8, exceptionMessageUTF8);
                (*env)->ReleaseStringUTFChars(env, exceptionMessage, exceptionMessageUTF8);
            } else {
                SDL_SetError("%s", exceptionNameUTF8);
            }

            (*env)->ReleaseStringUTFChars(env, exceptionName, exceptionNameUTF8);
        }

        return true;
    }

    return false;
}

static void Internal_Android_Create_AssetManager(void)
{

    struct LocalReferenceHolder refs = LocalReferenceHolder_Setup(__FUNCTION__);
    JNIEnv *env = Android_JNI_GetEnv();
    jmethodID mid;
    jobject context;
    jobject javaAssetManager;

    if (!LocalReferenceHolder_Init(&refs, env)) {
        LocalReferenceHolder_Cleanup(&refs);
        return;
    }

    // context = SDLActivity.getContext();
    context = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);

    // javaAssetManager = context.getAssets();
    mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, context),
                              "getAssets", "()Landroid/content/res/AssetManager;");
    javaAssetManager = (*env)->CallObjectMethod(env, context, mid);

    /**
     * Given a Dalvik AssetManager object, obtain the corresponding native AAssetManager
     * object.  Note that the caller is responsible for obtaining and holding a VM reference
     * to the jobject to prevent its being garbage collected while the native object is
     * in use.
     */
    javaAssetManagerRef = (*env)->NewGlobalRef(env, javaAssetManager);
    asset_manager = AAssetManager_fromJava(env, javaAssetManagerRef);

    if (!asset_manager) {
        (*env)->DeleteGlobalRef(env, javaAssetManagerRef);
        Android_JNI_ExceptionOccurred(true);
    }

    LocalReferenceHolder_Cleanup(&refs);
}

static void Internal_Android_Destroy_AssetManager(void)
{
    JNIEnv *env = Android_JNI_GetEnv();

    if (asset_manager) {
        (*env)->DeleteGlobalRef(env, javaAssetManagerRef);
        asset_manager = NULL;
    }
}

bool Android_JNI_FileOpen(void **puserdata, const char *fileName, const char *mode)
{
    SDL_assert(puserdata != NULL);

    AAsset *asset = NULL;
    *puserdata = NULL;

    if (!asset_manager) {
        Internal_Android_Create_AssetManager();
    }

    if (!asset_manager) {
        return SDL_SetError("Couldn't create asset manager");
    }

    asset = AAssetManager_open(asset_manager, fileName, AASSET_MODE_UNKNOWN);
    if (!asset) {
        return SDL_SetError("Couldn't open asset '%s'", fileName);
    }

    *puserdata = (void *)asset;
    return true;
}

size_t Android_JNI_FileRead(void *userdata, void *buffer, size_t size, SDL_IOStatus *status)
{
    const int bytes = AAsset_read((AAsset *)userdata, buffer, size);
    if (bytes < 0) {
        SDL_SetError("AAsset_read() failed");
        return 0;
    }
    return (size_t)bytes;
}

size_t Android_JNI_FileWrite(void *userdata, const void *buffer, size_t size, SDL_IOStatus *status)
{
    SDL_SetError("Cannot write to Android package filesystem");
    return 0;
}

Sint64 Android_JNI_FileSize(void *userdata)
{
    return (Sint64) AAsset_getLength64((AAsset *)userdata);
}

Sint64 Android_JNI_FileSeek(void *userdata, Sint64 offset, SDL_IOWhence whence)
{
    return (Sint64) AAsset_seek64((AAsset *)userdata, offset, (int)whence);
}

bool Android_JNI_FileClose(void *userdata)
{
    AAsset_close((AAsset *)userdata);
    return true;
}

bool Android_JNI_SetClipboardText(const char *text)
{
    JNIEnv *env = Android_JNI_GetEnv();
    jstring string = (*env)->NewStringUTF(env, text);
    (*env)->CallStaticVoidMethod(env, mActivityClass, midClipboardSetText, string);
    (*env)->DeleteLocalRef(env, string);
    return true;
}

char *Android_JNI_GetClipboardText(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    char *text = NULL;
    jstring string;

    string = (*env)->CallStaticObjectMethod(env, mActivityClass, midClipboardGetText);
    if (string) {
        const char *utf = (*env)->GetStringUTFChars(env, string, 0);
        if (utf) {
            text = SDL_strdup(utf);
            (*env)->ReleaseStringUTFChars(env, string, utf);
        }
        (*env)->DeleteLocalRef(env, string);
    }

    return (!text) ? SDL_strdup("") : text;
}

bool Android_JNI_HasClipboardText(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midClipboardHasText);
}

/* returns 0 on success or -1 on error (others undefined then)
 * returns truthy or falsy value in plugged, charged and battery
 * returns the value in seconds and percent or -1 if not available
 */
int Android_JNI_GetPowerInfo(int *plugged, int *charged, int *battery, int *seconds, int *percent)
{
    struct LocalReferenceHolder refs = LocalReferenceHolder_Setup(__FUNCTION__);
    JNIEnv *env = Android_JNI_GetEnv();
    jmethodID mid;
    jobject context;
    jstring action;
    jclass cls;
    jobject filter;
    jobject intent;
    jstring iname;
    jmethodID imid;
    jstring bname;
    jmethodID bmid;
    if (!LocalReferenceHolder_Init(&refs, env)) {
        LocalReferenceHolder_Cleanup(&refs);
        return -1;
    }

    // context = SDLActivity.getContext();
    context = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);

    action = (*env)->NewStringUTF(env, "android.intent.action.BATTERY_CHANGED");

    cls = (*env)->FindClass(env, "android/content/IntentFilter");

    mid = (*env)->GetMethodID(env, cls, "<init>", "(Ljava/lang/String;)V");
    filter = (*env)->NewObject(env, cls, mid, action);

    (*env)->DeleteLocalRef(env, action);

    mid = (*env)->GetMethodID(env, mActivityClass, "registerReceiver", "(Landroid/content/BroadcastReceiver;Landroid/content/IntentFilter;)Landroid/content/Intent;");
    intent = (*env)->CallObjectMethod(env, context, mid, NULL, filter);

    (*env)->DeleteLocalRef(env, filter);

    cls = (*env)->GetObjectClass(env, intent);

    imid = (*env)->GetMethodID(env, cls, "getIntExtra", "(Ljava/lang/String;I)I");

    // Watch out for C89 scoping rules because of the macro
#define GET_INT_EXTRA(var, key)                                  \
    int var;                                                     \
    iname = (*env)->NewStringUTF(env, key);                      \
    (var) = (*env)->CallIntMethod(env, intent, imid, iname, -1); \
    (*env)->DeleteLocalRef(env, iname);

    bmid = (*env)->GetMethodID(env, cls, "getBooleanExtra", "(Ljava/lang/String;Z)Z");

    // Watch out for C89 scoping rules because of the macro
#define GET_BOOL_EXTRA(var, key)                                            \
    int var;                                                                \
    bname = (*env)->NewStringUTF(env, key);                                 \
    (var) = (*env)->CallBooleanMethod(env, intent, bmid, bname, JNI_FALSE); \
    (*env)->DeleteLocalRef(env, bname);

    if (plugged) {
        // Watch out for C89 scoping rules because of the macro
        GET_INT_EXTRA(plug, "plugged") // == BatteryManager.EXTRA_PLUGGED (API 5)
        if (plug == -1) {
            LocalReferenceHolder_Cleanup(&refs);
            return -1;
        }
        // 1 == BatteryManager.BATTERY_PLUGGED_AC
        // 2 == BatteryManager.BATTERY_PLUGGED_USB
        *plugged = (0 < plug) ? 1 : 0;
    }

    if (charged) {
        // Watch out for C89 scoping rules because of the macro
        GET_INT_EXTRA(status, "status") // == BatteryManager.EXTRA_STATUS (API 5)
        if (status == -1) {
            LocalReferenceHolder_Cleanup(&refs);
            return -1;
        }
        // 5 == BatteryManager.BATTERY_STATUS_FULL
        *charged = (status == 5) ? 1 : 0;
    }

    if (battery) {
        GET_BOOL_EXTRA(present, "present") // == BatteryManager.EXTRA_PRESENT (API 5)
        *battery = present ? 1 : 0;
    }

    if (seconds) {
        *seconds = -1; // not possible
    }

    if (percent) {
        int level;
        int scale;

        // Watch out for C89 scoping rules because of the macro
        {
            GET_INT_EXTRA(level_temp, "level") // == BatteryManager.EXTRA_LEVEL (API 5)
            level = level_temp;
        }
        // Watch out for C89 scoping rules because of the macro
        {
            GET_INT_EXTRA(scale_temp, "scale") // == BatteryManager.EXTRA_SCALE (API 5)
            scale = scale_temp;
        }

        if ((level == -1) || (scale == -1)) {
            LocalReferenceHolder_Cleanup(&refs);
            return -1;
        }
        *percent = level * 100 / scale;
    }

    (*env)->DeleteLocalRef(env, intent);

    LocalReferenceHolder_Cleanup(&refs);
    return 0;
}

// Add all touch devices
void Android_JNI_InitTouch(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mActivityClass, midInitTouch);
}

void Android_JNI_PollInputDevices(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mControllerManagerClass, midPollInputDevices);
}

void Android_JNI_PollHapticDevices(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mControllerManagerClass, midPollHapticDevices);
}

void Android_JNI_HapticRun(int device_id, float intensity, int length)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mControllerManagerClass, midHapticRun, device_id, intensity, length);
}

void Android_JNI_HapticRumble(int device_id, float low_frequency_intensity, float high_frequency_intensity, int length)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mControllerManagerClass, midHapticRumble, device_id, low_frequency_intensity, high_frequency_intensity, length);
}

void Android_JNI_HapticStop(int device_id)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mControllerManagerClass, midHapticStop, device_id);
}

// See SDLActivity.java for constants.
#define COMMAND_SET_KEEP_SCREEN_ON 5

bool SDL_SendAndroidMessage(Uint32 command, int param)
{
    if (command < 0x8000) {
        return SDL_InvalidParamError("command");
    }
    return Android_JNI_SendMessage(command, param);
}

// sends message to be handled on the UI event dispatch thread
bool Android_JNI_SendMessage(int command, int param)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midSendMessage, command, param);
}

bool Android_JNI_SuspendScreenSaver(bool suspend)
{
    return Android_JNI_SendMessage(COMMAND_SET_KEEP_SCREEN_ON, (suspend == false) ? 0 : 1);
}

void Android_JNI_ShowScreenKeyboard(int input_type, SDL_Rect *inputRect)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticBooleanMethod(env, mActivityClass, midShowTextInput,
                                    input_type,
                                    inputRect->x,
                                    inputRect->y,
                                    inputRect->w,
                                    inputRect->h);
}

void Android_JNI_HideScreenKeyboard(void)
{
    // has to match Activity constant
    const int COMMAND_TEXTEDIT_HIDE = 3;
    Android_JNI_SendMessage(COMMAND_TEXTEDIT_HIDE, 0);
}

bool Android_JNI_IsScreenKeyboardShown(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    jboolean is_shown = 0;
    is_shown = (*env)->CallStaticBooleanMethod(env, mActivityClass, midIsScreenKeyboardShown);
    return is_shown;
}

bool Android_JNI_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    JNIEnv *env;
    jclass clazz;
    jmethodID mid;
    jobject context;
    jstring title;
    jstring message;
    jintArray button_flags;
    jintArray button_ids;
    jobjectArray button_texts;
    jintArray colors;
    jobject text;
    jint temp;
    int i;

    env = Android_JNI_GetEnv();

    // convert parameters

    clazz = (*env)->FindClass(env, "java/lang/String");

    title = (*env)->NewStringUTF(env, messageboxdata->title);
    message = (*env)->NewStringUTF(env, messageboxdata->message);

    button_flags = (*env)->NewIntArray(env, messageboxdata->numbuttons);
    button_ids = (*env)->NewIntArray(env, messageboxdata->numbuttons);
    button_texts = (*env)->NewObjectArray(env, messageboxdata->numbuttons,
                                          clazz, NULL);
    for (i = 0; i < messageboxdata->numbuttons; ++i) {
        const SDL_MessageBoxButtonData *sdlButton;

        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
            sdlButton = &messageboxdata->buttons[messageboxdata->numbuttons - 1 - i];
        } else {
            sdlButton = &messageboxdata->buttons[i];
        }

        temp = sdlButton->flags;
        (*env)->SetIntArrayRegion(env, button_flags, i, 1, &temp);
        temp = sdlButton->buttonID;
        (*env)->SetIntArrayRegion(env, button_ids, i, 1, &temp);
        text = (*env)->NewStringUTF(env, sdlButton->text);
        (*env)->SetObjectArrayElement(env, button_texts, i, text);
        (*env)->DeleteLocalRef(env, text);
    }

    if (messageboxdata->colorScheme) {
        colors = (*env)->NewIntArray(env, SDL_MESSAGEBOX_COLOR_COUNT);
        for (i = 0; i < SDL_MESSAGEBOX_COLOR_COUNT; ++i) {
            temp = (0xFFU << 24) |
                   (messageboxdata->colorScheme->colors[i].r << 16) |
                   (messageboxdata->colorScheme->colors[i].g << 8) |
                   (messageboxdata->colorScheme->colors[i].b << 0);
            (*env)->SetIntArrayRegion(env, colors, i, 1, &temp);
        }
    } else {
        colors = NULL;
    }

    (*env)->DeleteLocalRef(env, clazz);

    // context = SDLActivity.getContext();
    context = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);

    clazz = (*env)->GetObjectClass(env, context);

    mid = (*env)->GetMethodID(env, clazz,
                              "messageboxShowMessageBox", "(ILjava/lang/String;Ljava/lang/String;[I[I[Ljava/lang/String;[I)I");
    *buttonID = (*env)->CallIntMethod(env, context, mid,
                                      messageboxdata->flags,
                                      title,
                                      message,
                                      button_flags,
                                      button_ids,
                                      button_texts,
                                      colors);

    (*env)->DeleteLocalRef(env, context);
    (*env)->DeleteLocalRef(env, clazz);

    // delete parameters

    (*env)->DeleteLocalRef(env, title);
    (*env)->DeleteLocalRef(env, message);
    (*env)->DeleteLocalRef(env, button_flags);
    (*env)->DeleteLocalRef(env, button_ids);
    (*env)->DeleteLocalRef(env, button_texts);
    (*env)->DeleteLocalRef(env, colors);

    return true;
}

/*
//////////////////////////////////////////////////////////////////////////////
//
// Functions exposed to SDL applications in SDL_system.h
//////////////////////////////////////////////////////////////////////////////
*/

void *SDL_GetAndroidJNIEnv(void)
{
    return Android_JNI_GetEnv();
}

void *SDL_GetAndroidActivity(void)
{
    // See SDL_system.h for caveats on using this function.

    JNIEnv *env = Android_JNI_GetEnv();
    if (!env) {
        return NULL;
    }

    // return SDLActivity.getContext();
    return (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);
}

int SDL_GetAndroidSDKVersion(void)
{
    static int sdk_version;
    if (!sdk_version) {
        char sdk[PROP_VALUE_MAX] = { 0 };
        if (__system_property_get("ro.build.version.sdk", sdk) != 0) {
            sdk_version = SDL_atoi(sdk);
        }
    }
    return sdk_version;
}

bool SDL_IsAndroidTablet(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midIsTablet);
}

bool SDL_IsAndroidTV(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midIsAndroidTV);
}

bool SDL_IsChromebook(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midIsChromebook);
}

bool SDL_IsDeXMode(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midIsDeXMode);
}

void SDL_SendAndroidBackButton(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mActivityClass, midManualBackButton);
}

const char *SDL_GetAndroidInternalStoragePath(void)
{
    static char *s_AndroidInternalFilesPath = NULL;

    if (!s_AndroidInternalFilesPath) {
        struct LocalReferenceHolder refs = LocalReferenceHolder_Setup(__FUNCTION__);
        jmethodID mid;
        jobject context;
        jobject fileObject;
        jstring pathString;
        const char *path;

        JNIEnv *env = Android_JNI_GetEnv();
        if (!LocalReferenceHolder_Init(&refs, env)) {
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // context = SDLActivity.getContext();
        context = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);
        if (!context) {
            SDL_SetError("Couldn't get Android context!");
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // fileObj = context.getFilesDir();
        mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, context),
                                  "getFilesDir", "()Ljava/io/File;");
        fileObject = (*env)->CallObjectMethod(env, context, mid);
        if (!fileObject) {
            SDL_SetError("Couldn't get internal directory");
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // path = fileObject.getCanonicalPath();
        mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, fileObject),
                                  "getCanonicalPath", "()Ljava/lang/String;");
        pathString = (jstring)(*env)->CallObjectMethod(env, fileObject, mid);
        if (Android_JNI_ExceptionOccurred(false)) {
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        path = (*env)->GetStringUTFChars(env, pathString, NULL);
        s_AndroidInternalFilesPath = SDL_strdup(path);
        (*env)->ReleaseStringUTFChars(env, pathString, path);

        LocalReferenceHolder_Cleanup(&refs);
    }
    return s_AndroidInternalFilesPath;
}

Uint32 SDL_GetAndroidExternalStorageState(void)
{
    struct LocalReferenceHolder refs = LocalReferenceHolder_Setup(__FUNCTION__);
    jmethodID mid;
    jclass cls;
    jstring stateString;
    const char *state_string;
    Uint32 stateFlags;

    JNIEnv *env = Android_JNI_GetEnv();
    if (!LocalReferenceHolder_Init(&refs, env)) {
        LocalReferenceHolder_Cleanup(&refs);
        return 0;
    }

    cls = (*env)->FindClass(env, "android/os/Environment");
    mid = (*env)->GetStaticMethodID(env, cls,
                                    "getExternalStorageState", "()Ljava/lang/String;");
    stateString = (jstring)(*env)->CallStaticObjectMethod(env, cls, mid);

    state_string = (*env)->GetStringUTFChars(env, stateString, NULL);

    // Print an info message so people debugging know the storage state
    __android_log_print(ANDROID_LOG_INFO, "SDL", "external storage state: %s", state_string);

    if (SDL_strcmp(state_string, "mounted") == 0) {
        stateFlags = SDL_ANDROID_EXTERNAL_STORAGE_READ |
                     SDL_ANDROID_EXTERNAL_STORAGE_WRITE;
    } else if (SDL_strcmp(state_string, "mounted_ro") == 0) {
        stateFlags = SDL_ANDROID_EXTERNAL_STORAGE_READ;
    } else {
        stateFlags = 0;
    }
    (*env)->ReleaseStringUTFChars(env, stateString, state_string);

    LocalReferenceHolder_Cleanup(&refs);

    return stateFlags;
}

const char *SDL_GetAndroidExternalStoragePath(void)
{
    static char *s_AndroidExternalFilesPath = NULL;

    if (!s_AndroidExternalFilesPath) {
        struct LocalReferenceHolder refs = LocalReferenceHolder_Setup(__FUNCTION__);
        jmethodID mid;
        jobject context;
        jobject fileObject;
        jstring pathString;
        const char *path;

        JNIEnv *env = Android_JNI_GetEnv();
        if (!LocalReferenceHolder_Init(&refs, env)) {
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // context = SDLActivity.getContext();
        context = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);

        // fileObj = context.getExternalFilesDir();
        mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, context),
                                  "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
        fileObject = (*env)->CallObjectMethod(env, context, mid, NULL);
        if (!fileObject) {
            SDL_SetError("Couldn't get external directory");
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // path = fileObject.getAbsolutePath();
        mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, fileObject),
                                  "getAbsolutePath", "()Ljava/lang/String;");
        pathString = (jstring)(*env)->CallObjectMethod(env, fileObject, mid);

        path = (*env)->GetStringUTFChars(env, pathString, NULL);
        s_AndroidExternalFilesPath = SDL_strdup(path);
        (*env)->ReleaseStringUTFChars(env, pathString, path);

        LocalReferenceHolder_Cleanup(&refs);
    }
    return s_AndroidExternalFilesPath;
}

const char *SDL_GetAndroidCachePath(void)
{
    // !!! FIXME: lots of duplication with SDL_GetAndroidExternalStoragePath and SDL_GetAndroidInternalStoragePath; consolidate these functions!
    static char *s_AndroidCachePath = NULL;

    if (!s_AndroidCachePath) {
        struct LocalReferenceHolder refs = LocalReferenceHolder_Setup(__FUNCTION__);
        jmethodID mid;
        jobject context;
        jobject fileObject;
        jstring pathString;
        const char *path;

        JNIEnv *env = Android_JNI_GetEnv();
        if (!LocalReferenceHolder_Init(&refs, env)) {
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // context = SDLActivity.getContext();
        context = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetContext);

        // fileObj = context.getExternalFilesDir();
        mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, context),
                                  "getCacheDir", "()Ljava/io/File;");
        fileObject = (*env)->CallObjectMethod(env, context, mid, NULL);
        if (!fileObject) {
            SDL_SetError("Couldn't get cache directory");
            LocalReferenceHolder_Cleanup(&refs);
            return NULL;
        }

        // path = fileObject.getAbsolutePath();
        mid = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, fileObject),
                                  "getAbsolutePath", "()Ljava/lang/String;");
        pathString = (jstring)(*env)->CallObjectMethod(env, fileObject, mid);

        path = (*env)->GetStringUTFChars(env, pathString, NULL);
        s_AndroidCachePath = SDL_strdup(path);
        (*env)->ReleaseStringUTFChars(env, pathString, path);

        LocalReferenceHolder_Cleanup(&refs);
    }
    return s_AndroidCachePath;
}

bool SDL_ShowAndroidToast(const char *message, int duration, int gravity, int xOffset, int yOffset)
{
    return Android_JNI_ShowToast(message, duration, gravity, xOffset, yOffset);
}

void Android_JNI_GetManifestEnvironmentVariables(void)
{
    if (!mActivityClass || !midGetManifestEnvironmentVariables) {
        __android_log_print(ANDROID_LOG_WARN, "SDL", "Request to get environment variables before JNI is ready");
        return;
    }

    if (!bHasEnvironmentVariables) {
        JNIEnv *env = Android_JNI_GetEnv();
        bool ret = (*env)->CallStaticBooleanMethod(env, mActivityClass, midGetManifestEnvironmentVariables);
        if (ret) {
            bHasEnvironmentVariables = true;
        }
    }
}

int Android_JNI_CreateCustomCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    JNIEnv *env = Android_JNI_GetEnv();
    int custom_cursor = 0;
    jintArray pixels;
    pixels = (*env)->NewIntArray(env, surface->w * surface->h);
    if (pixels) {
        (*env)->SetIntArrayRegion(env, pixels, 0, surface->w * surface->h, (int *)surface->pixels);
        custom_cursor = (*env)->CallStaticIntMethod(env, mActivityClass, midCreateCustomCursor, pixels, surface->w, surface->h, hot_x, hot_y);
        (*env)->DeleteLocalRef(env, pixels);
    } else {
        SDL_OutOfMemory();
    }
    return custom_cursor;
}

void Android_JNI_DestroyCustomCursor(int cursorID)
{
    JNIEnv *env = Android_JNI_GetEnv();
    (*env)->CallStaticVoidMethod(env, mActivityClass, midDestroyCustomCursor, cursorID);
}

bool Android_JNI_SetCustomCursor(int cursorID)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midSetCustomCursor, cursorID);
}

bool Android_JNI_SetSystemCursor(int cursorID)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midSetSystemCursor, cursorID);
}

bool Android_JNI_SupportsRelativeMouse(void)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midSupportsRelativeMouse);
}

bool Android_JNI_SetRelativeMouseEnabled(bool enabled)
{
    JNIEnv *env = Android_JNI_GetEnv();
    return (*env)->CallStaticBooleanMethod(env, mActivityClass, midSetRelativeMouseEnabled, (enabled == 1));
}

typedef struct NativePermissionRequestInfo
{
    int request_code;
    char *permission;
    SDL_RequestAndroidPermissionCallback callback;
    void *userdata;
    struct NativePermissionRequestInfo *next;
} NativePermissionRequestInfo;

static NativePermissionRequestInfo pending_permissions;

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(nativePermissionResult)(
    JNIEnv *env, jclass cls,
    jint requestCode, jboolean result)
{
    SDL_LockMutex(Android_ActivityMutex);
    NativePermissionRequestInfo *prev = &pending_permissions;
    for (NativePermissionRequestInfo *info = prev->next; info != NULL; info = info->next) {
        if (info->request_code == (int) requestCode) {
            prev->next = info->next;
            SDL_UnlockMutex(Android_ActivityMutex);
            info->callback(info->userdata, info->permission, result ? true : false);
            SDL_free(info->permission);
            SDL_free(info);
            return;
        }
        prev = info;
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

bool SDL_RequestAndroidPermission(const char *permission, SDL_RequestAndroidPermissionCallback cb, void *userdata)
{
    if (!permission) {
        return SDL_InvalidParamError("permission");
    } else if (!cb) {
        return SDL_InvalidParamError("cb");
    }

    NativePermissionRequestInfo *info = (NativePermissionRequestInfo *) SDL_calloc(1, sizeof (NativePermissionRequestInfo));
    if (!info) {
        return false;
    }

    info->permission = SDL_strdup(permission);
    if (!info->permission) {
        SDL_free(info);
        return false;
    }

    static SDL_AtomicInt next_request_code;
    info->request_code = SDL_AddAtomicInt(&next_request_code, 1);

    info->callback = cb;
    info->userdata = userdata;

    SDL_LockMutex(Android_ActivityMutex);
    info->next = pending_permissions.next;
    pending_permissions.next = info;
    SDL_UnlockMutex(Android_ActivityMutex);

    JNIEnv *env = Android_JNI_GetEnv();
    jstring jpermission = (*env)->NewStringUTF(env, permission);
    (*env)->CallStaticVoidMethod(env, mActivityClass, midRequestPermission, jpermission, info->request_code);
    (*env)->DeleteLocalRef(env, jpermission);

    return true;
}

// Show toast notification
bool Android_JNI_ShowToast(const char *message, int duration, int gravity, int xOffset, int yOffset)
{
    bool result;
    JNIEnv *env = Android_JNI_GetEnv();
    jstring jmessage = (*env)->NewStringUTF(env, message);
    result = (*env)->CallStaticBooleanMethod(env, mActivityClass, midShowToast, jmessage, duration, gravity, xOffset, yOffset);
    (*env)->DeleteLocalRef(env, jmessage);
    return result;
}

bool Android_JNI_GetLocale(char *buf, size_t buflen)
{
    AConfiguration *cfg;

    SDL_assert(buflen > 6);

    // Need to re-create the asset manager if locale has changed (SDL_EVENT_LOCALE_CHANGED)
    Internal_Android_Destroy_AssetManager();

    if (!asset_manager) {
        Internal_Android_Create_AssetManager();
    }

    if (!asset_manager) {
        return false;
    }

    cfg = AConfiguration_new();
    if (!cfg) {
        return false;
    }

    {
        char language[2] = {};
        char country[2] = {};
        size_t id = 0;

        AConfiguration_fromAssetManager(cfg, asset_manager);
        AConfiguration_getLanguage(cfg, language);
        AConfiguration_getCountry(cfg, country);

        // Indonesian is "id" according to ISO 639.2, but on Android is "in" because of Java backwards compatibility
        if (language[0] == 'i' && language[1] == 'n') {
            language[1] = 'd';
        }

        // copy language (not null terminated)
        if (language[0]) {
            buf[id++] = language[0];
            if (language[1]) {
                buf[id++] = language[1];
            }
        }

        buf[id++] = '_';

        // copy country (not null terminated)
        if (country[0]) {
            buf[id++] = country[0];
            if (country[1]) {
                buf[id++] = country[1];
            }
        }

        buf[id++] = '\0';
        SDL_assert(id <= buflen);
    }

    AConfiguration_delete(cfg);

    return true;
}

bool Android_JNI_OpenURL(const char *url)
{
    bool result;
    JNIEnv *env = Android_JNI_GetEnv();
    jstring jurl = (*env)->NewStringUTF(env, url);
    result = (*env)->CallStaticBooleanMethod(env, mActivityClass, midOpenURL, jurl);
    (*env)->DeleteLocalRef(env, jurl);
    return result;
}

int Android_JNI_OpenFileDescriptor(const char *uri, const char *mode)
{
    // Get fopen-style modes
    int moderead = 0, modewrite = 0, modeappend = 0, modeupdate = 0;

    for (const char *cmode = mode; *cmode; cmode++) {
        switch (*cmode) {
            case 'a':
                modeappend = 1;
                break;
            case 'r':
                moderead = 1;
                break;
            case 'w':
                modewrite = 1;
                break;
            case '+':
                modeupdate = 1;
                break;
            default:
                break;
        }
    }

    // Translate fopen-style modes to ContentResolver modes.
    // Android only allows "r", "w", "wt", "wa", "rw" or "rwt".
    const char *contentResolverMode = "r";

    if (moderead) {
        if (modewrite) {
            contentResolverMode = "rwt";
        } else {
            contentResolverMode = modeupdate ? "rw" : "r";
        }
    } else if (modewrite) {
        contentResolverMode = modeupdate ? "rwt" : "wt";
    } else if (modeappend) {
        contentResolverMode = modeupdate ? "rw" : "wa";
    }

    JNIEnv *env = Android_JNI_GetEnv();
    jstring jstringUri = (*env)->NewStringUTF(env, uri);
    jstring jstringMode = (*env)->NewStringUTF(env, contentResolverMode);
    jint fd = (*env)->CallStaticIntMethod(env, mActivityClass, midOpenFileDescriptor, jstringUri, jstringMode);
    (*env)->DeleteLocalRef(env, jstringUri);
    (*env)->DeleteLocalRef(env, jstringMode);

    if (fd == -1) {
        SDL_SetError("Unspecified error in JNI");
    }

    return fd;
}

static struct AndroidFileDialog
{
    int request_code;
    SDL_DialogFileCallback callback;
    void *userdata;
} mAndroidFileDialogData;

JNIEXPORT void JNICALL SDL_JAVA_INTERFACE(onNativeFileDialog)(
    JNIEnv *env, jclass jcls,
    jint requestCode, jobjectArray fileList, jint filter)
{
    if (mAndroidFileDialogData.callback != NULL && mAndroidFileDialogData.request_code == requestCode) {
        if (fileList == NULL) {
            SDL_SetError("Unspecified error in JNI");
            mAndroidFileDialogData.callback(mAndroidFileDialogData.userdata, NULL, -1);
            mAndroidFileDialogData.callback = NULL;
            return;
        }

        // Convert fileList to string
        size_t count = (*env)->GetArrayLength(env, fileList);
        char **charFileList = SDL_calloc(count + 1, sizeof(char*));

        if (charFileList == NULL) {
            mAndroidFileDialogData.callback(mAndroidFileDialogData.userdata, NULL, -1);
            mAndroidFileDialogData.callback = NULL;
            return;
        }

        // Convert to UTF-8
        // TODO: Fix modified UTF-8 to classic UTF-8
        for (int i = 0; i < count; i++) {
            jstring string = (*env)->GetObjectArrayElement(env, fileList, i);
            if (!string) {
                continue;
            }

            const char *utf8string = (*env)->GetStringUTFChars(env, string, NULL);
            if (!utf8string) {
                (*env)->DeleteLocalRef(env, string);
                continue;
            }

            char *newFile = SDL_strdup(utf8string);
            if (!newFile) {
                (*env)->ReleaseStringUTFChars(env, string, utf8string);
                (*env)->DeleteLocalRef(env, string);
                mAndroidFileDialogData.callback(mAndroidFileDialogData.userdata, NULL, -1);
                mAndroidFileDialogData.callback = NULL;

                // Cleanup memory
                for (int j = 0; j < i; j++) {
                    SDL_free(charFileList[j]);
                }
                SDL_free(charFileList);
                return;
            }

            charFileList[i] = newFile;
            (*env)->ReleaseStringUTFChars(env, string, utf8string);
            (*env)->DeleteLocalRef(env, string);
        }

        // Call user-provided callback
        SDL_ClearError();
        mAndroidFileDialogData.callback(mAndroidFileDialogData.userdata, (const char *const *) charFileList, filter);
        mAndroidFileDialogData.callback = NULL;

        // Cleanup memory
        for (int i = 0; i < count; i++) {
            SDL_free(charFileList[i]);
        }
        SDL_free(charFileList);
    }
}

bool Android_JNI_OpenFileDialog(
    SDL_DialogFileCallback callback, void* userdata,
    const SDL_DialogFileFilter *filters, int nfilters, bool forwrite,
    bool multiple)
{
    if (mAndroidFileDialogData.callback != NULL) {
        SDL_SetError("Only one file dialog can be run at a time.");
        return false;
    }

    if (forwrite) {
        multiple = false;
    }

    JNIEnv *env = Android_JNI_GetEnv();

    // Setup filters
    jobjectArray filtersArray = NULL;
    if (filters) {
        jclass stringClass = (*env)->FindClass(env, "java/lang/String");
        filtersArray = (*env)->NewObjectArray(env, nfilters, stringClass, NULL);

        // Convert to string
        for (int i = 0; i < nfilters; i++) {
            jstring str = (*env)->NewStringUTF(env, filters[i].pattern);
            (*env)->SetObjectArrayElement(env, filtersArray, i, str);
            (*env)->DeleteLocalRef(env, str);
        }
    }

    // Setup data
    static SDL_AtomicInt next_request_code;
    mAndroidFileDialogData.request_code = SDL_AddAtomicInt(&next_request_code, 1);
    mAndroidFileDialogData.userdata = userdata;
    mAndroidFileDialogData.callback = callback;

    // Invoke JNI
    jboolean success = (*env)->CallStaticBooleanMethod(env, mActivityClass,
        midShowFileDialog, filtersArray, (jboolean) multiple, (jboolean) forwrite, mAndroidFileDialogData.request_code);
    (*env)->DeleteLocalRef(env, filtersArray);
    if (!success) {
        mAndroidFileDialogData.callback = NULL;
        SDL_AddAtomicInt(&next_request_code, -1);
        SDL_SetError("Unspecified error in JNI");

        return false;
    }

    return true;
}

#endif // SDL_PLATFORM_ANDROID
