/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Copyright (C) 2015 Joey Ferwerda
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Android Driver */

#ifndef ANDROID_H
#define ANDROID_H

#include "../openhmdi.h"

typedef enum {
    DROID_DUROVIS_OPEN_DIVE   = 1,
    DROID_DUROVIS_DIVE_5      = 2,
    DROID_DUROVIS_DIVE_7      = 3,
    DROID_CARL_ZEISS_VRONE    = 4,
    DROID_GOOGLE_CARDBOARD    = 5,

    DROID_NONE                = 0,
} android_hmd_profile;

//Android copy-paste from android_native_app_glue to be able to cast data to something useful
#include <poll.h>
#include <pthread.h>
#include <sched.h>

#include <android/configuration.h>
#include <android/looper.h>
#include <android/native_activity.h>

struct android_app;

struct android_poll_source {
    int32_t id;

    struct android_app* app;

    void (*process)(struct android_app* app, struct android_poll_source* source);
};

typedef struct android_app {
    void* userData;

    void (*onAppCmd)(struct android_app* app, int32_t cmd);
    int32_t (*onInputEvent)(struct android_app* app, AInputEvent* event);

    ANativeActivity* activity;
    AConfiguration* config;

    void* savedState;
    size_t savedStateSize;

    ALooper* looper;
    AInputQueue* inputQueue;
    ANativeWindow* window;
    ARect contentRect;

    int activityState;
    int destroyRequested;

    pthread_mutex_t mutex;
    pthread_cond_t cond;

    int msgread;
    int msgwrite;

    pthread_t thread;

    struct android_poll_source cmdPollSource;
    struct android_poll_source inputPollSource;

    int running;
    int stateSaved;
    int destroyed;
    int redrawNeeded;
    AInputQueue* pendingInputQueue;
    ANativeWindow* pendingWindow;
    ARect pendingContentRect;
} android_app;

enum {
    LOOPER_ID_MAIN = 1,
    LOOPER_ID_INPUT = 2,
    LOOPER_ID_USER = 3
};

#endif // ANDROID_H
