/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.android.vending.expansion.downloader.impl;

import android.app.Service;
import android.content.Intent;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

/**
 * This service differs from IntentService in a few minor ways/ It will not
 * auto-stop itself after the intent is handled unless the target returns "true"
 * in should stop. Since the goal of this service is to handle a single kind of
 * intent, it does not queue up batches of intents of the same type.
 */
public abstract class CustomIntentService extends Service {
    private String mName;
    private boolean mRedelivery;
    private volatile ServiceHandler mServiceHandler;
    private volatile Looper mServiceLooper;
    private static final String LOG_TAG = "CustomIntentService";
    private static final int WHAT_MESSAGE = -10;

    public CustomIntentService(String paramString) {
        this.mName = paramString;
    }

    @Override
    public IBinder onBind(Intent paramIntent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        HandlerThread localHandlerThread = new HandlerThread("IntentService["
                + this.mName + "]");
        localHandlerThread.start();
        this.mServiceLooper = localHandlerThread.getLooper();
        this.mServiceHandler = new ServiceHandler(this.mServiceLooper);
    }

    @Override
    public void onDestroy() {
        Thread localThread = this.mServiceLooper.getThread();
        if ((localThread != null) && (localThread.isAlive())) {
            localThread.interrupt();
        }
        this.mServiceLooper.quit();
        Log.d(LOG_TAG, "onDestroy");
    }

    protected abstract void onHandleIntent(Intent paramIntent);

    protected abstract boolean shouldStop();

    @Override
    public void onStart(Intent paramIntent, int startId) {
        if (!this.mServiceHandler.hasMessages(WHAT_MESSAGE)) {
            Message localMessage = this.mServiceHandler.obtainMessage();
            localMessage.arg1 = startId;
            localMessage.obj = paramIntent;
            localMessage.what = WHAT_MESSAGE;
            this.mServiceHandler.sendMessage(localMessage);
        }
    }

    @Override
    public int onStartCommand(Intent paramIntent, int flags, int startId) {
        onStart(paramIntent, startId);
        return mRedelivery ? START_REDELIVER_INTENT : START_NOT_STICKY;
    }

    public void setIntentRedelivery(boolean enabled) {
        this.mRedelivery = enabled;
    }

    private final class ServiceHandler extends Handler {
        public ServiceHandler(Looper looper) {
            super(looper);
        }

        @Override
        public void handleMessage(Message paramMessage) {
            CustomIntentService.this
                    .onHandleIntent((Intent) paramMessage.obj);
            if (shouldStop()) {
                Log.d(LOG_TAG, "stopSelf");
                CustomIntentService.this.stopSelf(paramMessage.arg1);
                Log.d(LOG_TAG, "afterStopSelf");
            }
        }
    }
}
