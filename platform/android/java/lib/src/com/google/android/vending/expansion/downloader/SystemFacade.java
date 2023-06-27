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

package com.google.android.vending.expansion.downloader;

import android.app.Notification;
import android.app.NotificationManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager.NameNotFoundException;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.telephony.TelephonyManager;
import android.util.Log;

// -- GODOT start --
import android.annotation.SuppressLint;
// -- GODOT end --

/**
 * Contains useful helper functions, typically tied to the application context.
 */
class SystemFacade {
    private Context mContext;
    private NotificationManager mNotificationManager;

    public SystemFacade(Context context) {
        mContext = context;
        mNotificationManager = (NotificationManager)
                mContext.getSystemService(Context.NOTIFICATION_SERVICE);
    }

    public long currentTimeMillis() {
        return System.currentTimeMillis();
    }

    public Integer getActiveNetworkType() {
        ConnectivityManager connectivity =
                (ConnectivityManager) mContext.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (connectivity == null) {
            Log.w(Constants.TAG, "couldn't get connectivity manager");
            return null;
        }

        @SuppressLint("MissingPermission")
        NetworkInfo activeInfo = connectivity.getActiveNetworkInfo();
        if (activeInfo == null) {
            if (Constants.LOGVV) {
                Log.v(Constants.TAG, "network is not available");
            }
            return null;
        }
        return activeInfo.getType();
    }

    public boolean isNetworkRoaming() {
        ConnectivityManager connectivity =
                (ConnectivityManager) mContext.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (connectivity == null) {
            Log.w(Constants.TAG, "couldn't get connectivity manager");
            return false;
        }

        @SuppressLint("MissingPermission")
        NetworkInfo info = connectivity.getActiveNetworkInfo();
        boolean isMobile = (info != null && info.getType() == ConnectivityManager.TYPE_MOBILE);
        TelephonyManager tm = (TelephonyManager) mContext
                .getSystemService(Context.TELEPHONY_SERVICE);
        if (null == tm) {
            Log.w(Constants.TAG, "couldn't get telephony manager");
            return false;
        }
        boolean isRoaming = isMobile && tm.isNetworkRoaming();
        if (Constants.LOGVV && isRoaming) {
            Log.v(Constants.TAG, "network is roaming");
        }
        return isRoaming;
    }

    public Long getMaxBytesOverMobile() {
        return (long) Integer.MAX_VALUE;
    }

    public Long getRecommendedMaxBytesOverMobile() {
        return 2097152L;
    }

    public void sendBroadcast(Intent intent) {
        mContext.sendBroadcast(intent);
    }

    public boolean userOwnsPackage(int uid, String packageName) throws NameNotFoundException {
        return mContext.getPackageManager().getApplicationInfo(packageName, 0).uid == uid;
    }

    public void postNotification(long id, Notification notification) {
        /**
         * TODO: The system notification manager takes ints, not longs, as IDs,
         * but the download manager uses IDs take straight from the database,
         * which are longs. This will have to be dealt with at some point.
         */
        mNotificationManager.notify((int) id, notification);
    }

    public void cancelNotification(long id) {
        mNotificationManager.cancel((int) id);
    }

    public void cancelAllNotifications() {
        mNotificationManager.cancelAll();
    }

    public void startThread(Thread thread) {
        thread.start();
    }
}
