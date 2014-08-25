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

import com.android.vending.expansion.downloader.R;
import com.google.android.vending.expansion.downloader.Helpers;

import android.app.Notification;
import android.app.PendingIntent;
import android.content.Context;
import android.graphics.BitmapFactory;
import android.view.View;
import android.widget.RemoteViews;

public class V3CustomNotification implements DownloadNotification.ICustomNotification {

    CharSequence mTitle;
    CharSequence mTicker;
    int mIcon;
    long mTotalBytes = -1;
    long mCurrentBytes = -1;
    long mTimeRemaining;
    PendingIntent mPendingIntent;
    Notification mNotification = new Notification();

    @Override
    public void setIcon(int icon) {
        mIcon = icon;
    }

    @Override
    public void setTitle(CharSequence title) {
        mTitle = title;
    }

    @Override
    public void setTotalBytes(long totalBytes) {
        mTotalBytes = totalBytes;
    }

    @Override
    public void setCurrentBytes(long currentBytes) {
        mCurrentBytes = currentBytes;
    }

    @Override
    public Notification updateNotification(Context c) {
        Notification n = mNotification;

        n.icon = mIcon;

        n.flags |= Notification.FLAG_ONGOING_EVENT;

        if (android.os.Build.VERSION.SDK_INT > 10) {
            n.flags |= Notification.FLAG_ONLY_ALERT_ONCE; // only matters for
                                                          // Honeycomb
        }

        // Build the RemoteView object
        RemoteViews expandedView = new RemoteViews(
                c.getPackageName(),
                R.layout.status_bar_ongoing_event_progress_bar);

        expandedView.setTextViewText(R.id.title, mTitle);
        // look at strings
        expandedView.setViewVisibility(R.id.description, View.VISIBLE);
        expandedView.setTextViewText(R.id.description,
                Helpers.getDownloadProgressString(mCurrentBytes, mTotalBytes));
        expandedView.setViewVisibility(R.id.progress_bar_frame, View.VISIBLE);
        expandedView.setProgressBar(R.id.progress_bar,
                (int) (mTotalBytes >> 8),
                (int) (mCurrentBytes >> 8),
                mTotalBytes <= 0);
        expandedView.setViewVisibility(R.id.time_remaining, View.VISIBLE);
        expandedView.setTextViewText(
                R.id.time_remaining,
                c.getString(R.string.time_remaining_notification,
                        Helpers.getTimeRemaining(mTimeRemaining)));
        expandedView.setTextViewText(R.id.progress_text,
                Helpers.getDownloadProgressPercent(mCurrentBytes, mTotalBytes));
        expandedView.setImageViewResource(R.id.appIcon, mIcon);
        n.contentView = expandedView;
        n.contentIntent = mPendingIntent;
        return n;
    }

    @Override
    public void setPendingIntent(PendingIntent contentIntent) {
        mPendingIntent = contentIntent;
    }

    @Override
    public void setTicker(CharSequence ticker) {
        mTicker = ticker;
    }

    @Override
    public void setTimeRemaining(long timeRemaining) {
        mTimeRemaining = timeRemaining;
    }

}
