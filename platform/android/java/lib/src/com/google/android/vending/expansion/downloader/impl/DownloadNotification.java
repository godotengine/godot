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

// -- GODOT start --
//import com.android.vending.expansion.downloader.R;
import org.godotengine.godot.R;
// -- GODOT end --

import com.google.android.vending.expansion.downloader.DownloadProgressInfo;
import com.google.android.vending.expansion.downloader.DownloaderClientMarshaller;
import com.google.android.vending.expansion.downloader.Helpers;
import com.google.android.vending.expansion.downloader.IDownloaderClient;

import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.os.Messenger;

import androidx.core.app.NotificationCompat;

/**
 * This class handles displaying the notification associated with the download
 * queue going on in the download manager. It handles multiple status types;
 * Some require user interaction and some do not. Some of the user interactions
 * may be transient. (for example: the user is queried to continue the download
 * on 3G when it started on WiFi, but then the phone locks onto WiFi again so
 * the prompt automatically goes away)
 * <p/>
 * The application interface for the downloader also needs to understand and
 * handle these transient states.
 */
public class DownloadNotification implements IDownloaderClient {

    private int mState;
    private final Context mContext;
    private final NotificationManager mNotificationManager;
    private CharSequence mCurrentTitle;

    private IDownloaderClient mClientProxy;
    private NotificationCompat.Builder mActiveDownloadBuilder;
    private NotificationCompat.Builder mBuilder;
    private NotificationCompat.Builder mCurrentBuilder;
    private CharSequence mLabel;
    private String mCurrentText;
    private DownloadProgressInfo mProgressInfo;
    private PendingIntent mContentIntent;

    static final String LOGTAG = "DownloadNotification";
    static final int NOTIFICATION_ID = LOGTAG.hashCode();

    public PendingIntent getClientIntent() {
        return mContentIntent;
    }

    public void setClientIntent(PendingIntent clientIntent) {
        this.mBuilder.setContentIntent(clientIntent);
        this.mActiveDownloadBuilder.setContentIntent(clientIntent);
        this.mContentIntent = clientIntent;
    }

    public void resendState() {
        if (null != mClientProxy) {
            mClientProxy.onDownloadStateChanged(mState);
        }
    }

    @Override
    public void onDownloadStateChanged(int newState) {
        if (null != mClientProxy) {
            mClientProxy.onDownloadStateChanged(newState);
        }
        if (newState != mState) {
            mState = newState;
            if (newState == IDownloaderClient.STATE_IDLE || null == mContentIntent) {
                return;
            }
            int stringDownloadID;
            int iconResource;
            boolean ongoingEvent;

            // get the new title string and paused text
            switch (newState) {
                case 0:
                    iconResource = android.R.drawable.stat_sys_warning;
                    stringDownloadID = R.string.state_unknown;
                    ongoingEvent = false;
                    break;

                case IDownloaderClient.STATE_DOWNLOADING:
                    iconResource = android.R.drawable.stat_sys_download;
                    stringDownloadID = Helpers.getDownloaderStringResourceIDFromState(newState);
                    ongoingEvent = true;
                    break;

                case IDownloaderClient.STATE_FETCHING_URL:
                case IDownloaderClient.STATE_CONNECTING:
                    iconResource = android.R.drawable.stat_sys_download_done;
                    stringDownloadID = Helpers.getDownloaderStringResourceIDFromState(newState);
                    ongoingEvent = true;
                    break;

                case IDownloaderClient.STATE_COMPLETED:
                case IDownloaderClient.STATE_PAUSED_BY_REQUEST:
                    iconResource = android.R.drawable.stat_sys_download_done;
                    stringDownloadID = Helpers.getDownloaderStringResourceIDFromState(newState);
                    ongoingEvent = false;
                    break;

                case IDownloaderClient.STATE_FAILED:
                case IDownloaderClient.STATE_FAILED_CANCELED:
                case IDownloaderClient.STATE_FAILED_FETCHING_URL:
                case IDownloaderClient.STATE_FAILED_SDCARD_FULL:
                case IDownloaderClient.STATE_FAILED_UNLICENSED:
                    iconResource = android.R.drawable.stat_sys_warning;
                    stringDownloadID = Helpers.getDownloaderStringResourceIDFromState(newState);
                    ongoingEvent = false;
                    break;

                default:
                    iconResource = android.R.drawable.stat_sys_warning;
                    stringDownloadID = Helpers.getDownloaderStringResourceIDFromState(newState);
                    ongoingEvent = true;
                    break;
            }

            mCurrentText = mContext.getString(stringDownloadID);
            mCurrentTitle = mLabel;
            mCurrentBuilder.setTicker(mLabel + ": " + mCurrentText);
            mCurrentBuilder.setSmallIcon(iconResource);
            mCurrentBuilder.setContentTitle(mCurrentTitle);
            mCurrentBuilder.setContentText(mCurrentText);
            if (ongoingEvent) {
                mCurrentBuilder.setOngoing(true);
            } else {
                mCurrentBuilder.setOngoing(false);
                mCurrentBuilder.setAutoCancel(true);
            }
            mNotificationManager.notify(NOTIFICATION_ID, mCurrentBuilder.build());
        }
    }

    @Override
    public void onDownloadProgress(DownloadProgressInfo progress) {
        mProgressInfo = progress;
        if (null != mClientProxy) {
            mClientProxy.onDownloadProgress(progress);
        }
        if (progress.mOverallTotal <= 0) {
            // we just show the text
            mBuilder.setTicker(mCurrentTitle);
            mBuilder.setSmallIcon(android.R.drawable.stat_sys_download);
            mBuilder.setContentTitle(mCurrentTitle);
            mBuilder.setContentText(mCurrentText);
            mCurrentBuilder = mBuilder;
        } else {
            mActiveDownloadBuilder.setProgress((int) progress.mOverallTotal, (int) progress.mOverallProgress, false);
            mActiveDownloadBuilder.setContentText(Helpers.getDownloadProgressString(progress.mOverallProgress, progress.mOverallTotal));
            mActiveDownloadBuilder.setSmallIcon(android.R.drawable.stat_sys_download);
            mActiveDownloadBuilder.setTicker(mLabel + ": " + mCurrentText);
            mActiveDownloadBuilder.setContentTitle(mLabel);
            mActiveDownloadBuilder.setContentInfo(mContext.getString(R.string.time_remaining_notification,
                    Helpers.getTimeRemaining(progress.mTimeRemaining)));
            mCurrentBuilder = mActiveDownloadBuilder;
        }
        mNotificationManager.notify(NOTIFICATION_ID, mCurrentBuilder.build());
    }

    /**
     * Called in response to onClientUpdated. Creates a new proxy and notifies
     * it of the current state.
     *
     * @param msg the client Messenger to notify
     */
    public void setMessenger(Messenger msg) {
        mClientProxy = DownloaderClientMarshaller.CreateProxy(msg);
        if (null != mProgressInfo) {
            mClientProxy.onDownloadProgress(mProgressInfo);
        }
        if (mState != -1) {
            mClientProxy.onDownloadStateChanged(mState);
        }
    }

    /**
     * Constructor
     *
     * @param ctx The context to use to obtain access to the Notification
     *            Service
     */
    DownloadNotification(Context ctx, CharSequence applicationLabel) {
        mState = -1;
        mContext = ctx;
        mLabel = applicationLabel;
        mNotificationManager = (NotificationManager)
                mContext.getSystemService(Context.NOTIFICATION_SERVICE);
        mActiveDownloadBuilder = new NotificationCompat.Builder(ctx);
        mBuilder = new NotificationCompat.Builder(ctx);

        // Set Notification category and priorities to something that makes sense for a long
        // lived background task.
        mActiveDownloadBuilder.setPriority(NotificationCompat.PRIORITY_LOW);
        mActiveDownloadBuilder.setCategory(NotificationCompat.CATEGORY_PROGRESS);

        mBuilder.setPriority(NotificationCompat.PRIORITY_LOW);
        mBuilder.setCategory(NotificationCompat.CATEGORY_PROGRESS);

        mCurrentBuilder = mBuilder;
    }

    @Override
    public void onServiceConnected(Messenger m) {
    }

}
