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
import com.google.android.vending.expansion.downloader.DownloadProgressInfo;
import com.google.android.vending.expansion.downloader.DownloaderClientMarshaller;
import com.google.android.vending.expansion.downloader.Helpers;
import com.google.android.vending.expansion.downloader.IDownloaderClient;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.os.Messenger;

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
    private String mCurrentTitle;

    private IDownloaderClient mClientProxy;
    final ICustomNotification mCustomNotification;
    private Notification mNotification;
    private Notification mCurrentNotification;
    private CharSequence mLabel;
    private String mCurrentText;
    private PendingIntent mContentIntent;
    private DownloadProgressInfo mProgressInfo;

    static final String LOGTAG = "DownloadNotification";
    static final int NOTIFICATION_ID = LOGTAG.hashCode();

    public PendingIntent getClientIntent() {
        return mContentIntent;
    }

    public void setClientIntent(PendingIntent mClientIntent) {
        this.mContentIntent = mClientIntent;
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
            mCurrentTitle = mLabel.toString();
            mCurrentNotification.tickerText = mLabel + ": " + mCurrentText;
            mCurrentNotification.icon = iconResource;
            mCurrentNotification.setLatestEventInfo(mContext, mCurrentTitle, mCurrentText,
                    mContentIntent);
            if (ongoingEvent) {
                mCurrentNotification.flags |= Notification.FLAG_ONGOING_EVENT;
            } else {
                mCurrentNotification.flags &= ~Notification.FLAG_ONGOING_EVENT;
                mCurrentNotification.flags |= Notification.FLAG_AUTO_CANCEL;
            }
            mNotificationManager.notify(NOTIFICATION_ID, mCurrentNotification);
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
            mNotification.tickerText = mCurrentTitle;
            mNotification.icon = android.R.drawable.stat_sys_download;
            mNotification.setLatestEventInfo(mContext, mLabel, mCurrentText, mContentIntent);
            mCurrentNotification = mNotification;
        } else {
            mCustomNotification.setCurrentBytes(progress.mOverallProgress);
            mCustomNotification.setTotalBytes(progress.mOverallTotal);
            mCustomNotification.setIcon(android.R.drawable.stat_sys_download);
            mCustomNotification.setPendingIntent(mContentIntent);
            mCustomNotification.setTicker(mLabel + ": " + mCurrentText);
            mCustomNotification.setTitle(mLabel);
            mCustomNotification.setTimeRemaining(progress.mTimeRemaining);
            mCurrentNotification = mCustomNotification.updateNotification(mContext);
        }
        mNotificationManager.notify(NOTIFICATION_ID, mCurrentNotification);
    }

    public interface ICustomNotification {
        void setTitle(CharSequence title);

        void setTicker(CharSequence ticker);

        void setPendingIntent(PendingIntent mContentIntent);

        void setTotalBytes(long totalBytes);

        void setCurrentBytes(long currentBytes);

        void setIcon(int iconResource);

        void setTimeRemaining(long timeRemaining);

        Notification updateNotification(Context c);
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
        mCustomNotification = CustomNotificationFactory
                .createCustomNotification();
        mNotification = new Notification();
        mCurrentNotification = mNotification;

    }

    @Override
    public void onServiceConnected(Messenger m) {
    }

}
