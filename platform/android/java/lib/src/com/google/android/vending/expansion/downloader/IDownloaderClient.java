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

import android.os.Messenger;

/**
 * This interface should be implemented by the client activity for the
 * downloader. It is used to pass status from the service to the client.
 */
public interface IDownloaderClient {
    static final int STATE_IDLE = 1;
    static final int STATE_FETCHING_URL = 2;
    static final int STATE_CONNECTING = 3;
    static final int STATE_DOWNLOADING = 4;
    static final int STATE_COMPLETED = 5;

    static final int STATE_PAUSED_NETWORK_UNAVAILABLE = 6;
    static final int STATE_PAUSED_BY_REQUEST = 7;

    /**
     * Both STATE_PAUSED_WIFI_DISABLED_NEED_CELLULAR_PERMISSION and
     * STATE_PAUSED_NEED_CELLULAR_PERMISSION imply that Wi-Fi is unavailable and
     * cellular permission will restart the service. Wi-Fi disabled means that
     * the Wi-Fi manager is returning that Wi-Fi is not enabled, while in the
     * other case Wi-Fi is enabled but not available.
     */
    static final int STATE_PAUSED_WIFI_DISABLED_NEED_CELLULAR_PERMISSION = 8;
    static final int STATE_PAUSED_NEED_CELLULAR_PERMISSION = 9;

    /**
     * Both STATE_PAUSED_WIFI_DISABLED and STATE_PAUSED_NEED_WIFI imply that
     * Wi-Fi is unavailable and cellular permission will NOT restart the
     * service. Wi-Fi disabled means that the Wi-Fi manager is returning that
     * Wi-Fi is not enabled, while in the other case Wi-Fi is enabled but not
     * available.
     * <p>
     * The service does not return these values. We recommend that app
     * developers with very large payloads do not allow these payloads to be
     * downloaded over cellular connections.
     */
    static final int STATE_PAUSED_WIFI_DISABLED = 10;
    static final int STATE_PAUSED_NEED_WIFI = 11;

    static final int STATE_PAUSED_ROAMING = 12;

    /**
     * Scary case. We were on a network that redirected us to another website
     * that delivered us the wrong file.
     */
    static final int STATE_PAUSED_NETWORK_SETUP_FAILURE = 13;

    static final int STATE_PAUSED_SDCARD_UNAVAILABLE = 14;

    static final int STATE_FAILED_UNLICENSED = 15;
    static final int STATE_FAILED_FETCHING_URL = 16;
    static final int STATE_FAILED_SDCARD_FULL = 17;
    static final int STATE_FAILED_CANCELED = 18;

    static final int STATE_FAILED = 19;

    /**
     * Called internally by the stub when the service is bound to the client.
     * <p>
     * Critical implementation detail. In onServiceConnected we create the
     * remote service and marshaler. This is how we pass the client information
     * back to the service so the client can be properly notified of changes. We
     * must do this every time we reconnect to the service.
     * <p>
     * That is, when you receive this callback, you should call
     * {@link DownloaderServiceMarshaller#CreateProxy} to instantiate a member
     * instance of {@link IDownloaderService}, then call
     * {@link IDownloaderService#onClientUpdated} with the Messenger retrieved
     * from your {@link IStub} proxy object.
     *
     * @param m the service Messenger. This Messenger is used to call the
     *            service API from the client.
     */
    void onServiceConnected(Messenger m);

    /**
     * Called when the download state changes. Depending on the state, there may
     * be user requests. The service is free to change the download state in the
     * middle of a user request, so the client should be able to handle this.
     * <p>
     * The Downloader Library includes a collection of string resources that
     * correspond to each of the states, which you can use to provide users a
     * useful message based on the state provided in this callback. To fetch the
     * appropriate string for a state, call
     * {@link Helpers#getDownloaderStringResourceIDFromState}.
     * <p>
     * What this means to the developer: The application has gotten a message
     * that the download has paused due to lack of WiFi. The developer should
     * then show UI asking the user if they want to enable downloading over
     * cellular connections with appropriate warnings. If the application
     * suddenly starts downloading, the application should revert to showing the
     * progress again, rather than leaving up the download over cellular UI up.
     *
     * @param newState one of the STATE_* values defined in IDownloaderClient
     */
    void onDownloadStateChanged(int newState);

    /**
     * Shows the download progress. This is intended to be used to fill out a
     * client UI. This progress should only be shown in a few states such as
     * STATE_DOWNLOADING.
     *
     * @param progress the DownloadProgressInfo object containing the current
     *            progress of all downloads.
     */
    void onDownloadProgress(DownloadProgressInfo progress);
}
