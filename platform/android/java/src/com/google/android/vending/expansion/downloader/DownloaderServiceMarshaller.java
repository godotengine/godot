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

import com.google.android.vending.expansion.downloader.impl.DownloaderService;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;



/**
 * This class is used by the client activity to proxy requests to the Downloader
 * Service.
 *
 * Most importantly, you must call {@link #CreateProxy} during the {@link
 * IDownloaderClient#onServiceConnected} callback in your activity in order to instantiate
 * an {@link IDownloaderService} object that you can then use to issue commands to the {@link
 * DownloaderService} (such as to pause and resume downloads).
 */
public class DownloaderServiceMarshaller {

    public static final int MSG_REQUEST_ABORT_DOWNLOAD =
            1;
    public static final int MSG_REQUEST_PAUSE_DOWNLOAD =
            2;
    public static final int MSG_SET_DOWNLOAD_FLAGS =
            3;
    public static final int MSG_REQUEST_CONTINUE_DOWNLOAD =
            4;
    public static final int MSG_REQUEST_DOWNLOAD_STATE =
            5;
    public static final int MSG_REQUEST_CLIENT_UPDATE =
            6;

    public static final String PARAMS_FLAGS = "flags";
    public static final String PARAM_MESSENGER = DownloaderService.EXTRA_MESSAGE_HANDLER;

    private static class Proxy implements IDownloaderService {
        private Messenger mMsg;

        private void send(int method, Bundle params) {
            Message m = Message.obtain(null, method);
            m.setData(params);
            try {
                mMsg.send(m);
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        }

        public Proxy(Messenger msg) {
            mMsg = msg;
        }

        @Override
        public void requestAbortDownload() {
            send(MSG_REQUEST_ABORT_DOWNLOAD, new Bundle());
        }

        @Override
        public void requestPauseDownload() {
            send(MSG_REQUEST_PAUSE_DOWNLOAD, new Bundle());
        }

        @Override
        public void setDownloadFlags(int flags) {
            Bundle params = new Bundle();
            params.putInt(PARAMS_FLAGS, flags);
            send(MSG_SET_DOWNLOAD_FLAGS, params);
        }

        @Override
        public void requestContinueDownload() {
            send(MSG_REQUEST_CONTINUE_DOWNLOAD, new Bundle());
        }

        @Override
        public void requestDownloadStatus() {
            send(MSG_REQUEST_DOWNLOAD_STATE, new Bundle());
        }

        @Override
        public void onClientUpdated(Messenger clientMessenger) {
            Bundle bundle = new Bundle(1);
            bundle.putParcelable(PARAM_MESSENGER, clientMessenger);
            send(MSG_REQUEST_CLIENT_UPDATE, bundle);
        }
    }

    private static class Stub implements IStub {
        private IDownloaderService mItf = null;
        final Messenger mMessenger = new Messenger(new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case MSG_REQUEST_ABORT_DOWNLOAD:
                        mItf.requestAbortDownload();
                        break;
                    case MSG_REQUEST_CONTINUE_DOWNLOAD:
                        mItf.requestContinueDownload();
                        break;
                    case MSG_REQUEST_PAUSE_DOWNLOAD:
                        mItf.requestPauseDownload();
                        break;
                    case MSG_SET_DOWNLOAD_FLAGS:
                        mItf.setDownloadFlags(msg.getData().getInt(PARAMS_FLAGS));
                        break;
                    case MSG_REQUEST_DOWNLOAD_STATE:
                        mItf.requestDownloadStatus();
                        break;
                    case MSG_REQUEST_CLIENT_UPDATE:
                        mItf.onClientUpdated((Messenger) msg.getData().getParcelable(
                                PARAM_MESSENGER));
                        break;
                }
            }
        });

        public Stub(IDownloaderService itf) {
            mItf = itf;
        }

        @Override
        public Messenger getMessenger() {
            return mMessenger;
        }

        @Override
        public void connect(Context c) {

        }

        @Override
        public void disconnect(Context c) {

        }
    }

    /**
     * Returns a proxy that will marshall calls to IDownloaderService methods
     * 
     * @param ctx
     * @return
     */
    public static IDownloaderService CreateProxy(Messenger msg) {
        return new Proxy(msg);
    }

    /**
     * Returns a stub object that, when connected, will listen for marshalled
     * IDownloaderService methods and translate them into calls to the supplied
     * interface.
     * 
     * @param itf An implementation of IDownloaderService that will be called
     *            when remote method calls are unmarshalled.
     * @return
     */
    public static IStub CreateStub(IDownloaderService itf) {
        return new Stub(itf);
    }

}
