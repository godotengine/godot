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

import android.app.PendingIntent;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager.NameNotFoundException;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.util.Log;



/**
 * This class binds the service API to your application client.  It contains the IDownloaderClient proxy,
 * which is used to call functions in your client as well as the Stub, which is used to call functions
 * in the client implementation of IDownloaderClient.
 * 
 * <p>The IPC is implemented using an Android Messenger and a service Binder.  The connect method
 * should be called whenever the client wants to bind to the service.  It opens up a service connection
 * that ends up calling the onServiceConnected client API that passes the service messenger
 * in.  If the client wants to be notified by the service, it is responsible for then passing its
 * messenger to the service in a separate call.
 *
 * <p>Critical methods are {@link #startDownloadServiceIfRequired} and {@link #CreateStub}.
 *
 * <p>When your application first starts, you should first check whether your app's expansion files are
 * already on the device. If not, you should then call {@link #startDownloadServiceIfRequired}, which
 * starts your {@link impl.DownloaderService} to download the expansion files if necessary. The method
 * returns a value indicating whether download is required or not.
 *
 * <p>If a download is required, {@link #startDownloadServiceIfRequired} begins the download through
 * the specified service and you should then call {@link #CreateStub} to instantiate a member {@link
 * IStub} object that you need in order to receive calls through your {@link IDownloaderClient}
 * interface.
 */
public class DownloaderClientMarshaller {
    public static final int MSG_ONDOWNLOADSTATE_CHANGED = 10;
    public static final int MSG_ONDOWNLOADPROGRESS = 11;
    public static final int MSG_ONSERVICECONNECTED = 12;

    public static final String PARAM_NEW_STATE = "newState";
    public static final String PARAM_PROGRESS = "progress";
    public static final String PARAM_MESSENGER = DownloaderService.EXTRA_MESSAGE_HANDLER;

    public static final int NO_DOWNLOAD_REQUIRED = DownloaderService.NO_DOWNLOAD_REQUIRED;
    public static final int LVL_CHECK_REQUIRED = DownloaderService.LVL_CHECK_REQUIRED;
    public static final int DOWNLOAD_REQUIRED = DownloaderService.DOWNLOAD_REQUIRED;

    private static class Proxy implements IDownloaderClient {
        private Messenger mServiceMessenger;

        @Override
        public void onDownloadStateChanged(int newState) {
            Bundle params = new Bundle(1);
            params.putInt(PARAM_NEW_STATE, newState);
            send(MSG_ONDOWNLOADSTATE_CHANGED, params);
        }

        @Override
        public void onDownloadProgress(DownloadProgressInfo progress) {
            Bundle params = new Bundle(1);
            params.putParcelable(PARAM_PROGRESS, progress);
            send(MSG_ONDOWNLOADPROGRESS, params);
        }

        private void send(int method, Bundle params) {
            Message m = Message.obtain(null, method);
            m.setData(params);
            try {
                mServiceMessenger.send(m);
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        }
        
        public Proxy(Messenger msg) {
            mServiceMessenger = msg;
        }

        @Override
        public void onServiceConnected(Messenger m) {
            /**
             * This is never called through the proxy.
             */
        }
    }

    private static class Stub implements IStub {
        private IDownloaderClient mItf = null;
        private Class<?> mDownloaderServiceClass;
        private boolean mBound;
        private Messenger mServiceMessenger;
        private Context mContext;
        /**
         * Target we publish for clients to send messages to IncomingHandler.
         */
        final Messenger mMessenger = new Messenger(new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case MSG_ONDOWNLOADPROGRESS:                        
                        Bundle bun = msg.getData();
                        if ( null != mContext ) {
                            bun.setClassLoader(mContext.getClassLoader());
                            DownloadProgressInfo dpi = (DownloadProgressInfo) msg.getData()
                                    .getParcelable(PARAM_PROGRESS);
                            mItf.onDownloadProgress(dpi);
                        }
                        break;
                    case MSG_ONDOWNLOADSTATE_CHANGED:
                        mItf.onDownloadStateChanged(msg.getData().getInt(PARAM_NEW_STATE));
                        break;
                    case MSG_ONSERVICECONNECTED:
                        mItf.onServiceConnected(
                                (Messenger) msg.getData().getParcelable(PARAM_MESSENGER));
                        break;
                }
            }
        });

        public Stub(IDownloaderClient itf, Class<?> downloaderService) {
            mItf = itf;
            mDownloaderServiceClass = downloaderService;
        }

        /**
         * Class for interacting with the main interface of the service.
         */
        private ServiceConnection mConnection = new ServiceConnection() {
            public void onServiceConnected(ComponentName className, IBinder service) {
                // This is called when the connection with the service has been
                // established, giving us the object we can use to
                // interact with the service. We are communicating with the
                // service using a Messenger, so here we get a client-side
                // representation of that from the raw IBinder object.
                mServiceMessenger = new Messenger(service);
                mItf.onServiceConnected(
                        mServiceMessenger);
            }

            public void onServiceDisconnected(ComponentName className) {
                // This is called when the connection with the service has been
                // unexpectedly disconnected -- that is, its process crashed.
                mServiceMessenger = null;
            }
        };

        @Override
        public void connect(Context c) {
            mContext = c;
            Intent bindIntent = new Intent(c, mDownloaderServiceClass);
            bindIntent.putExtra(PARAM_MESSENGER, mMessenger);
            if ( !c.bindService(bindIntent, mConnection, Context.BIND_DEBUG_UNBIND) ) {
                if ( Constants.LOGVV ) {
                    Log.d(Constants.TAG, "Service Unbound");
                }
            } else {
                mBound = true;
            }
                
        }

        @Override
        public void disconnect(Context c) {
            if (mBound) {
                c.unbindService(mConnection);
                mBound = false;
            }
            mContext = null;
        }

        @Override
        public Messenger getMessenger() {
            return mMessenger;
        }
    }

    /**
     * Returns a proxy that will marshal calls to IDownloaderClient methods
     * 
     * @param msg
     * @return
     */
    public static IDownloaderClient CreateProxy(Messenger msg) {
        return new Proxy(msg);
    }

    /**
     * Returns a stub object that, when connected, will listen for marshaled
     * {@link IDownloaderClient} methods and translate them into calls to the supplied
     * interface.
     * 
     * @param itf An implementation of IDownloaderClient that will be called
     *            when remote method calls are unmarshaled.
     * @param downloaderService The class for your implementation of {@link
     * impl.DownloaderService}.
     * @return The {@link IStub} that allows you to connect to the service such that
     * your {@link IDownloaderClient} receives status updates.
     */
    public static IStub CreateStub(IDownloaderClient itf, Class<?> downloaderService) {
        return new Stub(itf, downloaderService);
    }
    
    /**
     * Starts the download if necessary. This function starts a flow that does `
     * many things. 1) Checks to see if the APK version has been checked and
     * the metadata database updated 2) If the APK version does not match,
     * checks the new LVL status to see if a new download is required 3) If the
     * APK version does match, then checks to see if the download(s) have been
     * completed 4) If the downloads have been completed, returns
     * NO_DOWNLOAD_REQUIRED The idea is that this can be called during the
     * startup of an application to quickly ascertain if the application needs
     * to wait to hear about any updated APK expansion files. Note that this does
     * mean that the application MUST be run for the first time with a network
     * connection, even if Market delivers all of the files.
     * 
     * @param context Your application Context.
     * @param notificationClient A PendingIntent to start the Activity in your application
     * that shows the download progress and which will also start the application when download
     * completes.
     * @param serviceClass the class of your {@link imp.DownloaderService} implementation
     * @return whether the service was started and the reason for starting the service.
     * Either {@link #NO_DOWNLOAD_REQUIRED}, {@link #LVL_CHECK_REQUIRED}, or {@link
     * #DOWNLOAD_REQUIRED}.
     * @throws NameNotFoundException
     */
    public static int startDownloadServiceIfRequired(Context context, PendingIntent notificationClient, 
            Class<?> serviceClass)
            throws NameNotFoundException {
        return DownloaderService.startDownloadServiceIfRequired(context, notificationClient,
                serviceClass);
    }
    
    /**
     * This version assumes that the intent contains the pending intent as a parameter. This
     * is used for responding to alarms.
     * <p>The pending intent must be in an extra with the key {@link 
     * impl.DownloaderService#EXTRA_PENDING_INTENT}.
     * 
     * @param context
     * @param notificationClient
     * @param serviceClass the class of the service to start
     * @return
     * @throws NameNotFoundException
     */
    public static int startDownloadServiceIfRequired(Context context, Intent notificationClient, 
            Class<?> serviceClass)
            throws NameNotFoundException {
        return DownloaderService.startDownloadServiceIfRequired(context, notificationClient,
                serviceClass);
    }    

}
