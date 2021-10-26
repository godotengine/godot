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

import android.content.Context;
import android.os.Messenger;

/**
 * This is the interface that is used to connect/disconnect from the downloader
 * service.
 * <p>
 * You should get a proxy object that implements this interface by calling
 * {@link DownloaderClientMarshaller#CreateStub} in your activity when the
 * downloader service starts. Then, call {@link #connect} during your activity's
 * onResume() and call {@link #disconnect} during onStop().
 * <p>
 * Then during the {@link IDownloaderClient#onServiceConnected} callback, you
 * should call {@link #getMessenger} to pass the stub's Messenger object to
 * {@link IDownloaderService#onClientUpdated}.
 */
public interface IStub {
    Messenger getMessenger();

    void connect(Context c);

    void disconnect(Context c);
}
