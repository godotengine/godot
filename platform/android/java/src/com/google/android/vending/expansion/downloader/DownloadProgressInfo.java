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

import android.os.Parcel;
import android.os.Parcelable;


/**
 * This class contains progress information about the active download(s).
 *
 * When you build the Activity that initiates a download and tracks the
 * progress by implementing the {@link IDownloaderClient} interface, you'll
 * receive a DownloadProgressInfo object in each call to the {@link
 * IDownloaderClient#onDownloadProgress} method. This allows you to update
 * your activity's UI with information about the download progress, such
 * as the progress so far, time remaining and current speed.
 */
public class DownloadProgressInfo implements Parcelable {
    public long mOverallTotal;
    public long mOverallProgress;
    public long mTimeRemaining; // time remaining
    public float mCurrentSpeed; // speed in KB/S

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel p, int i) {
        p.writeLong(mOverallTotal);
        p.writeLong(mOverallProgress);
        p.writeLong(mTimeRemaining);
        p.writeFloat(mCurrentSpeed);
    }

    public DownloadProgressInfo(Parcel p) {
        mOverallTotal = p.readLong();
        mOverallProgress = p.readLong();
        mTimeRemaining = p.readLong();
        mCurrentSpeed = p.readFloat();
    }

    public DownloadProgressInfo(long overallTotal, long overallProgress,
            long timeRemaining,
            float currentSpeed) {
        this.mOverallTotal = overallTotal;
        this.mOverallProgress = overallProgress;
        this.mTimeRemaining = timeRemaining;
        this.mCurrentSpeed = currentSpeed;
    }

    public static final Creator<DownloadProgressInfo> CREATOR = new Creator<DownloadProgressInfo>() {
        @Override
        public DownloadProgressInfo createFromParcel(Parcel parcel) {
            return new DownloadProgressInfo(parcel);
        }

        @Override
        public DownloadProgressInfo[] newArray(int i) {
            return new DownloadProgressInfo[i];
        }
    };

}
