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

import java.io.File;


/**
 * Contains the internal constants that are used in the download manager.
 * As a general rule, modifying these constants should be done with care.
 */
public class Constants {
    /** Tag used for debugging/logging */
    public static final String TAG = "LVLDL";

    /**
     * Expansion path where we store obb files
     */
    public static final String EXP_PATH = File.separator + "Android"
            + File.separator + "obb" + File.separator;

    /** The intent that gets sent when the service must wake up for a retry */
    public static final String ACTION_RETRY = "android.intent.action.DOWNLOAD_WAKEUP";

    /** the intent that gets sent when clicking a successful download */
    public static final String ACTION_OPEN = "android.intent.action.DOWNLOAD_OPEN";

    /** the intent that gets sent when clicking an incomplete/failed download  */
    public static final String ACTION_LIST = "android.intent.action.DOWNLOAD_LIST";

    /** the intent that gets sent when deleting the notification of a completed download */
    public static final String ACTION_HIDE = "android.intent.action.DOWNLOAD_HIDE";

    /**
     * When a number has to be appended to the filename, this string is used to separate the
     * base filename from the sequence number
     */
    public static final String FILENAME_SEQUENCE_SEPARATOR = "-";

    /** The default user agent used for downloads */
    public static final String DEFAULT_USER_AGENT = "Android.LVLDM";

    /** The buffer size used to stream the data */
    public static final int BUFFER_SIZE = 4096;

    /** The minimum amount of progress that has to be done before the progress bar gets updated */
    public static final int MIN_PROGRESS_STEP = 4096;

    /** The minimum amount of time that has to elapse before the progress bar gets updated, in ms */
    public static final long MIN_PROGRESS_TIME = 1000;

    /** The maximum number of rows in the database (FIFO) */
    public static final int MAX_DOWNLOADS = 1000;

    /**
     * The number of times that the download manager will retry its network
     * operations when no progress is happening before it gives up.
     */
    public static final int MAX_RETRIES = 5;

    /**
     * The minimum amount of time that the download manager accepts for
     * a Retry-After response header with a parameter in delta-seconds.
     */
    public static final int MIN_RETRY_AFTER = 30; // 30s

    /**
     * The maximum amount of time that the download manager accepts for
     * a Retry-After response header with a parameter in delta-seconds.
     */
    public static final int MAX_RETRY_AFTER = 24 * 60 * 60; // 24h

    /**
     * The maximum number of redirects.
     */
    public static final int MAX_REDIRECTS = 5; // can't be more than 7.

    /**
     * The time between a failure and the first retry after an IOException.
     * Each subsequent retry grows exponentially, doubling each time.
     * The time is in seconds.
     */
    public static final int RETRY_FIRST_DELAY = 30;

    /** Enable separate connectivity logging */
    public static final boolean LOGX = true;

    /** Enable verbose logging */
    public static final boolean LOGV = false;

    /** Enable super-verbose logging */
    private static final boolean LOCAL_LOGVV = false;
    public static final boolean LOGVV = LOCAL_LOGVV && LOGV;

    /**
     * This download has successfully completed.
     * Warning: there might be other status values that indicate success
     * in the future.
     * Use isSucccess() to capture the entire category.
     */
    public static final int STATUS_SUCCESS = 200;

    /**
     * This request couldn't be parsed. This is also used when processing
     * requests with unknown/unsupported URI schemes.
     */
    public static final int STATUS_BAD_REQUEST = 400;

    /**
     * This download can't be performed because the content type cannot be
     * handled.
     */
    public static final int STATUS_NOT_ACCEPTABLE = 406;

    /**
     * This download cannot be performed because the length cannot be
     * determined accurately. This is the code for the HTTP error "Length
     * Required", which is typically used when making requests that require
     * a content length but don't have one, and it is also used in the
     * client when a response is received whose length cannot be determined
     * accurately (therefore making it impossible to know when a download
     * completes).
     */
    public static final int STATUS_LENGTH_REQUIRED = 411;

    /**
     * This download was interrupted and cannot be resumed.
     * This is the code for the HTTP error "Precondition Failed", and it is
     * also used in situations where the client doesn't have an ETag at all.
     */
    public static final int STATUS_PRECONDITION_FAILED = 412;

    /**
     * The lowest-valued error status that is not an actual HTTP status code.
     */
    public static final int MIN_ARTIFICIAL_ERROR_STATUS = 488;

    /**
     * The requested destination file already exists.
     */
    public static final int STATUS_FILE_ALREADY_EXISTS_ERROR = 488;

    /**
     * Some possibly transient error occurred, but we can't resume the download.
     */
    public static final int STATUS_CANNOT_RESUME = 489;

    /**
     * This download was canceled
     */
    public static final int STATUS_CANCELED = 490;

    /**
     * This download has completed with an error.
     * Warning: there will be other status values that indicate errors in
     * the future. Use isStatusError() to capture the entire category.
     */
    public static final int STATUS_UNKNOWN_ERROR = 491;

    /**
     * This download couldn't be completed because of a storage issue.
     * Typically, that's because the filesystem is missing or full.
     * Use the more specific {@link #STATUS_INSUFFICIENT_SPACE_ERROR}
     * and {@link #STATUS_DEVICE_NOT_FOUND_ERROR} when appropriate.
     */
    public static final int STATUS_FILE_ERROR = 492;

    /**
     * This download couldn't be completed because of an HTTP
     * redirect response that the download manager couldn't
     * handle.
     */
    public static final int STATUS_UNHANDLED_REDIRECT = 493;

    /**
     * This download couldn't be completed because of an
     * unspecified unhandled HTTP code.
     */
    public static final int STATUS_UNHANDLED_HTTP_CODE = 494;

    /**
     * This download couldn't be completed because of an
     * error receiving or processing data at the HTTP level.
     */
    public static final int STATUS_HTTP_DATA_ERROR = 495;

    /**
     * This download couldn't be completed because of an
     * HttpException while setting up the request.
     */
    public static final int STATUS_HTTP_EXCEPTION = 496;

    /**
     * This download couldn't be completed because there were
     * too many redirects.
     */
    public static final int STATUS_TOO_MANY_REDIRECTS = 497;

    /**
     * This download couldn't be completed due to insufficient storage
     * space.  Typically, this is because the SD card is full.
     */
    public static final int STATUS_INSUFFICIENT_SPACE_ERROR = 498;

    /**
     * This download couldn't be completed because no external storage
     * device was found.  Typically, this is because the SD card is not
     * mounted.
     */
    public static final int STATUS_DEVICE_NOT_FOUND_ERROR = 499;

    /**
     * The wake duration to check to see if a download is possible.
     */
    public static final long WATCHDOG_WAKE_TIMER = 60*1000;

    /**
     * The wake duration to check to see if the process was killed.
     */
    public static final long ACTIVE_THREAD_WATCHDOG = 5*1000;

}
