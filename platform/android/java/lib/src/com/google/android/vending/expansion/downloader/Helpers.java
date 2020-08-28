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

import android.annotation.TargetApi;
import android.content.Context;
import android.os.Build;
import android.os.Environment;
import android.os.StatFs;
import android.os.SystemClock;
import android.util.Log;

// -- GODOT start --
//import com.android.vending.expansion.downloader.R;
import org.godotengine.godot.R;
// -- GODOT end --

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Random;
import java.util.TimeZone;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Some helper functions for the download manager
 */
public class Helpers {

    public static Random sRandom = new Random(SystemClock.uptimeMillis());

    /** Regex used to parse content-disposition headers */
    private static final Pattern CONTENT_DISPOSITION_PATTERN = Pattern
            .compile("attachment;\\s*filename\\s*=\\s*\"([^\"]*)\"");

    private Helpers() {
    }

    /*
     * Parse the Content-Disposition HTTP Header. The format of the header is defined here:
     * http://www.w3.org/Protocols/rfc2616/rfc2616-sec19.html This header provides a filename for
     * content that is going to be downloaded to the file system. We only support the attachment
     * type.
     */
    static String parseContentDisposition(String contentDisposition) {
        try {
            Matcher m = CONTENT_DISPOSITION_PATTERN.matcher(contentDisposition);
            if (m.find()) {
                return m.group(1);
            }
        } catch (IllegalStateException ex) {
            // This function is defined as returning null when it can't parse
            // the header
        }
        return null;
    }

    /**
     * @return the root of the filesystem containing the given path
     */
    public static File getFilesystemRoot(String path) {
        File cache = Environment.getDownloadCacheDirectory();
        if (path.startsWith(cache.getPath())) {
            return cache;
        }
        File external = Environment.getExternalStorageDirectory();
        if (path.startsWith(external.getPath())) {
            return external;
        }
        throw new IllegalArgumentException(
                "Cannot determine filesystem root for " + path);
    }

    public static boolean isExternalMediaMounted() {
        if (!Environment.getExternalStorageState().equals(
                Environment.MEDIA_MOUNTED)) {
            // No SD card found.
            if (Constants.LOGVV) {
                Log.d(Constants.TAG, "no external storage");
            }
            return false;
        }
        return true;
    }

    /**
     * @return the number of bytes available on the filesystem rooted at the given File
     */
    public static long getAvailableBytes(File root) {
        StatFs stat = new StatFs(root.getPath());
        // put a bit of margin (in case creating the file grows the system by a
        // few blocks)
        long availableBlocks = (long) stat.getAvailableBlocks() - 4;
        return stat.getBlockSize() * availableBlocks;
    }

    /**
     * Checks whether the filename looks legitimate
     */
    public static boolean isFilenameValid(String filename) {
        filename = filename.replaceFirst("/+", "/"); // normalize leading
                                                     // slashes
        return filename.startsWith(Environment.getDownloadCacheDirectory().toString())
                || filename.startsWith(Environment.getExternalStorageDirectory().toString());
    }

    /*
     * Delete the given file from device
     */
    /* package */static void deleteFile(String path) {
        try {
            File file = new File(path);
            file.delete();
        } catch (Exception e) {
            Log.w(Constants.TAG, "file: '" + path + "' couldn't be deleted", e);
        }
    }

    /**
     * Showing progress in MB here. It would be nice to choose the unit (KB, MB, GB) based on total
     * file size, but given what we know about the expected ranges of file sizes for APK expansion
     * files, it's probably not necessary.
     *
     * @param overallProgress
     * @param overallTotal
     * @return
     */

    static public String getDownloadProgressString(long overallProgress, long overallTotal) {
        if (overallTotal == 0) {
            if (Constants.LOGVV) {
                Log.e(Constants.TAG, "Notification called when total is zero");
            }
            return "";
        }
        // -- GODOT start --
        return String.format(Locale.ENGLISH, "%.2f",
                (float) overallProgress / (1024.0f * 1024.0f))
                + "MB /" +
                String.format(Locale.ENGLISH, "%.2f", (float) overallTotal /
                        (1024.0f * 1024.0f))
                + "MB";
        // -- GODOT end --
    }

    /**
     * Adds a percentile to getDownloadProgressString.
     *
     * @param overallProgress
     * @param overallTotal
     * @return
     */
    static public String getDownloadProgressStringNotification(long overallProgress,
            long overallTotal) {
        if (overallTotal == 0) {
            if (Constants.LOGVV) {
                Log.e(Constants.TAG, "Notification called when total is zero");
            }
            return "";
        }
        return getDownloadProgressString(overallProgress, overallTotal) + " (" +
                getDownloadProgressPercent(overallProgress, overallTotal) + ")";
    }

    public static String getDownloadProgressPercent(long overallProgress, long overallTotal) {
        if (overallTotal == 0) {
            if (Constants.LOGVV) {
                Log.e(Constants.TAG, "Notification called when total is zero");
            }
            return "";
        }
        return Long.toString(overallProgress * 100 / overallTotal) + "%";
    }

    public static String getSpeedString(float bytesPerMillisecond) {
        // -- GODOT start --
        return String.format(Locale.ENGLISH, "%.2f", bytesPerMillisecond * 1000 / 1024);
        // -- GODOT end --
    }

    public static String getTimeRemaining(long durationInMilliseconds) {
        SimpleDateFormat sdf;
        if (durationInMilliseconds > 1000 * 60 * 60) {
            sdf = new SimpleDateFormat("HH:mm", Locale.getDefault());
        } else {
            sdf = new SimpleDateFormat("mm:ss", Locale.getDefault());
        }
        return sdf.format(new Date(durationInMilliseconds - TimeZone.getDefault().getRawOffset()));
    }

    /**
     * Returns the file name (without full path) for an Expansion APK file from the given context.
     *
     * @param c the context
     * @param mainFile true for main file, false for patch file
     * @param versionCode the version of the file
     * @return String the file name of the expansion file
     */
    public static String getExpansionAPKFileName(Context c, boolean mainFile, int versionCode) {
        return (mainFile ? "main." : "patch.") + versionCode + "." + c.getPackageName() + ".obb";
    }

    /**
     * Returns the filename (where the file should be saved) from info about a download
     */
    static public String generateSaveFileName(Context c, String fileName) {
        String path = getSaveFilePath(c)
                + File.separator + fileName;
        return path;
    }

    @TargetApi(Build.VERSION_CODES.HONEYCOMB)
    static public String getSaveFilePath(Context c) {
        // This technically existed since Honeycomb, but it is critical
        // on KitKat and greater versions since it will create the
        // directory if needed
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            return c.getObbDir().toString();
        } else {
            File root = Environment.getExternalStorageDirectory();
            String path = root.toString() + Constants.EXP_PATH + c.getPackageName();
            return path;
        }
    }

    /**
     * Helper function to ascertain the existence of a file and return true/false appropriately
     *
     * @param c the app/activity/service context
     * @param fileName the name (sans path) of the file to query
     * @param fileSize the size that the file must match
     * @param deleteFileOnMismatch if the file sizes do not match, delete the file
     * @return true if it does exist, false otherwise
     */
    static public boolean doesFileExist(Context c, String fileName, long fileSize,
            boolean deleteFileOnMismatch) {
        // the file may have been delivered by Play --- let's make sure
        // it's the size we expect
        File fileForNewFile = new File(Helpers.generateSaveFileName(c, fileName));
        if (fileForNewFile.exists()) {
            if (fileForNewFile.length() == fileSize) {
                return true;
            }
            if (deleteFileOnMismatch) {
                // delete the file --- we won't be able to resume
                // because we cannot confirm the integrity of the file
                fileForNewFile.delete();
            }
        }
        return false;
    }

    public static final int FS_READABLE = 0;
    public static final int FS_DOES_NOT_EXIST = 1;
    public static final int FS_CANNOT_READ = 2;

    /**
     * Helper function to ascertain whether a file can be read.
     *
     * @param c the app/activity/service context
     * @param fileName the name (sans path) of the file to query
     * @return true if it does exist, false otherwise
     */
    static public int getFileStatus(Context c, String fileName) {
        // the file may have been delivered by Play --- let's make sure
        // it's the size we expect
        File fileForNewFile = new File(Helpers.generateSaveFileName(c, fileName));
        int returnValue;
        if (fileForNewFile.exists()) {
            if (fileForNewFile.canRead()) {
                returnValue = FS_READABLE;
            } else {
                returnValue = FS_CANNOT_READ;
            }
        } else {
            returnValue = FS_DOES_NOT_EXIST;
        }
        return returnValue;
    }

    /**
     * Helper function to ascertain whether the application has the correct access to the OBB
     * directory to allow an OBB file to be written.
     * 
     * @param c the app/activity/service context
     * @return true if the application can write an OBB file, false otherwise
     */
    static public boolean canWriteOBBFile(Context c) {
        String path = getSaveFilePath(c);
        File fileForNewFile = new File(path);
        boolean canWrite;
        if (fileForNewFile.exists()) {
            canWrite = fileForNewFile.isDirectory() && fileForNewFile.canWrite();
        } else {
            canWrite = fileForNewFile.mkdirs();
        }
        return canWrite;
    }

    /**
     * Converts download states that are returned by the
     * {@link IDownloaderClient#onDownloadStateChanged} callback into usable strings. This is useful
     * if using the state strings built into the library to display user messages.
     * 
     * @param state One of the STATE_* constants from {@link IDownloaderClient}.
     * @return string resource ID for the corresponding string.
     */
    static public int getDownloaderStringResourceIDFromState(int state) {
        switch (state) {
            case IDownloaderClient.STATE_IDLE:
                return R.string.state_idle;
            case IDownloaderClient.STATE_FETCHING_URL:
                return R.string.state_fetching_url;
            case IDownloaderClient.STATE_CONNECTING:
                return R.string.state_connecting;
            case IDownloaderClient.STATE_DOWNLOADING:
                return R.string.state_downloading;
            case IDownloaderClient.STATE_COMPLETED:
                return R.string.state_completed;
            case IDownloaderClient.STATE_PAUSED_NETWORK_UNAVAILABLE:
                return R.string.state_paused_network_unavailable;
            case IDownloaderClient.STATE_PAUSED_BY_REQUEST:
                return R.string.state_paused_by_request;
            case IDownloaderClient.STATE_PAUSED_WIFI_DISABLED_NEED_CELLULAR_PERMISSION:
                return R.string.state_paused_wifi_disabled;
            case IDownloaderClient.STATE_PAUSED_NEED_CELLULAR_PERMISSION:
                return R.string.state_paused_wifi_unavailable;
            case IDownloaderClient.STATE_PAUSED_WIFI_DISABLED:
                return R.string.state_paused_wifi_disabled;
            case IDownloaderClient.STATE_PAUSED_NEED_WIFI:
                return R.string.state_paused_wifi_unavailable;
            case IDownloaderClient.STATE_PAUSED_ROAMING:
                return R.string.state_paused_roaming;
            case IDownloaderClient.STATE_PAUSED_NETWORK_SETUP_FAILURE:
                return R.string.state_paused_network_setup_failure;
            case IDownloaderClient.STATE_PAUSED_SDCARD_UNAVAILABLE:
                return R.string.state_paused_sdcard_unavailable;
            case IDownloaderClient.STATE_FAILED_UNLICENSED:
                return R.string.state_failed_unlicensed;
            case IDownloaderClient.STATE_FAILED_FETCHING_URL:
                return R.string.state_failed_fetching_url;
            case IDownloaderClient.STATE_FAILED_SDCARD_FULL:
                return R.string.state_failed_sdcard_full;
            case IDownloaderClient.STATE_FAILED_CANCELED:
                return R.string.state_failed_cancelled;
            default:
                return R.string.state_unknown;
        }
    }

}
