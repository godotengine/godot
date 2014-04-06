/*
 * Copyright (C) 2010 The Android Open Source Project
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

package com.google.android.vending.licensing;

import org.apache.http.NameValuePair;
import org.apache.http.client.utils.URLEncodedUtils;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * Default policy. All policy decisions are based off of response data received
 * from the licensing service. Specifically, the licensing server sends the
 * following information: response validity period, error retry period, and
 * error retry count.
 * <p>
 * These values will vary based on the the way the application is configured in
 * the Android Market publishing console, such as whether the application is
 * marked as free or is within its refund period, as well as how often an
 * application is checking with the licensing service.
 * <p>
 * Developers who need more fine grained control over their application's
 * licensing policy should implement a custom Policy.
 */
public class ServerManagedPolicy implements Policy {

    private static final String TAG = "ServerManagedPolicy";
    private static final String PREFS_FILE = "com.android.vending.licensing.ServerManagedPolicy";
    private static final String PREF_LAST_RESPONSE = "lastResponse";
    private static final String PREF_VALIDITY_TIMESTAMP = "validityTimestamp";
    private static final String PREF_RETRY_UNTIL = "retryUntil";
    private static final String PREF_MAX_RETRIES = "maxRetries";
    private static final String PREF_RETRY_COUNT = "retryCount";
    private static final String DEFAULT_VALIDITY_TIMESTAMP = "0";
    private static final String DEFAULT_RETRY_UNTIL = "0";
    private static final String DEFAULT_MAX_RETRIES = "0";
    private static final String DEFAULT_RETRY_COUNT = "0";

    private static final long MILLIS_PER_MINUTE = 60 * 1000;

    private long mValidityTimestamp;
    private long mRetryUntil;
    private long mMaxRetries;
    private long mRetryCount;
    private long mLastResponseTime = 0;
    private int mLastResponse;
    private PreferenceObfuscator mPreferences;

    /**
     * @param context The context for the current application
     * @param obfuscator An obfuscator to be used with preferences.
     */
    public ServerManagedPolicy(Context context, Obfuscator obfuscator) {
        // Import old values
        SharedPreferences sp = context.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE);
        mPreferences = new PreferenceObfuscator(sp, obfuscator);
        mLastResponse = Integer.parseInt(
            mPreferences.getString(PREF_LAST_RESPONSE, Integer.toString(Policy.RETRY)));
        mValidityTimestamp = Long.parseLong(mPreferences.getString(PREF_VALIDITY_TIMESTAMP,
                DEFAULT_VALIDITY_TIMESTAMP));
        mRetryUntil = Long.parseLong(mPreferences.getString(PREF_RETRY_UNTIL, DEFAULT_RETRY_UNTIL));
        mMaxRetries = Long.parseLong(mPreferences.getString(PREF_MAX_RETRIES, DEFAULT_MAX_RETRIES));
        mRetryCount = Long.parseLong(mPreferences.getString(PREF_RETRY_COUNT, DEFAULT_RETRY_COUNT));
    }

    /**
     * Process a new response from the license server.
     * <p>
     * This data will be used for computing future policy decisions. The
     * following parameters are processed:
     * <ul>
     * <li>VT: the timestamp that the client should consider the response
     *   valid until
     * <li>GT: the timestamp that the client should ignore retry errors until
     * <li>GR: the number of retry errors that the client should ignore
     * </ul>
     *
     * @param response the result from validating the server response
     * @param rawData the raw server response data
     */
    public void processServerResponse(int response, ResponseData rawData) {

        // Update retry counter
        if (response != Policy.RETRY) {
            setRetryCount(0);
        } else {
            setRetryCount(mRetryCount + 1);
        }

        if (response == Policy.LICENSED) {
            // Update server policy data
            Map<String, String> extras = decodeExtras(rawData.extra);
            mLastResponse = response;
            setValidityTimestamp(extras.get("VT"));
            setRetryUntil(extras.get("GT"));
            setMaxRetries(extras.get("GR"));
        } else if (response == Policy.NOT_LICENSED) {
            // Clear out stale policy data
            setValidityTimestamp(DEFAULT_VALIDITY_TIMESTAMP);
            setRetryUntil(DEFAULT_RETRY_UNTIL);
            setMaxRetries(DEFAULT_MAX_RETRIES);
        }

        setLastResponse(response);
        mPreferences.commit();
    }

    /**
     * Set the last license response received from the server and add to
     * preferences. You must manually call PreferenceObfuscator.commit() to
     * commit these changes to disk.
     *
     * @param l the response
     */
    private void setLastResponse(int l) {
        mLastResponseTime = System.currentTimeMillis();
        mLastResponse = l;
        mPreferences.putString(PREF_LAST_RESPONSE, Integer.toString(l));
    }

    /**
     * Set the current retry count and add to preferences. You must manually
     * call PreferenceObfuscator.commit() to commit these changes to disk.
     *
     * @param c the new retry count
     */
    private void setRetryCount(long c) {
        mRetryCount = c;
        mPreferences.putString(PREF_RETRY_COUNT, Long.toString(c));
    }

    public long getRetryCount() {
        return mRetryCount;
    }

    /**
     * Set the last validity timestamp (VT) received from the server and add to
     * preferences. You must manually call PreferenceObfuscator.commit() to
     * commit these changes to disk.
     *
     * @param validityTimestamp the VT string received
     */
    private void setValidityTimestamp(String validityTimestamp) {
        Long lValidityTimestamp;
        try {
            lValidityTimestamp = Long.parseLong(validityTimestamp);
        } catch (NumberFormatException e) {
            // No response or not parsable, expire in one minute.
            Log.w(TAG, "License validity timestamp (VT) missing, caching for a minute");
            lValidityTimestamp = System.currentTimeMillis() + MILLIS_PER_MINUTE;
            validityTimestamp = Long.toString(lValidityTimestamp);
        }

        mValidityTimestamp = lValidityTimestamp;
        mPreferences.putString(PREF_VALIDITY_TIMESTAMP, validityTimestamp);
    }

    public long getValidityTimestamp() {
        return mValidityTimestamp;
    }

    /**
     * Set the retry until timestamp (GT) received from the server and add to
     * preferences. You must manually call PreferenceObfuscator.commit() to
     * commit these changes to disk.
     *
     * @param retryUntil the GT string received
     */
    private void setRetryUntil(String retryUntil) {
        Long lRetryUntil;
        try {
            lRetryUntil = Long.parseLong(retryUntil);
        } catch (NumberFormatException e) {
            // No response or not parsable, expire immediately
            Log.w(TAG, "License retry timestamp (GT) missing, grace period disabled");
            retryUntil = "0";
            lRetryUntil = 0l;
        }

        mRetryUntil = lRetryUntil;
        mPreferences.putString(PREF_RETRY_UNTIL, retryUntil);
    }

    public long getRetryUntil() {
      return mRetryUntil;
    }

    /**
     * Set the max retries value (GR) as received from the server and add to
     * preferences. You must manually call PreferenceObfuscator.commit() to
     * commit these changes to disk.
     *
     * @param maxRetries the GR string received
     */
    private void setMaxRetries(String maxRetries) {
        Long lMaxRetries;
        try {
            lMaxRetries = Long.parseLong(maxRetries);
        } catch (NumberFormatException e) {
            // No response or not parsable, expire immediately
            Log.w(TAG, "Licence retry count (GR) missing, grace period disabled");
            maxRetries = "0";
            lMaxRetries = 0l;
        }

        mMaxRetries = lMaxRetries;
        mPreferences.putString(PREF_MAX_RETRIES, maxRetries);
    }

    public long getMaxRetries() {
        return mMaxRetries;
    }

    /**
     * {@inheritDoc}
     *
     * This implementation allows access if either:<br>
     * <ol>
     * <li>a LICENSED response was received within the validity period
     * <li>a RETRY response was received in the last minute, and we are under
     * the RETRY count or in the RETRY period.
     * </ol>
     */
    public boolean allowAccess() {
        long ts = System.currentTimeMillis();
        if (mLastResponse == Policy.LICENSED) {
            // Check if the LICENSED response occurred within the validity timeout.
            if (ts <= mValidityTimestamp) {
                // Cached LICENSED response is still valid.
                return true;
            }
        } else if (mLastResponse == Policy.RETRY &&
                   ts < mLastResponseTime + MILLIS_PER_MINUTE) {
            // Only allow access if we are within the retry period or we haven't used up our
            // max retries.
            return (ts <= mRetryUntil || mRetryCount <= mMaxRetries);
        }
        return false;
    }

    private Map<String, String> decodeExtras(String extras) {
        Map<String, String> results = new HashMap<String, String>();
        try {
            URI rawExtras = new URI("?" + extras);
            List<NameValuePair> extraList = URLEncodedUtils.parse(rawExtras, "UTF-8");
            for (NameValuePair item : extraList) {
                results.put(item.getName(), item.getValue());
            }
        } catch (URISyntaxException e) {
          Log.w(TAG, "Invalid syntax error while decoding extras data from server.");
        }
        return results;
    }

}
