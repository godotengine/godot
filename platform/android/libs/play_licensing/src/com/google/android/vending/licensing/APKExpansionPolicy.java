
package com.google.android.vending.licensing;

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

import org.apache.http.NameValuePair;
import org.apache.http.client.utils.URLEncodedUtils;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

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
public class APKExpansionPolicy implements Policy {

    private static final String TAG = "APKExpansionPolicy";
    private static final String PREFS_FILE = "com.android.vending.licensing.APKExpansionPolicy";
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
    private Vector<String> mExpansionURLs = new Vector<String>();
    private Vector<String> mExpansionFileNames = new Vector<String>();
    private Vector<Long> mExpansionFileSizes = new Vector<Long>();

    /**
     * The design of the protocol supports n files. Currently the market can
     * only deliver two files. To accommodate this, we have these two constants,
     * but the order is the only relevant thing here.
     */
    public static final int MAIN_FILE_URL_INDEX = 0;
    public static final int PATCH_FILE_URL_INDEX = 1;

    /**
     * @param context The context for the current application
     * @param obfuscator An obfuscator to be used with preferences.
     */
    public APKExpansionPolicy(Context context, Obfuscator obfuscator) {
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
     * We call this to guarantee that we fetch a fresh policy from the server.
     * This is to be used if the URL is invalid.
     */
    public void resetPolicy() {
        mPreferences.putString(PREF_LAST_RESPONSE, Integer.toString(Policy.RETRY));
        setRetryUntil(DEFAULT_RETRY_UNTIL);
        setMaxRetries(DEFAULT_MAX_RETRIES);
        setRetryCount(Long.parseLong(DEFAULT_RETRY_COUNT));
        setValidityTimestamp(DEFAULT_VALIDITY_TIMESTAMP);
        mPreferences.commit();
    }

    /**
     * Process a new response from the license server.
     * <p>
     * This data will be used for computing future policy decisions. The
     * following parameters are processed:
     * <ul>
     * <li>VT: the timestamp that the client should consider the response valid
     * until
     * <li>GT: the timestamp that the client should ignore retry errors until
     * <li>GR: the number of retry errors that the client should ignore
     * </ul>
     * 
     * @param response the result from validating the server response
     * @param rawData the raw server response data
     */
    public void processServerResponse(int response,
            com.google.android.vending.licensing.ResponseData rawData) {

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
            setValidityTimestamp(Long.toString(System.currentTimeMillis() + MILLIS_PER_MINUTE));
            Set<String> keys = extras.keySet();
            for (String key : keys) {
                if (key.equals("VT")) {
                    setValidityTimestamp(extras.get(key));
                } else if (key.equals("GT")) {
                    setRetryUntil(extras.get(key));
                } else if (key.equals("GR")) {
                    setMaxRetries(extras.get(key));
                } else if (key.startsWith("FILE_URL")) {
                    int index = Integer.parseInt(key.substring("FILE_URL".length())) - 1;
                    setExpansionURL(index, extras.get(key));
                } else if (key.startsWith("FILE_NAME")) {
                    int index = Integer.parseInt(key.substring("FILE_NAME".length())) - 1;
                    setExpansionFileName(index, extras.get(key));
                } else if (key.startsWith("FILE_SIZE")) {
                    int index = Integer.parseInt(key.substring("FILE_SIZE".length())) - 1;
                    setExpansionFileSize(index, Long.parseLong(extras.get(key)));
                }
            }
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
            // No response or not parseable, expire in one minute.
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
            // No response or not parseable, expire immediately
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
            // No response or not parseable, expire immediately
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
     * Gets the count of expansion URLs. Since expansionURLs are not committed
     * to preferences, this will return zero if there has been no LVL fetch
     * in the current session.
     * 
     * @return the number of expansion URLs. (0,1,2)
     */
    public int getExpansionURLCount() {
        return mExpansionURLs.size();
    }

    /**
     * Gets the expansion URL. Since these URLs are not committed to
     * preferences, this will always return null if there has not been an LVL
     * fetch in the current session.
     * 
     * @param index the index of the URL to fetch. This value will be either
     *            MAIN_FILE_URL_INDEX or PATCH_FILE_URL_INDEX
     * @param URL the URL to set
     */
    public String getExpansionURL(int index) {
        if (index < mExpansionURLs.size()) {
            return mExpansionURLs.elementAt(index);
        }
        return null;
    }

    /**
     * Sets the expansion URL. Expansion URL's are not committed to preferences,
     * but are instead intended to be stored when the license response is
     * processed by the front-end.
     * 
     * @param index the index of the expansion URL. This value will be either
     *            MAIN_FILE_URL_INDEX or PATCH_FILE_URL_INDEX
     * @param URL the URL to set
     */
    public void setExpansionURL(int index, String URL) {
        if (index >= mExpansionURLs.size()) {
            mExpansionURLs.setSize(index + 1);
        }
        mExpansionURLs.set(index, URL);
    }

    public String getExpansionFileName(int index) {
        if (index < mExpansionFileNames.size()) {
            return mExpansionFileNames.elementAt(index);
        }
        return null;
    }

    public void setExpansionFileName(int index, String name) {
        if (index >= mExpansionFileNames.size()) {
            mExpansionFileNames.setSize(index + 1);
        }
        mExpansionFileNames.set(index, name);
    }

    public long getExpansionFileSize(int index) {
        if (index < mExpansionFileSizes.size()) {
            return mExpansionFileSizes.elementAt(index);
        }
        return -1;
    }

    public void setExpansionFileSize(int index, long size) {
        if (index >= mExpansionFileSizes.size()) {
            mExpansionFileSizes.setSize(index + 1);
        }
        mExpansionFileSizes.set(index, size);
    }

    /**
     * {@inheritDoc} This implementation allows access if either:<br>
     * <ol>
     * <li>a LICENSED response was received within the validity period
     * <li>a RETRY response was received in the last minute, and we are under
     * the RETRY count or in the RETRY period.
     * </ol>
     */
    public boolean allowAccess() {
        long ts = System.currentTimeMillis();
        if (mLastResponse == Policy.LICENSED) {
            // Check if the LICENSED response occurred within the validity
            // timeout.
            if (ts <= mValidityTimestamp) {
                // Cached LICENSED response is still valid.
                return true;
            }
        } else if (mLastResponse == Policy.RETRY &&
                ts < mLastResponseTime + MILLIS_PER_MINUTE) {
            // Only allow access if we are within the retry period or we haven't
            // used up our
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
                String name = item.getName();
                int i = 0;
                while (results.containsKey(name)) {
                    name = item.getName() + ++i;
                }
                results.put(name, item.getValue());
            }
        } catch (URISyntaxException e) {
            Log.w(TAG, "Invalid syntax error while decoding extras data from server.");
        }
        return results;
    }

}
