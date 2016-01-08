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

/**
 * Policy used by {@link LicenseChecker} to determine whether a user should have
 * access to the application.
 */
public interface Policy {

    /**
     * Change these values to make it more difficult for tools to automatically
     * strip LVL protection from your APK.
     */

    /**
     * LICENSED means that the server returned back a valid license response
     */
    public static final int LICENSED = 0x0100;
    /**
     * NOT_LICENSED means that the server returned back a valid license response
     * that indicated that the user definitively is not licensed
     */
    public static final int NOT_LICENSED = 0x0231;
    /**
     * RETRY means that the license response was unable to be determined ---
     * perhaps as a result of faulty networking
     */
    public static final int RETRY = 0x0123;

    /**
     * Provide results from contact with the license server. Retry counts are
     * incremented if the current value of response is RETRY. Results will be
     * used for any future policy decisions.
     * 
     * @param response the result from validating the server response
     * @param rawData the raw server response data, can be null for RETRY
     */
    void processServerResponse(int response, ResponseData rawData);

    /**
     * Check if the user should be allowed access to the application.
     */
    boolean allowAccess();
}
