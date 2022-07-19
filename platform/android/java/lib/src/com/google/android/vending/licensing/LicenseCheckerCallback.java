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
 * Callback for the license checker library.
 * <p>
 * Upon checking with the Market server and conferring with the {@link Policy},
 * the library calls the appropriate callback method to communicate the result.
 * <p>
 * <b>The callback does not occur in the original checking thread.</b> Your
 * application should post to the appropriate handling thread or lock
 * accordingly.
 * <p>
 * The reason that is passed back with allow/dontAllow is the base status handed
 * to the policy for allowed/disallowing the license. Policy.RETRY will call
 * allow or dontAllow depending on other statistics associated with the policy,
 * while in most cases Policy.NOT_LICENSED will call dontAllow and
 * Policy.LICENSED will Allow.
 */
public interface LicenseCheckerCallback {

    /**
     * Allow use. App should proceed as normal.
     *
     * @param reason Policy.LICENSED or Policy.RETRY typically. (although in
     *            theory the policy can return Policy.NOT_LICENSED here as well)
     */
    public void allow(int reason);

    /**
     * Don't allow use. App should inform user and take appropriate action.
     *
     * @param reason Policy.NOT_LICENSED or Policy.RETRY. (although in theory
     *            the policy can return Policy.LICENSED here as well ---
     *            perhaps the call to the LVL took too long, for example)
     */
    public void dontAllow(int reason);

    /** Application error codes. */
    public static final int ERROR_INVALID_PACKAGE_NAME = 1;
    public static final int ERROR_NON_MATCHING_UID = 2;
    public static final int ERROR_NOT_MARKET_MANAGED = 3;
    public static final int ERROR_CHECK_IN_PROGRESS = 4;
    public static final int ERROR_INVALID_PUBLIC_KEY = 5;
    public static final int ERROR_MISSING_PERMISSION = 6;

    /**
     * Error in application code. Caller did not call or set up license checker
     * correctly. Should be considered fatal.
     */
    public void applicationError(int errorCode);
}
