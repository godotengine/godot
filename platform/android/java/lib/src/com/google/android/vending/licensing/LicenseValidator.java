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

import com.google.android.vending.licensing.util.Base64;
import com.google.android.vending.licensing.util.Base64DecoderException;

import android.text.TextUtils;
import android.util.Log;

import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;

/**
 * Contains data related to a licensing request and methods to verify
 * and process the response.
 */
class LicenseValidator {
    private static final String TAG = "LicenseValidator";

    // Server response codes.
    private static final int LICENSED = 0x0;
    private static final int NOT_LICENSED = 0x1;
    private static final int LICENSED_OLD_KEY = 0x2;
    private static final int ERROR_NOT_MARKET_MANAGED = 0x3;
    private static final int ERROR_SERVER_FAILURE = 0x4;
    private static final int ERROR_OVER_QUOTA = 0x5;

    private static final int ERROR_CONTACTING_SERVER = 0x101;
    private static final int ERROR_INVALID_PACKAGE_NAME = 0x102;
    private static final int ERROR_NON_MATCHING_UID = 0x103;

    private final Policy mPolicy;
    private final LicenseCheckerCallback mCallback;
    private final int mNonce;
    private final String mPackageName;
    private final String mVersionCode;
    private final DeviceLimiter mDeviceLimiter;

    LicenseValidator(Policy policy, DeviceLimiter deviceLimiter, LicenseCheckerCallback callback,
             int nonce, String packageName, String versionCode) {
        mPolicy = policy;
        mDeviceLimiter = deviceLimiter;
        mCallback = callback;
        mNonce = nonce;
        mPackageName = packageName;
        mVersionCode = versionCode;
    }

    public LicenseCheckerCallback getCallback() {
        return mCallback;
    }

    public int getNonce() {
        return mNonce;
    }

    public String getPackageName() {
        return mPackageName;
    }

    private static final String SIGNATURE_ALGORITHM = "SHA1withRSA";

    /**
     * Verifies the response from server and calls appropriate callback method.
     *
     * @param publicKey public key associated with the developer account
     * @param responseCode server response code
     * @param signedData signed data from server
     * @param signature server signature
     */
    public void verify(PublicKey publicKey, int responseCode, String signedData, String signature) {
        String userId = null;
        // Skip signature check for unsuccessful requests
        ResponseData data = null;
        if (responseCode == LICENSED || responseCode == NOT_LICENSED ||
                responseCode == LICENSED_OLD_KEY) {
            // Verify signature.
            try {
                if (TextUtils.isEmpty(signedData)) {
                    Log.e(TAG, "Signature verification failed: signedData is empty. " +
                            "(Device not signed-in to any Google accounts?)");
                    handleInvalidResponse();
                    return;
                }

                Signature sig = Signature.getInstance(SIGNATURE_ALGORITHM);
                sig.initVerify(publicKey);
                sig.update(signedData.getBytes());

                if (!sig.verify(Base64.decode(signature))) {
                    Log.e(TAG, "Signature verification failed.");
                    handleInvalidResponse();
                    return;
                }
            } catch (NoSuchAlgorithmException e) {
                // This can't happen on an Android compatible device.
                throw new RuntimeException(e);
            } catch (InvalidKeyException e) {
                handleApplicationError(LicenseCheckerCallback.ERROR_INVALID_PUBLIC_KEY);
                return;
            } catch (SignatureException e) {
                throw new RuntimeException(e);
            } catch (Base64DecoderException e) {
                Log.e(TAG, "Could not Base64-decode signature.");
                handleInvalidResponse();
                return;
            }

            // Parse and validate response.
            try {
                data = ResponseData.parse(signedData);
            } catch (IllegalArgumentException e) {
                Log.e(TAG, "Could not parse response.");
                handleInvalidResponse();
                return;
            }

            if (data.responseCode != responseCode) {
                Log.e(TAG, "Response codes don't match.");
                handleInvalidResponse();
                return;
            }

            if (data.nonce != mNonce) {
                Log.e(TAG, "Nonce doesn't match.");
                handleInvalidResponse();
                return;
            }

            if (!data.packageName.equals(mPackageName)) {
                Log.e(TAG, "Package name doesn't match.");
                handleInvalidResponse();
                return;
            }

            if (!data.versionCode.equals(mVersionCode)) {
                Log.e(TAG, "Version codes don't match.");
                handleInvalidResponse();
                return;
            }

            // Application-specific user identifier.
            userId = data.userId;
            if (TextUtils.isEmpty(userId)) {
                Log.e(TAG, "User identifier is empty.");
                handleInvalidResponse();
                return;
            }
        }

        switch (responseCode) {
            case LICENSED:
            case LICENSED_OLD_KEY:
                int limiterResponse = mDeviceLimiter.isDeviceAllowed(userId);
                handleResponse(limiterResponse, data);
                break;
            case NOT_LICENSED:
                handleResponse(Policy.NOT_LICENSED, data);
                break;
            case ERROR_CONTACTING_SERVER:
                Log.w(TAG, "Error contacting licensing server.");
                handleResponse(Policy.RETRY, data);
                break;
            case ERROR_SERVER_FAILURE:
                Log.w(TAG, "An error has occurred on the licensing server.");
                handleResponse(Policy.RETRY, data);
                break;
            case ERROR_OVER_QUOTA:
                Log.w(TAG, "Licensing server is refusing to talk to this device, over quota.");
                handleResponse(Policy.RETRY, data);
                break;
            case ERROR_INVALID_PACKAGE_NAME:
                handleApplicationError(LicenseCheckerCallback.ERROR_INVALID_PACKAGE_NAME);
                break;
            case ERROR_NON_MATCHING_UID:
                handleApplicationError(LicenseCheckerCallback.ERROR_NON_MATCHING_UID);
                break;
            case ERROR_NOT_MARKET_MANAGED:
                handleApplicationError(LicenseCheckerCallback.ERROR_NOT_MARKET_MANAGED);
                break;
            default:
                Log.e(TAG, "Unknown response code for license check.");
                handleInvalidResponse();
        }
    }

    /**
     * Confers with policy and calls appropriate callback method.
     *
     * @param response
     * @param rawData
     */
    private void handleResponse(int response, ResponseData rawData) {
        // Update policy data and increment retry counter (if needed)
        mPolicy.processServerResponse(response, rawData);

        // Given everything we know, including cached data, ask the policy if we should grant
        // access.
        if (mPolicy.allowAccess()) {
            mCallback.allow(response);
        } else {
            mCallback.dontAllow(response);
        }
    }

    private void handleApplicationError(int code) {
        mCallback.applicationError(code);
    }

    private void handleInvalidResponse() {
        mCallback.dontAllow(Policy.NOT_LICENSED);
    }
}
