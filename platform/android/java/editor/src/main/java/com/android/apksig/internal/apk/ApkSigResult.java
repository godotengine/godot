/*
 * Copyright (C) 2020 The Android Open Source Project
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

package com.android.apksig.internal.apk;

import com.android.apksig.ApkVerificationIssue;

import java.util.ArrayList;
import java.util.List;

/**
 * Base implementation of an APK signature verification result.
 */
public class ApkSigResult {
    public final int signatureSchemeVersion;

    /** Whether the APK's Signature Scheme signature verifies. */
    public boolean verified;

    public final List<ApkSignerInfo> mSigners = new ArrayList<>();
    private final List<ApkVerificationIssue> mWarnings = new ArrayList<>();
    private final List<ApkVerificationIssue> mErrors = new ArrayList<>();

    public ApkSigResult(int signatureSchemeVersion) {
        this.signatureSchemeVersion = signatureSchemeVersion;
    }

    /**
     * Returns {@code true} if this result encountered errors during verification.
     */
    public boolean containsErrors() {
        if (!mErrors.isEmpty()) {
            return true;
        }
        if (!mSigners.isEmpty()) {
            for (ApkSignerInfo signer : mSigners) {
                if (signer.containsErrors()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Returns {@code true} if this result encountered warnings during verification.
     */
    public boolean containsWarnings() {
        if (!mWarnings.isEmpty()) {
            return true;
        }
        if (!mSigners.isEmpty()) {
            for (ApkSignerInfo signer : mSigners) {
                if (signer.containsWarnings()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Adds a new {@link ApkVerificationIssue} as an error to this result using the provided {@code
     * issueId} and {@code params}.
     */
    public void addError(int issueId, Object... parameters) {
        mErrors.add(new ApkVerificationIssue(issueId, parameters));
    }

    /**
     * Adds a new {@link ApkVerificationIssue} as a warning to this result using the provided {@code
     * issueId} and {@code params}.
     */
    public void addWarning(int issueId, Object... parameters) {
        mWarnings.add(new ApkVerificationIssue(issueId, parameters));
    }

    /**
     * Returns the errors encountered during verification.
     */
    public List<? extends ApkVerificationIssue> getErrors() {
        return mErrors;
    }

    /**
     * Returns the warnings encountered during verification.
     */
    public List<? extends ApkVerificationIssue> getWarnings() {
        return mWarnings;
    }
}
