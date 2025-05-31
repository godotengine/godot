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

import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.List;

/**
 * Base implementation of an APK signer.
 */
public class ApkSignerInfo {
    public int index;
    public long timestamp;
    public List<X509Certificate> certs = new ArrayList<>();
    public List<X509Certificate> certificateLineage = new ArrayList<>();

    private final List<ApkVerificationIssue> mInfoMessages = new ArrayList<>();
    private final List<ApkVerificationIssue> mWarnings = new ArrayList<>();
    private final List<ApkVerificationIssue> mErrors = new ArrayList<>();

    /**
     * Adds a new {@link ApkVerificationIssue} as an error to this signer using the provided {@code
     * issueId} and {@code params}.
     */
    public void addError(int issueId, Object... params) {
        mErrors.add(new ApkVerificationIssue(issueId, params));
    }

    /**
     * Adds a new {@link ApkVerificationIssue} as a warning to this signer using the provided {@code
     * issueId} and {@code params}.
     */
    public void addWarning(int issueId, Object... params) {
        mWarnings.add(new ApkVerificationIssue(issueId, params));
    }

    /**
     * Adds a new {@link ApkVerificationIssue} as an info message to this signer config using the
     * provided {@code issueId} and {@code params}.
     */
    public void addInfoMessage(int issueId, Object... params) {
        mInfoMessages.add(new ApkVerificationIssue(issueId, params));
    }

    /**
     * Returns {@code true} if any errors were encountered during verification for this signer.
     */
    public boolean containsErrors() {
        return !mErrors.isEmpty();
    }

    /**
     * Returns {@code true} if any warnings were encountered during verification for this signer.
     */
    public boolean containsWarnings() {
        return !mWarnings.isEmpty();
    }

    /**
     * Returns {@code true} if any info messages were encountered during verification of this
     * signer.
     */
    public boolean containsInfoMessages() {
        return !mInfoMessages.isEmpty();
    }

    /**
     * Returns the errors encountered during verification for this signer.
     */
    public List<? extends ApkVerificationIssue> getErrors() {
        return mErrors;
    }

    /**
     * Returns the warnings encountered during verification for this signer.
     */
    public List<? extends ApkVerificationIssue> getWarnings() {
        return mWarnings;
    }

    /**
     * Returns the info messages encountered during verification of this signer.
     */
    public List<? extends ApkVerificationIssue> getInfoMessages() {
        return mInfoMessages;
    }
}
