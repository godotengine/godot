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

package com.android.apksig;

/**
 * This class is intended as a lightweight representation of an APK signature verification issue
 * where the client does not require the additional textual details provided by a subclass.
 */
public class ApkVerificationIssue {
    /* The V2 signer(s) could not be read from the V2 signature block */
    public static final int V2_SIG_MALFORMED_SIGNERS = 1;
    /* A V2 signature block exists without any V2 signers */
    public static final int V2_SIG_NO_SIGNERS = 2;
    /* Failed to parse a signer's block in the V2 signature block */
    public static final int V2_SIG_MALFORMED_SIGNER = 3;
    /* Failed to parse the signer's signature record in the V2 signature block */
    public static final int V2_SIG_MALFORMED_SIGNATURE = 4;
    /* The V2 signer contained no signatures */
    public static final int V2_SIG_NO_SIGNATURES = 5;
    /* The V2 signer's certificate could not be parsed */
    public static final int V2_SIG_MALFORMED_CERTIFICATE = 6;
    /* No signing certificates exist for the V2 signer */
    public static final int V2_SIG_NO_CERTIFICATES = 7;
    /* Failed to parse the V2 signer's digest record */
    public static final int V2_SIG_MALFORMED_DIGEST = 8;
    /* The V3 signer(s) could not be read from the V3 signature block */
    public static final int V3_SIG_MALFORMED_SIGNERS = 9;
    /* A V3 signature block exists without any V3 signers */
    public static final int V3_SIG_NO_SIGNERS = 10;
    /* Failed to parse a signer's block in the V3 signature block */
    public static final int V3_SIG_MALFORMED_SIGNER = 11;
    /* Failed to parse the signer's signature record in the V3 signature block */
    public static final int V3_SIG_MALFORMED_SIGNATURE = 12;
    /* The V3 signer contained no signatures */
    public static final int V3_SIG_NO_SIGNATURES = 13;
    /* The V3 signer's certificate could not be parsed */
    public static final int V3_SIG_MALFORMED_CERTIFICATE = 14;
    /* No signing certificates exist for the V3 signer */
    public static final int V3_SIG_NO_CERTIFICATES = 15;
    /* Failed to parse the V3 signer's digest record */
    public static final int V3_SIG_MALFORMED_DIGEST = 16;
    /* The source stamp signer contained no signatures */
    public static final int SOURCE_STAMP_NO_SIGNATURE = 17;
    /* The source stamp signer's certificate could not be parsed */
    public static final int SOURCE_STAMP_MALFORMED_CERTIFICATE = 18;
    /* The source stamp contains a signature produced using an unknown algorithm */
    public static final int SOURCE_STAMP_UNKNOWN_SIG_ALGORITHM = 19;
    /* Failed to parse the signer's signature in the source stamp signature block */
    public static final int SOURCE_STAMP_MALFORMED_SIGNATURE = 20;
    /* The source stamp's signature block failed verification */
    public static final int SOURCE_STAMP_DID_NOT_VERIFY = 21;
    /* An exception was encountered when verifying the source stamp */
    public static final int SOURCE_STAMP_VERIFY_EXCEPTION = 22;
    /* The certificate digest in the APK does not match the expected digest */
    public static final int SOURCE_STAMP_EXPECTED_DIGEST_MISMATCH = 23;
    /*
     * The APK contains a source stamp signature block without a corresponding stamp certificate
     * digest in the APK contents.
     */
    public static final int SOURCE_STAMP_SIGNATURE_BLOCK_WITHOUT_CERT_DIGEST = 24;
    /*
     * The APK does not contain the source stamp certificate digest file nor the source stamp
     * signature block.
     */
    public static final int SOURCE_STAMP_CERT_DIGEST_AND_SIG_BLOCK_MISSING = 25;
    /*
     * None of the signatures provided by the source stamp were produced with a known signature
     * algorithm.
     */
    public static final int SOURCE_STAMP_NO_SUPPORTED_SIGNATURE = 26;
    /*
     * The source stamp signer's certificate in the signing block does not match the certificate in
     * the APK.
     */
    public static final int SOURCE_STAMP_CERTIFICATE_MISMATCH_BETWEEN_SIGNATURE_BLOCK_AND_APK = 27;
    /* The APK could not be properly parsed due to a ZIP or APK format exception */
    public static final int MALFORMED_APK = 28;
    /* An unexpected exception was caught when attempting to verify the APK's signatures */
    public static final int UNEXPECTED_EXCEPTION = 29;
    /* The APK contains the certificate digest file but does not contain a stamp signature block */
    public static final int SOURCE_STAMP_SIG_MISSING = 30;
    /* Source stamp block contains a malformed attribute. */
    public static final int SOURCE_STAMP_MALFORMED_ATTRIBUTE = 31;
    /* Source stamp block contains an unknown attribute. */
    public static final int SOURCE_STAMP_UNKNOWN_ATTRIBUTE = 32;
    /**
     * Failed to parse the SigningCertificateLineage structure in the source stamp
     * attributes section.
     */
    public static final int SOURCE_STAMP_MALFORMED_LINEAGE = 33;
    /**
     * The source stamp certificate does not match the terminal node in the provided
     * proof-of-rotation structure describing the stamp certificate history.
     */
    public static final int SOURCE_STAMP_POR_CERT_MISMATCH = 34;
    /**
     * The source stamp SigningCertificateLineage attribute contains a proof-of-rotation record
     * with signature(s) that did not verify.
     */
    public static final int SOURCE_STAMP_POR_DID_NOT_VERIFY = 35;
    /** No V1 / jar signing signature blocks were found in the APK. */
    public static final int JAR_SIG_NO_SIGNATURES = 36;
    /** An exception was encountered when parsing the V1 / jar signer in the signature block. */
    public static final int JAR_SIG_PARSE_EXCEPTION = 37;
    /** The source stamp timestamp attribute has an invalid value. */
    public static final int SOURCE_STAMP_INVALID_TIMESTAMP = 38;

    private final int mIssueId;
    private final String mFormat;
    private final Object[] mParams;

    /**
     * Constructs a new {@code ApkVerificationIssue} using the provided {@code format} string and
     * {@code params}.
     */
    public ApkVerificationIssue(String format, Object... params) {
        mIssueId = -1;
        mFormat = format;
        mParams = params;
    }

    /**
     * Constructs a new {@code ApkVerificationIssue} using the provided {@code issueId} and {@code
     * params}.
     */
    public ApkVerificationIssue(int issueId, Object... params) {
        mIssueId = issueId;
        mFormat = null;
        mParams = params;
    }

    /**
     * Returns the numeric ID for this issue.
     */
    public int getIssueId() {
        return mIssueId;
    }

    /**
     * Returns the optional parameters for this issue.
     */
    public Object[] getParams() {
        return mParams;
    }

    @Override
    public String toString() {
        // If this instance was created by a subclass with a format string then return the same
        // formatted String as the subclass.
        if (mFormat != null) {
            return String.format(mFormat, mParams);
        }
        StringBuilder result = new StringBuilder("mIssueId: ").append(mIssueId);
        for (Object param : mParams) {
            result.append(", ").append(param.toString());
        }
        return result.toString();
    }
}
