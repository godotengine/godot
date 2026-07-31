/*
 * Copyright (C) 2016 The Android Open Source Project
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

package com.android.apksig.internal.apk.v1;

import java.util.Comparator;

/**
 * Digest algorithm used with JAR signing (aka v1 signing scheme).
 */
public enum DigestAlgorithm {
    /** SHA-1 */
    SHA1("SHA-1"),

    /** SHA2-256 */
    SHA256("SHA-256");

    private final String mJcaMessageDigestAlgorithm;

    private DigestAlgorithm(String jcaMessageDigestAlgoritm) {
        mJcaMessageDigestAlgorithm = jcaMessageDigestAlgoritm;
    }

    /**
     * Returns the {@link java.security.MessageDigest} algorithm represented by this digest
     * algorithm.
     */
    String getJcaMessageDigestAlgorithm() {
        return mJcaMessageDigestAlgorithm;
    }

    public static Comparator<DigestAlgorithm> BY_STRENGTH_COMPARATOR = new StrengthComparator();

    private static class StrengthComparator implements Comparator<DigestAlgorithm> {
        @Override
        public int compare(DigestAlgorithm a1, DigestAlgorithm a2) {
            switch (a1) {
                case SHA1:
                    switch (a2) {
                        case SHA1:
                            return 0;
                        case SHA256:
                            return -1;
                    }
                    throw new RuntimeException("Unsupported algorithm: " + a2);

                case SHA256:
                    switch (a2) {
                        case SHA1:
                            return 1;
                        case SHA256:
                            return 0;
                    }
                    throw new RuntimeException("Unsupported algorithm: " + a2);

                default:
                    throw new RuntimeException("Unsupported algorithm: " + a1);
            }
        }
    }
}
