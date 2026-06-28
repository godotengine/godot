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

package com.android.apksig.internal.oid;

import com.android.apksig.internal.util.InclusiveIntRange;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OidConstants {
    public static final String OID_DIGEST_MD5 = "1.2.840.113549.2.5";
    public static final String OID_DIGEST_SHA1 = "1.3.14.3.2.26";
    public static final String OID_DIGEST_SHA224 = "2.16.840.1.101.3.4.2.4";
    public static final String OID_DIGEST_SHA256 = "2.16.840.1.101.3.4.2.1";
    public static final String OID_DIGEST_SHA384 = "2.16.840.1.101.3.4.2.2";
    public static final String OID_DIGEST_SHA512 = "2.16.840.1.101.3.4.2.3";

    public static final String OID_SIG_RSA = "1.2.840.113549.1.1.1";
    public static final String OID_SIG_MD5_WITH_RSA = "1.2.840.113549.1.1.4";
    public static final String OID_SIG_SHA1_WITH_RSA = "1.2.840.113549.1.1.5";
    public static final String OID_SIG_SHA224_WITH_RSA = "1.2.840.113549.1.1.14";
    public static final String OID_SIG_SHA256_WITH_RSA = "1.2.840.113549.1.1.11";
    public static final String OID_SIG_SHA384_WITH_RSA = "1.2.840.113549.1.1.12";
    public static final String OID_SIG_SHA512_WITH_RSA = "1.2.840.113549.1.1.13";

    public static final String OID_SIG_DSA = "1.2.840.10040.4.1";
    public static final String OID_SIG_SHA1_WITH_DSA = "1.2.840.10040.4.3";
    public static final String OID_SIG_SHA224_WITH_DSA = "2.16.840.1.101.3.4.3.1";
    public static final String OID_SIG_SHA256_WITH_DSA = "2.16.840.1.101.3.4.3.2";
    public static final String OID_SIG_SHA384_WITH_DSA = "2.16.840.1.101.3.4.3.3";
    public static final String OID_SIG_SHA512_WITH_DSA = "2.16.840.1.101.3.4.3.4";

    public static final String OID_SIG_EC_PUBLIC_KEY = "1.2.840.10045.2.1";
    public static final String OID_SIG_SHA1_WITH_ECDSA = "1.2.840.10045.4.1";
    public static final String OID_SIG_SHA224_WITH_ECDSA = "1.2.840.10045.4.3.1";
    public static final String OID_SIG_SHA256_WITH_ECDSA = "1.2.840.10045.4.3.2";
    public static final String OID_SIG_SHA384_WITH_ECDSA = "1.2.840.10045.4.3.3";
    public static final String OID_SIG_SHA512_WITH_ECDSA = "1.2.840.10045.4.3.4";

    public static final Map<String, List<InclusiveIntRange>> SUPPORTED_SIG_ALG_OIDS =
            new HashMap<>();
    static {
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_RSA,
                InclusiveIntRange.from(0));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_MD5_WITH_RSA,
                InclusiveIntRange.fromTo(0, 8), InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA1_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA224_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA256_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA384_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA512_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_RSA,
                InclusiveIntRange.from(0));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_MD5_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA1_WITH_RSA,
                InclusiveIntRange.from(0));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA224_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA256_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA384_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA512_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_RSA,
                InclusiveIntRange.fromTo(0, 8), InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_MD5_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA1_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA224_WITH_RSA,
                InclusiveIntRange.fromTo(0, 8), InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA256_WITH_RSA,
                InclusiveIntRange.fromTo(21, 21));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA384_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA512_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_RSA,
                InclusiveIntRange.fromTo(0, 8), InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_MD5_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA1_WITH_RSA,
                InclusiveIntRange.fromTo(21, 21));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA224_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA256_WITH_RSA,
                InclusiveIntRange.fromTo(0, 8), InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA384_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA512_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_RSA,
                InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_MD5_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA1_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA224_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA256_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA384_WITH_RSA,
                InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA512_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_RSA,
                InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_MD5_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA1_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA224_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA256_WITH_RSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA384_WITH_RSA,
                InclusiveIntRange.fromTo(21, 21));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA512_WITH_RSA,
                InclusiveIntRange.from(21));

        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA1_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA224_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA256_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_DSA,
                InclusiveIntRange.from(0));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA1_WITH_DSA,
                InclusiveIntRange.from(9));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA224_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA256_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_DSA,
                InclusiveIntRange.from(22));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA1_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA224_WITH_DSA,
                InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA256_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_DSA,
                InclusiveIntRange.from(22));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA1_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA224_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA256_WITH_DSA,
                InclusiveIntRange.from(21));

        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA1_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA224_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA256_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA1_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA224_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA256_WITH_DSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_EC_PUBLIC_KEY,
                InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_EC_PUBLIC_KEY,
                InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_EC_PUBLIC_KEY,
                InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_EC_PUBLIC_KEY,
                InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_EC_PUBLIC_KEY,
                InclusiveIntRange.from(18));

        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA1_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA224_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA256_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA384_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_MD5, OID_SIG_SHA512_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA1_WITH_ECDSA,
                InclusiveIntRange.from(18));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA224_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA256_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA384_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA1, OID_SIG_SHA512_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA1_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA224_WITH_ECDSA,
                InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA256_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA384_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA224, OID_SIG_SHA512_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA1_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA224_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA256_WITH_ECDSA,
                InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA384_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA256, OID_SIG_SHA512_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA1_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA224_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA256_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA384_WITH_ECDSA,
                InclusiveIntRange.from(21));
        addSupportedSigAlg(
                OID_DIGEST_SHA384, OID_SIG_SHA512_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));

        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA1_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA224_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA256_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA384_WITH_ECDSA,
                InclusiveIntRange.fromTo(21, 23));
        addSupportedSigAlg(
                OID_DIGEST_SHA512, OID_SIG_SHA512_WITH_ECDSA,
                InclusiveIntRange.from(21));
    }

    public static void addSupportedSigAlg(
            String digestAlgorithmOid,
            String signatureAlgorithmOid,
            InclusiveIntRange... supportedApiLevels) {
        SUPPORTED_SIG_ALG_OIDS.put(
                digestAlgorithmOid + "with" + signatureAlgorithmOid,
                Arrays.asList(supportedApiLevels));
    }

    public static List<InclusiveIntRange> getSigAlgSupportedApiLevels(
            String digestAlgorithmOid,
            String signatureAlgorithmOid) {
        List<InclusiveIntRange> result =
                SUPPORTED_SIG_ALG_OIDS.get(digestAlgorithmOid + "with" + signatureAlgorithmOid);
        return (result != null) ? result : Collections.emptyList();
    }

    public static class OidToUserFriendlyNameMapper {
        private OidToUserFriendlyNameMapper() {}

        private static final Map<String, String> OID_TO_USER_FRIENDLY_NAME = new HashMap<>();
        static {
            OID_TO_USER_FRIENDLY_NAME.put(OID_DIGEST_MD5, "MD5");
            OID_TO_USER_FRIENDLY_NAME.put(OID_DIGEST_SHA1, "SHA-1");
            OID_TO_USER_FRIENDLY_NAME.put(OID_DIGEST_SHA224, "SHA-224");
            OID_TO_USER_FRIENDLY_NAME.put(OID_DIGEST_SHA256, "SHA-256");
            OID_TO_USER_FRIENDLY_NAME.put(OID_DIGEST_SHA384, "SHA-384");
            OID_TO_USER_FRIENDLY_NAME.put(OID_DIGEST_SHA512, "SHA-512");

            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_RSA, "RSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_MD5_WITH_RSA, "MD5 with RSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA1_WITH_RSA, "SHA-1 with RSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA224_WITH_RSA, "SHA-224 with RSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA256_WITH_RSA, "SHA-256 with RSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA384_WITH_RSA, "SHA-384 with RSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA512_WITH_RSA, "SHA-512 with RSA");


            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_DSA, "DSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA1_WITH_DSA, "SHA-1 with DSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA224_WITH_DSA, "SHA-224 with DSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA256_WITH_DSA, "SHA-256 with DSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA384_WITH_DSA, "SHA-384 with DSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA512_WITH_DSA, "SHA-512 with DSA");

            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_EC_PUBLIC_KEY, "ECDSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA1_WITH_ECDSA, "SHA-1 with ECDSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA224_WITH_ECDSA, "SHA-224 with ECDSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA256_WITH_ECDSA, "SHA-256 with ECDSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA384_WITH_ECDSA, "SHA-384 with ECDSA");
            OID_TO_USER_FRIENDLY_NAME.put(OID_SIG_SHA512_WITH_ECDSA, "SHA-512 with ECDSA");
        }

        public static String getUserFriendlyNameForOid(String oid) {
            return OID_TO_USER_FRIENDLY_NAME.get(oid);
        }
    }

    public static final Map<String, String> OID_TO_JCA_DIGEST_ALG = new HashMap<>();
    static {
        OID_TO_JCA_DIGEST_ALG.put(OID_DIGEST_MD5, "MD5");
        OID_TO_JCA_DIGEST_ALG.put(OID_DIGEST_SHA1, "SHA-1");
        OID_TO_JCA_DIGEST_ALG.put(OID_DIGEST_SHA224, "SHA-224");
        OID_TO_JCA_DIGEST_ALG.put(OID_DIGEST_SHA256, "SHA-256");
        OID_TO_JCA_DIGEST_ALG.put(OID_DIGEST_SHA384, "SHA-384");
        OID_TO_JCA_DIGEST_ALG.put(OID_DIGEST_SHA512, "SHA-512");
    }

    public static final Map<String, String> OID_TO_JCA_SIGNATURE_ALG = new HashMap<>();
    static {
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_MD5_WITH_RSA, "MD5withRSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA1_WITH_RSA, "SHA1withRSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA224_WITH_RSA, "SHA224withRSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA256_WITH_RSA, "SHA256withRSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA384_WITH_RSA, "SHA384withRSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA512_WITH_RSA, "SHA512withRSA");

        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA1_WITH_DSA, "SHA1withDSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA224_WITH_DSA, "SHA224withDSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA256_WITH_DSA, "SHA256withDSA");

        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA1_WITH_ECDSA, "SHA1withECDSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA224_WITH_ECDSA, "SHA224withECDSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA256_WITH_ECDSA, "SHA256withECDSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA384_WITH_ECDSA, "SHA384withECDSA");
        OID_TO_JCA_SIGNATURE_ALG.put(OID_SIG_SHA512_WITH_ECDSA, "SHA512withECDSA");
    }

    private OidConstants() {}
}
