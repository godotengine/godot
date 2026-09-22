/*
 * Copyright (C) 2017 The Android Open Source Project
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

package com.android.apksig.internal.pkcs7;

/**
 * Assorted PKCS #7 constants from RFC 5652.
 */
public abstract class Pkcs7Constants {
    private Pkcs7Constants() {}

    public static final String OID_DATA = "1.2.840.113549.1.7.1";
    public static final String OID_SIGNED_DATA = "1.2.840.113549.1.7.2";
    public static final String OID_CONTENT_TYPE = "1.2.840.113549.1.9.3";
    public static final String OID_MESSAGE_DIGEST = "1.2.840.113549.1.9.4";
}
