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
 * Indicates that an error was encountered while decoding a PKCS #7 structure.
 */
public class Pkcs7DecodingException extends Exception {
    private static final long serialVersionUID = 1L;

    public Pkcs7DecodingException(String message) {
        super(message);
    }

    public Pkcs7DecodingException(String message, Throwable cause) {
        super(message, cause);
    }
}
