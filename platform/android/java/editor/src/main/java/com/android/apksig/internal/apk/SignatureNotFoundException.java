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

/**
 * Base exception that is thrown when the APK is not signed with the requested signature scheme.
 */
public class SignatureNotFoundException extends Exception {
    public SignatureNotFoundException(String message) {
        super(message);
    }

    public SignatureNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}