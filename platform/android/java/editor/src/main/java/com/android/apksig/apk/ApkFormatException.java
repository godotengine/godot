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

package com.android.apksig.apk;

/**
 * Indicates that an APK is not well-formed. For example, this may indicate that the APK is not a
 * well-formed ZIP archive, in which case {@link #getCause()} will return a
 * {@link com.android.apksig.zip.ZipFormatException ZipFormatException}, or that the APK contains
 * multiple ZIP entries with the same name.
 */
public class ApkFormatException extends Exception {
    private static final long serialVersionUID = 1L;

    public ApkFormatException(String message) {
        super(message);
    }

    public ApkFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
