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
 * Indicates that there was an issue determining the minimum Android platform version supported by
 * an APK.
 */
public class MinSdkVersionException extends ApkFormatException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new {@code MinSdkVersionException} with the provided message.
     */
    public MinSdkVersionException(String message) {
        super(message);
    }

    /**
     * Constructs a new {@code MinSdkVersionException} with the provided message and cause.
     */
    public MinSdkVersionException(String message, Throwable cause) {
        super(message, cause);
    }
}
