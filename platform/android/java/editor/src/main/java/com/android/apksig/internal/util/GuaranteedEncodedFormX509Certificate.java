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

package com.android.apksig.internal.util;

import java.security.cert.CertificateEncodingException;
import java.security.cert.X509Certificate;
import java.util.Arrays;

/**
 * {@link X509Certificate} whose {@link #getEncoded()} returns the data provided at construction
 * time.
 */
public class GuaranteedEncodedFormX509Certificate extends DelegatingX509Certificate {
    private static final long serialVersionUID = 1L;

    private final byte[] mEncodedForm;
    private int mHash = -1;

    public GuaranteedEncodedFormX509Certificate(X509Certificate wrapped, byte[] encodedForm) {
        super(wrapped);
        this.mEncodedForm = (encodedForm != null) ? encodedForm.clone() : null;
    }

    @Override
    public byte[] getEncoded() throws CertificateEncodingException {
        return (mEncodedForm != null) ? mEncodedForm.clone() : null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof X509Certificate)) return false;

        try {
            byte[] a = this.getEncoded();
            byte[] b = ((X509Certificate) o).getEncoded();
            return Arrays.equals(a, b);
        } catch (CertificateEncodingException e) {
            return false;
        }
    }

    @Override
    public int hashCode() {
        if (mHash == -1) {
            try {
                mHash = Arrays.hashCode(this.getEncoded());
            } catch (CertificateEncodingException e) {
                mHash = 0;
            }
        }
        return mHash;
    }
}
