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

package com.android.apksig.internal.util;

import java.math.BigInteger;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.Principal;
import java.security.Provider;
import java.security.PublicKey;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateExpiredException;
import java.security.cert.CertificateNotYetValidException;
import java.security.cert.CertificateParsingException;
import java.security.cert.X509Certificate;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Set;

import javax.security.auth.x500.X500Principal;

/**
 * {@link X509Certificate} which delegates all method invocations to the provided delegate
 * {@code X509Certificate}.
 */
public class DelegatingX509Certificate extends X509Certificate {
    private static final long serialVersionUID = 1L;

    private final X509Certificate mDelegate;

    public DelegatingX509Certificate(X509Certificate delegate) {
        this.mDelegate = delegate;
    }

    @Override
    public Set<String> getCriticalExtensionOIDs() {
        return mDelegate.getCriticalExtensionOIDs();
    }

    @Override
    public byte[] getExtensionValue(String oid) {
        return mDelegate.getExtensionValue(oid);
    }

    @Override
    public Set<String> getNonCriticalExtensionOIDs() {
        return mDelegate.getNonCriticalExtensionOIDs();
    }

    @Override
    public boolean hasUnsupportedCriticalExtension() {
        return mDelegate.hasUnsupportedCriticalExtension();
    }

    @Override
    public void checkValidity()
            throws CertificateExpiredException, CertificateNotYetValidException {
        mDelegate.checkValidity();
    }

    @Override
    public void checkValidity(Date date)
            throws CertificateExpiredException, CertificateNotYetValidException {
        mDelegate.checkValidity(date);
    }

    @Override
    public int getVersion() {
        return mDelegate.getVersion();
    }

    @Override
    public BigInteger getSerialNumber() {
        return mDelegate.getSerialNumber();
    }

    @Override
    public Principal getIssuerDN() {
        return mDelegate.getIssuerDN();
    }

    @Override
    public Principal getSubjectDN() {
        return mDelegate.getSubjectDN();
    }

    @Override
    public Date getNotBefore() {
        return mDelegate.getNotBefore();
    }

    @Override
    public Date getNotAfter() {
        return mDelegate.getNotAfter();
    }

    @Override
    public byte[] getTBSCertificate() throws CertificateEncodingException {
        return mDelegate.getTBSCertificate();
    }

    @Override
    public byte[] getSignature() {
        return mDelegate.getSignature();
    }

    @Override
    public String getSigAlgName() {
        return mDelegate.getSigAlgName();
    }

    @Override
    public String getSigAlgOID() {
        return mDelegate.getSigAlgOID();
    }

    @Override
    public byte[] getSigAlgParams() {
        return mDelegate.getSigAlgParams();
    }

    @Override
    public boolean[] getIssuerUniqueID() {
        return mDelegate.getIssuerUniqueID();
    }

    @Override
    public boolean[] getSubjectUniqueID() {
        return mDelegate.getSubjectUniqueID();
    }

    @Override
    public boolean[] getKeyUsage() {
        return mDelegate.getKeyUsage();
    }

    @Override
    public int getBasicConstraints() {
        return mDelegate.getBasicConstraints();
    }

    @Override
    public byte[] getEncoded() throws CertificateEncodingException {
        return mDelegate.getEncoded();
    }

    @Override
    public void verify(PublicKey key) throws CertificateException, NoSuchAlgorithmException,
            InvalidKeyException, NoSuchProviderException, SignatureException {
        mDelegate.verify(key);
    }

    @Override
    public void verify(PublicKey key, String sigProvider)
            throws CertificateException, NoSuchAlgorithmException, InvalidKeyException,
            NoSuchProviderException, SignatureException {
        mDelegate.verify(key, sigProvider);
    }

    @Override
    public String toString() {
        return mDelegate.toString();
    }

    @Override
    public PublicKey getPublicKey() {
        return mDelegate.getPublicKey();
    }

    @Override
    public X500Principal getIssuerX500Principal() {
        return mDelegate.getIssuerX500Principal();
    }

    @Override
    public X500Principal getSubjectX500Principal() {
        return mDelegate.getSubjectX500Principal();
    }

    @Override
    public List<String> getExtendedKeyUsage() throws CertificateParsingException {
        return mDelegate.getExtendedKeyUsage();
    }

    @Override
    public Collection<List<?>> getSubjectAlternativeNames() throws CertificateParsingException {
        return mDelegate.getSubjectAlternativeNames();
    }

    @Override
    public Collection<List<?>> getIssuerAlternativeNames() throws CertificateParsingException {
        return mDelegate.getIssuerAlternativeNames();
    }

    @Override
    @SuppressWarnings("AndroidJdkLibsChecker")
    public void verify(PublicKey key, Provider sigProvider) throws CertificateException,
            NoSuchAlgorithmException, InvalidKeyException, SignatureException {
        mDelegate.verify(key, sigProvider);
    }
}
