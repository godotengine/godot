/*
 * Copyright (C) 2018 The Android Open Source Project
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

import com.android.apksig.internal.asn1.Asn1BerParser;
import com.android.apksig.internal.asn1.Asn1DecodingException;
import com.android.apksig.internal.asn1.Asn1DerEncoder;
import com.android.apksig.internal.asn1.Asn1EncodingException;
import com.android.apksig.internal.x509.Certificate;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;

/**
 * Provides methods to generate {@code X509Certificate}s from their encoded form. These methods
 * can be used to generate certificates that would be rejected by the Java {@code
 * CertificateFactory}.
 */
public class X509CertificateUtils {

    private static volatile CertificateFactory sCertFactory = null;

    // The PEM certificate header and footer as specified in RFC 7468:
    //   There is exactly one space character (SP) separating the "BEGIN" or
    //   "END" from the label.  There are exactly five hyphen-minus (also
    //   known as dash) characters ("-") on both ends of the encapsulation
    //   boundaries, no more, no less.
    public static final byte[] BEGIN_CERT_HEADER = "-----BEGIN CERTIFICATE-----".getBytes();
    public static final byte[] END_CERT_FOOTER = "-----END CERTIFICATE-----".getBytes();

    private static void buildCertFactory() {
        if (sCertFactory != null) {
            return;
        }

        buildCertFactoryHelper();
    }

    private static synchronized void buildCertFactoryHelper() {
        if (sCertFactory != null) {
            return;
        }
        try {
            sCertFactory = CertificateFactory.getInstance("X.509");
        } catch (CertificateException e) {
            throw new RuntimeException("Failed to create X.509 CertificateFactory", e);
        }
    }

    /**
     * Generates an {@code X509Certificate} from the {@code InputStream}.
     *
     * @throws CertificateException if the {@code InputStream} cannot be decoded to a valid
     *                              certificate.
     */
    public static X509Certificate generateCertificate(InputStream in) throws CertificateException {
        byte[] encodedForm;
        try {
            encodedForm = ByteStreams.toByteArray(in);
        } catch (IOException e) {
            throw new CertificateException("Failed to parse certificate", e);
        }
        return generateCertificate(encodedForm);
    }

    /**
     * Generates an {@code X509Certificate} from the encoded form.
     *
     * @throws CertificateException if the encodedForm cannot be decoded to a valid certificate.
     */
    public static X509Certificate generateCertificate(byte[] encodedForm)
            throws CertificateException {
        buildCertFactory();
        return generateCertificate(encodedForm, sCertFactory);
    }

    /**
     * Generates an {@code X509Certificate} from the encoded form using the provided
     * {@code CertificateFactory}.
     *
     * @throws CertificateException if the encodedForm cannot be decoded to a valid certificate.
     */
    public static X509Certificate generateCertificate(byte[] encodedForm,
            CertificateFactory certFactory) throws CertificateException {
        X509Certificate certificate;
        try {
            certificate = (X509Certificate) certFactory.generateCertificate(
                    new ByteArrayInputStream(encodedForm));
            return certificate;
        } catch (CertificateException e) {
            // This could be expected if the certificate is encoded using a BER encoding that does
            // not use the minimum number of bytes to represent the length of the contents; attempt
            // to decode the certificate using the BER parser and re-encode using the DER encoder
            // below.
        }
        try {
            // Some apps were previously signed with a BER encoded certificate that now results
            // in exceptions from the CertificateFactory generateCertificate(s) methods. Since
            // the original BER encoding of the certificate is used as the signature for these
            // apps that original encoding must be maintained when signing updated versions of
            // these apps and any new apps that may require capabilities guarded by the
            // signature. To maintain the same signature the BER parser can be used to parse
            // the certificate, then it can be re-encoded to its DER equivalent which is
            // accepted by the generateCertificate method. The positions in the ByteBuffer can
            // then be used with the GuaranteedEncodedFormX509Certificate object to ensure the
            // getEncoded method returns the original signature of the app.
            ByteBuffer encodedCertBuffer = getNextDEREncodedCertificateBlock(
                    ByteBuffer.wrap(encodedForm));
            int startingPos = encodedCertBuffer.position();
            Certificate reencodedCert = Asn1BerParser.parse(encodedCertBuffer, Certificate.class);
            byte[] reencodedForm = Asn1DerEncoder.encode(reencodedCert);
            certificate = (X509Certificate) certFactory.generateCertificate(
                    new ByteArrayInputStream(reencodedForm));
            // If the reencodedForm is successfully accepted by the CertificateFactory then copy the
            // original encoding from the ByteBuffer and use that encoding in the Guaranteed object.
            byte[] originalEncoding = new byte[encodedCertBuffer.position() - startingPos];
            encodedCertBuffer.position(startingPos);
            encodedCertBuffer.get(originalEncoding);
            GuaranteedEncodedFormX509Certificate guaranteedEncodedCert =
                    new GuaranteedEncodedFormX509Certificate(certificate, originalEncoding);
            return guaranteedEncodedCert;
        } catch (Asn1DecodingException | Asn1EncodingException | CertificateException e) {
            throw new CertificateException("Failed to parse certificate", e);
        }
    }

    /**
     * Generates a {@code Collection} of {@code Certificate} objects from the encoded {@code
     * InputStream}.
     *
     * @throws CertificateException if the InputStream cannot be decoded to zero or more valid
     *                              {@code Certificate} objects.
     */
    public static Collection<? extends java.security.cert.Certificate> generateCertificates(
            InputStream in) throws CertificateException {
        buildCertFactory();
        return generateCertificates(in, sCertFactory);
    }

    /**
     * Generates a {@code Collection} of {@code Certificate} objects from the encoded {@code
     * InputStream} using the provided {@code CertificateFactory}.
     *
     * @throws CertificateException if the InputStream cannot be decoded to zero or more valid
     *                              {@code Certificates} objects.
     */
    public static Collection<? extends java.security.cert.Certificate> generateCertificates(
            InputStream in, CertificateFactory certFactory) throws CertificateException {
        // Since the InputStream is not guaranteed to support mark / reset operations first read it
        // into a byte array to allow using the BER parser / DER encoder if it cannot be read by
        // the CertificateFactory.
        byte[] encodedCerts;
        try {
            encodedCerts = ByteStreams.toByteArray(in);
        } catch (IOException e) {
            throw new CertificateException("Failed to read the input stream", e);
        }
        try {
            return certFactory.generateCertificates(new ByteArrayInputStream(encodedCerts));
        } catch (CertificateException e) {
            // This could be expected if the certificates are encoded using a BER encoding that does
            // not use the minimum number of bytes to represent the length of the contents; attempt
            // to decode the certificates using the BER parser and re-encode using the DER encoder
            // below.
        }
        try {
            Collection<X509Certificate> certificates = new ArrayList<>(1);
            ByteBuffer encodedCertsBuffer = ByteBuffer.wrap(encodedCerts);
            while (encodedCertsBuffer.hasRemaining()) {
                ByteBuffer certBuffer = getNextDEREncodedCertificateBlock(encodedCertsBuffer);
                int startingPos = certBuffer.position();
                Certificate reencodedCert = Asn1BerParser.parse(certBuffer, Certificate.class);
                byte[] reencodedForm = Asn1DerEncoder.encode(reencodedCert);
                X509Certificate certificate = (X509Certificate) certFactory.generateCertificate(
                        new ByteArrayInputStream(reencodedForm));
                byte[] originalEncoding = new byte[certBuffer.position() - startingPos];
                certBuffer.position(startingPos);
                certBuffer.get(originalEncoding);
                GuaranteedEncodedFormX509Certificate guaranteedEncodedCert =
                        new GuaranteedEncodedFormX509Certificate(certificate, originalEncoding);
                certificates.add(guaranteedEncodedCert);
            }
            return certificates;
        } catch (Asn1DecodingException | Asn1EncodingException e) {
            throw new CertificateException("Failed to parse certificates", e);
        }
    }

    /**
     * Parses the provided ByteBuffer to obtain the next certificate in DER encoding. If the buffer
     * does not begin with the PEM certificate header then it is returned with the assumption that
     * it is already DER encoded. If the buffer does begin with the PEM certificate header then the
     * certificate data is read from the buffer until the PEM certificate footer is reached; this
     * data is then base64 decoded and returned in a new ByteBuffer.
     *
     * If the buffer is in PEM format then the position of the buffer is moved to the end of the
     * current certificate; if the buffer is already DER encoded then the position of the buffer is
     * not modified.
     *
     * @throws CertificateException if the buffer contains the PEM certificate header but does not
     *                              contain the expected footer.
     */
    private static ByteBuffer getNextDEREncodedCertificateBlock(ByteBuffer certificateBuffer)
            throws CertificateException {
        if (certificateBuffer == null) {
            throw new NullPointerException("The certificateBuffer cannot be null");
        }
        // if the buffer does not contain enough data for the PEM cert header then just return the
        // provided buffer.
        if (certificateBuffer.remaining() < BEGIN_CERT_HEADER.length) {
            return certificateBuffer;
        }
        certificateBuffer.mark();
        for (int i = 0; i < BEGIN_CERT_HEADER.length; i++) {
            if (certificateBuffer.get() != BEGIN_CERT_HEADER[i]) {
                certificateBuffer.reset();
                return certificateBuffer;
            }
        }
        StringBuilder pemEncoding = new StringBuilder();
        while (certificateBuffer.hasRemaining()) {
            char encodedChar = (char) certificateBuffer.get();
            // if the current character is a '-' then the beginning of the footer has been reached
            if (encodedChar == '-') {
                break;
            } else if (Character.isWhitespace(encodedChar)) {
                continue;
            } else {
                pemEncoding.append(encodedChar);
            }
        }
        // start from the second index in the certificate footer since the first '-' should have
        // been consumed above.
        for (int i = 1; i < END_CERT_FOOTER.length; i++) {
            if (!certificateBuffer.hasRemaining()) {
                throw new CertificateException(
                        "The provided input contains the PEM certificate header but does not "
                                + "contain sufficient data for the footer");
            }
            if (certificateBuffer.get() != END_CERT_FOOTER[i]) {
                throw new CertificateException(
                        "The provided input contains the PEM certificate header without a "
                                + "valid certificate footer");
            }
        }
        byte[] derEncoding = Base64.getDecoder().decode(pemEncoding.toString());
        // consume any trailing whitespace in the byte buffer
        int nextEncodedChar = certificateBuffer.position();
        while (certificateBuffer.hasRemaining()) {
            char trailingChar = (char) certificateBuffer.get();
            if (Character.isWhitespace(trailingChar)) {
                nextEncodedChar++;
            } else {
                break;
            }
        }
        certificateBuffer.position(nextEncodedChar);
        return ByteBuffer.wrap(derEncoding);
    }
}
