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

package com.android.apksig.internal.jar;

import java.io.IOException;
import java.io.OutputStream;
import java.util.SortedMap;
import java.util.jar.Attributes;

/**
 * Producer of JAR signature file ({@code *.SF}).
 *
 * @see <a href="https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#JAR_Manifest">JAR Manifest format</a>
 */
public abstract class SignatureFileWriter {
    private SignatureFileWriter() {}

    public static void writeMainSection(OutputStream out, Attributes attributes)
            throws IOException {

        // Main section must start with the Signature-Version attribute.
        // See https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File.
        String signatureVersion = attributes.getValue(Attributes.Name.SIGNATURE_VERSION);
        if (signatureVersion == null) {
            throw new IllegalArgumentException(
                    "Mandatory " + Attributes.Name.SIGNATURE_VERSION + " attribute missing");
        }
        ManifestWriter.writeAttribute(out, Attributes.Name.SIGNATURE_VERSION, signatureVersion);

        if (attributes.size() > 1) {
            SortedMap<String, String> namedAttributes =
                    ManifestWriter.getAttributesSortedByName(attributes);
            namedAttributes.remove(Attributes.Name.SIGNATURE_VERSION.toString());
            ManifestWriter.writeAttributes(out, namedAttributes);
        }
        writeSectionDelimiter(out);
    }

    public static void writeIndividualSection(OutputStream out, String name, Attributes attributes)
            throws IOException {
        ManifestWriter.writeIndividualSection(out, name, attributes);
    }

    public static void writeSectionDelimiter(OutputStream out) throws IOException {
        ManifestWriter.writeSectionDelimiter(out);
    }
}
