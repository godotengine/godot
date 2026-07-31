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
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.jar.Attributes;

/**
 * Producer of {@code META-INF/MANIFEST.MF} file.
 *
 * @see <a href="https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#JAR_Manifest">JAR Manifest format</a>
 */
public abstract class ManifestWriter {

    private static final byte[] CRLF = new byte[] {'\r', '\n'};
    private static final int MAX_LINE_LENGTH = 70;

    private ManifestWriter() {}

    public static void writeMainSection(OutputStream out, Attributes attributes)
            throws IOException {

        // Main section must start with the Manifest-Version attribute.
        // See https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File.
        String manifestVersion = attributes.getValue(Attributes.Name.MANIFEST_VERSION);
        if (manifestVersion == null) {
            throw new IllegalArgumentException(
                    "Mandatory " + Attributes.Name.MANIFEST_VERSION + " attribute missing");
        }
        writeAttribute(out, Attributes.Name.MANIFEST_VERSION, manifestVersion);

        if (attributes.size() > 1) {
            SortedMap<String, String> namedAttributes = getAttributesSortedByName(attributes);
            namedAttributes.remove(Attributes.Name.MANIFEST_VERSION.toString());
            writeAttributes(out, namedAttributes);
        }
        writeSectionDelimiter(out);
    }

    public static void writeIndividualSection(OutputStream out, String name, Attributes attributes)
            throws IOException {
        writeAttribute(out, "Name", name);

        if (!attributes.isEmpty()) {
            writeAttributes(out, getAttributesSortedByName(attributes));
        }
        writeSectionDelimiter(out);
    }

    static void writeSectionDelimiter(OutputStream out) throws IOException {
        out.write(CRLF);
    }

    static void writeAttribute(OutputStream  out, Attributes.Name name, String value)
            throws IOException {
        writeAttribute(out, name.toString(), value);
    }

    private static void writeAttribute(OutputStream  out, String name, String value)
            throws IOException {
        writeLine(out, name + ": " + value);
    }

    private static void writeLine(OutputStream  out, String line) throws IOException {
        byte[] lineBytes = line.getBytes(StandardCharsets.UTF_8);
        int offset = 0;
        int remaining = lineBytes.length;
        boolean firstLine = true;
        while (remaining > 0) {
            int chunkLength;
            if (firstLine) {
                // First line
                chunkLength = Math.min(remaining, MAX_LINE_LENGTH);
            } else {
                // Continuation line
                out.write(CRLF);
                out.write(' ');
                chunkLength = Math.min(remaining, MAX_LINE_LENGTH - 1);
            }
            out.write(lineBytes, offset, chunkLength);
            offset += chunkLength;
            remaining -= chunkLength;
            firstLine = false;
        }
        out.write(CRLF);
    }

    static SortedMap<String, String> getAttributesSortedByName(Attributes attributes) {
        Set<Map.Entry<Object, Object>> attributesEntries = attributes.entrySet();
        SortedMap<String, String> namedAttributes = new TreeMap<String, String>();
        for (Map.Entry<Object, Object> attribute : attributesEntries) {
            String attrName = attribute.getKey().toString();
            String attrValue = attribute.getValue().toString();
            namedAttributes.put(attrName, attrValue);
        }
        return namedAttributes;
    }

    static void writeAttributes(
            OutputStream out, SortedMap<String, String> attributesSortedByName) throws IOException {
        for (Map.Entry<String, String> attribute : attributesSortedByName.entrySet()) {
            String attrName = attribute.getKey();
            String attrValue = attribute.getValue();
            writeAttribute(out, attrName, attrValue);
        }
    }
}
