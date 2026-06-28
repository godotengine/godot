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

import com.android.apksig.internal.apk.AndroidBinXmlParser;
import com.android.apksig.internal.apk.stamp.SourceStampConstants;
import com.android.apksig.internal.apk.v1.V1SchemeVerifier;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.internal.zip.CentralDirectoryRecord;
import com.android.apksig.internal.zip.LocalFileRecord;
import com.android.apksig.internal.zip.ZipUtils;
import com.android.apksig.util.DataSource;
import com.android.apksig.zip.ZipFormatException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * APK utilities.
 */
public abstract class ApkUtils {

    /**
     * Name of the Android manifest ZIP entry in APKs.
     */
    public static final String ANDROID_MANIFEST_ZIP_ENTRY_NAME = "AndroidManifest.xml";

    /** Name of the SourceStamp certificate hash ZIP entry in APKs. */
    public static final String SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME =
            SourceStampConstants.SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME;

    private ApkUtils() {}

    /**
     * Finds the main ZIP sections of the provided APK.
     *
     * @throws IOException if an I/O error occurred while reading the APK
     * @throws ZipFormatException if the APK is malformed
     */
    public static ZipSections findZipSections(DataSource apk)
            throws IOException, ZipFormatException {
        com.android.apksig.zip.ZipSections zipSections = ApkUtilsLite.findZipSections(apk);
        return new ZipSections(
                zipSections.getZipCentralDirectoryOffset(),
                zipSections.getZipCentralDirectorySizeBytes(),
                zipSections.getZipCentralDirectoryRecordCount(),
                zipSections.getZipEndOfCentralDirectoryOffset(),
                zipSections.getZipEndOfCentralDirectory());
    }

    /**
     * Information about the ZIP sections of an APK.
     */
    public static class ZipSections extends com.android.apksig.zip.ZipSections {
        public ZipSections(
                long centralDirectoryOffset,
                long centralDirectorySizeBytes,
                int centralDirectoryRecordCount,
                long eocdOffset,
                ByteBuffer eocd) {
            super(centralDirectoryOffset, centralDirectorySizeBytes, centralDirectoryRecordCount,
                    eocdOffset, eocd);
        }
    }

    /**
     * Sets the offset of the start of the ZIP Central Directory in the APK's ZIP End of Central
     * Directory record.
     *
     * @param zipEndOfCentralDirectory APK's ZIP End of Central Directory record
     * @param offset offset of the ZIP Central Directory relative to the start of the archive. Must
     *        be between {@code 0} and {@code 2^32 - 1} inclusive.
     */
    public static void setZipEocdCentralDirectoryOffset(
            ByteBuffer zipEndOfCentralDirectory, long offset) {
        ByteBuffer eocd = zipEndOfCentralDirectory.slice();
        eocd.order(ByteOrder.LITTLE_ENDIAN);
        ZipUtils.setZipEocdCentralDirectoryOffset(eocd, offset);
    }

    /**
     * Updates the length of EOCD comment.
     *
     * @param zipEndOfCentralDirectory APK's ZIP End of Central Directory record
     */
    public static void updateZipEocdCommentLen(ByteBuffer zipEndOfCentralDirectory) {
        ByteBuffer eocd = zipEndOfCentralDirectory.slice();
        eocd.order(ByteOrder.LITTLE_ENDIAN);
        ZipUtils.updateZipEocdCommentLen(eocd);
    }

    /**
     * Returns the APK Signing Block of the provided {@code apk}.
     *
     * @throws ApkFormatException if the APK is not a valid ZIP archive
     * @throws IOException if an I/O error occurs
     * @throws ApkSigningBlockNotFoundException if there is no APK Signing Block in the APK
     *
     * @see <a href="https://source.android.com/security/apksigning/v2.html">APK Signature Scheme v2
     * </a>
     */
    public static ApkSigningBlock findApkSigningBlock(DataSource apk)
            throws ApkFormatException, IOException, ApkSigningBlockNotFoundException {
        ApkUtils.ZipSections inputZipSections;
        try {
            inputZipSections = ApkUtils.findZipSections(apk);
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Malformed APK: not a ZIP archive", e);
        }
        return findApkSigningBlock(apk, inputZipSections);
    }

    /**
     * Returns the APK Signing Block of the provided APK.
     *
     * @throws IOException if an I/O error occurs
     * @throws ApkSigningBlockNotFoundException if there is no APK Signing Block in the APK
     *
     * @see <a href="https://source.android.com/security/apksigning/v2.html">APK Signature Scheme v2
     * </a>
     */
    public static ApkSigningBlock findApkSigningBlock(DataSource apk, ZipSections zipSections)
            throws IOException, ApkSigningBlockNotFoundException {
        ApkUtilsLite.ApkSigningBlock apkSigningBlock = ApkUtilsLite.findApkSigningBlock(apk,
                zipSections);
        return new ApkSigningBlock(apkSigningBlock.getStartOffset(), apkSigningBlock.getContents());
    }

    /**
     * Information about the location of the APK Signing Block inside an APK.
     */
    public static class ApkSigningBlock extends ApkUtilsLite.ApkSigningBlock {
        /**
         * Constructs a new {@code ApkSigningBlock}.
         *
         * @param startOffsetInApk start offset (in bytes, relative to start of file) of the APK
         *        Signing Block inside the APK file
         * @param contents contents of the APK Signing Block
         */
        public ApkSigningBlock(long startOffsetInApk, DataSource contents) {
            super(startOffsetInApk, contents);
        }
    }

    /**
     * Returns the contents of the APK's {@code AndroidManifest.xml}.
     *
     * @throws IOException if an I/O error occurs while reading the APK
     * @throws ApkFormatException if the APK is malformed
     */
    public static ByteBuffer getAndroidManifest(DataSource apk)
            throws IOException, ApkFormatException {
        ZipSections zipSections;
        try {
            zipSections = findZipSections(apk);
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Not a valid ZIP archive", e);
        }
        List<CentralDirectoryRecord> cdRecords =
                V1SchemeVerifier.parseZipCentralDirectory(apk, zipSections);
        CentralDirectoryRecord androidManifestCdRecord = null;
        for (CentralDirectoryRecord cdRecord : cdRecords) {
            if (ANDROID_MANIFEST_ZIP_ENTRY_NAME.equals(cdRecord.getName())) {
                androidManifestCdRecord = cdRecord;
                break;
            }
        }
        if (androidManifestCdRecord == null) {
            throw new ApkFormatException("Missing " + ANDROID_MANIFEST_ZIP_ENTRY_NAME);
        }
        DataSource lfhSection = apk.slice(0, zipSections.getZipCentralDirectoryOffset());

        try {
            return ByteBuffer.wrap(
                    LocalFileRecord.getUncompressedData(
                            lfhSection, androidManifestCdRecord, lfhSection.size()));
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Failed to read " + ANDROID_MANIFEST_ZIP_ENTRY_NAME, e);
        }
    }

    /**
     * Android resource ID of the {@code android:minSdkVersion} attribute in AndroidManifest.xml.
     */
    private static final int MIN_SDK_VERSION_ATTR_ID = 0x0101020c;

    /**
     * Android resource ID of the {@code android:debuggable} attribute in AndroidManifest.xml.
     */
    private static final int DEBUGGABLE_ATTR_ID = 0x0101000f;

    /**
     * Android resource ID of the {@code android:targetSandboxVersion} attribute in
     * AndroidManifest.xml.
     */
    private static final int TARGET_SANDBOX_VERSION_ATTR_ID = 0x0101054c;

    /**
     * Android resource ID of the {@code android:targetSdkVersion} attribute in
     * AndroidManifest.xml.
     */
    private static final int TARGET_SDK_VERSION_ATTR_ID = 0x01010270;
    private static final String USES_SDK_ELEMENT_TAG = "uses-sdk";

    /**
     * Android resource ID of the {@code android:versionCode} attribute in AndroidManifest.xml.
     */
    private static final int VERSION_CODE_ATTR_ID = 0x0101021b;
    private static final String MANIFEST_ELEMENT_TAG = "manifest";

    /**
     * Android resource ID of the {@code android:versionCodeMajor} attribute in AndroidManifest.xml.
     */
    private static final int VERSION_CODE_MAJOR_ATTR_ID = 0x01010576;

    /**
     * Returns the lowest Android platform version (API Level) supported by an APK with the
     * provided {@code AndroidManifest.xml}.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *        resource format
     *
     * @throws MinSdkVersionException if an error occurred while determining the API Level
     */
    public static int getMinSdkVersionFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents) throws MinSdkVersionException {
        // IMPLEMENTATION NOTE: Minimum supported Android platform version number is declared using
        // uses-sdk elements which are children of the top-level manifest element. uses-sdk element
        // declares the minimum supported platform version using the android:minSdkVersion attribute
        // whose default value is 1.
        // For each encountered uses-sdk element, the Android runtime checks that its minSdkVersion
        // is not higher than the runtime's API Level and rejects APKs if it is higher. Thus, the
        // effective minSdkVersion value is the maximum over the encountered minSdkVersion values.

        try {
            // If no uses-sdk elements are encountered, Android accepts the APK. We treat this
            // scenario as though the minimum supported API Level is 1.
            int result = 1;

            AndroidBinXmlParser parser = new AndroidBinXmlParser(androidManifestContents);
            int eventType = parser.getEventType();
            while (eventType != AndroidBinXmlParser.EVENT_END_DOCUMENT) {
                if ((eventType == AndroidBinXmlParser.EVENT_START_ELEMENT)
                        && (parser.getDepth() == 2)
                        && ("uses-sdk".equals(parser.getName()))
                        && (parser.getNamespace().isEmpty())) {
                    // In each uses-sdk element, minSdkVersion defaults to 1
                    int minSdkVersion = 1;
                    for (int i = 0; i < parser.getAttributeCount(); i++) {
                        if (parser.getAttributeNameResourceId(i) == MIN_SDK_VERSION_ATTR_ID) {
                            int valueType = parser.getAttributeValueType(i);
                            switch (valueType) {
                                case AndroidBinXmlParser.VALUE_TYPE_INT:
                                    minSdkVersion = parser.getAttributeIntValue(i);
                                    break;
                                case AndroidBinXmlParser.VALUE_TYPE_STRING:
                                    minSdkVersion =
                                            getMinSdkVersionForCodename(
                                                    parser.getAttributeStringValue(i));
                                    break;
                                default:
                                    throw new MinSdkVersionException(
                                            "Unable to determine APK's minimum supported Android"
                                                    + ": unsupported value type in "
                                                    + ANDROID_MANIFEST_ZIP_ENTRY_NAME + "'s"
                                                    + " minSdkVersion"
                                                    + ". Only integer values supported.");
                            }
                            break;
                        }
                    }
                    result = Math.max(result, minSdkVersion);
                }
                eventType = parser.next();
            }

            return result;
        } catch (AndroidBinXmlParser.XmlParserException e) {
            throw new MinSdkVersionException(
                    "Unable to determine APK's minimum supported Android platform version"
                            + ": malformed binary resource: " + ANDROID_MANIFEST_ZIP_ENTRY_NAME,
                    e);
        }
    }

    private static class CodenamesLazyInitializer {

        /**
         * List of platform codename (first letter of) to API Level mappings. The list must be
         * sorted by the first letter. For codenames not in the list, the assumption is that the API
         * Level is incremented by one for every increase in the codename's first letter.
         */
        @SuppressWarnings({"rawtypes", "unchecked"})
        private static final Pair<Character, Integer>[] SORTED_CODENAMES_FIRST_CHAR_TO_API_LEVEL =
                new Pair[] {
            Pair.of('C', 2),
            Pair.of('D', 3),
            Pair.of('E', 4),
            Pair.of('F', 7),
            Pair.of('G', 8),
            Pair.of('H', 10),
            Pair.of('I', 13),
            Pair.of('J', 15),
            Pair.of('K', 18),
            Pair.of('L', 20),
            Pair.of('M', 22),
            Pair.of('N', 23),
            Pair.of('O', 25),
        };

        private static final Comparator<Pair<Character, Integer>> CODENAME_FIRST_CHAR_COMPARATOR =
                new ByFirstComparator();

        private static class ByFirstComparator implements Comparator<Pair<Character, Integer>> {
            @Override
            public int compare(Pair<Character, Integer> o1, Pair<Character, Integer> o2) {
                char c1 = o1.getFirst();
                char c2 = o2.getFirst();
                return c1 - c2;
            }
        }
    }

    /**
     * Returns the API Level corresponding to the provided platform codename.
     *
     * <p>This method is pessimistic. It returns a value one lower than the API Level with which the
     * platform is actually released (e.g., 23 for N which was released as API Level 24). This is
     * because new features which first appear in an API Level are not available in the early days
     * of that platform version's existence, when the platform only has a codename. Moreover, this
     * method currently doesn't differentiate between initial and MR releases, meaning API Level
     * returned for MR releases may be more than one lower than the API Level with which the
     * platform version is actually released.
     *
     * @throws CodenameMinSdkVersionException if the {@code codename} is not supported
     */
    static int getMinSdkVersionForCodename(String codename) throws CodenameMinSdkVersionException {
        char firstChar = codename.isEmpty() ? ' ' : codename.charAt(0);
        // Codenames are case-sensitive. Only codenames starting with A-Z are supported for now.
        // We only look at the first letter of the codename as this is the most important letter.
        if ((firstChar >= 'A') && (firstChar <= 'Z')) {
            Pair<Character, Integer>[] sortedCodenamesFirstCharToApiLevel =
                    CodenamesLazyInitializer.SORTED_CODENAMES_FIRST_CHAR_TO_API_LEVEL;
            int searchResult =
                    Arrays.binarySearch(
                            sortedCodenamesFirstCharToApiLevel,
                            Pair.of(firstChar, null), // second element of the pair is ignored here
                            CodenamesLazyInitializer.CODENAME_FIRST_CHAR_COMPARATOR);
            if (searchResult >= 0) {
                // Exact match -- searchResult is the index of the matching element
                return sortedCodenamesFirstCharToApiLevel[searchResult].getSecond();
            }
            // Not an exact match -- searchResult is negative and is -(insertion index) - 1.
            // The element at insertionIndex - 1 (if present) is smaller than firstChar and the
            // element at insertionIndex (if present) is greater than firstChar.
            int insertionIndex = -1 - searchResult; // insertionIndex is in [0; array length]
            if (insertionIndex == 0) {
                // 'A' or 'B' -- never released to public
                return 1;
            } else {
                // The element at insertionIndex - 1 is the newest older codename.
                // API Level bumped by at least 1 for every change in the first letter of codename
                Pair<Character, Integer> newestOlderCodenameMapping =
                        sortedCodenamesFirstCharToApiLevel[insertionIndex - 1];
                char newestOlderCodenameFirstChar = newestOlderCodenameMapping.getFirst();
                int newestOlderCodenameApiLevel = newestOlderCodenameMapping.getSecond();
                return newestOlderCodenameApiLevel + (firstChar - newestOlderCodenameFirstChar);
            }
        }

        throw new CodenameMinSdkVersionException(
                "Unable to determine APK's minimum supported Android platform version"
                        + " : Unsupported codename in " + ANDROID_MANIFEST_ZIP_ENTRY_NAME
                        + "'s minSdkVersion: \"" + codename + "\"",
                codename);
    }

    /**
     * Returns {@code true} if the APK is debuggable according to its {@code AndroidManifest.xml}.
     * See the {@code android:debuggable} attribute of the {@code application} element.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *        resource format
     *
     * @throws ApkFormatException if the manifest is malformed
     */
    public static boolean getDebuggableFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents) throws ApkFormatException {
        // IMPLEMENTATION NOTE: Whether the package is debuggable is declared using the first
        // "application" element which is a child of the top-level manifest element. The debuggable
        // attribute of this application element is coerced to a boolean value. If there is no
        // application element or if it doesn't declare the debuggable attribute, the package is
        // considered not debuggable.

        try {
            AndroidBinXmlParser parser = new AndroidBinXmlParser(androidManifestContents);
            int eventType = parser.getEventType();
            while (eventType != AndroidBinXmlParser.EVENT_END_DOCUMENT) {
                if ((eventType == AndroidBinXmlParser.EVENT_START_ELEMENT)
                        && (parser.getDepth() == 2)
                        && ("application".equals(parser.getName()))
                        && (parser.getNamespace().isEmpty())) {
                    for (int i = 0; i < parser.getAttributeCount(); i++) {
                        if (parser.getAttributeNameResourceId(i) == DEBUGGABLE_ATTR_ID) {
                            int valueType = parser.getAttributeValueType(i);
                            switch (valueType) {
                                case AndroidBinXmlParser.VALUE_TYPE_BOOLEAN:
                                case AndroidBinXmlParser.VALUE_TYPE_STRING:
                                case AndroidBinXmlParser.VALUE_TYPE_INT:
                                    String value = parser.getAttributeStringValue(i);
                                    return ("true".equals(value))
                                            || ("TRUE".equals(value))
                                            || ("1".equals(value));
                                case AndroidBinXmlParser.VALUE_TYPE_REFERENCE:
                                    // References to resources are not supported on purpose. The
                                    // reason is that the resolved value depends on the resource
                                    // configuration (e.g, MNC/MCC, locale, screen density) used
                                    // at resolution time. As a result, the same APK may appear as
                                    // debuggable in one situation and as non-debuggable in another
                                    // situation. Such APKs may put users at risk.
                                    throw new ApkFormatException(
                                            "Unable to determine whether APK is debuggable"
                                                    + ": " + ANDROID_MANIFEST_ZIP_ENTRY_NAME + "'s"
                                                    + " android:debuggable attribute references a"
                                                    + " resource. References are not supported for"
                                                    + " security reasons. Only constant boolean,"
                                                    + " string and int values are supported.");
                                default:
                                    throw new ApkFormatException(
                                            "Unable to determine whether APK is debuggable"
                                                    + ": " + ANDROID_MANIFEST_ZIP_ENTRY_NAME + "'s"
                                                    + " android:debuggable attribute uses"
                                                    + " unsupported value type. Only boolean,"
                                                    + " string and int values are supported.");
                            }
                        }
                    }
                    // This application element does not declare the debuggable attribute
                    return false;
                }
                eventType = parser.next();
            }

            // No application element found
            return false;
        } catch (AndroidBinXmlParser.XmlParserException e) {
            throw new ApkFormatException(
                    "Unable to determine whether APK is debuggable: malformed binary resource: "
                            + ANDROID_MANIFEST_ZIP_ENTRY_NAME,
                    e);
        }
    }

    /**
     * Returns the package name of the APK according to its {@code AndroidManifest.xml} or
     * {@code null} if package name is not declared. See the {@code package} attribute of the
     * {@code manifest} element.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *        resource format
     *
     * @throws ApkFormatException if the manifest is malformed
     */
    public static String getPackageNameFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents) throws ApkFormatException {
        // IMPLEMENTATION NOTE: Package name is declared as the "package" attribute of the top-level
        // manifest element. Interestingly, as opposed to most other attributes, Android Package
        // Manager looks up this attribute by its name rather than by its resource ID.

        try {
            AndroidBinXmlParser parser = new AndroidBinXmlParser(androidManifestContents);
            int eventType = parser.getEventType();
            while (eventType != AndroidBinXmlParser.EVENT_END_DOCUMENT) {
                if ((eventType == AndroidBinXmlParser.EVENT_START_ELEMENT)
                        && (parser.getDepth() == 1)
                        && ("manifest".equals(parser.getName()))
                        && (parser.getNamespace().isEmpty())) {
                    for (int i = 0; i < parser.getAttributeCount(); i++) {
                        if ("package".equals(parser.getAttributeName(i))
                                && (parser.getNamespace().isEmpty())) {
                            return parser.getAttributeStringValue(i);
                        }
                    }
                    // No "package" attribute found
                    return null;
                }
                eventType = parser.next();
            }

            // No manifest element found
            return null;
        } catch (AndroidBinXmlParser.XmlParserException e) {
            throw new ApkFormatException(
                    "Unable to determine APK package name: malformed binary resource: "
                            + ANDROID_MANIFEST_ZIP_ENTRY_NAME,
                    e);
        }
    }

    /**
     * Returns the security sandbox version targeted by an APK with the provided
     * {@code AndroidManifest.xml}.
     *
     * <p>If the security sandbox version is not specified in the manifest a default value of 1 is
     * returned.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *                                resource format
     */
    public static int getTargetSandboxVersionFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents) {
        try {
            return getAttributeValueFromBinaryAndroidManifest(androidManifestContents,
                    MANIFEST_ELEMENT_TAG, TARGET_SANDBOX_VERSION_ATTR_ID);
        } catch (ApkFormatException e) {
            // An ApkFormatException indicates the target sandbox is not specified in the manifest;
            // return a default value of 1.
            return 1;
        }
    }

    /**
     * Returns the SDK version targeted by an APK with the provided {@code AndroidManifest.xml}.
     *
     * <p>If the targetSdkVersion is not specified the minimumSdkVersion is returned. If neither
     * value is specified then a value of 1 is returned.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *                                resource format
     */
    public static int getTargetSdkVersionFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents) {
        // If the targetSdkVersion is not specified then the platform will use the value of the
        // minSdkVersion; if neither is specified then the platform will use a value of 1.
        int minSdkVersion = 1;
        try {
            return getAttributeValueFromBinaryAndroidManifest(androidManifestContents,
                    USES_SDK_ELEMENT_TAG, TARGET_SDK_VERSION_ATTR_ID);
        } catch (ApkFormatException e) {
            // Expected if the APK does not contain a targetSdkVersion attribute or the uses-sdk
            // element is not specified at all.
        }
        androidManifestContents.rewind();
        try {
            minSdkVersion = getMinSdkVersionFromBinaryAndroidManifest(androidManifestContents);
        } catch (ApkFormatException e) {
            // Similar to above, expected if the APK does not contain a minSdkVersion attribute, or
            // the uses-sdk element is not specified at all.
        }
        return minSdkVersion;
    }

    /**
     * Returns the versionCode of the APK according to its {@code AndroidManifest.xml}.
     *
     * <p>If the versionCode is not specified in the {@code AndroidManifest.xml} or is not a valid
     * integer an ApkFormatException is thrown.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *                                resource format
     * @throws ApkFormatException if an error occurred while determining the versionCode, or if the
     *                            versionCode attribute value is not available.
     */
    public static int getVersionCodeFromBinaryAndroidManifest(ByteBuffer androidManifestContents)
            throws ApkFormatException {
        return getAttributeValueFromBinaryAndroidManifest(androidManifestContents,
                MANIFEST_ELEMENT_TAG, VERSION_CODE_ATTR_ID);
    }

    /**
     * Returns the versionCode and versionCodeMajor of the APK according to its {@code
     * AndroidManifest.xml} combined together as a single long value.
     *
     * <p>The versionCodeMajor is placed in the upper 32 bits, and the versionCode is in the lower
     * 32 bits. If the versionCodeMajor is not specified then the versionCode is returned.
     *
     * @param androidManifestContents contents of {@code AndroidManifest.xml} in binary Android
     *                                resource format
     * @throws ApkFormatException if an error occurred while determining the version, or if the
     *                            versionCode attribute value is not available.
     */
    public static long getLongVersionCodeFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents) throws ApkFormatException {
        // If the versionCode is not found then allow the ApkFormatException to be thrown to notify
        // the caller that the versionCode is not available.
        int versionCode = getVersionCodeFromBinaryAndroidManifest(androidManifestContents);
        long versionCodeMajor = 0;
        try {
            androidManifestContents.rewind();
            versionCodeMajor = getAttributeValueFromBinaryAndroidManifest(androidManifestContents,
                    MANIFEST_ELEMENT_TAG, VERSION_CODE_MAJOR_ATTR_ID);
        } catch (ApkFormatException e) {
            // This is expected if the versionCodeMajor has not been defined for the APK; in this
            // case the return value is just the versionCode.
        }
        return (versionCodeMajor << 32) | versionCode;
    }

    /**
     * Returns the integer value of the requested {@code attributeId} in the specified {@code
     * elementName} from the provided {@code androidManifestContents} in binary Android resource
     * format.
     *
     * @throws ApkFormatException if an error occurred while attempting to obtain the attribute, or
     *                            if the requested attribute is not found.
     */
    private static int getAttributeValueFromBinaryAndroidManifest(
            ByteBuffer androidManifestContents, String elementName, int attributeId)
            throws ApkFormatException {
        if (elementName == null) {
            throw new NullPointerException("elementName cannot be null");
        }
        try {
            AndroidBinXmlParser parser = new AndroidBinXmlParser(androidManifestContents);
            int eventType = parser.getEventType();
            while (eventType != AndroidBinXmlParser.EVENT_END_DOCUMENT) {
                if ((eventType == AndroidBinXmlParser.EVENT_START_ELEMENT)
                        && (elementName.equals(parser.getName()))) {
                    for (int i = 0; i < parser.getAttributeCount(); i++) {
                        if (parser.getAttributeNameResourceId(i) == attributeId) {
                            int valueType = parser.getAttributeValueType(i);
                            switch (valueType) {
                                case AndroidBinXmlParser.VALUE_TYPE_INT:
                                case AndroidBinXmlParser.VALUE_TYPE_STRING:
                                    return parser.getAttributeIntValue(i);
                                default:
                                    throw new ApkFormatException(
                                            "Unsupported value type, " + valueType
                                                    + ", for attribute " + String.format("0x%08X",
                                                    attributeId) + " under element " + elementName);

                            }
                        }
                    }
                }
                eventType = parser.next();
            }
            throw new ApkFormatException(
                    "Failed to determine APK's " + elementName + " attribute "
                            + String.format("0x%08X", attributeId) + " value");
        } catch (AndroidBinXmlParser.XmlParserException e) {
            throw new ApkFormatException(
                    "Unable to determine value for attribute " + String.format("0x%08X",
                            attributeId) + " under element " + elementName
                            + "; malformed binary resource: " + ANDROID_MANIFEST_ZIP_ENTRY_NAME, e);
        }
    }

    public static byte[] computeSha256DigestBytes(byte[] data) {
        return ApkUtilsLite.computeSha256DigestBytes(data);
    }
}
