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

package com.android.apksig;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.RunnablesExecutor;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.util.List;
import java.util.Set;

/**
 * APK signing logic which is independent of how input and output APKs are stored, parsed, and
 * generated.
 *
 * <p><h3>Operating Model</h3>
 *
 * The abstract operating model is that there is an input APK which is being signed, thus producing
 * an output APK. In reality, there may be just an output APK being built from scratch, or the input
 * APK and the output APK may be the same file. Because this engine does not deal with reading and
 * writing files, it can handle all of these scenarios.
 *
 * <p>The engine is stateful and thus cannot be used for signing multiple APKs. However, once
 * the engine signed an APK, the engine can be used to re-sign the APK after it has been modified.
 * This may be more efficient than signing the APK using a new instance of the engine. See
 * <a href="#incremental">Incremental Operation</a>.
 *
 * <p>In the engine's operating model, a signed APK is produced as follows.
 * <ol>
 * <li>JAR entries to be signed are output,</li>
 * <li>JAR archive is signed using JAR signing, thus adding the so-called v1 signature to the
 *     output,</li>
 * <li>JAR archive is signed using APK Signature Scheme v2, thus adding the so-called v2 signature
 *     to the output.</li>
 * </ol>
 *
 * <p>The input APK may contain JAR entries which, depending on the engine's configuration, may or
 * may not be output (e.g., existing signatures may need to be preserved or stripped) or which the
 * engine will overwrite as part of signing. The engine thus offers {@link #inputJarEntry(String)}
 * which tells the client whether the input JAR entry needs to be output. This avoids the need for
 * the client to hard-code the aspects of APK signing which determine which parts of input must be
 * ignored. Similarly, the engine offers {@link #inputApkSigningBlock(DataSource)} to help the
 * client avoid dealing with preserving or stripping APK Signature Scheme v2 signature of the input
 * APK.
 *
 * <p>To use the engine to sign an input APK (or a collection of JAR entries), follow these
 * steps:
 * <ol>
 * <li>Obtain a new instance of the engine -- engine instances are stateful and thus cannot be used
 *     for signing multiple APKs.</li>
 * <li>Locate the input APK's APK Signing Block and provide it to
 *     {@link #inputApkSigningBlock(DataSource)}.</li>
 * <li>For each JAR entry in the input APK, invoke {@link #inputJarEntry(String)} to determine
 *     whether this entry should be output. The engine may request to inspect the entry.</li>
 * <li>For each output JAR entry, invoke {@link #outputJarEntry(String)} which may request to
 *     inspect the entry.</li>
 * <li>Once all JAR entries have been output, invoke {@link #outputJarEntries()} which may request
 *     that additional JAR entries are output. These entries comprise the output APK's JAR
 *     signature.</li>
 * <li>Locate the ZIP Central Directory and ZIP End of Central Directory sections in the output and
 *     invoke {@link #outputZipSections2(DataSource, DataSource, DataSource)} which may request that
 *     an APK Signature Block is inserted before the ZIP Central Directory. The block contains the
 *     output APK's APK Signature Scheme v2 signature.</li>
 * <li>Invoke {@link #outputDone()} to signal that the APK was output in full. The engine will
 *     confirm that the output APK is signed.</li>
 * <li>Invoke {@link #close()} to signal that the engine will no longer be used. This lets the
 *     engine free any resources it no longer needs.
 * </ol>
 *
 * <p>Some invocations of the engine may provide the client with a task to perform. The client is
 * expected to perform all requested tasks before proceeding to the next stage of signing. See
 * documentation of each method about the deadlines for performing the tasks requested by the
 * method.
 *
 * <p><h3 id="incremental">Incremental Operation</h3></a>
 *
 * The engine supports incremental operation where a signed APK is produced, then modified and
 * re-signed. This may be useful for IDEs, where an app is frequently re-signed after small changes
 * by the developer. Re-signing may be more efficient than signing from scratch.
 *
 * <p>To use the engine in incremental mode, keep notifying the engine of changes to the APK through
 * {@link #inputApkSigningBlock(DataSource)}, {@link #inputJarEntry(String)},
 * {@link #inputJarEntryRemoved(String)}, {@link #outputJarEntry(String)},
 * and {@link #outputJarEntryRemoved(String)}, perform the tasks requested by the engine through
 * these methods, and, when a new signed APK is desired, run through steps 5 onwards to re-sign the
 * APK.
 *
 * <p><h3>Output-only Operation</h3>
 *
 * The engine's abstract operating model consists of an input APK and an output APK. However, it is
 * possible to use the engine in output-only mode where the engine's {@code input...} methods are
 * not invoked. In this mode, the engine has less control over output because it cannot request that
 * some JAR entries are not output. Nevertheless, the engine will attempt to make the output APK
 * signed and will report an error if cannot do so.
 *
 * @see <a href="https://source.android.com/security/apksigning/index.html">Application Signing</a>
 */
public interface ApkSignerEngine extends Closeable {

    default void setExecutor(RunnablesExecutor executor) {
        throw new UnsupportedOperationException("setExecutor method is not implemented");
    }

    /**
     * Initializes the signer engine with the data already present in the apk (if any). There
     * might already be data that can be reused if the entries has not been changed.
     *
     * @param manifestBytes
     * @param entryNames
     * @return set of entry names which were processed by the engine during the initialization, a
     *         subset of entryNames
     */
    default Set<String> initWith(byte[] manifestBytes, Set<String> entryNames) {
        throw new UnsupportedOperationException("initWith method is not implemented");
    }

    /**
     * Indicates to this engine that the input APK contains the provided APK Signing Block. The
     * block may contain signatures of the input APK, such as APK Signature Scheme v2 signatures.
     *
     * @param apkSigningBlock APK signing block of the input APK. The provided data source is
     *        guaranteed to not be used by the engine after this method terminates.
     *
     * @throws IOException if an I/O error occurs while reading the APK Signing Block
     * @throws ApkFormatException if the APK Signing Block is malformed
     * @throws IllegalStateException if this engine is closed
     */
    void inputApkSigningBlock(DataSource apkSigningBlock)
            throws IOException, ApkFormatException, IllegalStateException;

    /**
     * Indicates to this engine that the specified JAR entry was encountered in the input APK.
     *
     * <p>When an input entry is updated/changed, it's OK to not invoke
     * {@link #inputJarEntryRemoved(String)} before invoking this method.
     *
     * @return instructions about how to proceed with this entry
     *
     * @throws IllegalStateException if this engine is closed
     */
    InputJarEntryInstructions inputJarEntry(String entryName) throws IllegalStateException;

    /**
     * Indicates to this engine that the specified JAR entry was output.
     *
     * <p>It is unnecessary to invoke this method for entries added to output by this engine (e.g.,
     * requested by {@link #outputJarEntries()}) provided the entries were output with exactly the
     * data requested by the engine.
     *
     * <p>When an already output entry is updated/changed, it's OK to not invoke
     * {@link #outputJarEntryRemoved(String)} before invoking this method.
     *
     * @return request to inspect the entry or {@code null} if the engine does not need to inspect
     *         the entry. The request must be fulfilled before {@link #outputJarEntries()} is
     *         invoked.
     *
     * @throws IllegalStateException if this engine is closed
     */
    InspectJarEntryRequest outputJarEntry(String entryName) throws IllegalStateException;

    /**
     * Indicates to this engine that the specified JAR entry was removed from the input. It's safe
     * to invoke this for entries for which {@link #inputJarEntry(String)} hasn't been invoked.
     *
     * @return output policy of this JAR entry. The policy indicates how this input entry affects
     *         the output APK. The client of this engine should use this information to determine
     *         how the removal of this input APK's JAR entry affects the output APK.
     *
     * @throws IllegalStateException if this engine is closed
     */
    InputJarEntryInstructions.OutputPolicy inputJarEntryRemoved(String entryName)
            throws IllegalStateException;

    /**
     * Indicates to this engine that the specified JAR entry was removed from the output. It's safe
     * to invoke this for entries for which {@link #outputJarEntry(String)} hasn't been invoked.
     *
     * @throws IllegalStateException if this engine is closed
     */
    void outputJarEntryRemoved(String entryName) throws IllegalStateException;

    /**
     * Indicates to this engine that all JAR entries have been output.
     *
     * @return request to add JAR signature to the output or {@code null} if there is no need to add
     *         a JAR signature. The request will contain additional JAR entries to be output. The
     *         request must be fulfilled before
     *         {@link #outputZipSections2(DataSource, DataSource, DataSource)} is invoked.
     *
     * @throws ApkFormatException if the APK is malformed in a way which is preventing this engine
     *         from producing a valid signature. For example, if the engine uses the provided
     *         {@code META-INF/MANIFEST.MF} as a template and the file is malformed.
     * @throws NoSuchAlgorithmException if a signature could not be generated because a required
     *         cryptographic algorithm implementation is missing
     * @throws InvalidKeyException if a signature could not be generated because a signing key is
     *         not suitable for generating the signature
     * @throws SignatureException if an error occurred while generating a signature
     * @throws IllegalStateException if there are unfulfilled requests, such as to inspect some JAR
     *         entries, or if the engine is closed
     */
    OutputJarSignatureRequest outputJarEntries()
            throws ApkFormatException, NoSuchAlgorithmException, InvalidKeyException,
            SignatureException, IllegalStateException;

    /**
     * Indicates to this engine that the ZIP sections comprising the output APK have been output.
     *
     * <p>The provided data sources are guaranteed to not be used by the engine after this method
     * terminates.
     *
     * @deprecated This is now superseded by {@link #outputZipSections2(DataSource, DataSource,
     * DataSource)}.
     *
     * @param zipEntries the section of ZIP archive containing Local File Header records and data of
     *        the ZIP entries. In a well-formed archive, this section starts at the start of the
     *        archive and extends all the way to the ZIP Central Directory.
     * @param zipCentralDirectory ZIP Central Directory section
     * @param zipEocd ZIP End of Central Directory (EoCD) record
     *
     * @return request to add an APK Signing Block to the output or {@code null} if the output must
     *         not contain an APK Signing Block. The request must be fulfilled before
     *         {@link #outputDone()} is invoked.
     *
     * @throws IOException if an I/O error occurs while reading the provided ZIP sections
     * @throws ApkFormatException if the provided APK is malformed in a way which prevents this
     *         engine from producing a valid signature. For example, if the APK Signing Block
     *         provided to the engine is malformed.
     * @throws NoSuchAlgorithmException if a signature could not be generated because a required
     *         cryptographic algorithm implementation is missing
     * @throws InvalidKeyException if a signature could not be generated because a signing key is
     *         not suitable for generating the signature
     * @throws SignatureException if an error occurred while generating a signature
     * @throws IllegalStateException if there are unfulfilled requests, such as to inspect some JAR
     *         entries or to output JAR signature, or if the engine is closed
     */
    @Deprecated
    OutputApkSigningBlockRequest outputZipSections(
            DataSource zipEntries,
            DataSource zipCentralDirectory,
            DataSource zipEocd)
            throws IOException, ApkFormatException, NoSuchAlgorithmException,
            InvalidKeyException, SignatureException, IllegalStateException;

    /**
     * Indicates to this engine that the ZIP sections comprising the output APK have been output.
     *
     * <p>The provided data sources are guaranteed to not be used by the engine after this method
     * terminates.
     *
     * @param zipEntries the section of ZIP archive containing Local File Header records and data of
     *        the ZIP entries. In a well-formed archive, this section starts at the start of the
     *        archive and extends all the way to the ZIP Central Directory.
     * @param zipCentralDirectory ZIP Central Directory section
     * @param zipEocd ZIP End of Central Directory (EoCD) record
     *
     * @return request to add an APK Signing Block to the output or {@code null} if the output must
     *         not contain an APK Signing Block. The request must be fulfilled before
     *         {@link #outputDone()} is invoked.
     *
     * @throws IOException if an I/O error occurs while reading the provided ZIP sections
     * @throws ApkFormatException if the provided APK is malformed in a way which prevents this
     *         engine from producing a valid signature. For example, if the APK Signing Block
     *         provided to the engine is malformed.
     * @throws NoSuchAlgorithmException if a signature could not be generated because a required
     *         cryptographic algorithm implementation is missing
     * @throws InvalidKeyException if a signature could not be generated because a signing key is
     *         not suitable for generating the signature
     * @throws SignatureException if an error occurred while generating a signature
     * @throws IllegalStateException if there are unfulfilled requests, such as to inspect some JAR
     *         entries or to output JAR signature, or if the engine is closed
     */
    OutputApkSigningBlockRequest2 outputZipSections2(
            DataSource zipEntries,
            DataSource zipCentralDirectory,
            DataSource zipEocd)
            throws IOException, ApkFormatException, NoSuchAlgorithmException,
            InvalidKeyException, SignatureException, IllegalStateException;

    /**
     * Indicates to this engine that the signed APK was output.
     *
     * <p>This does not change the output APK. The method helps the client confirm that the current
     * output is signed.
     *
     * @throws IllegalStateException if there are unfulfilled requests, such as to inspect some JAR
     *         entries or to output signatures, or if the engine is closed
     */
    void outputDone() throws IllegalStateException;

    /**
     * Generates a V4 signature proto and write to output file.
     *
     * @param data Input data to calculate a verity hash tree and hash root
     * @param outputFile To store the serialized V4 Signature.
     * @param ignoreFailures Whether any failures will be silently ignored.
     * @throws InvalidKeyException if a signature could not be generated because a signing key is
     *         not suitable for generating the signature
     * @throws NoSuchAlgorithmException if a signature could not be generated because a required
     *         cryptographic algorithm implementation is missing
     * @throws SignatureException if an error occurred while generating a signature
     * @throws IOException if protobuf fails to be serialized and written to file
     */
    void signV4(DataSource data, File outputFile, boolean ignoreFailures)
            throws InvalidKeyException, NoSuchAlgorithmException, SignatureException, IOException;

    /**
     * Checks if the signing configuration provided to the engine is capable of creating a
     * SourceStamp.
     */
    default boolean isEligibleForSourceStamp() {
        return false;
    }

    /** Generates the digest of the certificate used to sign the source stamp. */
    default byte[] generateSourceStampCertificateDigest() throws SignatureException {
        return new byte[0];
    }

    /**
     * Indicates to this engine that it will no longer be used. Invoking this on an already closed
     * engine is OK.
     *
     * <p>This does not change the output APK. For example, if the output APK is not yet fully
     * signed, it will remain so after this method terminates.
     */
    @Override
    void close();

    /**
     * Instructions about how to handle an input APK's JAR entry.
     *
     * <p>The instructions indicate whether to output the entry (see {@link #getOutputPolicy()}) and
     * may contain a request to inspect the entry (see {@link #getInspectJarEntryRequest()}), in
     * which case the request must be fulfilled before {@link ApkSignerEngine#outputJarEntries()} is
     * invoked.
     */
    public static class InputJarEntryInstructions {
        private final OutputPolicy mOutputPolicy;
        private final InspectJarEntryRequest mInspectJarEntryRequest;

        /**
         * Constructs a new {@code InputJarEntryInstructions} instance with the provided entry
         * output policy and without a request to inspect the entry.
         */
        public InputJarEntryInstructions(OutputPolicy outputPolicy) {
            this(outputPolicy, null);
        }

        /**
         * Constructs a new {@code InputJarEntryInstructions} instance with the provided entry
         * output mode and with the provided request to inspect the entry.
         *
         * @param inspectJarEntryRequest request to inspect the entry or {@code null} if there's no
         *        need to inspect the entry.
         */
        public InputJarEntryInstructions(
                OutputPolicy outputPolicy,
                InspectJarEntryRequest inspectJarEntryRequest) {
            mOutputPolicy = outputPolicy;
            mInspectJarEntryRequest = inspectJarEntryRequest;
        }

        /**
         * Returns the output policy for this entry.
         */
        public OutputPolicy getOutputPolicy() {
            return mOutputPolicy;
        }

        /**
         * Returns the request to inspect the JAR entry or {@code null} if there is no need to
         * inspect the entry.
         */
        public InspectJarEntryRequest getInspectJarEntryRequest() {
            return mInspectJarEntryRequest;
        }

        /**
         * Output policy for an input APK's JAR entry.
         */
        public static enum OutputPolicy {
            /** Entry must not be output. */
            SKIP,

            /** Entry should be output. */
            OUTPUT,

            /** Entry will be output by the engine. The client can thus ignore this input entry. */
            OUTPUT_BY_ENGINE,
        }
    }

    /**
     * Request to inspect the specified JAR entry.
     *
     * <p>The entry's uncompressed data must be provided to the data sink returned by
     * {@link #getDataSink()}. Once the entry's data has been provided to the sink, {@link #done()}
     * must be invoked.
     */
    interface InspectJarEntryRequest {

        /**
         * Returns the data sink into which the entry's uncompressed data should be sent.
         */
        DataSink getDataSink();

        /**
         * Indicates that entry's data has been provided in full.
         */
        void done();

        /**
         * Returns the name of the JAR entry.
         */
        String getEntryName();
    }

    /**
     * Request to add JAR signature (aka v1 signature) to the output APK.
     *
     * <p>Entries listed in {@link #getAdditionalJarEntries()} must be added to the output APK after
     * which {@link #done()} must be invoked.
     */
    interface OutputJarSignatureRequest {

        /**
         * Returns JAR entries that must be added to the output APK.
         */
        List<JarEntry> getAdditionalJarEntries();

        /**
         * Indicates that the JAR entries contained in this request were added to the output APK.
         */
        void done();

        /**
         * JAR entry.
         */
        public static class JarEntry {
            private final String mName;
            private final byte[] mData;

            /**
             * Constructs a new {@code JarEntry} with the provided name and data.
             *
             * @param data uncompressed data of the entry. Changes to this array will not be
             *        reflected in {@link #getData()}.
             */
            public JarEntry(String name, byte[] data) {
                mName = name;
                mData = data.clone();
            }

            /**
             * Returns the name of this ZIP entry.
             */
            public String getName() {
                return mName;
            }

            /**
             * Returns the uncompressed data of this JAR entry.
             */
            public byte[] getData() {
                return mData.clone();
            }
        }
    }

    /**
     * Request to add the specified APK Signing Block to the output APK. APK Signature Scheme v2
     * signature(s) of the APK are contained in this block.
     *
     * <p>The APK Signing Block returned by {@link #getApkSigningBlock()} must be placed into the
     * output APK such that the block is immediately before the ZIP Central Directory, the offset of
     * ZIP Central Directory in the ZIP End of Central Directory record must be adjusted
     * accordingly, and then {@link #done()} must be invoked.
     *
     * <p>If the output contains an APK Signing Block, that block must be replaced by the block
     * contained in this request.
     *
     * @deprecated This is now superseded by {@link OutputApkSigningBlockRequest2}.
     */
    @Deprecated
    interface OutputApkSigningBlockRequest {

        /**
         * Returns the APK Signing Block.
         */
        byte[] getApkSigningBlock();

        /**
         * Indicates that the APK Signing Block was output as requested.
         */
        void done();
    }

    /**
     * Request to add the specified APK Signing Block to the output APK. APK Signature Scheme v2
     * signature(s) of the APK are contained in this block.
     *
     * <p>The APK Signing Block returned by {@link #getApkSigningBlock()} must be placed into the
     * output APK such that the block is immediately before the ZIP Central Directory. Immediately
     * before the APK Signing Block must be padding consists of the number of 0x00 bytes returned by
     * {@link #getPaddingSizeBeforeApkSigningBlock()}. The offset of ZIP Central Directory in the
     * ZIP End of Central Directory record must be adjusted accordingly, and then {@link #done()}
     * must be invoked.
     *
     * <p>If the output contains an APK Signing Block, that block must be replaced by the block
     * contained in this request.
     */
    interface OutputApkSigningBlockRequest2 {
        /**
         * Returns the APK Signing Block.
         */
        byte[] getApkSigningBlock();

        /**
         * Indicates that the APK Signing Block was output as requested.
         */
        void done();

        /**
         * Returns the number of 0x00 bytes the caller must place immediately before APK Signing
         * Block.
         */
        int getPaddingSizeBeforeApkSigningBlock();
    }
}
