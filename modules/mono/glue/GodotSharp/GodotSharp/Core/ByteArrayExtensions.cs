using Godot.NativeInterop;


namespace Godot
{
    /// <summary>
    /// Extension methods to manipulate byte arrays.
    /// </summary>
    public static class ByteArrayExtensions
    {
        /// <summary>
        ///	Returns a new byte array with the data compressed.
        /// </summary>
        /// <param name="instance">The byte array to compress.</param>
        /// <param name="compressionMode">The compression mode, one of <see cref="FileAccess.CompressionMode"/></param>
        /// <returns>The compressed byte array.</returns>
        public static byte[] Compress(this byte[] instance, FileAccess.CompressionMode compressionMode)
        {
            using godot_packed_byte_array src = Marshaling.ConvertSystemArrayToNativePackedByteArray(instance);
            NativeFuncs.godotsharp_packed_byte_array_compress(src, (int)compressionMode, out var ret);
            using (ret)
                return Marshaling.ConvertNativePackedByteArrayToSystemArray(ret);
        }

        /// <summary>
        /// Returns a new byte array with the data decompressed.
        /// <para>Note: Decompression is not guaranteed to work with data not compressed by Godot, for example if data compressed with the deflate compression mode lacks a checksum or header.</para>
        /// </summary>
        /// <param name="instance">The byte array to decompress.</param>
        /// <param name="bufferSize">The size of the uncompressed data.</param>
        /// <param name="compressionMode">The compression mode, one of <see cref="FileAccess.CompressionMode"/></param>
        /// <returns>The decompressed byte array.</returns>
        public static byte[] Decompress(this byte[] instance, long bufferSize, FileAccess.CompressionMode compressionMode)
        {
            using godot_packed_byte_array src = Marshaling.ConvertSystemArrayToNativePackedByteArray(instance);
            NativeFuncs.godotsharp_packed_byte_array_decompress(src, bufferSize, (int)compressionMode, out var ret);
            using (ret)
                return Marshaling.ConvertNativePackedByteArrayToSystemArray(ret);
        }

        /// <summary>
        ///	Returns a new byte array with the data decompressed. <b>This method only accepts brotli, gzip, and deflate compression modes</b>.
        /// <para>This method is potentially slower than <see cref="Decompress"/>, as it may have to re-allocate its output buffer multiple times while decompressing, whereas <see cref="Decompress"/> knows it's output buffer size from the beginning.</para>
        /// <para>GZIP has a maximal compression ratio of 1032:1, meaning it's very possible for a small compressed payload to decompress to a potentially very large output. To guard against this, you may provide a maximum size this function is allowed to allocate in bytes via [param max_output_size]. Passing -1 will allow for unbounded output. If any positive value is passed, and the decompression exceeds that amount in bytes, then an error will be returned.</para>
        /// <para>Note: Decompression is not guaranteed to work with data not compressed by Godot, for example if data compressed with the deflate compression mode lacks a checksum or header.</para>
        /// </summary>
        /// <param name="instance">The byte array to decompress.</param>
        /// <param name="maxOutputSize">The maximum size this function is allowed to allocate in bytes.</param>
        /// <param name="compressionMode">The compression mode, one of <see cref="FileAccess.CompressionMode"/></param>
        /// <returns>The decompressed byte array.</returns>
        public static byte[] DecompressDynamic(this byte[] instance, long maxOutputSize, FileAccess.CompressionMode compressionMode)
        {
            using godot_packed_byte_array src = Marshaling.ConvertSystemArrayToNativePackedByteArray(instance);
            NativeFuncs.godotsharp_packed_byte_array_decompress_dynamic(src, maxOutputSize, (int)compressionMode, out var ret);
            using (ret)
                return Marshaling.ConvertNativePackedByteArrayToSystemArray(ret);
        }
    }
}
