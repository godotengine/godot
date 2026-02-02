#pragma once

#include <cstddef>
#include <string>

namespace tinyusdz {

// Based on USD' TfFastCompression

class LZ4Compression
{
public:
    /// Return the largest input buffer size that can be compressed with these
    /// functions.  Guaranteed to be at least 200 GB.
    static size_t
    GetMaxInputSize();

    /// Return the largest possible compressed size for the given \p inputSize
    /// in the worst case (input is not compressible).  This is larger than
    /// \p inputSize.  If inputSize is larger than GetMaxInputSize(), return 0.
    static size_t
    GetCompressedBufferSize(size_t inputSize);

    /// Compress \p inputSize bytes in \p input and store the result in
    /// \p compressed.  The \p compressed buffer must point to at least
    /// GetCompressedBufferSize(uncompressedSize) bytes.  Return the number of
    /// bytes written to the \p compressed buffer.  Issue a runtime error and
    /// return ~0 in case of an error.
    static size_t
    CompressToBuffer(char const *input, char *compressed, size_t inputSize, std::string *err);

    /// Decompress \p compressedSize bytes in \p compressed and store the
    /// result in \p output.  No more than \p maxOutputSize bytes will be
    /// written to \p output.
    static size_t
    DecompressFromBuffer(char const *compressed, char *output,
                         size_t compressedSize, size_t maxOutputSize, std::string *err);
};



} // namespace tinyusdz
