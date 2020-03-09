/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file Defines the StreamReader class which reads data from
 *  a binary stream with a well-defined endianness.
 */
#pragma once
#ifndef AI_STREAMREADER_H_INCLUDED
#define AI_STREAMREADER_H_INCLUDED

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/IOStream.hpp>
#include <assimp/Defines.h>
#include <assimp/ByteSwapper.h>
#include <assimp/Exceptional.h>

#include <memory>

namespace Assimp {

// --------------------------------------------------------------------------------------------
/** Wrapper class around IOStream to allow for consistent reading of binary data in both
 *  little and big endian format. Don't attempt to instance the template directly. Use
 *  StreamReaderLE to read from a little-endian stream and StreamReaderBE to read from a
 *  BE stream. The class expects that the endianness of any input data is known at
 *  compile-time, which should usually be true (#BaseImporter::ConvertToUTF8 implements
 *  runtime endianness conversions for text files).
 *
 *  XXX switch from unsigned int for size types to size_t? or ptrdiff_t?*/
// --------------------------------------------------------------------------------------------
template <bool SwapEndianess = false, bool RuntimeSwitch = false>
class StreamReader {
public:
    // FIXME: use these data types throughout the whole library,
    // then change them to 64 bit values :-)
    using diff = int;
    using pos  = unsigned int;

    // ---------------------------------------------------------------------
    /** Construction from a given stream with a well-defined endianness.
     *
     *  The StreamReader holds a permanent strong reference to the
     *  stream, which is released upon destruction.
     *  @param stream Input stream. The stream is not restarted if
     *    its file pointer is not at 0. Instead, the stream reader
     *    reads from the current position to the end of the stream.
     *  @param le If @c RuntimeSwitch is true: specifies whether the
     *    stream is in little endian byte order. Otherwise the
     *    endianness information is contained in the @c SwapEndianess
     *    template parameter and this parameter is meaningless.  */
    StreamReader(std::shared_ptr<IOStream> stream, bool le = false)
        : stream(stream)
        , le(le)
    {
        ai_assert(stream);
        InternBegin();
    }

    // ---------------------------------------------------------------------
    StreamReader(IOStream* stream, bool le = false)
        : stream(std::shared_ptr<IOStream>(stream))
        , le(le)
    {
        ai_assert(stream);
        InternBegin();
    }

    // ---------------------------------------------------------------------
    ~StreamReader() {
        delete[] buffer;
    }

    // deprecated, use overloaded operator>> instead

    // ---------------------------------------------------------------------
    /** Read a float from the stream  */
    float GetF4()
    {
        return Get<float>();
    }

    // ---------------------------------------------------------------------
    /** Read a double from the stream  */
    double GetF8()  {
        return Get<double>();
    }

    // ---------------------------------------------------------------------
    /** Read a signed 16 bit integer from the stream */
    int16_t GetI2() {
        return Get<int16_t>();
    }

    // ---------------------------------------------------------------------
    /** Read a signed 8 bit integer from the stream */
    int8_t GetI1()  {
        return Get<int8_t>();
    }

    // ---------------------------------------------------------------------
    /** Read an signed 32 bit integer from the stream */
    int32_t GetI4() {
        return Get<int32_t>();
    }

    // ---------------------------------------------------------------------
    /** Read a signed 64 bit integer from the stream */
    int64_t GetI8() {
        return Get<int64_t>();
    }

    // ---------------------------------------------------------------------
    /** Read a unsigned 16 bit integer from the stream */
    uint16_t GetU2()    {
        return Get<uint16_t>();
    }

    // ---------------------------------------------------------------------
    /** Read a unsigned 8 bit integer from the stream */
    uint8_t GetU1() {
        return Get<uint8_t>();
    }

    // ---------------------------------------------------------------------
    /** Read an unsigned 32 bit integer from the stream */
    uint32_t GetU4()    {
        return Get<uint32_t>();
    }

    // ---------------------------------------------------------------------
    /** Read a unsigned 64 bit integer from the stream */
    uint64_t GetU8()    {
        return Get<uint64_t>();
    }

    // ---------------------------------------------------------------------
    /** Get the remaining stream size (to the end of the stream) */
    unsigned int GetRemainingSize() const {
        return (unsigned int)(end - current);
    }

    // ---------------------------------------------------------------------
    /** Get the remaining stream size (to the current read limit). The
     *  return value is the remaining size of the stream if no custom
     *  read limit has been set. */
    unsigned int GetRemainingSizeToLimit() const {
        return (unsigned int)(limit - current);
    }

    // ---------------------------------------------------------------------
    /** Increase the file pointer (relative seeking)  */
    void IncPtr(intptr_t plus)    {
        current += plus;
        if (current > limit) {
            throw DeadlyImportError("End of file or read limit was reached");
        }
    }

    // ---------------------------------------------------------------------
    /** Get the current file pointer */
    int8_t* GetPtr() const  {
        return current;
    }

    // ---------------------------------------------------------------------
    /** Set current file pointer (Get it from #GetPtr). This is if you
     *  prefer to do pointer arithmetics on your own or want to copy
     *  large chunks of data at once.
     *  @param p The new pointer, which is validated against the size
     *    limit and buffer boundaries. */
    void SetPtr(int8_t* p)  {
        current = p;
        if (current > limit || current < buffer) {
            throw DeadlyImportError("End of file or read limit was reached");
        }
    }

    // ---------------------------------------------------------------------
    /** Copy n bytes to an external buffer
     *  @param out Destination for copying
     *  @param bytes Number of bytes to copy */
    void CopyAndAdvance(void* out, size_t bytes)    {
        int8_t* ur = GetPtr();
        SetPtr(ur+bytes); // fire exception if eof

        ::memcpy(out,ur,bytes);
    }

    // ---------------------------------------------------------------------
    /** Get the current offset from the beginning of the file */
    int GetCurrentPos() const   {
        return (unsigned int)(current - buffer);
    }

    void SetCurrentPos(size_t pos) {
        SetPtr(buffer + pos);
    }

    // ---------------------------------------------------------------------
    /** Setup a temporary read limit
     *
     *  @param limit Maximum number of bytes to be read from
     *    the beginning of the file. Specifying UINT_MAX
     *    resets the limit to the original end of the stream.
     *  Returns the previously set limit. */
    unsigned int SetReadLimit(unsigned int _limit)  {
        unsigned int prev = GetReadLimit();
        if (UINT_MAX == _limit) {
            limit = end;
            return prev;
        }

        limit = buffer + _limit;
        if (limit > end) {
            throw DeadlyImportError("StreamReader: Invalid read limit");
        }
        return prev;
    }

    // ---------------------------------------------------------------------
    /** Get the current read limit in bytes. Reading over this limit
     *  accidentally raises an exception.  */
    unsigned int GetReadLimit() const    {
        return (unsigned int)(limit - buffer);
    }

    // ---------------------------------------------------------------------
    /** Skip to the read limit in bytes. Reading over this limit
     *  accidentally raises an exception. */
    void SkipToReadLimit()  {
        current = limit;
    }

    // ---------------------------------------------------------------------
    /** overload operator>> and allow chaining of >> ops. */
    template <typename T>
    StreamReader& operator >> (T& f) {
        f = Get<T>();
        return *this;
    }

    // ---------------------------------------------------------------------
    /** Generic read method. ByteSwap::Swap(T*) *must* be defined */
    template <typename T>
    T Get() {
        if ( current + sizeof(T) > limit) {
            throw DeadlyImportError("End of file or stream limit was reached");
        }

        T f;
        ::memcpy (&f, current, sizeof(T));
        Intern::Getter<SwapEndianess,T,RuntimeSwitch>() (&f,le);
        current += sizeof(T);

        return f;
    }

private:
    // ---------------------------------------------------------------------
    void InternBegin() {
        if (!stream) {
            // in case someone wonders: StreamReader is frequently invoked with
            // no prior validation whether the input stream is valid. Since
            // no one bothers changing the error message, this message here
            // is passed down to the caller and 'unable to open file'
            // simply describes best what happened.
            throw DeadlyImportError("StreamReader: Unable to open file");
        }

        const size_t s = stream->FileSize() - stream->Tell();
        if (!s) {
            throw DeadlyImportError("StreamReader: File is empty or EOF is already reached");
        }

        current = buffer = new int8_t[s];
        const size_t read = stream->Read(current,1,s);
        // (read < s) can only happen if the stream was opened in text mode, in which case FileSize() is not reliable
        ai_assert(read <= s);
        end = limit = &buffer[read-1] + 1;
    }

private:
    std::shared_ptr<IOStream> stream;
    int8_t *buffer, *current, *end, *limit;
    bool le;
};

// --------------------------------------------------------------------------------------------
// `static` StreamReaders. Their byte order is fixed and they might be a little bit faster.
#ifdef AI_BUILD_BIG_ENDIAN
    typedef StreamReader<true>  StreamReaderLE;
    typedef StreamReader<false> StreamReaderBE;
#else
    typedef StreamReader<true>  StreamReaderBE;
    typedef StreamReader<false> StreamReaderLE;
#endif

// `dynamic` StreamReader. The byte order of the input data is specified in the
// c'tor. This involves runtime branching and might be a little bit slower.
typedef StreamReader<true,true> StreamReaderAny;

} // end namespace Assimp

#endif // !! AI_STREAMREADER_H_INCLUDED
