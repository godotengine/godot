// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

//#pragma once
//#ifndef NV_CORE_STDSTREAM_H
//#define NV_CORE_STDSTREAM_H

#include "nvcore.h"
#include "Stream.h"
#include "Array.h"

#include <stdio.h> // fopen
#include <string.h> // memcpy

namespace nv
{

    // Portable version of fopen.
    inline FILE * fileOpen(const char * fileName, const char * mode)
    {
        nvCheck(fileName != NULL);
#if NV_CC_MSVC && _MSC_VER >= 1400
        FILE * fp;
        if (fopen_s(&fp, fileName, mode) == 0) {
            return fp;
        }
        return NULL;
#else
        return fopen(fileName, mode);
#endif
    }


    /// Base stdio stream.
    class NVCORE_CLASS StdStream : public Stream
    {
        NV_FORBID_COPY(StdStream);
    public:

        /// Ctor.
        StdStream( FILE * fp, bool autoclose ) : m_fp(fp), m_autoclose(autoclose) { }

        /// Dtor. 
        virtual ~StdStream()
        {
            if( m_fp != NULL && m_autoclose ) {
#if NV_OS_WIN32
                _fclose_nolock( m_fp );
#else
                fclose( m_fp );
#endif
            }
        }


        /** @name Stream implementation. */
        //@{
        virtual void seek( uint pos )
        {
            nvDebugCheck(m_fp != NULL);
            nvDebugCheck(pos <= size());
#if NV_OS_WIN32
            _fseek_nolock(m_fp, pos, SEEK_SET);
#else
            fseek(m_fp, pos, SEEK_SET);
#endif
        }

        virtual uint tell() const
        {
            nvDebugCheck(m_fp != NULL);
#if NV_OS_WIN32
            return _ftell_nolock(m_fp);
#else
            return (uint)ftell(m_fp);
#endif
        }

        virtual uint size() const
        {
            nvDebugCheck(m_fp != NULL);
#if NV_OS_WIN32
            uint pos = _ftell_nolock(m_fp);
            _fseek_nolock(m_fp, 0, SEEK_END);
            uint end = _ftell_nolock(m_fp);
            _fseek_nolock(m_fp, pos, SEEK_SET);
#else
            uint pos = (uint)ftell(m_fp);
            fseek(m_fp, 0, SEEK_END);
            uint end = (uint)ftell(m_fp);
            fseek(m_fp, pos, SEEK_SET);
#endif
            return end;
        }

        virtual bool isError() const
        {
            return m_fp == NULL || ferror( m_fp ) != 0;
        }

        virtual void clearError()
        {
            nvDebugCheck(m_fp != NULL);
            clearerr(m_fp);
        }

        // @@ The original implementation uses feof, which only returns true when we attempt to read *past* the end of the stream. 
        // That is, if we read the last byte of a file, then isAtEnd would still return false, even though the stream pointer is at the file end. This is not the intent and was inconsistent with the implementation of the MemoryStream, a better 
        // implementation uses use ftell and fseek to determine our location within the file.
        virtual bool isAtEnd() const
        {
            if (m_fp == NULL) return true;
            //nvDebugCheck(m_fp != NULL);
            //return feof( m_fp ) != 0;
#if NV_OS_WIN32
            uint pos = _ftell_nolock(m_fp);
            _fseek_nolock(m_fp, 0, SEEK_END);
            uint end = _ftell_nolock(m_fp);
            _fseek_nolock(m_fp, pos, SEEK_SET);
#else
            uint pos = (uint)ftell(m_fp);
            fseek(m_fp, 0, SEEK_END);
            uint end = (uint)ftell(m_fp);
            fseek(m_fp, pos, SEEK_SET);
#endif
            return pos == end;
        }

        /// Always true.
        virtual bool isSeekable() const { return true; }
        //@}

    protected:

        FILE * m_fp;
        bool m_autoclose;

    };


    /// Standard output stream.
    class NVCORE_CLASS StdOutputStream : public StdStream
    {
        NV_FORBID_COPY(StdOutputStream);
    public:

        /// Construct stream by file name.
        StdOutputStream( const char * name ) : StdStream(fileOpen(name, "wb"), /*autoclose=*/true) { }

        /// Construct stream by file handle.
        StdOutputStream( FILE * fp, bool autoclose ) : StdStream(fp, autoclose)
        {
        }

        /** @name Stream implementation. */
        //@{
        /// Write data.
        virtual uint serialize( void * data, uint len )
        {
            nvDebugCheck(data != NULL);
            nvDebugCheck(m_fp != NULL);
#if NV_OS_WIN32
            return (uint)_fwrite_nolock(data, 1, len, m_fp);
#elif NV_OS_LINUX
            return (uint)fwrite_unlocked(data, 1, len, m_fp);
#elif NV_OS_DARWIN
            // @@ No error checking, always returns len.
            for (uint i = 0; i < len; i++) {
                putc_unlocked(((char *)data)[i], m_fp);
            }
            return len;
#else
            return (uint)fwrite(data, 1, len, m_fp);
#endif
        }

        virtual bool isLoading() const
        {
            return false;
        }

        virtual bool isSaving() const
        {
            return true;
        }
        //@}

    };


    /// Standard input stream.
    class NVCORE_CLASS StdInputStream : public StdStream
    {
        NV_FORBID_COPY(StdInputStream);
    public:

        /// Construct stream by file name.
        StdInputStream( const char * name ) : StdStream(fileOpen(name, "rb"), /*autoclose=*/true) { }

        /// Construct stream by file handle.
        StdInputStream( FILE * fp, bool autoclose=true ) : StdStream(fp, autoclose)
        {
        }

        /** @name Stream implementation. */
        //@{
        /// Read data.
        virtual uint serialize( void * data, uint len )
        {
            nvDebugCheck(data != NULL);
            nvDebugCheck(m_fp != NULL);
#if NV_OS_WIN32
            return (uint)_fread_nolock(data, 1, len, m_fp);
#elif NV_OS_LINUX
            return (uint)fread_unlocked(data, 1, len, m_fp);
#elif NV_OS_DARWIN
            // This is rather lame. Not sure if it's faster than the locked version.
            for (uint i = 0; i < len; i++) {
                ((char *)data)[i] = getc_unlocked(m_fp);
                if (feof_unlocked(m_fp) != 0) {
                    return i;
                }
            }
            return len;
#else
            return (uint)fread(data, 1, len, m_fp);
#endif
            
        }

        virtual bool isLoading() const
        {
            return true;
        }

        virtual bool isSaving() const
        {
            return false;
        }
        //@}
    };



    /// Memory input stream.
    class NVCORE_CLASS MemoryInputStream : public Stream
    {
        NV_FORBID_COPY(MemoryInputStream);
    public:

        /// Ctor.
        MemoryInputStream( const uint8 * mem, uint size ) : m_mem(mem), m_ptr(mem), m_size(size) { }

        /** @name Stream implementation. */
        //@{
        /// Read data.
        virtual uint serialize( void * data, uint len )
        {
            nvDebugCheck(data != NULL);
            nvDebugCheck(!isError());

            uint left = m_size - tell();
            if (len > left) len = left;

            memcpy( data, m_ptr, len );
            m_ptr += len;

            return len;
        }

        virtual void seek( uint pos )
        {
            nvDebugCheck(!isError());
            m_ptr = m_mem + pos;
            nvDebugCheck(!isError());
        }

        virtual uint tell() const
        {
            nvDebugCheck(m_ptr >= m_mem);
            return uint(m_ptr - m_mem);
        }

        virtual uint size() const
        {
            return m_size;
        }

        virtual bool isError() const
        {
            return m_mem == NULL || m_ptr > m_mem + m_size || m_ptr < m_mem;
        }

        virtual void clearError()
        {
            // Nothing to do.
        }

        virtual bool isAtEnd() const
        {
            return m_ptr == m_mem + m_size;
        }

        /// Always true.
        virtual bool isSeekable() const
        {
            return true;
        }

        virtual bool isLoading() const
        {
            return true;
        }

        virtual bool isSaving() const
        {
            return false;
        }
        //@}

        const uint8 * ptr() const { return m_ptr; }


    private:

        const uint8 * m_mem;
        const uint8 * m_ptr;
        uint m_size;

    };


    /// Buffer output stream.
    class NVCORE_CLASS BufferOutputStream : public Stream
    {
        NV_FORBID_COPY(BufferOutputStream);
    public:

        BufferOutputStream(Array<uint8> & buffer) : m_buffer(buffer) { }

        virtual uint serialize( void * data, uint len )
        {
            nvDebugCheck(data != NULL);
            m_buffer.append((uint8 *)data, len);
            return len;
        }

        virtual void seek( uint /*pos*/ ) { /*Not implemented*/ }
        virtual uint tell() const { return m_buffer.size(); }
        virtual uint size() const { return m_buffer.size(); }

        virtual bool isError() const { return false; }
        virtual void clearError() {}

        virtual bool isAtEnd() const { return true; }
        virtual bool isSeekable() const { return false; }
        virtual bool isLoading() const { return false; }
        virtual bool isSaving() const { return true; }

    private:
        Array<uint8> & m_buffer;
    };


    /// Protected input stream.
    class NVCORE_CLASS ProtectedStream : public Stream
    {
        NV_FORBID_COPY(ProtectedStream);
    public:

        /// Ctor.
        ProtectedStream( Stream & s ) : m_s(&s), m_autodelete(false)
        { 
        }

        /// Ctor.
        ProtectedStream( Stream * s, bool autodelete = true ) : 
        m_s(s), m_autodelete(autodelete) 
        {
            nvDebugCheck(m_s != NULL);
        }

        /// Dtor.
        virtual ~ProtectedStream()
        {
            if( m_autodelete ) {
                delete m_s;
            }
        }

        /** @name Stream implementation. */
        //@{
        /// Read data.
        virtual uint serialize( void * data, uint len )
        {
            nvDebugCheck(data != NULL);
            len = m_s->serialize( data, len );

            if( m_s->isError() ) {
#if NV_OS_ORBIS
                //SBtodoORBIS disabled (no exceptions)
#else
                throw;
#endif
            }

            return len;
        }

        virtual void seek( uint pos )
        {
            m_s->seek( pos );

            if( m_s->isError() ) {
#if NV_OS_ORBIS
                //SBtodoORBIS disabled (no exceptions)
#else
                throw;
#endif
            }
        }

        virtual uint tell() const
        {
            return m_s->tell();
        }

        virtual uint size() const
        {
            return m_s->size();
        }

        virtual bool isError() const
        {
            return m_s->isError();
        }

        virtual void clearError()
        {
            m_s->clearError();
        }

        virtual bool isAtEnd() const
        {
            return m_s->isAtEnd();
        }

        virtual bool isSeekable() const
        {
            return m_s->isSeekable();
        }

        virtual bool isLoading() const
        {
            return m_s->isLoading();
        }

        virtual bool isSaving() const
        {
            return m_s->isSaving();
        }
        //@}


    private:

        Stream * const m_s;
        bool const m_autodelete;

    };

} // nv namespace


//#endif // NV_CORE_STDSTREAM_H
