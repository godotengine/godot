// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_STRING_H
#define NV_CORE_STRING_H

#include "Debug.h"
#include "Hash.h" // hash

//#include <string.h> // strlen, etc.

#if NV_OS_WIN32
#define NV_PATH_SEPARATOR '\\'
#else
#define NV_PATH_SEPARATOR '/'
#endif

namespace nv
{

    NVCORE_API uint strHash(const char * str, uint h) NV_PURE;

    /// String hash based on Bernstein's hash.
    inline uint strHash(const char * data, uint h = 5381)
    {
        uint i = 0;
        while(data[i] != 0) {
            h = (33 * h) ^ uint(data[i]);
            i++;
        }
        return h;
    }

    template <> struct Hash<const char *> {
        uint operator()(const char * str) const { return strHash(str); }
    };

    NVCORE_API uint strLen(const char * str) NV_PURE;                       // Asserts on NULL strings.

    NVCORE_API int strDiff(const char * s1, const char * s2) NV_PURE;       // Asserts on NULL strings.
    NVCORE_API int strCaseDiff(const char * s1, const char * s2) NV_PURE;   // Asserts on NULL strings.
    NVCORE_API bool strEqual(const char * s1, const char * s2) NV_PURE;     // Accepts NULL strings.
    NVCORE_API bool strCaseEqual(const char * s1, const char * s2) NV_PURE; // Accepts NULL strings.

    template <> struct Equal<const char *> {
        bool operator()(const char * a, const char * b) const { return strEqual(a, b); }
    };

    NVCORE_API bool strBeginsWith(const char * dst, const char * prefix) NV_PURE;
    NVCORE_API bool strEndsWith(const char * dst, const char * suffix) NV_PURE;


    NVCORE_API void strCpy(char * dst, uint size, const char * src);
    NVCORE_API void strCpy(char * dst, uint size, const char * src, uint len);
    NVCORE_API void strCat(char * dst, uint size, const char * src);

    NVCORE_API const char * strSkipWhiteSpace(const char * str);
    NVCORE_API char * strSkipWhiteSpace(char * str);

    NVCORE_API bool strMatch(const char * str, const char * pat) NV_PURE;

    NVCORE_API bool isNumber(const char * str) NV_PURE;

    /* @@ Implement these two functions and modify StringBuilder to use them?
    NVCORE_API void strFormat(const char * dst, const char * fmt, ...);
    NVCORE_API void strFormatList(const char * dst, const char * fmt, va_list arg);

    template <size_t count> void strFormatSafe(char (&buffer)[count], const char *fmt, ...) __attribute__((format (printf, 2, 3)));
    template <size_t count> void strFormatSafe(char (&buffer)[count], const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        strFormatList(buffer, count, fmt, args);
        va_end(args);
    }
    template <size_t count> void strFormatListSafe(char (&buffer)[count], const char *fmt, va_list arg) {
        va_list tmp;
        va_copy(tmp, args);
        strFormatList(buffer, count, fmt, tmp);
        va_end(tmp);
    }*/

    template <int count> void strCpySafe(char (&buffer)[count], const char *src) {
        strCpy(buffer, count, src);
    }

    template <int count> void strCatSafe(char (&buffer)[count], const char * src) {
        strCat(buffer, count, src);
    }



    /// String builder.
    class NVCORE_CLASS StringBuilder
    {
    public:

        StringBuilder();
        explicit StringBuilder( uint size_hint );
        StringBuilder(const char * str);
        StringBuilder(const char * str, uint len);
        StringBuilder(const StringBuilder & other);

        ~StringBuilder();

        StringBuilder & format( const char * format, ... ) __attribute__((format (printf, 2, 3)));
        StringBuilder & formatList( const char * format, va_list arg );

        StringBuilder & append(char c);
        StringBuilder & append(const char * str);
        StringBuilder & append(const char * str, uint len);
        StringBuilder & append(const StringBuilder & str);
        StringBuilder & appendFormat(const char * format, ...) __attribute__((format (printf, 2, 3)));
        StringBuilder & appendFormatList(const char * format, va_list arg);

        StringBuilder & appendSpace(uint n);

        StringBuilder & number( int i, int base = 10 );
        StringBuilder & number( uint i, int base = 10 );

        StringBuilder & reserve(uint size_hint);
        StringBuilder & copy(const char * str);
        StringBuilder & copy(const char * str, uint len);
        StringBuilder & copy(const StringBuilder & str);

        StringBuilder & toLower();
        StringBuilder & toUpper();

        bool endsWith(const char * str) const;
        bool beginsWith(const char * str) const;

        char * reverseFind(char c);

        void reset();
        bool isNull() const { return m_size == 0; }

        // const char * accessors
        //operator const char * () const { return m_str; }
        //operator char * () { return m_str; }
        const char * str() const { return m_str; }
        char * str() { return m_str; }

        char * release();       // Release ownership of string.
        void acquire(char *);   // Take ownership of string.

        /// Implement value semantics.
        StringBuilder & operator=( const StringBuilder & s ) {
            return copy(s);
        }

        /// Implement value semantics.
        StringBuilder & operator=( const char * s ) {
            return copy(s);
        }

        /// Equal operator.
        bool operator==( const StringBuilder & s ) const {
            return strMatch(s.m_str, m_str);
        }

        /// Return the exact length.
        uint length() const { return isNull() ? 0 : strLen(m_str); }

        /// Return the size of the string container.
        uint capacity() const { return m_size; }

        /// Return the hash of the string.
        uint hash() const { return isNull() ? 0 : strHash(m_str); }

        // Swap strings.
        friend void swap(StringBuilder & a, StringBuilder & b);

    protected:

        /// Size of the string container.
        uint m_size;

        /// String.
        char * m_str;

    };


    /// Path string. @@ This should be called PathBuilder.
    class NVCORE_CLASS Path : public StringBuilder
    {
    public:
        Path() : StringBuilder() {}
        explicit Path(int size_hint) : StringBuilder(size_hint) {}
        Path(const char * str) : StringBuilder(str) {}
        Path(const Path & path) : StringBuilder(path) {}

        const char * fileName() const;
        const char * extension() const;

        void translatePath(char pathSeparator = NV_PATH_SEPARATOR);

        void appendSeparator(char pathSeparator = NV_PATH_SEPARATOR);

        void stripFileName();
        void stripExtension();

        // statics
        NVCORE_API static char separator();
        NVCORE_API static const char * fileName(const char *);
        NVCORE_API static const char * extension(const char *);

        NVCORE_API static void translatePath(char * path, char pathSeparator = NV_PATH_SEPARATOR);
    };


    /// String class.
    class NVCORE_CLASS String
    {
    public:

        /// Constructs a null string. @sa isNull()
        String()
        {
            data = NULL;
        }

        /// Constructs a shared copy of str.
        String(const String & str)
        {
            data = str.data;
            if (data != NULL) addRef();
        }

        /// Constructs a shared string from a standard string.
        String(const char * str)
        {
            setString(str);
        }

        /// Constructs a shared string from a standard string.
        String(const char * str, int length)
        {
            setString(str, length);
        }

        /// Constructs a shared string from a StringBuilder.
        String(const StringBuilder & str)
        {
            setString(str);
        }

        /// Dtor.
        ~String()
        {
            release();
        }

        String clone() const;

        /// Release the current string and allocate a new one.
        const String & operator=( const char * str )
        {
            release();
            setString( str );
            return *this;
        }

        /// Release the current string and allocate a new one.
        const String & operator=( const StringBuilder & str )
        {
            release();
            setString( str );
            return *this;
        }

        /// Implement value semantics.
        String & operator=( const String & str )
        {
            if (str.data != data)
            {
                release();
                data = str.data;
                addRef();
            }
            return *this;
        }

        /// Equal operator.
        bool operator==( const String & str ) const
        {
            return strMatch(str.data, data);
        }

        /// Equal operator.
        bool operator==( const char * str ) const
        {
            return strMatch(str, data);
        }

        /// Not equal operator.
        bool operator!=( const String & str ) const
        {
            return !strMatch(str.data, data);
        }

        /// Not equal operator.
        bool operator!=( const char * str ) const
        {
            return !strMatch(str, data);
        }

        /// Returns true if this string is the null string.
        bool isNull() const { return data == NULL; }

        /// Return the exact length.
        uint length() const { nvDebugCheck(data != NULL); return strLen(data); }

        /// Return the hash of the string.
        uint hash() const { nvDebugCheck(data != NULL); return strHash(data); }

        /// const char * cast operator.
        operator const char * () const { return data; }

        /// Get string pointer.
        const char * str() const { return data; }


    private:

        // Add reference count.
        void addRef();

        // Decrease reference count.
        void release();

        uint16 getRefCount() const
        {
            nvDebugCheck(data != NULL);
            return *reinterpret_cast<const uint16 *>(data - 2);
        }

        void setRefCount(uint16 count) {
            nvDebugCheck(data != NULL);
            nvCheck(count < 0xFFFF);
            *reinterpret_cast<uint16 *>(const_cast<char *>(data - 2)) = uint16(count);
        }

        void setData(const char * str) {
            data = str + 2;
        }

        void allocString(const char * str)
        {
            allocString(str, strLen(str));
        }

        void allocString(const char * str, uint length);

        void setString(const char * str);
        void setString(const char * str, uint length);
        void setString(const StringBuilder & str);

        // Swap strings.
        friend void swap(String & a, String & b);

    private:

        const char * data;

    };

    template <> struct Hash<String> {
        uint operator()(const String & str) const { return str.hash(); }
    };


    // Like AutoPtr, but for const char strings.
    class AutoString
    {
        NV_FORBID_COPY(AutoString);
        NV_FORBID_HEAPALLOC();
    public:

        // Ctor.
        AutoString(const char * p = NULL) : m_ptr(p) { }

#if NV_CC_CPP11
        // Move ctor.
        AutoString(AutoString && ap) : m_ptr(ap.m_ptr) { ap.m_ptr = NULL; }
#endif
        
        // Dtor. Deletes owned pointer.
        ~AutoString() {
            delete [] m_ptr;
            m_ptr = NULL;
        }

        // Delete owned pointer and assign new one.
        void operator=(const char * p) {
            if (p != m_ptr) 
            {
                delete [] m_ptr;
                m_ptr = p;
            }
        }

        // Get pointer.
        const char * ptr() const { return m_ptr; }
        operator const char *() const { return m_ptr; }

        // Relinquish ownership of the underlying pointer and returns that pointer.
        const char * release() {
            const char * tmp = m_ptr;
            m_ptr = NULL;
            return tmp;
        }

        // comparison operators.
        friend bool operator == (const AutoString & ap, const char * const p) {
            return (ap.ptr() == p);
        }
        friend bool operator != (const AutoString & ap, const char * const p) {
            return (ap.ptr() != p);
        }
        friend bool operator == (const char * const p, const AutoString & ap) {
            return (ap.ptr() == p);
        }
        friend bool operator != (const char * const p, const AutoString & ap) {
            return (ap.ptr() != p);
        }

    private:
        const char * m_ptr;
    };

} // nv namespace

#endif // NV_CORE_STRING_H
