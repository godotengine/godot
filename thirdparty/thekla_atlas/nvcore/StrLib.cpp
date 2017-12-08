// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "StrLib.h"

#include "Memory.h"
#include "Utils.h" // swap

#include <math.h>   // log
#include <stdio.h>  // vsnprintf
#include <string.h> // strlen, strcmp, etc.

#if NV_CC_MSVC
#include <stdarg.h> // vsnprintf
#endif

using namespace nv;

namespace 
{
    static char * strAlloc(uint size)
    {
        return malloc<char>(size);
    }

    static char * strReAlloc(char * str, uint size)
    {
        return realloc<char>(str, size);
    }

    static void strFree(const char * str)
    {
        return free<char>(str);
    }

    /*static char * strDup( const char * str )
    {
        nvDebugCheck( str != NULL );
        uint len = uint(strlen( str ) + 1);
        char * dup = strAlloc( len );
        memcpy( dup, str, len );
        return dup;
    }*/

    // helper function for integer to string conversion.
    static char * i2a( uint i, char *a, uint r )
    {
        if( i / r > 0 ) {
            a = i2a( i / r, a, r );
        }
        *a = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i % r];
        return a + 1;
    }

    // Locale independent functions.
    static inline char toUpper( char c ) {
        return (c<'a' || c>'z') ? (c) : (c+'A'-'a');
    }
    static inline char toLower( char c ) {
        return (c<'A' || c>'Z') ? (c) : (c+'a'-'A');
    }
    static inline bool isAlpha( char c ) {
        return (c>='a' && c<='z') || (c>='A' && c<='Z');
    }
    static inline bool isDigit( char c ) {
        return c>='0' && c<='9';
    }
    static inline bool isAlnum( char c ) {
        return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9');
    }

}

uint nv::strLen(const char * str)
{
    nvDebugCheck(str != NULL);
    return U32(strlen(str));
}

int nv::strDiff(const char * s1, const char * s2)
{
    nvDebugCheck(s1 != NULL);
    nvDebugCheck(s2 != NULL);
    return strcmp(s1, s2);
}

int nv::strCaseDiff(const char * s1, const char * s2)
{
    nvDebugCheck(s1 != NULL);
    nvDebugCheck(s1 != NULL);
#if NV_CC_MSVC
    return _stricmp(s1, s2);
#else
    return strcasecmp(s1, s2);
#endif
}

bool nv::strEqual(const char * s1, const char * s2)
{
    if (s1 == s2) return true;
    if (s1 == NULL || s2 == NULL) return false;
    return strcmp(s1, s2) == 0;
}

bool nv::strCaseEqual(const char * s1, const char * s2)
{
    if (s1 == s2) return true;
    if (s1 == NULL || s2 == NULL) return false;
    return strCaseDiff(s1, s2) == 0;
}

bool nv::strBeginsWith(const char * str, const char * prefix)
{
    //return strstr(str, prefix) == dst;
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

bool nv::strEndsWith(const char * str, const char * suffix)
{
    uint ml = strLen(str);
    uint sl = strLen(suffix);
    if (ml < sl) return false;
    return strncmp(str + ml - sl, suffix, sl) == 0;
}

// @@ Add asserts to detect overlap between dst and src?
void nv::strCpy(char * dst, uint size, const char * src)
{
    nvDebugCheck(dst != NULL);
    nvDebugCheck(src != NULL);
#if NV_CC_MSVC && _MSC_VER >= 1400
    strcpy_s(dst, size, src);
#else
    NV_UNUSED(size);
    strcpy(dst, src);
#endif
}

void nv::strCpy(char * dst, uint size, const char * src, uint len)
{
    nvDebugCheck(dst != NULL);
    nvDebugCheck(src != NULL);
#if NV_CC_MSVC && _MSC_VER >= 1400
    strncpy_s(dst, size, src, len);
#else
    int n = min(len+1, size);
    strncpy(dst, src, n);
    dst[n-1] = '\0';
#endif
}

void nv::strCat(char * dst, uint size, const char * src)
{
    nvDebugCheck(dst != NULL);
    nvDebugCheck(src != NULL);
#if NV_CC_MSVC && _MSC_VER >= 1400
    strcat_s(dst, size, src);
#else
    NV_UNUSED(size);
    strcat(dst, src);
#endif
}

NVCORE_API const char * nv::strSkipWhiteSpace(const char * str)
{
    nvDebugCheck(str != NULL);
    while (*str == ' ') str++;
    return str;
}

NVCORE_API char * nv::strSkipWhiteSpace(char * str)
{
    nvDebugCheck(str != NULL);
    while (*str == ' ') str++;
    return str;
}


/** Pattern matching routine. I don't remember where did I get this. */
bool nv::strMatch(const char * str, const char * pat)
{
    nvDebugCheck(str != NULL);
    nvDebugCheck(pat != NULL);

    char c2;

    while (true) {
        if (*pat==0) {
            if (*str==0) return true;
            else         return false;
        }
        if ((*str==0) && (*pat!='*')) return false;
        if (*pat=='*') {
            pat++;
            if (*pat==0) return true;
            while (true) {
                if (strMatch(str, pat)) return true;
                if (*str==0) return false;
                str++;
            }
        }
        if (*pat=='?') goto match;
        if (*pat=='[') {
            pat++;
            while (true) {
                if ((*pat==']') || (*pat==0)) return false;
                if (*pat==*str) break;
                if (pat[1] == '-') {
                    c2 = pat[2];
                    if (c2==0) return false;
                    if ((*pat<=*str) && (c2>=*str)) break;
                    if ((*pat>=*str) && (c2<=*str)) break;
                    pat+=2;
                }
                pat++;
            }
            while (*pat!=']') {
                if (*pat==0) {
                    pat--;
                    break;
                }
                pat++;
            }
            goto match;
        }

        if (*pat == NV_PATH_SEPARATOR) {
            pat++;
            if (*pat==0) return false;
        }
        if (*pat!=*str) return false;

match:
        pat++;
        str++;
    }
}

bool nv::isNumber(const char * str) {
    while(*str != '\0') {
        if (!isDigit(*str)) return false;
        str++;
    }
    return true;
}


/** Empty string. */
StringBuilder::StringBuilder() : m_size(0), m_str(NULL)
{
}

/** Preallocate space. */
StringBuilder::StringBuilder( uint size_hint ) : m_size(size_hint)
{
    nvDebugCheck(m_size > 0);
    m_str = strAlloc(m_size);
    *m_str = '\0';
}

/** Copy ctor. */
StringBuilder::StringBuilder( const StringBuilder & s ) : m_size(0), m_str(NULL)
{
    copy(s);
}

/** Copy string. */
StringBuilder::StringBuilder(const char * s) : m_size(0), m_str(NULL)
{
    if (s != NULL) {
        copy(s);
    }
}

/** Copy string. */
StringBuilder::StringBuilder(const char * s, uint len) : m_size(0), m_str(NULL)
{
    copy(s, len);
}

/** Delete the string. */
StringBuilder::~StringBuilder()
{
    strFree(m_str);
}


/** Format a string safely. */
StringBuilder & StringBuilder::format( const char * fmt, ... )
{
    nvDebugCheck(fmt != NULL);
    va_list arg;
    va_start( arg, fmt );

    formatList( fmt, arg );

    va_end( arg );

    return *this;
}


/** Format a string safely. */
StringBuilder & StringBuilder::formatList( const char * fmt, va_list arg )
{
    nvDebugCheck(fmt != NULL);

    if (m_size == 0) {
        m_size = 64;
        m_str = strAlloc( m_size );
    }

    va_list tmp;
    va_copy(tmp, arg);
#if NV_CC_MSVC && _MSC_VER >= 1400
    int n = vsnprintf_s(m_str, m_size, _TRUNCATE, fmt, tmp);
#else
    int n = vsnprintf(m_str, m_size, fmt, tmp);
#endif
    va_end(tmp);

    while( n < 0 || n >= int(m_size) ) {
        if( n > -1 ) {
            m_size = n + 1;
        }
        else {
            m_size *= 2;
        }

        m_str = strReAlloc(m_str, m_size);

        va_copy(tmp, arg);
#if NV_CC_MSVC && _MSC_VER >= 1400
        n = vsnprintf_s(m_str, m_size, _TRUNCATE, fmt, tmp);
#else
        n = vsnprintf(m_str, m_size, fmt, tmp);
#endif
        va_end(tmp);
    }

    nvDebugCheck(n < int(m_size));

    // Make sure it's null terminated.
    nvDebugCheck(m_str[n] == '\0');
    //str[n] = '\0';

    return *this;
}


// Append a character.
StringBuilder & StringBuilder::append( char c )
{
    return append(&c, 1);
}

// Append a string.
StringBuilder & StringBuilder::append( const char * s )
{
    return append(s, U32(strlen( s )));
}

// Append a string.
StringBuilder & StringBuilder::append(const char * s, uint len)
{
    nvDebugCheck(s != NULL);

    uint offset = length();
    const uint size = offset + len + 1;
    reserve(size);
    strCpy(m_str + offset, len + 1, s, len);

    return *this;
}

StringBuilder & StringBuilder::append(const StringBuilder & str)
{
    return append(str.m_str, str.length());
}


/** Append a formatted string. */
StringBuilder & StringBuilder::appendFormat( const char * fmt, ... )
{
    nvDebugCheck( fmt != NULL );

    va_list arg;
    va_start( arg, fmt );

    appendFormatList( fmt, arg );

    va_end( arg );

    return *this;
}


/** Append a formatted string. */
StringBuilder & StringBuilder::appendFormatList( const char * fmt, va_list arg )
{
    nvDebugCheck( fmt != NULL );

    va_list tmp;
    va_copy(tmp, arg);

    if (m_size == 0) {
        formatList(fmt, arg);
    }
    else {
        StringBuilder tmp_str;
        tmp_str.formatList( fmt, tmp );
        append( tmp_str.str() );
    }

    va_end(tmp);

    return *this;
}

// Append n spaces.
StringBuilder & StringBuilder::appendSpace(uint n)
{
    if (m_str == NULL) {
        m_size = n + 1;
        m_str = strAlloc(m_size);
        memset(m_str, ' ', m_size);
        m_str[n] = '\0';
    }
    else {
        const uint len = strLen(m_str);
        if (m_size < len + n + 1) {
            m_size = len + n + 1;
            m_str = strReAlloc(m_str, m_size);
        }
        memset(m_str + len, ' ', n);
        m_str[len+n] = '\0';
    }

    return *this;
}


/** Convert number to string in the given base. */
StringBuilder & StringBuilder::number( int i, int base )
{
    nvCheck( base >= 2 );
    nvCheck( base <= 36 );

    // @@ This needs to be done correctly.
    // length = floor(log(i, base));
    uint len = uint(log(float(i)) / log(float(base)) + 2); // one more if negative
    reserve(len);

    if( i < 0 ) {
        *m_str = '-';
        *i2a(uint(-i), m_str+1, base) = 0;
    }
    else {
        *i2a(i, m_str, base) = 0;
    }

    return *this;
}


/** Convert number to string in the given base. */
StringBuilder & StringBuilder::number( uint i, int base )
{
    nvCheck( base >= 2 );
    nvCheck( base <= 36 );

    // @@ This needs to be done correctly.
    // length = floor(log(i, base));
    uint len = uint(log(float(i)) / log(float(base)) - 0.5f + 1);
    reserve(len);

    *i2a(i, m_str, base) = 0;

    return *this;
}


/** Resize the string preserving the contents. */
StringBuilder & StringBuilder::reserve( uint size_hint )
{
    nvCheck(size_hint != 0);
    if (size_hint > m_size) {
        m_str = strReAlloc(m_str, size_hint);
        m_size = size_hint;
    }
    return *this;
}


/** Copy a string safely. */
StringBuilder & StringBuilder::copy(const char * s)
{
    nvCheck( s != NULL );
    const uint str_size = uint(strlen( s )) + 1;
    reserve(str_size);
    memcpy(m_str, s, str_size);
    return *this;
}

/** Copy a string safely. */
StringBuilder & StringBuilder::copy(const char * s, uint len)
{
    nvCheck( s != NULL );
    const uint str_size = len + 1;
    reserve(str_size);
    strCpy(m_str, str_size, s, len);
    return *this;
}


/** Copy an StringBuilder. */
StringBuilder & StringBuilder::copy( const StringBuilder & s )
{
    if (s.m_str == NULL) {
        nvCheck( s.m_size == 0 );
        reset();
    }
    else {
        reserve( s.m_size );
        strCpy( m_str, s.m_size, s.m_str );
    }
    return *this;
}

bool StringBuilder::endsWith(const char * str) const
{
    uint l = uint(strlen(str));
    uint ml = uint(strlen(m_str));
    if (ml < l) return false;
    return strncmp(m_str + ml - l, str, l) == 0;
}

bool StringBuilder::beginsWith(const char * str) const 
{
    size_t l = strlen(str);
    return strncmp(m_str, str, l) == 0;
}

// Find given char starting from the end.
char * StringBuilder::reverseFind(char c)
{
    int length = (int)strlen(m_str) - 1;
    while (length >= 0 && m_str[length] != c) {
        length--;
    }
    if (length >= 0) {
        return m_str + length;
    }
    else {
        return NULL;
    }
}


/** Reset the string. */
void StringBuilder::reset()
{
    m_size = 0;
    strFree( m_str );
    m_str = NULL;
}

/** Release the allocated string. */
char * StringBuilder::release()
{
    char * str = m_str;
    m_size = 0;
    m_str = NULL;
    return str;
}

// Take ownership of string.
void StringBuilder::acquire(char * str)
{
    if (str) {
        m_size = strLen(str) + 1;
        m_str = str;
    }
    else {
        m_size = 0;
        m_str = NULL;
    }
}

// Swap strings.
void nv::swap(StringBuilder & a, StringBuilder & b) {
    swap(a.m_size, b.m_size);
    swap(a.m_str, b.m_str);
}


/// Get the file name from a path.
const char * Path::fileName() const
{
    return fileName(m_str);
}


/// Get the extension from a file path.
const char * Path::extension() const
{
    return extension(m_str);
}


/*static */void Path::translatePath(char * path, char pathSeparator/*= NV_PATH_SEPARATOR*/) {
    if (path != NULL) {
        for (int i = 0;; i++) {
            if (path[i] == '\0') break;
            if (path[i] == '\\' || path[i] == '/') path[i] = pathSeparator;
        }
    }
}

/// Toggles path separators (ie. \\ into /).
void Path::translatePath(char pathSeparator/*=NV_PATH_SEPARATOR*/)
{
    if (!isNull()) {
        translatePath(m_str, pathSeparator);
    }
}

void Path::appendSeparator(char pathSeparator/*=NV_PATH_SEPARATOR*/)
{
    nvCheck(!isNull());

    const uint l = length();
    
    if (m_str[l] != '\\' && m_str[l] != '/') {
        char separatorString[] = { pathSeparator, '\0' };
        append(separatorString);
    }
}


/**
* Strip the file name from a path.
* @warning path cannot end with '/' o '\\', can't it?
*/
void Path::stripFileName()
{
    nvCheck( m_str != NULL );

    int length = (int)strlen(m_str) - 1;
    while (length > 0 && m_str[length] != '/' && m_str[length] != '\\'){
        length--;
    }
    if( length ) {
        m_str[length+1] = 0;
    }
    else {
        m_str[0] = 0;
    }
}


/// Strip the extension from a path name.
void Path::stripExtension()
{
    nvCheck( m_str != NULL );

    int length = (int)strlen(m_str) - 1;
    while (length > 0 && m_str[length] != '.') {
        length--;
        if( m_str[length] == NV_PATH_SEPARATOR ) {
            return; // no extension
        }
    }
    if (length > 0) {
        m_str[length] = 0;
    }
}


/// Get the path separator.
// static
char Path::separator()
{
    return NV_PATH_SEPARATOR;
}

// static 
const char * Path::fileName(const char * str)
{
    nvCheck( str != NULL );

    int length = (int)strlen(str) - 1;
    while (length >= 0 && str[length] != '\\' && str[length] != '/') {
        length--;
    }

    return &str[length+1];
}

// static 
const char * Path::extension(const char * str)
{
    nvCheck( str != NULL );

    int length, l;
    l = length = (int)strlen( str );
    while (length > 0 && str[length] != '.') {
        length--;
        if (str[length] == '\\' || str[length] == '/') {
            return &str[l]; // no extension
        }
    }
    if (length == 0) {
        return &str[l];
    }
    return &str[length];
}



/// Clone this string
String String::clone() const
{
    String str(data);
    return str;
}

void String::setString(const char * str)
{
    if (str == NULL) {
        data = NULL;
    }
    else {
        allocString( str );
        addRef();
    }
}

void String::setString(const char * str, uint length)
{
    nvDebugCheck(str != NULL);

    allocString(str, length);
    addRef();
}

void String::setString(const StringBuilder & str)
{
    if (str.str() == NULL) {
        data =	NULL;
    }
    else {
        allocString(str.str());
        addRef();
    }
}	

// Add reference count.
void String::addRef()
{
    if (data != NULL)
    {
        setRefCount(getRefCount() + 1);
    }
}

// Decrease reference count.
void String::release()
{
    if (data != NULL)
    {
        const uint16 count = getRefCount();
        setRefCount(count - 1);
        if (count - 1 == 0) {
            free(data - 2);
            data = NULL;
        }
    }
}

void String::allocString(const char * str, uint len)
{
    const char * ptr = malloc<char>(2 + len + 1);

    setData( ptr );
    setRefCount( 0 );

    // Copy string.
    strCpy(const_cast<char *>(data), len+1, str, len);

    // Add terminating character.
    const_cast<char *>(data)[len] = '\0';
}

void nv::swap(String & a, String & b) {
    swap(a.data, b.data);
}
