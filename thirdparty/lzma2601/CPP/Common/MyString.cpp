// Common/MyString.cpp

#include "StdAfx.h"

#ifdef _WIN32
#include <wchar.h>
#else
#include <ctype.h>
#endif

#include "IntToString.h"

#if !defined(_UNICODE) || !defined(USE_UNICODE_FSTRING)
#include "StringConvert.h"
#endif

#include "MyString.h"

#define MY_STRING_NEW(_T_, _size_) new _T_[_size_]
// #define MY_STRING_NEW(_T_, _size_) ((_T_ *)my_new((size_t)(_size_) * sizeof(_T_)))

/*
inline const char* MyStringGetNextCharPointer(const char *p) throw()
{
  #if defined(_WIN32) && !defined(UNDER_CE)
  return CharNextA(p);
  #else
  return p + 1;
  #endif
}
*/

#define MY_STRING_NEW_char(_size_) MY_STRING_NEW(char, (_size_))
#define MY_STRING_NEW_wchar_t(_size_) MY_STRING_NEW(wchar_t, (_size_))


int FindCharPosInString(const char *s, char c) throw()
{
  for (const char *p = s;; p++)
  {
    if (*p == c)
      return (int)(p - s);
    if (*p == 0)
      return -1;
    // MyStringGetNextCharPointer(p);
  }
}

int FindCharPosInString(const wchar_t *s, wchar_t c) throw()
{
  for (const wchar_t *p = s;; p++)
  {
    if (*p == c)
      return (int)(p - s);
    if (*p == 0)
      return -1;
  }
}

/*
void MyStringUpper_Ascii(char *s) throw()
{
  for (;;)
  {
    const char c = *s;
    if (c == 0)
      return;
    *s++ = MyCharUpper_Ascii(c);
  }
}

void MyStringUpper_Ascii(wchar_t *s) throw()
{
  for (;;)
  {
    const wchar_t c = *s;
    if (c == 0)
      return;
    *s++ = MyCharUpper_Ascii(c);
  }
}
*/

void MyStringLower_Ascii(char *s) throw()
{
  for (;;)
  {
    const char c = *s;
    if (c == 0)
      return;
    *s++ = MyCharLower_Ascii(c);
  }
}

void MyStringLower_Ascii(wchar_t *s) throw()
{
  for (;;)
  {
    const wchar_t c = *s;
    if (c == 0)
      return;
    *s++ = MyCharLower_Ascii(c);
  }
}

#ifdef _WIN32

#ifdef _UNICODE

// wchar_t * MyStringUpper(wchar_t *s) { return CharUpperW(s); }
// wchar_t * MyStringLower(wchar_t *s) { return CharLowerW(s); }
// for WinCE - FString - char
// const char *MyStringGetPrevCharPointer(const char * /* base */, const char *p) { return p - 1; }

#else

// const char * MyStringGetPrevCharPointer(const char *base, const char *p) throw() { return CharPrevA(base, p); }
// char * MyStringUpper(char *s) { return CharUpperA(s); }
// char * MyStringLower(char *s) { return CharLowerA(s); }

wchar_t MyCharUpper_WIN(wchar_t c) throw()
{
  wchar_t *res = CharUpperW((LPWSTR)(UINT_PTR)(unsigned)c);
  if (res != 0 || ::GetLastError() != ERROR_CALL_NOT_IMPLEMENTED)
    return (wchar_t)(unsigned)(UINT_PTR)res;
  const int kBufSize = 4;
  char s[kBufSize + 1];
  int numChars = ::WideCharToMultiByte(CP_ACP, 0, &c, 1, s, kBufSize, 0, 0);
  if (numChars == 0 || numChars > kBufSize)
    return c;
  s[numChars] = 0;
  ::CharUpperA(s);
  ::MultiByteToWideChar(CP_ACP, 0, s, numChars, &c, 1);
  return c;
}

/*
wchar_t MyCharLower_WIN(wchar_t c)
{
  wchar_t *res = CharLowerW((LPWSTR)(UINT_PTR)(unsigned)c);
  if (res != 0 || ::GetLastError() != ERROR_CALL_NOT_IMPLEMENTED)
    return (wchar_t)(unsigned)(UINT_PTR)res;
  const int kBufSize = 4;
  char s[kBufSize + 1];
  int numChars = ::WideCharToMultiByte(CP_ACP, 0, &c, 1, s, kBufSize, 0, 0);
  if (numChars == 0 || numChars > kBufSize)
    return c;
  s[numChars] = 0;
  ::CharLowerA(s);
  ::MultiByteToWideChar(CP_ACP, 0, s, numChars, &c, 1);
  return c;
}
*/

/*
wchar_t * MyStringUpper(wchar_t *s)
{
  if (s == 0)
    return 0;
  wchar_t *res = CharUpperW(s);
  if (res != 0 || ::GetLastError() != ERROR_CALL_NOT_IMPLEMENTED)
    return res;
  AString a = UnicodeStringToMultiByte(s);
  a.MakeUpper();
  MyStringCopy(s, (const wchar_t *)MultiByteToUnicodeString(a));
  return s;
}
*/

/*
wchar_t * MyStringLower(wchar_t *s)
{
  if (s == 0)
    return 0;
  wchar_t *res = CharLowerW(s);
  if (res != 0 || ::GetLastError() != ERROR_CALL_NOT_IMPLEMENTED)
    return res;
  AString a = UnicodeStringToMultiByte(s);
  a.MakeLower();
  MyStringCopy(s, (const wchar_t *)MultiByteToUnicodeString(a));
  return s;
}
*/

#endif

#endif

bool IsString1PrefixedByString2(const char *s1, const char *s2) throw()
{
  for (;;)
  {
    const char c2 = *s2++; if (c2 == 0) return true;
    const char c1 = *s1++; if (c1 != c2) return false;
  }
}

bool StringsAreEqualNoCase(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    const wchar_t c1 = *s1++;
    const wchar_t c2 = *s2++;
    if (c1 != c2 && MyCharUpper(c1) != MyCharUpper(c2)) return false;
    if (c1 == 0) return true;
  }
}

// ---------- ASCII ----------

bool StringsAreEqual_Ascii(const char *u, const char *a) throw()
{
  for (;;)
  {
    const char c = *a;
    if (c != *u)
      return false;
    if (c == 0)
      return true;
    a++;
    u++;
  }
}

bool StringsAreEqual_Ascii(const wchar_t *u, const char *a) throw()
{
  for (;;)
  {
    const unsigned char c = (unsigned char)*a;
    if (c != *u)
      return false;
    if (c == 0)
      return true;
    a++;
    u++;
  }
}

bool StringsAreEqualNoCase_Ascii(const char *s1, const char *s2) throw()
{
  for (;;)
  {
    const char c1 = *s1++;
    const char c2 = *s2++;
    if (c1 != c2 && MyCharLower_Ascii(c1) != MyCharLower_Ascii(c2))
      return false;
    if (c1 == 0)
      return true;
  }
}

bool StringsAreEqualNoCase_Ascii(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    const wchar_t c1 = *s1++;
    const wchar_t c2 = *s2++;
    if (c1 != c2 && MyCharLower_Ascii(c1) != MyCharLower_Ascii(c2))
      return false;
    if (c1 == 0)
      return true;
  }
}

bool StringsAreEqualNoCase_Ascii(const wchar_t *s1, const char *s2) throw()
{
  for (;;)
  {
    const wchar_t c1 = *s1++;
    const char c2 = *s2++;
    if (c1 != (unsigned char)c2 && (c1 > 0x7F || MyCharLower_Ascii(c1) != (unsigned char)MyCharLower_Ascii(c2)))
      return false;
    if (c1 == 0)
      return true;
  }
}

bool IsString1PrefixedByString2(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    const wchar_t c2 = *s2++; if (c2 == 0) return true;
    const wchar_t c1 = *s1++; if (c1 != c2) return false;
  }
}

bool IsString1PrefixedByString2(const wchar_t *s1, const char *s2) throw()
{
  for (;;)
  {
    const unsigned char c2 = (unsigned char)(*s2++); if (c2 == 0) return true;
    const wchar_t c1 = *s1++; if (c1 != c2) return false;
  }
}

bool IsString1PrefixedByString2_NoCase_Ascii(const char *s1, const char *s2) throw()
{
  for (;;)
  {
    const char c2 = *s2++; if (c2 == 0) return true;
    const char c1 = *s1++;
    if (c1 != c2 && MyCharLower_Ascii(c1) != MyCharLower_Ascii(c2))
      return false;
  }
}

bool IsString1PrefixedByString2_NoCase_Ascii(const wchar_t *s1, const char *s2) throw()
{
  for (;;)
  {
    const char c2 = *s2++; if (c2 == 0) return true;
    const wchar_t c1 = *s1++;
    if (c1 != (unsigned char)c2 && MyCharLower_Ascii(c1) != (unsigned char)MyCharLower_Ascii(c2))
      return false;
  }
}

bool IsString1PrefixedByString2_NoCase(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    const wchar_t c2 = *s2++; if (c2 == 0) return true;
    const wchar_t c1 = *s1++;
    if (c1 != c2 && MyCharUpper(c1) != MyCharUpper(c2))
      return false;
  }
}

// NTFS order: uses upper case
int MyStringCompareNoCase(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    const wchar_t c1 = *s1++;
    const wchar_t c2 = *s2++;
    if (c1 != c2)
    {
      const wchar_t u1 = MyCharUpper(c1);
      const wchar_t u2 = MyCharUpper(c2);
      if (u1 < u2) return -1;
      if (u1 > u2) return 1;
    }
    if (c1 == 0) return 0;
  }
}

/*
int MyStringCompareNoCase_N(const wchar_t *s1, const wchar_t *s2, unsigned num)
{
  for (; num != 0; num--)
  {
    wchar_t c1 = *s1++;
    wchar_t c2 = *s2++;
    if (c1 != c2)
    {
      wchar_t u1 = MyCharUpper(c1);
      wchar_t u2 = MyCharUpper(c2);
      if (u1 < u2) return -1;
      if (u1 > u2) return 1;
    }
    if (c1 == 0) return 0;
  }
  return 0;
}
*/

// ---------- AString ----------

void AString::InsertSpace(unsigned &index, unsigned size)
{
  Grow(size);
  MoveItems(index + size, index);
}

#define k_Alloc_Len_Limit (0x40000000 - 2)
// #define k_Alloc_Len_Limit (((unsigned)1 << (sizeof(unsigned) * 8 - 2)) - 2)

void AString::ReAlloc(unsigned newLimit)
{
  // MY_STRING_REALLOC(_chars, char, (size_t)newLimit + 1, (size_t)_len + 1);
  char *newBuf = MY_STRING_NEW_char((size_t)newLimit + 1);
  memcpy(newBuf, _chars, (size_t)_len + 1);
  MY_STRING_DELETE(_chars)
  _chars = newBuf;
  _limit = newLimit;
}

#define THROW_STRING_ALLOC_EXCEPTION  { throw 20130220; }

#define CHECK_STRING_ALLOC_LEN(len) \
  { if ((len) > k_Alloc_Len_Limit) THROW_STRING_ALLOC_EXCEPTION }

void AString::ReAlloc2(unsigned newLimit)
{
  CHECK_STRING_ALLOC_LEN(newLimit)
  // MY_STRING_REALLOC(_chars, char, (size_t)newLimit + 1, 0);
  char *newBuf = MY_STRING_NEW_char((size_t)newLimit + 1);
  newBuf[0] = 0;
  MY_STRING_DELETE(_chars)
  _chars = newBuf;
  _limit = newLimit;
  _len = 0;
}

void AString::SetStartLen(unsigned len)
{
  _chars = NULL;
  _chars = MY_STRING_NEW_char((size_t)len + 1);
  _len = len;
  _limit = len;
}

Z7_NO_INLINE
void AString::Grow_1()
{
  unsigned next = _len;
  next += next / 2;
  next += 16;
  next &= ~(unsigned)15;
  next--;
  if (next < _len || next > k_Alloc_Len_Limit)
    next = k_Alloc_Len_Limit;
  if (next <= _len)
    THROW_STRING_ALLOC_EXCEPTION
  ReAlloc(next);
  // Grow(1);
}

void AString::Grow(unsigned n)
{
  const unsigned freeSize = _limit - _len;
  if (n <= freeSize)
    return;
  unsigned next = _len + n;
  next += next / 2;
  next += 16;
  next &= ~(unsigned)15;
  next--;
  if (next < _len || next > k_Alloc_Len_Limit)
    next = k_Alloc_Len_Limit;
  if (next <= _len || next - _len < n)
    THROW_STRING_ALLOC_EXCEPTION
  ReAlloc(next);
}

AString::AString(unsigned num, const char *s)
{
  unsigned len = MyStringLen(s);
  if (num > len)
    num = len;
  SetStartLen(num);
  memcpy(_chars, s, num);
  _chars[num] = 0;
}

AString::AString(unsigned num, const AString &s)
{
  if (num > s._len)
    num = s._len;
  SetStartLen(num);
  memcpy(_chars, s._chars, num);
  _chars[num] = 0;
}

AString::AString(const AString &s, char c)
{
  SetStartLen(s.Len() + 1);
  char *chars = _chars;
  unsigned len = s.Len();
  memcpy(chars, s, len);
  chars[len] = c;
  chars[(size_t)len + 1] = 0;
}

AString::AString(const char *s1, unsigned num1, const char *s2, unsigned num2)
{
  SetStartLen(num1 + num2);
  char *chars = _chars;
  memcpy(chars, s1, num1);
  memcpy(chars + num1, s2, num2 + 1);
}

AString operator+(const AString &s1, const AString &s2) { return AString(s1, s1.Len(), s2, s2.Len()); }
AString operator+(const AString &s1, const char    *s2) { return AString(s1, s1.Len(), s2, MyStringLen(s2)); }
AString operator+(const char    *s1, const AString &s2) { return AString(s1, MyStringLen(s1), s2, s2.Len()); }

static const unsigned kStartStringCapacity = 4;
 
AString::AString()
{
  _chars = NULL;
  _chars = MY_STRING_NEW_char(kStartStringCapacity);
  _len = 0;
  _limit = kStartStringCapacity - 1;
  _chars[0] = 0;
}

AString::AString(char c)
{
  SetStartLen(1);
  char *chars = _chars;
  chars[0] = c;
  chars[1] = 0;
}

AString::AString(const char *s)
{
  SetStartLen(MyStringLen(s));
  MyStringCopy(_chars, s);
}

AString::AString(const AString &s)
{
  SetStartLen(s._len);
  MyStringCopy(_chars, s._chars);
}

AString &AString::operator=(char c)
{
  if (1 > _limit)
  {
    char *newBuf = MY_STRING_NEW_char(1 + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = 1;
  }
  _len = 1;
  char *chars = _chars;
  chars[0] = c;
  chars[1] = 0;
  return *this;
}

AString &AString::operator=(const char *s)
{
  unsigned len = MyStringLen(s);
  if (len > _limit)
  {
    char *newBuf = MY_STRING_NEW_char((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;
  MyStringCopy(_chars, s);
  return *this;
}

AString &AString::operator=(const AString &s)
{
  if (&s == this)
    return *this;
  unsigned len = s._len;
  if (len > _limit)
  {
    char *newBuf = MY_STRING_NEW_char((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;
  MyStringCopy(_chars, s._chars);
  return *this;
}

void AString::SetFromWStr_if_Ascii(const wchar_t *s)
{
  unsigned len = 0;
  {
    for (;; len++)
    {
      wchar_t c = s[len];
      if (c == 0)
        break;
      if (c >= 0x80)
        return;
    }
  }
  if (len > _limit)
  {
    char *newBuf = MY_STRING_NEW_char((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;
  char *dest = _chars;
  unsigned i;
  for (i = 0; i < len; i++)
    dest[i] = (char)s[i];
  dest[i] = 0;
}

/*
void AString::SetFromBstr_if_Ascii(BSTR s)
{
  unsigned len = ::SysStringLen(s);
  {
    for (unsigned i = 0; i < len; i++)
      if (s[i] <= 0 || s[i] >= 0x80)
        return;
  }
  if (len > _limit)
  {
    char *newBuf = MY_STRING_NEW_char((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;
  char *dest = _chars;
  unsigned i;
  for (i = 0; i < len; i++)
    dest[i] = (char)s[i];
  dest[i] = 0;
}
*/

void AString::Add_Char(char c) { operator+=(c); }
void AString::Add_Space() { operator+=(' '); }
void AString::Add_Space_if_NotEmpty() { if (!IsEmpty()) Add_Space(); }
void AString::Add_LF() { operator+=('\n'); }
void AString::Add_Slash() { operator+=('/'); }
void AString::Add_Dot() { operator+=('.'); }
void AString::Add_Minus() { operator+=('-'); }
void AString::Add_Colon() { operator+=(':'); }

AString &AString::operator+=(const char *s)
{
  unsigned len = MyStringLen(s);
  Grow(len);
  MyStringCopy(_chars + _len, s);
  _len += len;
  return *this;
}

void AString::Add_OptSpaced(const char *s)
{
  Add_Space_if_NotEmpty();
  (*this) += s;
}

AString &AString::operator+=(const AString &s)
{
  Grow(s._len);
  MyStringCopy(_chars + _len, s._chars);
  _len += s._len;
  return *this;
}

void AString::Add_UInt32(UInt32 v)
{
  Grow(10);
  _len = (unsigned)(ConvertUInt32ToString(v, _chars + _len) - _chars);
}

void UString::Add_UInt64(UInt64 v)
{
  Grow(20);
  _len = (unsigned)(ConvertUInt64ToString(v, _chars + _len) - _chars);
}

void AString::AddFrom(const char *s, unsigned len) // no check
{
  if (len != 0)
  {
    Grow(len);
    memcpy(_chars + _len, s, len);
    len += _len;
    _chars[len] = 0;
    _len = len;
  }
}

void AString::SetFrom(const char *s, unsigned len) // no check
{
  if (len > _limit)
  {
    CHECK_STRING_ALLOC_LEN(len)
    char *newBuf = MY_STRING_NEW_char((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  if (len != 0)
    memcpy(_chars, s, len);
  _chars[len] = 0;
  _len = len;
}

void AString::SetFrom_Chars_SizeT(const char *s, size_t len)
{
  CHECK_STRING_ALLOC_LEN(len)
  SetFrom(s, (unsigned)len);
}

void AString::SetFrom_CalcLen(const char *s, unsigned len) // no check
{
  unsigned i;
  for (i = 0; i < len; i++)
    if (s[i] == 0)
      break;
  SetFrom(s, i);
}

int AString::Find(const char *s, unsigned startIndex) const throw()
{
  const char *fs = strstr(_chars + startIndex, s);
  if (!fs)
    return -1;
  return (int)(fs - _chars);

  /*
  if (s[0] == 0)
    return startIndex;
  unsigned len = MyStringLen(s);
  const char *p = _chars + startIndex;
  for (;; p++)
  {
    const char c = *p;
    if (c != s[0])
    {
      if (c == 0)
        return -1;
      continue;
    }
    unsigned i;
    for (i = 1; i < len; i++)
      if (p[i] != s[i])
        break;
    if (i == len)
      return (int)(p - _chars);
  }
  */
}

int AString::ReverseFind(char c) const throw()
{
  if (_len == 0)
    return -1;
  const char *p = _chars + _len - 1;
  for (;;)
  {
    if (*p == c)
      return (int)(p - _chars);
    if (p == _chars)
      return -1;
    p--; // p = GetPrevCharPointer(_chars, p);
  }
}

int AString::ReverseFind_PathSepar() const throw()
{
  if (_len == 0)
    return -1;
  const char *p = _chars + _len - 1;
  for (;;)
  {
    const char c = *p;
    if (IS_PATH_SEPAR(c))
      return (int)(p - _chars);
    if (p == _chars)
      return -1;
    p--;
  }
}

void AString::TrimLeft() throw()
{
  const char *p = _chars;
  for (;; p++)
  {
    char c = *p;
    if (c != ' ' && c != '\n' && c != '\t')
      break;
  }
  unsigned pos = (unsigned)(p - _chars);
  if (pos != 0)
  {
    MoveItems(0, pos);
    _len -= pos;
  }
}

void AString::TrimRight() throw()
{
  const char *p = _chars;
  unsigned i;
  for (i = _len; i != 0; i--)
  {
    char c = p[(size_t)i - 1];
    if (c != ' ' && c != '\n' && c != '\t')
      break;
  }
  if (i != _len)
  {
    _chars[i] = 0;
    _len = i;
  }
}

void AString::InsertAtFront(char c)
{
  if (_limit == _len)
    Grow_1();
  MoveItems(1, 0);
  _chars[0] = c;
  _len++;
}

/*
void AString::Insert(unsigned index, char c)
{
  InsertSpace(index, 1);
  _chars[index] = c;
  _len++;
}
*/

void AString::Insert(unsigned index, const char *s)
{
  unsigned num = MyStringLen(s);
  if (num != 0)
  {
    InsertSpace(index, num);
    memcpy(_chars + index, s, num);
    _len += num;
  }
}

void AString::Insert(unsigned index, const AString &s)
{
  unsigned num = s.Len();
  if (num != 0)
  {
    InsertSpace(index, num);
    memcpy(_chars + index, s, num);
    _len += num;
  }
}

void AString::RemoveChar(char ch) throw()
{
  char *src = _chars;
  
  for (;;)
  {
    char c = *src++;
    if (c == 0)
      return;
    if (c == ch)
      break;
  }

  char *dest = src - 1;
  
  for (;;)
  {
    char c = *src++;
    if (c == 0)
      break;
    if (c != ch)
      *dest++ = c;
  }
  
  *dest = 0;
  _len = (unsigned)(dest - _chars);
}

// !!!!!!!!!!!!!!! test it if newChar = '\0'
void AString::Replace(char oldChar, char newChar) throw()
{
  if (oldChar == newChar)
    return; // 0;
  // unsigned number = 0;
  int pos = 0;
  char *chars = _chars;
  while ((unsigned)pos < _len)
  {
    pos = Find(oldChar, (unsigned)pos);
    if (pos < 0)
      break;
    chars[(unsigned)pos] = newChar;
    pos++;
    // number++;
  }
  return; //  number;
}

void AString::Replace(const AString &oldString, const AString &newString)
{
  if (oldString.IsEmpty())
    return; // 0;
  if (oldString == newString)
    return; // 0;
  const unsigned oldLen = oldString.Len();
  const unsigned newLen = newString.Len();
  // unsigned number = 0;
  int pos = 0;
  while ((unsigned)pos < _len)
  {
    pos = Find(oldString, (unsigned)pos);
    if (pos < 0)
      break;
    Delete((unsigned)pos, oldLen);
    Insert((unsigned)pos, newString);
    pos += newLen;
    // number++;
  }
  // return number;
}

void AString::Delete(unsigned index) throw()
{
  MoveItems(index, index + 1);
  _len--;
}

void AString::Delete(unsigned index, unsigned count) throw()
{
  if (index + count > _len)
    count = _len - index;
  if (count > 0)
  {
    MoveItems(index, index + count);
    _len -= count;
  }
}

void AString::DeleteFrontal(unsigned num) throw()
{
  if (num != 0)
  {
    MoveItems(0, num);
    _len -= num;
  }
}

/*
AString operator+(const AString &s1, const AString &s2)
{
  AString result(s1);
  result += s2;
  return result;
}

AString operator+(const AString &s, const char *chars)
{
  AString result(s);
  result += chars;
  return result;
}

AString operator+(const char *chars, const AString &s)
{
  AString result(chars);
  result += s;
  return result;
}

AString operator+(const AString &s, char c)
{
  AString result(s);
  result += c;
  return result;
}
*/

/*
AString operator+(char c, const AString &s)
{
  AString result(c);
  result += s;
  return result;
}
*/




// ---------- UString ----------

void UString::InsertSpace(unsigned index, unsigned size)
{
  Grow(size);
  MoveItems(index + size, index);
}

void UString::ReAlloc(unsigned newLimit)
{
  // MY_STRING_REALLOC(_chars, wchar_t, (size_t)newLimit + 1, (size_t)_len + 1);
  wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)newLimit + 1);
  wmemcpy(newBuf, _chars, _len + 1);
  MY_STRING_DELETE(_chars)
  _chars = newBuf;
  _limit = newLimit;
}

void UString::ReAlloc2(unsigned newLimit)
{
  CHECK_STRING_ALLOC_LEN(newLimit)
  // MY_STRING_REALLOC(_chars, wchar_t, newLimit + 1, 0);
  wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)newLimit + 1);
  newBuf[0] = 0;
  MY_STRING_DELETE(_chars)
  _chars = newBuf;
  _limit = newLimit;
  _len = 0;
}

void UString::SetStartLen(unsigned len)
{
  _chars = NULL;
  _chars = MY_STRING_NEW_wchar_t((size_t)len + 1);
  _len = len;
  _limit = len;
}

Z7_NO_INLINE
void UString::Grow_1()
{
  unsigned next = _len;
  next += next / 2;
  next += 16;
  next &= ~(unsigned)15;
  next--;
  if (next < _len || next > k_Alloc_Len_Limit)
    next = k_Alloc_Len_Limit;
  if (next <= _len)
    THROW_STRING_ALLOC_EXCEPTION
  ReAlloc(next);
}

void UString::Grow(unsigned n)
{
  const unsigned freeSize = _limit - _len;
  if (n <= freeSize)
    return;
  unsigned next = _len + n;
  next += next / 2;
  next += 16;
  next &= ~(unsigned)15;
  next--;
  if (next < _len || next > k_Alloc_Len_Limit)
    next = k_Alloc_Len_Limit;
  if (next <= _len || next - _len < n)
    THROW_STRING_ALLOC_EXCEPTION
  ReAlloc(next - 1);
}


UString::UString(unsigned num, const wchar_t *s)
{
  unsigned len = MyStringLen(s);
  if (num > len)
    num = len;
  SetStartLen(num);
  wmemcpy(_chars, s, num);
  _chars[num] = 0;
}


UString::UString(unsigned num, const UString &s)
{
  if (num > s._len)
    num = s._len;
  SetStartLen(num);
  wmemcpy(_chars, s._chars, num);
  _chars[num] = 0;
}

UString::UString(const UString &s, wchar_t c)
{
  SetStartLen(s.Len() + 1);
  wchar_t *chars = _chars;
  unsigned len = s.Len();
  wmemcpy(chars, s, len);
  chars[len] = c;
  chars[(size_t)len + 1] = 0;
}

UString::UString(const wchar_t *s1, unsigned num1, const wchar_t *s2, unsigned num2)
{
  SetStartLen(num1 + num2);
  wchar_t *chars = _chars;
  wmemcpy(chars, s1, num1);
  wmemcpy(chars + num1, s2, num2 + 1);
}

UString operator+(const UString &s1, const UString &s2) { return UString(s1, s1.Len(), s2, s2.Len()); }
UString operator+(const UString &s1, const wchar_t *s2) { return UString(s1, s1.Len(), s2, MyStringLen(s2)); }
UString operator+(const wchar_t *s1, const UString &s2) { return UString(s1, MyStringLen(s1), s2, s2.Len()); }

UString::UString()
{
  _chars = NULL;
  _chars = MY_STRING_NEW_wchar_t(kStartStringCapacity);
  _len = 0;
  _limit = kStartStringCapacity - 1;
  _chars[0] = 0;
}

UString::UString(wchar_t c)
{
  SetStartLen(1);
  wchar_t *chars = _chars;
  chars[0] = c;
  chars[1] = 0;
}

UString::UString(char c)
{
  SetStartLen(1);
  wchar_t *chars = _chars;
  chars[0] = (unsigned char)c;
  chars[1] = 0;
}

UString::UString(const wchar_t *s)
{
  const unsigned len = MyStringLen(s);
  SetStartLen(len);
  wmemcpy(_chars, s, len + 1);
}

UString::UString(const char *s)
{
  const unsigned len = MyStringLen(s);
  SetStartLen(len);
  wchar_t *chars = _chars;
  for (unsigned i = 0; i < len; i++)
    chars[i] = (unsigned char)s[i];
  chars[len] = 0;
}

UString::UString(const AString &s)
{
  const unsigned len = s.Len();
  SetStartLen(len);
  wchar_t *chars = _chars;
  const char *s2 = s.Ptr();
  for (unsigned i = 0; i < len; i++)
    chars[i] = (unsigned char)s2[i];
  chars[len] = 0;
}

UString::UString(const UString &s)
{
  SetStartLen(s._len);
  wmemcpy(_chars, s._chars, s._len + 1);
}

UString &UString::operator=(wchar_t c)
{
  if (1 > _limit)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t(1 + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = 1;
  }
  _len = 1;
  wchar_t *chars = _chars;
  chars[0] = c;
  chars[1] = 0;
  return *this;
}

UString &UString::operator=(const wchar_t *s)
{
  unsigned len = MyStringLen(s);
  if (len > _limit)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;
  wmemcpy(_chars, s, len + 1);
  return *this;
}

UString &UString::operator=(const UString &s)
{
  if (&s == this)
    return *this;
  unsigned len = s._len;
  if (len > _limit)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;
  wmemcpy(_chars, s._chars, len + 1);
  return *this;
}

void UString::SetFrom(const wchar_t *s, unsigned len) // no check
{
  if (len > _limit)
  {
    CHECK_STRING_ALLOC_LEN(len)
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  if (len != 0)
    wmemcpy(_chars, s, len);
  _chars[len] = 0;
  _len = len;
}

void UString::SetFromBstr(LPCOLESTR s)
{
  unsigned len = ::SysStringLen((BSTR)(void *)(s));

  /*
  #if WCHAR_MAX > 0xffff
  size_t num_wchars = 0;
  for (size_t i = 0; i < len;)
  {
    wchar_t c = s[i++];
    if (c >= 0xd800 && c < 0xdc00 && i + 1 != len)
    {
      wchar_t c2 = s[i];
      if (c2 >= 0xdc00 && c2 < 0xe000)
      {
        c = 0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff);
        i++;
      }
    }
    num_wchars++;
  }
  len = num_wchars;
  #endif
  */

  if (len > _limit)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  _len = len;

  /*
  #if WCHAR_MAX > 0xffff

  wchar_t *chars = _chars;
  for (size_t i = 0; i <= len; i++)
  {
    wchar_t c = *s++;
    if (c >= 0xd800 && c < 0xdc00 && i + 1 != len)
    {
      wchar_t c2 = *s;
      if (c2 >= 0xdc00 && c2 < 0xe000)
      {
        s++;
        c = 0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff);
      }
    }
    chars[i] = c;
  }

  #else
  */

  // if (s)
    wmemcpy(_chars, s, len + 1);
  
  // #endif
}

UString &UString::operator=(const char *s)
{
  unsigned len = MyStringLen(s);
  if (len > _limit)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    MY_STRING_DELETE(_chars)
    _chars = newBuf;
    _limit = len;
  }
  wchar_t *chars = _chars;
  for (unsigned i = 0; i < len; i++)
    chars[i] = (unsigned char)s[i];
  chars[len] = 0;
  _len = len;
  return *this;
}

void UString::Add_Char(char c) { operator+=((wchar_t)(unsigned char)c); }
// void UString::Add_WChar(wchar_t c) { operator+=(c); }
void UString::Add_Dot() { operator+=(L'.'); }
void UString::Add_Space() { operator+=(L' '); }
void UString::Add_Minus() { operator+=(L'-'); }
void UString::Add_Colon() { operator+=(L':'); }
void UString::Add_LF() { operator+=(L'\n'); }
void UString::Add_Space_if_NotEmpty() { if (!IsEmpty()) Add_Space(); }

UString &UString::operator+=(const wchar_t *s)
{
  unsigned len = MyStringLen(s);
  Grow(len);
  wmemcpy(_chars + _len, s, len + 1);
  _len += len;
  return *this;
}

UString &UString::operator+=(const UString &s)
{
  Grow(s._len);
  wmemcpy(_chars + _len, s._chars, s._len + 1);
  _len += s._len;
  return *this;
}

UString &UString::operator+=(const char *s)
{
  unsigned len = MyStringLen(s);
  Grow(len);
  wchar_t *chars = _chars + _len;
  for (unsigned i = 0; i < len; i++)
    chars[i] = (unsigned char)s[i];
  chars[len] = 0;
  _len += len;
  return *this;
}


void UString::Add_UInt32(UInt32 v)
{
  Grow(10);
  _len = (unsigned)(ConvertUInt32ToString(v, _chars + _len) - _chars);
}

void AString::Add_UInt64(UInt64 v)
{
  Grow(20);
  _len = (unsigned)(ConvertUInt64ToString(v, _chars + _len) - _chars);
}


int UString::Find(const wchar_t *s, unsigned startIndex) const throw()
{
  const wchar_t *fs = wcsstr(_chars + startIndex, s);
  if (!fs)
    return -1;
  return (int)(fs - _chars);

  /*
  if (s[0] == 0)
    return startIndex;
  unsigned len = MyStringLen(s);
  const wchar_t *p = _chars + startIndex;
  for (;; p++)
  {
    const wchar_t c = *p;
    if (c != s[0])
    {
      if (c == 0)
        return -1;
      continue;
    }
    unsigned i;
    for (i = 1; i < len; i++)
      if (p[i] != s[i])
        break;
    if (i == len)
      return (int)(p - _chars);
  }
  */
}

int UString::ReverseFind(wchar_t c) const throw()
{
  if (_len == 0)
    return -1;
  const wchar_t *p = _chars + _len;
  do
  {
    if (*(--p) == c)
      return (int)(p - _chars);
  }
  while (p != _chars);
  return -1;
}

int UString::ReverseFind_PathSepar() const throw()
{
  const wchar_t *p = _chars + _len;
  while (p != _chars)
  {
    const wchar_t c = *(--p);
    if (IS_PATH_SEPAR(c))
      return (int)(p - _chars);
  }
  return -1;
}

void UString::TrimLeft() throw()
{
  const wchar_t *p = _chars;
  for (;; p++)
  {
    wchar_t c = *p;
    if (c != ' ' && c != '\n' && c != '\t')
      break;
  }
  unsigned pos = (unsigned)(p - _chars);
  if (pos != 0)
  {
    MoveItems(0, pos);
    _len -= pos;
  }
}

void UString::TrimRight() throw()
{
  const wchar_t *p = _chars;
  unsigned i;
  for (i = _len; i != 0; i--)
  {
    wchar_t c = p[(size_t)i - 1];
    if (c != ' ' && c != '\n' && c != '\t')
      break;
  }
  if (i != _len)
  {
    _chars[i] = 0;
    _len = i;
  }
}

void UString::InsertAtFront(wchar_t c)
{
  if (_limit == _len)
    Grow_1();
  MoveItems(1, 0);
  _chars[0] = c;
  _len++;
}

/*
void UString::Insert_wchar_t(unsigned index, wchar_t c)
{
  InsertSpace(index, 1);
  _chars[index] = c;
  _len++;
}
*/

void UString::Insert(unsigned index, const wchar_t *s)
{
  unsigned num = MyStringLen(s);
  if (num != 0)
  {
    InsertSpace(index, num);
    wmemcpy(_chars + index, s, num);
    _len += num;
  }
}

void UString::Insert(unsigned index, const UString &s)
{
  unsigned num = s.Len();
  if (num != 0)
  {
    InsertSpace(index, num);
    wmemcpy(_chars + index, s, num);
    _len += num;
  }
}

void UString::RemoveChar(wchar_t ch) throw()
{
  wchar_t *src = _chars;
  
  for (;;)
  {
    wchar_t c = *src++;
    if (c == 0)
      return;
    if (c == ch)
      break;
  }

  wchar_t *dest = src - 1;
  
  for (;;)
  {
    wchar_t c = *src++;
    if (c == 0)
      break;
    if (c != ch)
      *dest++ = c;
  }
  
  *dest = 0;
  _len = (unsigned)(dest - _chars);
}

// !!!!!!!!!!!!!!! test it if newChar = '\0'
void UString::Replace(wchar_t oldChar, wchar_t newChar) throw()
{
  if (oldChar == newChar)
    return; // 0;
  // unsigned number = 0;
  int pos = 0;
  wchar_t *chars = _chars;
  while ((unsigned)pos < _len)
  {
    pos = Find(oldChar, (unsigned)pos);
    if (pos < 0)
      break;
    chars[(unsigned)pos] = newChar;
    pos++;
    // number++;
  }
  return; //  number;
}

void UString::Replace(const UString &oldString, const UString &newString)
{
  if (oldString.IsEmpty())
    return; // 0;
  if (oldString == newString)
    return; // 0;
  unsigned oldLen = oldString.Len();
  unsigned newLen = newString.Len();
  // unsigned number = 0;
  int pos = 0;
  while ((unsigned)pos < _len)
  {
    pos = Find(oldString, (unsigned)pos);
    if (pos < 0)
      break;
    Delete((unsigned)pos, oldLen);
    Insert((unsigned)pos, newString);
    pos += newLen;
    // number++;
  }
  // return number;
}

void UString::Delete(unsigned index) throw()
{
  MoveItems(index, index + 1);
  _len--;
}

void UString::Delete(unsigned index, unsigned count) throw()
{
  if (index + count > _len)
    count = _len - index;
  if (count > 0)
  {
    MoveItems(index, index + count);
    _len -= count;
  }
}

void UString::DeleteFrontal(unsigned num) throw()
{
  if (num != 0)
  {
    MoveItems(0, num);
    _len -= num;
  }
}


// ---------- UString2 ----------

void UString2::ReAlloc2(unsigned newLimit)
{
  // wrong (_len) is allowed after this function
  CHECK_STRING_ALLOC_LEN(newLimit)
  // MY_STRING_REALLOC(_chars, wchar_t, newLimit + 1, 0);
  if (_chars)
  {
    MY_STRING_DELETE(_chars)
    _chars = NULL;
    // _len = 0;
  }
  _chars = MY_STRING_NEW_wchar_t((size_t)newLimit + 1);
  _chars[0] = 0;
  // _len = newLimit;
}

void UString2::SetStartLen(unsigned len)
{
  _chars = NULL;
  _chars = MY_STRING_NEW_wchar_t((size_t)len + 1);
  _len = len;
}


/*
UString2::UString2(wchar_t c)
{
  SetStartLen(1);
  wchar_t *chars = _chars;
  chars[0] = c;
  chars[1] = 0;
}
*/

UString2::UString2(const wchar_t *s)
{
  const unsigned len = MyStringLen(s);
  SetStartLen(len);
  wmemcpy(_chars, s, len + 1);
}

UString2::UString2(const UString2 &s): _chars(NULL), _len(0)
{
  if (s._chars)
  {
    SetStartLen(s._len);
    wmemcpy(_chars, s._chars, s._len + 1);
  }
}

/*
UString2 &UString2::operator=(wchar_t c)
{
  if (1 > _len)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t(1 + 1);
    if (_chars)
      MY_STRING_DELETE(_chars)
    _chars = newBuf;
  }
  _len = 1;
  wchar_t *chars = _chars;
  chars[0] = c;
  chars[1] = 0;
  return *this;
}
*/

UString2 &UString2::operator=(const wchar_t *s)
{
  unsigned len = MyStringLen(s);
  if (len > _len)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    if (_chars)
      MY_STRING_DELETE(_chars)
    _chars = newBuf;
  }
  _len = len;
  MyStringCopy(_chars, s);
  return *this;
}

void UString2::SetFromAscii(const char *s)
{
  unsigned len = MyStringLen(s);
  if (len > _len)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    if (_chars)
      MY_STRING_DELETE(_chars)
    _chars = newBuf;
  }
  wchar_t *chars = _chars;
  for (unsigned i = 0; i < len; i++)
    chars[i] = (unsigned char)s[i];
  chars[len] = 0;
  _len = len;
}

UString2 &UString2::operator=(const UString2 &s)
{
  if (&s == this)
    return *this;
  unsigned len = s._len;
  if (len > _len)
  {
    wchar_t *newBuf = MY_STRING_NEW_wchar_t((size_t)len + 1);
    if (_chars)
      MY_STRING_DELETE(_chars)
    _chars = newBuf;
  }
  _len = len;
  MyStringCopy(_chars, s._chars);
  return *this;
}

bool operator==(const UString2 &s1, const UString2 &s2)
{
  return s1.Len() == s2.Len() && (s1.IsEmpty() || wcscmp(s1.GetRawPtr(), s2.GetRawPtr()) == 0);
}

bool operator==(const UString2 &s1, const wchar_t *s2)
{
  if (s1.IsEmpty())
    return (*s2 == 0);
  return wcscmp(s1.GetRawPtr(), s2) == 0;
}

bool operator==(const wchar_t *s1, const UString2 &s2)
{
  if (s2.IsEmpty())
    return (*s1 == 0);
  return wcscmp(s1, s2.GetRawPtr()) == 0;
}



// ----------------------------------------

/*
int MyStringCompareNoCase(const char *s1, const char *s2)
{
  return MyStringCompareNoCase(MultiByteToUnicodeString(s1), MultiByteToUnicodeString(s2));
}
*/

#if !defined(USE_UNICODE_FSTRING) || !defined(_UNICODE)

static inline UINT GetCurrentCodePage()
{
  #if defined(UNDER_CE) || !defined(_WIN32)
  return CP_ACP;
  #else
  return ::AreFileApisANSI() ? CP_ACP : CP_OEMCP;
  #endif
}

#endif

#ifdef USE_UNICODE_FSTRING

#ifndef _UNICODE

AString fs2fas(CFSTR s)
{
  return UnicodeStringToMultiByte(s, GetCurrentCodePage());
}

FString fas2fs(const char *s)
{
  return MultiByteToUnicodeString(s, GetCurrentCodePage());
}

FString fas2fs(const AString &s)
{
  return MultiByteToUnicodeString(s, GetCurrentCodePage());
}

#endif //  _UNICODE

#else // USE_UNICODE_FSTRING

UString fs2us(const FChar *s)
{
  return MultiByteToUnicodeString(s, GetCurrentCodePage());
}

UString fs2us(const FString &s)
{
  return MultiByteToUnicodeString(s, GetCurrentCodePage());
}

FString us2fs(const wchar_t *s)
{
  return UnicodeStringToMultiByte(s, GetCurrentCodePage());
}

#endif // USE_UNICODE_FSTRING


bool CStringFinder::FindWord_In_LowCaseAsciiList_NoCase(const char *p, const wchar_t *str)
{
  _temp.Empty();
  for (;;)
  {
    const wchar_t c = *str++;
    if (c == 0)
      break;
    if (c <= 0x20 || c > 0x7f)
      return false;
    _temp.Add_Char((char)MyCharLower_Ascii((char)c));
  }

  while (*p != 0)
  {
    const char *s2 = _temp.Ptr();
    char c, c2;
    do
    {
      c = *p++;
      c2 = *s2++;
    }
    while (c == c2);
    
    if (c == ' ')
    {
      if (c2 == 0)
        return true;
      continue;
    }
    
    while (*p++ != ' ');
  }
  
  return false;
}


void SplitString(const UString &srcString, UStringVector &destStrings)
{
  destStrings.Clear();
  unsigned len = srcString.Len();
  if (len == 0)
    return;
  UString s;
  for (unsigned i = 0; i < len; i++)
  {
    const wchar_t c = srcString[i];
    if (c == ' ')
    {
      if (!s.IsEmpty())
      {
        destStrings.Add(s);
        s.Empty();
      }
    }
    else
      s += c;
  }
  if (!s.IsEmpty())
    destStrings.Add(s);
}
