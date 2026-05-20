// Common/StdInStream.cpp

#include "StdAfx.h"

#ifdef _WIN32
#include <tchar.h>
#endif

#include "StdInStream.h"
#include "StringConvert.h"
#include "UTFConvert.h"

// #define kEOFMessage "Unexpected end of input stream"
// #define kReadErrorMessage "Error reading input stream"
// #define kIllegalCharMessage "Illegal zero character in input stream"


CStdInStream g_StdIn(stdin);

/*
#define kFileOpenMode TEXT("r")

bool CStdInStream::Open(LPCTSTR fileName) throw()
{
  Close();
  _stream =
    #ifdef _WIN32
      _tfopen
    #else
      fopen
    #endif
      (fileName, kFileOpenMode);
  _streamIsOpen = (_stream != 0);
  return _streamIsOpen;
}

bool CStdInStream::Close() throw()
{
  if (!_streamIsOpen)
    return true;
  _streamIsOpen = (fclose(_stream) != 0);
  return !_streamIsOpen;
}
*/

bool CStdInStream::ScanAStringUntilNewLine(AString &s)
{
  s.Empty();
  for (;;)
  {
    const int intChar = GetChar();
    if (intChar == EOF)
      return true;
    const char c = (char)intChar;
    if (c == 0)
      return false;
    if (c == '\n')
      return true;
    s.Add_Char(c);
  }
}

bool CStdInStream::ScanUStringUntilNewLine(UString &dest)
{
  dest.Empty();
  AString s;
  const bool res = ScanAStringUntilNewLine(s);
  int codePage = CodePage;
  if (codePage == -1)
    codePage = CP_OEMCP;
  if ((unsigned)codePage == CP_UTF8)
    ConvertUTF8ToUnicode(s, dest);
  else
    MultiByteToUnicodeString2(dest, s, (UINT)(unsigned)codePage);
  return res;
}

/*
bool CStdInStream::ReadToString(AString &resultString)
{
  resultString.Empty();
  for (;;)
  {
    int intChar = GetChar();
    if (intChar == EOF)
      return !Error();
    char c = (char)intChar;
    if (c == 0)
      return false;
    resultString += c;
  }
}
*/

int CStdInStream::GetChar()
{
  return fgetc(_stream); // getc() doesn't work in BeOS?
}
