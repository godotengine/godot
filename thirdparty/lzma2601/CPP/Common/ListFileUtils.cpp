// Common/ListFileUtils.cpp

#include "StdAfx.h"

#include "../../C/CpuArch.h"

#include "ListFileUtils.h"
#include "MyBuffer.h"
#include "StringConvert.h"
#include "UTFConvert.h"

#include "../Windows/FileIO.h"

#define CSysInFile NWindows::NFile::NIO::CInFile
#define MY_GET_LAST_ERROR ::GetLastError()


#define kQuoteChar '\"'


static void AddName(UStringVector &strings, UString &s)
{
  s.Trim();
  if (s.Len() >= 2 && s[0] == kQuoteChar && s.Back() == kQuoteChar)
  {
    s.DeleteBack();
    s.Delete(0);
  }
  if (!s.IsEmpty())
    strings.Add(s);
}


static bool My_File_Read(CSysInFile &file, void *data, size_t size, DWORD &lastError)
{
  size_t processed;
  if (!file.ReadFull(data, size, processed))
  {
    lastError = MY_GET_LAST_ERROR;
    return false;
  }
  if (processed != size)
  {
    lastError = 1; // error: size of listfile was changed
    return false;
  }
  return true;
}


bool ReadNamesFromListFile2(CFSTR fileName, UStringVector &strings, UINT codePage, DWORD &lastError)
{
  lastError = 0;
  CSysInFile file;
  if (!file.Open(fileName))
  {
    lastError = MY_GET_LAST_ERROR;
    return false;
  }
  UInt64 fileSize;
  if (!file.GetLength(fileSize))
  {
    lastError = MY_GET_LAST_ERROR;
    return false;
  }
  if (fileSize >= ((UInt32)1 << 31) - 32)
    return false;
  UString u;
  if (codePage == Z7_WIN_CP_UTF16 || codePage == Z7_WIN_CP_UTF16BE)
  {
    if ((fileSize & 1) != 0)
      return false;
    CByteArr buf((size_t)fileSize);

    if (!My_File_Read(file, buf, (size_t)fileSize, lastError))
      return false;

    file.Close();
    const size_t num = (size_t)fileSize / 2;
    wchar_t *p = u.GetBuf((unsigned)num);
    if (codePage == Z7_WIN_CP_UTF16)
      for (size_t i = 0; i < num; i++)
      {
        const wchar_t c = GetUi16(buf.ConstData() + (size_t)i * 2);
        if (c == 0)
          return false;
        p[i] = c;
      }
    else
      for (size_t i = 0; i < num; i++)
      {
        const wchar_t c = (wchar_t)GetBe16(buf.ConstData() + (size_t)i * 2);
        if (c == 0)
          return false;
        p[i] = c;
      }
    p[num] = 0;
    u.ReleaseBuf_SetLen((unsigned)num);
  }
  else
  {
    AString s;
    char *p = s.GetBuf((unsigned)fileSize);

    if (!My_File_Read(file, p, (size_t)fileSize, lastError))
      return false;

    file.Close();
    s.ReleaseBuf_CalcLen((unsigned)fileSize);
    if (s.Len() != fileSize)
      return false;
    
    // #ifdef CP_UTF8
    if (codePage == CP_UTF8)
    {
      // we must check UTF8 here, if convert function doesn't check
      if (!CheckUTF8_AString(s))
        return false;
      if (!ConvertUTF8ToUnicode(s, u))
        return false;
    }
    else
    // #endif
      MultiByteToUnicodeString2(u, s, codePage);
  }

  const wchar_t kGoodBOM = 0xFEFF;
  // const wchar_t kBadBOM  = 0xFFFE;
  
  UString s;
  unsigned i = 0;
  for (; i < u.Len() && u[i] == kGoodBOM; i++);
  for (; i < u.Len(); i++)
  {
    wchar_t c = u[i];
    /*
    if (c == kGoodBOM || c == kBadBOM)
      return false;
    */
    if (c == '\n' || c == 0xD)
    {
      AddName(strings, s);
      s.Empty();
    }
    else
      s += c;
  }
  AddName(strings, s);
  return true;
}
