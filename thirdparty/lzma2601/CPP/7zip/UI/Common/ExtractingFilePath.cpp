// ExtractingFilePath.cpp

#include "StdAfx.h"

#include "../../../Common/Wildcard.h"

#include "../../../Windows/FileName.h"

#include "ExtractingFilePath.h"

extern
bool g_PathTrailReplaceMode;
bool g_PathTrailReplaceMode =
    #ifdef _WIN32
      true
    #else
      false
    #endif
    ;


// #ifdef _WIN32
static void ReplaceIncorrectChars(UString &s)
{
  {
    for (unsigned i = 0; i < s.Len(); i++)
    {
      wchar_t c = s[i];
      if (
          #ifdef _WIN32
          c == ':' || c == '*' || c == '?' || c < 0x20 || c == '<' || c == '>' || c == '|' || c == '"'
          || c == '/'
          // || c == 0x202E // RLO
          ||
          #endif
          c == WCHAR_PATH_SEPARATOR)
      {
       #if WCHAR_PATH_SEPARATOR != L'/'
        // 22.00 : WSL replacement for backslash
        if (c == WCHAR_PATH_SEPARATOR)
          c = WCHAR_IN_FILE_NAME_BACKSLASH_REPLACEMENT;
        else
       #endif
          c = '_';
        s.ReplaceOneCharAtPos(i,
          c
          // (wchar_t)(0xf000 + c) // 21.02 debug: WSL encoding for unsupported characters
          );
      }
    }
  }
  
  if (g_PathTrailReplaceMode)
  {
    /*
    // if (g_PathTrailReplaceMode == 1)
    {
      if (!s.IsEmpty())
      {
        wchar_t c = s.Back();
        if (c == '.' || c == ' ')
        {
          // s += (wchar_t)(0x9c); // STRING TERMINATOR
          s += (wchar_t)'_';
        }
      }
    }
    else
    */
    {
      unsigned i;
      for (i = s.Len(); i != 0;)
      {
        wchar_t c = s[i - 1];
        if (c != '.' && c != ' ')
          break;
        i--;
        s.ReplaceOneCharAtPos(i, '_');
        // s.ReplaceOneCharAtPos(i, (c == ' ' ? (wchar_t)(0x2423) : (wchar_t)0x00B7));
      }
      /*
      if (g_PathTrailReplaceMode > 1 && i != s.Len())
      {
        s.DeleteFrom(i);
      }
      */
    }
  }
}
// #endif

/* WinXP-64 doesn't support ':', '\\' and '/' symbols in name of alt stream.
   But colon in postfix ":$DATA" is allowed.
   WIN32 functions don't allow empty alt stream name "name:" */

void Correct_AltStream_Name(UString &s)
{
  unsigned len = s.Len();
  const unsigned kPostfixSize = 6;
  if (s.Len() >= kPostfixSize
      && StringsAreEqualNoCase_Ascii(s.RightPtr(kPostfixSize), ":$DATA"))
    len -= kPostfixSize;
  for (unsigned i = 0; i < len; i++)
  {
    wchar_t c = s[i];
    if (c == ':' || c == '\\' || c == '/'
        || c == 0x202E // RLO
        )
      s.ReplaceOneCharAtPos(i, '_');
  }
  if (s.IsEmpty())
    s = '_';
}

#ifdef _WIN32

static const unsigned g_ReservedWithNum_Index = 4;

static const char * const g_ReservedNames[] =
{
  "CON", "PRN", "AUX", "NUL",
  "COM", "LPT"
};

static bool IsSupportedName(const UString &name)
{
  for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_ReservedNames); i++)
  {
    const char *reservedName = g_ReservedNames[i];
    unsigned len = MyStringLen(reservedName);
    if (name.Len() < len)
      continue;
    if (!name.IsPrefixedBy_Ascii_NoCase(reservedName))
      continue;
    if (i >= g_ReservedWithNum_Index)
    {
      wchar_t c = name[len];
      if (c < L'0' || c > L'9')
        continue;
      len++;
    }
    for (;;)
    {
      wchar_t c = name[len++];
      if (c == 0 || c == '.')
        return false;
      if (c != ' ')
        break;
    }
  }
  return true;
}

static void CorrectUnsupportedName(UString &name)
{
  if (!IsSupportedName(name))
    name.InsertAtFront(L'_');
}

#endif

static void Correct_PathPart(UString &s)
{
  // "." and ".."
  if (s.IsEmpty())
    return;

  if (s[0] == '.' && (s[1] == 0 || (s[1] == '.' && s[2] == 0)))
    s.Empty();
  // #ifdef _WIN32
  else
    ReplaceIncorrectChars(s);
  // #endif
}

// static const char * const k_EmptyReplaceName = "[]";
static const char k_EmptyReplaceName = '_';

UString Get_Correct_FsFile_Name(const UString &name)
{
  UString res = name;
  Correct_PathPart(res);
  
  #ifdef _WIN32
  CorrectUnsupportedName(res);
  #endif
  
  if (res.IsEmpty())
    res = k_EmptyReplaceName;
  return res;
}


void Correct_FsPath(bool absIsAllowed, bool keepAndReplaceEmptyPrefixes, UStringVector &parts, bool isDir)
{
  unsigned i = 0;

  if (absIsAllowed)
  {
    #if defined(_WIN32) && !defined(UNDER_CE)
    bool isDrive = false;
    #endif
    
    if (parts[0].IsEmpty())
    {
      i = 1;
      #if defined(_WIN32) && !defined(UNDER_CE)
      if (parts.Size() > 1 && parts[1].IsEmpty())
      {
        i = 2;
        if (parts.Size() > 2 && parts[2].IsEqualTo("?"))
        {
          i = 3;
          if (parts.Size() > 3 && NWindows::NFile::NName::IsDrivePath2(parts[3]))
          {
            isDrive = true;
            i = 4;
          }
        }
      }
      #endif
    }
    #if defined(_WIN32) && !defined(UNDER_CE)
    else if (NWindows::NFile::NName::IsDrivePath2(parts[0]))
    {
      isDrive = true;
      i = 1;
    }

    if (isDrive)
    {
      // we convert "c:name" to "c:\name", if absIsAllowed path.
      UString &ds = parts[i - 1];
      if (ds.Len() > 2)
      {
        parts.Insert(i, ds.Ptr(2));
        ds.DeleteFrom(2);
      }
    }
    #endif
  }

  if (i != 0)
    keepAndReplaceEmptyPrefixes = false;

  for (; i < parts.Size();)
  {
    UString &s = parts[i];

    Correct_PathPart(s);

    if (s.IsEmpty())
    {
      if (!keepAndReplaceEmptyPrefixes)
        if (isDir || i != parts.Size() - 1)
        {
          parts.Delete(i);
          continue;
        }
      s = k_EmptyReplaceName;
    }
    else
    {
      keepAndReplaceEmptyPrefixes = false;
      #ifdef _WIN32
      CorrectUnsupportedName(s);
      #endif
    }
    
    i++;
  }

  if (!isDir)
  {
    if (parts.IsEmpty())
      parts.Add((UString)k_EmptyReplaceName);
    else
    {
      UString &s = parts.Back();
      if (s.IsEmpty())
        s = k_EmptyReplaceName;
    }
  }
}

UString MakePathFromParts(const UStringVector &parts)
{
  UString s;
  FOR_VECTOR (i, parts)
  {
    if (i != 0)
      s.Add_PathSepar();
    s += parts[i];
  }
  return s;
}
