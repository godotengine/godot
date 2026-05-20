// CommandLineParser.cpp

#include "StdAfx.h"

#include "CommandLineParser.h"

namespace NCommandLineParser {

#ifdef _WIN32

bool SplitCommandLine(const UString &src, UString &dest1, UString &dest2)
{
  dest1.Empty();
  dest2.Empty();
  bool quoteMode = false;
  unsigned i;
  for (i = 0; i < src.Len(); i++)
  {
    const wchar_t c = src[i];
    if ((c == L' ' || c == L'\t') && !quoteMode)
    {
      dest2 = src.Ptr(i + 1);
      return i != 0;
    }
    if (c == L'\"')
      quoteMode = !quoteMode;
    else
      dest1 += c;
  }
  return i != 0;
}

void SplitCommandLine(const UString &s, UStringVector &parts)
{
#if 0
/* we don't use CommandLineToArgvW() because
   it can remove tail backslash:
     "1\"
   converted to
     1"
*/
  parts.Clear();
  {
    int nArgs;
    LPWSTR *szArgList = CommandLineToArgvW(s, &nArgs);
    if (szArgList)
    {
      for (int i = 0; i < nArgs; i++)
      {
        // printf("%2d: |%S|\n", i, szArglist[i]);
        parts.Add(szArgList[i]);
      }
      LocalFree(szArgList);
      return;
    }
  }
#endif
/*
#ifdef _UNICODE
  throw 20240406;
#else
*/
  UString sTemp (s);
  sTemp.Trim();
  parts.Clear();
  for (;;)
  {
    UString s1, s2;
    if (SplitCommandLine(sTemp, s1, s2))
      parts.Add(s1);
    if (s2.IsEmpty())
      break;
    sTemp = s2;
  }
// #endif
}
#endif


static const char * const kStopSwitchParsing = "--";

static bool inline IsItSwitchChar(wchar_t c)
{
  return (c == '-');
}

CParser::CParser():
  _switches(NULL),
  StopSwitchIndex(-1)
{
}

CParser::~CParser()
{
  delete []_switches;
}


// if (s) contains switch then function updates switch structures
// out: true, if (s) is a switch
bool CParser::ParseString(const UString &s, const CSwitchForm *switchForms, unsigned numSwitches)
{
  if (s.IsEmpty() || !IsItSwitchChar(s[0]))
    return false;

  unsigned pos = 1;
  unsigned switchIndex = 0;
  int maxLen = -1;
  
  for (unsigned i = 0; i < numSwitches; i++)
  {
    const char * const key = switchForms[i].Key;
    const unsigned switchLen = MyStringLen(key);
    if ((int)switchLen <= maxLen || pos + switchLen > s.Len())
      continue;
    if (IsString1PrefixedByString2_NoCase_Ascii((const wchar_t *)s + pos, key))
    {
      switchIndex = i;
      maxLen = (int)switchLen;
    }
  }

  if (maxLen < 0)
  {
    ErrorMessage = "Unknown switch:";
    return false;
  }

  pos += (unsigned)maxLen;
  
  CSwitchResult &sw = _switches[switchIndex];
  const CSwitchForm &form = switchForms[switchIndex];
  
  if (!form.Multi && sw.ThereIs)
  {
    ErrorMessage = "Multiple instances for switch:";
    return false;
  }

  sw.ThereIs = true;

  const unsigned rem = s.Len() - pos;
  if (rem < form.MinLen)
  {
    ErrorMessage = "Too short switch:";
    return false;
  }
  
  sw.WithMinus = false;
  sw.PostCharIndex = -1;
  
  switch (form.Type)
  {
    case NSwitchType::kMinus:
      if (rem == 1)
      {
        sw.WithMinus = (s[pos] == '-');
        if (sw.WithMinus)
          return true;
        ErrorMessage = "Incorrect switch postfix:";
        return false;
      }
      break;
      
    case NSwitchType::kChar:
      if (rem == 1)
      {
        const wchar_t c = s[pos];
        if (c <= 0x7F)
        {
          sw.PostCharIndex = FindCharPosInString(form.PostCharSet, (char)c);
          if (sw.PostCharIndex >= 0)
            return true;
        }
        ErrorMessage = "Incorrect switch postfix:";
        return false;
      }
      break;
      
    case NSwitchType::kString:
    {
      sw.PostStrings.Add(s.Ptr(pos));
      return true;
    }
    // case NSwitchType::kSimple:
    default: break;
  }

  if (pos != s.Len())
  {
    ErrorMessage = "Too long switch:";
    return false;
  }
  return true;
}


bool CParser::ParseStrings(const CSwitchForm *switchForms, unsigned numSwitches, const UStringVector &commandStrings)
{
  StopSwitchIndex = -1;
  ErrorMessage.Empty();
  ErrorLine.Empty();
  NonSwitchStrings.Clear();
  delete []_switches;
  _switches = NULL;
  _switches = new CSwitchResult[numSwitches];
  
  FOR_VECTOR (i, commandStrings)
  {
    const UString &s = commandStrings[i];
    if (StopSwitchIndex < 0)
    {
      if (s.IsEqualTo(kStopSwitchParsing))
      {
        StopSwitchIndex = (int)NonSwitchStrings.Size();
        continue;
      }
      if (!s.IsEmpty() && IsItSwitchChar(s[0]))
      {
        if (ParseString(s, switchForms, numSwitches))
          continue;
        ErrorLine = s;
        return false;
      }
    }
    NonSwitchStrings.Add(s);
  }
  return true;
}

}
