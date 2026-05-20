// HashCon.h

#ifndef ZIP7_INC_HASH_CON_H
#define ZIP7_INC_HASH_CON_H

#include "../Common/HashCalc.h"

#include "UpdateCallbackConsole.h"

class CHashCallbackConsole Z7_final:
    public IHashCallbackUI,
    public CCallbackConsoleBase
{
  Z7_IFACE_IMP(IDirItemsCallback)
  Z7_IFACE_IMP(IHashCallbackUI)

  UString _fileName;
  AString _s;

  void AddSpace()
  {
    _s.Add_Space_if_NotEmpty();
  }

  void AddSpacesBeforeName()
  {
    if (!_s.IsEmpty())
    {
      _s.Add_Space();
      _s.Add_Space();
    }
  }

  void PrintSeparatorLine(const CObjectVector<CHasherState> &hashers);
  void PrintResultLine(UInt64 fileSize,
      const CObjectVector<CHasherState> &hashers, unsigned digestIndex, bool showHash, const AString &path);
  void PrintProperty(const char *name, UInt64 value);

public:
  bool PrintNameInPercents;
  bool PrintHeaders;
  // bool PrintSize;
  // bool PrintNewLine; // set it too (false), if you need only hash for single file without LF char.
  AString PrintFields;

  AString GetFields() const;

  CHashCallbackConsole():
      PrintNameInPercents(true),
      PrintHeaders(false)
      // , PrintSize(true),
      // , PrintNewLine(true)
    {}
};

void PrintHashStat(CStdOutStream &so, const CHashBundle &hb);

#endif
