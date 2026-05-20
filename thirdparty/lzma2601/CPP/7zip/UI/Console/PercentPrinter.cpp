// PercentPrinter.cpp

#include "StdAfx.h"

#include "../../../Common/IntToString.h"

#include "PercentPrinter.h"

static const unsigned kPercentsSize = 4;

CPercentPrinter::~CPercentPrinter()
{
  ClosePrint(false);
}

void CPercentPrinterState::ClearCurState()
{
  Completed = 0;
  Total = ((UInt64)(Int64)-1);
  Files = 0;
  Command.Empty();
  FileName.Empty();
}

void CPercentPrinter::ClosePrint(bool needFlush)
{
  unsigned num = _printedString.Len();
  if (num != 0)
  {

  unsigned i;
    
  /* '\r' in old MAC OS means "new line".
     So we can't use '\r' in some systems */
    
  #ifdef _WIN32
    char *start = _temp.GetBuf(num  + 2);
    char *p = start;
    *p++ = '\r';
    for (i = 0; i < num; i++) *p++ = ' ';
    *p++ = '\r';
  #else
    char *start = _temp.GetBuf(num * 3);
    char *p = start;
    for (i = 0; i < num; i++) *p++ = '\b';
    for (i = 0; i < num; i++) *p++ = ' ';
    for (i = 0; i < num; i++) *p++ = '\b';
  #endif
    
  *p = 0;
  _temp.ReleaseBuf_SetLen((unsigned)(p - start));
  *_so << _temp;
  }
  if (needFlush)
    _so->Flush();
  _printedString.Empty();
}

void CPercentPrinter::GetPercents()
{
  char s[32];
  unsigned size;
  {
    char c = '%';
    UInt64 val = 0;
    if (Total == (UInt64)(Int64)-1 ||
        (Total == 0 && Completed != 0))
    {
      val = Completed >> 20;
      c = 'M';
    }
    else if (Total != 0)
      val = Completed * 100 / Total;
    ConvertUInt64ToString(val, s);
    size = (unsigned)strlen(s);
    s[size++] = c;
    s[size] = 0;
  }

  while (size < kPercentsSize)
  {
    _s.Add_Space();
    size++;
  }

  _s += s;
}

void CPercentPrinter::Print()
{
  if (DisablePrint)
    return;
  DWORD tick = 0;
  if (_tickStep != 0)
    tick = GetTickCount();

  bool onlyPercentsChanged = false;

  if (!_printedString.IsEmpty())
  {
    if (_tickStep != 0 && (UInt32)(tick - _prevTick) < _tickStep)
      return;
    
    CPercentPrinterState &st = *this;
    if (_printedState.Command == st.Command
        && _printedState.FileName == st.FileName
        && _printedState.Files == st.Files)
    {
      if (_printedState.Total == st.Total
          && _printedState.Completed == st.Completed)
        return;
      onlyPercentsChanged = true;
    }
  }

  _s.Empty();

  GetPercents();
  
  if (onlyPercentsChanged && _s == _printedPercents)
    return;

  _printedPercents = _s;

  if (Files != 0)
  {
    char s[32];
    ConvertUInt64ToString(Files, s);
    // unsigned size = (unsigned)strlen(s);
    // for (; size < 3; size++) _s.Add_Space();
    _s.Add_Space();
    _s += s;
    // _s += "f";
  }


  if (!Command.IsEmpty())
  {
    _s.Add_Space();
    _s += Command;
  }

  if (!FileName.IsEmpty() && _s.Len() < MaxLen)
  {
    _s.Add_Space();

    _tempU = FileName;
    _so->Normalize_UString_Path(_tempU);
    _so->Convert_UString_to_AString(_tempU, _temp);
    if (_s.Len() + _temp.Len() > MaxLen)
    {
      unsigned len = FileName.Len();
      for (; len != 0;)
      {
        unsigned delta = len / 8;
        if (delta == 0)
          delta = 1;
        len -= delta;
        _tempU = FileName;
        _tempU.Delete(len / 2, _tempU.Len() - len);
        _tempU.Insert(len / 2, L" . ");
        _so->Normalize_UString_Path(_tempU);
        _so->Convert_UString_to_AString(_tempU, _temp);
        if (_s.Len() + _temp.Len() <= MaxLen)
          break;
      }
      if (len == 0)
        _temp.Empty();
    }
    _s += _temp;
  }
  
  if (_printedString != _s)
  {
    ClosePrint(false);
    *_so << _s;
    if (NeedFlush)
      _so->Flush();
    _printedString = _s;
  }

  _printedState = *this;

  if (_tickStep != 0)
    _prevTick = tick;
}
