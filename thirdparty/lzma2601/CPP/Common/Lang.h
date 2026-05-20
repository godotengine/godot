// Common/Lang.h

#ifndef ZIP7_INC_COMMON_LANG_H
#define ZIP7_INC_COMMON_LANG_H

#include "MyString.h"

class CLang
{
  wchar_t *_text;

  bool OpenFromString(const AString &s);
public:
  CRecordVector<UInt32> _ids;
  CRecordVector<UInt32> _offsets;
  UStringVector Comments;

  CLang(): _text(NULL) {}
  ~CLang() { Clear(); }
  bool Open(CFSTR fileName, const char *id);
  void Clear() throw();
  bool IsEmpty() const { return _ids.IsEmpty(); }
  const wchar_t *Get(UInt32 id) const throw();
  const wchar_t *Get_by_index(unsigned index) const throw()
  {
    return _text + (size_t)_offsets[index];
  }
};

#endif
