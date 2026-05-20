// TempFiles.h

#ifndef ZIP7_INC_TEMP_FILES_H
#define ZIP7_INC_TEMP_FILES_H

#include "../../../Common/MyString.h"

class CTempFiles
{
  void Clear();
public:
  FStringVector Paths;
  bool NeedDeleteFiles;

  CTempFiles(): NeedDeleteFiles(true) {}
  ~CTempFiles() { Clear(); }
};

#endif
