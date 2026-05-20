// TempFiles.cpp

#include "StdAfx.h"

#include "../../../Windows/FileDir.h"

#include "TempFiles.h"

using namespace NWindows;
using namespace NFile;

void CTempFiles::Clear()
{
  while (!Paths.IsEmpty())
  {
    if (NeedDeleteFiles)
      NDir::DeleteFileAlways(Paths.Back());
    Paths.DeleteBack();
  }
}
