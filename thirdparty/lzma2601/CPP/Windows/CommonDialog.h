// Windows/CommonDialog.h

#ifndef ZIP7_INC_WINDOWS_COMMON_DIALOG_H
#define ZIP7_INC_WINDOWS_COMMON_DIALOG_H

#include "../Common/MyString.h"

namespace NWindows {

struct CCommonDialogInfo
{
  /* (FilterIndex == -1) means no selected filter.
       and (-1) also is reserved for unsupported custom filter.
     if (FilterIndex >= 0), then FilterIndex is index of filter */
  int FilterIndex;    // [in / out]
  bool SaveMode;
 #ifdef UNDER_CE
  bool OpenFolderMode;
 #endif
  HWND hwndOwner;
  // LPCWSTR lpstrInitialDir;
  LPCWSTR lpstrTitle;
  UString FilePath;   // [in / out]

  CCommonDialogInfo()
  {
    FilterIndex = -1;
    SaveMode = false;
   #ifdef UNDER_CE
    OpenFolderMode = false;
   #endif
    hwndOwner = NULL;
    // lpstrInitialDir = NULL;
    lpstrTitle = NULL;
  }
  
  /* (filters) : 2 sequential vector strings (Description, Masks) represent each filter */
  bool CommonDlg_BrowseForFile(LPCWSTR lpstrInitialDir, const UStringVector &filters);
};

}

#endif
