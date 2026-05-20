// BrowseDialog.h

#ifndef ZIP7_INC_BROWSE_DIALOG_H
#define ZIP7_INC_BROWSE_DIALOG_H

#include "../../../Windows/CommonDialog.h"

bool MyBrowseForFolder(HWND owner, LPCWSTR title, LPCWSTR path, UString &resultPath);

struct CBrowseFilterInfo
{
  UStringVector Masks;
  UString Description;
};

struct CBrowseInfo: public NWindows::CCommonDialogInfo
{
  bool BrowseForFile(const CObjectVector<CBrowseFilterInfo> &filters);
};


/* CorrectFsPath removes undesirable characters in names (dots and spaces at the end of file)
   But it doesn't change "bad" name in any of the following cases:
     - path is Super Path (with \\?\ prefix)
     - path is relative and relBase is Super Path
     - there is file or dir in filesystem with specified "bad" name */

bool CorrectFsPath(const UString &relBase, const UString &path, UString &result);

bool Dlg_CreateFolder(HWND wnd, UString &destName);

#endif
