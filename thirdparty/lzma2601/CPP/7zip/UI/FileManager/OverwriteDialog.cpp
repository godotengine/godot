// OverwriteDialog.cpp

#include "StdAfx.h"

#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"

#include "../../../Windows/FileFind.h"
#include "../../../Windows/PropVariantConv.h"
#include "../../../Windows/ResourceString.h"

#include "../../../Windows/Control/Static.h"

#include "FormatUtils.h"
#include "LangUtils.h"
#include "OverwriteDialog.h"

#include "PropertyNameRes.h"

using namespace NWindows;

#ifdef Z7_LANG
static const UInt32 kLangIDs[] =
{
  IDT_OVERWRITE_HEADER,
  IDT_OVERWRITE_QUESTION_BEGIN,
  IDT_OVERWRITE_QUESTION_END,
  IDB_YES_TO_ALL,
  IDB_NO_TO_ALL,
  IDB_AUTO_RENAME
};
#endif

static const unsigned kCurrentFileNameSizeLimit = 72;

void COverwriteDialog::ReduceString(UString &s)
{
  const unsigned size =
#ifdef UNDER_CE
      !_isBig ? 30 : // kCurrentFileNameSizeLimit2
#endif
      kCurrentFileNameSizeLimit;

  if (s.Len() > size)
  {
    s.Delete(size / 2, s.Len() - size);
    s.Insert(size / 2, L" ... ");
  }
  if (!s.IsEmpty() && s.Back() == ' ')
  {
    // s += (wchar_t)(0x2423); // visible space
    s.InsertAtFront(L'\"');
    s.Add_Char('\"');
  }
}


void COverwriteDialog::SetItemIcon(unsigned iconID, HICON hIcon)
{
  NControl::CStatic staticContol;
  staticContol.Attach(GetItem(iconID));
  hIcon = staticContol.SetIcon(hIcon);
  if (hIcon)
    DestroyIcon(hIcon);
}

void AddSizeValue(UString &s, UInt64 value);
void AddSizeValue(UString &s, UInt64 value)
{
  {
    wchar_t sz[32];
    ConvertUInt64ToString(value, sz);
    s += MyFormatNew(IDS_FILE_SIZE, sz);
  }
  if (value >= (1 << 10))
  {
    char c;
          if (value >= ((UInt64)10 << 30)) { value >>= 30; c = 'G'; }
    else  if (value >=         (10 << 20)) { value >>= 20; c = 'M'; }
    else                                   { value >>= 10; c = 'K'; }
    s += " : ";
    s.Add_UInt64(value);
    s.Add_Space();
    s.Add_Char(c);
    s += "iB";
  }
}


void COverwriteDialog::SetFileInfoControl(
    const NOverwriteDialog::CFileInfo &fileInfo,
    unsigned textID,
    unsigned iconID,
    unsigned iconID_2)
{
  {
    const UString &path = fileInfo.Path;
    const int slashPos = path.ReverseFind_PathSepar();
    UString s = path.Left((unsigned)(slashPos + 1));
    ReduceString(s);
    s.Add_LF();
    {
      UString s2 = path.Ptr((unsigned)(slashPos + 1));
      ReduceString(s2);
      s += s2;
    }
    s.Add_LF();
    if (fileInfo.Size_IsDefined)
      AddSizeValue(s, fileInfo.Size);
    s.Add_LF();
    if (fileInfo.Time_IsDefined)
    {
      AddLangString(s, IDS_PROP_MTIME);
      s += ": ";
      char t[64];
      ConvertUtcFileTimeToString(fileInfo.Time, t);
      s += t;
    }
    SetItemText(textID, s);
  }
/*
  SHGetFileInfo():
    DOCs: If uFlags does not contain SHGFI_EXETYPE or SHGFI_SYSICONINDEX,
          the return value is nonzero if successful, or zero otherwise.
    We don't use SHGFI_EXETYPE or SHGFI_SYSICONINDEX here.
  win10: we call with SHGFI_ICON flag set.
    it returns 0: if error : (shFileInfo::*) members are not set.
    it returns non_0, if successful, and retrieve:
      { shFileInfo.hIcon != NULL : the handle to icon (must be destroyed by our code)
        shFileInfo.iIcon is index of the icon image within the system image list.
      }
  Note:
    If we send path to ".exe" file,
    SHGFI_USEFILEATTRIBUTES flag is ignored, and it tries to open file.
    and return icon from that exe file.
    So we still need to reduce path, if want to get raw icon of exe file.
    
  if (name.Len() >= MAX_PATH))
  {
    it can return:
      return 0.
      return 1 and:
        { shFileInfo.hIcon != NULL : is some default icon for file
          shFileInfo.iIcon == 0
        }
    return results (0 or 1) can depend from:
      - unicode/non-unicode
      - (SHGFI_USEFILEATTRIBUTES) flag
      - exact file extension (.exe).
  }
*/
  int iconIndex = -1;
  for (unsigned i = 0; i < 2; i++)
  {
    CSysString name = GetSystemString(fileInfo.Path);
    if (i != 0)
    {
      if (!fileInfo.Is_FileSystemFile)
        break;
      if (name.Len() < 4 ||
          (!StringsAreEqualNoCase_Ascii(name.RightPtr(4), ".exe") &&
           !StringsAreEqualNoCase_Ascii(name.RightPtr(4), ".ico")))
        break;
      // if path for ".exe" file is long, it returns default icon (shFileInfo.iIcon == 0).
      // We don't want to show that default icon.
      // But we will check for default icon later instead of MAX_PATH check here.
      // if (name.Len() >= MAX_PATH) break; // optional
    }
    else
    {
      // we need only file extension with dot
      const int separ = name.ReverseFind_PathSepar();
      name.DeleteFrontal((unsigned)(separ + 1));
      // if (name.Len() >= MAX_PATH)
      {
        const int dot = name.ReverseFind_Dot();
        if (dot >= 0)
          name.DeleteFrontal((unsigned)dot);
        // else name.Empty(); to set default name below
      }
      // name.Empty(); // for debug
    }

    if (name.IsEmpty())
    {
      // If we send empty name, SHGetFileInfo() returns some strange icon.
      // So we use common dummy name without extension,
      // and SHGetFileInfo() will return default icon (iIcon == 0)
      name = "__file__";
    }

    DWORD attrib = FILE_ATTRIBUTE_ARCHIVE;
    if (fileInfo.Is_FileSystemFile)
    {
      NFile::NFind::CFileInfo fi;
      if (fi.Find(us2fs(fileInfo.Path)) && !fi.IsAltStream && !fi.IsDir())
        attrib = fi.Attrib;
    }

    SHFILEINFO shFileInfo;
    // ZeroMemory(&shFileInfo, sizeof(shFileInfo)); // optional
    shFileInfo.hIcon = NULL; // optional
    shFileInfo.iIcon = -1;   // optional
    // memset(&shFileInfo, 1, sizeof(shFileInfo)); // for debug
    const DWORD_PTR res = ::SHGetFileInfo(name, attrib,
        &shFileInfo, sizeof(shFileInfo),
        SHGFI_ICON | SHGFI_LARGEICON | SHGFI_SHELLICONSIZE |
        // (i == 0 ? SHGFI_USEFILEATTRIBUTES : 0)
        SHGFI_USEFILEATTRIBUTES
        // we use SHGFI_USEFILEATTRIBUTES for second icon, because
        // it still returns real icon from exe files
        );
    if (res && shFileInfo.hIcon)
    {
      // we don't show second icon, if icon index (iIcon) is same
      // as first icon index of first shown icon (exe file without icon)
      if (   shFileInfo.iIcon >= 0
          && shFileInfo.iIcon != iconIndex
          && (shFileInfo.iIcon != 0 || i == 0)) // we don't want default icon for second icon
      {
        iconIndex = shFileInfo.iIcon;
        SetItemIcon(i == 0 ? iconID : iconID_2, shFileInfo.hIcon);
      }
      else
        DestroyIcon(shFileInfo.hIcon);
    }
  }
}



bool COverwriteDialog::OnInit()
{
  #ifdef Z7_LANG
  LangSetWindowText(*this, IDD_OVERWRITE);
  LangSetDlgItems(*this, kLangIDs, Z7_ARRAY_SIZE(kLangIDs));
  #endif
  SetFileInfoControl(OldFileInfo,
      IDT_OVERWRITE_OLD_FILE_SIZE_TIME,
      IDI_OVERWRITE_OLD_FILE,
      IDI_OVERWRITE_OLD_FILE_2);
  SetFileInfoControl(NewFileInfo,
      IDT_OVERWRITE_NEW_FILE_SIZE_TIME,
      IDI_OVERWRITE_NEW_FILE,
      IDI_OVERWRITE_NEW_FILE_2);
  NormalizePosition();

  if (!ShowExtraButtons)
  {
    HideItem(IDB_YES_TO_ALL);
    HideItem(IDB_NO_TO_ALL);
    HideItem(IDB_AUTO_RENAME);
  }

  if (DefaultButton_is_NO)
  {
    PostMsg(DM_SETDEFID, IDNO);
    HWND h = GetItem(IDNO);
    PostMsg(WM_NEXTDLGCTL, (WPARAM)h, TRUE);
    // ::SetFocus(h);
  }

  return CModalDialog::OnInit();
}

bool COverwriteDialog::OnDestroy()
{
  SetItemIcon(IDI_OVERWRITE_OLD_FILE, NULL);
  SetItemIcon(IDI_OVERWRITE_OLD_FILE_2, NULL);
  SetItemIcon(IDI_OVERWRITE_NEW_FILE, NULL);
  SetItemIcon(IDI_OVERWRITE_NEW_FILE_2, NULL);
  return false; // we return (false) to perform default dialog operation
}

bool COverwriteDialog::OnButtonClicked(unsigned buttonID, HWND buttonHWND)
{
  switch (buttonID)
  {
    case IDYES:
    case IDNO:
    case IDB_YES_TO_ALL:
    case IDB_NO_TO_ALL:
    case IDB_AUTO_RENAME:
      End((INT_PTR)buttonID);
      return true;
  }
  return CModalDialog::OnButtonClicked(buttonID, buttonHWND);
}
