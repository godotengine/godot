// BrowseDialog.cpp
 
#include "StdAfx.h"

#include "../../../Common/MyWindows.h"

#include "../../../Common/IntToString.h"

#ifndef UNDER_CE
#include "../../../Windows/CommonDialog.h"
#include "../../../Windows/Shell.h"
#endif

#include "../../../Windows/FileName.h"
#include "../../../Windows/FileFind.h"

#ifdef UNDER_CE
#include <commdlg.h>
#endif

#include "BrowseDialog.h"

#define USE_MY_BROWSE_DIALOG

#ifdef USE_MY_BROWSE_DIALOG

#include "../../../Common/Defs.h"
#include "../../../Common/Wildcard.h"

#include "../../../Windows/FileDir.h"
#include "../../../Windows/PropVariantConv.h"

#include "../../../Windows/Control/ComboBox.h"
#include "../../../Windows/Control/Dialog.h"
#include "../../../Windows/Control/Edit.h"
#include "../../../Windows/Control/ListView.h"

#include "BrowseDialogRes.h"
#include "PropertyNameRes.h"
#include "SysIconUtils.h"

#ifndef Z7_SFX
#include "RegistryUtils.h"
#endif

#endif // USE_MY_BROWSE_DIALOG

#include "ComboDialog.h"
#include "LangUtils.h"

#include "resource.h"

using namespace NWindows;
using namespace NFile;
using namespace NName;
using namespace NFind;

static void MessageBox_Error_Global(HWND wnd, const wchar_t *message)
{
  ::MessageBoxW(wnd, message, L"7-Zip", MB_ICONERROR);
}

#ifdef USE_MY_BROWSE_DIALOG

#if 0
extern HINSTANCE g_hInstance;
#endif
extern bool g_LVN_ITEMACTIVATE_Support;

static const int kParentIndex = -1;
static const UINT k_Message_RefreshPathEdit = WM_APP + 1;

extern UString HResultToMessage(HRESULT errorCode);

static void MessageBox_HResError(HWND wnd, HRESULT errorCode, const wchar_t *name)
{
  UString s = HResultToMessage(errorCode);
  if (name)
  {
    s.Add_LF();
    s += name;
  }
  MessageBox_Error_Global(wnd, s);
}

class CBrowseDialog: public NControl::CModalDialog
{
  NControl::CListView _list;
  NControl::CEdit _pathEdit;
  NControl::CComboBox _filterCombo;

  CObjectVector<CFileInfo> _files;

  CExtToIconMap _extToIconMap;
  int _sortIndex;
  bool _ascending;
 #ifndef Z7_SFX
  bool _showDots;
 #endif
  UString _topDirPrefix; // we don't open parent of that folder
  UString DirPrefix;

  virtual bool OnInit() Z7_override;
  virtual bool OnSize(WPARAM wParam, int xSize, int ySize) Z7_override;
  virtual bool OnMessage(UINT message, WPARAM wParam, LPARAM lParam) Z7_override;
  virtual bool OnNotify(UINT controlID, LPNMHDR header) Z7_override;
  virtual bool OnCommand(unsigned code, unsigned itemID, LPARAM lParam) Z7_override;
  virtual bool OnButtonClicked(unsigned buttonID, HWND buttonHWND) Z7_override;
  virtual void OnOK() Z7_override;

  bool OnKeyDown(LPNMLVKEYDOWN keyDownInfo);

  void Post_RefreshPathEdit() { PostMsg(k_Message_RefreshPathEdit); }

  bool GetParentPath(const UString &path, UString &parentPrefix, UString &name);
  // Reload changes DirPrefix. Don't send DirPrefix in pathPrefix parameter
  HRESULT Reload(const UString &pathPrefix, const UString &selectedName);
  HRESULT Reload();
  
  void OpenParentFolder();
  void SetPathEditText();
  void OnCreateDir();
  void OnItemEnter();
  void FinishOnOK();

  int GetRealItemIndex(int indexInListView) const
  {
    LPARAM param;
    if (!_list.GetItemParam((unsigned)indexInListView, param))
      return (int)-1;
    return (int)param;
  }

public:

  bool SaveMode;
  bool FolderMode;
  int FilterIndex;  // [in / out]
  CObjectVector<CBrowseFilterInfo> Filters;

  UString FilePath;   // [in / out]
  UString Title;

  CBrowseDialog():
   #ifndef Z7_SFX
      _showDots(false),
   #endif
      SaveMode(false)
      , FolderMode(false)
      , FilterIndex(-1)
    {}
  INT_PTR Create(HWND parent = NULL) { return CModalDialog::Create(IDD_BROWSE, parent); }
  int CompareItems(LPARAM lParam1, LPARAM lParam2) const;
};


bool CBrowseDialog::OnInit()
{
  #ifdef Z7_LANG
  LangSetDlgItems(*this, NULL, 0);
  #endif
  if (!Title.IsEmpty())
    SetText(Title);
  _list.Attach(GetItem(IDL_BROWSE));
  _filterCombo.Attach(GetItem(IDC_BROWSE_FILTER));
  _pathEdit.Attach(GetItem(IDE_BROWSE_PATH));

  #ifndef UNDER_CE
  _list.SetUnicodeFormat();
  #endif

  #ifndef Z7_SFX
  CFmSettings st;
  st.Load();
  if (st.SingleClick)
    _list.SetExtendedListViewStyle(LVS_EX_ONECLICKACTIVATE | LVS_EX_TRACKSELECT);
  _showDots = st.ShowDots;
  #endif

  {
    /*
    Filters.Clear(); // for debug
    if (Filters.IsEmpty() && !FolderMode)
    {
      CBrowseFilterInfo &f = Filters.AddNew();
      const UString mask("*.*");
      f.Masks.Add(mask);
      // f.Description = "(";
      f.Description += mask;
      // f.Description += ")";
    }
    */

    FOR_VECTOR (i, Filters)
    {
      _filterCombo.AddString(Filters[i].Description);
    }

    if (Filters.Size() <= 1)
    {
      if (FolderMode)
        HideItem(IDC_BROWSE_FILTER);
      else
        EnableItem(IDC_BROWSE_FILTER, false);
    }
    
    if (/* FilterIndex >= 0 && */ (unsigned)FilterIndex < Filters.Size())
      _filterCombo.SetCurSel(FilterIndex);
  }

  _list.SetImageList(Shell_Get_SysImageList_smallIcons(true), LVSIL_SMALL);
  _list.SetImageList(Shell_Get_SysImageList_smallIcons(false), LVSIL_NORMAL);

  _list.InsertColumn(0, LangString(IDS_PROP_NAME), 100);
  _list.InsertColumn(1, LangString(IDS_PROP_MTIME), 100);
  {
    LV_COLUMNW column;
    column.iSubItem = 2;
    column.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM;
    column.fmt = LVCFMT_RIGHT;
    column.cx = 100;
    const UString s = LangString(IDS_PROP_SIZE);
    column.pszText = s.Ptr_non_const();
    _list.InsertColumn(2, &column);
  }

  _list.InsertItem(0, L"12345678901234567"
      #ifndef UNDER_CE
      L"1234567890"
      #endif
      );
  _list.SetSubItem(0, 1, L"2009-09-09"
      #ifndef UNDER_CE
      L" 09:09"
      #endif
      );
  _list.SetSubItem(0, 2, L"9999 MB");
  for (int i = 0; i < 3; i++)
    _list.SetColumnWidthAuto(i);
  _list.DeleteAllItems();

  _ascending = true;
  _sortIndex = 0;

  NormalizeSize();

  _topDirPrefix.Empty();
  {
    unsigned rootSize = GetRootPrefixSize(FilePath);
    #if defined(_WIN32) && !defined(UNDER_CE)
    // We can go up from root folder to drives list
    if (IsDrivePath(FilePath))
      rootSize = 0;
    else if (IsSuperPath(FilePath))
    {
      if (IsDrivePath(FilePath.Ptr(kSuperPathPrefixSize)))
        rootSize = kSuperPathPrefixSize;
    }
    #endif
    _topDirPrefix.SetFrom(FilePath, rootSize);
  }

  UString name;
  if (!GetParentPath(FilePath, DirPrefix, name))
    DirPrefix = _topDirPrefix;

  for (;;)
  {
    UString baseFolder = DirPrefix;
    if (Reload(baseFolder, name) == S_OK)
      break;
    name.Empty();
    if (DirPrefix.IsEmpty())
      break;
    UString parent, name2;
    GetParentPath(DirPrefix, parent, name2);
    DirPrefix = parent;
  }

  if (name.IsEmpty())
    name = FilePath;
  if (FolderMode)
    NormalizeDirPathPrefix(name);
  _pathEdit.SetText(name);

  #ifndef UNDER_CE
  /* If we clear UISF_HIDEFOCUS, the focus rectangle in ListView will be visible,
     even if we use mouse for pressing the button to open this dialog. */
  PostMsg(Z7_WIN_WM_UPDATEUISTATE, MAKEWPARAM(Z7_WIN_UIS_CLEAR, Z7_WIN_UISF_HIDEFOCUS));
  #endif

#if 0
  {
    const HWND hwndTool = GetItem(IDB_BROWSE_CREATE_DIR);
    if (hwndTool)
    {
      // Create the tooltip:
      const HWND hwndTip = CreateWindowEx(0, TOOLTIPS_CLASS, NULL,
          WS_POPUP | TTS_ALWAYSTIP
          // | TTS_BALLOON
          , CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
          *this, NULL, g_hInstance, NULL);
      if (hwndTip)
      {
        // Associate the tooltip with the tool:
        TOOLINFOW toolInfo;
        memset(&toolInfo, 0, sizeof(toolInfo));
        toolInfo.cbSize = sizeof(toolInfo);
        toolInfo.hwnd = *this;
        toolInfo.uFlags = TTF_IDISHWND | TTF_SUBCLASS;
        toolInfo.uId = (UINT_PTR)hwndTool;
        UString s;
#ifdef Z7_LANG
        LangString_OnlyFromLangFile(IDM_CREATE_FOLDER, s);
        s.RemoveChar(L'&');
        if (s.IsEmpty())
#endif
          s = "Create Folder";
        toolInfo.lpszText = s.Ptr_non_const();
        SendMessage(hwndTip, TTM_ADDTOOLW, 0, (LPARAM)&toolInfo);
      }
    }
  }
#endif
  return CModalDialog::OnInit();
}

bool CBrowseDialog::OnSize(WPARAM /* wParam */, int xSize, int ySize)
{
  int mx, my;
  {
    RECT r;
    GetClientRectOfItem(IDB_BROWSE_PARENT, r);
    mx = r.left;
    my = r.top;
  }
  InvalidateRect(NULL);

  int xLim = xSize - mx;
  {
    RECT r;
    GetClientRectOfItem(IDT_BROWSE_FOLDER, r);
    MoveItem(IDT_BROWSE_FOLDER, r.left, r.top, xLim - r.left, RECT_SIZE_Y(r));
  }

  int bx1, bx2, by;
  GetItemSizes(IDCANCEL, bx1, by);
  GetItemSizes(IDOK, bx2, by);
  int y = ySize - my - by;
  int x = xLim - bx1;
  MoveItem(IDCANCEL, x, y, bx1, by);
  MoveItem(IDOK, x - mx - bx2, y, bx2, by);

  // Y_Size of ComboBox is tricky. So we use Y_Size of _pathEdit instead

  int yPathSize;
  {
    RECT r;
    GetClientRectOfItem(IDE_BROWSE_PATH, r);
    yPathSize = RECT_SIZE_Y(r);
    _pathEdit.Move(r.left, y - my - yPathSize - my - yPathSize, xLim - r.left, yPathSize);
  }

  {
    RECT r;
    GetClientRectOfItem(IDC_BROWSE_FILTER, r);
    _filterCombo.Move(r.left, y - my - yPathSize, xLim - r.left, RECT_SIZE_Y(r));
  }

  {
    RECT r;
    GetClientRectOfItem(IDL_BROWSE, r);
    _list.Move(r.left, r.top, xLim - r.left, y - my - yPathSize - my - yPathSize - my - r.top);
  }

  return false;
}

bool CBrowseDialog::OnMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
  if (message == k_Message_RefreshPathEdit)
  {
    SetPathEditText();
    return true;
  }
  return CModalDialog::OnMessage(message, wParam, lParam);
}


bool CBrowseDialog::OnCommand(unsigned code, unsigned itemID, LPARAM lParam)
{
  if (code == CBN_SELCHANGE)
  {
    switch (itemID)
    {
      case IDC_BROWSE_FILTER:
      {
        Reload();
        return true;
      }
    }
  }
  return CModalDialog::OnCommand(code, itemID, lParam);
}


bool CBrowseDialog::OnNotify(UINT /* controlID */, LPNMHDR header)
{
  if (header->hwndFrom != _list)
    return false;
  switch (header->code)
  {
    case LVN_ITEMACTIVATE:
      if (g_LVN_ITEMACTIVATE_Support)
        OnItemEnter();
      break;
    case NM_DBLCLK:
    case NM_RETURN: // probably it's unused
      if (!g_LVN_ITEMACTIVATE_Support)
        OnItemEnter();
      break;
    case LVN_COLUMNCLICK:
    {
      const int index = LPNMLISTVIEW(header)->iSubItem;
      if (index == _sortIndex)
        _ascending = !_ascending;
      else
      {
        _ascending = (index == 0);
        _sortIndex = index;
      }
      Reload();
      return false;
    }
    case LVN_KEYDOWN:
    {
      bool boolResult = OnKeyDown(LPNMLVKEYDOWN(header));
      Post_RefreshPathEdit();
      return boolResult;
    }
    case NM_RCLICK:
    case NM_CLICK:
    case LVN_BEGINDRAG:
      Post_RefreshPathEdit();
      break;
  }
  return false;
}

bool CBrowseDialog::OnKeyDown(LPNMLVKEYDOWN keyDownInfo)
{
  const bool ctrl = IsKeyDown(VK_CONTROL);

  switch (keyDownInfo->wVKey)
  {
    case VK_BACK:
      OpenParentFolder();
      return true;
    case 'R':
      if (ctrl)
      {
        Reload();
        return true;
      }
      return false;
    case VK_F7:
      OnCreateDir();
      return true;
  }
  return false;
}


bool CBrowseDialog::OnButtonClicked(unsigned buttonID, HWND buttonHWND)
{
  switch (buttonID)
  {
    case IDB_BROWSE_PARENT: OpenParentFolder(); break;
    case IDB_BROWSE_CREATE_DIR: OnCreateDir(); break;
    default: return CModalDialog::OnButtonClicked(buttonID, buttonHWND);
  }
  _list.SetFocus();
  return true;
}

void CBrowseDialog::OnOK()
{
  /* When we press "Enter" in listview, Windows sends message to first Button.
     We check that message was from ListView; */
  if (GetFocus() == _list)
  {
    OnItemEnter();
    return;
  }
  FinishOnOK();
}


bool CBrowseDialog::GetParentPath(const UString &path, UString &parentPrefix, UString &name)
{
  parentPrefix.Empty();
  name.Empty();
  if (path.IsEmpty())
    return false;
  if (_topDirPrefix == path)
    return false;
  UString s = path;
  if (IS_PATH_SEPAR(s.Back()))
    s.DeleteBack();
  if (s.IsEmpty())
    return false;
  if (IS_PATH_SEPAR(s.Back()))
    return false;
  const unsigned pos1 = (unsigned)(s.ReverseFind_PathSepar() + 1);
  parentPrefix.SetFrom(s, pos1);
  name = s.Ptr(pos1);
  return true;
}

int CBrowseDialog::CompareItems(LPARAM lParam1, LPARAM lParam2) const
{
  if (lParam1 == lParam2)      return 0;
  if (lParam1 == kParentIndex) return -1;
  if (lParam2 == kParentIndex) return 1;

  const CFileInfo &f1 = _files[(int)lParam1];
  const CFileInfo &f2 = _files[(int)lParam2];

  const bool isDir2 = f2.IsDir();
  if (f1.IsDir())
  {
    if (!isDir2) return -1;
  }
  else if (isDir2) return 1;
  
  int res = 0;
  switch (_sortIndex)
  {
    case 0: res = CompareFileNames(fs2us(f1.Name), fs2us(f2.Name)); break;
    case 1: res = CompareFileTime(&f1.MTime, &f2.MTime); break;
    case 2: res = MyCompare(f1.Size, f2.Size); break;
  }
  return _ascending ? res: -res;
}

static int CALLBACK CompareItems2(LPARAM lParam1, LPARAM lParam2, LPARAM lpData)
{
  return ((CBrowseDialog *)lpData)->CompareItems(lParam1, lParam2);
}

wchar_t *Browse_ConvertSizeToString(UInt64 v, wchar_t *s);
wchar_t *Browse_ConvertSizeToString(UInt64 v, wchar_t *s)
{
  char c = 0;
       if (v >= ((UInt64)10000 << 20)) { v >>= 30; c = 'G'; }
  else if (v >= ((UInt64)10000 << 10)) { v >>= 20; c = 'M'; }
  else if (v >= ((UInt64)10000 <<  0)) { v >>= 10; c = 'K'; }
  s = ConvertUInt64ToString(v, s);
  if (c != 0)
  {
    *s++ = ' ';
    *s++ = (wchar_t)c;
    *s++ = 'B';
    *s = 0;
  }
  return s;
}

// Reload changes DirPrefix. Don't send DirPrefix in pathPrefix parameter

HRESULT CBrowseDialog::Reload(const UString &pathPrefix, const UString &selectedName)
{
  CObjectVector<CFileInfo> files;
  
  #ifndef UNDER_CE
  bool isDrive = false;
  if (pathPrefix.IsEmpty() || pathPrefix.IsEqualTo(kSuperPathPrefix))
  {
    isDrive = true;
    FStringVector drives;
    if (!MyGetLogicalDriveStrings(drives))
      return GetLastError_noZero_HRESULT();
    FOR_VECTOR (i, drives)
    {
      const FString &d = drives[i];
      if (d.Len() < 2 || d.Back() != '\\')
        return E_FAIL;
      CFileInfo &fi = files.AddNew();
      fi.SetAsDir();
      fi.Name = d;
      fi.Name.DeleteBack();
    }
  }
  else
  #endif
  {
    const UStringVector *masks = NULL;
    if (!Filters.IsEmpty() && _filterCombo.GetCount() > 0)
    {
      const int selected = _filterCombo.GetCurSel();
            // GetItemData_of_CurSel(); // we don't use data field
      if (/* selected >= 0 && */ (unsigned)selected < Filters.Size())
      {
        const UStringVector &m = Filters[selected].Masks;
        if (m.Size() > 1 || (m.Size() == 1
              && !m[0].IsEqualTo("*.*")
              && !m[0].IsEqualTo("*")))
          masks = &m;
      }
    }
    CEnumerator enumerator;
    enumerator.SetDirPrefix(us2fs(pathPrefix));
    CFileInfo fi;
    for (;;)
    {
      bool found;
      if (!enumerator.Next(fi, found))
        return GetLastError_noZero_HRESULT();
      if (!found)
        break;
      if (!fi.IsDir())
      {
        if (FolderMode)
          continue;
        if (masks)
        {
          unsigned i;
          const unsigned numMasks = masks->Size();
          for (i = 0; i < numMasks; i++)
            if (DoesWildcardMatchName((*masks)[i], fs2us(fi.Name)))
              break;
          if (i == numMasks)
            continue;
        }
      }
      files.Add(fi);
    }
  }

  DirPrefix = pathPrefix;

  _files = files;

  SetItemText(IDT_BROWSE_FOLDER, DirPrefix);

  _list.SetRedraw(false);
  _list.DeleteAllItems();

  LVITEMW item;

  unsigned index = 0;
  int cursorIndex = -1;

  #ifndef Z7_SFX
  if (_showDots && _topDirPrefix != DirPrefix)
  {
    item.iItem = (int)index;
    const UString itemName ("..");
    if (selectedName.IsEmpty())
      cursorIndex = (int)index;
    item.mask = LVIF_TEXT | LVIF_PARAM | LVIF_IMAGE;
    unsigned subItem = 0;
    item.iSubItem = (int)(subItem++);
    item.lParam = kParentIndex;
    item.pszText = itemName.Ptr_non_const();
    item.iImage = _extToIconMap.GetIconIndex(FILE_ATTRIBUTE_DIRECTORY, DirPrefix);
    if (item.iImage < 0)
      item.iImage = 0;
    _list.InsertItem(&item);
    _list.SetSubItem(index, subItem++, L"");
    _list.SetSubItem(index, subItem++, L"");
    index++;
  }
  #endif

  for (unsigned i = 0; i < _files.Size(); i++, index++)
  {
    item.iItem = (int)index;
    const CFileInfo &fi = _files[i];
    const UString name = fs2us(fi.Name);
    if (!selectedName.IsEmpty() && CompareFileNames(name, selectedName) == 0)
      cursorIndex = (int)index;
    item.mask = LVIF_TEXT | LVIF_PARAM | LVIF_IMAGE;
    unsigned subItem = 0;
    item.iSubItem = (int)(subItem++);
    item.lParam = (LPARAM)i;
    item.pszText = name.Ptr_non_const();

    const UString fullPath = DirPrefix + name;
    #ifndef UNDER_CE
    if (isDrive)
    {
      item.iImage = Shell_GetFileInfo_SysIconIndex_for_Path(
          fi.Name + FCHAR_PATH_SEPARATOR,
          FILE_ATTRIBUTE_DIRECTORY);
    }
    else
    #endif
      item.iImage = _extToIconMap.GetIconIndex(fi.Attrib, fullPath);
    if (item.iImage < 0)
        item.iImage = 0;
    _list.InsertItem(&item);
    wchar_t s[64];
    {
      s[0] = 0;
      if (!FILETIME_IsZero(fi.MTime))
        ConvertUtcFileTimeToString(fi.MTime, s,
            #ifndef UNDER_CE
              kTimestampPrintLevel_MIN
            #else
              kTimestampPrintLevel_DAY
            #endif
              );
      _list.SetSubItem(index, subItem++, s);
    }
    {
      s[0] = 0;
      if (!fi.IsDir())
        Browse_ConvertSizeToString(fi.Size, s);
      _list.SetSubItem(index, subItem++, s);
    }
  }

  if (_list.GetItemCount() > 0 && cursorIndex >= 0)
    _list.SetItemState_FocusedSelected(cursorIndex);
  _list.SortItems(CompareItems2, (LPARAM)this);
  if (_list.GetItemCount() > 0 && cursorIndex < 0)
    _list.SetItemState(0, LVIS_FOCUSED, LVIS_FOCUSED);
  _list.EnsureVisible(_list.GetFocusedItem(), false);
  _list.SetRedraw(true);
  _list.InvalidateRect(NULL, true);
  return S_OK;
}

HRESULT CBrowseDialog::Reload()
{
  UString selected;
  const int index = _list.GetNextSelectedItem(-1);
  if (index >= 0)
  {
    const int fileIndex = GetRealItemIndex(index);
    if (fileIndex != kParentIndex)
      selected = fs2us(_files[fileIndex].Name);
  }
  const UString dirPathTemp = DirPrefix;
  return Reload(dirPathTemp, selected);
}

void CBrowseDialog::OpenParentFolder()
{
  UString parent, selected;
  if (GetParentPath(DirPrefix, parent, selected))
  {
    Reload(parent, selected);
    SetPathEditText();
  }
}

void CBrowseDialog::SetPathEditText()
{
  const int index = _list.GetNextSelectedItem(-1);
  if (index < 0)
  {
    if (FolderMode)
      _pathEdit.SetText(DirPrefix);
    return;
  }
  const int fileIndex = GetRealItemIndex(index);
  if (fileIndex == kParentIndex)
  {
    if (FolderMode)
      _pathEdit.SetText(L".." WSTRING_PATH_SEPARATOR);
    return;
  }
  const CFileInfo &file = _files[fileIndex];
  if (file.IsDir())
  {
    if (!FolderMode)
      return;
    _pathEdit.SetText(fs2us(file.Name) + WCHAR_PATH_SEPARATOR);
  }
  else
    _pathEdit.SetText(fs2us(file.Name));
}

void CBrowseDialog::OnCreateDir()
{
  UString name;
  {
    UString enteredName;
    Dlg_CreateFolder((HWND)*this, enteredName);
    if (enteredName.IsEmpty())
      return;
    if (!CorrectFsPath(DirPrefix, enteredName, name))
    {
      MessageBox_HResError((HWND)*this, ERROR_INVALID_NAME, name);
      return;
    }
  }
  if (name.IsEmpty())
    return;

  FString destPath;
  if (GetFullPath(us2fs(DirPrefix), us2fs(name), destPath))
  {
    if (!NDir::CreateComplexDir(destPath))
    {
      MessageBox_HResError((HWND)*this, GetLastError_noZero_HRESULT(), fs2us(destPath));
    }
    else
    {
      UString tempPath = DirPrefix;
      Reload(tempPath, name);
      SetPathEditText();
    }
    _list.SetFocus();
  }
}

void CBrowseDialog::OnItemEnter()
{
  const int index = _list.GetNextSelectedItem(-1);
  if (index < 0)
    return;
  const int fileIndex = GetRealItemIndex(index);
  if (fileIndex == kParentIndex)
    OpenParentFolder();
  else
  {
    const CFileInfo &file = _files[fileIndex];
    if (!file.IsDir())
    {
      if (!FolderMode)
        FinishOnOK();
      /*
      MessageBox_Error_Global(*this, FolderMode ?
            L"You must select some folder":
            L"You must select some file");
      */
      return;
    }
    UString s = DirPrefix;
    s += fs2us(file.Name);
    s.Add_PathSepar();
    const HRESULT res = Reload(s, UString());
    if (res != S_OK)
      MessageBox_HResError(*this, res, s);
    SetPathEditText();
  }
}

void CBrowseDialog::FinishOnOK()
{
  UString s;
  _pathEdit.GetText(s);
  FString destPath;
  if (!GetFullPath(us2fs(DirPrefix), us2fs(s), destPath))
  {
    MessageBox_HResError((HWND)*this, ERROR_INVALID_NAME, s);
    return;
  }
  FilePath = fs2us(destPath);
  if (FolderMode)
    NormalizeDirPathPrefix(FilePath);
  FilterIndex = _filterCombo.GetCurSel();
  End(IDOK);
}

#endif // USE_MY_BROWSE_DIALOG



bool MyBrowseForFolder(HWND owner, LPCWSTR title, LPCWSTR path, UString &resultPath)
{
  resultPath.Empty();

  #ifndef UNDER_CE

#ifdef USE_MY_BROWSE_DIALOG
  if (!IsSuperOrDevicePath(path))
  if (MyStringLen(path) < MAX_PATH)
#endif
    return NShell::BrowseForFolder(owner, title, path, resultPath);

  #endif //  UNDER_CE

  #ifdef USE_MY_BROWSE_DIALOG

  CBrowseDialog dialog;
  dialog.FolderMode = true;
  if (title)
    dialog.Title = title;
  if (path)
    dialog.FilePath = path;
  if (dialog.Create(owner) != IDOK)
    return false;
  resultPath = dialog.FilePath;
  return true;

  #endif
}


// LPCWSTR filterDescription, LPCWSTR filter,

bool CBrowseInfo::BrowseForFile(const CObjectVector<CBrowseFilterInfo> &filters)
{
#ifndef UNDER_CE
#ifdef USE_MY_BROWSE_DIALOG
  /* win10:
     GetOpenFileName() for FilePath doesn't support super prefix "\\\\?\\"
     GetOpenFileName() for FilePath doesn't support long path
  */
  if (!IsSuperOrDevicePath(FilePath))
  // if (filters.Size() > 100) // for debug
#endif
  {
    const UString filePath_Store = FilePath;
    UString dirPrefix;
    {
      FString prefix, name;
      if (NDir::GetFullPathAndSplit(us2fs(FilePath), prefix, name))
      {
        dirPrefix = fs2us(prefix);
        FilePath = fs2us(name);
      }
    }
    UStringVector filters2;
    FOR_VECTOR (i, filters)
    {
      const CBrowseFilterInfo &fi = filters[i];
      filters2.Add(fi.Description);
      UString s;
      FOR_VECTOR (k, fi.Masks)
      {
        if (k != 0)
          s += ";";
        s += fi.Masks[k];
      }
      filters2.Add(s);
    }
    if (CommonDlg_BrowseForFile(!dirPrefix.IsEmpty() ? dirPrefix.Ptr(): NULL, filters2))
      return true;
    FilePath = filePath_Store;

  #ifdef UNDER_CE
    return false;
  #else
    // maybe we must use GetLastError in WinCE.
    const DWORD errorCode = CommDlgExtendedError();
  #ifdef USE_MY_BROWSE_DIALOG
    // FNERR_INVALIDFILENAME is expected error, if long path was used
    if (errorCode != FNERR_INVALIDFILENAME
        || FilePath.Len() < MAX_PATH)
  #endif
    {
      if (errorCode == 0)  // cancel or close on dialog
        return false;
      const char *message = NULL;
      if (errorCode == FNERR_INVALIDFILENAME)
        message = "Invalid file name";
      UString s ("Open Dialog Error:");
      s.Add_LF();
      if (message)
        s += message;
      else
      {
        char temp[16];
        ConvertUInt32ToHex8Digits(errorCode, temp);
        s += "Error #";
        s += temp;
      }
      s.Add_LF();
      s += FilePath;
      MessageBox_Error_Global(hwndOwner, s);
    }
  #endif // UNDER_CE
  }

#endif // UNDER_CE
  
#ifdef USE_MY_BROWSE_DIALOG

  CBrowseDialog dialog;

  dialog.FolderMode = false;
  dialog.SaveMode = SaveMode;
  dialog.FilterIndex = FilterIndex;
  dialog.Filters = filters;

  if (lpstrTitle)
    dialog.Title = lpstrTitle;
  dialog.FilePath = FilePath;
  if (dialog.Create(hwndOwner) != IDOK)
    return false;
  FilePath = dialog.FilePath;
  FilterIndex = dialog.FilterIndex;
#endif

  return true;
}


#ifdef _WIN32

static void RemoveDotsAndSpaces(UString &path)
{
  while (!path.IsEmpty())
  {
    wchar_t c = path.Back();
    if (c != ' ' && c != '.')
      return;
    path.DeleteBack();
  }
}


bool CorrectFsPath(const UString &relBase, const UString &path2, UString &result)
{
  result.Empty();

  UString path = path2;
  #ifdef _WIN32
  path.Replace(L'/', WCHAR_PATH_SEPARATOR);
  #endif
  unsigned start = 0;
  UString base;
  
  if (IsAbsolutePath(path))
  {
    #if defined(_WIN32) && !defined(UNDER_CE)
    if (IsSuperOrDevicePath(path))
    {
      result = path;
      return true;
    }
    #endif
    start = GetRootPrefixSize(path);
  }
  else
  {
    #if defined(_WIN32) && !defined(UNDER_CE)
    if (IsSuperOrDevicePath(relBase))
    {
      result = path;
      return true;
    }
    #endif
    base = relBase;
  }

  /* We can't use backward, since we must change only disk paths */
  /*
  for (;;)
  {
    if (path.Len() <= start)
      break;
    if (DoesFileOrDirExist(us2fs(path)))
      break;
    if (path.Back() == WCHAR_PATH_SEPARATOR)
    {
      path.DeleteBack();
      result.Insert(0, WCHAR_PATH_SEPARATOR);
    }
    int pos = path.ReverseFind(WCHAR_PATH_SEPARATOR) + 1;
    UString cur = path.Ptr(pos);
    RemoveDotsAndSpaces(cur);
    result.Insert(0, cur);
    path.DeleteFrom(pos);
  }
  result.Insert(0, path);
  return true;
  */

  result += path.Left(start);
  bool checkExist = true;
  UString cur;

  for (;;)
  {
    if (start == path.Len())
      break;
    const int slashPos = path.Find(WCHAR_PATH_SEPARATOR, start);
    cur.SetFrom(path.Ptr(start), (slashPos < 0 ? path.Len() : (unsigned)slashPos) - start);
    if (checkExist)
    {
      CFileInfo fi;
      if (fi.Find(us2fs(base + result + cur)))
      {
        if (!fi.IsDir())
        {
          result = path;
          break;
        }
      }
      else
        checkExist = false;
    }
    if (!checkExist)
      RemoveDotsAndSpaces(cur);
    result += cur;
    if (slashPos < 0)
      break;
    start = (unsigned)(slashPos + 1);
    result.Add_PathSepar();
  }
  
  return true;
}

#else

bool CorrectFsPath(const UString & /* relBase */, const UString &path, UString &result)
{
  result = path;
  return true;
}

#endif

bool Dlg_CreateFolder(HWND wnd, UString &destName)
{
  destName.Empty();
  CComboDialog dlg;
  LangString(IDS_CREATE_FOLDER, dlg.Title);
  LangString(IDS_CREATE_FOLDER_NAME, dlg.Static);
  LangString(IDS_CREATE_FOLDER_DEFAULT_NAME, dlg.Value);
  if (dlg.Create(wnd) != IDOK)
    return false;
  destName = dlg.Value;
  return true;
}
