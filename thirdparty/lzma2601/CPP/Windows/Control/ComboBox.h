// Windows/Control/ComboBox.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_COMBOBOX_H
#define ZIP7_INC_WINDOWS_CONTROL_COMBOBOX_H

#include "../../Common/MyWindows.h"

#include <CommCtrl.h>

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CComboBox: public CWindow
{
public:
  void ResetContent() { SendMsg(CB_RESETCONTENT, 0, 0); }
  LRESULT AddString(LPCTSTR s) { return SendMsg(CB_ADDSTRING, 0, (LPARAM)s); }
  #ifndef _UNICODE
  LRESULT AddString(LPCWSTR s);
  #endif

  LRESULT AddString_SetItemData(LPCWSTR s, LPARAM lParam);

  /* If this parameter is -1, any current selection in the list is removed and the edit control is cleared.*/
  LRESULT SetCurSel(int index) { return SendMsg(CB_SETCURSEL, MY_int_TO_WPARAM(index), 0); }
  LRESULT SetCurSel(unsigned index) { return SendMsg(CB_SETCURSEL, index, 0); }

  /* If no item is selected, it returns CB_ERR (-1) */
  int GetCurSel() { return (int)SendMsg(CB_GETCURSEL, 0, 0); }
  
  /*  If an error occurs, it is CB_ERR (-1) */
  int GetCount() { return (int)SendMsg(CB_GETCOUNT, 0, 0); }
  
  LRESULT GetLBTextLen(int index) { return SendMsg(CB_GETLBTEXTLEN, MY_int_TO_WPARAM(index), 0); }
  LRESULT GetLBText(int index, LPTSTR s) { return SendMsg(CB_GETLBTEXT, MY_int_TO_WPARAM(index), (LPARAM)s); }
  LRESULT GetLBText(int index, CSysString &s);
  #ifndef _UNICODE
  LRESULT GetLBText(int index, UString &s);
  #endif

  LRESULT SetItemData(int index, LPARAM lParam) { return SendMsg(CB_SETITEMDATA, MY_int_TO_WPARAM(index), lParam); }
  LRESULT GetItemData(int index) { return SendMsg(CB_GETITEMDATA, MY_int_TO_WPARAM(index), 0); }
  LRESULT GetItemData(unsigned index) { return SendMsg(CB_GETITEMDATA, index, 0); }

  LRESULT GetItemData_of_CurSel() { return GetItemData(GetCurSel()); }

  void ShowDropDown(bool show = true) { SendMsg(CB_SHOWDROPDOWN, show ? TRUE : FALSE, 0);  }
};

#ifndef UNDER_CE

class CComboBoxEx: public CComboBox
{
public:
  bool SetUnicodeFormat(bool fUnicode) { return LRESULTToBool(SendMsg(CBEM_SETUNICODEFORMAT, BOOLToBool(fUnicode), 0)); }

  /* Returns:
      an INT value that represents the number of items remaining in the control.
      If (index) is invalid, the message returns CB_ERR. */
  LRESULT DeleteItem(int index) { return SendMsg(CBEM_DELETEITEM, MY_int_TO_WPARAM(index), 0); }

  LRESULT InsertItem(COMBOBOXEXITEM *item) { return SendMsg(CBEM_INSERTITEM, 0, (LPARAM)item); }
  #ifndef _UNICODE
  LRESULT InsertItem(COMBOBOXEXITEMW *item) { return SendMsg(CBEM_INSERTITEMW, 0, (LPARAM)item); }
  #endif

  LRESULT SetItem(COMBOBOXEXITEM *item) { return SendMsg(CBEM_SETITEM, 0, (LPARAM)item); }
  DWORD SetExtendedStyle(DWORD exMask, DWORD exStyle) { return (DWORD)SendMsg(CBEM_SETEXTENDEDSTYLE, exMask, (LPARAM)exStyle); }
  HWND GetEditControl() { return (HWND)SendMsg(CBEM_GETEDITCONTROL, 0, 0); }
  HIMAGELIST SetImageList(HIMAGELIST imageList) { return (HIMAGELIST)SendMsg(CBEM_SETIMAGELIST, 0, (LPARAM)imageList); }
};

#endif

}}

#endif
