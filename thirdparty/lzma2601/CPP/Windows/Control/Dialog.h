// Windows/Control/Dialog.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_DIALOG_H
#define ZIP7_INC_WINDOWS_CONTROL_DIALOG_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

#ifndef IDCONTINUE
#define IDCONTINUE 11
#endif

class CDialog: public CWindow
{
  // Z7_CLASS_NO_COPY(CDialog)
public:
  CDialog(HWND wnd = NULL): CWindow(wnd) {}
  virtual ~CDialog() {}

  HWND GetItem(unsigned itemID) const
    { return GetDlgItem(_window, (int)itemID); }

  bool EnableItem(unsigned itemID, bool enable) const
    { return BOOLToBool(::EnableWindow(GetItem(itemID), BoolToBOOL(enable))); }

  bool ShowItem(unsigned itemID, int cmdShow) const
    { return BOOLToBool(::ShowWindow(GetItem(itemID), cmdShow)); }

  bool ShowItem_Bool(unsigned itemID, bool show) const
    { return ShowItem(itemID, show ? SW_SHOW: SW_HIDE); }

  bool HideItem(unsigned itemID) const { return ShowItem(itemID, SW_HIDE); }

  bool SetItemText(unsigned itemID, LPCTSTR s)
    { return BOOLToBool(SetDlgItemText(_window, (int)itemID, s)); }

  bool SetItemTextA(unsigned itemID, LPCSTR s)
    { return BOOLToBool(SetDlgItemTextA(_window, (int)itemID, s)); }

  bool SetItemText_Empty(unsigned itemID)
    { return SetItemText(itemID, TEXT("")); }

  #ifndef _UNICODE
  bool SetItemText(unsigned itemID, LPCWSTR s)
  {
    CWindow window(GetItem(itemID));
    return window.SetText(s);
  }
  #endif

  UINT GetItemText(unsigned itemID, LPTSTR string, unsigned maxCount)
    { return GetDlgItemText(_window, (int)itemID, string, (int)maxCount); }
  #ifndef _UNICODE
  /*
  bool GetItemText(unsigned itemID, LPWSTR string, int maxCount)
  {
    CWindow window(GetItem(unsigned));
    return window.GetText(string, maxCount);
  }
  */
  #endif

  bool GetItemText(unsigned itemID, UString &s)
  {
    CWindow window(GetItem(itemID));
    return window.GetText(s);
  }

/*
  bool SetItemInt(unsigned itemID, UINT value, bool isSigned)
    { return BOOLToBool(SetDlgItemInt(_window, (int)itemID, value, BoolToBOOL(isSigned))); }
*/
  bool SetItemUInt(unsigned itemID, UINT value)
    { return BOOLToBool(SetDlgItemInt(_window, (int)itemID, value, FALSE)); }
/*
  bool GetItemInt(unsigned itemID, bool isSigned, UINT &value)
  {
    BOOL result;
    value = GetDlgItemInt(_window, (int)itemID, &result, BoolToBOOL(isSigned));
    return BOOLToBool(result);
  }
*/
  bool GetItemUInt(unsigned itemID, UINT &value)
  {
    BOOL result;
    value = GetDlgItemInt(_window, (int)itemID, &result, FALSE);
    return BOOLToBool(result);
  }

  HWND GetNextGroupItem(HWND control, bool previous)
    { return GetNextDlgGroupItem(_window, control, BoolToBOOL(previous)); }
  HWND GetNextTabItem(HWND control, bool previous)
    { return GetNextDlgTabItem(_window, control, BoolToBOOL(previous)); }

  LRESULT SendMsg_NextDlgCtl(WPARAM wParam, LPARAM lParam)
    { return SendMsg(WM_NEXTDLGCTL, wParam, lParam); }
  LRESULT SendMsg_NextDlgCtl_HWND(HWND hwnd) { return SendMsg_NextDlgCtl((WPARAM)hwnd, TRUE); }
  LRESULT SendMsg_NextDlgCtl_CtlId(unsigned id)   { return SendMsg_NextDlgCtl_HWND(GetItem(id)); }
  LRESULT SendMsg_NextDlgCtl_Next()          { return SendMsg_NextDlgCtl(0, FALSE); }
  LRESULT SendMsg_NextDlgCtl_Prev()          { return SendMsg_NextDlgCtl(1, FALSE); }

  bool MapRect(LPRECT rect)
    { return BOOLToBool(MapDialogRect(_window, rect)); }

  bool IsMessage(LPMSG message)
    { return BOOLToBool(IsDialogMessage(_window, message)); }

  LRESULT SendItemMessage(unsigned itemID, UINT message, WPARAM wParam, LPARAM lParam)
    { return SendDlgItemMessage(_window, (int)itemID, message, wParam, lParam); }

  bool CheckButton(unsigned buttonID, UINT checkState)
    { return BOOLToBool(CheckDlgButton(_window, (int)buttonID, checkState)); }
  bool CheckButton(unsigned buttonID, bool checkState)
    { return CheckButton(buttonID, UINT(checkState ? BST_CHECKED : BST_UNCHECKED)); }

  UINT IsButtonChecked_BST(unsigned buttonID) const
    { return IsDlgButtonChecked(_window, (int)buttonID); }
  bool IsButtonCheckedBool(unsigned buttonID) const
    { return (IsButtonChecked_BST(buttonID) == BST_CHECKED); }

  bool CheckRadioButton(unsigned firstButtonID, unsigned lastButtonID, unsigned checkButtonID)
    { return BOOLToBool(::CheckRadioButton(_window,
        (int)firstButtonID, (int)lastButtonID, (int)checkButtonID)); }

  virtual bool OnMessage(UINT message, WPARAM wParam, LPARAM lParam);
  virtual bool OnInit() { return true; }
  // virtual bool OnCommand2(WPARAM wParam, LPARAM lParam);
  virtual bool OnCommand(unsigned code, unsigned itemID, LPARAM lParam);
  virtual bool OnSize(WPARAM /* wParam */, int /* xSize */, int /* ySize */) { return false; }
  virtual bool OnDestroy() { return false; }

  /*
  #ifdef UNDER_CE
  virtual void OnHelp(void *) { OnHelp(); }
  #else
  virtual void OnHelp(LPHELPINFO) { OnHelp(); }
  #endif
  */
  virtual void OnHelp() {}

  virtual bool OnButtonClicked(unsigned buttonID, HWND buttonHWND);
  virtual void OnOK() {}
  virtual void OnContinue() {}
  virtual void OnCancel() {}
  virtual void OnClose() {}
  virtual bool OnNotify(UINT /* controlID */, LPNMHDR /* lParam */) { return false; }
  virtual bool OnTimer(WPARAM /* timerID */, LPARAM /* callback */) { return false; }

  LONG_PTR SetMsgResult(LONG_PTR newLongPtr )
    { return SetLongPtr(DWLP_MSGRESULT, newLongPtr); }
  LONG_PTR GetMsgResult() const
    { return GetLongPtr(DWLP_MSGRESULT); }

  bool GetMargins(int margin, int &x, int &y);
  int Units_To_Pixels_X(int units);
  bool GetItemSizes(unsigned id, int &x, int &y);
  void GetClientRectOfItem(unsigned id, RECT &rect);
  bool MoveItem(unsigned id, int x, int y, int width, int height, bool repaint = true);
  bool MoveItem_RECT(unsigned id, const RECT &r, bool repaint = true)
    { return MoveItem(id, r.left, r.top, RECT_SIZE_X(r), RECT_SIZE_Y(r), repaint); }

  void NormalizeSize(bool fullNormalize = false);
  void NormalizePosition();
};

class CModelessDialog: public CDialog
{
public:
  bool Create(LPCTSTR templateName, HWND parentWindow);
  bool Create(UINT resID, HWND parentWindow) { return Create(MAKEINTRESOURCEW(resID), parentWindow); }
  #ifndef _UNICODE
  bool Create(LPCWSTR templateName, HWND parentWindow);
  #endif
  virtual void OnOK() Z7_override { Destroy(); }
  virtual void OnContinue() Z7_override { Destroy(); }
  virtual void OnCancel() Z7_override { Destroy(); }
  virtual void OnClose() Z7_override { Destroy(); }
};

class CModalDialog: public CDialog
{
public:
  INT_PTR Create(LPCTSTR templateName, HWND parentWindow);
  INT_PTR Create(UINT resID, HWND parentWindow) { return Create(MAKEINTRESOURCEW(resID), parentWindow); }
  #ifndef _UNICODE
  INT_PTR Create(LPCWSTR templateName, HWND parentWindow);
  #endif

  bool End(INT_PTR result) { return BOOLToBool(::EndDialog(_window, result)); }
  virtual void OnOK() Z7_override { End(IDOK); }
  virtual void OnContinue() Z7_override { End(IDCONTINUE); }
  virtual void OnCancel() Z7_override { End(IDCANCEL); }
  virtual void OnClose() Z7_override { End(IDCLOSE); }
};

class CDialogChildControl: public NWindows::CWindow
{
  // unsigned m_ID;
public:
  void Init(const NWindows::NControl::CDialog &parentDialog, unsigned id)
  {
    // m_ID = id;
    Attach(parentDialog.GetItem(id));
  }
};

bool IsDialogSizeOK(int xSize, int ySize, HWND hwnd = NULL);

}}

#endif
