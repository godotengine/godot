// Windows/Control/Window2.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_WINDOW2_H
#define ZIP7_INC_WINDOWS_CONTROL_WINDOW2_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CWindow2: public CWindow
{
  // Z7_CLASS_NO_COPY(CWindow2)

  LRESULT DefProc(UINT message, WPARAM wParam, LPARAM lParam);
public:
  CWindow2(HWND newWindow = NULL): CWindow(newWindow) {}
  virtual ~CWindow2() {}

  bool CreateEx(DWORD exStyle, LPCTSTR className, LPCTSTR windowName,
      DWORD style, int x, int y, int width, int height,
      HWND parentWindow, HMENU idOrHMenu, HINSTANCE instance);

  #ifndef _UNICODE
  bool CreateEx(DWORD exStyle, LPCWSTR className, LPCWSTR windowName,
      DWORD style, int x, int y, int width, int height,
      HWND parentWindow, HMENU idOrHMenu, HINSTANCE instance);
  #endif

  virtual LRESULT OnMessage(UINT message, WPARAM wParam, LPARAM lParam);
  virtual bool OnCreate(CREATESTRUCT * /* createStruct */) { return true; }
  // virtual LRESULT OnCommand(WPARAM wParam, LPARAM lParam);
  // bool OnCommand2(WPARAM wParam, LPARAM lParam, LRESULT &result);
  virtual bool OnCommand(unsigned code, unsigned itemID, LPARAM lParam, LRESULT &result);
  virtual bool OnSize(WPARAM /* wParam */, int /* xSize */, int /* ySize */) { return false; }
  virtual bool OnNotify(UINT /* controlID */, LPNMHDR /* lParam */, LRESULT & /* result */) { return false; }
  virtual void OnDestroy() { PostQuitMessage(0); }
  virtual void OnClose() { Destroy(); }
  /*
  virtual LRESULT  OnHelp(LPHELPINFO helpInfo) { OnHelp(); }
  virtual LRESULT  OnHelp() {};
  virtual bool OnButtonClicked(unsigned buttonID, HWND buttonHWND);
  virtual void OnOK() {};
  virtual void OnCancel() {};
  */

  LONG_PTR SetMsgResult(LONG_PTR newLongPtr) { return SetLongPtr(DWLP_MSGRESULT, newLongPtr); }
  LONG_PTR GetMsgResult() const { return GetLongPtr(DWLP_MSGRESULT); }
};

}}

#endif
