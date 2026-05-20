// Windows/Control/Window2.cpp

#include "StdAfx.h"

#ifndef _UNICODE
#include "../../Common/StringConvert.h"
#endif

#include "Window2.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {

#ifndef _UNICODE
ATOM MyRegisterClass(CONST WNDCLASSW *wndClass);
#endif

namespace NControl {

#ifdef UNDER_CE
#define MY_START_WM_CREATE WM_CREATE
#else
#define MY_START_WM_CREATE WM_NCCREATE
#endif

static LRESULT CALLBACK WindowProcedure(HWND aHWND, UINT message, WPARAM wParam, LPARAM lParam)
{
  CWindow tempWindow(aHWND);
  if (message == MY_START_WM_CREATE)
    tempWindow.SetUserDataLongPtr((LONG_PTR)(((LPCREATESTRUCT)lParam)->lpCreateParams));
  CWindow2 *window = (CWindow2 *)(tempWindow.GetUserDataLongPtr());
  if (window && message == MY_START_WM_CREATE)
    window->Attach(aHWND);
  if (!window)
  {
    #ifndef _UNICODE
    if (g_IsNT)
      return DefWindowProcW(aHWND, message, wParam, lParam);
    else
    #endif
      return DefWindowProc(aHWND, message, wParam, lParam);
  }
  return window->OnMessage(message, wParam, lParam);
}

bool CWindow2::CreateEx(DWORD exStyle, LPCTSTR className, LPCTSTR windowName,
    DWORD style, int x, int y, int width, int height,
    HWND parentWindow, HMENU idOrHMenu, HINSTANCE instance)
{
  WNDCLASS wc;
  if (!::GetClassInfo(instance, className, &wc))
  {
    // wc.style          = CS_HREDRAW | CS_VREDRAW;
    wc.style          = 0;
    wc.lpfnWndProc    = WindowProcedure;
    wc.cbClsExtra     = 0;
    wc.cbWndExtra     = 0;
    wc.hInstance      = instance;
    wc.hIcon          = NULL;
    wc.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName   = NULL;
    wc.lpszClassName  = className;
    if (::RegisterClass(&wc) == 0)
      return false;
  }
  return CWindow::CreateEx(exStyle, className, windowName, style,
      x, y, width, height, parentWindow, idOrHMenu, instance, this);
}

#ifndef _UNICODE

bool CWindow2::CreateEx(DWORD exStyle, LPCWSTR className, LPCWSTR windowName,
    DWORD style, int x, int y, int width, int height,
    HWND parentWindow, HMENU idOrHMenu, HINSTANCE instance)
{
  bool needRegister;
  if (g_IsNT)
  {
    WNDCLASSW wc;
    needRegister = ::GetClassInfoW(instance, className, &wc) == 0;
  }
  else
  {
    WNDCLASSA windowClassA;
    AString classNameA;
    LPCSTR classNameP;
    if (IS_INTRESOURCE(className))
      classNameP = (LPCSTR)className;
    else
    {
      classNameA = GetSystemString(className);
      classNameP = classNameA;
    }
    needRegister = ::GetClassInfoA(instance, classNameP, &windowClassA) == 0;
  }
  if (needRegister)
  {
    WNDCLASSW wc;
    // wc.style          = CS_HREDRAW | CS_VREDRAW;
    wc.style          = 0;
    wc.lpfnWndProc    = WindowProcedure;
    wc.cbClsExtra     = 0;
    wc.cbWndExtra     = 0;
    wc.hInstance      = instance;
    wc.hIcon          = NULL;
    wc.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName   = NULL;
    wc.lpszClassName  = className;
    if (MyRegisterClass(&wc) == 0)
      return false;
  }
  return CWindow::CreateEx(exStyle, className, windowName, style,
      x, y, width, height, parentWindow, idOrHMenu, instance, this);
}

#endif

LRESULT CWindow2::DefProc(UINT message, WPARAM wParam, LPARAM lParam)
{
  #ifndef _UNICODE
  if (g_IsNT)
    return DefWindowProcW(_window, message, wParam, lParam);
  else
  #endif
    return DefWindowProc(_window, message, wParam, lParam);
}

LRESULT CWindow2::OnMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
  LRESULT result;
  switch (message)
  {
    case WM_CREATE:
      if (!OnCreate((CREATESTRUCT *)lParam))
        return -1;
      break;
    case WM_COMMAND:
      if (OnCommand(HIWORD(wParam), LOWORD(wParam), lParam, result))
        return result;
      break;
    case WM_NOTIFY:
      if (OnNotify((UINT)wParam, (LPNMHDR) lParam, result))
        return result;
      break;
    case WM_DESTROY:
      OnDestroy();
      break;
    case WM_CLOSE:
      OnClose();
      return 0;
    case WM_SIZE:
      if (OnSize(wParam, LOWORD(lParam), HIWORD(lParam)))
        return 0;
  }
  return DefProc(message, wParam, lParam);
}

/*
bool CWindow2::OnCommand2(WPARAM wParam, LPARAM lParam, LRESULT &result)
{
  return OnCommand(HIWORD(wParam), LOWORD(wParam), lParam, result);
}
*/

bool CWindow2::OnCommand(unsigned /* code */, unsigned /* itemID */, LPARAM /* lParam */, LRESULT & /* result */)
{
  return false;
  // return DefProc(message, wParam, lParam);
  /*
  if (code == BN_CLICKED)
    return OnButtonClicked(itemID, (HWND)lParam);
  */
}

/*
bool CDialog::OnButtonClicked(unsigned buttonID, HWND buttonHWND)
{
  switch (buttonID)
  {
    case IDOK:
      OnOK();
      break;
    case IDCANCEL:
      OnCancel();
      break;
    case IDHELP:
      OnHelp();
      break;
    default:
      return false;
  }
  return true;
}

*/

}}
