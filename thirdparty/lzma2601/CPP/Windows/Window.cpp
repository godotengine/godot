// Windows/Window.cpp

#include "StdAfx.h"

#ifndef _UNICODE
#include "../Common/StringConvert.h"
#endif
#include "Window.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {

#ifndef _UNICODE
ATOM MyRegisterClass(CONST WNDCLASSW *wndClass)
{
  if (g_IsNT)
    return RegisterClassW(wndClass);
  WNDCLASSA wndClassA;
  wndClassA.style = wndClass->style;
  wndClassA.lpfnWndProc = wndClass->lpfnWndProc;
  wndClassA.cbClsExtra = wndClass->cbClsExtra;
  wndClassA.cbWndExtra = wndClass->cbWndExtra;
  wndClassA.hInstance = wndClass->hInstance;
  wndClassA.hIcon = wndClass->hIcon;
  wndClassA.hCursor = wndClass->hCursor;
  wndClassA.hbrBackground = wndClass->hbrBackground;
  AString menuName;
  AString className;
  if (IS_INTRESOURCE(wndClass->lpszMenuName))
    wndClassA.lpszMenuName = (LPCSTR)wndClass->lpszMenuName;
  else
  {
    menuName = GetSystemString(wndClass->lpszMenuName);
    wndClassA.lpszMenuName = menuName;
  }
  if (IS_INTRESOURCE(wndClass->lpszClassName))
    wndClassA.lpszClassName = (LPCSTR)wndClass->lpszClassName;
  else
  {
    className = GetSystemString(wndClass->lpszClassName);
    wndClassA.lpszClassName = className;
  }
  return RegisterClassA(&wndClassA);
}

bool CWindow::Create(LPCWSTR className,
      LPCWSTR windowName, DWORD style,
      int x, int y, int width, int height,
      HWND parentWindow, HMENU idOrHMenu,
      HINSTANCE instance, LPVOID createParam)
{
  if (g_IsNT)
  {
    _window = ::CreateWindowW(className, windowName,
        style, x, y, width, height, parentWindow,
        idOrHMenu, instance, createParam);
     return (_window != NULL);
  }
  return Create(GetSystemString(className), GetSystemString(windowName),
        style, x, y, width, height, parentWindow,
        idOrHMenu, instance, createParam);
}

bool CWindow::CreateEx(DWORD exStyle, LPCWSTR className,
      LPCWSTR windowName, DWORD style,
      int x, int y, int width, int height,
      HWND parentWindow, HMENU idOrHMenu,
      HINSTANCE instance, LPVOID createParam)
{
  if (g_IsNT)
  {
    _window = ::CreateWindowExW(exStyle, className, windowName,
      style, x, y, width, height, parentWindow,
      idOrHMenu, instance, createParam);
     return (_window != NULL);
  }
  AString classNameA;
  LPCSTR classNameP;
  if (IS_INTRESOURCE(className))
    classNameP = (LPCSTR)className;
  else
  {
    classNameA = GetSystemString(className);
    classNameP = classNameA;
  }
  AString windowNameA;
  LPCSTR windowNameP;
  if (IS_INTRESOURCE(windowName))
    windowNameP = (LPCSTR)windowName;
  else
  {
    windowNameA = GetSystemString(windowName);
    windowNameP = windowNameA;
  }
  return CreateEx(exStyle, classNameP, windowNameP,
      style, x, y, width, height, parentWindow,
      idOrHMenu, instance, createParam);
}

#endif

#ifndef _UNICODE
bool MySetWindowText(HWND wnd, LPCWSTR s)
{
  if (g_IsNT)
    return BOOLToBool(::SetWindowTextW(wnd, s));
  return BOOLToBool(::SetWindowTextA(wnd, UnicodeStringToMultiByte(s)));
}
#endif

bool CWindow::GetText(CSysString &s) const
{
  s.Empty();
  unsigned len = (unsigned)GetTextLength();
  if (len == 0)
    return (::GetLastError() == ERROR_SUCCESS);
  TCHAR *p = s.GetBuf(len);
  {
    const unsigned len2 = (unsigned)GetText(p, (int)(len + 1));
    if (len > len2)
      len = len2;
  }
  s.ReleaseBuf_CalcLen(len);
  if (len == 0)
    return (::GetLastError() == ERROR_SUCCESS);
  return true;
}

#ifndef _UNICODE
bool CWindow::GetText(UString &s) const
{
  if (g_IsNT)
  {
    s.Empty();
    unsigned len = (unsigned)GetWindowTextLengthW(_window);
    if (len == 0)
      return (::GetLastError() == ERROR_SUCCESS);
    wchar_t *p = s.GetBuf(len);
    {
      const unsigned len2 = (unsigned)GetWindowTextW(_window, p, (int)(len + 1));
      if (len > len2)
        len = len2;
    }
    s.ReleaseBuf_CalcLen(len);
    if (len == 0)
      return (::GetLastError() == ERROR_SUCCESS);
    return true;
  }
  CSysString sysString;
  const bool result = GetText(sysString);
  MultiByteToUnicodeString2(s, sysString);
  return result;
}
#endif

 
/*
bool CWindow::ModifyStyleBase(int styleOffset,
  DWORD remove, DWORD add, UINT flags)
{
  DWORD style = GetWindowLong(styleOffset);
  DWORD newStyle = (style & ~remove) | add;
  if (style == newStyle)
    return false; // it is not good

  SetWindowLong(styleOffset, newStyle);
  if (flags != 0)
  {
    ::SetWindowPos(_window, NULL, 0, 0, 0, 0,
      SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | flags);
  }
  return TRUE;
}
*/

}
