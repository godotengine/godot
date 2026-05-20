// Windows/Control/Dialog.cpp

#include "StdAfx.h"

// #include "../../Windows/DLL.h"

#ifndef _UNICODE
#include "../../Common/StringConvert.h"
#endif

#include "Dialog.h"

extern HINSTANCE g_hInstance;
#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {
namespace NControl {

static
#ifdef Z7_OLD_WIN_SDK
  BOOL
#else
  INT_PTR
#endif
APIENTRY
DialogProcedure(HWND dialogHWND, UINT message, WPARAM wParam, LPARAM lParam)
{
  CWindow tempDialog(dialogHWND);
  if (message == WM_INITDIALOG)
    tempDialog.SetUserDataLongPtr(lParam);
  CDialog *dialog = (CDialog *)(tempDialog.GetUserDataLongPtr());
  if (dialog == NULL)
    return FALSE;
  if (message == WM_INITDIALOG)
    dialog->Attach(dialogHWND);

  /* MSDN: The dialog box procedure should return
       TRUE  - if it processed the message
       FALSE - if it did not process the message
     If the dialog box procedure returns FALSE,
     the dialog manager performs the default dialog operation in response to the message.
  */

  try { return BoolToBOOL(dialog->OnMessage(message, wParam, lParam)); }
  catch(...) { return TRUE; }
}

bool CDialog::OnMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
  switch (message)
  {
    case WM_INITDIALOG: return OnInit();
    case WM_COMMAND: return OnCommand(HIWORD(wParam), LOWORD(wParam), lParam);
    case WM_NOTIFY: return OnNotify((UINT)wParam, (LPNMHDR) lParam);
    case WM_TIMER: return OnTimer(wParam, lParam);
    case WM_SIZE: return OnSize(wParam, LOWORD(lParam), HIWORD(lParam));
    case WM_DESTROY: return OnDestroy();
    case WM_HELP: OnHelp(); return true;
    /*
        OnHelp(
          #ifdef UNDER_CE
          (void *)
          #else
          (LPHELPINFO)
          #endif
          lParam);
        return true;
    */
    default: return false;
  }
}

/*
bool CDialog::OnCommand2(WPARAM wParam, LPARAM lParam)
{
  return OnCommand(HIWORD(wParam), LOWORD(wParam), lParam);
}
*/

bool CDialog::OnCommand(unsigned code, unsigned itemID, LPARAM lParam)
{
  if (code == BN_CLICKED)
    return OnButtonClicked(itemID, (HWND)lParam);
  return false;
}

bool CDialog::OnButtonClicked(unsigned buttonID, HWND /* buttonHWND */)
{
  switch (buttonID)
  {
    case IDOK: OnOK(); break;
    case IDCANCEL: OnCancel(); break;
    case IDCLOSE: OnClose(); break;
    case IDCONTINUE: OnContinue(); break;
    case IDHELP: OnHelp(); break;
    default: return false;
  }
  return true;
}

#ifndef UNDER_CE
/* in win2000/win98 : monitor functions are supported.
   We need dynamic linking, if we want nt4/win95 support in program.
   Even if we compile the code with low (WINVER) value, we still
   want to use monitor functions. So we declare missing functions here */
// #if (WINVER < 0x0500)
#ifndef MONITOR_DEFAULTTOPRIMARY
extern "C" {
DECLARE_HANDLE(HMONITOR);
#define MONITOR_DEFAULTTOPRIMARY    0x00000001
typedef struct tagMONITORINFO
{
    DWORD   cbSize;
    RECT    rcMonitor;
    RECT    rcWork;
    DWORD   dwFlags;
} MONITORINFO, *LPMONITORINFO;
WINUSERAPI HMONITOR WINAPI MonitorFromWindow(HWND hwnd, DWORD dwFlags);
WINUSERAPI BOOL WINAPI GetMonitorInfoA(HMONITOR hMonitor, LPMONITORINFO lpmi);
}
#endif
#endif

static bool GetWorkAreaRect(RECT *rect, HWND hwnd)
{
  if (hwnd)
  {
    #ifndef UNDER_CE
    /* MonitorFromWindow() is supported in Win2000+
       MonitorFromWindow() : retrieves a handle to the display monitor that has the
         largest area of intersection with the bounding rectangle of a specified window.
       dwFlags: Determines the function's return value if the window does not intersect any display monitor.
         MONITOR_DEFAULTTONEAREST : Returns display that is nearest to the window.
         MONITOR_DEFAULTTONULL    : Returns NULL.
         MONITOR_DEFAULTTOPRIMARY : Returns the primary display monitor.
    */
    const HMONITOR hmon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
    if (hmon)
    {
      MONITORINFO mi;
      memset(&mi, 0, sizeof(mi));
      mi.cbSize = sizeof(mi);
      if (GetMonitorInfoA(hmon, &mi))
      {
        *rect = mi.rcWork;
        return true;
      }
    }
    #endif
  }

  /* Retrieves the size of the work area on the primary display monitor.
     The work area is the portion of the screen not obscured
     by the system taskbar or by application desktop toolbars.
     Any DPI virtualization mode of the caller has no effect on this output. */

  return BOOLToBool(::SystemParametersInfo(SPI_GETWORKAREA, 0, rect, 0));
}


bool IsDialogSizeOK(int xSize, int ySize, HWND hwnd)
{
  // it returns for system font. Real font uses another values
  const LONG v = GetDialogBaseUnits();
  const int x = LOWORD(v);
  const int y = HIWORD(v);

  RECT rect;
  GetWorkAreaRect(&rect, hwnd);
  const int wx = RECT_SIZE_X(rect);
  const int wy = RECT_SIZE_Y(rect);
  return
    xSize / 4 * x <= wx &&
    ySize / 8 * y <= wy;
}

bool CDialog::GetMargins(int margin, int &x, int &y)
{
  x = margin;
  y = margin;
  RECT rect;
  rect.left = 0;
  rect.top = 0;
  rect.right = margin;
  rect.bottom = margin;
  if (!MapRect(&rect))
    return false;
  x = rect.right - rect.left;
  y = rect.bottom - rect.top;
  return true;
}

int CDialog::Units_To_Pixels_X(int units)
{
  RECT rect;
  rect.left = 0;
  rect.top = 0;
  rect.right = units;
  rect.bottom = units;
  if (!MapRect(&rect))
    return units * 3 / 2;
  return rect.right - rect.left;
}

bool CDialog::GetItemSizes(unsigned id, int &x, int &y)
{
  RECT rect;
  if (!::GetWindowRect(GetItem(id), &rect))
    return false;
  x = RECT_SIZE_X(rect);
  y = RECT_SIZE_Y(rect);
  return true;
}

void CDialog::GetClientRectOfItem(unsigned id, RECT &rect)
{
  ::GetWindowRect(GetItem(id), &rect);
  ScreenToClient(&rect);
}

bool CDialog::MoveItem(unsigned id, int x, int y, int width, int height, bool repaint)
{
  return BOOLToBool(::MoveWindow(GetItem(id), x, y, width, height, BoolToBOOL(repaint)));
}


/*
typedef BOOL (WINAPI * Func_DwmGetWindowAttribute)(
    HWND hwnd, DWORD dwAttribute, PVOID pvAttribute, DWORD cbAttribute);

static bool GetWindowsRect_DWM(HWND hwnd, RECT *rect)
{
  // dll load and free is too slow : 300 calls in second.
  NDLL::CLibrary dll;
  if (!dll.Load(FTEXT("dwmapi.dll")))
    return false;
  Func_DwmGetWindowAttribute f = (Func_DwmGetWindowAttribute)dll.GetProc("DwmGetWindowAttribute" );
  if (f)
  {
    #define MY__DWMWA_EXTENDED_FRAME_BOUNDS 9
    // 30000 per second
    RECT r;
    if (f(hwnd, MY__DWMWA_EXTENDED_FRAME_BOUNDS, &r, sizeof(RECT)) == S_OK)
    {
      *rect = r;
      return true;
    }
  }
  return false;
}
*/


static bool IsRect_Small_Inside_Big(const RECT &sm, const RECT &big)
{
  return sm.left   >= big.left
      && sm.right  <= big.right
      && sm.top    >= big.top
      && sm.bottom <= big.bottom;
}


static bool AreRectsOverlapped(const RECT &r1, const RECT &r2)
{
  return r1.left   < r2.right
      && r1.right  > r2.left
      && r1.top    < r2.bottom
      && r1.bottom > r2.top;
}


static bool AreRectsEqual(const RECT &r1, const RECT &r2)
{
  return r1.left   == r2.left
      && r1.right  == r2.right
      && r1.top    == r2.top
      && r1.bottom == r2.bottom;
}


void CDialog::NormalizeSize(bool fullNormalize)
{
  RECT workRect;
  if (!GetWorkAreaRect(&workRect, *this))
    return;
  RECT rect;
  if (!GetWindowRect(&rect))
    return;
  int xs = RECT_SIZE_X(rect);
  int ys = RECT_SIZE_Y(rect);

  // we don't want to change size using workRect, if window is outside of WorkArea
  if (!AreRectsOverlapped(rect, workRect))
    return;

  /* here rect and workRect are overlapped, but it can be false
     overlapping of small shadow when window in another display. */

  const int xsW = RECT_SIZE_X(workRect);
  const int ysW = RECT_SIZE_Y(workRect);
  if (xs <= xsW && ys <= ysW)
    return; // size of window is OK
  if (fullNormalize)
  {
    Show(SW_SHOWMAXIMIZED);
    return;
  }
  int x = workRect.left;
  int y = workRect.top;
  if (xs < xsW)  x += (xsW - xs) / 2;  else xs = xsW;
  if (ys < ysW)  y += (ysW - ys) / 2;  else ys = ysW;
  Move(x, y, xs, ys, true);
}


void CDialog::NormalizePosition()
{
  RECT workRect;
  if (!GetWorkAreaRect(&workRect, *this))
    return;

  RECT rect2 = workRect;
  bool useWorkArea = true;
  const HWND parentHWND = GetParent();

  if (parentHWND)
  {
    RECT workRectParent;
    if (!GetWorkAreaRect(&workRectParent, parentHWND))
      return;

    // if windows are in different monitors, we use only workArea of current window

    if (AreRectsEqual(workRectParent, workRect))
    {
      // RECT rect3; if (GetWindowsRect_DWM(parentHWND, &rect3)) {}
      CWindow wnd(parentHWND);
      if (wnd.GetWindowRect(&rect2))
      {
        // it's same monitor. So we try to use parentHWND rect.
        /* we don't want to change position, if parent window is not inside work area.
           In Win10 : parent window rect is 8 pixels larger for each corner than window size for shadow.
           In maximize mode : window is outside of workRect.
           if parent window is inside workRect, we will use parent window instead of workRect */
        if (IsRect_Small_Inside_Big(rect2, workRect))
          useWorkArea = false;
      }
    }
  }

  RECT rect;
  if (!GetWindowRect(&rect))
    return;

  if (useWorkArea)
  {
    // we don't want to move window, if it's already inside.
    if (IsRect_Small_Inside_Big(rect, workRect))
      return;
    // we don't want to move window, if it's outside of workArea
    if (!AreRectsOverlapped(rect, workRect))
      return;
    rect2 = workRect;
  }

  {
    const int xs = RECT_SIZE_X(rect);
    const int ys = RECT_SIZE_Y(rect);
    const int xs2 = RECT_SIZE_X(rect2);
    const int ys2 = RECT_SIZE_Y(rect2);
    // we don't want to change position if parent is smaller.
    if (xs <= xs2 && ys <= ys2)
    {
      const int x = rect2.left + (xs2 - xs) / 2;
      const int y = rect2.top  + (ys2 - ys) / 2;

      if (x != rect.left || y != rect.top)
        Move(x, y, xs, ys, true);
      // SetWindowPos(*this, HWND_TOP, x, y, 0, 0, SWP_NOSIZE);
      return;
    }
  }
}



bool CModelessDialog::Create(LPCTSTR templateName, HWND parentWindow)
{
  const HWND aHWND = CreateDialogParam(g_hInstance, templateName, parentWindow, DialogProcedure, (LPARAM)this);
  if (!aHWND)
    return false;
  Attach(aHWND);
  return true;
}

INT_PTR CModalDialog::Create(LPCTSTR templateName, HWND parentWindow)
{
  return DialogBoxParam(g_hInstance, templateName, parentWindow, DialogProcedure, (LPARAM)this);
}

#ifndef _UNICODE

bool CModelessDialog::Create(LPCWSTR templateName, HWND parentWindow)
{
  HWND aHWND;
  if (g_IsNT)
    aHWND = CreateDialogParamW(g_hInstance, templateName, parentWindow, DialogProcedure, (LPARAM)this);
  else
  {
    AString name;
    LPCSTR templateNameA;
    if (IS_INTRESOURCE(templateName))
      templateNameA = (LPCSTR)templateName;
    else
    {
      name = GetSystemString(templateName);
      templateNameA = name;
    }
    aHWND = CreateDialogParamA(g_hInstance, templateNameA, parentWindow, DialogProcedure, (LPARAM)this);
  }
  if (aHWND == 0)
    return false;
  Attach(aHWND);
  return true;
}

INT_PTR CModalDialog::Create(LPCWSTR templateName, HWND parentWindow)
{
  if (g_IsNT)
    return DialogBoxParamW(g_hInstance, templateName, parentWindow, DialogProcedure, (LPARAM)this);
  AString name;
  LPCSTR templateNameA;
  if (IS_INTRESOURCE(templateName))
    templateNameA = (LPCSTR)templateName;
  else
  {
    name = GetSystemString(templateName);
    templateNameA = name;
  }
  return DialogBoxParamA(g_hInstance, templateNameA, parentWindow, DialogProcedure, (LPARAM)this);
}
#endif

}}
