// Windows/Control/ListView.cpp

#include "StdAfx.h"

#include "ListView.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {
namespace NControl {

bool CListView::CreateEx(DWORD exStyle, DWORD style,
      int x, int y, int width, int height,
      HWND parentWindow, HMENU idOrHMenu,
      HINSTANCE instance, LPVOID createParam)
{
  return CWindow::CreateEx(exStyle, WC_LISTVIEW, TEXT(""), style, x, y, width,
      height, parentWindow, idOrHMenu, instance, createParam);
}

/* note: LVITEM and LVCOLUMN structures contain optional fields
   depending from preprocessor macros:
      #if (_WIN32_IE >= 0x0300)
      #if (_WIN32_WINNT >= 0x0501)
      #if (_WIN32_WINNT >= 0x0600)
*/

bool CListView::GetItemParam(unsigned index, LPARAM &param) const
{
  LVITEM item;
  item.iItem = (int)index;
  item.iSubItem = 0;
  item.mask = LVIF_PARAM;
  const bool res = GetItem(&item);
  param = item.lParam;
  return res;
}

int CListView::InsertColumn(unsigned columnIndex, LPCTSTR text, int width)
{
  LVCOLUMN ci;
  ci.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;
  ci.pszText = (LPTSTR)(void *)text;
  ci.iSubItem = (int)columnIndex;
  ci.cx = width;
  return InsertColumn(columnIndex, &ci);
}

int CListView::InsertItem(unsigned index, LPCTSTR text)
{
  LVITEM item;
  item.mask = LVIF_TEXT | LVIF_PARAM;
  item.iItem = (int)index;
  item.lParam = (LPARAM)index;
  item.pszText = (LPTSTR)(void *)text;
  item.iSubItem = 0;
  return InsertItem(&item);
}

int CListView::SetSubItem(unsigned index, unsigned subIndex, LPCTSTR text)
{
  LVITEM item;
  item.mask = LVIF_TEXT;
  item.iItem = (int)index;
  item.pszText = (LPTSTR)(void *)text;
  item.iSubItem = (int)subIndex;
  return SetItem(&item);
}

#ifndef _UNICODE

int CListView::InsertColumn(unsigned columnIndex, LPCWSTR text, int width)
{
  LVCOLUMNW ci;
  ci.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;
  ci.pszText = (LPWSTR)(void *)text;
  ci.iSubItem = (int)columnIndex;
  ci.cx = width;
  return InsertColumn(columnIndex, &ci);
}

int CListView::InsertItem(unsigned index, LPCWSTR text)
{
  LVITEMW item;
  item.mask = LVIF_TEXT | LVIF_PARAM;
  item.iItem = (int)index;
  item.lParam = (LPARAM)index;
  item.pszText = (LPWSTR)(void *)text;
  item.iSubItem = 0;
  return InsertItem(&item);
}

int CListView::SetSubItem(unsigned index, unsigned subIndex, LPCWSTR text)
{
  LVITEMW item;
  item.mask = LVIF_TEXT;
  item.iItem = (int)index;
  item.pszText = (LPWSTR)(void *)text;
  item.iSubItem = (int)subIndex;
  return SetItem(&item);
}

#endif

static LRESULT APIENTRY ListViewSubclassProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  CWindow window(hwnd);
  CListView2 *w = (CListView2 *)(window.GetUserDataLongPtr());
  if (w == NULL)
    return 0;
  return w->OnMessage(message, wParam, lParam);
}

LRESULT CListView2::OnMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
  #ifndef _UNICODE
  if (g_IsNT)
    return CallWindowProcW(_origWindowProc, *this, message, wParam, lParam);
  else
  #endif
    return CallWindowProc(_origWindowProc, *this, message, wParam, lParam);
}

void CListView2::SetWindowProc()
{
  SetUserDataLongPtr((LONG_PTR)this);
  #ifndef _UNICODE
  if (g_IsNT)
    _origWindowProc = (WNDPROC)SetLongPtrW(GWLP_WNDPROC, (LONG_PTR)ListViewSubclassProc);
  else
  #endif
    _origWindowProc = (WNDPROC)SetLongPtr(GWLP_WNDPROC, (LONG_PTR)ListViewSubclassProc);
}

/*
LRESULT CListView3::OnMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
  LRESULT res = CListView2::OnMessage(message, wParam, lParam);
  if (message == WM_GETDLGCODE)
  {
    // when user presses RETURN, windows sends default (first) button command to parent dialog.
    // we disable this:
    MSG *msg = (MSG *)lParam;
    WPARAM key = wParam;
    bool change = false;
    if (msg)
    {
      if (msg->message == WM_KEYDOWN && msg->wParam == VK_RETURN)
        change = true;
    }
    else if (wParam == VK_RETURN)
      change = true;
    if (change)
      res |= DLGC_WANTALLKEYS;
  }
  return res;
}
*/
  
}}
