// Windows/Control/PropertyPage.cpp

#include "StdAfx.h"

#ifndef _UNICODE
#include "../../Common/StringConvert.h"
#endif

#include "PropertyPage.h"

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
APIENTRY MyProperyPageProcedure(HWND dialogHWND, UINT message, WPARAM wParam, LPARAM lParam)
{
  CWindow tempDialog(dialogHWND);
  if (message == WM_INITDIALOG)
    tempDialog.SetUserDataLongPtr(((PROPSHEETPAGE *)lParam)->lParam);
  CDialog *dialog = (CDialog *)(tempDialog.GetUserDataLongPtr());
  if (dialog == NULL)
    return FALSE;
  if (message == WM_INITDIALOG)
    dialog->Attach(dialogHWND);
  try { return BoolToBOOL(dialog->OnMessage(message, wParam, lParam)); }
  catch(...) { return TRUE; }
}

bool CPropertyPage::OnNotify(UINT /* controlID */, LPNMHDR lParam)
{
  switch (lParam->code)
  {
    case PSN_APPLY: SetMsgResult(OnApply2(LPPSHNOTIFY(lParam))); break;
    case PSN_KILLACTIVE: SetMsgResult(BoolToBOOL(OnKillActive2(LPPSHNOTIFY(lParam)))); break;
    case PSN_SETACTIVE: SetMsgResult(OnSetActive2(LPPSHNOTIFY(lParam))); break;
    case PSN_RESET: OnReset2(LPPSHNOTIFY(lParam)); break;
    case PSN_HELP: OnNotifyHelp2(LPPSHNOTIFY(lParam)); break;
    default: return false;
  }
  return true;
}

/*
PROPSHEETPAGE fields depend from
#if   (_WIN32_WINNT >= 0x0600)
#elif (_WIN32_WINNT >= 0x0501)
#elif (_WIN32_IE >= 0x0400)
PROPSHEETHEADER fields depend from
#if (_WIN32_IE >= 0x0400)
*/
#if defined(PROPSHEETPAGEA_V1_SIZE) && !defined(Z7_OLD_WIN_SDK)
#ifndef _UNICODE
#define my_compatib_PROPSHEETPAGEA PROPSHEETPAGEA_V1
#endif
#define my_compatib_PROPSHEETPAGEW PROPSHEETPAGEW_V1
#else
// for old mingw:
#ifndef _UNICODE
#define my_compatib_PROPSHEETPAGEA PROPSHEETPAGEA
#endif
#define my_compatib_PROPSHEETPAGEW PROPSHEETPAGEW
#endif

INT_PTR MyPropertySheet(const CObjectVector<CPageInfo> &pagesInfo, HWND hwndParent, const UString &title)
{
  unsigned i;
  #ifndef _UNICODE
  AStringVector titles;
  for (i = 0; i < pagesInfo.Size(); i++)
    titles.Add(GetSystemString(pagesInfo[i].Title));
  CRecordVector<my_compatib_PROPSHEETPAGEA> pagesA;
  #endif
  CRecordVector<my_compatib_PROPSHEETPAGEW> pagesW;

  for (i = 0; i < pagesInfo.Size(); i++)
  {
    const CPageInfo &pageInfo = pagesInfo[i];
    #ifndef _UNICODE
    {
      my_compatib_PROPSHEETPAGEA page;
      memset(&page, 0, sizeof(page));
      page.dwSize = sizeof(page);
      page.dwFlags = PSP_HASHELP;
      page.hInstance = g_hInstance;
      page.pszTemplate = MAKEINTRESOURCEA(pageInfo.ID);
      // page.pszIcon = NULL;
      page.pfnDlgProc = NWindows::NControl::MyProperyPageProcedure;
      
      if (!titles[i].IsEmpty())
      {
        page.pszTitle = titles[i];
        page.dwFlags |= PSP_USETITLE;
      }
      // else page.pszTitle = NULL;
      page.lParam = (LPARAM)pageInfo.Page;
      // page.pfnCallback = NULL;
      pagesA.Add(page);
    }
    #endif
    {
      my_compatib_PROPSHEETPAGEW page;
      memset(&page, 0, sizeof(page));
      page.dwSize = sizeof(page);
      page.dwFlags = PSP_HASHELP;
      page.hInstance = g_hInstance;
      page.pszTemplate = MAKEINTRESOURCEW(pageInfo.ID);
      // page.pszIcon = NULL;
      page.pfnDlgProc = NWindows::NControl::MyProperyPageProcedure;
      
      if (!pageInfo.Title.IsEmpty())
      {
        page.pszTitle = pageInfo.Title;
        page.dwFlags |= PSP_USETITLE;
      }
      // else page.pszTitle = NULL;
      page.lParam = (LPARAM)pageInfo.Page;
      // page.pfnCallback = NULL;
      pagesW.Add(page);
    }
  }

  #ifndef _UNICODE
  if (!g_IsNT)
  {
    PROPSHEETHEADERA sheet;
    sheet.dwSize = sizeof(sheet);
    sheet.dwFlags = PSH_PROPSHEETPAGE;
    sheet.hwndParent = hwndParent;
    sheet.hInstance = g_hInstance;
    AString titleA (GetSystemString(title));
    sheet.pszCaption = titleA;
    sheet.nPages = pagesA.Size();
    sheet.nStartPage = 0;
    sheet.ppsp = (LPCPROPSHEETPAGEA)(const void *)pagesA.ConstData();
    sheet.pfnCallback = NULL;
    return ::PropertySheetA(&sheet);
  }
  else
  #endif
  {
    PROPSHEETHEADERW sheet;
    sheet.dwSize = sizeof(sheet);
    sheet.dwFlags = PSH_PROPSHEETPAGE;
    sheet.hwndParent = hwndParent;
    sheet.hInstance = g_hInstance;
    sheet.pszCaption = title;
    sheet.nPages = pagesW.Size();
    sheet.nStartPage = 0;
    sheet.ppsp = (LPCPROPSHEETPAGEW)(const void *)pagesW.ConstData();
    sheet.pfnCallback = NULL;
    return ::PropertySheetW(&sheet);
  }
}

}}
