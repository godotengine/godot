// MyWindowsNew.h

#ifndef ZIP7_INC_MY_WINDOWS_NEW_H
#define ZIP7_INC_MY_WINDOWS_NEW_H

#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__MINGW32_VERSION)
#include <shobjidl.h>

#if defined(__MINGW32_VERSION) && !defined(__ITaskbarList3_INTERFACE_DEFINED__)
// for old mingw
extern "C" {
DEFINE_GUID(IID_ITaskbarList3, 0xEA1AFB91, 0x9E28, 0x4B86, 0x90, 0xE9, 0x9E, 0x9F, 0x8A, 0x5E, 0xEF, 0xAF);
DEFINE_GUID(CLSID_TaskbarList, 0x56fdf344, 0xfd6d, 0x11d0, 0x95,0x8a, 0x00,0x60,0x97,0xc9,0xa0,0x90);
}
#endif

#else // is not __MINGW*

#ifndef Z7_OLD_WIN_SDK
#include <ShObjIdl.h>
#else

#ifndef HIMAGELIST
struct _IMAGELIST;
typedef struct _IMAGELIST* HIMAGELIST;
#endif

#ifndef __ITaskbarList_INTERFACE_DEFINED__
#define __ITaskbarList_INTERFACE_DEFINED__
DEFINE_GUID(IID_ITaskbarList, 0x56FDF342, 0xFD6D, 0x11d0, 0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90);
struct ITaskbarList: public IUnknown
{
  STDMETHOD(HrInit)(void) = 0;
  STDMETHOD(AddTab)(HWND hwnd) = 0;
  STDMETHOD(DeleteTab)(HWND hwnd) = 0;
  STDMETHOD(ActivateTab)(HWND hwnd) = 0;
  STDMETHOD(SetActiveAlt)(HWND hwnd) = 0;
};
#endif // __ITaskbarList_INTERFACE_DEFINED__

#ifndef __ITaskbarList2_INTERFACE_DEFINED__
#define __ITaskbarList2_INTERFACE_DEFINED__
DEFINE_GUID(IID_ITaskbarList2, 0x602D4995, 0xB13A, 0x429b, 0xA6, 0x6E, 0x19, 0x35, 0xE4, 0x4F, 0x43, 0x17);
struct ITaskbarList2: public ITaskbarList
{
  STDMETHOD(MarkFullscreenWindow)(HWND hwnd, BOOL fFullscreen) = 0;
};
#endif // __ITaskbarList2_INTERFACE_DEFINED__

#endif // Z7_OLD_WIN_SDK


#ifndef __ITaskbarList3_INTERFACE_DEFINED__
#define __ITaskbarList3_INTERFACE_DEFINED__

typedef enum THUMBBUTTONFLAGS
{
  THBF_ENABLED = 0,
  THBF_DISABLED = 0x1,
  THBF_DISMISSONCLICK = 0x2,
  THBF_NOBACKGROUND = 0x4,
  THBF_HIDDEN = 0x8,
  THBF_NONINTERACTIVE = 0x10
} THUMBBUTTONFLAGS;

typedef enum THUMBBUTTONMASK
{
  THB_BITMAP = 0x1,
  THB_ICON = 0x2,
  THB_TOOLTIP = 0x4,
  THB_FLAGS = 0x8
} THUMBBUTTONMASK;

// #include <pshpack8.h>

typedef struct THUMBBUTTON
{
  THUMBBUTTONMASK dwMask;
  UINT iId;
  UINT iBitmap;
  HICON hIcon;
  WCHAR szTip[260];
  THUMBBUTTONFLAGS dwFlags;
} THUMBBUTTON;

typedef struct THUMBBUTTON *LPTHUMBBUTTON;

typedef enum TBPFLAG
{
  TBPF_NOPROGRESS = 0,
  TBPF_INDETERMINATE = 0x1,
  TBPF_NORMAL = 0x2,
  TBPF_ERROR = 0x4,
  TBPF_PAUSED = 0x8
} TBPFLAG;

DEFINE_GUID(IID_ITaskbarList3, 0xEA1AFB91, 0x9E28, 0x4B86, 0x90, 0xE9, 0x9E, 0x9F, 0x8A, 0x5E, 0xEF, 0xAF);

struct ITaskbarList3: public ITaskbarList2
{
  STDMETHOD(SetProgressValue)(HWND hwnd, ULONGLONG ullCompleted, ULONGLONG ullTotal) = 0;
  STDMETHOD(SetProgressState)(HWND hwnd, TBPFLAG tbpFlags) = 0;
  STDMETHOD(RegisterTab)(HWND hwndTab, HWND hwndMDI) = 0;
  STDMETHOD(UnregisterTab)(HWND hwndTab) = 0;
  STDMETHOD(SetTabOrder)(HWND hwndTab, HWND hwndInsertBefore) = 0;
  STDMETHOD(SetTabActive)(HWND hwndTab, HWND hwndMDI, DWORD dwReserved) = 0;
  STDMETHOD(ThumbBarAddButtons)(HWND hwnd, UINT cButtons, LPTHUMBBUTTON pButton) = 0;
  STDMETHOD(ThumbBarUpdateButtons)(HWND hwnd, UINT cButtons, LPTHUMBBUTTON pButton) = 0;
  STDMETHOD(ThumbBarSetImageList)(HWND hwnd, HIMAGELIST himl) = 0;
  STDMETHOD(SetOverlayIcon)(HWND hwnd, HICON hIcon, LPCWSTR pszDescription) = 0;
  STDMETHOD(SetThumbnailTooltip)(HWND hwnd, LPCWSTR pszTip) = 0;
  STDMETHOD(SetThumbnailClip)(HWND hwnd, RECT *prcClip) = 0;
};

#endif // __ITaskbarList3_INTERFACE_DEFINED__

#endif // __MINGW*

#endif
