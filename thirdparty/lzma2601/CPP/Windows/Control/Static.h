// Windows/Control/Static.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_STATIC_H
#define ZIP7_INC_WINDOWS_CONTROL_STATIC_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CStatic: public CWindow
{
public:
  HANDLE SetImage(WPARAM imageType, HANDLE handle) { return (HANDLE)SendMsg(STM_SETIMAGE, imageType, (LPARAM)handle); }
  HANDLE GetImage(WPARAM imageType) { return (HANDLE)SendMsg(STM_GETIMAGE, imageType, 0); }

  #ifdef UNDER_CE
  HICON SetIcon(HICON icon) { return (HICON)SetImage(IMAGE_ICON, icon); }
  HICON GetIcon() { return (HICON)GetImage(IMAGE_ICON); }
  #else
  HICON SetIcon(HICON icon) { return (HICON)SendMsg(STM_SETICON, (WPARAM)icon, 0); }
  HICON GetIcon() { return (HICON)SendMsg(STM_GETICON, 0, 0); }
  #endif
};

}}

#endif
