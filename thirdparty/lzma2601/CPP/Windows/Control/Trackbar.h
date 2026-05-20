// Windows/Control/Trackbar.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_TRACKBAR_H
#define ZIP7_INC_WINDOWS_CONTROL_TRACKBAR_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CTrackbar: public CWindow
{
public:
  void SetRange(int minimum, int maximum, bool redraw = true)
    { SendMsg(TBM_SETRANGE, BoolToBOOL(redraw), MAKELONG(minimum, maximum)); }
  void SetPos(int pos, bool redraw = true)
    { SendMsg(TBM_SETPOS, BoolToBOOL(redraw), pos); }
  void SetTicFreq(int freq)
    { SendMsg(TBM_SETTICFREQ, freq); }
  
  int GetPos()
    { return (int)SendMsg(TBM_GETPOS); }
};

}}

#endif
