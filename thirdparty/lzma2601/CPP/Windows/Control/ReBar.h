// Windows/Control/ReBar.h
  
#ifndef ZIP7_INC_WINDOWS_CONTROL_REBAR_H
#define ZIP7_INC_WINDOWS_CONTROL_REBAR_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CReBar: public NWindows::CWindow
{
public:
  bool SetBarInfo(LPREBARINFO barInfo)
    { return LRESULTToBool(SendMsg(RB_SETBARINFO, 0, (LPARAM)barInfo)); }
  bool InsertBand(int index, LPREBARBANDINFO bandInfo)
    { return LRESULTToBool(SendMsg(RB_INSERTBAND, MY_int_TO_WPARAM(index), (LPARAM)bandInfo)); }
  bool SetBandInfo(unsigned index, LPREBARBANDINFO bandInfo)
    { return LRESULTToBool(SendMsg(RB_SETBANDINFO, index, (LPARAM)bandInfo)); }
  void MaximizeBand(unsigned index, bool ideal)
    { SendMsg(RB_MAXIMIZEBAND, index, BoolToBOOL(ideal)); }
  bool SizeToRect(LPRECT rect)
    { return LRESULTToBool(SendMsg(RB_SIZETORECT, 0, (LPARAM)rect)); }
  UINT GetHeight()
    { return (UINT)SendMsg(RB_GETBARHEIGHT); }
  UINT GetBandCount()
    { return (UINT)SendMsg(RB_GETBANDCOUNT); }
  bool DeleteBand(UINT index)
    { return LRESULTToBool(SendMsg(RB_DELETEBAND, index)); }
};

}}

#endif
