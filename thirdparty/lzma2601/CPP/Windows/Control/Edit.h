// Windows/Control/Edit.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_EDIT_H
#define ZIP7_INC_WINDOWS_CONTROL_EDIT_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CEdit: public CWindow
{
public:
  void SetPasswordChar(WPARAM c) { SendMsg(EM_SETPASSWORDCHAR, c); }
};

}}

#endif
