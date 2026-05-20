// DialogSize.h

#ifndef ZIP7_INC_DIALOG_SIZE_H
#define ZIP7_INC_DIALOG_SIZE_H

#include "../../../Windows/Control/Dialog.h"

#ifdef UNDER_CE
#define BIG_DIALOG_SIZE(x, y) bool isBig = NWindows::NControl::IsDialogSizeOK(x, y);
#define SIZED_DIALOG(big) (isBig ? big : big ## _2)
#else
#define BIG_DIALOG_SIZE(x, y)
#define SIZED_DIALOG(big) big
#endif

#endif
