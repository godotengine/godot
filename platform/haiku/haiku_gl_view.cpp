#include "main/main.h"
#include "haiku_gl_view.h"

HaikuGLView::HaikuGLView(BRect frame, uint32 type)
   : BGLView(frame, "GodotGLView", B_FOLLOW_ALL_SIDES, 0, type)
{
}

void HaikuGLView::AttachedToWindow(void) {
	LockGL();
	BGLView::AttachedToWindow();
	UnlockGL();
	MakeFocus();
}

void HaikuGLView::Draw(BRect updateRect) {
	Main::force_redraw();
}
