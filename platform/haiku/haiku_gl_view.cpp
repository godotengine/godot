#include "main/main.h"
#include "haiku_gl_view.h"

HaikuGLView::HaikuGLView(BRect frame, uint32 type)
   : BGLView(frame, "SampleGLView", B_FOLLOW_ALL_SIDES, 0, type)
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

void HaikuGLView::MessageReceived(BMessage* msg)
{
	// TODO: remove if not needed
	switch (msg->what) {
		default:	
			BGLView::MessageReceived(msg);
	}
}

void HaikuGLView::MouseMoved (BPoint where, uint32 code, const BMessage *dragMessage) {
	ERR_PRINT("MouseMoved()");
}
