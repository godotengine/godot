#include "haiku_gl_view.h"

HaikuGLView::HaikuGLView(BRect frame, uint32 type)
   : BGLView(frame, "SampleGLView", B_FOLLOW_ALL_SIDES, 0, type), rotate(0)
{
	width = frame.right-frame.left;
	height = frame.bottom-frame.top;
}

void HaikuGLView::AttachedToWindow(void)
{
	LockGL();
	BGLView::AttachedToWindow();
	UnlockGL();
	MakeFocus();
}

void HaikuGLView::FrameResized(float newWidth, float newHeight) 
{
}

void HaikuGLView::gDraw(float rotation)
{
}

void HaikuGLView::gReshape(int width, int height)
{
}

void HaikuGLView::Render(void)
{
	LockGL();
	SwapBuffers();
	UnlockGL();
}

void HaikuGLView::MessageReceived(BMessage * msg)
{
	switch (msg->what) {
	case 'rdrw':
		Render();
		/* Rotate a bit more */
		rotate++;
		break;

	default:	
		BGLView::MessageReceived(msg);
	}
}

void HaikuGLView::KeyDown(const char *bytes, int32 numBytes)
{

}
