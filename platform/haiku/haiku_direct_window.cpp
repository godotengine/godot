#include "haiku_direct_window.h"

HaikuDirectWindow::HaikuDirectWindow(BRect p_frame)
   : BDirectWindow(p_frame, "Godot", B_TITLED_WINDOW, 0)
{
	// TODO: formatting
   float minWidth = 0.0f; 
   float maxWidth = 0.0f; 
   float minHeight = 0.0f; 
   float maxHeight = 0.0f; 
 	
   GetSizeLimits(&minWidth, &maxWidth, &minHeight, &maxHeight); 
   SetSizeLimits(50.0f, maxWidth, 50.0f, maxHeight);
}


HaikuDirectWindow::~HaikuDirectWindow()
{
	delete update_runner;
}

void HaikuDirectWindow::SetHaikuGLView(HaikuGLView* p_view) {
	view = p_view;
}

void HaikuDirectWindow::InitMessageRunner() {
	update_runner = new BMessageRunner(BMessenger(view),
		new BMessage(REDRAW_MSG), 1000000/60 /* 60 fps */);
}


bool HaikuDirectWindow::QuitRequested()
{
	view->EnableDirectMode(false);
	be_app->PostMessage(B_QUIT_REQUESTED);
	return true;
}


void HaikuDirectWindow::DirectConnected(direct_buffer_info *info)
{
	view->DirectConnected(info);	
	view->EnableDirectMode(true);
}

