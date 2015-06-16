#ifndef HAIKU_DIRECT_WINDOW_H
#define HAIKU_DIRECT_WINDOW_H

#include <kernel/image.h> // needed for image_id
#include <DirectWindow.h>

#include "os/input.h"
#include "haiku_gl_view.h"

#define REDRAW_MSG 'rdrw'

class HaikuDirectWindow : public BDirectWindow 
{
private:
	unsigned int event_id;
	Point2i last_mouse_pos;
	bool last_mouse_pos_valid;
	uint32 last_buttons_state;

	InputDefault* input;
	HaikuGLView* view;
	BMessageRunner* update_runner;

	void DispatchMouseButton(BMessage* message);
	void DispatchMouseMoved(BMessage* message);

public:
	HaikuDirectWindow(BRect p_frame);
	~HaikuDirectWindow();

	void SetHaikuGLView(HaikuGLView* p_view);
	void StartMessageRunner();
	void StopMessageRunner();
	void SetInput(InputDefault* p_input);
	virtual bool QuitRequested();
	virtual void DirectConnected(direct_buffer_info* info);
	virtual void MessageReceived(BMessage* message);
	virtual void DispatchMessage(BMessage* message, BHandler* handler);
};

#endif
