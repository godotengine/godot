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
	Point2i last_mouse_position;
	bool last_mouse_pos_valid;
	uint32 last_buttons_state;
	uint32 last_key_modifier_state;
	int last_button_mask;

	MainLoop* main_loop;
	InputDefault* input;
	HaikuGLView* view;
	BMessageRunner* update_runner;

	void DispatchMouseButton(BMessage* message);
	void DispatchMouseMoved(BMessage* message);
	void DispatchMouseWheelChanged(BMessage* message);
	inline InputModifierState GetKeyModifierState(uint32 p_state);
	inline int GetMouseButtonState(uint32 p_state);

public:
	HaikuDirectWindow(BRect p_frame);
	~HaikuDirectWindow();

	void SetHaikuGLView(HaikuGLView* p_view);
	void StartMessageRunner();
	void StopMessageRunner();
	void SetInput(InputDefault* p_input);
	void SetMainLoop(MainLoop* p_main_loop);
	virtual bool QuitRequested();
	virtual void DirectConnected(direct_buffer_info* info);
	virtual void MessageReceived(BMessage* message);
	virtual void DispatchMessage(BMessage* message, BHandler* handler);

	inline Point2i GetLastMousePosition() { return last_mouse_position; };
	inline int GetLastButtonMask() { return last_button_mask; };
};

#endif
