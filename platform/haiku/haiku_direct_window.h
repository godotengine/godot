#ifndef HAIKU_DIRECT_WINDOW_H
#define HAIKU_DIRECT_WINDOW_H

#include <kernel/image.h> // needed for image_id
#include <DirectWindow.h>

#include "core/os/os.h"
#include "main/input_default.h"

#include "haiku_gl_view.h"

#define REDRAW_MSG 'rdrw'
#define LOCKGL_MSG 'glck'
#define UNLOCKGL_MSG 'ulck'

class HaikuDirectWindow : public BDirectWindow
{
private:
	unsigned int event_id;
	Point2i last_mouse_position;
	bool last_mouse_pos_valid;
	uint32 last_buttons_state;
	uint32 last_key_modifier_state;
	int last_button_mask;
	OS::VideoMode* current_video_mode;

	MainLoop* main_loop;
	InputDefault* input;
	HaikuGLView* view;
	BMessageRunner* update_runner;

	void HandleMouseButton(BMessage* message);
	void HandleMouseMoved(BMessage* message);
	void HandleMouseWheelChanged(BMessage* message);
	void HandleWindowResized(BMessage* message);
	void HandleKeyboardEvent(BMessage* message);
	void HandleKeyboardModifierEvent(BMessage* message);
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
	inline void SetVideoMode(OS::VideoMode* video_mode) { current_video_mode = video_mode; };
	virtual bool QuitRequested();
	virtual void DirectConnected(direct_buffer_info* info);
	virtual void MessageReceived(BMessage* message);
	virtual void DispatchMessage(BMessage* message, BHandler* handler);

	inline Point2i GetLastMousePosition() { return last_mouse_position; };
	inline int GetLastButtonMask() { return last_button_mask; };
};

#endif
