#ifndef HAIKU_DIRECT_WINDOW_H
#define HAIKU_DIRECT_WINDOW_H

#include <kernel/image.h> // needed for image_id
#include <DirectWindow.h>

#include "haiku_gl_view.h"

#define REDRAW_MSG 'rdrw'

class HaikuDirectWindow : public BDirectWindow 
{
public:
	HaikuDirectWindow(BRect p_frame);
	~HaikuDirectWindow();

	void SetHaikuGLView(HaikuGLView* p_view);
	void InitMessageRunner();
	virtual bool QuitRequested();
	virtual void DirectConnected(direct_buffer_info *info);

private:
	HaikuGLView* view;
	BMessageRunner* update_runner;
};

#endif
