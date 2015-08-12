#ifndef HAIKU_GL_VIEW_H
#define HAIKU_GL_VIEW_H

#include <kernel/image.h> // needed for image_id
#include <GLView.h>

class HaikuGLView : public BGLView
{
public:
	HaikuGLView(BRect frame, uint32 type);
	virtual void AttachedToWindow(void);
	virtual void Draw(BRect updateRect);
};

#endif
