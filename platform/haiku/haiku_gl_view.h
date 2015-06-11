#ifndef HAIKU_GL_VIEW_H
#define HAIKU_GL_VIEW_H

#include <kernel/image.h> // needed for image_id
#include <GLView.h>

class HaikuGLView : public BGLView
{
public:
   HaikuGLView(BRect frame, uint32 type);
   virtual void   AttachedToWindow(void);
   virtual void   FrameResized(float newWidth, float newHeight);
   virtual void   MessageReceived(BMessage * msg);
   virtual void   KeyDown(const char* bytes, int32 numBytes);
   
   void         Render(void);
   
private:
   void         gDraw(float rotation = 0);
   void         gReshape(int width, int height);
         
   float        width;
   float        height;
   float		rotate;
};

#endif
