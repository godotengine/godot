#ifndef HAIKU_APPLICATION_H
#define HAIKU_APPLICATION_H

#include <kernel/image.h> // needed for image_id
#include <Application.h>

class HaikuApplication : public BApplication
{
public:
   HaikuApplication();
};

#endif
