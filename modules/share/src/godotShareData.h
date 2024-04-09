#ifndef GODOT_SHARE_DATA_H
#define GODOT_SHARE_DATA_H

#include <version_generated.gen.h>
#include "core/object/ref_counted.h"

class GodotShareData : public RefCounted {
    GDCLASS(GodotShareData, RefCounted);
    

protected:
    static void _bind_methods();
    
    GodotShareData* instance;
    
public:

    void shareText(const String &title, const String &subject, const String &text);
    void shareImage(const String &path, const String &title, const String &subject, const String &text);

    GodotShareData();
    ~GodotShareData();
};

#endif
