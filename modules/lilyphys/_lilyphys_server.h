//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS__LILYPHYS_SERVER_H
#define LILYPHYS__LILYPHYS_SERVER_H

#include "core/object.h"
#include "lilyphys_server.h"

class _LilyphysServer : public Object {
    GDCLASS(_LilyphysServer, Object);

    friend class LilyphysServer;
    static _LilyphysServer *singleton;

protected:
    static void _bind_methods();

private:

public:
    static _LilyphysServer *get_singleton();
    _LilyphysServer();
    ~_LilyphysServer();
    String get_pee_storage();

    RID create_physics_body();
};

#endif //LILYPHYS__LILYPHYS_SERVER_H
