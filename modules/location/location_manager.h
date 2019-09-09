/* location_manager.h */

#ifndef LOCATION_MANAGER_H
#define LOCATION_MANAGER_H

#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/os/thread_safe.h"
#include "location_result.h"
#include "location_param.h"

class LocationManager : public Object {
    GDCLASS(LocationManager, Object);

    LocationResult *location_result;
    static LocationManager *singleton;

protected:
    static void _bind_methods();

public:
    static LocationManager *get_singleton();

    void _send_location_data(OS::Location location);
    void request_location_updates(const Ref<LocationParam> &p_location_param);
    void stop_request_location();
    LocationManager();
    ~LocationManager();

};

#endif // LOCATION_MANAGER_H
