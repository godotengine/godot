/* location_result.h */

#ifndef LOCATION_RESULT_H
#define LOCATION_RESULT_H

#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/os/thread_safe.h"

class LocationResult : public Object {
    GDCLASS(LocationResult, Object);

    /**
     *
     * public Reference {

	GDCLASS(LocationResult, Reference);
	OBJ_CATEGORY("Resources");
	RES_BASE_EXTENSION("res"); */

protected:
    static void _bind_methods();

public:
    real_t longitude;
    real_t latitude;
    real_t horizontal_accuracy;
    real_t vertical_accuracy;
    real_t altitude;
    real_t speed;
    uint64_t time;

    real_t get_longitude() const;
    real_t get_latitude() const;
    real_t get_horizontal_accuracy() const;
    real_t get_vertical_accuracy() const;
    real_t get_altitude() const;
    real_t get_speed() const;
    uint64_t get_time() const;

    LocationResult();

};

#endif // LOCATION_RESULT_H
