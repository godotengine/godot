/* location_param.h */

#ifndef LOCATION_PARAM_H
#define LOCATION_PARAM_H

#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/os/thread_safe.h"

/**
	@author Cagdas Caglak <cagdascaglak@gmail.com>
*/

class LocationParam : public Resource {
	GDCLASS(LocationParam, Resource);

	int interval;
	int max_wait_time;

protected:
	static void _bind_methods();

public:
	void set_interval(int p_interval);
	void set_max_wait_time(int p_max_wait_time);

	int get_interval();
	int get_max_wait_time();

	LocationParam();
};

#endif // LOCATION_PARAM_H
