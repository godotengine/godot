#ifndef IOS_H
#define IOS_H

#include "core/object.h"

class iOS : public Object {

	OBJ_TYPE(iOS, Object);

	static void _bind_methods();

public:

	String get_rate_url(int p_app_id) const;

	iOS();

};

#endif
