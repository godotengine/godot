//
// Created by Harris.Lu on 2024/1/7.
//

#ifndef GODOT_GLOBE_H
#define GODOT_GLOBE_H

#include "scene/main/node.h"

namespace Cesium {

class Globe : public Node {
	GDCLASS(Globe, Node);

protected:
	static void _bind_methods();
	void _notification(int p_what);

	Globe() {}
};

} //namespace Cesium

#endif //GODOT_GLOBE_H
