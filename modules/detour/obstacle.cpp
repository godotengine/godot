#include "obstacle.h"
#include "detour.h"
DetourNavigationObstacle::DetourNavigationObstacle() :
		mesh(0),
		id(0),
		radius(5.0f),
		height(5.0f) {}

DetourNavigationObstacle::~DetourNavigationObstacle() {
	if (mesh && id > 0)
		mesh->remove_obstacle(id);
}
void DetourNavigationObstacle::_bind_methods() {}
