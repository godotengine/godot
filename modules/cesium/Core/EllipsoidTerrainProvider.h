//
// Created by Harris.Lu on 2024/1/7.
//

#ifndef GODOT_ELLIPSOIDTERRAINPROVIDER_H
#define GODOT_ELLIPSOIDTERRAINPROVIDER_H
#include "TerrainProvider.h"

namespace Cesium {

class EllipsoidTerrainProvider : public TerrainProvider {

public:
	bool getTileDataAvailable(int x, int y, int level) override;

};

} //namespace Cesium

#endif //GODOT_ELLIPSOIDTERRAINPROVIDER_H
