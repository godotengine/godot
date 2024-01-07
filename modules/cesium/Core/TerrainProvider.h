//
// Created by Harris.Lu on 2024/1/7.
//

#ifndef GODOT_TERRAINPROVIDER_H
#define GODOT_TERRAINPROVIDER_H

namespace Cesium {

class TerrainProvider {

public:
	virtual bool getTileDataAvailable(int x, int y, int level);
};

} //namespace Cesium

#endif //GODOT_TERRAINPROVIDER_H
