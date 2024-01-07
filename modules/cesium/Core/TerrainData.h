//
// Created by Harris.Lu on 2024/1/7.
//

#ifndef GODOT_TERRAINDATA_H
#define GODOT_TERRAINDATA_H

namespace Cesium {

class TerrainData {

public:
	virtual bool isChildAvailable(int thisX, int thisY, int childX ) = 0;
};

} //namespace Cesium

#endif //GODOT_TERRAINDATA_H
