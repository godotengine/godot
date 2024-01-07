//
// Created by Harris.Lu on 2024/1/7.
//

#ifndef GODOT_HEIGHTMAPTERRAINDATA_H
#define GODOT_HEIGHTMAPTERRAINDATA_H

#include "TerrainData.h"
namespace Cesium {

class HeightmapTerrainData : public TerrainData {

public:
	bool isChildAvailable(int thisX, int thisY, int childX ) override;
};

} //namespace Cesium

#endif //GODOT_HEIGHTMAPTERRAINDATA_H
