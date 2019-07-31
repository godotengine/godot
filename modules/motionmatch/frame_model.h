#ifndef FRAME_MODEL_H
#define FRAME_MODEL_H

#include "scene\main\node.h"

struct frame_model {

  PoolVector<PoolRealArray> *bone_data = new PoolVector<PoolRealArray>();
  PoolRealArray *traj = new PoolRealArray();
  float time = 0.0f;
  int anim_num = 0;
};

#endif
