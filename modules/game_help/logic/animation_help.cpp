#include "animation_help.h"
AnimationHelp* AnimationHelp::singleton = nullptr;
Ref<Animation>  AnimationHelp::nullAnimation = Ref<Animation>(memnew(Animation));