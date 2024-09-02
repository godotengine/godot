#pragma once
#include "core/variant/dictionary.h"
#include "scene/resources/animation.h"

class UnityAnimationImport
{
public:
    static void ImportAnimation(Dictionary anima_dict, int mirror, Ref<Animation> anim);
};