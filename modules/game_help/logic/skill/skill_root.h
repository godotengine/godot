#pragma once

#include "scene/main/node.h"
#include "scene/3d/node_3d.h"

class SkillTimelineCompoent : public RefCounted
{
    GDCLASS(SkillTimelineCompoent,RefCounted);
public:
    SkillTimelineCompoent();
    ~SkillTimelineCompoent();

    NodePath compoent_path;
    // 开始时间
    float start_time;
    // 持续时间
    float duration = 0;
    bool enabled = false;
};
class SkillTimeline : public RefCounted
{
    GDCLASS(SkillTimeline, RefCounted);
public:
    SkillTimeline();
    ~SkillTimeline();
    StringName name;
    LocalVector<SkillTimelineCompoent> timelines;
    float curr_time;
    bool enabled = false;

};


class SkillRoot : public Node3D
{
    GDCLASS(SkillRoot, Node3D);
public:
    SkillRoot();
    ~SkillRoot();
    class CharacterBodyMain* owner = nullptr;
    StringName default_enable_timeline;
    LocalVector<Ref<SkillTimeline>> timelines;

    List<Ref<SkillAutoDelete>> auto_deletes;
};