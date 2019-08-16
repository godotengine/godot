#ifndef ANIMATION_MOTION_MATCH_EDITOR_H
#define ANIMATION_MOTION_MATCH_EDITOR_H

#ifdef TOOLS_ENABLED

#include "editor/plugins/animation_tree_editor_plugin.h"
#include "modules/motionmatch/animation_node_motion_match.h"
#include "core/reference.h"
#include "frame_model.h"

class AnimationNodeMotionMatchEditor : public AnimationTreeNodeEditorPlugin {

  GDCLASS(AnimationNodeMotionMatchEditor, AnimationTreeNodeEditorPlugin)

  Ref<AnimationNodeMotionMatch> motion_match;

  AcceptDialog *match_tracks_dialog;
  Tree *match_tracks;

  Button *edit_match_tracks;
  Button *update_tracks;
  Button *clear_tree;
  SpinBox *snap_x;

  Skeleton *skeleton;

  bool updating;
  PoolVector<frame_model *> *keys = new PoolVector<frame_model *>();

  void _match_tracks_edited();

  void _edit_match_tracks();
  void _update_match_tracks();
  void _update_tracks();
  void _clear_tree();

protected:
  static void _bind_methods();

public:
  virtual bool can_edit(const Ref<AnimationNode> &p_node);
  virtual void edit(const Ref<AnimationNode> &p_node);

  int fill_tracks(AnimationPlayer *player, Animation *anim, NodePath &root);

  AnimationNodeMotionMatchEditor();
};
#endif // ANIMATION_MOTION_MATCH_EDITOR_H
#endif
