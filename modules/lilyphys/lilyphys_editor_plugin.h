//
// Created by amara on 26/11/2021.
//

#ifndef LILYPHYS_LILYPHYS_EDITOR_PLUGIN_H
#define LILYPHYS_LILYPHYS_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"

class LilyphysEditorPlugin : public EditorPlugin {
GDCLASS(LilyphysEditorPlugin, EditorPlugin);
public:
    LilyphysEditorPlugin(EditorNode *p_editor);
};


#endif //LILYPHYS_LILYPHYS_EDITOR_PLUGIN_H
