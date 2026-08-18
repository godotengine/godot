#pragma once
#include "scene/gui/rich_text_label.h"

class EditorLogRichTextLabel : public RichTextLabel {
    GDCLASS(EditorLogRichTextLabel, RichTextLabel);

    public:
        void set_filter_keytext(const String &p_keytext);

    protected:
        virtual void _highlighting_pass_callback(Line *l, RID rid, RID ci_rid, Color highlight_color, Vector2 p_ofs, Vector2 off, float l_ascent, Vector2 l_size) override;
    
    private:
        String filter_keytext;
        bool filter_enabled = false;
};