#pragma once
#include "scene/gui/rich_text_label.h"

class EditorLogRichTextLabel : public RichTextLabel {
    GDCLASS(EditorLogRichTextLabel, RichTextLabel);

public:
    void set_bbcode_parser(RichTextLabel *p_label);
    void set_filter_keytext(const String &p_keytext);
    void set_filter_case_sensitive(bool p_case_sensitive);

protected:
    virtual void _highlighting_pass_callback(const String &line_text, RID rid, RID ci_rid, Color highlight_color, Vector2 p_ofs, Vector2 off, float l_ascent, Vector2 l_size) override;
    
private:
    RichTextLabel *bbcode_parser = nullptr;    

    String filter_keytext;
    bool filter_enabled = false;
    bool filter_case_sensitive = false;

    Vector<int> _get_line_search_query_positions(const String &p_line);

    bool _contains_case_sensitive(const String &p_base, const String &p_target);
    int _find_case_sensitive(const String &p_base, const String &p_target);
    int _count_case_sensitive(const String &p_base, const String &p_target);
    String _strip_bbcode_from_line(const String &p_line);

};
