#include "editor_log_rich_text_label.h"
#include "servers/rendering/rendering_server.h"

void EditorLogRichTextLabel::_highlighting_pass_callback(Line *l, RID rid, RID ci_rid, Color highlight_color, Vector2 p_ofs, Vector2 off, float l_ascent, Vector2 l_size) {
    Vector<Vector2> sel = TS->shaped_text_get_selection(rid, 0, 1);
    for (int i = 0; i < sel.size(); i++) {
		Rect2 rect = Rect2(sel[i].x + p_ofs.x + off.x, p_ofs.y + off.y - l_ascent, sel[i].y - sel[i].x, l_size.y); // Note: use "off" not "off_step", selection is relative to the line start.
		RenderingServer::get_singleton()->canvas_item_add_rect(ci_rid, rect, highlight_color);
    }
}

void EditorLogRichTextLabel::set_filter_keytext(const String &p_keytext) {
    filter_keytext = p_keytext;
    queue_redraw();
}