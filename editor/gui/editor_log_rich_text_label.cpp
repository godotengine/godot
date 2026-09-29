#include "editor_log_rich_text_label.h"
#include "servers/rendering/rendering_server.h"

void EditorLogRichTextLabel::_highlighting_pass_callback(const String &line_text, RID rid, RID ci_rid, Color highlight_color, Vector2 p_ofs, Vector2 off, float l_ascent, Vector2 l_size) {
    Vector<int> positions = _get_line_search_query_positions(line_text);

	for (int i = 2; i + 1 < positions.size(); i = i + 2) {
		// Start at the first match, so index 2 (a pair consists of elements at index i and index i - 1, so this starts with elements 1 and 2 as a pair).
		// Then increase by two because we only want to read out matches.

        Vector<Vector2> sel = TS->shaped_text_get_selection(rid, positions[i - 1], positions[i]);
        for (int j = 0; j < sel.size(); j++) {
            Rect2 rect = Rect2((sel[j].x + p_ofs.x + off.x), p_ofs.y + off.y - l_ascent, (sel[j].y - sel[j].x), l_size.y);
            RenderingServer::get_singleton()->canvas_item_add_rect(ci_rid, rect, highlight_color);
        }
    }
}

Vector<int> EditorLogRichTextLabel::_get_line_search_query_positions(const String &p_line) {
	int keytext_length = filter_keytext.length();

	String iterator_line = p_line;
	int cursor_position = 0;

	Vector<int> positions; // Array of substring positions. Every pair will be cut into a substring from p_line.
	positions.append(0);

	if (_find_case_sensitive(iterator_line, filter_keytext) == 0) {
		positions.append(0);
		positions.append(0); // This last zero will be replaced in a few lines, which fixes the order in case the first characters are immediately matches.
	}

	// Map which segments of p_line contain the target string. Every uneven pair of ints will be a non-match, and every even pair will be a match.
	while (_contains_case_sensitive(iterator_line, filter_keytext)) {
		int keytext_pos = _find_case_sensitive(iterator_line, filter_keytext);

		if (keytext_pos == 0) {
			int last_pos = positions[positions.size() - 1];
			positions.resize(positions.size() - 1);
			positions.append(last_pos + keytext_length);
		} else {
			positions.append(keytext_pos + cursor_position);
			positions.append(keytext_pos + cursor_position + keytext_length);
		}

		cursor_position += keytext_pos + keytext_length;

		iterator_line = p_line.substr(cursor_position);
	}

	// Imagine p_line "Lullaby" and p_keytext "l".
	// positions will be [0,0,1,2,4].
	// - The pair 0,0 was inserted due to 20 lines up and is considered not a match.
	// - The pair 0,1 ("L") is a match
	// - The pair 1,2 ("u") is not a match
	// - The pair 2,4 ("ll") is a match once again

	// The last pair will always describe a match at this point and in the case p_line does not end with a match, that would cut off p_line after the last match...
	if (positions[positions.size() - 1] != p_line.size() - 1) {
		positions.append(p_line.size() - 1); // ...so we add a final position. In the case of "Lullaby", it'd append 6 so that positions becomes [0,0,1,2,4,6]. That prevents the mistake described 2 lines up.
	}

	return positions;
}

bool EditorLogRichTextLabel::_contains_case_sensitive(const String &p_base, const String &p_contains) {
	if (filter_case_sensitive) {
		return p_base.contains(p_contains);
	} else {
		return p_base.containsn(p_contains);
	}
}

int EditorLogRichTextLabel::_find_case_sensitive(const String &p_base, const String &p_target) {
	if (filter_case_sensitive) {
		return p_base.find(p_target);
	} else {
		return p_base.findn(p_target);
	}
}

int EditorLogRichTextLabel::_count_case_sensitive(const String &p_base, const String &p_target) {
	if (filter_case_sensitive) {
		return p_base.count(p_target);
	} else {
		return p_base.countn(p_target);
	}
}

void EditorLogRichTextLabel::set_bbcode_parser(RichTextLabel *p_label) {
    bbcode_parser = p_label;
}

void EditorLogRichTextLabel::set_filter_keytext(const String &p_keytext) {
    filter_keytext = p_keytext;
    queue_redraw();
}

void EditorLogRichTextLabel::set_filter_case_sensitive(bool p_case_sensitive) {
    filter_case_sensitive = p_case_sensitive;
}

String EditorLogRichTextLabel::_strip_bbcode_from_line(const String &p_line) {
	String result;

	bbcode_parser->clear();
	bbcode_parser->parse_bbcode(p_line);
	result = bbcode_parser->get_parsed_text();
	return result;
}