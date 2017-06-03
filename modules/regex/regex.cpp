/*************************************************************************/
/*  regex.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "regex.h"
#include <wchar.h>
#include <wctype.h>

static int RegEx_hex2int(const CharType c) {
	if ('0' <= c && c <= '9')
		return int(c - '0');
	else if ('a' <= c && c <= 'f')
		return int(c - 'a') + 10;
	else if ('A' <= c && c <= 'F')
		return int(c - 'A') + 10;
	return -1;
}

struct RegExSearch {

	Ref<RegExMatch> match;
	const CharType *str;
	int end;
	int eof;

	// For standard quantifier behaviour, test_parent is used to check the
	// rest of the pattern. If the pattern matches, to prevent the parent
	// from testing again, the complete flag is used as a shortcut out.
	bool complete;

	// With lookahead, the position needs to rewind to its starting position
	// when test_parent is used. Due to functional programming, this state
	// has to be kept as a parameter.
	Vector<int> lookahead_pos;

	CharType at(int p_pos) {
		return str[p_pos];
	}

	RegExSearch(Ref<RegExMatch> &p_match, int p_end, int p_lookahead)
		: match(p_match) {

		str = p_match->string.c_str();
		end = p_end;
		eof = p_match->string.length();
		complete = false;
		lookahead_pos.resize(p_lookahead);
	}
};

struct RegExNode {

	RegExNode *next;
	RegExNode *previous;
	RegExNode *parent;
	bool quantifiable;
	int length;

	RegExNode() {

		next = NULL;
		previous = NULL;
		parent = NULL;
		quantifiable = false;
		length = -1;
	}

	virtual ~RegExNode() {

		if (next)
			memdelete(next);
	}

	// For avoiding RTTI
	virtual bool is_look_behind() { return false; }

	virtual int test(RegExSearch &s, int pos) const {

		return next ? next->test(s, pos) : -1;
	}

	virtual int test_parent(RegExSearch &s, int pos) const {

		if (next)
			pos = next->test(s, pos);

		if (pos >= 0) {
			s.complete = true;
			if (parent)
				pos = parent->test_parent(s, pos);
		}

		if (pos < 0)
			s.complete = false;

		return pos;
	}

	void increment_length(int amount, bool subtract = false) {

		if (amount >= 0 && length >= 0) {
			if (!subtract)
				length += amount;
			else
				length -= amount;
		} else {
			length = -1;
		}

		if (parent)
			parent->increment_length(amount, subtract);
	}
};

struct RegExNodeChar : public RegExNode {

	CharType ch;

	RegExNodeChar(CharType p_char) {

		length = 1;
		quantifiable = true;
		ch = p_char;
	}

	virtual int test(RegExSearch &s, int pos) const {

		if (s.end <= pos || 0 > pos || s.at(pos) != ch)
			return -1;

		return next ? next->test(s, pos + 1) : pos + 1;
	}

	static CharType parse_escape(const CharType *&c) {

		int point = 0;
		switch (c[1]) {
			case 'x':
				for (int i = 2; i <= 3; ++i) {
					int res = RegEx_hex2int(c[i]);
					if (res == -1)
						return '\0';
					point = (point << 4) + res;
				}
				c = &c[3];
				return CharType(point);
			case 'u':
				for (int i = 2; i <= 5; ++i) {
					int res = RegEx_hex2int(c[i]);
					if (res == -1)
						return '\0';
					point = (point << 4) + res;
				}
				c = &c[5];
				return CharType(point);
			case '0': ++c; return '\0';
			case 'a': ++c; return '\a';
			case 'e': ++c; return '\e';
			case 'f': ++c; return '\f';
			case 'n': ++c; return '\n';
			case 'r': ++c; return '\r';
			case 't': ++c; return '\t';
			case 'v': ++c; return '\v';
			case 'b': ++c; return '\b';
			default: break;
		}
		return (++c)[0];
	}
};

struct RegExNodeRange : public RegExNode {

	CharType start;
	CharType end;

	RegExNodeRange(CharType p_start, CharType p_end) {

		length = 1;
		quantifiable = true;
		start = p_start;
		end = p_end;
	}

	virtual int test(RegExSearch &s, int pos) const {

		if (s.end <= pos || 0 > pos)
			return -1;

		CharType c = s.at(pos);
		if (c < start || end < c)
			return -1;

		return next ? next->test(s, pos + 1) : pos + 1;
	}
};

struct RegExNodeShorthand : public RegExNode {

	CharType repr;

	RegExNodeShorthand(CharType p_repr) {

		length = 1;
		quantifiable = true;
		repr = p_repr;
	}

	virtual int test(RegExSearch &s, int pos) const {

		if (s.end <= pos || 0 > pos)
			return -1;

		bool found = false;
		bool invert = false;
		CharType c = s.at(pos);
		switch (repr) {
			case '.':
				found = true;
				break;
			case 'W':
				invert = true;
			case 'w':
				found = (c == '_' || iswalnum(c) != 0);
				break;
			case 'D':
				invert = true;
			case 'd':
				found = ('0' <= c && c <= '9');
				break;
			case 'S':
				invert = true;
			case 's':
				found = (iswspace(c) != 0);
				break;
			default:
				break;
		}

		if (found == invert)
			return -1;

		return next ? next->test(s, pos + 1) : pos + 1;
	}
};

struct RegExNodeClass : public RegExNode {

	enum Type {
		Type_none,
		Type_alnum,
		Type_alpha,
		Type_ascii,
		Type_blank,
		Type_cntrl,
		Type_digit,
		Type_graph,
		Type_lower,
		Type_print,
		Type_punct,
		Type_space,
		Type_upper,
		Type_xdigit,
		Type_word
	};

	Type type;

	bool test_class(CharType c) const {

		static Vector<CharType> REGEX_NODE_SPACE = String(" \t\r\n\f");
		static Vector<CharType> REGEX_NODE_PUNCT = String("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");

		switch (type) {
			case Type_alnum:
				if ('0' <= c && c <= '9') return true;
				if ('a' <= c && c <= 'z') return true;
				if ('A' <= c && c <= 'Z') return true;
				return false;
			case Type_alpha:
				if ('a' <= c && c <= 'z') return true;
				if ('A' <= c && c <= 'Z') return true;
				return false;
			case Type_ascii:
				return (0x00 <= c && c <= 0x7F);
			case Type_blank:
				return (c == ' ' || c == '\t');
			case Type_cntrl:
				return ((0x00 <= c && c <= 0x1F) || c == 0x7F);
			case Type_digit:
				return ('0' <= c && c <= '9');
			case Type_graph:
				return (0x20 < c && c < 0x7F);
			case Type_lower:
				return ('a' <= c && c <= 'z');
			case Type_print:
				return (0x20 < c && c < 0x7f);
			case Type_punct:
				return (REGEX_NODE_PUNCT.find(c) >= 0);
			case Type_space:
				return (REGEX_NODE_SPACE.find(c) >= 0);
			case Type_upper:
				return ('A' <= c && c <= 'Z');
			case Type_xdigit:
				if ('0' <= c && c <= '9') return true;
				if ('a' <= c && c <= 'f') return true;
				if ('A' <= c && c <= 'F') return true;
				return false;
			case Type_word:
				if ('0' <= c && c <= '9') return true;
				if ('a' <= c && c <= 'z') return true;
				if ('A' <= c && c <= 'Z') return true;
				return (c == '_');
			default:
				return false;
		}
		return false;
	}

	RegExNodeClass(Type p_type) {

		length = 1;
		quantifiable = true;
		type = p_type;
	}

	virtual int test(RegExSearch &s, int pos) const {

		if (s.end <= pos || 0 > pos)
			return -1;

		if (!test_class(s.at(pos)))
			return -1;

		return next ? next->test(s, pos + 1) : pos + 1;
	}

#define REGEX_CMP_CLASS(POS, NAME) \
	if (cmp_class(POS, #NAME)) return Type_##NAME

	static Type parse_type(const CharType *&p_pos) {

		REGEX_CMP_CLASS(p_pos, alnum);
		REGEX_CMP_CLASS(p_pos, alpha);
		REGEX_CMP_CLASS(p_pos, ascii);
		REGEX_CMP_CLASS(p_pos, blank);
		REGEX_CMP_CLASS(p_pos, cntrl);
		REGEX_CMP_CLASS(p_pos, digit);
		REGEX_CMP_CLASS(p_pos, graph);
		REGEX_CMP_CLASS(p_pos, lower);
		REGEX_CMP_CLASS(p_pos, print);
		REGEX_CMP_CLASS(p_pos, punct);
		REGEX_CMP_CLASS(p_pos, space);
		REGEX_CMP_CLASS(p_pos, upper);
		REGEX_CMP_CLASS(p_pos, xdigit);
		REGEX_CMP_CLASS(p_pos, word);
		return Type_none;
	}

	static bool cmp_class(const CharType *&p_pos, const char *p_text) {

		unsigned int i = 0;
		for (i = 0; p_text[i] != '\0'; ++i)
			if (p_pos[i] != p_text[i])
				return false;

		if (p_pos[i++] != ':' || p_pos[i] != ']')
			return false;

		p_pos = &p_pos[i];
		return true;
	}
};

struct RegExNodeAnchorStart : public RegExNode {

	RegExNodeAnchorStart() {

		length = 0;
	}

	virtual int test(RegExSearch &s, int pos) const {

		if (pos != 0)
			return -1;

		return next ? next->test(s, pos) : pos;
	}
};

struct RegExNodeAnchorEnd : public RegExNode {

	RegExNodeAnchorEnd() {

		length = 0;
	}

	virtual int test(RegExSearch &s, int pos) const {

		if (pos != s.eof)
			return -1;

		return next ? next->test(s, pos) : pos;
	}
};

struct RegExNodeWordBoundary : public RegExNode {

	bool inverse;

	RegExNodeWordBoundary(bool p_inverse) {

		length = 0;
		inverse = p_inverse;
	}

	virtual int test(RegExSearch &s, int pos) const {

		bool left = false;
		bool right = false;

		if (pos != 0) {
			CharType c = s.at(pos - 1);
			if (c == '_' || iswalnum(c))
				left = true;
		}

		if (pos != s.eof) {
			CharType c = s.at(pos);
			if (c == '_' || iswalnum(c))
				right = true;
		}

		if ((left == right) != inverse)
			return -1;

		return next ? next->test(s, pos) : pos;
	}
};

struct RegExNodeQuantifier : public RegExNode {

	int min;
	int max;
	bool greedy;
	RegExNode *child;

	RegExNodeQuantifier(int p_min, int p_max) {

		min = p_min;
		max = p_max;
		greedy = true;
		child = NULL;
	}

	~RegExNodeQuantifier() {

		if (child)
			memdelete(child);
	}

	virtual int test(RegExSearch &s, int pos) const {

		return test_step(s, pos, 0, pos);
	}

	virtual int test_parent(RegExSearch &s, int pos) const {

		s.complete = false;
		return pos;
	}

	int test_step(RegExSearch &s, int pos, int level, int start) const {

		if (pos > s.end)
			return -1;

		if (!greedy && level > min) {
			int res = next ? next->test(s, pos) : pos;
			if (s.complete)
				return res;

			if (res >= 0 && parent->test_parent(s, res) >= 0)
				return res;
		}

		if (max >= 0 && level > max)
			return -1;

		int res = pos;
		if (level >= 1) {
			if (level > min + 1 && pos == start)
				return -1;

			res = child->test(s, pos);
			if (s.complete)
				return res;
		}

		if (res >= 0) {

			int res_step = test_step(s, res, level + 1, start);
			if (res_step >= 0)
				return res_step;

			if (greedy && level >= min) {
				if (next)
					res = next->test(s, res);
				if (s.complete)
					return res;

				if (res >= 0 && parent->test_parent(s, res) >= 0)
					return res;
			}
		}
		return -1;
	}
};

struct RegExNodeBackReference : public RegExNode {

	int id;

	RegExNodeBackReference(int p_id) {

		length = -1;
		quantifiable = true;
		id = p_id;
	}

	virtual int test(RegExSearch &s, int pos) const {

		RegExMatch::Group &ref = s.match->captures[id];
		for (int i = 0; i < ref.length; ++i) {

			if (pos + i >= s.end)
				return -1;

			if (s.at(ref.start + i) != s.at(pos + i))
				return -1;
		}
		return next ? next->test(s, pos + ref.length) : pos + ref.length;
	}
};

struct RegExNodeGroup : public RegExNode {

	bool inverse;
	bool reset_pos;
	Vector<RegExNode *> childset;
	RegExNode *back;

	RegExNodeGroup() {

		length = 0;
		quantifiable = true;
		inverse = false;
		reset_pos = false;
		back = NULL;
	}

	virtual ~RegExNodeGroup() {

		for (int i = 0; i < childset.size(); ++i)
			memdelete(childset[i]);
	}

	virtual int test(RegExSearch &s, int pos) const {

		for (int i = 0; i < childset.size(); ++i) {

			s.complete = false;

			int res = childset[i]->test(s, pos);

			if (s.complete)
				return res;

			if (inverse) {
				if (res < 0)
					res = pos + 1;
				else
					return -1;

				if (i + 1 < childset.size())
					continue;
			}

			if (res >= 0) {
				if (reset_pos)
					res = pos;
				return next ? next->test(s, res) : res;
			}
		}
		return -1;
	}

	void add_child(RegExNode *node) {

		node->parent = this;
		node->previous = back;

		if (back)
			back->next = node;
		else
			childset.push_back(node);

		increment_length(node->length);

		back = node;
	}

	void add_childset() {

		if (childset.size() > 0)
			length = -1;
		back = NULL;
	}

	RegExNode *swap_back(RegExNode *node) {

		RegExNode *old = back;

		if (old) {
			if (!old->previous)
				childset.remove(childset.size() - 1);
			back = old->previous;
			increment_length(old->length, true);
		}

		add_child(node);

		return old;
	}
};

struct RegExNodeCapturing : public RegExNodeGroup {

	int id;

	RegExNodeCapturing(int p_id = 0) {

		id = p_id;
	}

	virtual int test(RegExSearch &s, int pos) const {

		RegExMatch::Group &ref = s.match->captures[id];
		int old_start = ref.start;
		ref.start = pos;

		int res = RegExNodeGroup::test(s, pos);

		if (res >= 0) {
			if (!s.complete)
				ref.length = res - pos;
		} else {
			ref.start = old_start;
		}

		return res;
	}

	virtual int test_parent(RegExSearch &s, int pos) const {

		RegExMatch::Group &ref = s.match->captures[id];
		ref.length = pos - ref.start;
		return RegExNode::test_parent(s, pos);
	}

	static Variant parse_name(const CharType *&c, bool p_allow_numeric) {

		if (c[1] == '0') {
			return -1;
		} else if ('1' <= c[1] && c[1] <= '9') {
			if (!p_allow_numeric)
				return -1;
			int res = (++c)[0] - '0';
			while ('0' <= c[1] && c[1] <= '9')
				res = res * 10 + int((++c)[0] - '0');
			if ((++c)[0] != '>')
				return -1;
			return res;
		} else if (iswalnum(c[1])) {
			String res(++c, 1);
			while (iswalnum(c[1]))
				res += String(++c, 1);
			if ((++c)[0] != '>')
				return -1;
			return res;
		}
		return -1;
	}
};

struct RegExNodeLookAhead : public RegExNodeGroup {

	int id;

	RegExNodeLookAhead(bool p_inverse, int p_id = 0) {

		quantifiable = false;
		inverse = p_inverse;
		reset_pos = true;
		id = p_id;
	}

	virtual int test(RegExSearch &s, int pos) const {

		s.lookahead_pos[id] = pos;
		return RegExNodeGroup::test(s, pos);
	}

	virtual int test_parent(RegExSearch &s, int pos) const {

		return RegExNode::test_parent(s, s.lookahead_pos[id]);
	}
};

struct RegExNodeLookBehind : public RegExNodeGroup {

	RegExNodeLookBehind(bool p_inverse, int p_id = 0) {

		quantifiable = false;
		inverse = p_inverse;
		reset_pos = true;
	}

	virtual bool is_look_behind() { return true; }

	virtual int test(RegExSearch &s, int pos) const {

		if (pos < length)
			return -1;
		return RegExNodeGroup::test(s, pos - length);
	}
};

struct RegExNodeBracket : public RegExNode {

	bool inverse;
	Vector<RegExNode *> children;

	RegExNodeBracket() {

		length = 1;
		quantifiable = true;
		inverse = false;
	}

	virtual ~RegExNodeBracket() {

		for (int i = 0; i < children.size(); ++i)
			memdelete(children[i]);
	}

	virtual int test(RegExSearch &s, int pos) const {

		for (int i = 0; i < children.size(); ++i) {

			int res = children[i]->test(s, pos);

			if (inverse) {
				if (res < 0)
					res = pos + 1;
				else
					return -1;

				if (i + 1 < children.size())
					continue;
			}

			if (res >= 0)
				return next ? next->test(s, res) : res;
		}
		return -1;
	}

	void add_child(RegExNode *node) {

		node->parent = this;
		children.push_back(node);
	}

	void pop_back() {

		memdelete(children[children.size() - 1]);
		children.remove(children.size() - 1);
	}
};

#define REGEX_EXPAND_FAIL(MSG) \
	{                          \
		ERR_PRINT(MSG);        \
		return String();       \
	}

String RegExMatch::expand(const String &p_template) const {

	String res;
	for (const CharType *c = p_template.c_str(); *c != '\0'; ++c) {
		if (c[0] == '\\') {
			if (('1' <= c[1] && c[1] <= '9') || (c[1] == 'g' && c[2] == '{')) {

				int ref = 0;
				bool unclosed = false;

				if (c[1] == 'g') {
					unclosed = true;
					c = &c[2];
				}

				while ('0' <= c[1] && c[1] <= '9') {
					ref = ref * 10 + int(c[1] - '0');
					++c;
				}

				if (unclosed) {
					if (c[1] != '}')
						REGEX_EXPAND_FAIL("unclosed backreference '{'");
					++c;
				}

				res += get_string(ref);

			} else if (c[1] == 'g' && c[2] == '<') {

				const CharType *d = &c[2];

				Variant name = RegExNodeCapturing::parse_name(d, true);
				if (name == Variant(-1))
					REGEX_EXPAND_FAIL("unrecognised character for group name");

				c = d;

				res += get_string(name);

			} else {

				const CharType *d = c;
				CharType ch = RegExNodeChar::parse_escape(d);
				if (c == d)
					REGEX_EXPAND_FAIL("invalid escape token");
				res += String(&ch, 1);
				c = d;
			}
		} else {
			res += String(c, 1);
		}
	}
	return res;
}

int RegExMatch::get_group_count() const {

	int count = 0;
	for (int i = 1; i < captures.size(); ++i)
		if (captures[i].name.get_type() == Variant::INT)
			++count;
	return count;
}

Array RegExMatch::get_group_array() const {

	Array res;
	for (int i = 1; i < captures.size(); ++i) {
		const RegExMatch::Group &capture = captures[i];
		if (capture.name.get_type() != Variant::INT)
			continue;

		if (capture.start >= 0)
			res.push_back(string.substr(capture.start, capture.length));
		else
			res.push_back(String());
	}
	return res;
}

Array RegExMatch::get_names() const {

	Array res;
	for (int i = 1; i < captures.size(); ++i)
		if (captures[i].name.get_type() == Variant::STRING)
			res.push_back(captures[i].name);
	return res;
}

Dictionary RegExMatch::get_name_dict() const {

	Dictionary res;
	for (int i = 1; i < captures.size(); ++i) {
		const RegExMatch::Group &capture = captures[i];
		if (capture.name.get_type() != Variant::STRING)
			continue;

		if (capture.start >= 0)
			res[capture.name] = string.substr(capture.start, capture.length);
		else
			res[capture.name] = String();
	}
	return res;
}

String RegExMatch::get_string(const Variant &p_name) const {

	for (int i = 0; i < captures.size(); ++i) {

		const RegExMatch::Group &capture = captures[i];

		if (capture.name != p_name)
			continue;

		if (capture.start == -1)
			return String();

		return string.substr(capture.start, capture.length);
	}
	return String();
}

int RegExMatch::get_start(const Variant &p_name) const {

	for (int i = 0; i < captures.size(); ++i)
		if (captures[i].name == p_name)
			return captures[i].start;
	return -1;
}

int RegExMatch::get_end(const Variant &p_name) const {

	for (int i = 0; i < captures.size(); ++i)
		if (captures[i].name == p_name)
			return captures[i].start + captures[i].length;
	return -1;
}

RegExMatch::RegExMatch() {
}

static bool RegEx_is_shorthand(CharType ch) {

	switch (ch) {
		case 'w':
		case 'W':
		case 'd':
		case 'D':
		case 's':
		case 'S':
			return true;
		default:
			break;
	}
	return false;
}

#define REGEX_COMPILE_FAIL(MSG) \
	{                           \
		ERR_PRINT(MSG);         \
		clear();                \
		return FAILED;          \
	}

Error RegEx::compile(const String &p_pattern) {

	ERR_FAIL_COND_V(p_pattern.length() == 0, FAILED);

	if (pattern == p_pattern && root)
		return OK;

	clear();
	pattern = p_pattern;
	group_names.push_back(0);
	RegExNodeGroup *root_group = memnew(RegExNodeCapturing(0));
	root = root_group;
	Vector<RegExNodeGroup *> stack;
	stack.push_back(root_group);
	int lookahead_level = 0;
	int numeric_groups = 0;
	const int numeric_max = 9;

	for (const CharType *c = p_pattern.c_str(); *c != '\0'; ++c) {

		switch (c[0]) {
			case '(':
				if (c[1] == '?') {

					RegExNodeGroup *group = NULL;
					switch (c[2]) {
						case ':':
							c = &c[2];
							group = memnew(RegExNodeGroup());
							break;
						case '!':
						case '=':
							group = memnew(RegExNodeLookAhead((c[2] == '!'), lookahead_level++));
							if (lookahead_depth < lookahead_level)
								lookahead_depth = lookahead_level;
							c = &c[2];
							break;
						case '<':
							if (c[3] == '!' || c[3] == '=') {
								group = memnew(RegExNodeLookBehind((c[3] == '!'), lookahead_level++));
								c = &c[3];
							}
							break;
						case 'P':
							if (c[3] == '<') {
								const CharType *d = &c[3];
								Variant name = RegExNodeCapturing::parse_name(d, false);
								if (name == Variant(-1))
									REGEX_COMPILE_FAIL("unrecognised character for group name");
								group = memnew(RegExNodeCapturing(group_names.size()));
								group_names.push_back(name);
								c = d;
							}
						default:
							break;
					}
					if (!group)
						REGEX_COMPILE_FAIL("unrecognised qualifier for group");
					stack[0]->add_child(group);
					stack.insert(0, group);

				} else if (numeric_groups < numeric_max) {

					RegExNodeCapturing *group = memnew(RegExNodeCapturing(group_names.size()));
					group_names.push_back(++numeric_groups);
					stack[0]->add_child(group);
					stack.insert(0, group);

				} else {

					RegExNodeGroup *group = memnew(RegExNodeGroup());
					stack[0]->add_child(group);
					stack.insert(0, group);
				}
				break;
			case ')':
				if (stack.size() == 1)
					REGEX_COMPILE_FAIL("unexpected ')'");
				stack.remove(0);
				break;
			case '\\':
				if (('1' <= c[1] && c[1] <= '9') || (c[1] == 'g' && c[2] == '{')) {

					int ref = 0;
					bool unclosed = false;

					if (c[1] == 'g') {
						unclosed = true;
						c = &c[2];
					}

					while ('0' <= c[1] && c[1] <= '9') {
						ref = ref * 10 + int(c[1] - '0');
						++c;
					}

					if (unclosed) {
						if (c[1] != '}')
							REGEX_COMPILE_FAIL("unclosed backreference '{'");
						++c;
					}

					if (ref > numeric_groups || ref <= 0)
						REGEX_COMPILE_FAIL("backreference not found");

					for (int i = 0; i < stack.size(); ++i)
						if (stack[i]->is_look_behind())
							REGEX_COMPILE_FAIL("backreferences inside lookbehind not supported");

					for (int i = 0; i < group_names.size(); ++i) {
						if (group_names[i].get_type() == Variant::INT && int(group_names[i]) == ref) {
							ref = group_names[i];
							break;
						}
					}

					stack[0]->add_child(memnew(RegExNodeBackReference(ref)));
				}
				if (c[1] == 'g' && c[2] == '<') {

					const CharType *d = &c[2];

					Variant name = RegExNodeCapturing::parse_name(d, true);
					if (name == Variant(-1))
						REGEX_COMPILE_FAIL("unrecognised character for group name");

					c = d;

					for (int i = 0; i < stack.size(); ++i)
						if (stack[i]->is_look_behind())
							REGEX_COMPILE_FAIL("backreferences inside lookbehind not supported");

					int ref = -1;

					for (int i = 0; i < group_names.size(); ++i) {
						if (group_names[i].get_type() == Variant::INT && int(group_names[i]) == ref) {
							ref = group_names[i];
							break;
						}
					}

					if (ref == -1)
						REGEX_COMPILE_FAIL("backreference not found");

					stack[0]->add_child(memnew(RegExNodeBackReference(ref)));

				} else if (c[1] == 'b' || c[1] == 'B') {

					stack[0]->add_child(memnew(RegExNodeWordBoundary(*(++c) == 'B')));

				} else if (RegEx_is_shorthand(c[1])) {

					stack[0]->add_child(memnew(RegExNodeShorthand(*(++c))));

				} else {

					const CharType *d = c;
					CharType ch = RegExNodeChar::parse_escape(d);
					if (c == d)
						REGEX_COMPILE_FAIL("invalid escape token");
					stack[0]->add_child(memnew(RegExNodeChar(ch)));
					c = d;
				}
				break;
			case '[': {
				RegExNodeBracket *bracket = memnew(RegExNodeBracket());
				stack[0]->add_child(bracket);
				if (c[1] == '^') {
					bracket->inverse = true;
					++c;
				}
				bool first_child = true;
				CharType previous_child;
				bool previous_child_single = false;
				while (true) {
					++c;
					if (!first_child && c[0] == ']') {

						break;

					} else if (c[0] == '\0') {

						REGEX_COMPILE_FAIL("unclosed bracket expression '['");

					} else if (c[0] == '\\') {

						if (RegEx_is_shorthand(c[1])) {
							bracket->add_child(memnew(RegExNodeShorthand(*(++c))));
						} else {
							const CharType *d = c;
							CharType ch = RegExNodeChar::parse_escape(d);
							if (c == d)
								REGEX_COMPILE_FAIL("invalid escape token");
							bracket->add_child(memnew(RegExNodeChar(ch)));
							c = d;
							previous_child = ch;
							previous_child_single = true;
						}

					} else if (c[0] == ']' && c[1] == ':') {

						const CharType *d = &c[2];
						RegExNodeClass::Type type = RegExNodeClass::parse_type(d);
						if (type != RegExNodeClass::Type_none) {

							c = d;
							previous_child_single = false;

						} else {

							bracket->add_child(memnew(RegExNodeChar('[')));
							previous_child = '[';
							previous_child_single = true;
						}
					} else if (previous_child_single && c[0] == '-') {

						if (c[1] != '\0' && c[1] != ']') {

							CharType next;

							if (c[1] == '\\') {
								const CharType *d = ++c;
								next = RegExNodeChar::parse_escape(d);
								if (c == d)
									REGEX_COMPILE_FAIL("invalid escape token");
							} else {
								next = *(++c);
							}

							if (next < previous_child)
								REGEX_COMPILE_FAIL("text range out of order");

							bracket->pop_back();
							bracket->add_child(memnew(RegExNodeRange(previous_child, next)));
							previous_child_single = false;
						} else {

							bracket->add_child(memnew(RegExNodeChar('-')));
							previous_child = '-';
							previous_child_single = true;
						}
					} else {

						bracket->add_child(memnew(RegExNodeChar(c[0])));
						previous_child = c[0];
						previous_child_single = true;
					}
					first_child = false;
				}
			} break;
			case '|':
				for (int i = 0; i < stack.size(); ++i)
					if (stack[i]->is_look_behind())
						REGEX_COMPILE_FAIL("alternations inside lookbehind not supported");
				stack[0]->add_childset();
				break;
			case '^':
				stack[0]->add_child(memnew(RegExNodeAnchorStart()));
				break;
			case '$':
				stack[0]->add_child(memnew(RegExNodeAnchorEnd()));
				break;
			case '.':
				stack[0]->add_child(memnew(RegExNodeShorthand('.')));
				break;
			case '?':
			case '*':
			case '+':
			case '{': {
				int min_val = 0;
				int max_val = -1;
				bool valid = true;
				const CharType *d = c;
				bool max_set = true;
				switch (c[0]) {
					case '?':
						min_val = 0;
						max_val = 1;
						break;
					case '*':
						min_val = 0;
						max_val = -1;
						break;
					case '+':
						min_val = 1;
						max_val = -1;
						break;
					case '{':
						max_set = false;
						while (valid) {
							++d;
							if (d[0] == '}') {
								break;
							} else if (d[0] == ',') {
								max_set = true;
							} else if ('0' <= d[0] && d[0] <= '9') {
								if (max_set) {
									if (max_val < 0)
										max_val = int(d[0] - '0');
									else
										max_val = max_val * 10 + int(d[0] - '0');
								} else {
									min_val = min_val * 10 + int(d[0] - '0');
								}
							} else {
								valid = false;
							}
						}
						break;
					default:
						break;
				}

				if (!max_set)
					max_val = min_val;

				if (valid) {

					c = d;

					if (stack[0]->back == NULL || !stack[0]->back->quantifiable)
						REGEX_COMPILE_FAIL("element not quantifiable");

					if (min_val != max_val)
						for (int i = 0; i < stack.size(); ++i)
							if (stack[i]->is_look_behind())
								REGEX_COMPILE_FAIL("variable length quantifiers inside lookbehind not supported");

					RegExNodeQuantifier *quant = memnew(RegExNodeQuantifier(min_val, max_val));
					quant->child = stack[0]->swap_back(quant);
					quant->child->previous = NULL;
					quant->child->parent = quant;

					if (min_val == max_val && quant->child->length >= 0)
						quant->length = max_val * quant->child->length;

					if (c[1] == '?') {
						quant->greedy = false;
						++c;
					}
					break;
				}
			}
			default:
				stack[0]->add_child(memnew(RegExNodeChar(c[0])));
				break;
		}
	}
	if (stack.size() > 1)
		REGEX_COMPILE_FAIL("unclosed group '('");
	return OK;
}

Ref<RegExMatch> RegEx::search(const String &p_text, int p_start, int p_end) const {

	ERR_FAIL_COND_V(!is_valid(), NULL);
	ERR_FAIL_COND_V(p_start < 0, NULL);
	ERR_FAIL_COND_V(p_start >= p_text.length(), NULL);
	ERR_FAIL_COND_V(p_end > p_text.length(), NULL);
	ERR_FAIL_COND_V(p_end != -1 && p_end < p_start, NULL);

	Ref<RegExMatch> res = memnew(RegExMatch());

	for (int i = 0; i < group_names.size(); ++i) {
		RegExMatch::Group group;
		group.name = group_names[i];
		res->captures.push_back(group);
	}

	res->string = p_text;

	if (p_end == -1)
		p_end = p_text.length();

	RegExSearch s(res, p_end, lookahead_depth);

	for (int i = p_start; i <= s.end; ++i) {
		for (int c = 0; c < group_names.size(); ++c) {
			res->captures[c].start = -1;
			res->captures[c].length = 0;
		}
		if (root->test(s, i) >= 0)
			break;
	}

	if (res->captures[0].start >= 0)
		return res;
	return NULL;
}

String RegEx::sub(const String &p_text, const String &p_replacement, bool p_all, int p_start, int p_end) const {

	ERR_FAIL_COND_V(!is_valid(), p_text);
	ERR_FAIL_COND_V(p_start < 0, p_text);
	ERR_FAIL_COND_V(p_start >= p_text.length(), p_text);
	ERR_FAIL_COND_V(p_end > p_text.length(), p_text);
	ERR_FAIL_COND_V(p_end != -1 && p_end < p_start, p_text);

	String text = p_text;
	int start = p_start;

	if (p_end == -1)
		p_end = p_text.length();

	while (start < text.length() && (p_all || start == p_start)) {

		Ref<RegExMatch> m = search(text, start, p_end);

		RegExMatch::Group &s = m->captures[0];

		if (s.start < 0)
			break;

		String res = text.substr(0, s.start) + m->expand(p_replacement);

		start = res.length();

		if (s.length == 0)
			++start;

		int sub_end = s.start + s.length;
		if (sub_end < text.length())
			res += text.substr(sub_end, text.length() - sub_end);

		p_end += res.length() - text.length();

		text = res;
	}
	return text;
}

void RegEx::clear() {

	if (root)
		memdelete(root);

	root = NULL;
	group_names.clear();
	lookahead_depth = 0;
}

bool RegEx::is_valid() const {

	return (root != NULL);
}

String RegEx::get_pattern() const {

	return pattern;
}

int RegEx::get_group_count() const {

	int count = 0;
	for (int i = 1; i < group_names.size(); ++i)
		if (group_names[i].get_type() == Variant::INT)
			++count;
	return count;
}

Array RegEx::get_names() const {

	Array res;
	for (int i = 1; i < group_names.size(); ++i)
		if (group_names[i].get_type() == Variant::STRING)
			res.push_back(group_names[i]);
	return res;
}

RegEx::RegEx() {

	root = NULL;
	lookahead_depth = 0;
}

RegEx::RegEx(const String &p_pattern) {

	root = NULL;
	compile(p_pattern);
}

RegEx::~RegEx() {

	if (root)
		memdelete(root);
}

void RegExMatch::_bind_methods() {

	ClassDB::bind_method(D_METHOD("expand", "template"), &RegExMatch::expand);
	ClassDB::bind_method(D_METHOD("get_group_count"), &RegExMatch::get_group_count);
	ClassDB::bind_method(D_METHOD("get_group_array"), &RegExMatch::get_group_array);
	ClassDB::bind_method(D_METHOD("get_names"), &RegExMatch::get_names);
	ClassDB::bind_method(D_METHOD("get_name_dict"), &RegExMatch::get_name_dict);
	ClassDB::bind_method(D_METHOD("get_string", "name"), &RegExMatch::get_string, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_start", "name"), &RegExMatch::get_start, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_end", "name"), &RegExMatch::get_end, DEFVAL(0));
}

void RegEx::_bind_methods() {

	ClassDB::bind_method(D_METHOD("clear"), &RegEx::clear);
	ClassDB::bind_method(D_METHOD("compile", "pattern"), &RegEx::compile);
	ClassDB::bind_method(D_METHOD("search", "text", "start", "end"), &RegEx::search, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("sub", "text", "replacement", "all", "start", "end"), &RegEx::sub, DEFVAL(false), DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("is_valid"), &RegEx::is_valid);
	ClassDB::bind_method(D_METHOD("get_pattern"), &RegEx::get_pattern);
	ClassDB::bind_method(D_METHOD("get_group_count"), &RegEx::get_group_count);
	ClassDB::bind_method(D_METHOD("get_names"), &RegEx::get_names);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "pattern"), "compile", "get_pattern");
}
