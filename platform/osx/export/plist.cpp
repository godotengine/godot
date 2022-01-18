/*************************************************************************/
/*  plist.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "modules/modules_enabled.gen.h" // For regex.

#include "plist.h"

#ifdef MODULE_REGEX_ENABLED

Ref<PListNode> PListNode::new_array() {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_ARRAY;
	return node;
}

Ref<PListNode> PListNode::new_dict() {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_DICT;
	return node;
}

Ref<PListNode> PListNode::new_string(const String &p_string) {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_STRING;
	node->data_string = p_string.utf8();
	return node;
}

Ref<PListNode> PListNode::new_data(const String &p_string) {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_DATA;
	node->data_string = p_string.utf8();
	return node;
}

Ref<PListNode> PListNode::new_date(const String &p_string) {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_DATE;
	node->data_string = p_string.utf8();
	return node;
}

Ref<PListNode> PListNode::new_bool(bool p_bool) {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_BOOLEAN;
	node->data_bool = p_bool;
	return node;
}

Ref<PListNode> PListNode::new_int(int32_t p_int) {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_INTEGER;
	node->data_int = p_int;
	return node;
}

Ref<PListNode> PListNode::new_real(float p_real) {
	Ref<PListNode> node = memnew(PListNode());
	ERR_FAIL_COND_V(node.is_null(), Ref<PListNode>());
	node->data_type = PList::PLNodeType::PL_NODE_TYPE_REAL;
	node->data_real = p_real;
	return node;
}

bool PListNode::push_subnode(const Ref<PListNode> &p_node, const String &p_key) {
	ERR_FAIL_COND_V(p_node.is_null(), false);
	if (data_type == PList::PLNodeType::PL_NODE_TYPE_DICT) {
		ERR_FAIL_COND_V(p_key.is_empty(), false);
		ERR_FAIL_COND_V(data_dict.has(p_key), false);
		data_dict[p_key] = p_node;
		return true;
	} else if (data_type == PList::PLNodeType::PL_NODE_TYPE_ARRAY) {
		data_array.push_back(p_node);
		return true;
	} else {
		ERR_FAIL_V_MSG(false, "PList: Invalid parent node type, should be DICT or ARRAY.");
	}
}

size_t PListNode::get_asn1_size(uint8_t p_len_octets) const {
	// Get size of all data, excluding type and size information.
	switch (data_type) {
		case PList::PLNodeType::PL_NODE_TYPE_NIL: {
			return 0;
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DATA:
		case PList::PLNodeType::PL_NODE_TYPE_DATE: {
			ERR_FAIL_V_MSG(0, "PList: DATE and DATA nodes are not supported by ASN.1 serialization.");
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_STRING: {
			return data_string.length();
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_BOOLEAN: {
			return 1;
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_INTEGER:
		case PList::PLNodeType::PL_NODE_TYPE_REAL: {
			return 4;
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_ARRAY: {
			size_t size = 0;
			for (int i = 0; i < data_array.size(); i++) {
				size += 1 + _asn1_size_len(p_len_octets) + data_array[i]->get_asn1_size(p_len_octets);
			}
			return size;
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DICT: {
			size_t size = 0;
			for (const Map<String, Ref<PListNode>>::Element *it = data_dict.front(); it; it = it->next()) {
				size += 1 + _asn1_size_len(p_len_octets); // Sequence.
				size += 1 + _asn1_size_len(p_len_octets) + it->key().utf8().length(); //Key.
				size += 1 + _asn1_size_len(p_len_octets) + it->value()->get_asn1_size(p_len_octets); // Value.
			}
			return size;
		} break;
		default: {
			return 0;
		} break;
	}
}

int PListNode::_asn1_size_len(uint8_t p_len_octets) {
	if (p_len_octets > 1) {
		return p_len_octets + 1;
	} else {
		return 1;
	}
}

void PListNode::store_asn1_size(PackedByteArray &p_stream, uint8_t p_len_octets) const {
	uint32_t size = get_asn1_size(p_len_octets);
	if (p_len_octets > 1) {
		p_stream.push_back(0x80 + p_len_octets);
	}
	for (int i = p_len_octets - 1; i >= 0; i--) {
		uint8_t x = (size >> i * 8) & 0xFF;
		p_stream.push_back(x);
	}
}

bool PListNode::store_asn1(PackedByteArray &p_stream, uint8_t p_len_octets) const {
	// Convert to binary ASN1 stream.
	bool valid = true;
	switch (data_type) {
		case PList::PLNodeType::PL_NODE_TYPE_NIL: {
			// Nothing to store.
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DATE:
		case PList::PLNodeType::PL_NODE_TYPE_DATA: {
			ERR_FAIL_V_MSG(false, "PList: DATE and DATA nodes are not supported by ASN.1 serialization.");
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_STRING: {
			p_stream.push_back(0x0C);
			store_asn1_size(p_stream, p_len_octets);
			for (int i = 0; i < data_string.size(); i++) {
				p_stream.push_back(data_string[i]);
			}
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_BOOLEAN: {
			p_stream.push_back(0x01);
			store_asn1_size(p_stream, p_len_octets);
			if (data_bool) {
				p_stream.push_back(0x01);
			} else {
				p_stream.push_back(0x00);
			}
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_INTEGER: {
			p_stream.push_back(0x02);
			store_asn1_size(p_stream, p_len_octets);
			for (int i = 4; i >= 0; i--) {
				uint8_t x = (data_int >> i * 8) & 0xFF;
				p_stream.push_back(x);
			}
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_REAL: {
			p_stream.push_back(0x03);
			store_asn1_size(p_stream, p_len_octets);
			for (int i = 4; i >= 0; i--) {
				uint8_t x = (data_int >> i * 8) & 0xFF;
				p_stream.push_back(x);
			}
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_ARRAY: {
			p_stream.push_back(0x30); // Sequence.
			store_asn1_size(p_stream, p_len_octets);
			for (int i = 0; i < data_array.size(); i++) {
				valid = valid && data_array[i]->store_asn1(p_stream, p_len_octets);
			}
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DICT: {
			p_stream.push_back(0x31); // Set.
			store_asn1_size(p_stream, p_len_octets);
			for (const Map<String, Ref<PListNode>>::Element *it = data_dict.front(); it; it = it->next()) {
				CharString cs = it->key().utf8();
				uint32_t size = cs.length();

				// Sequence.
				p_stream.push_back(0x30);
				uint32_t seq_size = 2 * (1 + _asn1_size_len(p_len_octets)) + size + it->value()->get_asn1_size(p_len_octets);
				if (p_len_octets > 1) {
					p_stream.push_back(0x80 + p_len_octets);
				}
				for (int i = p_len_octets - 1; i >= 0; i--) {
					uint8_t x = (seq_size >> i * 8) & 0xFF;
					p_stream.push_back(x);
				}
				// Key.
				p_stream.push_back(0x0C);
				if (p_len_octets > 1) {
					p_stream.push_back(0x80 + p_len_octets);
				}
				for (int i = p_len_octets - 1; i >= 0; i--) {
					uint8_t x = (size >> i * 8) & 0xFF;
					p_stream.push_back(x);
				}
				for (uint32_t i = 0; i < size; i++) {
					p_stream.push_back(cs[i]);
				}
				// Value.
				valid = valid && it->value()->store_asn1(p_stream, p_len_octets);
			}
		} break;
	}
	return valid;
}

void PListNode::store_text(String &p_stream, uint8_t p_indent) const {
	// Convert to text XML stream.
	switch (data_type) {
		case PList::PLNodeType::PL_NODE_TYPE_NIL: {
			// Nothing to store.
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DATA: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<data>\n";
			p_stream += String("\t").repeat(p_indent);
			p_stream += data_string + "\n";
			p_stream += String("\t").repeat(p_indent);
			p_stream += "</data>\n";
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DATE: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<date>";
			p_stream += data_string;
			p_stream += "</date>\n";
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_STRING: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<string>";
			p_stream += String::utf8(data_string);
			p_stream += "</string>\n";
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_BOOLEAN: {
			p_stream += String("\t").repeat(p_indent);
			if (data_bool) {
				p_stream += "<true/>\n";
			} else {
				p_stream += "<false/>\n";
			}
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_INTEGER: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<integer>";
			p_stream += itos(data_int);
			p_stream += "</integer>\n";
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_REAL: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<real>";
			p_stream += rtos(data_real);
			p_stream += "</real>\n";
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_ARRAY: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<array>\n";
			for (int i = 0; i < data_array.size(); i++) {
				data_array[i]->store_text(p_stream, p_indent + 1);
			}
			p_stream += String("\t").repeat(p_indent);
			p_stream += "</array>\n";
		} break;
		case PList::PLNodeType::PL_NODE_TYPE_DICT: {
			p_stream += String("\t").repeat(p_indent);
			p_stream += "<dict>\n";
			for (const Map<String, Ref<PListNode>>::Element *it = data_dict.front(); it; it = it->next()) {
				p_stream += String("\t").repeat(p_indent + 1);
				p_stream += "<key>";
				p_stream += it->key();
				p_stream += "</key>\n";
				it->value()->store_text(p_stream, p_indent + 1);
			}
			p_stream += String("\t").repeat(p_indent);
			p_stream += "</dict>\n";
		} break;
	}
}

/*************************************************************************/

PList::PList() {
	root = PListNode::new_dict();
}

PList::PList(const String &p_string) {
	load_string(p_string);
}

bool PList::load_file(const String &p_filename) {
	root = Ref<PListNode>();

	FileAccessRef fb = FileAccess::open(p_filename, FileAccess::READ);
	if (!fb) {
		return false;
	}

	unsigned char magic[8];
	fb->get_buffer(magic, 8);

	if (String((const char *)magic, 8) == "bplist00") {
		ERR_FAIL_V_MSG(false, "PList: Binary property lists are not supported.");
	} else {
		// Load text plist.
		Error err;
		Vector<uint8_t> array = FileAccess::get_file_as_array(p_filename, &err);
		ERR_FAIL_COND_V(err != OK, false);

		String ret;
		ret.parse_utf8((const char *)array.ptr(), array.size());
		return load_string(ret);
	}
}

bool PList::load_string(const String &p_string) {
	root = Ref<PListNode>();

	int pos = 0;
	bool in_plist = false;
	bool done_plist = false;
	List<Ref<PListNode>> stack;
	String key;
	while (pos >= 0) {
		int open_token_s = p_string.find("<", pos);
		if (open_token_s == -1) {
			ERR_FAIL_V_MSG(false, "PList: Unexpected end of data. No tags found.");
		}
		int open_token_e = p_string.find(">", open_token_s);
		pos = open_token_e;

		String token = p_string.substr(open_token_s + 1, open_token_e - open_token_s - 1);
		if (token.is_empty()) {
			ERR_FAIL_V_MSG(false, "PList: Invalid token name.");
		}
		String value;
		if (token[0] == '?' || token[0] == '!') { // Skip <?xml ... ?> and <!DOCTYPE ... >
			int end_token_e = p_string.find(">", open_token_s);
			pos = end_token_e;
			continue;
		}

		if (token.find("plist", 0) == 0) {
			in_plist = true;
			continue;
		}

		if (token == "/plist") {
			in_plist = false;
			done_plist = true;
			break;
		}

		if (!in_plist) {
			ERR_FAIL_V_MSG(false, "PList: Node outside of <plist> tag.");
		}

		if (token == "dict") {
			if (!stack.is_empty()) {
				// Add subnode end enter it.
				Ref<PListNode> dict = PListNode::new_dict();
				dict->data_type = PList::PLNodeType::PL_NODE_TYPE_DICT;
				if (!stack.back()->get()->push_subnode(dict, key)) {
					ERR_FAIL_V_MSG(false, "PList: Can't push subnode, invalid parent type.");
				}
				stack.push_back(dict);
			} else {
				// Add root node.
				if (!root.is_null()) {
					ERR_FAIL_V_MSG(false, "PList: Root node already set.");
				}
				Ref<PListNode> dict = PListNode::new_dict();
				stack.push_back(dict);
				root = dict;
			}
			continue;
		}

		if (token == "/dict") {
			// Exit current dict.
			if (stack.is_empty() || stack.back()->get()->data_type != PList::PLNodeType::PL_NODE_TYPE_DICT) {
				ERR_FAIL_V_MSG(false, "PList: Mismatched </dict> tag.");
			}
			stack.pop_back();
			continue;
		}

		if (token == "array") {
			if (!stack.is_empty()) {
				// Add subnode end enter it.
				Ref<PListNode> arr = PListNode::new_array();
				if (!stack.back()->get()->push_subnode(arr, key)) {
					ERR_FAIL_V_MSG(false, "PList: Can't push subnode, invalid parent type.");
				}
				stack.push_back(arr);
			} else {
				// Add root node.
				if (!root.is_null()) {
					ERR_FAIL_V_MSG(false, "PList: Root node already set.");
				}
				Ref<PListNode> arr = PListNode::new_array();
				stack.push_back(arr);
				root = arr;
			}
			continue;
		}

		if (token == "/array") {
			// Exit current array.
			if (stack.is_empty() || stack.back()->get()->data_type != PList::PLNodeType::PL_NODE_TYPE_ARRAY) {
				ERR_FAIL_V_MSG(false, "PList: Mismatched </array> tag.");
			}
			stack.pop_back();
			continue;
		}

		if (token[token.length() - 1] == '/') {
			token = token.substr(0, token.length() - 1);
		} else {
			int end_token_s = p_string.find("</", pos);
			if (end_token_s == -1) {
				ERR_FAIL_V_MSG(false, vformat("PList: Mismatched <%s> tag.", token));
			}
			int end_token_e = p_string.find(">", end_token_s);
			pos = end_token_e;
			String end_token = p_string.substr(end_token_s + 2, end_token_e - end_token_s - 2);
			if (end_token != token) {
				ERR_FAIL_V_MSG(false, vformat("PList: Mismatched <%s> and <%s> token pair.", token, end_token));
			}
			value = p_string.substr(open_token_e + 1, end_token_s - open_token_e - 1);
		}
		if (token == "key") {
			key = value;
		} else {
			Ref<PListNode> var = nullptr;
			if (token == "true") {
				var = PListNode::new_bool(true);
			} else if (token == "false") {
				var = PListNode::new_bool(false);
			} else if (token == "integer") {
				var = PListNode::new_int(value.to_int());
			} else if (token == "real") {
				var = PListNode::new_real(value.to_float());
			} else if (token == "string") {
				var = PListNode::new_string(value);
			} else if (token == "data") {
				var = PListNode::new_data(value);
			} else if (token == "date") {
				var = PListNode::new_date(value);
			} else {
				ERR_FAIL_V_MSG(false, "PList: Invalid value type.");
			}
			if (stack.is_empty() || !stack.back()->get()->push_subnode(var, key)) {
				ERR_FAIL_V_MSG(false, "PList: Can't push subnode, invalid parent type.");
			}
		}
	}
	if (!stack.is_empty() || !done_plist) {
		ERR_FAIL_V_MSG(false, "PList: Unexpected end of data. Root node is not closed.");
	}
	return true;
}

PackedByteArray PList::save_asn1() const {
	if (root == nullptr) {
		ERR_FAIL_V_MSG(PackedByteArray(), "PList: Invalid PList, no root node.");
	}
	size_t size = root->get_asn1_size(1);
	uint8_t len_octets = 0;
	if (size < 0x80) {
		len_octets = 1;
	} else {
		size = root->get_asn1_size(2);
		if (size < 0xFFFF) {
			len_octets = 2;
		} else {
			size = root->get_asn1_size(3);
			if (size < 0xFFFFFF) {
				len_octets = 3;
			} else {
				size = root->get_asn1_size(4);
				if (size < 0xFFFFFFFF) {
					len_octets = 4;
				} else {
					ERR_FAIL_V_MSG(PackedByteArray(), "PList: Data is too big for ASN.1 serializer, should be < 4 GiB.");
				}
			}
		}
	}

	PackedByteArray ret;
	if (!root->store_asn1(ret, len_octets)) {
		ERR_FAIL_V_MSG(PackedByteArray(), "PList: ASN.1 serializer error.");
	}
	return ret;
}

String PList::save_text() const {
	if (root == nullptr) {
		ERR_FAIL_V_MSG(String(), "PList: Invalid PList, no root node.");
	}

	String ret;
	ret += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	ret += "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n";
	ret += "<plist version=\"1.0\">\n";

	root->store_text(ret, 0);

	ret += "</plist>\n\n";
	return ret;
}

Ref<PListNode> PList::get_root() {
	return root;
}

#endif // MODULE_REGEX_ENABLED
