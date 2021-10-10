/**************************************************************************/
/*  packed_data_container.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "packed_data_container.h"

#include "core/core_string_names.h"
#include "core/io/marshalls.h"

Variant PackedDataContainer::getvar(const Variant &p_key, bool *r_valid) const {
	bool err = false;
	Variant ret = _key_at_ofs(0, p_key, err);
	if (r_valid) {
		*r_valid = !err;
	}
	return ret;
}

int PackedDataContainer::size() const {
	return _size(0);
};

Variant PackedDataContainer::_iter_init_ofs(const Array &p_iter, uint32_t p_offset) {
	Array ref = p_iter;
	uint32_t size = _size(p_offset);
	if (size == 0 || ref.size() != 1) {
		return false;
	} else {
		ref[0] = 0;
		return true;
	}
}

Variant PackedDataContainer::_iter_next_ofs(const Array &p_iter, uint32_t p_offset) {
	Array ref = p_iter;
	int size = _size(p_offset);
	if (ref.size() != 1) {
		return false;
	}
	int pos = ref[0];
	if (pos < 0 || pos >= size) {
		return false;
	}
	pos += 1;
	ref[0] = pos;
	return pos != size;
}

Variant PackedDataContainer::_iter_get_ofs(const Variant &p_iter, uint32_t p_offset) {
	int size = _size(p_offset);
	int pos = p_iter;
	if (pos < 0 || pos >= size) {
		return Variant();
	}

	PoolVector<uint8_t>::Read rd = data.read();
	const uint8_t *r = &rd[p_offset];
	uint32_t type = decode_uint32(r);

	bool err = false;
	if (type == TYPE_ARRAY) {
		uint32_t vpos = decode_uint32(rd.ptr() + p_offset + 8 + pos * 4);
		return _get_at_ofs(vpos, rd.ptr(), err);

	} else if (type == TYPE_DICT) {
		uint32_t vpos = decode_uint32(rd.ptr() + p_offset + 8 + pos * 12 + 4);
		return _get_at_ofs(vpos, rd.ptr(), err);
	} else {
		ERR_FAIL_V(Variant());
	}
}

Variant PackedDataContainer::_get_at_ofs(uint32_t p_ofs, const uint8_t *p_buf, bool &err) const {
	ERR_FAIL_COND_V(p_ofs + 4 > (uint32_t)data.size(), Variant());
	uint32_t type = decode_uint32(p_buf + p_ofs);

	if (type == TYPE_ARRAY || type == TYPE_DICT) {
		Ref<PackedDataContainerRef> pdcr = memnew(PackedDataContainerRef);
		Ref<PackedDataContainer> pdc = Ref<PackedDataContainer>((PackedDataContainer *)this);

		pdcr->from = pdc;
		pdcr->offset = p_ofs;
		return pdcr;
	} else {
		Variant v;
		Error rerr = decode_variant(v, p_buf + p_ofs, datalen - p_ofs, nullptr, false);

		if (rerr != OK) {
			err = true;
			ERR_FAIL_COND_V_MSG(err != OK, Variant(), "Error when trying to decode Variant.");
		}
		return v;
	}
}

uint32_t PackedDataContainer::_type_at_ofs(uint32_t p_ofs) const {
	ERR_FAIL_COND_V(p_ofs + 4 > (uint32_t)data.size(), 0);
	PoolVector<uint8_t>::Read rd = data.read();
	ERR_FAIL_COND_V(!rd.ptr(), 0);
	const uint8_t *r = &rd[p_ofs];
	uint32_t type = decode_uint32(r);

	return type;
};

int PackedDataContainer::_size(uint32_t p_ofs) const {
	ERR_FAIL_COND_V(p_ofs + 4 > (uint32_t)data.size(), 0);
	PoolVector<uint8_t>::Read rd = data.read();
	ERR_FAIL_COND_V(!rd.ptr(), 0);
	const uint8_t *r = &rd[p_ofs];
	uint32_t type = decode_uint32(r);

	if (type == TYPE_ARRAY) {
		uint32_t len = decode_uint32(r + 4);
		return len;

	} else if (type == TYPE_DICT) {
		uint32_t len = decode_uint32(r + 4);
		return len;
	};

	return -1;
};

Variant PackedDataContainer::_key_at_ofs(uint32_t p_ofs, const Variant &p_key, bool &err) const {
	ERR_FAIL_COND_V(p_ofs + 4 > (uint32_t)data.size(), Variant());
	PoolVector<uint8_t>::Read rd = data.read();
	if (!rd.ptr()) {
		err = true;
		ERR_FAIL_COND_V(!rd.ptr(), Variant());
	}
	const uint8_t *r = &rd[p_ofs];
	uint32_t type = decode_uint32(r);

	if (type == TYPE_ARRAY) {
		if (p_key.is_num()) {
			int idx = p_key;
			int len = decode_uint32(r + 4);
			if (idx < 0 || idx >= len) {
				err = true;
				return Variant();
			}
			uint32_t ofs = decode_uint32(r + 8 + 4 * idx);
			return _get_at_ofs(ofs, rd.ptr(), err);

		} else {
			err = true;
			return Variant();
		}

	} else if (type == TYPE_DICT) {
		uint32_t hash = p_key.hash();
		uint32_t len = decode_uint32(r + 4);

		bool found = false;
		for (uint32_t i = 0; i < len; i++) {
			uint32_t khash = decode_uint32(r + 8 + i * 12 + 0);
			if (khash == hash) {
				Variant key = _get_at_ofs(decode_uint32(r + 8 + i * 12 + 4), rd.ptr(), err);
				if (err) {
					return Variant();
				}
				if (key == p_key) {
					//key matches, return value
					return _get_at_ofs(decode_uint32(r + 8 + i * 12 + 8), rd.ptr(), err);
				}
				found = true;
			} else {
				if (found) {
					break;
				}
			}
		}

		err = true;
		return Variant();

	} else {
		err = true;
		return Variant();
	}
}

uint32_t PackedDataContainer::_pack(const Variant &p_data, Vector<uint8_t> &tmpdata, Map<String, uint32_t> &string_cache) {
	switch (p_data.get_type()) {
		case Variant::STRING: {
			String s = p_data;
			if (string_cache.has(s)) {
				return string_cache[s];
			}

			string_cache[s] = tmpdata.size();

			FALLTHROUGH;
		};
		case Variant::NIL:
		case Variant::BOOL:
		case Variant::INT:
		case Variant::REAL:
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
		case Variant::TRANSFORM2D:
		case Variant::PLANE:
		case Variant::QUAT:
		case Variant::AABB:
		case Variant::BASIS:
		case Variant::TRANSFORM:
		case Variant::POOL_BYTE_ARRAY:
		case Variant::POOL_INT_ARRAY:
		case Variant::POOL_REAL_ARRAY:
		case Variant::POOL_STRING_ARRAY:
		case Variant::POOL_VECTOR2_ARRAY:
		case Variant::POOL_VECTOR3_ARRAY:
		case Variant::POOL_COLOR_ARRAY:
		case Variant::NODE_PATH: {
			uint32_t pos = tmpdata.size();
			int len;
			encode_variant(p_data, nullptr, len, false);
			tmpdata.resize(tmpdata.size() + len);
			encode_variant(p_data, &tmpdata.write[pos], len, false);
			return pos;

		} break;
		// misc types
		case Variant::_RID:
		case Variant::OBJECT: {
			return _pack(Variant(), tmpdata, string_cache);
		} break;
		case Variant::DICTIONARY: {
			Dictionary d = p_data;
			//size is known, use sort
			uint32_t pos = tmpdata.size();
			int len = d.size();
			tmpdata.resize(tmpdata.size() + len * 12 + 8);
			encode_uint32(TYPE_DICT, &tmpdata.write[pos + 0]);
			encode_uint32(len, &tmpdata.write[pos + 4]);

			List<Variant> keys;
			d.get_key_list(&keys);
			List<DictKey> sortk;

			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
				DictKey dk;
				dk.hash = E->get().hash();
				dk.key = E->get();
				sortk.push_back(dk);
			}

			sortk.sort();

			int idx = 0;
			for (List<DictKey>::Element *E = sortk.front(); E; E = E->next()) {
				encode_uint32(E->get().hash, &tmpdata.write[pos + 8 + idx * 12 + 0]);
				uint32_t ofs = _pack(E->get().key, tmpdata, string_cache);
				encode_uint32(ofs, &tmpdata.write[pos + 8 + idx * 12 + 4]);
				ofs = _pack(d[E->get().key], tmpdata, string_cache);
				encode_uint32(ofs, &tmpdata.write[pos + 8 + idx * 12 + 8]);
				idx++;
			}

			return pos;

		} break;
		case Variant::ARRAY: {
			Array a = p_data;
			//size is known, use sort
			uint32_t pos = tmpdata.size();
			int len = a.size();
			tmpdata.resize(tmpdata.size() + len * 4 + 8);
			encode_uint32(TYPE_ARRAY, &tmpdata.write[pos + 0]);
			encode_uint32(len, &tmpdata.write[pos + 4]);

			for (int i = 0; i < len; i++) {
				uint32_t ofs = _pack(a[i], tmpdata, string_cache);
				encode_uint32(ofs, &tmpdata.write[pos + 8 + i * 4]);
			}

			return pos;

		} break;

		default: {
		}
	}

	return OK;
}

Error PackedDataContainer::pack(const Variant &p_data) {
	Vector<uint8_t> tmpdata;
	Map<String, uint32_t> string_cache;
	_pack(p_data, tmpdata, string_cache);
	datalen = tmpdata.size();
	data.resize(tmpdata.size());
	PoolVector<uint8_t>::Write w = data.write();
	memcpy(w.ptr(), tmpdata.ptr(), tmpdata.size());

	return OK;
}

void PackedDataContainer::_set_data(const PoolVector<uint8_t> &p_data) {
	data = p_data;
	datalen = data.size();
}

PoolVector<uint8_t> PackedDataContainer::_get_data() const {
	return data;
}

Variant PackedDataContainer::_iter_init(const Array &p_iter) {
	return _iter_init_ofs(p_iter, 0);
}

Variant PackedDataContainer::_iter_next(const Array &p_iter) {
	return _iter_next_ofs(p_iter, 0);
}
Variant PackedDataContainer::_iter_get(const Variant &p_iter) {
	return _iter_get_ofs(p_iter, 0);
}

void PackedDataContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_data"), &PackedDataContainer::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &PackedDataContainer::_get_data);
	ClassDB::bind_method(D_METHOD("_iter_init"), &PackedDataContainer::_iter_init);
	ClassDB::bind_method(D_METHOD("_iter_get"), &PackedDataContainer::_iter_get);
	ClassDB::bind_method(D_METHOD("_iter_next"), &PackedDataContainer::_iter_next);
	ClassDB::bind_method(D_METHOD("pack", "value"), &PackedDataContainer::pack);
	ClassDB::bind_method(D_METHOD("size"), &PackedDataContainer::size);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "__data__"), "_set_data", "_get_data");
}

PackedDataContainer::PackedDataContainer() {
	datalen = 0;
}

//////////////////

Variant PackedDataContainerRef::_iter_init(const Array &p_iter) {
	return from->_iter_init_ofs(p_iter, offset);
}

Variant PackedDataContainerRef::_iter_next(const Array &p_iter) {
	return from->_iter_next_ofs(p_iter, offset);
}
Variant PackedDataContainerRef::_iter_get(const Variant &p_iter) {
	return from->_iter_get_ofs(p_iter, offset);
}

bool PackedDataContainerRef::_is_dictionary() const {
	return from->_type_at_ofs(offset) == PackedDataContainer::TYPE_DICT;
};

void PackedDataContainerRef::_bind_methods() {
	ClassDB::bind_method(D_METHOD("size"), &PackedDataContainerRef::size);
	ClassDB::bind_method(D_METHOD("_iter_init"), &PackedDataContainerRef::_iter_init);
	ClassDB::bind_method(D_METHOD("_iter_get"), &PackedDataContainerRef::_iter_get);
	ClassDB::bind_method(D_METHOD("_iter_next"), &PackedDataContainerRef::_iter_next);
	ClassDB::bind_method(D_METHOD("_is_dictionary"), &PackedDataContainerRef::_is_dictionary);
}

Variant PackedDataContainerRef::getvar(const Variant &p_key, bool *r_valid) const {
	bool err = false;
	Variant ret = from->_key_at_ofs(offset, p_key, err);
	if (r_valid) {
		*r_valid = !err;
	}
	return ret;
}

int PackedDataContainerRef::size() const {
	return from->_size(offset);
};

PackedDataContainerRef::PackedDataContainerRef() {
	offset = 0;
}
