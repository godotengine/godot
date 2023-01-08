#ifndef RCS_TYPES_H
#define RCS_TYPES_H

#include "core/ustring.h"
#include "core/print_string.h"

#include <cstdlib>
// #include <functional>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <memory>
#include <iterator>
#include <mutex>

#define P2PArrayCopy(from, to) {                                        \
	auto size = from.size();                                            \
	to.clear();                                                         \
	to.resize(size);                                                    \
	for (uint32_t i = 0; i < size; i++){                                \
		to[i] = from[i];                                                \
	}                                                                   \
}

#define P2PVectorCopy(from, to) {                                       \
	auto size = from.size();                                            \
	to.clear();                                                         \
	to.resize(size);                                                    \
	for (uint32_t i = 0; i < size; i++){                                \
		to.set(i, from[i]);                                             \
	}                                                                   \
}

#define P2PArrayWrite(from, to) {                                       \
	auto size = from.size();                                            \
	to.clear();                                                         \
	for (uint32_t i = 0; i < size; i++){                                \
		to.push_back(from[i]);                                          \
	}                                                                   \
}

#define List2Vector(from, to)                                           \
	uint32_t __iter = 0;                                                \
	for (auto E = from.front(); E; E = E->next() && __iter += 1){       \
		to.write[__iter] = E->get();                                    \
	}

// template <class T> class CArray {
// private:
// 	Array();
// 	struct CArrayPrivate {
// 		void* ptr;
// 		int size = 0;
// 		SafeRefCount refcount;
// 	};
// private:
// 	size_t type_size = 0;
// 	mutable CArrayPrivate* inner_array;
// 	void _ref(const CArray& from){
// 		if (!from.inner_array) return;
// 		inner_array = from.inner_array;
// 		inner_array->refcount.ref();
// 	}
// 	void _unref(){
// 		if (!inner_array) return;
// 		if (inner_array->refcount.unref()) delete inner_array;
// 		inner_array = nullptr;
// 	}
// 	void mass_copy(const void* from, void* to, const uint32_t& item_count){
// 		for (uint32_t i = 0; i < item_count; i++){
// 			// *(T*)((size_t)to + (type_size * i)) = *(T*)((size_t)from + (type_size * i));
// 			((T*)to)[i] = ((T*)from)[i];
// 		}
// 	}
// 	void mass_alloc(void* at, const uint32_t& item_count){
// 		for (uint32_t i = 0; i < item_count; i++){
// 			auto item_ptr = ((T*)at)[i];
// 			new (item_ptr) T();
// 		}
// 	}
// 	void mass_delete(void* at, const uint32_t& item_count){
// 		for (uint32_t i = 0; i < item_count; i++){
// 			auto item_ptr = ((T*)at)[i];
// 			item_ptr->~T();
// 		}
// 	}
// public:
// 	_FORCE_INLINE_ CArray(){
// 		type_size = sizeof(T);
// 		inner_array = new CArrayPrivate();
// 		inner_array->refcount.init();
// 	}
// 	_FORCE_INLINE_ CArray(const CArray& from){
// 		type_size = sizeof(T);
// 		_unref();
// 		_ref(from);
// 	}
// 	_FORCE_INLINE_ ~CArray(){
// 		_unref();
// 	}
// 	_FORCE_INLINE_ void reallocate(const uint32_t& new_size){
// 		ERR_FAIL_COND(!inner_array || new_size < 0);
// 		uint32_t delta = (new_size - inner_array->size);
// 		uint32_t retained = new_size > inner_array->size ? inner_array->size : new_size;
// 		delta = delta < 0 ? -delta : delta;
// 		if (delta == 0) return;
// 		void* new_ptr = malloc(type_size * new_size);
// 		if (inner_array->size == 0){
// 			mass_alloc(new_ptr, new_size);
// 		}
// 		else if (new_size == 0){
// 			mass_delete(inner_array->ptr, inner_array->size);
// 		}
// 		else if (new_size > inner_array->size){
// 			mass_copy(inner_array->ptr, new_ptr, retained);
// 			mass_alloc((size_t)new_ptr + (type_size * inner_array->size), delta);
// 		} else  {
// 			mass_copy(inner_array->ptr, new_ptr, retained);
// 			mass_delete((size_t)inner_array->ptr + (type_size * new_size), delta);
// 		}
// 		inner_array->ptr = new_ptr;
// 		inner_array->size = new_size;
// 	}
// 	_FORCE_INLINE_ uint32_t size() const noexcept { 
// 		ERR_FAIL_COND_V(!inner_array, 0);
// 		return inner_array->size;
// 	}
// 	_FORCE_INLINE_ bool empty() const noexcept { return size() == 0; }
// 	_FORCE_INLINE_ T& operator[](const uint32_t& idx){
// 		ERR_FAIL_COND_V(!inner_array, *(new T()));
// 		return (T*)(inner_array->ptr)[idx];
// 	}
// 	_FORCE_INLINE_ const T& operator[](const uint32_t& idx) const {
// 		ERR_FAIL_COND_V(!inner_array, *(new T()));
// 		return (T*)(inner_array->ptr)[idx];
// 	}
// 	_FORCE_INLINE_ void push_back(const T& val){
// 		ERR_FAIL_COND(!inner_array);
// 		reallocate(inner_array->size + 1);
// 		(T*)(inner_array->ptr)[inner_array->size - 1] = val;
// 	}
// };

template <class T> class WrappedVector {
private:
	std::shared_ptr<std::vector<T>> vec_ref;
public:
	_FORCE_INLINE_ WrappedVector() {
		vec_ref = std::make_shared<std::vector<T>>();
	}
	_FORCE_INLINE_ WrappedVector(const WrappedVector& vec) {
		vec_ref = vec.vec_ref;
	}
	_FORCE_INLINE_ WrappedVector(WrappedVector* vec) {
		vec_ref = vec->vec_ref;
	}
	_FORCE_INLINE_ WrappedVector(const std::vector<T>& real_vec) {
		vec_ref = std::make_shared<std::vector<T>>(real_vec);
	}
	_FORCE_INLINE_ WrappedVector(const std::initializer_list<T>& init_list) {
		vec_ref = std::make_shared<std::vector<T>>(init_list);
	}
	_FORCE_INLINE_ bool is_valid() const {
		return vec_ref.operator bool();
	}
	_FORCE_INLINE_ bool is_null() const { return !is_valid(); }
	_FORCE_INLINE_ bool operator==(const std::vector<T>& real_vec) const {
		ERR_FAIL_COND_V(is_null(), false);
		if (size() != real_vec.size()) return false;
		return std::equal(real_vec.begin(), real_vec.end(), vec_ref->begin());
	}
	_FORCE_INLINE_ bool operator==(const WrappedVector& vec) const {
		ERR_FAIL_COND_V(is_null(), false);
		if (size() != vec.size()) return false;
		return *this == (vec.vec_ref);
	}
	_FORCE_INLINE_ bool operator==(const std::initializer_list<T>& init_list) const {
		ERR_FAIL_COND_V(is_null(), false);
		if (size() != init_list.size()) return false;
		return *this == WrappedVector(init_list);
	}
	_FORCE_INLINE_ WrappedVector& operator=(const std::vector<T>& real_vec) {
		ERR_FAIL_COND_V(is_null(), *this);
		vec_ref->operator=(real_vec); return *this;
	}
	_FORCE_INLINE_ WrappedVector& operator=(const WrappedVector& vec) {
		ERR_FAIL_COND_V(is_null(), *this);
		vec_ref->operator=(vec->vec_ref); return *this;
	}
	_FORCE_INLINE_ WrappedVector& operator=(const std::initializer_list<T>& init_list) {
		ERR_FAIL_COND_V(is_null(), *this);
		vec_ref->operator=(std::vector<T>(init_list)); return *this;
	}
	_FORCE_INLINE_ size_t size() const {
		ERR_FAIL_COND_V(is_null(), 0);
		return vec_ref->size();
	}
	_FORCE_INLINE_ void push_back(const T& value) {
		ERR_FAIL_COND(is_null());
		vec_ref->push_back(value);
	}
	_FORCE_INLINE_ T& pop_back() {
		ERR_FAIL_COND_V(is_null(), *(new T()));
		T& re = vec_ref->operator[](size() - 1);
		vec_ref->pop_back();
		return re;
	}
	_FORCE_INLINE_ void insert(const int64_t& at, const T& value) {
		ERR_FAIL_COND(is_null());
		auto index = at;
		auto s = size();
		if (at < 0) index = s + at;
		ERR_FAIL_COND(index < 0);
		ERR_FAIL_COND(index >= s);
		vec_ref->insert(vec_ref->begin() + index, value);
	}
	_FORCE_INLINE_ T& operator[](int idx) {
		ERR_FAIL_COND_V(is_null(), *(new T()));
		auto index = idx;
		auto s = size();
		if (idx < 0) index = s + idx;
		ERR_FAIL_COND_V((index < 0), *(new T()));
		ERR_FAIL_COND_V(index >= s, *(new T()));
		return vec_ref->operator[](index);
	}
	_FORCE_INLINE_ operator bool() const {
		ERR_FAIL_COND_V(is_null(), false);
		return !empty();
	}
	_FORCE_INLINE_ const T& operator[](int idx) const {
		ERR_FAIL_COND_V(is_null(), *(new T()));
		auto index = idx;
		auto s = size();
		if (idx < 0) index = s + idx;
		ERR_FAIL_COND_V((index < 0), *(new T()));
		ERR_FAIL_COND_V(index >= s, *(new T()));
		return vec_ref->operator[](index);
	}
	// _FORCE_INLINE_ T& operator[](int64_t idx) {
	// 	auto index = idx;
	// 	auto s = size();
	// 	ERR_FAIL_COND_V((index < 0), *(new T()));
	// 	ERR_FAIL_COND(index >= s);
	// 	return vec_ref->operator[](index);
	// }
	_FORCE_INLINE_ int64_t find(const T& what, const int64_t& from = 0) const {
		ERR_FAIL_COND_V(is_null(), -1);
		if (empty()) return -1;
		int64_t iter = from;
		auto s = size();
		ERR_FAIL_COND_V_MSG(s < 0, -1, "Broken vector detected");
		// print_verbose(String("Finding item from index: ") + itos(from));
		// print_verbose(String("Finding item with whole size: ") + itos(s));
		for (; iter < s; iter += 1) {
			if (vec_ref->operator[](iter) == what) return iter;
		}
		return -1;
	}
	_FORCE_INLINE_ void remove(const int64_t& at) {
		ERR_FAIL_COND(is_null());
		if (empty()) return;
		auto index = at;
		auto s = size();
		if (at < 0) index = s + at;
		ERR_FAIL_COND(index < 0);
		ERR_FAIL_COND(index >= s);
		// print_verbose(String("Removing item at raw index: ") + itos(at));
		// print_verbose(String("Removing item at index: ") + itos(index));
		// print_verbose(String("Removing item with whole size: ") + itos(s));
		vec_ref->erase(vec_ref->begin() + index);
	}
	_FORCE_INLINE_ void erase(const T& what) {
		ERR_FAIL_COND(is_null());
		if (empty()) return;
		// print_verbose(String("Erasing..."));
		auto index = find(what);
		if (index == -1) return;
		// print_verbose(String("Erasing item at index: ") + itos(index));
		remove(index);
	}
	_FORCE_INLINE_ void change_pointer(const WrappedVector& vec){
		vec_ref = vec.vec_ref;
	}
	_FORCE_INLINE_ void change_pointer(const std::shared_ptr<std::vector<T>>& ptr){
		vec_ref = ptr;
	}
	_FORCE_INLINE_ void make_unique() {
		change_pointer(std::make_shared<std::vector<T>>());
	}
	_FORCE_INLINE_ void clear() {
		ERR_FAIL_COND(is_null());
		vec_ref->clear();
	}
	_FORCE_INLINE_ bool empty() const {
		ERR_FAIL_COND_V(is_null(), true);
		return vec_ref->empty();
	}
	_FORCE_INLINE_ void copy(const WrappedVector& vec){
		ERR_FAIL_COND(is_null());
		clear();
		auto other = vec.vec_ref;
		for (auto iter : *other){
			push_back(iter);
		}
	}
	_FORCE_INLINE_ void copy(const std::vector<T>& vec){
		ERR_FAIL_COND(is_null());
		clear();
		for (auto iter : vec){
			push_back(iter);
		}
	}
	_FORCE_INLINE_ void duplicate(const WrappedVector& vec){
		make_unique();
		copy(vec);
	}
	_FORCE_INLINE_ void duplicate(const std::vector<T>& vec){
		make_unique();
		copy(vec);
	}
	_FORCE_INLINE_ WrappedVector<T> make_duplicate() const{
		WrappedVector<T> new_dup();
		new_dup.duplicate(*this);
		return new_dup;
	}
	_FORCE_INLINE_ std::weak_ptr<std::vector<T>> get_weakptr() const {
		std::weak_ptr<std::vector<T>> ptr = vec_ref;
		return ptr;
	}
};

/*  
**  Safe-Wrapped Contigous Stack
**  A closed-circuit, thread-safe, reference-counted, semi-static size container
 */
template <class T> class SWContigousStack {
private:
	struct SWContigousStackPrivate;
public:
	class Element {
	private:
		friend class SWContigousStack<T>;
		friend struct SWContigousStack<T>::SWContigousStackPrivate;
		T value;
		Element* prev_ptr;
		Element* next_ptr;
	public:
		_FORCE_INLINE_ Element() {
			value = T();
			prev_ptr = nullptr;
			next_ptr = nullptr;
		}
		_FORCE_INLINE_ ~Element(){
			// print_verbose(itos((uint64_t)this) + String(" deallocated"));
		}
		_FORCE_INLINE_ void flush_back(const Element* start_at) {
			if (!next_ptr || next_ptr == start_at) return;
			next_ptr->flush_back(start_at);
			delete next_ptr;
			next_ptr = nullptr;
		}
		_FORCE_INLINE_ void flush_front(const Element* start_at) {
			if (!prev_ptr || prev_ptr == start_at) return;
			prev_ptr->flush_front(start_at);
			delete prev_ptr;
			prev_ptr = nullptr;
		}
		_FORCE_INLINE_ const Element* next() const {
			return next_ptr;
		};
		_FORCE_INLINE_ Element* next() {
			return next_ptr;
		};
		_FORCE_INLINE_ const Element* prev() const {
			return prev_ptr;
		};
		_FORCE_INLINE_ Element* prev() {
			return prev_ptr;
		};
		_FORCE_INLINE_ const T& operator*() const {
			return value;
		};
		_FORCE_INLINE_ const T* operator->() const {
			return &value;
		};
		_FORCE_INLINE_ T& operator*() {
			return value;
		};
		_FORCE_INLINE_ T* operator->() {
			return &value;
		};
		_FORCE_INLINE_ T& get() {
			return value;
		};
		_FORCE_INLINE_ const T& get() const {
			return value;
		};
		_FORCE_INLINE_ void set(const T& p_value) {
			value = (T&)p_value;
		};
	};
private:
	struct SWContigousStackPrivate {
	private:
		mutable std::recursive_mutex iter_lock;
		mutable std::recursive_mutex main_lock;
		Element* entry;
		Element* iterating;
		bool allocated = false;
	public:
		uint32_t max_size = 128;
		mutable uint32_t usage = 0;

		_FORCE_INLINE_ SWContigousStackPrivate() {
			allocate();
		}
		_FORCE_INLINE_ SWContigousStackPrivate(const uint32_t& pre_alloc) {
			if (pre_alloc < 2) max_size = 16;
			else max_size = pre_alloc;
			allocate();
		}
		_FORCE_INLINE_ ~SWContigousStackPrivate() {
			std::lock_guard<std::recursive_mutex> guard(main_lock);
			deallocate();
			// print_verbose(String("SWContigousStackPrivate deallocated"));
		}
		_FORCE_INLINE_ void allocate() {
			Element* curr = new Element();
			Element* last = new Element();
			curr->prev_ptr = last;
			last->next_ptr = curr;
			entry = last;
			iterating = entry;
			for (uint32_t i = 2; i < max_size; i++) {
				last = curr;
				curr = new Element();
				last->next_ptr = curr;
				curr->prev_ptr = last;
			}
			curr->next_ptr = entry;
			entry->prev_ptr = curr;
			allocated = true;
		}
		_FORCE_INLINE_ void deallocate() {
			entry->flush_back(entry);
			delete entry;
			entry = nullptr; iterating = nullptr;
			usage = 0;
			allocated = false;
		}
		_FORCE_INLINE_ void resize(const uint32_t& new_size) {
			std::lock_guard<std::recursive_mutex> guard(main_lock);
			if (new_size < 2) max_size = 16;
			else max_size = new_size;
			deallocate();
			allocate();
		}
		_FORCE_INLINE_ uint32_t get_max_size() const { return max_size; }
		_FORCE_INLINE_ uint32_t get_usage() const { return usage; }
		_FORCE_INLINE_ const Element* first() const {
			return entry;
		}
		_FORCE_INLINE_ Element* first() {
			return entry;
		}
		_FORCE_INLINE_ Element* next() {
			std::lock_guard<std::recursive_mutex> guard(iter_lock);
			auto re = iterating;
			iterating = iterating->next();
			usage += 1;
			if (usage > max_size) {
				usage = max_size;
				WARN_PRINT_ONCE("SWContigousStack overflowed.");
			}
			return re;
		}
		_FORCE_INLINE_ void decrease_usage(){
			// WARN_PRINT_ONCE("decrease_usage called");
			std::lock_guard<std::recursive_mutex> guard(iter_lock);
			usage -= 1;
			if (usage < 0) usage = 0;
		}
		_FORCE_INLINE_ const Element* next() const {
			std::lock_guard<std::recursive_mutex> guard(iter_lock);
			auto re = iterating;
			iterating = iterating->next();
			usage += 1;
			if (usage > max_size) {
				usage = max_size;
				WARN_PRINT_ONCE("SWContigousStack overflowed.");
			}
			return re;
		}
		_FORCE_INLINE_ const Element* iterate(const Element* from) const {
			// if (from == nullptr){
			// 	return (const Element*)entry;
			// }
			const SWContigousStack<T>::Element* re = from->next();
			if (re == entry) return nullptr;
			return re;
		}
		_FORCE_INLINE_ Element* iterate(Element* from) {
			// if (from == nullptr){
			// 	return entry;
			// }
			auto re = from->next();
			if (re == entry) return nullptr;
			return re;
		}
		_FORCE_INLINE_ Element* find(const T& val) {
			for (auto E = first(); E; E = iterate(E)) {
				if (val == E->get()) return E;
			}
			return nullptr;
		}
		_FORCE_INLINE_ const Element* find(const T& val) const {
			for (const SWContigousStack<T>::Element* E = first(); E; E = iterate(E)) {
				if (val == E->get()) return E;
			}
			return nullptr;
		}
		_FORCE_INLINE_ Element* patch(Element* elem) {
			if (!elem) return;
			std::lock_guard<std::recursive_mutex> guard(main_lock);
			// auto stich = new Element();
			// stich->next_ptr = elem->next_ptr;
			// stich->prev_ptr = elem->prev_ptr;
			// elem->prev_ptr->next_ptr = stich;
			// elem->next_ptr->prev_ptr = stich;
			// if (elem == entry) entry = stich;
			// delete elem;
			elem->get() = T();
			return elem;
		}
		_FORCE_INLINE_ void patch_counted(Element* elem) {
			if (!elem) return;
			std::lock_guard<std::recursive_mutex> guard(main_lock);
			usage -= 1;
			if (usage < 0) usage = 0;
			elem->get() = T();
			// return elem;
		}
	};
	std::shared_ptr<SWContigousStackPrivate> data_ptr;
public:
	_FORCE_INLINE_ SWContigousStack() {
		// allocate();
		data_ptr = std::make_shared<SWContigousStackPrivate>();
	}
	_FORCE_INLINE_ ~SWContigousStack(){
		// print_verbose(String("data_ptr use count: ") + itos(data_ptr.use_count()));
		// data_ptr.reset();
		// print_verbose(String("data_ptr use count: ") + itos(data_ptr.use_count()));
	}
	_FORCE_INLINE_ SWContigousStack(const std::initializer_list<T>& ilist) {
		auto size = ilist.size();
		if (size < 2) {
			WARN_PRINT("initializer_list does not reach the sufficient size");
			data_ptr = std::make_shared<SWContigousStackPrivate>();
			return;
		}
		data_ptr = std::make_shared<SWContigousStackPrivate>(size);
		Element* E = data_ptr->first();
		for (T elem : ilist) {
			E->get() = elem;
			E = E->next();
		}
		data_ptr->usage = size;
	}
	_FORCE_INLINE_ SWContigousStack(const SWContigousStack& another) {
		data_ptr = another.data_ptr;
	}
	_FORCE_INLINE_ SWContigousStack(const uint32_t& pre_alloc) {
		data_ptr = std::make_shared<SWContigousStackPrivate>(pre_alloc);
	}
	_FORCE_INLINE_ void resize(const uint32_t& amount) {
		data_ptr->resize(amount);
	}
	_FORCE_INLINE_ Element* get_entry() {
		return data_ptr->first();
	}
	_FORCE_INLINE_ const Element* get_entry() const {
		return data_ptr->first();
	}
	_FORCE_INLINE_ uint32_t get_usage() const {
		return data_ptr->get_usage();
	}
	_FORCE_INLINE_ uint32_t get_max_size() const {
		return data_ptr->get_max_size();
	}
	_FORCE_INLINE_ T& operator[](const uint32_t& index) {
		uint32_t i = 0;
		auto elem = get_entry();
		while (true) {
			if (i == index) break;
			elem = elem->next();
			i += 1;
		}
		return elem->get();
	}
	_FORCE_INLINE_ T operator[](const uint32_t& index) const {
		uint32_t i = 0;
		auto elem = get_entry();
		while (true) {
			if (i == index) break;
			elem = elem->next();
			i += 1;
		}
		return elem->get();
	}
	_FORCE_INLINE_ Element* next() {
		return data_ptr->next();
	}
	_FORCE_INLINE_ const Element* next() const {
		return data_ptr->next();
	}
	_FORCE_INLINE_ T& fetch() {
		return next()->get();
	}
	// _FORCE_INLINE_ void decrease_usage(){ data_ptr->decrease_usage(); }
	_FORCE_INLINE_ Element* iterate(Element* from = nullptr) {
		if (!from) return get_entry();
		return data_ptr->iterate(from);
	}
	// _FORCE_INLINE_ bool empty() noexcept { return get_usage() == 0; }
	// _FORCE_INLINE_ uint32_t size() noexcept { return get_usage(); }
	_FORCE_INLINE_ SWContigousStack duplicate() const {
		SWContigousStack<T> re(get_max_size());
		for (uint32_t i = 0, size = get_max_size(); i < size; i++) {
			re[i] = operator[](i);
		}
		return re;
	}
	_FORCE_INLINE_ SWContigousStack& operator=(const SWContigousStack& to) {
		data_ptr = to.duplicate().data_ptr;
		return *this;
	}
	_FORCE_INLINE_ Element* patch(Element* elem) {
		return data_ptr->patch(elem);
	}
	_FORCE_INLINE_ const Element* find(const T& val) const {
		return data_ptr->find(val);
	}
	_FORCE_INLINE_ Element* find(const T& val) {
		return data_ptr->find(val);
	}
	_FORCE_INLINE_ Element* replace(const T& val) {
		auto search_res = find(val);
		if (!search_res) return nullptr;
		return patch((Element*)search_res);
	}
	_FORCE_INLINE_ void erase(const T& val) {
		auto search_res = find(val);
		if (!search_res) return;
		data_ptr->patch_counted((Element*)search_res);
	}
private:
	_FORCE_INLINE_ void async_erase_internal(const T& val) {
		std::shared_ptr<SWContigousStackPrivate> hold_ref = data_ptr;
		erase(val);
	}
public:
	_FORCE_INLINE_ void async_erase(const T& val) {
		std::thread(&SWContigousStack<T>::async_erase_internal, this, val).detach();
	}
	_FORCE_INLINE_ uint32_t estimate_size() const {
		auto class_size       = sizeof(SWContigousStack<T>);
		auto data_size        = sizeof(SWContigousStackPrivate);
		auto elem_size        = sizeof(SWContigousStack<T>::Element);
		auto stack_chain_size = elem_size * get_max_size();
		return class_size + data_size + stack_chain_size;
	}
};

#define TIMER_MSEC() \
	TimerRAII __timer(__FILE__, FUNCTION_STR, __LINE__, false)
#define TIMER_USEC() \
	TimerRAII __timer(__FILE__, FUNCTION_STR, __LINE__, true)

class TimerRAII{
private:
	std::chrono::steady_clock::time_point start;
	bool print_additional_info;
	bool use_micro;
	const char* function;
	const char* file;
	uint32_t line;
	String postfix;
public:
	_FORCE_INLINE_ void decide_postfix(){
		postfix = (use_micro ? " usec" : " msec");
	}
	_FORCE_INLINE_ TimerRAII(){
		this->print_additional_info = false;
		this->use_micro = false;
		decide_postfix();
		this->start = std::chrono::high_resolution_clock::now();
	}
	_FORCE_INLINE_ TimerRAII(const char* file, const char* function, const uint32_t& line, const bool& micros = false){
		this->file = file;
		this->function = function;
		this->line = line;
		this->print_additional_info = true;
		this->use_micro = micros;
		decide_postfix();
		this->start = std::chrono::high_resolution_clock::now();
	}
	_FORCE_INLINE_ ~TimerRAII(){
		auto end = std::chrono::high_resolution_clock::now();;
		uint64_t count = 0;
		if (use_micro) {
			count = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		}
		else {
			count = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}
		print_verbose(String("TimerRAII concluded after: ") + itos(count) + postfix);
		if (print_additional_info){
			print_verbose(String("   at (") + String(file) + String(":") + String(function) + String(") - ") + itos(line));
		}
	}
};

#endif

