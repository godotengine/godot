#ifndef GODOT_JS_SPX_UTIL_H
#define GODOT_JS_SPX_UTIL_H

#include <vector>

template <typename T>
class ObjectPool {
public:
    ObjectPool(size_t size) {
        for (size_t i = 0; i < size; ++i) {
            pool.push_back(new T());
        }
    }

    ~ObjectPool() {
        for (auto obj : pool) {
            delete obj;
        }
    }

    T* acquire() {
        if (pool.empty()) {
            return new T();
        } else {
            T* obj = pool.back();
            pool.pop_back();
            return obj;
        }
    }

    void release(T* obj) {
        pool.push_back(obj);
    }

private:
    std::vector<T*> pool;
};

#endif // GODOT_JS_SPX_UTIL_H