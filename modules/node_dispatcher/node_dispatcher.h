#ifndef NODE_DISPATCHER_H
#define NODE_DISPATCHER_H

#include <thread>
#include <mutex>
#include <string>
#include <cmath>
#include <chrono>

#include "core/vector.h"
#include "core/object.h"
#include "core/func_ref.h"
#include "core/project_settings.h"
#include "scene/main/node.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include "core/string_name.h"
#include "core/array.h"
#include "core/dictionary.h"
#include "modules/hub/hub.h"

typedef std::thread cthread;
typedef std::mutex cmutex;
typedef std::string cstr;

class NodeDispatcher : public Object {
	GDCLASS(NodeDispatcher, Object)
private:
	bool is_active = true;
	bool auto_thread_cancel = false;
	int max_thread_count = 2;
	int min_thread_count = 2;
	int usable_threads = 2;
	float max_idle_time = 1.0;
	float max_physics_time = 1.0;

	float idle_delta = 0.0;
	float physics_delta = 0.0;

	Vector<Node*> node_pool;
	Vector<Ref<FuncRef>> exec_pool;
	Vector<Array> args_pool;

	cmutex primary_lock;
	cmutex fref_lock;
	// int internal_count_debug = 0;

	inline static int clampi(const int& value, const int& from, const int& to) { return (value < from ? from : (value > to ? to : value));}
	inline static int min(const int& a, const int& b) { return a < b ? a : b; }
	static bool is_node_valid(Object *n);
	static int get_nearest_two_exponent(const int& num);
protected:
	static NodeDispatcher *singleton;
	static void _bind_methods();
	// void _notification(int p_notification);

	void execute_external();
	void dispatch_all(const bool& is_physics);
	void dispatch_idle();
	void dispatch_physics();
	void dispatch_internal(const int& tid, const bool& is_physics);
	cthread *dispatch(const int& tid, const bool& is_physics);
public:
	NodeDispatcher();
	~NodeDispatcher();

	static NodeDispatcher* get_singleton() { return singleton; }
	friend class SceneTree;

	// Exposed methods
	bool add_node(Node* node);
	bool remove_node(const int& instance_id);
	void queue_execution(const Ref<FuncRef>& function, Array args);

	void set_active(const bool& trigger);
	inline bool get_active() const { return is_active; }
	
	void set_usable_thread_count(const int& new_thread_count);
	inline int get_usable_thread_count() const { return usable_threads; }

	inline int get_min_thread_count() const { return min_thread_count; }
	inline int get_max_thread_count() const { return max_thread_count; }
	inline int get_node_count() const { return node_pool.size(); }
	inline int get_queued_execution_count() const { return exec_pool.size(); }
	Array get_all_nodes();
	Dictionary get_all_nodes_by_handler();
};

#endif