#include "detour.h"
#ifndef CROWD_H
#define CROWD_H

class dtCrowd;
class dtQueryFilter;
class DetourNavigationQuery;
struct dtCrowdAgent;
#if 0
class Crowd;
#endif
class DetourCrowdManager : public Node {
	GDCLASS(DetourCrowdManager, Object);
	class CrowdAgent;
	struct AgentData {
		Spatial *obj;
		int mode;
		int id;
		float radius;
		float height;
		float max_accel;
		float max_speed;
		int filter_id;
		int oa_id;
		bool send_signals;
		AgentData();
		~AgentData();
	};
	Vector<AgentData *> agents;
	friend class CrowdAgent;
	dtCrowd *crowd;
	bool dirty, initialized;
	static void _bind_methods();
	DetourNavigationQuery *query;
	Transform transform;

protected:
	void process_agent(dtCrowdAgent *agent);
	void _notification(int p_what);
	int max_agents;
	float max_agent_radius;
	const dtCrowdAgent *get_detour_crowd_agent(int agent);
	const dtQueryFilter *get_detour_query_filter(int query_filter);
	bool create_crowd();
	dtCrowd *get_crowd() const { return crowd; }
	Ref<DetourNavigationMesh> navmesh;
	void update_crowd(float delta);

public:
	typedef uint64_t polyref_t;
	DetourCrowdManager();
	~DetourCrowdManager();
	void add_agent(Object *agent, int mode = 0, bool signals = false);
	void remove_agent(Object *agent);
	void clear_agent_list();
	Spatial *get_agent_obj(int id) const { return agents[id]->obj; }
	int get_agent_mode(int id) const { return agents[id]->mode; }
	int get_agent_count() const { return agents.size(); }
	void set_agent_target_position(int id, const Vector3 &position);
	void set_target(const Vector3 &position);
	// void set_crowd_target_node(const Spatial *target);
	void set_velocity(const Vector3 &position);
	void reset_target();
	void set_max_agents(int max_agents);
	int get_max_agents() const { return max_agents; }
	void set_max_agent_radius(float radius);
	float get_max_agent_radius() const { return max_agent_radius; }
	void set_navigation_mesh(const Ref<DetourNavigationMesh> &mesh, const Transform &xform);
	Transform get_navigation_transform() const { return transform; }
	Ref<DetourNavigationMesh> get_navigation_mesh() const;
	/* Query filter */
	void set_include_flags(int filter_id, unsigned short flags);
	unsigned short get_include_flags(int filter_id);
	void set_exclude_flags(int query_filter, unsigned short flags);
	unsigned short get_exclude_flags(int query_filter);
	void set_area_cost(int filter_id, int area_id, float cost);
	float get_area_cost(int filter_id, int area_id);
	int get_num_query_filters() const;
	int get_num_aread(int query_filter) const;
	/* Queries, FIXME */
	Vector3 nearest_point(const Vector3 &point, int query_filter);
	Vector3 _nearest_point(const Vector3 &point, int query_filter, polyref_t *nearest_ref);
	Vector3 move_along_surface(const Vector3 &start, const Vector3 &end, int query_filter, int maxVisited = 3);
	Vector<Vector3> find_path(const Vector3 &start, const Vector3 &end, int query_filter);
	Vector3 get_random_point(int query_filter);
	Vector3 get_random_point_in_circle(const Vector3 &center, float radius, int query_filter);
	float get_distance_to_wall(const Vector3 &point, float radius, int query_filter);
	Vector3 recast(const Vector3 &start, const Vector3 &end, int query_filter);
};
#if 0
class Crowd : public Spatial {
	GDCLASS(Crowd, Spatial);
	Vector<NodePath> agent_paths;
	Vector<int> modes;
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void update_agent_list();
	static void _bind_methods();
	DetourCrowdManager *manager;
protected:
	void _notification(int p_what);
public:
	String get_configuration_warning();
	Crowd();
	~Crowd();
};
#endif
#endif
