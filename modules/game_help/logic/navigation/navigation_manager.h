#ifndef _NAVIGATION_MANAGER_H
#define _NAVIGATION_MANAGER_H
#include "core/object/ref_counted.h"
#include "core/math/a_star.h"








class MainRoad : public RefCounted
{
public:
    // 路点信息
  struct RoadPoint
  {
    uint32_t astar_id = -1;
     int32_t id;
     Vector3 pos;
     Vector3 normal;
     float width;
     int32_t prev_point_id = -1;
     int32_t next_point_id = -1;
     // 道路类型
     int road_type;
     // 左边连接的道路id
     int32_t left_link_road_id;
     // 右边连接的道路id
     int32_t right_link_road_id;
     // 细分出来的道路信息
     LocalVector<Vector3> segment;
     AABB aabb;
  };
  // 路信息
  struct Road
  {
    // 增加一个路点
    int32_t add_point(int prev_id,Vector3 pos)
    {
        int32_t id = allocall_id();
        RoadPoint& rp = points[id];
        rp.pos = pos;
        rp.prev_id = prev_id;
        if(prev_id!=-1 && roads.has(prev_id)){
            RoadPoint& r = roads[prev_id];
            r.next_point_id = id;
            rp.next_point_id = r.next_point_id;
            if(r.next_point_id!=-1 && roads.has(r.next_point_id)){
                RoadPoint& r2 = roads[r.next_point_id];
                r2.prev_point_id = id;
            }
        }
        return id;
    }
    // 删除一个路点
    void remove_point(int32_t id)
    {
        if (!roads.has(id))
        {
            return ;
        }
        RoadPoint& rp = points[id];
        if(rp.prev_point_id!=-1 && roads.has(rp.prev_point_id)){
            RoadPoint& r = roads[rp.prev_point_id];
            r.next_point_id = rp.next_point_id;
            if(rp.next_point_id!=-1 && roads.has(rp.next_point_id)){
                RoadPoint& r2 = roads[rp.next_point_id];
                r2.prev_point_id = rp.prev_point_id;
            }
        }        
    }
    int32_t allocall_id()
    {
        int32_t id = 0;
        while(roads.has(id))id++;
        return id;
    }
    int32_t link_road_id;
    int32_t link_point_id;
     int32_t id;
     AABB aabb;

     HashMap<int32_t,RoadPoint> points;     
  };
  // 构建astar 寻路信息
  void build_astar(Ref<AStar3D> astar)
  {
     uint32_t astar_id = 0;
     // 初始化道路的ID
     for(auto it = roads.begin();it!=roads.end();it++){
        Road& r = it->value;
        if(auto& point : r.points){
            point.astar_id = astar_id;
            astar->add_point(point.astar_id,point.pos);
            astar_id++;
        }
     }
     
     for(auto it = roads.begin();it!=roads.end();it++){
        Road& r = it->value;
        if(auto& point : r.points){
            if(point.next_point_id!=-1 && roads.has(point.next_point_id)){
                Road& r2 = roads[point.next_point_id];
                astar->connect_points(point.astar_id,r2.astar_id);
            }
        }
     }
  }
  HashMap<int32_t,Road> roads;

};











class NavigationManager
{
public:
    NavigationManager();
    ~NavigationManager();

    // 构建寻路信息
    void build_astar(Ref<AStar3D> astar)
    {
        if(astar.is_valid()){
            astar->clear();
        }
        if(main_road.is_valid()){
            main_road->build_astar(astar);
        }
    }
protected:
    Ref<AStar3D> astar;
    Ref<MainRoad> main_road;
};
#endif