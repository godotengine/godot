#ifndef FOLIAGE_PROTO_TYPE_H
#define FOLIAGE_PROTO_TYPE_H

#include "core/math/math_funcs.h"
#include "core/math/aabb.h"
#include "core/io/file_access.h"
#include "core/object/worker_thread_pool.h"

namespace Foliage
{
    class FoliagePrototype
    {
    public:
        enum EHIZCullAccuracy
        {
            automatic = 0,
            low = 1,
            medium = 2,
            high = 3,
        };
    public:
		/// <summary>
		/// prefab代理体的guid
		/// </summary>
		String guid;

		/// <summary>
		/// 名称，方便调试
		/// </summary>
		String name;

		/// <summary>
		/// HIZ剔除精度
		/// </summary>
		EHIZCullAccuracy hizCullAccuracy;

		/// <summary>
		/// 3个LOD等级的Prefab文件名
		/// </summary>
		String lod0PrefabPath, lod1PrefabPath, lod2PrefabPath, lod3PrefabPath;
		String colliderPath;
        bool is_convex_collider;

		/// <summary>
		/// 8个LOD等级最远渲染距离
		/// </summary>
		float lod0EndDistance = 32, lod1EndDistance = 96, lod2EndDistance = 192, lod3EndDistance = 256;
		/// <summary>
		/// 移动端3个LOD等级最远渲染距离
		/// </summary>
		float mobileLod0EndDistance = 32, mobileLod1EndDistance = 96, mobileLod2EndDistance = 192;
		// 4-8 加载距离

		/// <summary>
		/// 3个Lod层级密度显示 3个LOD密度，花类植被lod2密度单独控制
		/// </summary>
		float lod0Density=1.0f,lod1Density=1.0f,lod2Density=1.0f;

		// 移动端阴影渲染最大lod层级
		int mobileShadowRenderLevel = 3;
		// PC端阴影渲染最大lod层级
		int pcShadowRenderLevel = 3;

		// 移动端加载距离缩放
		float mobileLoadScale = 0.75f;

		/// <summary>
		/// 是否启用lod1、lod2
		/// </summary>
		bool lod1Enabled, lod2Enabled, lod3Enabled;

		/// <summary>
		/// 与地表的混和级别
		/// </summary>
		//public int groundBlendLevel = 4;
		

		/// <summary>
		/// Mesh对象空间包围盒
		/// </summary>
		AABB boxOS;

		/// <summary>
		/// 树叶Mesh对象空间包围盒
		/// </summary>
		AABB leafBoxOS;


		/// <summary>
		/// 是否灌木
		/// </summary>
		bool isBush;
	public:
		int protypeId = 0;
		int refCount = 0;
		// 是否正在被使用
		bool _isUse = false;

    public:
        FoliagePrototype()
        {

        }
		void reset_use()
		{
			_isUse = false;
		}
		void set_use()
		{
			_isUse = true;
		}
		bool is_use()
		{
			return _isUse;
		}
    };
    // 原型资源
    class FoliagePrototypeAsset : public RefCounted
    {
        GDCLASS(FoliagePrototypeAsset, RefCounted)

        static void _bind_methods()
        {

        }
    public:
        FoliagePrototypeAsset()
        {

        }
        
		struct FileLoadData
		{
			Ref<FoliagePrototypeAsset>	dest;
			Ref<FileAccess> file;
		};
        void unload(class FoliageManager * manager)
        {
            
        }

    private:
        Vector<FoliagePrototype> prototypes;
    };
}
#endif