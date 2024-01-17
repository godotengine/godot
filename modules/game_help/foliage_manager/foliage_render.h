#ifndef FOLIAGE_RENDER_H
#define FOLIAGE_RENDER_H
#include "foliage_proto_type.h"
#include "scene/resources/mesh.h"

namespace Foliage
{


    /// <summary>
    /// 该结构体封装一个渲染信�?
    /// </summary>
    struct MeshRenderInfo
    {
        /// <summary>
        /// 被渲染的Mesh
        /// </summary>
        Ref<Mesh> mesh;

        /// <summary>
        /// 要渲染的Mesh子集
        /// </summary>
        int meshSubset;

        /// <summary>
        /// 渲染阴影用的Mesh
        /// </summary>
        Ref<Mesh> shadowCasterMesh;

        /// <summary>
        /// 阴影Mesh子集
        /// </summary>
        int shadowCasterMeshSubset;

        /// <summary>
        /// 美术制作的原始材�?
        /// </summary>
        Ref<Material> originMtl;



        /// <summary>
        /// 实际渲染Mesh用的材质，它是原始材质的克隆并启用了一些Shader关键�?
        /// </summary>
        Ref<Material> material;

        /// <summary>
        /// 渲染阴影用的材质
        /// </summary>
        Ref<Material> shadowMaterial;
        Ref<Material> shadowMaterialLod;


        /// <summary>
        /// 渲染Mesh的Pass索引
        /// </summary>
        int renderPass;

        /// <summary>
        /// 渲染阴影用的Pass索引
        /// </summary>
        int shadowPass;

        /// <summary>
        /// 渲染深度或深度法线用的pass
        /// </summary>
        int depthPass, depthNormalPass;

        /// <summary>
        /// Motion Vectors的pass
        /// </summary>
        int motionVectorsPass;

        /// <summary>
        /// 材质ID的pass
        /// </summary>
        int materialIDPass;

        /// <summary>
        /// 是否叶子
        /// </summary>
        bool isLeaf;

        /// <summary>
        /// 渲染该Mesh用的渲染参数缓存在全局缓存中的偏移，以字节为单�?
        /// </summary>
        int drawArgsBufOffset;

        /// <summary>
        /// 正在visiblebuffer中的偏移,visibleBuffer中是吧每一�?
        /// </summary>
        int visibleCntOffset;
        /// <summary>
        /// 渲染反射的渲染参数缓存在全局缓存中的偏移，以字节为单�?
        /// 只有最后一个LOD才有数据
        /// </summary>
        int reflectionDrawArgsBufOffset;
        /// <summary>
        /// 相对于根节点的变�?
        /// </summary>
        Transform3D localTransform;

        /// <summary>
        /// localTransform是否单位变换
        /// </summary>
        bool localTransformIsIdentity;

        /// <summary>
        /// 渲染层掩�?
        /// </summary>
        float renderingLayerMask;

        /// <summary>
        /// 是否是有效材�?
        /// </summary>
        bool isInvalideMaterial;
        /// <summary>
        ///  渲染标签，标识使用的shader的标签，用来进行排序渲染
        /// </summary>
        long renderShaderTag;

        /// <summary>
        ///  渲染标签，标识使用的shader的标签，用来进行排序渲染
        /// </summary>
        int renderShaderTagIndex;

        /// <summary>
        ///  渲染标签，标识使用的shader的标签，用来进行排序渲染
        /// </summary>
        int renderShaderTagVersion;

        bool isShadowRender;
    };
   
    /// <summary>
    /// 该结构体封装每级LOD渲染信息
    /// </summary>
    struct LODRenderInfo
    {
        /// <summary>
        /// Mesh渲染信息数组，数组长度等于植被Mesh个数*植被Mesh子集个数
        /// </summary>
        Vector<MeshRenderInfo> meshRenderInfos;

        /// <summary>
        /// 用了几个DrawCall与Dispatch
        /// </summary>
        int numDrawCalls, numDispatch;

        /// <summary>
        /// 绘制了多少实�?
        /// </summary>
        int numInstancesDrawn, numShadowInstancesDrawn;

        /// <summary>
        /// 绘制阴影用了几个DrawCall与Dispatch
        /// </summary>
        int numShadowDrawCalls, numShadowDispatchs;

        bool isImpostor;
    };
    // 原型渲染信息
    class FiliageRender
    {
    public:
        struct FiliageRenderGpuResource
        {
            bool castShadow = true;
            int numInstances = 0;
            Vector<LODRenderInfo> lodRenderInfos;
        };
        // 對應的原型信息
        FoliagePrototype * prototype = nullptr;
        int renderIndex = 0;
        FiliageRenderGpuResource gpuResource;
        int heightType = 0;
        Vector4 lodEnabled;


        FiliageRender();
        ~FiliageRender();

        void render()
        {
            
        }
        void clear()
        {

        }

    };
}

#endif