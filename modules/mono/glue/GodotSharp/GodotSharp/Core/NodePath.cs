using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// 场景树中预解析的相对或绝对路径，
    /// 与 <see cref="Node.GetNode(NodePath)"/> 和类似函数一起使用。
    /// 它可以引用节点、节点内的资源或属性
    /// 一个节点或资源。
    /// 比如 <c>"Path2D/PathFollow2D/Sprite2D:texture:size"</c>
    /// 将引用 <c>texture</c> 的 <c>size</c> 属性
    /// 名为 <c>"Sprite2D"</c> 的节点上的资源，它是
    /// 路径中的其他命名节点。
    /// 您通常只需将字符串传递给 <see cref="Node.GetNode(NodePath)"/>
    /// 它会自动转换，但你可能偶尔会
    /// 想用 NodePath 提前解析路径。
    /// 导出 NodePath 变量将为您提供一个节点选择小部件
    /// 在编辑器的属性面板中，这通常很有用。
    /// NodePath 由斜线分隔的节点名称列表组成
    ///（如文件系统路径）和一个可选的以冒号分隔的列表
    /// “子名称”可以是资源或属性。
    ///
    /// 注意：在编辑器中，NodePath属性在移动时会自动更新，
    /// 重命名或删除场景树中的节点，但它们在运行时永远不会更新。
    /// </summary>
    /// <example>
    /// NodePaths 的一些示例包括：
    /// <code>
    /// // 没有前导斜杠意味着它是相对于当前节点的。
    /// new NodePath("A"); // Immediate child A.
    /// new NodePath("A/B"); // A's child B.
    /// new NodePath("."); // The current node.
    /// new NodePath(".."); // The parent node.
    /// new NodePath("../C"); // A sibling node C.
    /// // A leading slash means it is absolute from the SceneTree.
    /// new NodePath("/root"); // Equivalent to GetTree().Root
    /// new NodePath("/root/Main"); // If your main scene's root node were named "Main".
    /// new NodePath("/root/MyAutoload"); // If you have an autoloaded node or scene.
    /// </code>
    /// </example>
    public sealed partial class NodePath : IDisposable
    {
        private bool _disposed = false;

        private IntPtr ptr;

        internal static IntPtr GetPtr(NodePath instance)
        {
            if (instance == null)
                throw new NullReferenceException($"The instance of type {nameof(NodePath)} is null.");

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~NodePath()
        {
            Dispose(false);
        }

        /// <summary>
        /// 处理这个 <see cref="NodePath"/>。
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (ptr != IntPtr.Zero)
            {
                godot_icall_NodePath_Dtor(ptr);
                ptr = IntPtr.Zero;
            }

            _disposed = true;
        }

        internal NodePath(IntPtr ptr)
        {
            this.ptr = ptr;
        }

        /// <summary>
        /// 指向此 <see cref="NodePath"/> 的本机实例的指针。
        /// </summary>
        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        /// <summary>
        /// 构造一个空的 <see cref="NodePath"/>。
        /// </summary>
        public NodePath() : this(string.Empty) { }

        /// <summary>
        /// 从字符串 <paramref name="path"/> 构造一个 <see cref="NodePath"/>,
        /// 例如：<c>"Path2D/PathFollow2D/Sprite2D:texture:size"</c>.
        /// 如果路径以斜杠开头，则它是绝对路径。 绝对路径
        /// 仅在全局场景树中有效，在单个场景中无效
        /// 场景。 在相对路径中，<c>"."</c> 和 <c>".."</c> 表示
        /// 当前节点及其父节点。
        /// 目标路径后可选包含的“子名称”
        /// 节点可以指向资源或属性，也可以嵌套。
        /// </summary>
        /// <example>
        /// 有效节点路径的示例（假设这些节点存在并且
        /// 具有引用的资源或属性）：
        /// <code>
        /// // 指向 Sprite2D 节点。
        /// "Path2D/PathFollow2D/Sprite2D"
        /// // Points to the Sprite2D node and its "texture" resource.
        /// // GetNode() would retrieve "Sprite2D", while GetNodeAndResource()
        /// // would retrieve both the Sprite2D node and the "texture" resource.
        /// "Path2D/PathFollow2D/Sprite2D:texture"
        /// // Points to the Sprite2D node and its "position" property.
        /// "Path2D/PathFollow2D/Sprite2D:position"
        /// // Points to the Sprite2D node and the "x" component of its "position" property.
        /// "Path2D/PathFollow2D/Sprite2D:position:x"
        /// // Absolute path (from "root")
        /// "/root/Level/Path2D"
        /// </code>
        /// </example>
        /// <param name="path">A string that represents a path in a scene tree.</param>
        public NodePath(string path)
        {
            ptr = godot_icall_NodePath_Ctor(path);
        }

        /// <summary>
        /// 将字符串转换为 <see cref="NodePath"/>。
        /// </summary>
        /// <param name="from">要转换的字符串.</param>
        public static implicit operator NodePath(string from)
        {
            return new NodePath(from);
        }

        /// <summary>
        ///将此 <see cref="NodePath"/> 转换为字符串。
        /// </summary>
        /// <param name="from"><see cref="NodePath"/> 要转换.</param>
        public static implicit operator string(NodePath from)
        {
            return godot_icall_NodePath_operator_String(NodePath.GetPtr(from));
        }

        /// <summary>
        /// 将此 <see cref="NodePath"/> 转换为字符串。
        /// </summary>
        /// <returns>此 <see cref="NodePath"/> 的字符串表示形式.</returns>
        public override string ToString()
        {
            return (string)this;
        }

        /// <summary>
        /// 返回带有冒号字符 (<c>:</c>) 的节点路径，
        /// 将其转换为没有节点名称的纯属性路径（默认值
        /// 从当前节点解析）。
        /// </summary>
        /// <example>
        /// <code>
        /// // 这将被解析为“位置”节点中“x”属性的节点路径。
        /// var nodePath = new NodePath("position:x");
        /// // This will be parsed as a node path to the "x" component of the "position" property in the current node.
        /// NodePath propertyPath = nodePath.GetAsPropertyPath();
        /// GD.Print(propertyPath); // :position:x
        /// </code>
        /// </example>
        /// <returns><see cref="NodePath"/> 作为纯属性路径。</returns>
        public NodePath GetAsPropertyPath()
        {
            return new NodePath(godot_icall_NodePath_get_as_property_path(NodePath.GetPtr(this)));
        }

        /// <summary>
        /// 返回用冒号字符连接的所有子名称 (<c>:</c>)
        /// 作为分隔符，即节点路径中第一个冒号的右侧
        /// </summary>
        /// <example>
        /// <code>
        /// var nodepath = new NodePath("Path2D/PathFollow2D/Sprite2D:texture:load_path");
        /// GD.Print(nodepath.GetConcatenatedSubnames()); // texture:load_path
        /// </code>
        /// </example>
        /// <returns>与 <c>:</c> 连接的子名称。</returns>
        public string GetConcatenatedSubnames()
        {
            return godot_icall_NodePath_get_concatenated_subnames(NodePath.GetPtr(this));
        }

        /// <summary>
        /// 获取由 <paramref name="idx"/> 指示的节点名称（0 到 <see cref="GetNameCount"/>）。
        /// </summary>
        /// <example>
        /// <code>
        /// var nodePath = new NodePath("Path2D/PathFollow2D/Sprite2D");
        /// GD.Print(nodePath.GetName(0)); // Path2D
        /// GD.Print(nodePath.GetName(1)); // PathFollow2D
        /// GD.Print(nodePath.GetName(2)); // Sprite
        /// </code>
        /// </example>
        /// <param name="idx">The name index.</param>
        /// <returns>给定索引处的名称 <paramref name="idx"/>.</returns>
        public string GetName(int idx)
        {
            return godot_icall_NodePath_get_name(NodePath.GetPtr(this), idx);
        }

        /// <summary>
        /// 获取组成路径的节点名称的数量。
        /// 不包括子名称（参见 <see cref="GetSubnameCount"/>）。
        /// 例如<c>"Path2D/PathFollow2D/Sprite2D"</c>有3个名字。
        /// </summary>
        /// <returns>构成路径的节点名称的数量。</returns>
        public int GetNameCount()
        {
            return godot_icall_NodePath_get_name_count(NodePath.GetPtr(this));
        }

        /// <summary>
        /// 获取由 <paramref name="idx"/> 指示的资源或属性名称（0 到 <see cref="GetSubnameCount"/>）。
        /// </summary>
        /// <param name="idx">The subname index.</param>
        /// <returns>给定索引处的子名称 <paramref name="idx"/>.</returns>
        public string GetSubname(int idx)
        {
            return godot_icall_NodePath_get_subname(NodePath.GetPtr(this), idx);
        }

        /// <summary>
        /// 获取路径中资源或属性名称（“子名称”）的数量。
        /// 每个子名称在节点路径中的冒号字符 (<c>:</c>) 之后列出。
        /// 例如 <c>"Path2D/PathFollow2D/Sprite2D:texture:load_path"</c> 有 2 个子名。
        /// </summary>
        /// <returns>路径中的子名数。</returns>
        public int GetSubnameCount()
        {
            return godot_icall_NodePath_get_subname_count(NodePath.GetPtr(this));
        }

        /// <summary>
        /// 如果节点路径是绝对的（而不是相对的），则返回 <see langword="true"/>，
        /// 这意味着它以斜杠字符 (<c>/</c>) 开头。 绝对节点路径可以
        /// 用于访问根节点（<c>"/root"</c>）或自动加载（例如<c>"/global"</c>
        /// 如果注册了“全局”自动加载）。
        /// </summary>
        /// <returns>如果 <see cref="NodePath"/> 是绝对路径。</returns>
        public bool IsAbsolute()
        {
            return godot_icall_NodePath_is_absolute(NodePath.GetPtr(this));
        }

        /// <summary>
        /// 如果节点路径为空，则返回 <see langword="true"/>。
        /// </summary>
        /// <returns>如果 <see cref="NodePath"/> 为空。</returns>
        public bool IsEmpty()
        {
            return godot_icall_NodePath_is_empty(NodePath.GetPtr(this));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr godot_icall_NodePath_Ctor(string path);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void godot_icall_NodePath_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_operator_String(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr godot_icall_NodePath_get_as_property_path(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_get_concatenated_subnames(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_get_name(IntPtr ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern int godot_icall_NodePath_get_name_count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_get_subname(IntPtr ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern int godot_icall_NodePath_get_subname_count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool godot_icall_NodePath_is_absolute(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool godot_icall_NodePath_is_empty(IntPtr ptr);
    }
}
