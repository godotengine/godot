using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// A pre-parsed relative or absolute path in a scene tree,
    /// for use with <see cref="Node.GetNode(NodePath)"/> and similar functions.
    /// It can reference a node, a resource within a node, or a property
    /// of a node or resource.
    /// For instance, <c>"Path2D/PathFollow2D/Sprite2D:texture:size"</c>
    /// would refer to the <c>size</c> property of the <c>texture</c>
    /// resource on the node named <c>"Sprite2D"</c> which is a child of
    /// the other named nodes in the path.
    /// You will usually just pass a string to <see cref="Node.GetNode(NodePath)"/>
    /// and it will be automatically converted, but you may occasionally
    /// want to parse a path ahead of time with NodePath.
    /// Exporting a NodePath variable will give you a node selection widget
    /// in the properties panel of the editor, which can often be useful.
    /// A NodePath is composed of a list of slash-separated node names
    /// (like a filesystem path) and an optional colon-separated list of
    /// "subnames" which can be resources or properties.
    ///
    /// Note: In the editor, NodePath properties are automatically updated when moving,
    /// renaming or deleting a node in the scene tree, but they are never updated at runtime.
    /// </summary>
    /// <example>
    /// Some examples of NodePaths include the following:
    /// <code>
    /// // No leading slash means it is relative to the current node.
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
        /// Disposes of this <see cref="NodePath"/>.
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
        /// The pointer to the native instance of this <see cref="NodePath"/>.
        /// </summary>
        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        /// <summary>
        /// Constructs an empty <see cref="NodePath"/>.
        /// </summary>
        public NodePath() : this(string.Empty) { }

        /// <summary>
        /// Constructs a <see cref="NodePath"/> from a string <paramref name="path"/>,
        /// e.g.: <c>"Path2D/PathFollow2D/Sprite2D:texture:size"</c>.
        /// A path is absolute if it starts with a slash. Absolute paths
        /// are only valid in the global scene tree, not within individual
        /// scenes. In a relative path, <c>"."</c> and <c>".."</c> indicate
        /// the current node and its parent.
        /// The "subnames" optionally included after the path to the target
        /// node can point to resources or properties, and can also be nested.
        /// </summary>
        /// <example>
        /// Examples of valid NodePaths (assuming that those nodes exist and
        /// have the referenced resources or properties):
        /// <code>
        /// // Points to the Sprite2D node.
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
        /// Converts a string to a <see cref="NodePath"/>.
        /// </summary>
        /// <param name="from">The string to convert.</param>
        public static implicit operator NodePath(string from)
        {
            return new NodePath(from);
        }

        /// <summary>
        /// Converts this <see cref="NodePath"/> to a string.
        /// </summary>
        /// <param name="from">The <see cref="NodePath"/> to convert.</param>
        public static implicit operator string(NodePath from)
        {
            return godot_icall_NodePath_operator_String(NodePath.GetPtr(from));
        }

        /// <summary>
        /// Converts this <see cref="NodePath"/> to a string.
        /// </summary>
        /// <returns>A string representation of this <see cref="NodePath"/>.</returns>
        public override string ToString()
        {
            return (string)this;
        }

        /// <summary>
        /// Returns a node path with a colon character (<c>:</c>) prepended,
        /// transforming it to a pure property path with no node name (defaults
        /// to resolving from the current node).
        /// </summary>
        /// <example>
        /// <code>
        /// // This will be parsed as a node path to the "x" property in the "position" node.
        /// var nodePath = new NodePath("position:x");
        /// // This will be parsed as a node path to the "x" component of the "position" property in the current node.
        /// NodePath propertyPath = nodePath.GetAsPropertyPath();
        /// GD.Print(propertyPath); // :position:x
        /// </code>
        /// </example>
        /// <returns>The <see cref="NodePath"/> as a pure property path.</returns>
        public NodePath GetAsPropertyPath()
        {
            return new NodePath(godot_icall_NodePath_get_as_property_path(NodePath.GetPtr(this)));
        }

        /// <summary>
        /// Returns all subnames concatenated with a colon character (<c>:</c>)
        /// as separator, i.e. the right side of the first colon in a node path.
        /// </summary>
        /// <example>
        /// <code>
        /// var nodepath = new NodePath("Path2D/PathFollow2D/Sprite2D:texture:load_path");
        /// GD.Print(nodepath.GetConcatenatedSubnames()); // texture:load_path
        /// </code>
        /// </example>
        /// <returns>The subnames concatenated with <c>:</c>.</returns>
        public string GetConcatenatedSubnames()
        {
            return godot_icall_NodePath_get_concatenated_subnames(NodePath.GetPtr(this));
        }

        /// <summary>
        /// Gets the node name indicated by <paramref name="idx"/> (0 to <see cref="GetNameCount"/>).
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
        /// <returns>The name at the given index <paramref name="idx"/>.</returns>
        public string GetName(int idx)
        {
            return godot_icall_NodePath_get_name(NodePath.GetPtr(this), idx);
        }

        /// <summary>
        /// Gets the number of node names which make up the path.
        /// Subnames (see <see cref="GetSubnameCount"/>) are not included.
        /// For example, <c>"Path2D/PathFollow2D/Sprite2D"</c> has 3 names.
        /// </summary>
        /// <returns>The number of node names which make up the path.</returns>
        public int GetNameCount()
        {
            return godot_icall_NodePath_get_name_count(NodePath.GetPtr(this));
        }

        /// <summary>
        /// Gets the resource or property name indicated by <paramref name="idx"/> (0 to <see cref="GetSubnameCount"/>).
        /// </summary>
        /// <param name="idx">The subname index.</param>
        /// <returns>The subname at the given index <paramref name="idx"/>.</returns>
        public string GetSubname(int idx)
        {
            return godot_icall_NodePath_get_subname(NodePath.GetPtr(this), idx);
        }

        /// <summary>
        /// Gets the number of resource or property names ("subnames") in the path.
        /// Each subname is listed after a colon character (<c>:</c>) in the node path.
        /// For example, <c>"Path2D/PathFollow2D/Sprite2D:texture:load_path"</c> has 2 subnames.
        /// </summary>
        /// <returns>The number of subnames in the path.</returns>
        public int GetSubnameCount()
        {
            return godot_icall_NodePath_get_subname_count(NodePath.GetPtr(this));
        }

        /// <summary>
        /// Returns <see langword="true"/> if the node path is absolute (as opposed to relative),
        /// which means that it starts with a slash character (<c>/</c>). Absolute node paths can
        /// be used to access the root node (<c>"/root"</c>) or autoloads (e.g. <c>"/global"</c>
        /// if a "global" autoload was registered).
        /// </summary>
        /// <returns>If the <see cref="NodePath"/> is an absolute path.</returns>
        public bool IsAbsolute()
        {
            return godot_icall_NodePath_is_absolute(NodePath.GetPtr(this));
        }

        /// <summary>
        /// Returns <see langword="true"/> if the node path is empty.
        /// </summary>
        /// <returns>If the <see cref="NodePath"/> is empty.</returns>
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
