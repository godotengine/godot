namespace Godot
{
    public partial interface INode
    {
        /// <summary>
        /// Fetches a node and casts it to the specified type <typeparamref name="T"/>.
        /// </summary>
        T GetNode<T>(NodePath path) where T : class;

        /// <summary>
        /// Fetches a node and casts it to the specified type <typeparamref name="T"/>, or returns null if not found.
        /// </summary>
        T GetNodeOrNull<T>(NodePath path) where T : class;

        /// <summary>
        /// Returns a child node by its index and casts it to the specified type <typeparamref name="T"/>.
        /// </summary>
        T GetChild<T>(int idx, bool includeInternal = false) where T : class;

        /// <summary>
        /// Returns a child node by its index and casts it to the specified type <typeparamref name="T"/>, or returns null if not found.
        /// </summary>
        T GetChildOrNull<T>(int idx, bool includeInternal = false) where T : class;

        /// <summary>
        /// Returns the node owner and casts it to the specified type <typeparamref name="T"/>.
        /// </summary>
        T GetOwner<T>() where T : class;

        /// <summary>
        /// Returns the node owner and casts it to the specified type <typeparamref name="T"/>, or returns null if there is no owner.
        /// </summary>
        T GetOwnerOrNull<T>() where T : class;

        /// <summary>
        /// Returns the parent node and casts it to the specified type <typeparamref name="T"/>.
        /// </summary>
        T GetParent<T>() where T : class;

        /// <summary>
        /// Returns the parent node and casts it to the specified type <typeparamref name="T"/>, or returns null if the node has no parent.
        /// </summary>
        T GetParentOrNull<T>() where T : class;
    }
}
