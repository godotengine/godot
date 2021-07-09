namespace Godot
{
    public partial class PackedScene
    {
        /// <summary>
        /// Instantiates the scene's node hierarchy, erroring on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// `Node.NotificationInstanced` notification on the root node.
        /// </summary>
        /// <typeparam name="T">The type to cast to. Should be a descendant of Node.</typeparam>
        public T Instance<T>(GenEditState editState = (GenEditState)0) where T : class =>
            (T)(object)Instance(editState);

        /// <summary>
        /// Instantiates the scene's node hierarchy, returning null on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// `Node.NotificationInstanced` notification on the root node.
        /// </summary>
        /// <typeparam name="T">The type to cast to. Should be a descendant of Node.</typeparam>
        public T InstanceOrNull<T>(GenEditState editState = (GenEditState)0) where T : class =>
            Instance(editState) as T;
    }
}
