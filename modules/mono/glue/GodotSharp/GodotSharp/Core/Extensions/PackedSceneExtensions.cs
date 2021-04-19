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
        public T Instance<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return (T)(object)Instance(editState);
        }

        /// <summary>
        /// Instantiates the scene's node hierarchy, returning null on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// `Node.NotificationInstanced` notification on the root node.
        /// </summary>
        /// <typeparam name="T">The type to cast to. Should be a descendant of Node.</typeparam>
        public T InstanceOrNull<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return Instance(editState) as T;
        }
    }
}
