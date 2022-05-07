using System;

namespace Godot
{
    public partial class PackedScene
    {
        /// <summary>
        /// Instantiates the scene's node hierarchy, erroring on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// <see cref="Node.NotificationInstanced"/> notification on the root node.
        /// </summary>
        /// <seealso cref="InstanceOrNull{T}(GenEditState)"/>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the instantiated node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>The instantiated scene.</returns>
        public T Instance<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return (T)(object)Instance(editState);
        }

        /// <summary>
        /// Instantiates the scene's node hierarchy, returning <see langword="null"/> on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// <see cref="Node.NotificationInstanced"/> notification on the root node.
        /// </summary>
        /// <seealso cref="Instance{T}(GenEditState)"/>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>The instantiated scene.</returns>
        public T InstanceOrNull<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return Instance(editState) as T;
        }
    }
}
