using System;

namespace Godot
{
    public partial class PackedScene
    {
        /// <summary>
        /// Instantiates the scene's node hierarchy, erroring on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// <see cref="Node.NotificationSceneInstantiated"/> notification on the root node.
        /// </summary>
        /// <seealso cref="InstantiateOrNull{T}(GenEditState)"/>
        /// <exception cref="InvalidCastException">
        /// The instantiated node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>The instantiated scene.</returns>
        public T Instantiate<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return (T)(object)Instantiate(editState);
        }

        /// <summary>
        /// Instantiates the scene's node hierarchy, returning <see langword="null"/> on failure.
        /// Triggers child scene instantiation(s). Triggers a
        /// <see cref="Node.NotificationSceneInstantiated"/> notification on the root node.
        /// </summary>
        /// <seealso cref="Instantiate{T}(GenEditState)"/>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>The instantiated scene.</returns>
        public T InstantiateOrNull<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return Instantiate(editState) as T;
        }
    }
}
