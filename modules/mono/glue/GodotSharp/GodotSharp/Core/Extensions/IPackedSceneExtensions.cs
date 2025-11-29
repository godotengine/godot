namespace Godot
{
    public partial interface IPackedScene
    {
        /// <summary>
        /// Instantiates the scene's node hierarchy and casts it to the specified type <typeparamref name="T"/>.
        /// </summary>
        T Instantiate<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class;

        /// <summary>
        /// Instantiates the scene's node hierarchy and casts it to the specified type <typeparamref name="T"/>, or returns null on failure.
        /// </summary>
        T InstantiateOrNull<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class;
    }
}
