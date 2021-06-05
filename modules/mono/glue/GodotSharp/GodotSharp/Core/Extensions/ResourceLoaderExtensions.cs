namespace Godot
{
    public static partial class ResourceLoader
    {
        public static T Load<T>(string path, string typeHint = null, bool noCache = false) where T : class =>
            (T)(object)Load(path, typeHint, noCache);
    }
}
