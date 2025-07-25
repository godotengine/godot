namespace Godot
{
    /// <summary>
    /// An interface that requires a GetAwaiter() method to get a reference to the Awaiter.
    /// </summary>
    public interface IAwaitable
    {
        public IAwaiter GetAwaiter();
    }

    /// <summary>
    /// A templated interface that requires a GetAwaiter() method to get a reference to the Awaiter.
    /// </summary>
    /// <typeparam name="TResult">A reference to the result to be passed out.</typeparam>
    public interface IAwaitable<out TResult>
    {
        public IAwaiter<TResult> GetAwaiter();
    }
}
