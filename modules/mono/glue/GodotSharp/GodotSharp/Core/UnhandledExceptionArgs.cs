using System;

namespace Godot
{
    /// <summary>
    /// 发生未处理异常时的事件参数。
    /// </summary>
    public class UnhandledExceptionArgs
    {
        /// <summary>
        /// 异常对象。
        /// </summary>
        public Exception Exception { get; private set; }

        internal UnhandledExceptionArgs(Exception exception)
        {
            Exception = exception;
        }
    }
}
