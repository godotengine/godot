#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// Godot's global functions.
    /// </summary>
    public static partial class GD
    {
        /// <summary>
        /// 将字节数组解码回 <c>Variant</c> 值。
        /// 如果 <paramref name="allowObjects"/> 是 <see langword="true"/> 则允许解码对象。
        ///
        /// 警告：反序列化的对象可以包含被执行的代码。
        /// 不要将 <paramref name="allowObjects"/> 设置为 <see langword="true"/>
        /// 如果序列化对象来自不受信任的来源，要避免
        /// 潜在的安全威胁（远程代码执行）。
        /// </summary>
        /// <param name="bytes">将被解码为<c>Variant</c>的字节数组。</param>
        /// <param name="allowObjects">如果对象应该被解码。</param>
        /// <returns>解码后的<c>Variant</c>.</returns>
        public static object Bytes2Var(byte[] bytes, bool allowObjects = false)
        {
            return godot_icall_GD_bytes2var(bytes, allowObjects);
        }

        /// <summary>
        /// 以最佳方式从 <c>Variant</c> 类型转换为另一种类型。
        /// <paramref name="type"/> 参数使用 <see cref="Variant.Type"/> 值。
        /// </summary>
        /// <example>
        /// <code>
        /// var a = new Vector2(1, 0);
        /// // Prints 1
        /// GD.Print(a.Length());
        /// var b = GD.Convert(a, Variant.Type.String)
        /// // Prints 6 as "(1, 0)" is 6 characters
        /// GD.Print(b.Length);
        /// </code>
        /// </example>
        /// <returns><c>Variant</c> 转换为给定的 <paramref name="type"/>.</returns>
        public static object Convert(object what, Variant.Type type)
        {
            return godot_icall_GD_convert(what, type);
        }

        /// <summary>
        /// 从分贝转换为线性能量（音频）。
        /// </summary>
        /// <seealso cref="Linear2Db(real_t)"/>
        /// <param name="db">要转换的分贝数。</param>
        /// <returns>音频音量作为线性能量。</returns>
        public static real_t Db2Linear(real_t db)
        {
            return (real_t)Math.Exp(db * 0.11512925464970228420089957273422);
        }

        /// <summary>
        /// 返回 <paramref name="value"/> 减少的结果
        /// <paramref name="step"/> * <paramref name="amount"/>.
        /// </summary>
        /// <example>
        /// <code>
        /// // a = 59;
        /// // float a = GD.DecTime(60, 10, 0.1f);
        /// </code>
        /// </example>
        /// <param name="value">Value that will be decreased.</param>
        /// <param name="amount">
        /// 对于每个 <paramref name="step"/>，将从 <paramref name="value"/> 减少的数量。
        /// </param>
        /// <param name="step">Times the <paramref name="value"/> will be decreased by <paramref name="amount"/></param>
        /// <returns>减少的值。</returns>
        [Obsolete("DecTime has been deprecated and will be removed in Godot 4.0, use Mathf.MoveToward instead.")]
        public static real_t DecTime(real_t value, real_t amount, real_t step)
        {
            real_t sgn = Mathf.Sign(value);
            real_t val = Mathf.Abs(value);
            val -= amount * step;
            if (val < 0)
                val = 0;
            return val * sgn;
        }

        /// <summary>
        /// 获取引用函数的 <see cref="FuncRef"/>
        /// 使用给定的名称 <paramref name="funcname"/> 在
        /// 给定对象 <paramref name="instance"/>。
        /// </summary>
        /// <param name="instance">包含函数的对象。</param>
        /// <param name="funcname">函数名。</param>
        /// <returns>对给定对象函数的引用。</returns>
        public static FuncRef FuncRef(Object instance, string funcname)
        {
            var ret = new FuncRef();
            ret.SetInstance(instance);
            ret.SetFunction(funcname);
            return ret;
        }

        private static object[] GetPrintParams(object[] parameters)
        {
            if (parameters == null)
            {
                return new[] { "null" };
            }

            return Array.ConvertAll(parameters, x => x?.ToString() ?? "null");
        }

        /// <summary>
        /// 返回传递的变量的整数哈希。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(GD.Hash("a")); // Prints 177670
        /// </code>
        /// </example>
        /// <param name="var">将被散列的变量。</param>
        /// <returns>传递的变量的哈希值。</returns>
        public static int Hash(object var)
        {
            return godot_icall_GD_hash(var);
        }

        /// <summary>
        /// 返回对应于 <paramref name="instanceId"/> 的 <see cref="Object"/>。
        /// 所有对象都有一个唯一的实例 ID。
        /// </summary>
        /// <example>
        /// <code>
        /// public class MyNode : Node
        /// {
        ///     public string foo = "bar";
        ///
        ///     public override void _Ready()
        ///     {
        ///         ulong id = GetInstanceId();
        ///         var inst = (MyNode)GD.InstanceFromId(Id);
        ///         GD.Print(inst.foo); // Prints bar
        ///     }
        /// }
        /// </code>
        /// </example>
        /// <param name="instanceId">要检索的对象的实例ID。</param>
        /// <returns><see cref="Object"/> 实例。</returns>
        public static Object InstanceFromId(ulong instanceId)
        {
            return godot_icall_GD_instance_from_id(instanceId);
        }

        /// <summary>
        /// 从线性能量转换为分贝（音频）。
        /// 这可用于实现按预期运行的音量滑块（因为音量不是线性的）。
        /// </summary>
        /// <seealso cref="Db2Linear(real_t)"/>
        /// <example>
        /// <code>
        /// // "slider" refers to a node that inherits Range such as HSlider or VSlider.
        /// // Its range must be configured to go from 0 to 1.
        /// // Change the bus name if you'd like to change the volume of a specific bus only.
        /// AudioServer.SetBusVolumeDb(AudioServer.GetBusIndex("Master"), GD.Linear2Db(slider.value));
        /// </code>
        /// </example>
        /// <param name="linear">要转换的线性能量。</param>
        /// <returns>音频分贝数。</returns>
        public static real_t Linear2Db(real_t linear)
        {
            return (real_t)(Math.Log(linear) * 8.6858896380650365530225783783321);
        }

        /// <summary>
        /// 从位于 <paramref name="path"/> 的文件系统加载资源。
        /// 资源在方法调用时加载（除非它已经被引用
        /// 其他地方，例如 在另一个脚本或场景中），这可能会导致轻微的延迟，
        /// 特别是在加载场景时。 为了避免加载东西时不必要的延迟
        /// 多次，要么将资源存储在变量中。
        ///
        /// 注意：资源路径可以通过右键单击FileSystem中的资源来获取
        ///停靠并选择“复制路径”或将文件从文件系统停靠拖到脚本中。
        ///
        /// 重要：路径必须是绝对路径，本地路径只会返回<see langword="null"/>。
        /// 该方法是<see cref="ResourceLoader.Load"/>的简化版，可以使用
        /// 用于更高级的场景。
        /// </summary>
        /// <example>
        /// <code>
        /// // Load a scene called main located in the root of the project directory and cache it in a variable.
        /// var main = GD.Load("res://main.tscn"); // main will contain a PackedScene resource.
        /// </code>
        /// </example>
        /// <param name="path">要加载的 <see cref="Resource"/> 的路径。</param>
        /// <returns>加载的<see cref="Resource"/>.</returns>
        public static Resource Load(string path)
        {
            return ResourceLoader.Load(path);
        }

        /// <summary>
        /// 从位于 <paramref name="path"/> 的文件系统加载资源。
        /// 资源在方法调用时加载（除非它已经被引用
        /// 其他地方，例如 在另一个脚本或场景中），这可能会导致轻微的延迟，
        /// 特别是在加载场景时。 为了避免加载东西时不必要的延迟
        /// 多次，要么将资源存储在变量中。
        ///
        /// 注意：资源路径可以通过右键单击FileSystem中的资源来获取
        ///停靠并选择“复制路径”或将文件从文件系统停靠拖到脚本中。
        ///
        /// 重要：路径必须是绝对路径，本地路径只会返回<see langword="null"/>。
        /// 该方法是<see cref="ResourceLoader.Load"/>的简化版，可以使用
        /// 用于更高级的场景。
        /// </summary>
        /// <example>
        /// <code>
        /// // Load a scene called main located in the root of the project directory and cache it in a variable.
        /// var main = GD.Load&lt;PackedScene&gt;("res://main.tscn"); // main will contain a PackedScene resource.
        /// </code>
        /// </example>
        /// <param name="path">要加载的 <see cref="Resource"/> 的路径。</param>
        /// <typeparam name="T">要转换的类型。 应该是 <see cref="Resource"/> 的后代。</typeparam>
        public static T Load<T>(string path) where T : class
        {
            return ResourceLoader.Load<T>(path);
        }

        /// <summary>
        /// 将错误消息推送到 Godot 的内置调试器和 OS 终端。
        ///
        /// 注意：以这种方式打印的错误不会暂停项目执行。
        /// 要在调试版本中打印错误消息并暂停项目执行，
        /// 使用 [code]assert(false, "test error")[/code] 代替。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.PushError("test_error"); // Prints "test error" to debugger and terminal as error call
        /// </code>
        /// </example>
        /// <param name="message">错误消息.</param>
        public static void PushError(string message)
        {
            godot_icall_GD_pusherror(message);
        }

        /// <summary>
        /// 将警告消息推送到 Godot 的内置调试器和 OS 终端。
        /// </summary>
        /// <example>
        /// GD.PushWarning("test warning"); // Prints "test warning" to debugger and terminal as warning call
        /// </example>
        /// <param name="message">警告消息.</param>
        public static void PushWarning(string message)
        {
            godot_icall_GD_pushwarning(message);
        }

        /// <summary>
        /// 以尽可能最好的方式将任何类型的一个或多个参数转换为字符串
        /// 并将它们打印到控制台。
        ///
        /// 注意：考虑使用 <see cref="PushError(string)"/> 和 <see cref="PushWarning(string)"/>
        /// 打印错误和警告消息，而不是 <see cref="Print(object[])"/>。
        /// 这将它们与用于调试目的的打印消息区分开来，
        /// 同时在打印错误或警告时显示堆栈跟踪。
        /// </summary>
        /// <example>
        /// <code>
        /// var a = new int[] { 1, 2, 3 };
        /// GD.Print("a", "b", a); // Prints ab[1, 2, 3]
        /// </code>
        /// </example>
        /// <param name="what">将打印的参数.</param>
        public static void Print(params object[] what)
        {
            godot_icall_GD_print(GetPrintParams(what));
        }

        /// <summary>
        /// 将当前堆栈跟踪信息打印到控制台。
        /// </summary>
        public static void PrintStack()
        {
            Print(System.Environment.StackTrace);
        }

        /// <summary>
        /// 以标准错误行的最佳方式将一个或多个参数打印到字符串。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.PrintErr("prints to stderr");
        /// </code>
        /// </example>
        /// <param name="what">将打印的参数.</param>
        public static void PrintErr(params object[] what)
        {
            godot_icall_GD_printerr(GetPrintParams(what));
        }

        /// <summary>
        /// 以可能的最佳控制台方式将一个或多个参数打印到字符串。
        /// 最后不添加换行符。
        ///
        /// 注意：由于 Godot 内置控制台的限制，这只会打印到终端。
        /// 如果需要在编辑器中打印，请使用其他方法，例如<see cref="Print(object[])"/>。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.PrintRaw("A");
        /// GD.PrintRaw("B");
        /// // Prints AB
        /// </code>
        /// </example>
        /// <param name="what">将打印的参数.</param>
        public static void PrintRaw(params object[] what)
        {
            godot_icall_GD_printraw(GetPrintParams(what));
        }

        /// <summary>
        /// 将一个或多个参数打印到控制台，每个参数之间有一个空格。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.PrintS("A", "B", "C"); // Prints A B C
        /// </code>
        /// </example>
        /// <param name="what">将打印的参数.</param>
        public static void PrintS(params object[] what)
        {
            godot_icall_GD_prints(GetPrintParams(what));
        }

        /// <summary>
        /// 将一个或多个参数打印到控制台，每个参数之间有一个选项卡。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.PrintT("A", "B", "C"); // Prints A       B       C
        /// </code>
        /// </example>
        /// <param name="what">将打印的参数.</param>
        public static void PrintT(params object[] what)
        {
            godot_icall_GD_printt(GetPrintParams(what));
        }

        /// <summary>
        /// 返回 <c>0.0</c> 和 <c>1.0</c>（含）之间的随机浮点值。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Randf(); // Returns e.g. 0.375671
        /// </code>
        /// </example>
        /// <returns>一个随机的 <see langword="float"/> 数字.</returns>
        public static float Randf()
        {
            return godot_icall_GD_randf();
        }

        /// <summary>
        /// 返回一个随机无符号 32 位整数。
        /// 使用余数获得区间<c>[0, N - 1]</c>（其中N小于2^32）内的随机值。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Randi();           // Returns random integer between 0 and 2^32 - 1
        /// GD.Randi() % 20;      // Returns random integer between 0 and 19
        /// GD.Randi() % 100;     // Returns random integer between 0 and 99
        /// GD.Randi() % 100 + 1; // Returns random integer between 1 and 100
        /// </code>
        /// </example>
        /// <returns>一个随机的 <see langword="uint"/> 数字.</returns>
        public static uint Randi()
        {
            return godot_icall_GD_randi();
        }

        /// <summary>
        /// 随机化随机数生成器的种子（或内部状态）。
        /// 当前实现使用基于时间的数字重新播种。
        ///
        /// 注意：该方法在项目运行时自动调用。
        /// 如果您需要修复种子以获得可重现的结果，请使用 <see cref="Seed(ulong)"/>
        /// 初始化随机数生成器。
        /// </summary>
        public static void Randomize()
        {
            godot_icall_GD_randomize();
        }

        /// <summary>
        /// 在 <paramref name="from"/> 之间的间隔上返回一个随机浮点值
        /// 和 <paramref name="to"/>（包括）。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.PrintS(GD.RandRange(-10.0, 10.0), GD.RandRange(-10.0, 10.0)); // Prints e.g. -3.844535 7.45315
        /// </code>
        /// </example>
        /// <returns>给定范围内的随机 <see langword="double"/> 数字。</returns>
        public static double RandRange(double from, double to)
        {
            return godot_icall_GD_rand_range(from, to);
        }

        /// <summary>
        /// 使用给定的 <paramref name="seed"/> 返回一个随机无符号 32 位整数。
        /// <paramref name="newSeed"/> 将返回新种子。
        /// </summary>
        /// <param name="seed">用于生成随机数的种子。</param>
        /// <param name="newSeed">随机数生成器使用的种子。</param>
        /// <returns>一个随机的<see langword="uint"/>数字。</returns>
        public static uint RandSeed(ulong seed, out ulong newSeed)
        {
            return godot_icall_GD_rand_seed(seed, out newSeed);
        }

        /// <summary>
        /// 返回一个 <see cref="IEnumerable{T}"/> 迭代自
        /// <c>0</c> 到 <paramref name="end"/> 以 <c>1</c> 为步骤。
        /// </summary>
        /// <param name="end">最后一个索引.</param>
        public static IEnumerable<int> Range(int end)
        {
            return Range(0, end, 1);
        }

        /// <summary>
        /// 返回一个 <see cref="IEnumerable{T}"/> 迭代自
        /// <paramref name="start"/> 到 <paramref name="end"/> 在 <c>1</c> 的步骤中。
        /// </summary>
        /// <param name="start">第一个索引</param>
        /// <param name="end">最后一个索引。</param>
        public static IEnumerable<int> Range(int start, int end)
        {
            return Range(start, end, 1);
        }

        /// <summary>
        /// 返回一个 <see cref="IEnumerable{T}"/> 迭代自
        /// <paramref name="start"/> 到 <paramref name="end"/> 在 <paramref name="step"/> 的步骤中。
        /// </summary>
        /// <param name="start">第一个索引</param>
        /// <param name="end">最后一个索引。</param>
        /// <param name="step">每次迭代增加索引的数量。</param>
        public static IEnumerable<int> Range(int start, int end, int step)
        {
            if (end < start && step > 0)
                yield break;

            if (end > start && step < 0)
                yield break;

            if (step > 0)
            {
                for (int i = start; i < end; i += step)
                    yield return i;
            }
            else
            {
                for (int i = start; i > end; i += step)
                    yield return i;
            }
        }

        /// <summary>
        /// 为随机数生成器设置种子。
        /// </summary>
        /// <param name="seed">将使用的种子.</param>
        public static void Seed(ulong seed)
        {
            godot_icall_GD_seed(seed);
        }

        /// <summary>
        /// 以尽可能最好的方式将任何类型的一个或多个参数转换为字符串。
        /// </summary>
        /// <param name="what">将转换为字符串的参数。</param>
        /// <returns>给定参数形成的字符串。</returns>
        public static string Str(params object[] what)
        {
            return godot_icall_GD_str(what);
        }

        /// <summary>
        /// 将 <see cref="Var2Str(object)"/> 返回的格式化字符串转换为原始值。
        /// </summary>
        /// <example>
        /// <code>
        /// string a = "{\"a\": 1, \"b\": 2 }";
        /// var b = (Godot.Collections.Dictionary)GD.Str2Var(a);
        /// GD.Print(b["a"]); // Prints 1
        /// </code>
        /// </example>
        /// <param name="str">将被转换为Variant的字符串</param>
        /// <returns>解码后的<c>Variant</c>.</returns>
        public static object Str2Var(string str)
        {
            return godot_icall_GD_str2var(str);
        }

        /// <summary>
        /// 返回给定的类是否存在于 <see cref="ClassDB"/> 中。
        /// </summary>
        /// <returns>如果类存在于 <see cref="ClassDB"/>.</returns>
        public static bool TypeExists(string type)
        {
            return godot_icall_GD_type_exists(type);
        }

        /// <summary>
        /// 将 <c>Variant</c> 值编码为字节数组。
        /// 如果 <paramref name="fullObjects"/> 是 <see langword="true"/> 允许编码对象
        /// （并且可能包含代码）。
        /// 可以使用 <see cref="Bytes2Var(byte[], bool)"/> 完成反序列化。
        /// </summary>
        /// <param name="var">将被编码的变体。</param>
        /// <param name="fullObjects">如果对象应该被序列化。</param>
        /// <returns><c>Variant</c> 编码为字节数组。</returns>
        public static byte[] Var2Bytes(object var, bool fullObjects = false)
        {
            return godot_icall_GD_var2bytes(var, fullObjects);
        }

        /// <summary>
        /// 将 <c>Variant</c> <paramref name="var"/> 转换为格式化字符串
        /// 稍后可以使用 <see cref="Str2Var(string)"/> 解析。
        /// </summary>
        /// <example>
        /// <code>
        /// var a = new Godot.Collections.Dictionary { ["a"] = 1, ["b"] = 2 };
        /// GD.Print(GD.Var2Str(a));
        /// // Prints
        /// // {
        /// //    "a": 1,
        /// //    "b": 2
        /// // }
        /// </code>
        /// </example>
        /// <param name="var">将转换为字符串的变体。</param>
        /// <returns><c>Variant</c> 编码为字符串。</returns>
        public static string Var2Str(object var)
        {
            return godot_icall_GD_var2str(var);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_GD_bytes2var(byte[] bytes, bool allowObjects);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_GD_convert(object what, Variant.Type type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_GD_hash(object var);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Object godot_icall_GD_instance_from_id(ulong instanceId);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_print(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_printerr(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_printraw(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_prints(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_printt(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern float godot_icall_GD_randf();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern uint godot_icall_GD_randi();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_randomize();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern double godot_icall_GD_rand_range(double from, double to);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern uint godot_icall_GD_rand_seed(ulong seed, out ulong newSeed);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_seed(ulong seed);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_GD_str(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_GD_str2var(string str);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_GD_type_exists(string type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern byte[] godot_icall_GD_var2bytes(object what, bool fullObjects);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_GD_var2str(object var);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_pusherror(string type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_GD_pushwarning(string type);
    }
}
