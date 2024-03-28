using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Godot;
using System.Collections.Concurrent;

[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net80)]
[RPlotExporter]
public class BenchmarkObjectCollection
{
    private sealed class TestObject
    {
        private IntPtr fakeData;
    }

    private ConcurrentDictionary<TestObject, byte>? _collectionBased;
    private DisposablesTracker.FirstNonDuplicateLinkedList<TestObject>? _customTypeBased;

    [Params(100, 1000, 10_000)]
    public int NumberOfObjects;

    private TestObject CreateTestObject()
    {
        return new TestObject();
    }

    [Benchmark]
    public void Add_Object_Collection_Based()
    {
        _collectionBased = new();
        for (int i = 0; i < NumberOfObjects; i++)
        {
            _collectionBased.TryAdd(CreateTestObject(), 0);
        }
    }

    [Benchmark]
    public void Add_Object_Custom_Type_Based()
    {
        _customTypeBased = new();
        for (int i = 0; i < NumberOfObjects; i++)
        {
            _customTypeBased.Add(CreateTestObject());
        }
    }
}
