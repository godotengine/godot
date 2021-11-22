using Godot;
using NUnit.Framework;

namespace GodotSharp.UnitTest
{
    public class Vector3Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Vector3_constructor()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            Assert.AreEqual(1.1, vector3.x);
            Assert.AreEqual(2.2, vector3.y);
            Assert.AreEqual(3.3, vector3.z);
        }

        [Test]
        public void Vector3_constructor_from_other()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = new Vector3(vector3);
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.x, vector3Copy.x);
            Assert.AreEqual(vector3.y, vector3Copy.y);
            Assert.AreEqual(vector3.z, vector3Copy.z);
        }

        [Test]
        public void Vector3_xyz()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.xyz;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.x, vector3Copy.x);
            Assert.AreEqual(vector3.y, vector3Copy.y);
            Assert.AreEqual(vector3.z, vector3Copy.z);
        }

        [Test]
        public void Vector3_xzy()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.xzy;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.x, vector3Copy.x);
            Assert.AreEqual(vector3.z, vector3Copy.y);
            Assert.AreEqual(vector3.y, vector3Copy.z);
        }

        [Test]
        public void Vector3_yxz()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.yxz;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.y, vector3Copy.x);
            Assert.AreEqual(vector3.x, vector3Copy.y);
            Assert.AreEqual(vector3.z, vector3Copy.z);
        }

        [Test]
        public void Vector3_yzx()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.yzx;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.y, vector3Copy.x);
            Assert.AreEqual(vector3.z, vector3Copy.y);
            Assert.AreEqual(vector3.x, vector3Copy.z);
        }

        [Test]
        public void Vector3_zxy()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.zxy;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.z, vector3Copy.x);
            Assert.AreEqual(vector3.x, vector3Copy.y);
            Assert.AreEqual(vector3.y, vector3Copy.z);
        }

        [Test]
        public void Vector3_zyx()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.zyx;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.z, vector3Copy.x);
            Assert.AreEqual(vector3.y, vector3Copy.y);
            Assert.AreEqual(vector3.x, vector3Copy.z);
        }

        [Test]
        public void Vector3_xy()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.xy;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.x, vector3Copy.x);
            Assert.AreEqual(vector3.y, vector3Copy.y);
        }

        [Test]
        public void Vector3_xz()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.xz;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.x, vector3Copy.x);
            Assert.AreEqual(vector3.z, vector3Copy.y);
        }

        [Test]
        public void Vector3_yx()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.yx;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.y, vector3Copy.x);
            Assert.AreEqual(vector3.x, vector3Copy.y);
        }

        [Test]
        public void Vector3_yz()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.yz;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.y, vector3Copy.x);
            Assert.AreEqual(vector3.z, vector3Copy.y);
        }

        [Test]
        public void Vector3_zx()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.zx;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.z, vector3Copy.x);
            Assert.AreEqual(vector3.x, vector3Copy.y);
        }

        [Test]
        public void Vector3_zy()
        {
            var vector3 = new Vector3(1.1f, 2.2f, 3.3f);
            var vector3Copy = vector3.zy;
            Assert.AreNotSame(vector3, vector3Copy);
            Assert.AreEqual(vector3.z, vector3Copy.x);
            Assert.AreEqual(vector3.y, vector3Copy.y);
        }
    }
}
