struct Test0 {};
struct Test1 { float f; };

void main()
{
    {
        Test0 a;
        Test0 b = (Test0)a;
    }

    {
        Test1 a;
        Test1 b = (Test1)a;
    }

    {
        Test0 a[2];
        Test0 b[2] = (Test0[2])a;
    }

    {
        Test1 a[2];
        Test1 b[2] = (Test1[2])a;
    }
}
