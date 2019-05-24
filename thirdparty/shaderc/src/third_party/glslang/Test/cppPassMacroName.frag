#define f1(i) ((i)*(i))
#define I2(f, n) f(n) + f(n+1)
#define I3(f, n) I2(f, n) + f(n+2)

#define FL_f1(i) ((i)*(i))
#define FL_I2(f, n) f(n) + f(n+0.2)
#define FL_I3(f, n) FL_I2(f, n) + f(n+0.5)

void main()
{
    int f1 = 4;
    int f2 = f1;
    int f3 = f1(3);
    int f4 = I2(f1, 0);
    int f5 = I3(f1, 0);

    highp float fl_f5 = FL_I3(FL_f1, 0.1);
}

// f5 = I3(f1, 0)
//    = I2(f1, 0) + f1(0 + 2)
//    = f1(0) + f1(0+1) + f1(0+2)
//    = 0*0 + 1*1 + 2*2
//    = 5

// fl_f5 = FL_I3(FL_f1, 0.1)
//       = FL_I2(FL_f1, 0.1) + FL_f1(0.1 + 0.5)
//       = FL_f1(0.1) + FL_f1(0.1 + 0.2) + FL_f1(0.1 + 0.5)
//       = 0.1*0.1 + 0.3*0.3 + 0.6*0.6
//       = 0.46
