#include"iostream"
using namespace std;
class Base{
protected:
    int a;
private:
    int b;
    public:
    int c;
};
class Derived:protected Base{

};
int main(){
Base b;
Derived d;
cout<<d.b;
}
