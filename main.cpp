#include <iostream>
#include "swap.h"

using namespace std;

class Base
{
public:
    virtual ~Base()
    {
        cout << "Base free"<<endl;

    }
    virtual void play()
    {
        cout << "Base play" << endl;
    }
private:
        int x;
};

class Derive : public Base
{
public:
    ~Derive()
    {
        cout << "Derive free" << endl;
    }
    virtual void play()
    {
        cout << "Derive play" << endl;
    }

private:
    //int x;
};
class Derive2 : public Derive
{
public:
    ~Derive2()
    {
        cout << "Derive2 free" << endl;
    }
    virtual void play()
    {
        cout << "Derive2 play" << endl;
    }

private:
    // int x;
};

int main()
{

    Base *pA = new Derive2();
    pA->play();

    int b = 0;
    int a = 9;

    cout << "ÖÐÎÄ" << endl;
    cout << "aaa" << endl;

    cout << a << b << endl;
    swap(a, b);
    cout << a << b << endl;

    cout << !b << endl;
    cout << !!b << endl;

    delete pA;
    return 0;
}
