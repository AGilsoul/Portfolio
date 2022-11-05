#include <iostream>
#pragma once

using std::string;
using std::ostream;

class Car
{
    public:
        Car();
        Car(string make, string model, string powerType, string bodyType, string driveTrain, string mpg, string hp, string price);
        string getMake();
        string getModel();
        string getPowerType();
        string getBodyType();
        string getDriveTrain();
        int getMPG();
        int getHP();
        int getPrice();
        friend ostream& operator << (ostream &out, Car &s);
    private:
        string make, model, powerType, bodyType, driveTrain, mpg, hp, price;
};
