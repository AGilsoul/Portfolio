#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include "../include/Car.h"

using std::endl;
using std::setw;

Car::Car()
{

}

Car::Car(string make, string model, string powerType, string bodyType, string driveTrain, string mpg, string hp, string price) {
    this->make = make;
    this->model = model;
    this->powerType = powerType;
    this->bodyType = bodyType;
    this->driveTrain = driveTrain;
    this->mpg = mpg;
    this->hp = hp;
    this->price = price;

}

string Car::getMake() {
    return make;
}

string Car::getModel() {
    return model;
}

string Car::getPowerType() {
    return powerType;
}

string Car::getBodyType() {
    return bodyType;
}

string Car::getDriveTrain() {
    return driveTrain;
}

int Car::getMPG() {
    return stoi(mpg);
}

int Car::getHP() {
    return stoi(hp);
}

int Car::getPrice() {
    return stoi(price);
}

//overloads the << operator for the Car class
ostream& operator << (ostream &out, Car &s) {
    out << setw(12) << s.getMake() << setw(30) << s.getModel();
    out << "|" << setw(13) << s.getPowerType();
    out << "|" << setw(12) <<s.getBodyType() << "|" << setw(4) << s.getDriveTrain();
    out << "|";
    if (s.getPowerType() == "Combustion" || s.getPowerType() == "Hybrid")  {
        out << setw(10) <<"MPG: " << s.getMPG();
    }
    else {
        out << setw(9) <<"Range: " << s.getMPG();
    }
    if (s.getHP() >= 1000) {
        out << "|" << setw(9) << "Horsepower: " << s.getHP();
        out << setw(6) << "$" << s.getPrice();
    }
    else if (s.getHP() < 100) {
        out << "|" << setw(11) << "Horsepower: " << s.getHP();
        out << setw(8) << "$" << s.getPrice();
    }
    else {
        out << "|" << setw(10) << "Horsepower: " << s.getHP();
        out << setw(7) << "$" << s.getPrice();
    }
    return out;
}
