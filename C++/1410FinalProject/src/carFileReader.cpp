#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include "../include/carFileReader.h"
#include "../include/Car.h"

using std::getline;
using std::ifstream;
using std::ios;
using std::ostream;
using std::endl;
using std::cout;
using std::setw;



carFileReader::carFileReader() {
    ifstream fin("../res/CarDatabase.csv", ios::in);
    int carCounter = 0;

    //file is read a Car object is created for each line, and added to the cars vector
    cout << "Getting Car Information..." << endl;
    while (!fin.eof()) {
        carInfo newCar;
        getline(fin, newCar.make, ',');
        getline(fin, newCar.model, ',');
        getline(fin, newCar.powerType, ',');
        getline(fin, newCar.bodyType, ',');
        getline(fin, newCar.driveTrain, ',');
        getline(fin, newCar.mpg, ',');
        getline(fin, newCar.hp, ',');
        getline(fin, newCar.price, '\n');
        Car newestCar(newCar.make, newCar.model, newCar.powerType, newCar.bodyType, newCar.driveTrain, newCar.mpg, newCar.hp, newCar.price);
        carCounter++;
        cars.push_back(newestCar);
       }

    fin.close();
    carCounter--;
    //cout << setw(27) << "[" << (int)(percentLoaded) << "%] <" << loadCount << setw(49 - loadCount.length() - 1) << ">" << endl;
    cout << setw(64) << "Cars Loaded: " << carCounter << endl;
    cars.pop_back();
    carsBackup = cars;



}


//sorts cars by price (improved sorting algorithm over last version)
void carFileReader::priceSort() {
    vector<Car> tempCars;
    tempCars.push_back(cars[0]);

    for (int i = 1; i < cars.size(); i++) {
        for (int v = 0; v < tempCars.size(); v++) {
            if (cars[i].getPrice() > tempCars[v].getPrice()) {
                tempCars.insert(tempCars.begin() + v, cars[i]);
                break;
            }
            else if (v == tempCars.size() - 1) {
                tempCars.push_back(cars[i]);
                break;
            }
        }
    }
    cars = tempCars;
}

//sorts cars by mpg (improved sorting algorithm over last version)
void carFileReader::mpgSort() {
    vector<Car> tempCars;
    tempCars.push_back(cars[0]);

    for (int i = 1; i < cars.size(); i++) {
        for (int v = 0; v < tempCars.size(); v++) {
            if (cars[i].getMPG() < tempCars[v].getMPG()) {
                tempCars.insert(tempCars.begin() + v, cars[i]);
                break;
            }
            else if (v == tempCars.size() - 1) {
                tempCars.push_back(cars[i]);
                break;
            }
        }
    }
    cars = tempCars;
}

//sorts cars by horsepower (improved sorting algorithm over last version)
void carFileReader::hpSort() {
    vector<Car> tempCars;
    tempCars.push_back(cars[0]);

    for (int i = 1; i < cars.size(); i++) {
        for (int v = 0; v < tempCars.size(); v++) {
            if (cars[i].getHP() < tempCars[v].getHP()) {
                tempCars.insert(tempCars.begin() + v, cars[i]);
                break;
            }
            else if (v == tempCars.size() - 1) {
                tempCars.push_back(cars[i]);
                break;
            }
        }
    }
    cars = tempCars;
}

//remove cars from the vector if they do not pass the make filter
void carFileReader::makeFilter(vector<string> makeV) {
    vector<Car> newV;

    for (int i = 0; i < cars.size(); i++) {
        for (int v = 0; v < makeV.size(); v++) {
            if (cars[i].getMake() == makeV[v]) {
                newV.push_back(cars[i]);
                break;
            }
        }
    }

    cars = newV;
}

//remove cars from the vector if they do not pass the power type filter
void carFileReader::powerTypeFilter(vector<string> powerV) {
    vector<Car> newV;

    for (int i = 0; i < cars.size(); i++) {
        for (int v = 0; v < powerV.size(); v++) {
            if (cars[i].getPowerType() == powerV[v]) {
                newV.push_back(cars[i]);
                break;
            }
        }
    }
    cars = newV;
}

//remove cars from the vector if they do not pass the body type filter
void carFileReader::bodyTypeFilter(vector<string> bodyV) {
    vector<Car> newV;

    for (int i = 0; i < cars.size(); i++) {
        for (int v = 0; v < bodyV.size(); v++) {
            if (cars[i].getBodyType() == bodyV[v]) {
                newV.push_back(cars[i]);
                break;
            }
        }
    }
    cars = newV;
}

//remove cars from the vector if they do not pass the drivetrain filter
void carFileReader::driveTrainFilter(vector<string> driveV) {
    vector<Car> newV;

    for (int i = 0; i < cars.size(); i++) {
        for (int v = 0; v < driveV.size(); v++) {
            if (cars[i].getDriveTrain() == driveV[v]) {
                newV.push_back(cars[i]);
                break;
            }
        }
    }
    cars = newV;
}

//remove cars from the vector if they do not pass the mpg filter
void carFileReader::mpgFilter(int mpgMin, int mpgMax) {
    mpgSort();
    int sizeI = cars.size();

    for (int i = 0; i < sizeI; i++) {
        if ((cars[i].getMPG() > mpgMax) || (cars[i].getMPG() < mpgMin)) {
            cars.erase(cars.begin() + i);
            i--;
            sizeI--;
        }
    }
}

//remove cars from the vector if they do not pass the horsepower filter
void carFileReader::hpFilter(int hpMin, int hpMax) {
    hpSort();
    int sizeI = cars.size();

    for (int i = 0; i < sizeI; i++) {
        if ((cars[i].getHP() > hpMax) || (cars[i].getHP() < hpMin)) {
            cars.erase(cars.begin() + i);
            i--;
            sizeI--;
        }
    }
}

//remove cars from the vector if they do not pass the price filter
void carFileReader::priceFilter(int pMin, int pMax) {
    priceSort();
    int sizeI = cars.size();

    for (int i = 0; i < sizeI; i++) {
        if ((cars[i].getPrice() > pMax) || (cars[i].getPrice() < pMin)) {
            cars.erase(cars.begin() + i);
            i--;
            sizeI--;
        }
    }
}

void carFileReader::filterReset() {
    cars = carsBackup;
}

vector<Car> carFileReader::getVector() {
    return cars;
}

//overloads the << operator for the carFileReader class
ostream& operator << (ostream &out, carFileReader &s) {
    out << "------------------------------------------------------------------------------------------------------------------------" << endl;
    for (int i = 0; i < s.getVector().size(); i++) {
        out << s.getVector()[i] << endl;
        out << "------------------------------------------------------------------------------------------------------------------------";
    }

    return out;
}
