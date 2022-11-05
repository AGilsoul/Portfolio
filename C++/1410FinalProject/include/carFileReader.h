#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include "Car.h"
#pragma once

using std::vector;
using std::ostream;

struct carInfo
        {
            string make, model, powerType, bodyType, driveTrain, mpg, hp, price;
        };

class carFileReader
{
    public:
        //use *(arrayVariable + index) to return items from array
        carFileReader();
        void priceSort();
        void mpgSort();
        void hpSort();
        void makeFilter(vector<string> makeV);
        void powerTypeFilter(vector<string> powerTypeV);
        void bodyTypeFilter(vector<string> bodyType);
        void driveTrainFilter(vector<string> driveTrain);
        void mpgFilter(int mpgMin, int mpgMax);
        void hpFilter(int hpMin, int hpMax);
        void priceFilter(int pMin, int pMax);
        void filterReset();
        vector<Car> getVector();
        friend ostream& operator << (ostream &out, carFileReader &s);

    private:
        vector<Car> cars;
        vector<Car> carsBackup;
        string loadCount = "";
        double percentLoaded = 0;
};
