#include <iostream>
#include <iomanip>
#include <vector>
#include <windows.h>
#include <shellapi.h>
#include "../include/carFileReader.h"

using std::cout;
using std::endl;
using std::setw;
using std::string;
using std::cin;
using std::vector;

enum menuChoice {filterMake = 1, filterPower, filterStyle, filterDrive, filterMPG, filterHP, filterPrice, Sort, Reset, View, Search, Quit};
int menu();
void webSearch(carFileReader);

int main()
{
    //makes new carFileReader object called newProgram
    carFileReader newProgram;
    string title = "Car Selection Program";
    //formats the title to be output
    cout << setw(55 + title.length() * 0.75) << title << endl;
    bool inProgram = true;
    int userChoice;

    //while the user is still using the program
    while (inProgram) {
        userChoice = menu();

        //switch statement for menu choices
        switch(userChoice) {

            case filterMake:
                {
                    bool adding = true;
                    string makeChoice;
                    string againChoice;
                    vector<string> makeChoiceV;

                    while (adding) {
                        cin.clear();
                        cout << "Enter the make you would like to filter by (include hyphens and capital letters): ";
                        cin >> makeChoice;
                        makeChoiceV.push_back(makeChoice);
                        cout << "Would you like to add another make? (Y/N): ";
                        cin.clear();
                        cin >> againChoice;
                        if (againChoice == "N" || againChoice == "n") {
                            adding = false;
                        }
                    }

                    //sends the vector of the make choices to the makeFilter() method of carFileReader
                    newProgram.makeFilter(makeChoiceV);
                    break;
                }

            case filterPower:
                {
                    bool adding = true;
                    string powerChoice;
                    string againChoice;
                    vector<string> powerChoiceV;

                    while (adding) {
                        cin.clear();
                        cout << "Enter the power source you would like to filter by (Combustion, Electric, Hybrid, Hydrogen): ";
                        cin >> powerChoice;
                        powerChoiceV.push_back(powerChoice);
                        cout << "Would you like to add another power source? (Y/N): ";
                        cin.clear();
                        cin >> againChoice;
                        if (againChoice == "N" || againChoice == "n") {
                            adding = false;
                        }
                    }

                    //sends the vector of the power type choices to the powerTypeFilter() method of carFileReader
                    newProgram.powerTypeFilter(powerChoiceV);
                    break;
                }

            case filterStyle:
                {
                    bool adding = true;
                    string styleChoice;
                    string againChoice;
                    vector<string> styleChoiceV;

                    while (adding) {
                        cin.clear();
                        cout << "Enter the body style you would like to filter by (Coupe, Sports Car, Sedan, Hatchback, Convertible, Crossover, SUV, Minivan, Van, Truck): ";
                        cin >> styleChoice;
                        styleChoiceV.push_back(styleChoice);
                        cout << "Would you like to add another body style? (Y/N): ";
                        cin.clear();
                        cin >> againChoice;
                        if (againChoice == "N" || againChoice == "n") {
                            adding = false;
                        }
                    }

                    //sends the vector of the body choices to the bodyTypeFilter() method of carFileReader
                    newProgram.bodyTypeFilter(styleChoiceV);
                    break;
                }

            case filterDrive:
                {
                    bool adding = true;
                    string driveChoice;
                    string againChoice;
                    vector<string> driveChoiceV;

                    while (adding) {
                        cin.clear();
                        cout << "Enter the drive-train you would like to filter by (FWD, RWD, AWD): ";
                        cin >>driveChoice;
                        driveChoiceV.push_back(driveChoice);
                        cout << "Would you like to add another drive-train? (Y/N): ";
                        cin.clear();
                        cin >> againChoice;
                        if (againChoice == "N" || againChoice == "n") {
                            adding = false;
                        }
                    }

                    //sends the vector of the drivetrain choices to the driveTrainFilter() method of carFileReader
                    newProgram.driveTrainFilter(driveChoiceV);
                    break;
                }

            case filterMPG:
                {
                    int mpgMin;
                    int mpgMax;
                    cout << "Enter your minimum MPG: ";
                    cin >> mpgMin;
                    cout << "Enter your maximum MPG: ";
                    cin >> mpgMax;
                    //sends the vector of the mpg choices to the mpgFilter() method of carFileReader
                    newProgram.mpgFilter(mpgMin, mpgMax);
                    break;
                }

            case filterHP:
                {
                    int hMin;
                    int hMax;
                    cout << "Enter your minimum horsepower: ";
                    cin >> hMin;
                    cout << "Enter your maximum horsepower: ";
                    cin >> hMax;
                    //sends the vector of the hp choices to the hpFilter() method of carFileReader
                    newProgram.hpFilter(hMin, hMax);
                    break;
                }

            case filterPrice:
                {
                    int pMin;
                    int pMax;
                    cout << "Enter your minimum price: ";
                    cin >> pMin;
                    cout << "Enter your maximum price: ";
                    cin >> pMax;
                    //sends the vector of the price choices to the priceFilter() method of carFileReader
                    newProgram.priceFilter(pMin, pMax);
                    break;
                }

            //case for different sorting options
            case Sort:
                {
                    string sortChoice;
                    bool validChoice = false;
                    cout << "What would you like to sort by? (Price, Horsepower, MPG): ";

                    while (!validChoice) {
                        cin.clear();
                        cin >> sortChoice;
                        if (sortChoice == "Price" || sortChoice == "price") {
                            //sorts the cars vector of the carFileReader object by price
                            newProgram.priceSort();
                            validChoice = true;
                        }
                        else if (sortChoice == "Horsepower" || sortChoice == "horsepower") {
                            //sorts the cars vector of the carFileReader object by horsepower
                            newProgram.hpSort();
                            validChoice = true;
                        }
                        else if (sortChoice == "MPG" || sortChoice == "mpg") {
                            //sorts the cars vector of the carFileReader object by mpg
                            newProgram.mpgSort();
                            validChoice = true;
                        }
                        else {
                            cout << "Invalid choice, try again: ";
                        }
                    }
                    break;
                }

            case Reset:
                {
                    //resets the cars vector to its initial value, removing all filters
                    newProgram.filterReset();
                    break;
                }

            case View:
                {
                    //prints every car in the cars vector
                    cout << newProgram;
                    break;
                }

            case Search:
                {
                    //Google search of all cars in the cars vector
                    webSearch(newProgram);
                    break;
                }

            case Quit:
                {
                    //exits program
                    inProgram = false;
                    break;
                }
        }
    }
    return 0;
}

//prints menu and user makes choice here
int menu() {
    int choice;
    bool validChoice = false;
    cout << "\n\n1 - Filter by Make\n2 - Filter by Power Source\n3 - Filter by Body Style\n4 - Filter by Drive-Train\n5 - Filter by MPG/Range\n6 - Filter by Horsepower\n7 - Filter by Price\n8 - Sort List\n9 - Reset Filters\n10 - View Current List\n11 - Search\n12 - Quit\nChoose an Option: ";

    while (!validChoice) {

        cin.clear();
        cin >> choice;
        if (choice >= 1 && choice <= 12) {
            validChoice = true;
        }
        else  {
            cin.clear();
            cin.ignore();
            cout << "Invalid Choice, Try Again: ";
        }
    }

    return choice;
}

//function for a Google search
void webSearch(carFileReader cars) {
    for (int i = 0; i < cars.getVector().size(); i++) {
        string carName = cars.getVector()[i].getMake() + " " + cars.getVector()[i].getModel();
        //gets Google URL for a car
        string URL = "https://www.google.com/search?source=hp&ei=hQmpX4eOAdnN0PEP8tKRkAs&q=" + carName + "&oq=" + carName;
        ShellExecuteA(NULL, "open", URL.c_str(), NULL, NULL, SW_SHOWNORMAL);
    }
}
