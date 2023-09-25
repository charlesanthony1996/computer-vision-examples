// testing the visitor pattern?
#include <iostream>
#include <vector>
#include <string>

class CarElementVisitor;

class CarElement {
    public:
        virtual void accept(CarElementVisitor& visitor) const = 0;
        virtual ~CarElement() = default;
};

class Body;
class Car;
class Engine;
class Wheel;



