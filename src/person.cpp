#include "person.h"

Person::Person(const string &namePerson) :
    name(namePerson)
{
}
string Person::getName() const
{
    return name;
}

void Person::addImageId(const string &newId)
{
    listImagesId.push_back(newId);
}

