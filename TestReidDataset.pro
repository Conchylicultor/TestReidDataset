TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11

LIBS += `pkg-config opencv --libs`

SOURCES += src/main.cpp \
    src/dataset.cpp \
    src/person.cpp \
    src/adaptativedatabase.cpp

HEADERS += \
    src/dataset.h \
    src/person.h \
    src/adaptativedatabase.h
