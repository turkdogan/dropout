CXX=g++
CXXFLAGS=-g -O2 -std=c++11 -pedantic
BIN=prog

SRC=$(wildcard *.cpp examples/*.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) -o $(BIN) $^

.PHONY: clean
clean:
	rm -f *.o $(BIN)
