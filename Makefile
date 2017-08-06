CXX=g++
CXXFLAGS=-g -o2 -std=c++11 -pedantic
BIN=prog

SRC=$(wildcard *.cpp examples/*.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) -o $(BIN) $^

%.o: %.c
	$(CXX) $@ -c $<

clean:
	rm -f *.o
	rm $(BIN)
