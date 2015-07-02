CXX = g++
CPPFLAGS = -std=c++11
LIBS = -lm -lsndfile -lfftw3f
LDFLAGS =

obj bin:
	mkdir $@

obj/%:
	obj

bin/%:
	bin

obj/rbm.o: rbm.cpp
	$(CXX) $(CPPFLAGS) -c $^ -o $@

obj/normalize.o: normalize.cpp
	$(CXX) $(CPPFLAGS) -c $^ -o $@

obj/train.o: train.cpp
	$(CXX) $(CPPFLAGS) -c $^ -o $@

train: obj/rbm.o obj/normalize.o obj/train.o
	$(CXX) $(LDFLAGS) $^ -o bin/$@ $(LIBS)

clean:
	rm *.o

.PHONY: clean
