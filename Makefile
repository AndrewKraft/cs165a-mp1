CXX = g++
CXXFLAGS = -Wall -g

NaiveBayesClassifier: main.o
	$(CXX) $^ -o $@

clean: /.bin/rm -f *.o NaiveBayesClassifier *.gch
