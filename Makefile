all:
	g++ -shared -fPIC -Llibdl -o src/eigen_functions.so src/eigen_functions.cpp -O3

clean:
	rm -f src/eigen_functions.so