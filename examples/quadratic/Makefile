all: wrappers.so

clean:
	rm cost_functions/wrappers.so
	rm cost_functions/wrappers.cpp
	rm -rf build

wrappers.so:
	python setup.py build_ext --inplace --force
