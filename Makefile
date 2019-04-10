.PHONY: hw10

hw10:
	mkdir -p Build
	cd Build; cmake -DCMAKE_BUILD_TYPE=Debug ..; make
	cp Build/hw10 ./hw10
