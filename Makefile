.PHONY: gpu-image-processing

gpu-image-processing:
	mkdir -p Build
	cd Build; cmake -DCMAKE_BUILD_TYPE=Debug ..; make
	cp Build/gpu-image-processing ./gpu-image-processing
