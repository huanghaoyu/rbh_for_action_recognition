SOURCE_FILES = main.cpp
CFLAGS = -D__STDC_CONSTANT_MACROS -O3 -rdynamic
#CFLAGS = -D__STDC_CONSTANT_MACROS -O0 -rdynamic -g
LDFLAGS = -lc -lopencv_core -lopencv_imgproc -lavcodec -lavformat -lavutil -lswscale  

all: $(SOURCE_FILES)
	mkdir -p build
	$(CXX) $(SOURCE_FILES) -o build/src $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf build
