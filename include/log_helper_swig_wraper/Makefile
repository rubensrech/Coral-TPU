
#lib name, wraper file and output .so file
LIBRARY_NAME=log_helper
LIBRARY_NAME_WRAP=$(LIBRARY_NAME)_wrap

PYTHON_INCLUDES=$(shell python3-config --includes)
LOG_HELPER_DIR=../
INCLUDE=$(PYTHON_INCLUDES) -I$(LOG_HELPER_DIR)

CXX = g++
CC = gcc
SWIG = swig

CFLAGS=-fPIC

all:$(LIBRARY_NAME)_python
	
$(LIBRARY_NAME)_python: $(LIBRARY_NAME).c $(LIBRARY_NAME).i
	$(SWIG) -python $(LIBRARY_NAME).i
	$(CC) $(CFLAGS) -c $(LIBRARY_NAME).c $(LIBRARY_NAME_WRAP).c $(INCLUDE)
	$(CC) -shared $(LIBRARY_NAME).o $(LIBRARY_NAME_WRAP).o -o _$(LIBRARY_NAME).so

clean:
	rm -rf *.o *py *so *pyc $(LIBRARY_NAME_WRAP).c
