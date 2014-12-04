CC = gcc
LIBS = -lpthread
FLAGS = -g -Wall -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast
EXECS = philosopher_no_lock philosopher_lock trapezoid trapezoid_timed

all: $(EXECS)

philosopher_no_lock: philosopher_no_lock.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS)

philosopher_lock: philosopher_lock.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS)

trapezoid: trapezoid.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS) -lm

trapezoid_timed: trapezoid_timed.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS) -lm

clean:
	$(RM) $(EXECS)

