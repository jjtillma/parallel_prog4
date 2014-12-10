CC = gcc
MCC = mpicc
LIBS =
FLAGS = -g -Wall
EXECS = make_matrix prog4_shared prog4_dist

all: $(EXECS)

prog4_shared: prog4_shared.c
	$(CC) $(FLAGS) -fopenmp -lm -o $@ $? $(LIBS)

prog4_dist: prog4_dist.c
	$(MCC) $(FLAGS) -std=c99 -lm -o $@ $?

make_matrix: make_matrix.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS) -lm

clean:
	$(RM) $(EXECS)

