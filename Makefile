CC = gcc
MCC = mpicc
LIBS =
FLAGS = -g -Wall
EXECS = make_matrix prog4_shared prog4_shared_col prog4_dist prog4_dist_col

all: $(EXECS)

prog4_shared: prog4_shared.c
	$(CC) $(FLAGS) -O -fopenmp -lm -o $@ $? $(LIBS)

prog4_shared_col: prog4_shared_col.c
	$(CC) $(FLAGS) -O -fopenmp -lm -o $@ $? $(LIBS)

prog4_dist: prog4_dist.c
	$(MCC) $(FLAGS) -std=c99 -lm -o $@ $?

prog4_dist_col: prog4_dist_col.c
	$(MCC) $(FLAGS) -std=c99 -lm -o $@ $?

make_matrix: make_matrix.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS) -lm

clean:
	$(RM) $(EXECS)

