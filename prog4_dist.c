/******************************************************************************
File: prog4_dist.c

Purpose: Perform LU Decomposition on a matrix input by the user.

Compiling: mpicc -g -Wall -std=c99 -lm -o prog4_dist prog4_dist.c 

Usage: mpiexec -n <number of processes> -hostfile <list of hosts> ./prog4_dist

Notes: IMPORTANT NOTE, Redirected i/o is very strongly recommended for using
this program and an example input file should be shipped with this file.

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void printMatrix(unsigned int code, unsigned int ROWS, double *arr);
unsigned int getScalarsIndex(unsigned int row1, unsigned int row2, double *INPUT);
double* addRow(double *original, double* toChange, double multiplier, unsigned int SIZE);
void subtractRow(double *original, double* toChange, double multiplier, unsigned int SIZE);
void makeLMatrix();
void makeUMatrix();
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, int *INDEXES, double *SCALARS);

int main(int argc, char * argv[])
{
	double 		*input;
	unsigned int 	size;
	double 		*L;
	double 		*U;
	int		*indexes;
	double		*scalars;
	int		comm_sz;
	int		my_rank;
	double 		t_start, t_length;
	
	MPI_Init(NULL, NULL);

	//get comm size and rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	//get user input and set up the original states of the matrices
	if(my_rank == 0)
	{
		printf("Enter number of rows for the square matrix: ");
		scanf("%u", &size);
		
		//matrices as contiguous memory
		input = malloc(sizeof(double) * size * size);
		L = malloc(sizeof(double) * size * size);
		U = malloc(sizeof(double) * size * size);
		indexes = malloc(sizeof(unsigned int) * size);
		scalars = malloc(sizeof(double)*(size-1)*(size)/2);
		
		makeInput(size, input, L, U, indexes, scalars);
		t_start = MPI_Wtime();
	}

	//make the U matrix
	//makeUMatrix();
	//make the L matrix
	//makeLMatrix();
	
	if(my_rank == 0)
	{
		t_length = MPI_Wtime() - t_start;
		printf("The LU Decomposition took %lf seconds\n", t_length);
	}

	//print the input, then L, then U, then the result of multiplication
	printMatrix(3, size, input);
	
	//free all the arrays
	if(my_rank == 0)
	{
		free(scalars);
		free(indexes);
		free(input);
		free(L);
		free(U);
	}

	MPI_Finalize();
	return 0;
}

unsigned int getScalarsIndex(unsigned int row1, unsigned int row2, double *data)
{
	return row1 + row2 + data[row1];
}
/******************************************************************************
Gets the input of the matrix information from the user (redirect i/o) is
recommended. L is initialized to the identity matrix of the appropriate
dimensions. U is initialized to a copy of the INPUT matrix so that the INPUT
matrix is preserved for output.
******************************************************************************/
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, int *INDEXES, double *SCALARS)
{
	unsigned int i, j;

	for(i = 0; i < SIZE; i++)
	{
		for(j = 0; j < SIZE; j++)
		{
			scanf("%lf", &INPUT[i*SIZE+j]);
			U[i*SIZE+j] = INPUT[i*SIZE+j];
			if(i == j)
			{
				L[i*SIZE+j] = 1;
			}
			else
			{
				L[i*SIZE+j] = 0;
			}
		}
		if(i == 0 || i == 1)
		{
			INDEXES[i] = -1;
		}
		else
		{
			INDEXES[i] = INDEXES[i-1] + i - 1;
		}
	}
}

/******************************************************************************
Handles the making of the U matrix. The matrix starts as a copy of the INPUT
matrix and does Guassian Row Elimination up to the point where it becomes a
Right Upper Matrix. Multipliers for rows are stored in the SCALARS array by the
scaleURow method for later use by the makeLMatrix() method.
******************************************************************************/
/*void makeUMatrix()
{
	unsigned int i, j;
	double temp;
	double *rowI;
	double *rowJ;

	for(i = 0; i < ROWS; i++)
	{
		rowI = U[i];
#pragma omp parallel for num_threads(NUM_THREADS) private(j, temp, rowJ) shared(i, rowI, ROWS, SCALARS) schedule(static)
		for(j = i + 1; j < ROWS; j++)
		{
			rowJ = U[j];
			if(rowJ[i] == 0 || rowI[i] == 0)
			{
				SCALARS[getScalarsIndex(i,j)] = 0;
			}
			else
			{
				temp = rowJ[i]/rowI[i];
				SCALARS[getScalarsIndex(i,j)] = temp;
				subtractRow(rowI, rowJ, temp);
			}			
		}
	}
}*/
/******************************************************************************
Handles the making of the L matrix. The matrix starts at the identity matrix 
and then uses the multipliers stored in SCALARS (starting at the end) to build
the L matrix based on the process of building the U matrix which should be
complete when this function is called.
******************************************************************************/
/*void makeLMatrix()
{
	int i, j;
	double *newRow;

	for(i = ROWS - 1; i > 0; i--)
	{
#pragma omp parallel for num_threads(NUM_THREADS) private(j, newRow) shared(SCALARS, L, i) schedule(static)	
		for(j = i-1; j >= 0; j--)
		{
			
			if(SCALARS[getScalarsIndex(j,i)] != 0)
			{
				
				newRow = addRow(L[j], L[i], SCALARS[getScalarsIndex(j,i)]);
				#pragma omp critical
				{
					free(L[i]);
					L[i] = newRow;
				}
			printf("%d\n", getScalarsIndex(j,i));
			}
			
		}
	}
}*/
/******************************************************************************
Prints the matrix specified by the code passed into it. 1 prints L, 2 prints U,
3 prints INPUT.
******************************************************************************/
void printMatrix(unsigned int code, unsigned int ROWS, double *arr)
{
	unsigned int i, j;

	switch (code)
	{
		case 1:
		{
			printf("**********L Matrix**********\n");
			break;
		}
		case 2:
		{
			printf("**********U Matrix**********\n");
			break;
		}
		case 3:
		{
			printf("**********Input Matrix**********\n");
			break;
		}
		/*case 4:
		{
			double sum = 0;
			printf("**********Multiply Matrix**********\n");
			for(k = 0; k < ROWS; k++)
			{
				for(i = 0; i < ROWS; i++)
				{
					for(j = 0; j < ROWS; j++)
					{
						sum = sum + L[k][j] * U[j][i];
					}
					printf("%0.2lf    ", sum);
					sum = 0;
				}
				printf("\n");
			}
			break;
		}*/
	}
	for(i = 0; i < ROWS; i++)
	{
		for(j = 0; j < ROWS; j++)
		{
			printf("%lf    ", arr[i*ROWS+j]);
		}
		printf("\n");
	}
	return;
}

/******************************************************************************
This function adds one row to anotherwhile scaling the row that
corresponds to the "oroginal" indexer. It returns a new row rather than
assigning to the original row in an effort to shorten critical sections.
******************************************************************************/
double* addRow(double* original, double* toChange, double multiplier, unsigned int SIZE)
{
	unsigned int i;
	double * toReturn = malloc(sizeof(double)*SIZE);

	for(i = 0; i < SIZE; i++)
	{
		toReturn[i] = toChange[i] + original[i] * multiplier;
	}

	return toReturn;
}

/******************************************************************************
This function subtracts one row from another while scaling the row that
corresponds to the "oroginal" indexer. It returns a new row rather than
assigning to the original row in an effort to shorten critical sections.
******************************************************************************/
void subtractRow(double* original, double* toChange, double multiplier, unsigned int SIZE)
{
	unsigned int i;

	for(i = 0; i < SIZE; i++)
	{
		toChange[i] = toChange[i] - original[i] * multiplier;
	}

	//return toReturn;
}
