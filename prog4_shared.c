/******************************************************************************
File: prog4_shared.c

Purpose: Perform LU Decomposition on a matrix randomly generated with a set
number of rows. The matrix will be a square matrix because a square matrix is
required to actually use the LU Decomposition for many of its applications.

Compiling: gcc -O -g -Wall -fopenmp -o prog4_shared prog4_shared.c [-DDEBUG]

Usage: prog4_shared [number_of_threads]

Notes: Adding the -DDEBUG will use a static matrix of a set size that outputs
INPUT, L, U, and the result of L*U for debugging so that it can be seen that
the algorithm gets a proper result.

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

double **INPUT;
unsigned int ROWS;

double **L;
double **U;
double **P;

unsigned long NUM_THREADS;

void printMatrix(unsigned int code);
void subtractRow(double *original, double* toChange, double multiplier);
void handleRowSwap(unsigned int i);
void makeUMatrix();
void makeInput();

int main(int argc, char * argv[])
{
	unsigned int i;
	double start, length;
	if(argc == 2)
	{
		//set the number of threads
		NUM_THREADS = atol(argv[1]);
	}
	else
	{
		printf("This requires 1 command argument: the number of threads to use.\n");
		return 1;
	}

	//get user input and set up the original states of the matrices
	makeInput();

	start = omp_get_wtime();
	//make the U matrix
	makeUMatrix();
	length = omp_get_wtime() - start;
	printf("The LU Decomposition took %lf seconds\n", length);

	//print the input, then L, then U, then the result of multiplication
	#ifdef DEBUG
	printMatrix(3);
	printMatrix(4);
	printMatrix(1);
	printMatrix(2);
	printMatrix(5);
	#endif
	
	for(i = 0; i < ROWS; i++)
	{
		free(P[i]);
		free(U[i]);
		free(L[i]);
		free(INPUT[i]);
	}
	free(P);
	free(U);
	free(L);

	return 0;
}

/******************************************************************************
Gets the input of the matrix information from the user (redirect i/o) is
recommended. L is initialized to the identity matrix of the appropriate
dimensions. U is initialized to a copy of the INPUT matrix so that the INPUT
matrix is preserved for output.
******************************************************************************/
void makeInput()
{
	unsigned int i, j;

	scanf("%u", &ROWS);
	INPUT = malloc(sizeof(double *)*ROWS);

	L = malloc(sizeof(double *)*ROWS);
	U = malloc(sizeof(double *)*ROWS);
	P = malloc(sizeof(double *)*ROWS);

	for(i = 0; i < ROWS; i++)
	{
		INPUT[i] = malloc(sizeof(double)*ROWS);
		L[i] = malloc(sizeof(double)*ROWS);
		P[i] = malloc(sizeof(double *)*ROWS);
		U[i] = malloc(sizeof(double)*ROWS);
		for(j = 0; j < ROWS; j++)
		{
			scanf("%lf", &INPUT[i][j]);
			//initialize U to be INPUT
			U[i][j] = INPUT[i][j];
			//initialize L and P to be the identitiy matrix
			if(i == j)
			{
				L[i][j] = 1;
				P[i][j] = 1;
			}
			else if(j <= ROWS)
			{
				L[i][j] = 0;
				P[i][j] = 0;
			}
		}		
	}
}

/******************************************************************************
Handles the making of the U matrix. The matrix starts as a copy of the INPUT
matrix and does Guassian Row Elimination up to the point where it becomes a
Right Upper Matrix. Multipliers for rows are stored in the SCALARS array by the
scaleURow method for later use by the makeLMatrix() method.
******************************************************************************/
void makeUMatrix()
{
	unsigned int i, j;
	double tempScalar;
	double *rowI;
	double *rowJ;

	for(i = 0; i < ROWS; i++)
	{
		rowI = U[i];
		if(rowI[i] == 0)
		{
			//this condition means we need a row swap
			//perform the swap and reset rowI to the new U[i]
			handleRowSwap(i);
			rowI = U[i];
		}

		#pragma omp parallel for num_threads(NUM_THREADS) private(j, tempScalar, rowJ) shared(i, rowI, ROWS) schedule(static)
		for(j = i + 1; j < ROWS; j++)
		{
			rowJ = U[j];
			if(rowJ[i] == 0 || rowI[i] == 0)
			{
				//if the J is 0, don't do anything
				tempScalar = 0;
			}
			else
			{
				//otherwise, find the scalar and use it for the subtraction and put it in L
				tempScalar = rowJ[i]/rowI[i];
				L[j][i] = tempScalar;
				subtractRow(rowI, rowJ, tempScalar);
			}			
		}
	}
}

/******************************************************************************
This function handles the nightmare tha tis a row swap. It swaps entire rows in
P and U. And swaps anything below the diagonal in L. It also flags P as now
having meaning instead of just being the identity matrix.
******************************************************************************/
void handleRowSwap(unsigned int i)
{
	unsigned int j, maxIndex = 0;
	double temp, max = 0;
	double *tempPtr, *rowI, *rowJ;
	double *maxes = malloc(sizeof(double *) * NUM_THREADS);
	unsigned int *maxIndexes = malloc(sizeof(unsigned int *) * NUM_THREADS);


	#pragma omp parallel num_threads(NUM_THREADS) private(j) shared(maxIndexes, maxes, ROWS, U, i, rowJ)
	{
		unsigned long thread = omp_get_thread_num();
		unsigned int start = (unsigned int)(thread * ceil((double)i/ROWS)) + i;
		unsigned int stop = start + start - i;
		printf("stop %d\n", stop);
		for(j = start; j < stop && j < ROWS; j++)
		{
			rowJ = U[j];
			if(abs(rowJ[i]) > abs(maxes[thread]))
			{
				maxIndexes[thread] = j;
				maxes[thread] = rowJ[i];
			}
		}
	}

	for(j = 0; j < NUM_THREADS; j++)
	{
		if(abs(maxes[j]) > abs(max))
		{
			max = maxes[j];
			maxIndex = maxIndexes[j];
		}
	}
	if(maxIndex != i)
	{
		//swap a row in U
		tempPtr = U[i];
		U[i] = U[maxIndex];
		U[maxIndex] = tempPtr;
		//swap a row in P
		tempPtr = P[i];
		P[i] = P[maxIndex];
		P[maxIndex] = tempPtr;
	
		rowI = L[i];
		rowJ = L[maxIndex];
		#pragma omp parallel for num_threads(NUM_THREADS) private(j, temp) shared(i, rowI, rowJ) schedule(dynamic)
		for(j = 0; j < i; j++)
		{
			temp = rowI[j];
			rowI[j] = rowJ[j];
			rowJ[j] = temp;
		}
	}
	free(maxes);
	free(maxIndexes);
}
/******************************************************************************
Prints the matrix specified by the code passed into it. 1 prints L, 2 prints U,
3 prints INPUT, 4 prints a Multiplied matrix. These prints are not factored
into any timings taken.
******************************************************************************/
void printMatrix(unsigned int code)
{
	unsigned int i, j, k;
	switch (code)
	{
		case 1:
		{
			printf("**********L Matrix**********\n");
			for(i = 0; i < ROWS; i++)
			{
				for(j = 0; j < ROWS; j++)
				{
					printf("%6.2lf    ", L[i][j]);
				}
				printf("\n");
			}
			break;
		}
		case 2:
		{
			printf("**********U Matrix**********\n");
			for(i = 0; i < ROWS; i++)
			{
				for(j = 0; j < ROWS; j++)
				{
					printf("%6.2lf    ", U[i][j]);
				}
				printf("\n");
			}
			break;
		}
		case 3:
		{
			printf("**********Input Matrix**********\n");
			for(i = 0; i < ROWS; i++)
			{
				for(j = 0; j < ROWS; j++)
				{
					printf("%6.2lf    ", INPUT[i][j]);
				}
				printf("\n");
			}
			break;
		}
		case 5:
		{
			/*this mess of nonsense performs a P'(LU) multiplication where P' is
			the transformation of the matrix P. It first multiplies L and U,
			storing the result. Then does P' * result to get the overall result.
			This isn't even close to optimized, but the only point of this is to
			output some information for debugging that shows that the LU
			Decomposition actually succeeded.*/
			double sum = 0;
			double ** sumContainer = malloc(sizeof(double *)*ROWS);
			for(i = 0; i < ROWS; i++)
			{
				sumContainer[i] = malloc(sizeof(double)*ROWS);	
			}

			printf("**********Multiply Matrix**********\n");
			for(k = 0; k < ROWS; k++)
			{
				for(i = 0; i < ROWS; i++)
				{
					for(j = 0; j < ROWS; j++)
					{
						sum = sum + L[k][j] * U[j][i];
					}
					sumContainer[k][i] = sum;
					sum = 0;
				}
			}

			for(k = 0; k < ROWS; k++)
			{
				for(i = 0; i < ROWS; i++)
				{
					for(j = 0; j < ROWS; j++)
					{
						sum = sum + P[j][k] * sumContainer[j][i];
					}
					printf("%6.2lf    ", sum);
					sum = 0;
				}
				printf("\n");
			}

			for(i = 0; i < ROWS; i++)
			{
				free(sumContainer[i]);	
			}
			free(sumContainer);
			break;
		}
		case 4:
		{
			printf("**********P Matrix**********\n");
			for(i = 0; i < ROWS; i++)
			{
				for(j = 0; j < ROWS; j++)
				{
					printf("%6.2lf    ", P[i][j]);
				}
				printf("\n");
			}
			break;
		}
	}
}

/******************************************************************************
This function subtracts one row from another while scaling the row that
corresponds to the "oroginal" indexer. It returns a new row rather than
assigning to the original row in an effort to shorten critical sections.
******************************************************************************/
void subtractRow(double* original, double* toChange, double multiplier)
{
	unsigned int i;

	for(i = 0; i < ROWS; i++)
	{
		toChange[i] = toChange[i] - original[i] * multiplier;
	}
}
