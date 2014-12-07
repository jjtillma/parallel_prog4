/******************************************************************************
File: prog4_shared.c

Purpose: Perform LU Decomposition on a matrix input by the user.

Compiling: gcc -g -Wall -fopenmp -o prog4_shared prog4_shared.c

Usage: prog4_shared [number_of_threads]

Notes: IMPORTANT NOTE, Redirected i/o is very strongly recommended for using
this program and an example input file should be shipped with this file.

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#ifdef DEBUG
double INPUT[3][3] = {{4,2,2},{1,0,8},{2,4,4}};
unsigned int ROWS = 3;
#else
double **INPUT;
unsigned int ROWS;
#endif

double **L;
double **U;
int * INDEXES;
double *SCALARS;

unsigned long NUM_THREADS;

void printMatrix(unsigned int code);
unsigned int getScalarsIndex(unsigned int row1, unsigned int row2);
void subtractRow(double *original, double* toChange, double multiplier);
void addRow(double *original, double* toChange, double multiplier);
void makeLMatrix();
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
	//make the L matrix
	makeLMatrix();
	length = omp_get_wtime() - start;
	printf("The LU Decomposition took %lf seconds\n", length);

	//print the input, then L, then U, then the result of multiplication
	#ifdef DEBUG
	printMatrix(3);
	printMatrix(1);
	printMatrix(2);
	printMatrix(4);
	#endif

	//free all the arrays
	free(SCALARS);
	#ifndef DEBUG
	free(INDEXES);
	for(i = 0; i < ROWS; i++)
	{
		free(INPUT[i]);
	}
	free(INPUT);
	#endif
	for(i = 0; i < ROWS; i++)
	{
		free(L[i]);
	}
	free(L);
	for(i = 0; i < ROWS; i++)
	{
		free(U[i]);
	}
	free(U);

	return 0;
}

unsigned int getScalarsIndex(unsigned int row1, unsigned int row2)
{
	return row1 + row2 + INDEXES[row1];
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

	#ifndef DEBUG
	printf("Enter number of rows for the square matrix: ");
	scanf("%u", &ROWS);
	srand(time(NULL));
	INPUT = malloc(sizeof(double *)*ROWS);
	#endif
	
	SCALARS = malloc(sizeof(double)*(ROWS-1)*(ROWS)/2);

	L = malloc(sizeof(double *)*ROWS);
	U = malloc(sizeof(double *)*ROWS);
	INDEXES = malloc(sizeof(unsigned int *)*ROWS);

	for(i = 0; i < ROWS; i++)
	{
		#ifndef DEBUG
		INPUT[i] = malloc(sizeof(double)*ROWS);
		#endif
		L[i] = malloc(sizeof(double)*ROWS);
		U[i] = malloc(sizeof(double)*ROWS);
		for(j = 0; j < ROWS; j++)
		{
			#ifndef DEBUG
			INPUT[i][j] = rand() % 20 + 1;
			#endif
			U[i][j] = INPUT[i][j];
			if(i == j)
			{
				L[i][j] = 1;
			}
			else if(j <= ROWS)
			{
				L[i][j] = 0;
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
void makeUMatrix()
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
}
/******************************************************************************
Handles the making of the L matrix. The matrix starts at the identity matrix 
and then uses the multipliers stored in SCALARS (starting at the end) to build
the L matrix based on the process of building the U matrix which should be
complete when this function is called.
******************************************************************************/
void makeLMatrix()
{
	int i, j;
	double *newRow;

	for(i = ROWS - 1; i > 0; i--)
	{
		for(j = i-1; j >= 0; j--)
		{
			if(SCALARS[getScalarsIndex(j,i)] != 0)
			{
				printf("%d, %d, %d, %lf\n", j, i, getScalarsIndex(j,i), SCALARS[getScalarsIndex(j,i)]);
				addRow(L[j], L[i], SCALARS[getScalarsIndex(j,i)]);
			}
		}
	}
}
/******************************************************************************
Prints the matrix specified by the code passed into it. 1 prints L, 2 prints U,
3 prints INPUT.
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
					printf("%0.2f    ", L[i][j]);
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
					printf("%0.2lf    ", U[i][j]);
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
					printf("%0.2lf    ", INPUT[i][j]);
				}
				printf("\n");
			}
			break;
		}
		case 4:
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
		}
	}
}

/******************************************************************************
This function adds one row to anotherwhile scaling the row that
corresponds to the "oroginal" indexer. It returns a new row rather than
assigning to the original row in an effort to shorten critical sections.
******************************************************************************/
void addRow(double* original, double* toChange, double multiplier)
{
	unsigned int i;

	#pragma omp parallel for private(i), shared(ROWS, original, toChange, multiplier)
	for(i = 0; i < ROWS; i++)
	{
		toChange[i] = toChange[i] + original[i] * multiplier;
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
