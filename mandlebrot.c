#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <omp.h>


extern void matToImage(char* filename, int* mat, int* dims);
extern void matToImageColor(char* filename, int* mat, int* dims);

int main(int argc, char **argv){

    int numranks;
    int rank;
    MPI_Status stat;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numranks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int nx=600;
    int ny=400;
    int* matrix=(int*)malloc(nx*ny*sizeof(int));

    int iter = 0;
    int maxIter=255;
    double xStart=-2;
    double xEnd=1;
    double yStart=-1;
    double yEnd=1;

    double x=0;
    double y=0;
    double x0=0;
    double y0=0;

    int doneRanks=0;
    int starty,endy;
    int nextStarty;
    int disty = 10;
    bool done = false;
    int numelements = ny*nx;
    int rankelements = disty*nx;

    int *temparray = (int*)malloc(rankelements*sizeof(int));

    //master region
    if(rank==0){
        double stTime = MPI_Wtime();
        nextStarty=0;
        //for loop to kick off all the calculation on all ranks
        for(int i=1;i<numranks;i++){
            starty=nextStarty;
            endy=starty+disty-1;
            nextStarty+=disty;
            printf("Send to Rank: %d, Start Y: %d, End Y: %d\n",i,starty,endy);
            MPI_Send(&starty,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(&endy,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
        // wait to get values from every rank
        while(!done){
            MPI_Recv(temparray,rankelements,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&stat);
            MPI_Recv(&starty,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD,&stat);
            MPI_Recv(&endy,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD,&stat);
            int k = starty;
            for(int i=0;i<=(endy-starty);i++){
                for(int j=0;j<nx;j++){
                    matrix[k*nx+j] = temparray[i*nx+j];
                }
                k++;
            }
            if(nextStarty>ny){ 
                starty=-1;
                doneRanks++; 
            }else{ 
                starty=nextStarty;
                endy=starty+disty-1;
                nextStarty+=disty;
            }
            if(endy>ny){
                endy=ny-1;
            }

            MPI_Send(&starty,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD);
            MPI_Send(&endy,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD);
            if(doneRanks==numranks-1){
                done=true;
            }
        }
        double edTime = MPI_Wtime();
        printf("Full time: %f\n",edTime-stTime);
    }

    //worker area
    if(rank!=0){
        double startTime = MPI_Wtime();
        while(true){
            MPI_Recv(&starty,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat); 
            MPI_Recv(&endy,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat); 
            if(starty==-1){ 
                break;
            }
            int k = starty;
            int numoutside;
            #pragma omp parallel num_threads(6)
            {
                double threadsTime;
                if(rank==1){
                    threadsTime = omp_get_wtime();
                }
                #pragma omp for schedule(dynamic) private(x0,y0,x,y,iter)
                for(int i=0;i<=(endy-starty);i++){
                    for(int j=0;j<nx;j++){
                        x0=xStart+(1.0*j/nx)*(xEnd-xStart);
                        y0=yStart+(1.0*(k+i)/ny)*(yEnd-yStart);

                        x=0;
                        y=0;
                        iter=0;
                        while(iter<maxIter){
                            iter++;

                            double temp=x*x-y*y+x0;
                            y=2*x*y+y0;
                            x=temp;

                            if(x*x+y*y>4){
                                break;
                            }
                        }
                        temparray[i*nx+j]=iter;
                    }
                    
                }
                if(rank==1){
                    double threadeTime = omp_get_wtime();
                    printf("Thread %d compute time: %f", omp_get_thread_num(),threadeTime-threadsTime);
                }
            }
            printf("Rank %d asking for more:\n",rank);
            MPI_Send(temparray,rankelements,MPI_INT,0,0,MPI_COMM_WORLD);
            MPI_Send(&starty,1,MPI_INT,0,0,MPI_COMM_WORLD);
            MPI_Send(&endy,1,MPI_INT,0,0,MPI_COMM_WORLD);
        }
        double endTime = MPI_Wtime();
        printf("Rank %d calculation time: %f\n",rank,endTime-startTime);
    }
    
    if(rank==0){
        int dims[2];
        dims[0]=ny;
        dims[1]=nx;

        matToImage("mandelbrot.jpg", matrix, dims);
    }

    MPI_Finalize();

    return 0;
}

