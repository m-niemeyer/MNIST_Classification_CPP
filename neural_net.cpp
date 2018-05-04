#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <sstream>

using namespace std;

const float eta = 0.001; // Learning rate
const int dim1=784, dim2=28, dim3=28, dim4=10;
double W1[dim1+1][dim2], dW1[dim1+1][dim2];
double W2[dim2+1][dim3], dW2[dim2+1][dim3];
double W3[dim3+1][dim4], dW3[dim3+1][dim4];
double sum1[dim2], ac1[dim2], sum2[dim3], ac2[dim3], sum3[dim4], ac3[dim4];
double delta3[dim4], delta2[dim3], delta1[dim2];

int data_train[60000][784];
int data_test[10000][784];
int label_train[60000];
int label_test[10000];

double activation(double x) {
	/* Returns the value of the activation function */
	if (x>500) { x=500; }
	else if (x<-500) { x=-500; }
  return (1/(1+exp(-x)));
}

double d_activation(double x) {
	/* Returns the value of the derivative of the activation function */
  double y = activation(x);
  return y*(1-y);
}

double d_error(double y, double y_hat) {
	/* Returns the value of the derivative of the error function */
  //return -(y-y_hat);
	return y_hat-y;
}

double softmax_quotient(double *x) {
	double val = 0;
	for (int i=0; i<dim4; i++) {
		val += exp(x[i]);
	}

	return val;
}

void forward_pass(int *x) {

	for (int j=0; j<dim2; j++) {
			sum1[j] = 0;
			for(int i=0; i<dim1; i++) { 
				sum1[j] += W1[i][j]*x[i];
				//cout << "X:" << x[i] << " W1: " << W1[i][j] << endl;
			}
			sum1[j] += W1[dim1][j];
			//cout << "SUM: " << sum1[j] << endl;
			ac1[j] = activation(sum1[j]);
			//cout << "AC: " << ac1[j] << endl;
	}
	for (int j=0; j<dim3; j++) {
			sum2[j] = 0;
			for(int i=0; i<dim2; i++) sum2[j] += W2[i][j]*ac1[i];
			sum2[j] += W2[dim2][j];
			ac2[j] = activation(sum2[j]);
	}
	for (int j=0; j<dim4; j++) {
			sum3[j] = 0;
			for(int i=0; i<dim3; i++) sum3[j] += W3[i][j]*ac2[i];
			sum3[j] += W3[dim3][j];
			//ac3[j] = activation(sum3[j]);
			ac3[j] = exp(sum3[j]);
	}
	double quo = softmax_quotient(sum3);
	for (int i=0; i<dim4; i++) {
		ac3[i] /= quo;
	}
}

void update_weights() {
	for (int i=0;i<dim1+1;i++) {
		for (int j=0;j<dim2;j++) {
			W1[i][j] -= eta*dW1[i][j];
		}
	}
	for (int i=0;i<dim2+1;i++) {
		for (int j=0;j<dim3;j++) W2[i][j] -= eta*dW2[i][j];
	}
	for (int i=0;i<dim3+1;i++) {
		for (int j=0;j<dim4;j++) W3[i][j] -= eta*dW3[i][j];
	}
}

void calc_dw( int *x) {
	// Change of Weights for W3
	for (int i=0; i<dim3; i++) {
		for (int j=0; j<dim4; j++) { 
			dW3[i][j] = ac2[i] * delta3[j];
		}
	}
	for (int j=0; j<dim4; j++) dW3[dim3][j] = delta3[j]; // Bias weight
	//
	// Change of Weights for W2
	for (int i=0; i<dim2; i++) {
		for (int j=0; j<dim3; j++) dW2[i][j] = ac1[i] * delta2[j];
	}
	for (int j=0; j<dim3; j++) dW2[dim2][j] = delta2[j]; // Bias weight

	// Change of Weights for W1
	for (int i=0; i<dim1; i++) {
		for (int j=0; j<dim2; j++) dW1[i][j] = x[i] * delta1[j];
	}
	for (int j=0; j<dim2; j++) dW1[dim1][j] = delta1[j]; // Bias weight
}

void calc_deltas(int *y) {
	for (int i=0; i<dim4; i++) {
		//delta3[i] = d_error(static_cast<double>(y[i]), ac3[i])*d_activation(sum3[i]);
		delta3[i] = d_error(static_cast<double>(y[i]), ac3[i])*ac3[i]*(1-ac3[i]);
	}

	for (int i=0; i<dim3; i++) {
		delta2[i] = 0;
		for (int j=0; j<dim4; j++) delta2[i] += W3[i][j]*delta3[j];
		delta2[i] *= d_activation(sum2[i]);
	}

	for (int i=0; i<dim2; i++) {
		delta1[i] = 0;
		for (int j=0; j<dim3; j++) delta1[i] += W2[i][j]*delta2[j];
		delta1[i] *= d_activation(sum1[i]);
	}
}

void backward_pass(int *x, int *y) {
	calc_deltas(y);
	calc_dw(x);
	update_weights();
}


void initialise_weights() {
	/* Initialises the weight matrices */

	// First Weight Matrix
	for (int i=0; i<dim1+1; i++) {
		for (int j=0; j<dim2; j++) W1[i][j] = 2*double(rand())/RAND_MAX-1; // Random double value between -1 and 1
	}

	// Second Weight Matrix
	for (int i=0; i<dim2+1; i++) {
		for (int j=0; j<dim3; j++) W2[i][j] = 2*double(rand())/RAND_MAX-1; // Random double value between -1 and 1
	}

	// Third Weight Matrix
	for (int i=0; i<dim3+1; i++) {
		for (int j=0; j<dim4; j++) W3[i][j] = 2*double(rand())/RAND_MAX-1; // Random double value between -1 and 1
	}
}

int give_prediction(double *y_hat) {
	int arg_max = 0;
	double cur_max = y_hat[0];
	for (int i=1; i<dim4; i++) {
		if (y_hat[i]>cur_max) {
			arg_max = i;
			cur_max = y_hat[arg_max];
		}
	}
	return arg_max;
}

int *give_y(int y) {
	static int y_vector[dim4];
	for (int i=0; i<dim4;i++) y_vector[i]=0;
	y_vector[y] = 1;
	return y_vector;
}

void read_train_data() {
    ifstream csvread;
    csvread.open("mnist_train.csv", ios::in);
    if(csvread){
        //Datei bis Ende einlesen und bei ';' strings trennen
        string s;
				int data_pt = 0;
        while(getline(csvread, s)){
					stringstream ss(s);
					int pxl = 0;
					while( ss.good() ) {
						string substr;
						getline(ss, substr,',');
						if (pxl == 0) {
							label_train[data_pt] = stoi(substr);
						} else {
							data_train[data_pt][pxl-1] = stoi(substr);
						}
						pxl++;
					}
					data_pt++;
        }
        csvread.close();
    }
    else{
        cerr << "Fehler beim Lesen!" << endl;
    }
}
void read_test_data() {
    ifstream csvread;
    csvread.open("mnist_test.csv", ios::in);
    if(csvread){
        //Datei bis Ende einlesen und bei ';' strings trennen
        string s;
				int data_pt = 0;
        while(getline(csvread, s)){
					stringstream ss(s);
					int pxl = 0;
					while( ss.good() ) {
						string substr;
						getline(ss, substr,',');
						if (pxl == 0) {
							label_test[data_pt] = stoi(substr);
						} else {
							data_test[data_pt][pxl-1] = stoi(substr);
						}
						pxl++;
					}
					data_pt++;
        }
        csvread.close();
    }
    else{
        cerr << "Fehler beim Lesen!" << endl;
    }
}

int main() {
	initialise_weights();
	
	//Data
	read_train_data();
	read_test_data();

	// Train
	for (int j=0; j<40; j++) {
		for (int i=0; i<60000; i++) {
			forward_pass(data_train[i]);
			backward_pass(data_train[i], give_y(label_train[i]));
		}
		cout << "Epoch " << j << " done." << endl;

		// Eval
		int cor=0;
		for (int i=0; i<10000; i++) {
			forward_pass(data_test[i]);
			if (give_prediction(ac3) == label_test[i]) cor++;
		}
		cout << "Result: " << static_cast<double>(cor)/10000 << endl;
	}

	return 0;
}
