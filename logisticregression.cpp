#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

using std::cout, std::endl, std::string, std::vector, std::inner_product, std::cin, std::stod;

static vector<vector<double>> X;
static vector<double> Y;
static vector<double> theta;

double g(double z) { // Sigmoid function 
    return 1/(1 + exp(-1 * z));
}

double dot(vector<double>& v1, vector<double>& v2) { // Dot product
    return inner_product(v1.begin(), v1.end(), v2.begin(), 0); 
}

double likely(int num_of_training_ex) { //log likelyhood function
    double sum = 0;
    for (int i = 0; i < num_of_training_ex; ++i) {
        double h = g(dot(theta, X[i]));
        sum += Y[i] * log(h) + (1-Y[i]) * log(1-h);
    }
    return sum;
}

vector<string> split_string(string str, char delimiter){
    vector<string> result;
    string current = ""; 
    for(int i = 0; i < str.size(); ++i){
        if(str[i] == delimiter){
            if(current != ""){
                result.push_back(current);
                current = "";
            } 
            continue;
        }
        current += str[i];
    }
    if(current.size() != 0){
        result.push_back(current);  
    }
    return result;
}

void get_training_data() {
    int num_of_training_ex = 0;
    string x_values = "";
    string y_value = "";
    
    while (true) {
        num_of_training_ex++;
        cout << "Enter training example x-value(s) #" << num_of_training_ex << ", seperated by a comma (Type 'Q' to finish): ";
        cin >> x_values;
        if (x_values == "Q") {
            break;
        }

        vector<string> x_values_vec = split_string(x_values, ',');
        vector<double> x_values_to_push = {1};
        for (int i = 0; i < x_values_vec.size(); ++i) {
            x_values_to_push.push_back(stod(x_values_vec[i]));
        }
        X.push_back(x_values_to_push);
        
        cout << endl;
        do {
            cout << "Enter training example y-value {0, 1} #" << num_of_training_ex << ": ";
            cin >> y_value;

        } while (y_value != "0" && y_value != "1");
        
        Y.push_back(stod(y_value));
    }
    for (int i = 0; i < X[0].size(); ++i) { //setting thetas to 0's
        theta.push_back(0);
    }
}


void find_optimal_theta_parameters() {
    double learning_rate = 0.01;
    bool is_converged = false;
    int iters = 0;
    double epsilon = 0.01;

    while (!is_converged) {
        vector<double> temp(theta.size(), 0);

        for (int j = 0; j < theta.size(); ++j) {
            double result = 0;
            for (int i = 0; i < X.size(); ++i) {
                double h  = g(dot(theta, X[i]));
                result += (Y[i] - h) * X[i][j];
            }
            temp[j] = theta[j] + learning_rate * result;
        }
        theta = temp;
        iters++;

        if (iters == 100000 || fabs(likely(X.size())) <= epsilon) {
            is_converged = true;
        }
    }
}
void print_equation() {
    cout << "Predicted decision boundary equation: Y = ";
    for (int i = 0; i < theta.size(); ++i) {
        if (i == 0) {
            cout << theta[i] << " + ";
        }else if (i == theta.size() - 1){
            cout << theta[i] << "x" << i << endl;;       
        }else{
            cout << theta[i] << "x" << i << " + ";
        }
    }
}
void print_probability() {
    cout << "Probability of predictions accurate: " << exp(likely(X.size())) << endl; 
}
int main() {
    get_training_data();
    find_optimal_theta_parameters();
    print_equation();
    print_probability();
    return 0;   
}