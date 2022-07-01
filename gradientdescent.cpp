#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using std::cout, std::endl, std::cin, std::vector, std::string, std::stod;

static vector<vector<double>> X; // features
static vector<double> Y; // targets
static vector<double> theta; // parameters

// Helper method 
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
        cout << "Enter training example y-value #" << num_of_training_ex << ": ";
        cin >> y_value;
        
        Y.push_back(stod(y_value));
    }
    for (int i = 0; i < X[0].size(); ++i) { //setting thetas to 0's
        theta.push_back(0);
    }
}
// Hypothesis function
double h(vector<double> &x){
    double result = 0;
    for (int i = 0; i < theta.size(); ++i){
        result += theta[i] * x[i];
    }
    return result;
}
// Cost/error function
double J(){ 
    double result = 0;
    for (int i = 0; i < X.size(); ++i){
        result += pow(h(X[i]) - Y[i], 2); 
    }
    return result/(2 * X.size());
}

void find_optimal_theta_parameters() {
    double learning_rate = 0.01;
    bool is_converged = false;
    int iters = 0;
    double epsilon = 0.00000001;

    while (!is_converged) {
        double old_cost = J();
        vector<double> temp(theta.size(), 0);

        for (int j = 0; j < theta.size(); ++j) {
            double result = 0;
            for (int i = 0; i < X.size(); ++i) {
                result += (h(X[i]) - Y[i]) * X[i][j];
            }
            result /= (2 * X.size());
            temp[j] = theta[j] - learning_rate * result;
        }
        theta = temp;
        iters++;
        if (iters == 100000 || fabs(J() - old_cost) <= epsilon) {
            is_converged = true;
        }
    }
}

void print_equation() {
    cout << ":::::::::::::::::::::PREDICTED EQUATION::::::::::::::::::::::::" << endl; 
    cout << "Y = " << theta[1] << "x + " << theta[0] << endl;
}

int main() {
    get_training_data();
    find_optimal_theta_parameters();
    print_equation();
    return 0;
}