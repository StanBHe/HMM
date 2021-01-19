#include <stdio.h>
#include <iostream>
#include <ctime>
#include <random>
#include <thread>
#include <future>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <math.h>

using namespace std;

/**
* This is the struct for the model
* @param n, the number of states
* @param m, the max number of observations
* @param a, the a matrix
* @param b, the b matrix
* @param pi, the pi matrix
* @param c, the scaling factors for calculating
**/
struct Model
{
    double n;
    double m;
    double** a;
    double** b;
    double* pi;
    double* c;
};

/**
* Initializes a new Model with the given parameters
* @param N, the number of states
* @param M, the max number of observations
* @param T, the length of the observation sequence
* @return Model, the model to return
**/
Model initialize(int N, int M, int T)
{
    //std::cout << "initalizing model" << std::endl;
    struct Model model;
    model.n = N;
    model.m = M;
    model.a = new double* [N];
    model.b = new double* [N];
    model.pi = new double[N];
    model.c = new double[T];
    static std::mt19937 gen(std::time(nullptr));

    for (int i = 0; i < N; i++)
    {
        model.a[i] = new double[N];
        double mean = 1.0 / N;
        double stdif = mean * 0.2;
        std::normal_distribution<double> d(mean, stdif);
        double total = 0;
        for (int j = 0; j < N; j++)
        {
            double sample = d(gen);
            total += sample;
            model.a[i][j] = sample;
        }
        for (int j = 0; j < N; j++)
        {
            model.a[i][j] = model.a[i][j] / total;
        }
    }
    for (int i = 0; i < N; i++)
    {
        model.b[i] = new double[M];
        double mean = 1.0 / M;
        double stdif = mean * 0.2;
        std::normal_distribution<double> d(mean, stdif);
        double total = 0;
        for (int j = 0; j < M; j++)
        {
            double sample = d(gen);
            total += sample;
            model.b[i][j] = sample;
        }
        for (int j = 0; j < M; j++)
        {
            model.b[i][j] = model.b[i][j] / total;
        }
    }
    double total = 0;
    double mean = 1.0 / N;
    double stdif = mean * 0.15;
    std::normal_distribution<double> d(mean, stdif);
    for (int i = 0; i < N; i++)
    {
        double sample = d(gen);
        total += sample;
        model.pi[i] = sample;
    }
    for (int i = 0; i < N; i++)
    {
        model.pi[i] = model.pi[i] / total;
    }
    //std::cout << "model initalized" << std::endl;
    return model;
}

/**
* Initalizes a model read from the given path 
* @param path, the path to read from
* @return Model, the model to return
*/
Model readFrom(string path)
{
    std::ifstream MyReadFile(path);
    std::string myText;
    std::getline(MyReadFile, myText);
    const int N = stoi(myText);
    std::getline(MyReadFile, myText);
    const int M = stoi(myText);
    struct Model model;
    model.n = N;
    model.m = M;
    model.a = new double* [N];
    model.b = new double* [N];
    model.pi = new double[N];
    for (int i = 0; i < N; i++)
    {
        model.a[i] = new double[N];
    }
    for (int i = 0; i < N; i++)
    {
        model.b[i] = new double[M];
    }

    std::getline(MyReadFile, myText);
    istringstream ss(myText);
    string value;
    int x = 0;
    while (ss >> value)
    {
        model.pi[x] = stod(value);
        x++;
    }
    int y = 0;
    while (y < N)
    {
        std::getline(MyReadFile, myText);
        istringstream ss2(myText);
        string value;
        x = 0;
        while (ss2 >> value)
        {
            model.a[y][x] = stod(value);
            x++;
        }
        y++;
    }
    y = 0;
    while (y < N)
    {
        std::getline(MyReadFile, myText);
        istringstream ss2(myText);
        string value;
        x = 0;
        while (ss2 >> value)
        {
            //cout << value << endl;
            model.b[y][x] = stod(value);
            x++;
        }
        y++;
    }
    MyReadFile.close();
    return model;
}

/**
* The alpha pass of HMM
 * @param model, the model for HMM 
 * @param alpha, the 2dmatrix to store the probabilities for each state in
 * @param observations, the observed observation sequence
 * @param T, the length of the observation sequence
 **/
void alphaPass(Model model, double** alpha, int observations[], int T)
{
    //cout << model.n << endl;
    int N = model.n;
    model.c[0] = 0;
    //initalize the probabilites to see each state at the first observation
    for (int i = 0; i < N; i++)
    {
        alpha[0][i] = model.pi[i] * model.b[i][observations[0]];
        model.c[0] += alpha[0][i];
    }
    //calculate the scaling factor and apply it
    model.c[0] = 1 / model.c[0];
    for (int i = 0; i < N; i++)
    {
        alpha[0][i] = model.c[0] * alpha[0][i];
    }
    //for each observation, calculate the probability to see each state given the previous calculations
    for (int t = 1; t < T; t++)
    {
        model.c[t] = 0;
        for (int i = 0; i < N; i++)
        {
            alpha[t][i] = 0;
            for (int j = 0; j < N; j++)
            {
                alpha[t][i] = alpha[t][i] + alpha[t - 1][j] * model.a[j][i];
            }
            alpha[t][i] = alpha[t][i] * model.b[i][observations[t]];
            model.c[t] = model.c[t] + alpha[t][i];
        }
        model.c[t] = 1 / model.c[t];
        for (int i = 0; i < N; i++)
        {
            alpha[t][i] = model.c[t] * alpha[t][i];
        }
    }
 
}

/**
* The beta pass of HMM
 * @param model, the model for HMM 
 * @param beta, the 2dmatrix to store the probabilities for each state in
 * @param observations, the observed observation sequence
 * @param T, the length of the observation sequence
 **/
void betaPass(Model model, double** beta, int observations[], int T)
{
    int N = model.n;
    for (int i = 0; i < N; i++)
    {
        beta[T - 1][i] = model.c[T - 1];
    }

    for (int t = T - 2; t >= 0; t--)
    {
        for (int i = 0; i < N; i++)
        {
            beta[t][i] = 0;
            for (int j = 0; j < N; j++)
            {
                beta[t][i] = beta[t][i] + model.a[i][j] * model.b[j][observations[t + 1]] * beta[t + 1][j];
            }
            beta[t][i] = model.c[t] * beta[t][i];
        }
    }
}

/**
* The gamma pass of HMM
 * @param model, the model for HMM 
 * @param alpha, the 2dmatrix to retrieve the probabilities for each state from
 * @param beta, the 2dmatrix to retrieve the probabilities for each state from
 * @param digamma, the 3dmatrix to store the probabilities of each state in
 * @param gamma, the 3dmatrix to store the probabilities of each state in
 * @param observations, the observed observation sequence
 * @param T, the length of the observation sequence
 **/
void gammaPass(Model model, double** alpha, double** beta, double*** digamma, double** gamma, int observations[], int T)
{
    int N = model.n;
    for (int t = 0; t < T - 1; t++)
    {
        double denom = 0;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                denom = denom + alpha[t][i] * model.a[i][j] * model.b[j][observations[t + 1]] * beta[t + 1][j];
            }
        }

        for (int i = 0; i < N; i++)
        {
            gamma[t][i] = 0;
            for (int j = 0; j < N; j++)
            {
                digamma[t][i][j] = (alpha[t][i] * model.a[i][j] * model.b[j][observations[t + 1]] * beta[t + 1][j]) / denom;
                gamma[t][i] = gamma[t][i] + digamma[t][i][j];
            }
        }
    }
    double denom = 0;
    for (int i = 0; i < N; i++)
    {
        denom = denom + alpha[T - 1][i];
    }
    for (int i = 0; i < N; i++)
    {
        gamma[T - 1][i] = alpha[T - 1][i] / denom;
    }
}

/**
* Given the gamma and digammas, update the a, b, and pi matrices of the model
* @param model, the model of HMM
* @param gamma, the 3dmatrix to retrieve the probabilities for each state from
* @param digamma, the 3dmatrix to retrieve the probabilities for each state from
* @param observations, the observed observation sequence
* @param T, the length of the observation sequence
**/
void reestimate(Model model, double** gamma, double*** digamma, int observations[], int T)
{
    int N = model.n;
    int M = model.m;
    for (int i = 0; i < N; i++)
    {
        model.pi[i] = gamma[0][i];
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double numer = 0;
            double denom = 0;
            for (int t = 0; t < T - 1; t++)
            {
                numer = numer + digamma[t][i][j];
                denom = denom + gamma[t][i];
            }
            model.a[i][j] = numer / denom;
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            double numer = 0;
            double denom = 0;
            for (int t = 0; t < T; t++)
            {
                if (observations[t] == j)
                {
                    numer = numer + gamma[t][i];
                }
                denom = denom + gamma[t][i];
            }
            model.b[i][j] = numer / denom;
        }
    }
}

/**
* Given a model that has gone through alpha pass, calculate its accuracy (closer to 0 the better)
* @param model, the model of HMM
* @param T, the Length of the observation sequence
* @return double, the score of the model
**/
double evaluateModel(Model model, int T)
{
    double logProb = 0;
    for (int i = 0; i < T; i++)
    {
        //cout << model.c[i] << endl;
        logProb = logProb + log(model.c[i]);
    }
    logProb = -logProb;
    return logProb;
}

/**
* Generate trained HMM model completely randomly from the given parameters
* @param maxIters, how many iterations to train the model
* @param observations, the observation sequence
* @param N, the number of states
* @param M, the max number of observations
* @param T, the length of the observation sequence
* @return model, the trained model 
**/
Model generateHMM(int maxIters, int observations[], int N, int M, int T)
{
    Model trainingModel = initialize(N, M, T);
    //std::cout << "initalize alpha,beta,gamma,digamma" << std::endl;
    double** alpha = new double* [T];
    double** beta = new double* [T];
    double** gamma = new double* [T];
    double*** digamma = new double** [T];
    for (int i = 0; i < T; i++)
    {
        alpha[i] = new double[N];
        beta[i] = new double[N];
        gamma[i] = new double[N];
    }
    for (int i = 0; i < T; i++)
    {
        digamma[i] = new double* [N];
        for (int j = 0; j < N; j++)
        {
            digamma[i][j] = new double[N];
        }
    }
    //std::cout << "finish initalize alpha,beta,gamma,digamma" << std::endl;
    int iters = 0;
    double oldLog = -std::numeric_limits<double>::max();
    double currentLog = -std::numeric_limits<double>::max();
    //main loop
    do
    {
        alphaPass(trainingModel, alpha, observations, T);
        betaPass(trainingModel, beta, observations, T);
        gammaPass(trainingModel, alpha, beta, digamma, gamma, observations, T);
        reestimate(trainingModel, gamma, digamma, observations, T);
        oldLog = currentLog;
        currentLog = evaluateModel(trainingModel, T);
        iters++;
        //printf("%lf, %d\n", currentLog, iters);
    } while (iters < maxIters && currentLog > oldLog);
    //std::cout << "loop finished" << std::endl;
    for (int i = 0; i < T; i++)
    {
        delete[] alpha[i];
        delete[] beta[i];
        delete[] gamma[i];
    }
    delete[] alpha;
    delete[] beta;
    delete[] gamma;

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < N; j++)
        {
            delete[] digamma[i][j];
        }
        delete[] digamma[i];
    }
    delete[] digamma;
    return trainingModel;
}
/**
* Generate trained HMM model from the given model
* @param maxIters, how many iterations to train the model
* @param observations, the observation sequence
* @param N, the number of states
* @param M, the max number of observations
* @param T, the length of the observation sequence
* @return model, the trained model
**/
Model generateHMM(Model m, int maxIters, int observations[], int N, int M, int T)
{
    Model trainingModel = m;
    //std::cout << "initalize alpha,beta,gamma,digamma" << std::endl;
    double** alpha = new double* [T];
    double** beta = new double* [T];
    double** gamma = new double* [T];
    double*** digamma = new double** [T];
    for (int i = 0; i < T; i++)
    {
        alpha[i] = new double[N];
        beta[i] = new double[N];
        gamma[i] = new double[N];
    }
    for (int i = 0; i < T; i++)
    {
        digamma[i] = new double* [N];
        for (int j = 0; j < N; j++)
        {
            digamma[i][j] = new double[N];
        }
    }
    //std::cout << "finish initalize alpha,beta,gamma,digamma" << std::endl;
    int iters = 0;
    double oldLog = -std::numeric_limits<double>::max();
    double currentLog = -std::numeric_limits<double>::max();
    //main loop
    do
    {
        alphaPass(trainingModel, alpha, observations, T);
        betaPass(trainingModel, beta, observations, T);
        gammaPass(trainingModel, alpha, beta, digamma, gamma, observations, T);
        reestimate(trainingModel, gamma, digamma, observations, T);
        oldLog = currentLog;
        currentLog = evaluateModel(trainingModel, T);
        iters++;
        printf("%lf, %d\n", currentLog, iters);
        //std::cout << "old log: " << oldLog << " new log: " << currentLog << std::endl;

    } while (iters < maxIters && currentLog > oldLog);
    //std::cout << "loop finished" << std::endl;
    for (int i = 0; i < T; i++)
    {
        delete[] alpha[i];
        delete[] beta[i];
        delete[] gamma[i];
    }
    delete[] alpha;
    delete[] beta;
    delete[] gamma;

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < N; j++)
        {
            delete[] digamma[i][j];
        }
        delete[] digamma[i];
    }
    delete[] digamma;
    return trainingModel;
}

/*
* Helper function for boostedHMM, that calls generateHMM
* @param arr, the array to store the trained HMM model
* @param index, the index to save the model to
* @param maxIters, the max iterations to train the models
* @param observations, the observation sequence
* @param N, the number of states
* @param M, the max number of observations
* @param T, the length of the observation sequence
*/
void boostResult(Model arr[], int index, int maxIters, int observations[], int N, int M, int T)
{
    //std::cout << "boostresult" << index << std::endl;
    arr[index] = generateHMM(maxIters, observations, N, M, T);
    //std::cout << "boostresultfinished" << index << std::endl;
}

/**
* Applying boosting to get the best accuracy on an HMM model
* @param numberOfModels, the number of models to compare each other to
* @param maxIters, the max iterations to train each model
* @param observations, the observation sequence
* @param N, the number of states
* @param M, the max number of observations
* @param T, the length of the observation sequence
* @return model, the boosted HMM model
**/
Model boostedHMM(int numberOfModels, int maxIters, int observations[], int N, int M, int T)
{
    Model* trainedModels = new Model[numberOfModels];
    std::thread* threads = new std::thread[numberOfModels];
    //std::cout << "creating threads" << std::endl;
    for (int i = 0; i < numberOfModels; i++)
    {
        threads[i] = std::thread(boostResult, trainedModels, i, maxIters, observations, N, M, T);
    }
    //std::cout << "joining threads" << std::endl;
    for (int i = 0; i < numberOfModels; i++)
    {
        //std::cout << "joining" << i << std::endl;
        threads[i].join();
    }
    std::cout << "trying max" << std::endl;
    Model best = trainedModels[0];
    for (int i = 1; i < numberOfModels; i++)
    {
        if (evaluateModel(best, T) < evaluateModel(trainedModels[i], T))
        {
            best = trainedModels[i];
        }
    }
    std::cout << "done max" << std::endl;
    delete[] threads;
    std::cout << "delete" << std::endl;
    return best;
}

/**
* Save a model to the given path
* @param model, the model to save
* @param path, the path to save the model to
**/
void saveModel(Model m, string path)
{
    Model trained = m;
    const int N = trained.n;
    const int M = trained.m;
    ofstream MyFile(path);
    MyFile << N << "\n";
    MyFile << M << "\n";
    for (int i = 0; i < N; ++i)
    {
        MyFile << trained.pi[i];
        if (i != N - 1)
        {
            MyFile << ' ';
        }
    }
    MyFile << "\n";

    //save a
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            MyFile << trained.a[i][j];
            if (j != N - 1)
            {
                MyFile << ' ';
            }
        }
        MyFile << "\n";
    }
    //save b
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            MyFile << trained.b[i][j];
            if (j != M - 1)
            {
                MyFile << ' ';
            }
        }
        MyFile << "\n";
    }

    MyFile.close();
}

/**
*Used for getting the number of lines in a file
* @param path, the file path to read from
* @return int, the number of lines in the file
*/
int numOfLines(string path)
{
    std::ifstream MyReadFile(path);
    std::string line;
    int counter = 0;
    while (std::getline(MyReadFile, line))
    {
        counter++;
    }
    counter--;
    MyReadFile.close();
    return counter;
}

/**
* Used for creating HMM models of the given malware sample
* @param malName, the malware sample to read from
* @param n, the number of states
* @param m, the max number of observations
* @param numOfTrainModels, how many training models to make
* @param numOfTestModels, how many test models to make
*/
void createMalHMM(string malName, int n, int m, int numOfTrainModels, int numOfTestModels)
{
    //std::cout << "starting " << std::endl;
    const int N = n;
    const int M = m;

    //set up where to read from
    string trainingDataPath = "./ProcessedData/" + malName + "/trainData/"; 
    string testDataPath = "./ProcessedData/" + malName + "/testData/";

    //set up where to save to
    string trainingSavePath = "./ProcessedData/" + malName + "/models/training/";
    string testSavePath = "./ProcessedData/" + malName + "/models/test/";

    //for number specified, create an HMM model
    for (int i = 0; i < numOfTrainModels; i++)
    {
        cout << malName << " training " << i << endl;
        //create path to file to read from
        string filePath = trainingDataPath + to_string(i) + ".txt";
        //collect # of lines
        const int T = numOfLines(filePath);
        //cout << T << endl;
        int* observations = new int[T];
        //read from file
        std::ifstream MyReadFile(filePath);
        std::string myText;
        int counter = 0;
        std::getline(MyReadFile, myText);
        //cout << "creating array" << endl;
        //until file ends, populate the observations array
        while (std::getline(MyReadFile, myText))
        {
            observations[counter] = stoi(myText);
            counter++;
        }
        //cout << counter << endl;
        //std::cout << "starting" << std::endl;

        Model trained = generateHMM(100, observations, N, M, T);
        //std::cout << "done" << std::endl;

        //save model
        saveModel(trained, trainingSavePath + to_string(i)+".txt");
    }

    for (int i = 0; i < numOfTestModels; i++)
    {
        cout << malName << " testing " << i << endl;
        //create path to file to read from
        string filePath = testDataPath + to_string(i) + ".txt";
        //collect # of lines
        const int T = numOfLines(filePath);
        int* observations = new int[T];
        //read from file
        std::ifstream MyReadFile(filePath);
        std::string myText;
        int counter = 0;
        std::getline(MyReadFile, myText);
        //cout << "creating array" << endl;
        //until file ends, populate the observations array
        while (std::getline(MyReadFile, myText))
        {
            observations[counter] = stoi(myText);
            counter++;
        }
        //cout << counter << endl;
        //std::cout << "starting" << std::endl;

        Model trained = generateHMM(100, observations, N, M, T);
        //std::cout << "done" << std::endl;

        //save model
        saveModel(trained, testSavePath + to_string(i) +".txt");
    }
}

/**
void zeroaccessvsothers(int selection)
{
    const int N = 2;
    const int M = 30;
    string datasets[2] = {"winwebsec", "zbot"};
    string savePath = "./Experiments/zeroaccessvs" + datasets[selection] + "/results.txt";
    Model trained = readFrom("./Models/zeroaccess/model.txt");
    std::ifstream MyReadFile("./ProcessedData/zeroaccess/testDataFile" + datasets[selection] + ".txt");
    std::ofstream saveFile(savePath);
    std::string path;
    std::cout << "reading" << std::endl;
    while (std::getline(MyReadFile, path))
    {
        std::ifstream readData(path);
        std::string observation;
        //cout << "creating array" << endl;
        const int T = numOfLines(path);
        int *observations = new int[T];
        trained.c = new double[T];
        int counter = 0;
        while (std::getline(readData, observation))
        {
            observations[counter] = stoi(observation);
            counter++;
        }

        double **alpha = new double *[T];
        for (int i = 0; i < T; i++)
        {
            alpha[i] = new double[N];
        }
        alphaPass(trained, alpha, observations, N, T);
        for (int i = 0; i < T; i++)
        {
            delete[] alpha[i];
        }
        delete[] alpha;
        double result = evaluateModel(trained, T);
        delete observations;
        delete trained.c;
        saveFile << result << " " << 0 << "\n";
    }
    std::ifstream ReadOwnFiles("./ProcessedData/zeroaccess/testDataFile.txt");
    std::ofstream owntraining("./Experiments/zeroaccessvszeroaccess/results.txt");
    std::cout << "reading" << std::endl;
    while (std::getline(ReadOwnFiles, path))
    {
        std::ifstream readData(path);
        std::string observation;
        //cout << "creating array" << endl;
        const int T = numOfLines(path);
        int *observations = new int[T];
        trained.c = new double[T];
        int counter = 0;
        while (std::getline(readData, observation))
        {
            observations[counter] = stoi(observation);
            counter++;
        }

        double **alpha = new double *[T];
        for (int i = 0; i < T; i++)
        {
            alpha[i] = new double[N];
        }
        alphaPass(trained, alpha, observations, N, T);
        for (int i = 0; i < T; i++)
        {
            delete[] alpha[i];
        }
        delete[] alpha;
        double result = evaluateModel(trained, T);
        delete observations;
        delete trained.c;
        saveFile << result << " " << 1 << "\n";
        owntraining << result << " " << 1 << "\n";
    }
    std::cout << "done" << std::endl;
}

void zbotvsothers(int selection)
{
    const int N = 2;
    const int M = 30;
    string datasets[2] = {"winwebsec", "zeroaccess"};
    string savePath = "./Experiments/zbotvs" + datasets[selection] + "/results.txt";
    Model trained = readFrom("./Models/zbot/model.txt");
    std::ifstream MyReadFile("./ProcessedData/zbot/testDataFile" + datasets[selection] + ".txt");
    std::ofstream saveFile(savePath);
    std::string path;
    std::cout << "reading" << std::endl;
    while (std::getline(MyReadFile, path))
    {
        std::ifstream readData(path);
        std::string observation;
        //cout << "creating array" << endl;
        const int T = numOfLines(path);
        int *observations = new int[T];
        trained.c = new double[T];
        int counter = 0;
        while (std::getline(readData, observation))
        {
            observations[counter] = stoi(observation);
            counter++;
        }

        double **alpha = new double *[T];
        for (int i = 0; i < T; i++)
        {
            alpha[i] = new double[N];
        }
        alphaPass(trained, alpha, observations, N, T);
        for (int i = 0; i < T; i++)
        {
            delete[] alpha[i];
        }
        delete[] alpha;
        double result = evaluateModel(trained, T);
        delete observations;
        delete trained.c;
        saveFile << result << " " << 0 << "\n";
    }
    std::ifstream ReadOwnFiles("./ProcessedData/zbot/testDataFile.txt");
    std::ofstream owntraining("./Experiments/zbotvszbot/results.txt");
    std::cout << "reading" << std::endl;
    while (std::getline(ReadOwnFiles, path))
    {
        std::ifstream readData(path);
        std::string observation;
        //cout << "creating array" << endl;
        const int T = numOfLines(path);
        int *observations = new int[T];
        trained.c = new double[T];
        int counter = 0;
        while (std::getline(readData, observation))
        {
            observations[counter] = stoi(observation);
            counter++;
        }

        double **alpha = new double *[T];
        for (int i = 0; i < T; i++)
        {
            alpha[i] = new double[N];
        }
        alphaPass(trained, alpha, observations, N, T);
        for (int i = 0; i < T; i++)
        {
            delete[] alpha[i];
        }
        delete[] alpha;
        double result = evaluateModel(trained, T);
        delete observations;
        delete trained.c;
        saveFile << result << " " << 1 << "\n";
        owntraining << result << " " << 1 << "\n";
    }
    std::cout << "done" << std::endl;
}

void winwebsecvsothers(int selection)
{
    const int N = 2;
    const int M = 30;
    string datasets[2] = {"zbot", "zeroaccess"};
    string savePath = "./Experiments/winwebsecvs" + datasets[selection] + "/results.txt";
    Model trained = readFrom("./Models/winwebsec/model.txt");
    std::ifstream MyReadFile("./ProcessedData/winwebsec/testDataFile" + datasets[selection] + ".txt");
    std::ofstream saveFile(savePath);
    std::string path;
    std::cout << "reading" << std::endl;
    while (std::getline(MyReadFile, path))
    {
        std::ifstream readData(path);
        std::string observation;
        //cout << "creating array" << endl;
        const int T = numOfLines(path);
        int *observations = new int[T];
        trained.c = new double[T];
        int counter = 0;
        while (std::getline(readData, observation))
        {
            observations[counter] = stoi(observation);
            counter++;
        }

        double **alpha = new double *[T];
        for (int i = 0; i < T; i++)
        {
            alpha[i] = new double[N];
        }
        alphaPass(trained, alpha, observations, N, T);
        for (int i = 0; i < T; i++)
        {
            delete[] alpha[i];
        }
        delete[] alpha;
        double result = evaluateModel(trained, T);
        delete observations;
        delete trained.c;
        saveFile << result << " " << 0 << "\n";
    }
    std::ifstream ReadOwnFiles("./ProcessedData/winwebsec/testDataFile.txt");
    std::ofstream owntraining("./Experiments/winwebsecvswinwebsec/results.txt");
    std::cout << "reading" << std::endl;
    while (std::getline(ReadOwnFiles, path))
    {
        std::ifstream readData(path);
        std::string observation;
        //cout << "creating array" << endl;
        const int T = numOfLines(path);
        int *observations = new int[T];
        trained.c = new double[T];
        int counter = 0;
        while (std::getline(readData, observation))
        {
            observations[counter] = stoi(observation);
            counter++;
        }

        double **alpha = new double *[T];
        for (int i = 0; i < T; i++)
        {
            alpha[i] = new double[N];
        }
        alphaPass(trained, alpha, observations, N, T);
        for (int i = 0; i < T; i++)
        {
            delete[] alpha[i];
        }
        delete[] alpha;
        double result = evaluateModel(trained, T);
        delete observations;
        delete trained.c;
        saveFile << result << " " << 1 << "\n";
        owntraining << result << " " << 1 << "\n";
    }
    std::cout << "done" << std::endl;
}

void zeroaccess()
{
    std::cout << "starting " << std::endl;
    const int N = 2;
    const int M = 30;
    const int T = 5734264;
    int *observations = new int[T];
    std::cout << "reading" << std::endl;
    std::ifstream MyReadFile("./ProcessedData/zeroaccess/trainData.txt");
    std::string myText;
    int counter = 0;
    std::getline(MyReadFile, myText);
    cout << "creating array" << endl;
    while (std::getline(MyReadFile, myText))
    {
        observations[counter] = stoi(myText);
        counter++;
    }
    //cout << counter << endl;
    std::cout << "starting" << std::endl;

    Model trained = generateHMM(100, observations, N, M, T);
    std::cout << "done" << std::endl;

    //save model
    saveModel(trained, N, M, "./Models/zeroaccess/model.txt");
    //
    for (int i = 0; i < N; ++i)
    {
        std::cout << trained.pi[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << trained.a[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            std::cout << trained.b[i][j] << endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //
}

void zbot()
{
    std::cout << "starting " << std::endl;
    const int N = 2;
    const int M = 30;
    const int T = 1951246;
    int *observations = new int[T];
    std::cout << "reading" << std::endl;
    std::ifstream MyReadFile("./ProcessedData/zbot/trainData.txt");
    std::string myText;
    int counter = 0;
    std::getline(MyReadFile, myText);
    cout << "creating array" << endl;
    while (std::getline(MyReadFile, myText))
    {
        observations[counter] = stoi(myText);
        counter++;
    }
    //cout << counter << endl;
    std::cout << "starting" << std::endl;

    Model trained = generateHMM(100, observations, N, M, T);
    std::cout << "done" << std::endl;

    //save model
    saveModel(trained, N, M, "./Models/zbot/model.txt");
    //
    for (int i = 0; i < N; ++i)
    {
        std::cout << trained.pi[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << trained.a[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            std::cout << trained.b[i][j] << endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //
}

void winwebsec()
{
    std::cout << "starting " << std::endl;
    const int N = 2;
    const int M = 30;
    const int T = 3405216;
    int *observations = new int[T];
    std::cout << "reading" << std::endl;
    std::ifstream MyReadFile("./ProcessedData/winwebsec/trainData.txt");
    std::string myText;
    int counter = 0;
    std::getline(MyReadFile, myText);
    cout << "creating array" << endl;
    while (std::getline(MyReadFile, myText))
    {
        observations[counter] = stoi(myText);
        counter++;
    }
    //cout << counter << endl;
    std::cout << "starting" << std::endl;

    Model trained = generateHMM(50, observations, N, M, T);
    std::cout << "done" << std::endl;

    //save model
    saveModel(trained, N, M, "./Models/winwebsec/model.txt");
    //
    for (int i = 0; i < N; ++i)
    {
        std::cout << trained.pi[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << trained.a[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            std::cout << trained.b[i][j] << endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //
}
**/
int main()
{
    //createMalHMM("winwebsec", 2, 40, 333, 99);
    thread t1(createMalHMM, "winwebsec", 3, 40, 999, 99);
    thread t2(createMalHMM, "zbot", 3, 40, 999, 99);
    thread t3(createMalHMM, "zeroaccess", 3, 40, 999, 99);
    cout << "Threads are running" << endl;

    t1.join();
    t2.join();
    t3.join();
    cout << "threads are done" << endl;
    return 1;
}