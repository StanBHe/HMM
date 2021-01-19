#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <algorithm>
#include <iterator>
#include <windows.h>
using namespace std;

void process(int numOfTrainingModels, int numOfTestModels)
{
    string datasets[3] = { "winwebsec", "zbot", "zeroaccess" };
    string trainSavePath = "./NeuralNetworkData/trainData.txt";
    //string testSavePath = "./NeuralNetworkData/testData.txt";
    
    ofstream trainData(trainSavePath);
    //ofstream testData(testSavePath);
    for (int i = 0; i < 3; i++)
    {
        string malSample = datasets[i];
        string trainReadPath = "./ProcessedData/" + malSample + "/models/training/";
        string testReadPath = "./ProcessedData/" + malSample + "/models/test/";
        cout << "reading training" << endl;
        for (int j = 0; j < numOfTrainingModels; j++)
        {
            //read from model
            ifstream modelData(trainReadPath + to_string(j) + ".txt");
            string text;
            getline(modelData, text);
            const int N = stoi(text);
            getline(modelData, text);
            const int M = stoi(text);
           
            getline(modelData, text);
            int counter = 0;
            while (counter < N)
            {
                getline(modelData, text);
                istringstream ss(text);
                string value;

                while (ss >> value)
                {
                    trainData << value << ",";
                }
                counter++;
            }
            counter = 0;
            while (counter < N)
            {
                getline(modelData, text);
                istringstream ss(text);
                string value;
              
                while (ss >> value)
                {
                    trainData << value << ",";
                }
                counter++;
            } 

            modelData.close();
            trainData << i <<"\n";
        }
        cout << "reading testing" << endl;
        for (int j = 0; j < numOfTestModels; j++)
        {
            //read from model
            ifstream modelData(testReadPath + to_string(j) + ".txt");
            string text;
            getline(modelData, text);
            const int N = stoi(text);
            getline(modelData, text);
            const int M = stoi(text);

            getline(modelData, text);
            int counter = 0;
            while (counter < N)
            {
                getline(modelData, text);
                istringstream ss(text);
                string value;

                while (ss >> value)
                {
                    trainData << value << ",";
                }
                counter++;
            }
            counter = 0;
            while (counter < N)
            {
                getline(modelData, text);
                istringstream ss(text);
                string value;

                while (ss >> value)
                {
                    trainData << value << ",";
                }
                counter++;
            }

            modelData.close();
            trainData << i << "\n";
        }
        
    }
    trainData.close();
    cout << "done" << endl;
}

int main()
{
    cout << "starting" << endl;
    process(999,99);
    return 1;
}