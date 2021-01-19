#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <algorithm>
#include <iterator>
#include <windows.h>
using namespace std;

typedef std::vector<std::string> stringvec;

bool comparator(pair<string, int>& a, pair<string, int>& b)
{
    return a.second > b.second;
}

unordered_map<string, int> sortByValue(unordered_map<string, int> notSortedMap, int maxUniqueSymbols)
{
    vector<pair<string, int>> sortVector;
    unordered_map<string, int> sortedMap;

    for (auto& it : notSortedMap)
    {
        sortVector.push_back(it);
    }

    sort(sortVector.begin(), sortVector.end(), comparator);

    int counter = 0;
    for (auto& it : sortVector)
    {
        cout << it.first<<"  " << it.second << " " << counter << endl;
        sortedMap[it.first] = counter;
        if (counter < maxUniqueSymbols - 1)
        {
            counter++;
        }
    }
    return sortedMap;
}

void read_directory(const std::string& name, stringvec& v)
{
    std::string pattern(name);
    pattern.append("\\*");
    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (data.cFileName[0] != '.')
            {
                v.push_back(data.cFileName);
            }
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
}

void process()
{
    cout << "starting" << endl;
    string myText;
    unordered_map<string, int> occurrence;

    string datasetDirectory = "./DataSet/";
    string testDataDirectory = "./ProcessedData/";
    string datasets[3] = { "winwebsec", "zbot", "zeroaccess" };

    int symbols = 0;
    int symbols2 = 0;
    int maxSamples = 1098;
    int numSamples = 0;
    int maxUniqueSymbols = 40;
    int testSampleNum = 99;

    int currentTest = 0;

    for (int i = 0; i < 3; i++)
    {
        stringvec v;
        read_directory(datasetDirectory + datasets[i], v);
        string path = datasetDirectory + datasets[i] + "/";

        //iterate through all the files in the directory
        for (auto& it : v)
        {
            string totalPath = path + it;
            //std::cout << totalPath << std::endl;
            ifstream MyReadFile(totalPath);

            //count all the opcodes
            while (getline(MyReadFile, myText))
            {
                if (occurrence.count(myText) == 0)
                {
                    occurrence[myText] = 1;
                }
                else
                {
                    occurrence[myText] = occurrence[myText] + 1;
                }

                if (numSamples <= maxSamples - testSampleNum)
                {
                    symbols++;
                }
                else
                {
                    symbols2++;
                }
            }
            MyReadFile.close();
            numSamples++;
            if (numSamples >= maxSamples)
            {
                break;
            }
        }
    }
    
    cout << symbols << endl;
    cout << symbols2 << endl;

    unordered_map<string, int> observations = sortByValue(occurrence, maxUniqueSymbols);

    string symbolTablePath = testDataDirectory + "symboltable.txt";

    //save symboltable
    ofstream symbolTable(symbolTablePath);
    for (unordered_map<string, int>::iterator it = observations.begin(); it != observations.end(); it++)
    {
        symbolTable << it->first << " " << it->second << "\n";
    }
    symbolTable.close();

    //process data from each directory
    for (int i = 0; i < 3; i++)
    {
        stringvec v;
        read_directory(datasetDirectory + datasets[i], v);
        string path = datasetDirectory + datasets[i] + "/";
        string savePath = testDataDirectory + datasets[i] + "/";

        string totalPath2 = savePath + "trainData.txt";
        string totalPath3 = savePath + "testData.txt";
        string totalPath4 = savePath + "testDataFile.txt";
        string totalPath5 = savePath + "testData/";
        
        //start at the beginning of each raw data set
        auto it = std::begin(v);

        int newCounter = 0;
        while (newCounter <= maxSamples - testSampleNum)
        {
            // set up save path
            string trainingDataPath = savePath + "trainData/" + to_string(newCounter) + ".txt";
            ofstream MyFile(trainingDataPath);
            //read from raw file
            string totalPath = path + *it;
            ifstream MyReadFile(totalPath);
            //while file has opcodes left, replace it and write to the processed data path
            while (getline(MyReadFile, myText))
            {
                MyFile << observations[myText] << "\n";
            }
            // Close the file
            MyReadFile.close();

            it++;
            newCounter++;
            MyFile.close();
        }
    
        ofstream testFilesPath(totalPath4);

        int testFileCounter = 0;
        while (newCounter <= maxSamples)
        {
            //read from raw data
            string totalPath = path + *it;
            //set up save path
            string testDataPath = totalPath5 + to_string(testFileCounter) + ".txt";

            //read from raw data
            ifstream MyReadFile(totalPath);
            //open new file to store test data
            ofstream MyFile4(testDataPath);

            //store where the test data is being stored?
            testFilesPath << testDataPath << "\n";

            //read until there's nothing left to read
            while (getline(MyReadFile, myText))
            {
                MyFile4 << observations[myText] << "\n";
            }
            // Close the file
            MyReadFile.close();
            MyFile4.close();
            it++;
            newCounter++;
            testFileCounter++;
        }

        testFilesPath.close();
    }
 

    cout << "done" << endl;
}

int afasmain()
{
    process();
    return 0;
}