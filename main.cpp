#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <cmath>

using namespace std;

#define PRINT_DATASET(d) printDataset(d, #d)

int alpha = 1;

struct document
{
    bool pos = 0;
    int size = 0;
    unordered_map<string, double> words;
};

struct data
{
    int num=0, pos = 0, neg = 0;
};

struct feature
{
    double ppos = 0, pneg = 0;
};

vector<document> readDatasetFromFile(ifstream& i, string fileName)
{
    i.open(fileName);
    if (!i)
    {
        cout << "Error: file \"" << fileName << "\" could not be read" << endl;
        exit(1);
    }

    vector<document> output;
    string line, word;
    while (i.good() && getline(i, line))
    {
        document i;
        i.pos = line[line.size() - 1] == '1';
        line.erase(line.size() - 2);

        stringstream ss(line);
        while (ss >> word)
        {
            auto it = i.words.insert(pair<string, int>(word, 0));
            it.first->second++;
            i.size++;
        }
        output.push_back(i);
    }
    i.close();

    return output;
}

void tfIdfVectorize(vector<document>& raw)
{
    unordered_map<string, int> numDocs;

    for(auto i : raw)
    {
        for(auto j : i.words)
        {
            auto it = numDocs.insert(pair<string, int>(j.first, 0)).first;
            it->second++;
        }
    }

    for(auto i = raw.begin(); i != raw.end(); i++)
    {
        for(auto j = i->words.begin(); j != i->words.end(); j++)
        {
            double idf = log(raw.size() / (double)(numDocs.at(j->first)));
            j->second *= j->second * idf / (double)i->size;
        }
    }
}

unordered_map<string, feature> summary(const vector<document>& raw)
{
    unordered_map<string, feature> output;
    unordered_map<string, data> dMap;
    data d;
    feature f;
    int numPos = 0, numNeg = 0;
    for (auto i : raw)
    {
        for (auto j : i.words)
        {
            auto it = dMap.insert(pair<string, data>(j.first, d)).first;
            it->second.num++;
            if (i.pos)
            {
                it->second.pos += j.second;
                numPos += j.second;
            }
            else
            {
                it->second.neg += j.second;
                numNeg += j.second;
            }
        }
    }
    for (auto i : dMap)
    {
        auto it = output.insert(pair<string, feature>(i.first, f)).first;
        it->second.ppos = (i.second.pos + alpha) / (double)(numPos + alpha*dMap.size());
        it->second.pneg = (i.second.neg + alpha) / (double)(numNeg + alpha*dMap.size());
    }

    return output;
}

double test(const vector<document>& dataset, const unordered_map<string, feature>& classifier, bool out=0)
{
    int correct = 0;
    for (auto i : dataset)
    {
        int cnb;
        double logpos = 0, logneg = 0;
        for (auto j : i.words)
        {
            try
            {
                logpos += j.second * log(classifier.at(j.first).ppos);
                logneg += j.second * log(classifier.at(j.first).pneg);
            }
            catch (const exception& e) {}
        }
        cnb = logpos > logneg ? 1 : 0;
        if (cnb == i.pos)
            correct++;
        if (out)
            cout << cnb << endl;
    }
    return correct / (double)dataset.size();
}

void printDataset(vector<document> dataset, string name)
{
    cout << "---printing " << name << "---\n";
    for(auto i : dataset) 
    {
        for(auto j : i.words)
        {
            cout << j.first << " " << j.second << " ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " training.txt testing.txt" << endl;
        exit(1);
    }

    double training_score, testing_score;

    ifstream file;
    auto start = std::chrono::steady_clock::now();

    vector<document> training = readDatasetFromFile(file, argv[1]);

    unordered_map<string, feature> classifier = summary(training);

    auto train = std::chrono::steady_clock::now();

    vector<document> testing = readDatasetFromFile(file, argv[2]);

    training_score = test(training, classifier);
    testing_score = test(testing, classifier);//, true);

    auto test = std::chrono::steady_clock::now();

    cout << chrono::duration_cast<chrono::seconds>(train - start).count() << " seconds (training)" << endl;
    cout << chrono::duration_cast<chrono::seconds>(test - train).count() << " seconds (labeling)" << endl;
    cout << training_score << " (training)" << endl;
    cout << testing_score << " (testing)" << endl;

    return 0;
}