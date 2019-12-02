#include "mymulti.h"
#include <omp.h>
#include <cassert>
#include "PDSparse.h"
#include <string>

PDSparse::PDSparse() {}
PDSparse::~PDSparse() {}

StaticModel* PDSparse::readModel(/*char* file*/){
	string file = "model.txt";
	StaticModel* model = new StaticModel();
	
	ifstream fin(file);
	char* tmp = new char[LINE_LEN];
	fin >> tmp >> (model->K);
	
	fin >> tmp;
	string name;
	for(int k=0;k<model->K;k++){
		fin >> name;
		model->label_name_list->push_back(name);
		model->label_index_map->insert(make_pair(name,k));
	}
	
	fin >> tmp >> (model->D);
	model->w = new SparseVec[model->D];
	
	vector<string> ind_val;
	int nnz_j;
	for(int j=0;j<model->D;j++){
		fin >> nnz_j;
		model->w[j].resize(nnz_j);
		for(int r=0;r<nnz_j;r++){
			fin >> tmp;
			ind_val = split(tmp,":");
			int k = atoi(ind_val[0].c_str());
			Float val = atof(ind_val[1].c_str());
			model->w[j][r].first = k;
			model->w[j][r].second = val;
		}
	}
	
	delete[] tmp;
	return model;
}

vector<vector<int> > PDSparse::run(vector<vector<int> > X1, vector<vector<float> > X2, int T) {
	int S = T;
	StaticModel* model = readModel();
	/*
    if (T > model->K || S > model->K){
        cerr << "k or S is larger than domain size" << endl;
        exit(0);
    }
    */
	Problem* prob = new Problem();
	readData( X1, X2, prob );
	
	cerr << "Ntest=" << prob->N << endl;
	
	double start = omp_get_wtime();
	//compute accuracy
	vector<SparseVec*>* data = &(prob->data);
	//vector<Labels>* labels = &(prob->labels);
	Float hit=0.0;
	Float margin_hit = 0.0;
	Float* prod = new Float[model->K];
	int* max_indices = new int[model->K+1];
	for(int k = 0; k < model->K+1; k++){
		max_indices[k] = -1;
	}
	/*
    ofstream fout;
    if (S != 0){
        cerr << "Printing Top " << S << " <label>:<prediction score> pairs to " << outFname << ", one line per instance" << endl;
        fout.open(outFname);
    }
    */
    vector<vector<int> > y_pred(prob->N, vector<int>(T));
    int number_of_user_liked[model->K+1];	//the i'th element of this array stores the number of users who liked item i
    int number_of_user_for_which_predicted[model->K+1];	//the i'th element of this array stores the number of users for which item i was predicted
    //note both of these arrays are indexed by the reduces item indices, not the one mentioned in the testdata file
    memset(number_of_user_liked, 0, sizeof(number_of_user_liked));
    memset(number_of_user_for_which_predicted, 0, sizeof(number_of_user_for_which_predicted));
	for(int i=0;i<prob->N;i++){
		//N is the number of data points in the test file
		//this iteration is for the i'th data point
		memset(prod, 0.0, sizeof(Float)*model->K);
		
		SparseVec* xi = data->at(i);
		//Labels* yi = &(labels->at(i));
		//yi is the lab_indices for user i (refer to multi.h for more info on lab_indices)
		int Ti = T;
		/*
		if (Ti <= 0)
			Ti = yi->size();
		*/
        int top = max(Ti, S);
		for(int ind = 0; ind < model->K; ind++){
			max_indices[ind] = ind;
		}
        for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			
			int j= it->first;
			Float xij = it->second;
			if( j >= model->D )
				continue;
			SparseVec* wj = &(model->w[j]);
			for(SparseVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
				int k = it2->first;
				prod[k] += it2->second*xij;
			}
		}
        nth_element(max_indices, max_indices+top, max_indices+model->K, ScoreComp(prod));
        float myhit = 0.0;	//we want hit for each data point
        sort(max_indices, max_indices+top, ScoreComp(prod));
		for(int k=0;k<Ti;k++){
			y_pred[i][k] = stoi(model->label_name_list->at(max_indices[k]));
		}
	}
	return y_pred;
}
