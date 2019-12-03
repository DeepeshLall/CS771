#include "multi.h"
#include <omp.h>
#include <cassert>

StaticModel* readModel(char* file){
	
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

int main(int argc, char** argv){
	
	if( argc < 1+2 ){
		cerr << "multiPred [testfile] [model] (-p S <output_file>) (k)" << endl;
        cerr << "\t-p S <output_file>: print top S <label>:<prediction score> pairs to <output_file>, one line for each instance. (default S=0 and no file is generated)" << endl;
        cerr << "\tcompute top k accuracy, default k=1" << endl;
		exit(0);
	}

	char* testFile = argv[1];
	char* modelFile = argv[2];
    char* outFname;
    int S = 0, offset = 0;
    if (argc > 5 && strcmp(argv[3], "-p") == 0){
        S = atoi(argv[4]);
        outFname = argv[5];
        offset = 3;
    }
	int T = 1;
	if (argc > 3 + offset){
		T = atoi(argv[3 + offset]);
	}
	//T = 5;		//For now, we want the model to recommend us the top T items
	StaticModel* model = readModel(modelFile);

    if (T > model->K || S > model->K){
        cerr << "k or S is larger than domain size" << endl;
        exit(0);
    }
	Problem* prob = new Problem();
	readData( testFile, prob );
	
	cerr << "Ntest=" << prob->N << endl;
	
	double start = omp_get_wtime();
	//compute accuracy
	vector<SparseVec*>* data = &(prob->data);
	vector<Labels>* labels = &(prob->labels);
	Float hit=0.0;
	Float margin_hit = 0.0;
	Float* prod = new Float[model->K];
	int* max_indices = new int[model->K+1];
	for(int k = 0; k < model->K+1; k++){
		max_indices[k] = -1;
	}
    ofstream fout;
    if (S != 0){
        cerr << "Printing Top " << S << " <label>:<prediction score> pairs to " << outFname << ", one line per instance" << endl;
        fout.open(outFname);
    }
    int number_of_user_liked[prob->K+1];	//the i'th element of this array stores the number of users who liked item i
    int number_of_user_for_which_predicted[prob->K+1];	//the i'th element of this array stores the number of users for which item i was predicted
    //note both of these arrays are indexed by the reduces item indices, not the one mentioned in the testdata file
    memset(number_of_user_liked, 0, sizeof(number_of_user_liked));
    memset(number_of_user_for_which_predicted, 0, sizeof(number_of_user_for_which_predicted));
	for(int i=0;i<prob->N;i++){
		//N is the number of data points in the test file
		//this iteration is for the i'th data point
		memset(prod, 0.0, sizeof(Float)*model->K);
		
		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
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
			fout << model->label_name_list->at(max_indices[k])/* << ":" << prod[max_indices[k]]*/;
			if (k == Ti-1) fout << '\n';
			else fout << ' ';
			
			/*
			bool flag = false;
			for (int j = 0; j < yi->size(); j++){
				number_of_user_liked[yi->at(j)]++;
				if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(max_indices[k])){
					flag = true;
				}
			}
			if (flag) {
				hit += 1.0/Ti;
				myhit += 1.0 / Ti;
			}
			*/
		}
		/*
		fout << "microprecision@" << T << " for user " << i << " : " << myhit << endl;
		for(int k = 0; k < T; k++) {
			number_of_user_for_which_predicted[max_indices[k]]++;
		}
		*/

		//label_name_list and label_index_map together assign a name for each item
		//if the name of item k is "kitkat", then vector<string> label_name_list contains "kitkat"
		//and map<string, int> label_index_map will have the mapping label_index_map["kitkat"] = k

		//the top T items predicted for this user are max_indices[0], max_indices[1], ...
		//the items this user likes are for(int j = 0; j < yi.size(); j++) prob->label_name_list[yi->at(j)]
		/*
        if (S != 0){
            for (int k = 0; k < S; k++){
                if (k != 0){
                    fout << " ";
                }
                fout << model->label_name_list->at(max_indices[k]) << ":" << prod[max_indices[k]];
            }
            fout << endl;
        }
        */
	}
	/*
	for(int item = 0; item < prob->K; item++) {
		fout << "macroprecision@" << T << " for item " << prob->label_name_list[item] << " : " << (float)number_of_user_for_which_predicted[item] / number_of_user_liked[item] << endl;
	}
    if (S != 0){
	    fout.close();
    }
	*/
	fout.close();
	double end = omp_get_wtime();
	cerr << "Top " << T << " Acc=" << ((Float)hit/prob->N) << endl;
	cerr << "pred time=" << (end-start) << " s" << endl;
}
