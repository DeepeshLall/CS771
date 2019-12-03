#ifndef PDSPARSE_H
#define PDSPARSE_H

#include <string>
#include <vector>

class PDSparse {
	public:
		PDSparse();
		~PDSparse();
		StaticModel* readModel();
		vector<vector<int> > run(vector<vector<int> >, vector<vector<float> >, int);
};

#endif