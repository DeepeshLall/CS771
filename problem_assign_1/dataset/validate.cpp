// Program to print 1 with 75% probability and 0 
// with 25% probability 
#include <iostream> 
#include <fstream> 

using namespace std; 

// Random Function to that returns 0 or 1 with 
// equal probability 
int rand50() 
{ 
	// rand() function will generate odd or even 
	// number with equal probability. If rand() 
	// generates odd number, the function will 
	// return 1 else it will return 0. 
	return rand() & 1; 
} 

// Random Function to that returns 1 with 75% 
// probability and 0 with 25% probability using 
// Bitwise OR 
bool rand75() 
{ 
	return rand50() | rand50(); 
} 

// Random Function to that returns 1 with 75% 
// probability and 0 with 25% probability using 
// Bitwise OR 
bool rand50per() 
{ 
	return rand50();
} 

bool rand60per() 
{ 
	return (rand50() + rand50())%2;
}

bool rand80per() 
{ 
	return (rand50() + rand50() + rand50())%2;
}  

// Driver code to test above functions 
int main() 
{ 
	// Initialize random number generator 
	srand(time(NULL)); 

	//for(int i = 0; i < 50; i++) 
		//cout << rand75();
	ifstream fin; 

	ofstream fout_training;
	ofstream fout_validate; 

	fin.open("data"); 
	fout_validate.open("validate4");
	fout_training.open("training4");
	string line;
  
    // Execute a loop until EOF (End of File) 
    while (fin) { 
  
        // Read a Line from File 
        getline(fin, line); 
        int a  = rand80per();

  		if(a) fout_training << line << endl;
  		else fout_validate << line << endl;
        // Print line in Console 
        //cout << line << endl;
        cout << a<<endl; 
    } 

	return 0; 
} 
