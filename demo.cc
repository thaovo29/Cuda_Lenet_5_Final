#include "setup.h"

void inference_only(int batch_size) {

  std::cout<<"Loading fashion-mnist data...";
  MNIST dataset("./data/mnist/");
  dataset.read();
  std::cout<<"Done"<<std::endl;
  
  std::cout<<"Loading model...";
  Network dnn = createNetwork_GPU();
  std::cout<<"Done"<<std::endl;

  std::cout<<"Start Forward"<<std::endl;
  auto start = high_resolution_clock::now();

  dnn.forward(dataset.test_data);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  std::cout << "Time taken: "
         << duration.count() / 1000 << " milliseconds" << std::endl;
  
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout<<std::endl;
  std::cout<<"Test Accuracy: "<<acc<< std::endl;
  std::cout<<std::endl;
}

int main(int argc, char* argv[]) {

  int batch_size = 10000;
  
  if(argc == 2){
    batch_size = atoi(argv[1]);
  }

  std::cout<<"Test batch size: "<<batch_size<<std::endl;
  inference_only(batch_size);

  return 0;
}
