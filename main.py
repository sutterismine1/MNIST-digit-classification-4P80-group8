import drawing_input
from CNN import CNN

def get_matrix_from_input():
    matrix = drawing_input.run()
    if matrix is not None:
        print("Downsampled Image:")
        print(matrix)
        return matrix
    else:
        print("Something went wrong during input.")
        return matrix
            

if __name__ == "__main__":
    cnn = CNN(10) # initialize CNN with 10 classes
    cnn.train()
    matrix = get_matrix_from_input()
    if matrix is not None:
        cnn.test_specific_matrix(matrix)