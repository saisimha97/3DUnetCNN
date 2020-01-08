from brats.preprocess import convert_brats_data

source  ="/home/jbmai_sai/PycharmProjects/3DUnetCNN/brats/data/original"

target = "/home/jbmai_sai/PycharmProjects/3DUnetCNN/brats/data/preprocessed"

convert_brats_data(source, target)
print("done")