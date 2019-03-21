This is a introduction to the program file. This folder contians all the py file needed to run the code.
In order to load teacher model, go to the link https://drive.google.com/open?id=1zLOd_hqnCN-brWvi_XMPMxXQ8t54mwo1 and download following files:
resnet101.pth, resnet56.pth, resnet32.pth, PyramidNet.pth, densenet.ckpt.
Please also put these five files under the same file as 'Final_Project.py'.


resnet32.py: structure of resnet32
resnet56.py: structure of resnet56
resnet101.py: structure of resnet101
densenet.py: structure of densenet
pyramidnet.py: structure of pyramidnet
student.py: all structures of student model, including 1 4-layer CNN, 2 5-layer CNN, 1 7-layer CNN, and 1 MLP.
Final_Project.py: main file to train and test the whole project. The CIFAR-10 loader and different teacher and studnet models are included and importing in this file.

To run the project, use command 'python ~/Final_Project.py'. 
To use different student model, in the file Final_Project.py, change line 186 'student_net = ' to the student net that needed to be tested. 
Similarly, to used differnet teacher model, change line 187 'teacher_net = ' to the teacher net that needed to be utilized. 
 