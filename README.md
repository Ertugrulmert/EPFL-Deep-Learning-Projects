# Term projects for EPFL's Deep Learning (EE-559) course
## Team members:
- Mert Ertuğrul 
- Luis De Lima Carvalho
- Raphaël Gaiffe

This repository contains our submission for the two projects assigned in EE-559: Deep Learning course by Prof. François Fleuret. 

We thank Prof. François Fleuret for the resources and course material he provided us and the teaching assistants for their help throughout this course.

General info about the course: https://edu.epfl.ch/coursebook/en/deep-learning-EE-559

This repository consists of two projects, both were implemented with **PyTorch**, therefore make sure to install PyTorch before testing the projects. 
Explanation and official instructions for the two projects can be found in EE559-Miniprojects-Instructions.pdf

Brief explanations for project objectives:

## Project 1

In particular, it aims at showing the performance improvement that can be achieved through weight sharing, or using auxiliary losses. For the latter, the training in particular takes advantage of the availability of the classes of the two digits in each pair, beside the Boolean value truly of interest (comparison output). 

## Project 2

The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.

-----------------------------------

* The source code for these projects includes sample code from course material (**dlc_practical_prologue**). This module is responsible for providing and preparing data for Project 1.

* Both project folders contain a "test.py" file for showcasing the results.

* Detailed explanation of project code and results obtained can be found in the **project reports**.

* Style and structure of project code were inspired by the course material and lab assignment solutions provided by the course.
