# Assignment 3 – Building a Multilayer Artificial Neural Network

This repository contains our solution for the Assignment 3 – Building a Multilayer Artificial Neural Network. Given the code at chapter 11: “Implementing a Multi-layer Artificial Neural Network from Scratch” of the book “Machine Learning with PyTorch and Scikit-Learn” by Raschka et al. (2022). We extend the code to address multilayer hidden layers.

### Instruction:
[https://docs.google.com/document/d/19KZ-5a8XbB1EkJ6NK4b0S8B9HS_7OL8I/edit](https://docs.google.com/document/d/19KZ-5a8XbB1EkJ6NK4b0S8B9HS_7OL8I/edit)

[docs\ML_Assignment3_Instructions.pdf](docs\ML_Assignment3_Instructions.pdf)

### Students

| Name              | id                    | email                    |
|-------------------|-----------------------|--------------------------|
| Omer Eliyahu      | 206510828             | Omereliy@post.bgu.ac.il  |
| Nechi Berhe Weldu | 850164070             | weldu@post.bgu.ac.il     |
| Amit Ner Gaon     | 211649801             | amitner@post.bgu.ac.il   |


### Setting Up Your Python Environment

**Conda**

```bash
conda env create -f environment.yml
```

After creating this environment, you can activate it via

```bash
conda activate "pyml-book"
```

**Pip**

```bash
pip install virtualenv
cd /path/to/where/you/want/your/environment
virtualenv pyml-book
source pyml-book/bin/activate 
```

After activating your environment, you can install the required packages via

```bash
pip install numpy==1.21.2 scipy==1.7.0 scikit-learn==1.0 matplotlib==3.4.3 pandas==1.3.2, jupyterlab
```


### Acknowledgements
Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili.  
*Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing, 2022.

```bibtex
@book{mlbook2022,
  address   = {Birmingham, UK},
  author    = {Raschka, Sebastian and Liu, Yuxi (Hayden) and Mirjalili, Vahid},
  isbn      = {978-1801819312},
  publisher = {Packt Publishing},
  title     = {Machine Learning with PyTorch and Scikit-Learn},
  year      = {2022}
}
```

[https://sebastianraschka.com/books/#machine-learning-with-pytorch-and-scikit-learn](https://sebastianraschka.com/books/#machine-learning-with-pytorch-and-scikit-learn)
